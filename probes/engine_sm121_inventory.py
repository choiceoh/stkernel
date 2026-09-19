"""What the seed image already carries for engine/SM121_INTAKE.md, read inside the GB10 container (probe, single-GPU
lane; intake item U0).

Each intake item either binds a kernel the image has or vendors one it lacks (engine/kernels/SOURCES.json). Which is
which is a fact of the image -- flashinfer 0.6.18.dev20260819 as built into the seed, not the upstream tree of that
date -- so it is read here, where the engine runs:

    files       the installed flashinfer package's files whose names match an item's candidates (FlashKDA sources, the
                SM120 delta-rule prefill, b12x, the block-scaled GEMM, SM120 sparse attention, GQA decode)
    words       whether a candidate source says what the item needs (MXFP4 with FP8 activations, sinks and windows in
                paged decode, FP8 KV)
    modules     each candidate module imported, and its public callables' signatures -- or the import's error
    b12x        engine/kernels/b12x against the image's blackwell_sm12x, file by file (same bytes, differs, only one side)
    toolchain   device, capability, torch, triton, cutlass-dsl, flashinfer, nvcc

    python3 probes/engine_kernel_check.py --lanes sm121_inventory --output /cache/sm121-inventory.json

Reads only: nothing is compiled or launched. Each item's own ticket compiles and judges its kernel.
"""
from __future__ import annotations

import fnmatch
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# item -> file name patterns under the installed flashinfer package
FILES = {
    "U3 b12x mxfp4 w4a8": ("*b12x*", "fused_moe_mxfp8_mxfp4*"),
    "U5 dense nvfp4 gemm": ("blockscaled_gemm*", "*mm_fp4*", "gemm*.py"),
    "U6 sparse mla sm120": ("bsa_attn_sm120*", "flash_fwd_sm120*", "*sparse_mla*"),
    "U7 gqa paged decode": ("gqa_decode*", "batch_decode*", "decode.py"),
    "U9 flashkda": ("*flashkda*", "*kda*"),
    "U13 gdn prefill": ("delta_rule_sm120*", "gdn_prefill*", "gdn_decode*"),
}

# item -> (path relative to the package, words whose presence is recorded)
WORDS = {
    "U3 b12x mxfp4 w4a8": (("fused_moe/cute_dsl/b12x_moe.py", ("mxfp4", "mxfp8", "w4a8", "fp8", "activation")),),
    "U5 dense nvfp4 gemm": (("gemm/__init__.py", ("b12x", "mm_fp4", "cutlass", "cute_dsl")),
                            ("gemm.py", ("b12x", "mm_fp4", "cutlass", "cute_dsl"))),
    "U7 gqa paged decode": (("decode.py", ("window_left", "sinks", "sink", "kv_data_type", "logits_soft_cap")),),
    "U8 kv quant": (("decode.py", ("float8_e4m3fn", "fp8", "nvfp4", "kv_cache_sf")),),
}

# item -> modules to import (their public callables' signatures are recorded)
MODULES = {
    "U3 b12x mxfp4 w4a8": ("flashinfer.fused_moe.cute_dsl.b12x_moe", "flashinfer.fused_moe.cute_dsl.fused_moe_mxfp8_mxfp4"),
    "U5 dense nvfp4 gemm": ("flashinfer.gemm", "flashinfer.cute_dsl.blockscaled_gemm"),
    "U6 sparse mla sm120": ("flashinfer.cute_dsl.sparse.bsa_attn_sm120",),
    "U7 gqa paged decode": ("flashinfer.decode", "flashinfer.cute_dsl.attention.gqa_decode"),
    "U13 gdn prefill": ("flashinfer.gdn_prefill", "flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120",
                        "flashinfer.gdn_kernels.blackwell.gdn_prefill"),
}

SIGNATURES = 40          # public callables recorded a module
FILE_CAP = 200           # file names recorded an item


def _package(name):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def _version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as exc:  # noqa: BLE001 -- a missing distribution is the answer
        return f"absent ({type(exc).__name__})"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def files(pkg):
    paths = [p for p in pkg.rglob("*") if p.is_file()]
    out = {}
    for item, patterns in FILES.items():
        hits = sorted({p.relative_to(pkg).as_posix() for p in paths for pat in patterns if fnmatch.fnmatch(p.name, pat)})
        out[item] = {"count": len(hits), "files": hits[:FILE_CAP]}
    return out


def words(pkg):
    out = {}
    for item, sources in WORDS.items():
        rows = {}
        for rel, wanted in sources:
            path = pkg / rel
            if not path.is_file():
                rows[rel] = "absent"
                continue
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            rows[rel] = {w: text.count(w.lower()) for w in wanted}
        out[item] = rows
    return out


def _signatures(mod):
    rows = {}
    for name, value in sorted(vars(mod).items()):
        if name.startswith("_") or not callable(value) or getattr(value, "__module__", None) != mod.__name__:
            continue
        try:
            rows[name] = str(inspect.signature(value))[:400]
        except (TypeError, ValueError):
            rows[name] = "(no signature)"
        if len(rows) >= SIGNATURES:
            break
    return rows


def modules():
    out = {}
    for item, names in MODULES.items():
        rows = {}
        for name in names:
            t0 = time.perf_counter()
            try:
                mod = importlib.import_module(name)
                rows[name] = {"imported": True, "seconds": round(time.perf_counter() - t0, 2),
                              "file": getattr(mod, "__file__", None), "callables": _signatures(mod)}
            except Exception as exc:  # noqa: BLE001 -- the import's failure is what this lane records
                rows[name] = {"imported": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        out[item] = rows
    return out


def b12x(pkg):
    ours = ROOT / "engine" / "kernels" / "b12x"
    theirs = pkg / "fused_moe" / "cute_dsl" / "blackwell_sm12x"
    if not theirs.is_dir():
        return {"image": "absent"}
    mine = {p.relative_to(ours).as_posix(): p for p in ours.rglob("*.py")}
    image = {p.relative_to(theirs).as_posix(): p for p in theirs.rglob("*.py")}
    both = sorted(set(mine) & set(image))
    return {"same": [n for n in both if _sha(mine[n]) == _sha(image[n])],
            "differs": [n for n in both if _sha(mine[n]) != _sha(image[n])],
            "engine_only": sorted(set(mine) - set(image)), "image_only": sorted(set(image) - set(mine))}


def toolchain():
    row = {"python": sys.version.split()[0], "nvcc": shutil.which("nvcc"), "cuda_home": os.environ.get("CUDA_HOME")}
    for dist in ("torch", "triton", "flashinfer-python", "nvidia-cutlass-dsl", "deep_gemm", "tilelang"):
        row[dist] = _version(dist)
    try:
        import torch
        if torch.cuda.is_available():
            row["device"] = torch.cuda.get_device_name()
            row["capability"] = list(torch.cuda.get_device_capability())
            row["arch_list"] = torch.cuda.get_arch_list()
    except Exception as exc:  # noqa: BLE001
        row["torch_error"] = f"{type(exc).__name__}: {exc}"
    return row


def run(output=None) -> dict:
    pkg = _package("flashinfer")
    report = {"lane": "sm121_inventory", "toolchain": toolchain(), "flashinfer_dir": str(pkg) if pkg else None}
    if pkg is not None:
        report.update(files=files(pkg), words=words(pkg), b12x=b12x(pkg))
    report["modules"] = modules()
    text = json.dumps(report, indent=1, default=str)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
    print(text, flush=True)
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
