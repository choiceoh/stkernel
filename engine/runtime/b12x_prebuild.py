"""The b12x MoE kernels a tree's boot will ask for, compiled on the CPU before that boot (runtime).

engine/kernels/b12x_requests.py records, at every fleet boot, each dispatcher call that added a kernel. This replays
those calls through THIS tree's getters, in a container with no GPU, into the flashinfer modules a boot reads -- so the
key and module name are this tree's own, and its first boot loads instead of compiling. launchers/b12x-prebuild.sh
runs it on every node while the fleet still serves; st-deploy-watch.py runs that before a deploy, and
`start-st-qwen38.sh prebuild` before a Qwen3.8 window.

    CUDA_VISIBLE_DEVICES= python3 -m engine.runtime.b12x_prebuild prebuild \\
        --requests /cache/cu132/st-b12x-requests/qwen38.jsonl

It lives with the runtime because it reads the environment it runs in -- the GPU it must not see, the targets it sets
for three libraries -- and the kernel package reads none (D11).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from engine.kernels import b12x_requests as requests


def prebuild(paths, *, emit=print) -> dict:
    """Replay every request in `paths` through this tree's getters on the CPU -> the summary. Each request's line is
    emitted as it finishes (`hit`: every object it names was already on disk; `built`: at least one was compiled now;
    `cached`: an earlier request in this run already made its kernel; `failed`, with the error)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("prebuild compiles on the CPU: run it with CUDA_VISIBLE_DEVICES= (never beside production on its GPU)")
    # The target is the fleet's one GPU, said three times because three libraries ask: the CuTe DSL's target, the
    # capability the dispatcher checks, and flashinfer's workspace directory -- which it names after the visible
    # devices, and with none visible names nothing, so without this the objects land beside the ones a boot reads.
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.1")
    lines = requests.read(paths)
    import torch
    from unittest.mock import patch
    started = time.monotonic()
    results = []
    with patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        from flashinfer.jit import cute_dsl_core
        from flashinfer.jit import env as jit_env
        arch = jit_env.FLASHINFER_WORKSPACE_DIR.name
        if not arch:
            raise SystemExit(f"prebuild: flashinfer names no arch directory ({jit_env.FLASHINFER_WORKSPACE_DIR}); "
                             "a boot would never read what this writes")
        real = md.build_and_load_cute_dsl_kernel
        saved = {name: getattr(md, name) for name in requests.CONFIG}
        try:
            for line in lines:
                objects = []

                def build(module, name, compile_fn, extra_key_files=(), **kwargs):
                    try:
                        sha = cute_dsl_core._hash_source_files(tuple(extra_key_files))
                        hit = cute_dsl_core.JitSpecCuteDsl(module, name, compile_fn, sha).is_compiled
                    except (OSError, TypeError):
                        hit = False
                    objects.append({"module": module, "kernel": name, "hit": hit})
                    return real(module, name, compile_fn, extra_key_files=extra_key_files, **kwargs)

                began = time.monotonic()
                result = {"getter": line.get("getter"), "profile": line.get("profile")}
                try:
                    if line["getter"] not in requests.GETTERS:
                        raise ValueError(f"not a recorded getter: {line['getter']!r}")
                    device = line["device"]
                    if device.get("jit_arch", arch) != arch:
                        raise ValueError(f"the boot read objects under {device['jit_arch']!r}, this prebuild writes {arch!r}")
                    clusters = {int(size): int(n) for size, n in device["clusters"].items()}
                    for name, value in line["config"].items():
                        if name in requests.CONFIG:
                            setattr(md, name, requests.decode(value))
                    with patch.object(md, "get_num_sm", lambda *a, **k: int(device["sm"])), \
                            patch.object(md, "get_max_active_clusters", lambda size=1, *a, **k: clusters[int(size)]), \
                            patch.object(md, "build_and_load_cute_dsl_kernel", build):
                        getattr(md, line["getter"])(*requests.decode(line["args"]),
                                                    **{k: requests.decode(v) for k, v in line["kwargs"].items()})
                    result["status"] = ("cached" if not objects else
                                        "hit" if all(o["hit"] for o in objects) else "built")
                except Exception as exc:                  # noqa: BLE001 -- one request's failure is reported, not fatal
                    result["status"], result["error"] = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
                result["kernels"] = [f"{o['module']}/{o['kernel']}" for o in objects]
                result["seconds"] = round(time.monotonic() - began, 3)
                results.append(result)
                emit(json.dumps(result, sort_keys=True))
        finally:
            for name, value in saved.items():
                setattr(md, name, value)
    if torch.cuda.is_initialized():
        raise RuntimeError("prebuild initialised CUDA")
    summary = {"requests": len(lines), "seconds": round(time.monotonic() - started, 3), "into": str(jit_env.FLASHINFER_JIT_DIR)}
    for status in ("built", "hit", "cached", "failed"):
        summary[status] = sum(r["status"] == status for r in results)
    emit(json.dumps({"summary": summary}, sort_keys=True))
    return {"summary": summary, "requests": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m engine.runtime.b12x_prebuild", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prebuild", help="compile every recorded request for this tree, on the CPU")
    build.add_argument("--requests", nargs="+", required=True, help="request files written at boot (b12x_requests.record)")
    build.add_argument("--report", default=None, help="also write the per-request results here (JSON)")
    show = sub.add_parser("list", help="the distinct requests in the files")
    show.add_argument("--requests", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.command == "list":
        for line in requests.read(a.requests):
            print(json.dumps({k: line.get(k) for k in ("profile", "getter", "args", "kwargs")}, sort_keys=True))
        return 0
    report = prebuild(a.requests)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
