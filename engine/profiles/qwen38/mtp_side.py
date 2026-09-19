"""The MTP head's experts at a precision the rank files do not keep, as side files a rank loads beside its rank file
(profile).

    python3 -m engine.profiles.qwen38.mtp_side --precision bf16 --ckpt <a copy with the fused BF16 experts> \\
        --out /home/choiceoh/models/st-qwen38-mtp-bf16
    python3 -m engine.profiles.qwen38.mtp_side --precision fp8 --ckpt /home/choiceoh/models/qwen38-flash-next-nvidia-nvfp4 \\
        --out /home/choiceoh/models/st-qwen38-mtp-fp8

The rank files keep the MTP head's experts as NVFP4 re-encoded from the export's FP8 (specs.py: quantised twice, the
activations to FP4 as well). The operator's rule of 2026-09-19 -- the engine is NVFP4 by default, and invests where
the cost is small and the acceptance impact large -- pins them at the checkpoint's original BF16: "bf16" slices each
rank's 128 experts out of the older copy's fused BF16 tensors as they are (specs.mtp_bf16_specs); "fp8" writes the NVIDIA
export's own e4m3 bytes with their tile scales widened to FP32 exactly (specs.mtp_fp8_specs). Rank r's file is
`mtp-{precision}-r{r}of4.safetensors` (base/preshard.RankWriter: the loader's layout) in a directory of its own, not the
rank files' (theirs is immutable, preshard.py). Each file is read back and every tensor's bytes compared with what was
written; the directory is built as `<out>.incomplete` and renamed when every check passed, with a manifest. A boot
serving them: fleet.py --mtp-experts (kernels/moe_rows).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from engine.base.checkpoint import Checkpoint                          # noqa: E402
from engine.base.loader import RankLoader                              # noqa: E402
from engine.base.preshard import RankWriter                              # noqa: E402
from engine.profiles.qwen38 import facts, specs as layout             # noqa: E402
from engine.profiles.qwen38.preshard import file_hash, tensor_hash     # noqa: E402

LAYOUTS = {"fp8": "qwen38-mtp-fp8-v1", "bf16": "qwen38-mtp-bf16-v1"}
DIRS = {precision: f"/home/choiceoh/models/st-qwen38-mtp-{precision}" for precision in LAYOUTS}   # fleet --mtp-experts


SOURCES = {"bf16": ("bf16", 0), "fp8": ("fp8_block", 128)}   # what a checkpoint must keep to write a side file


def side_specs(F, precision: str):
    """The side file's tensors -- their names and shapes, what a boot binds whatever checkpoint its facts come from
    (the served rank files' facts are the NVIDIA export's, "fp8_block", whichever side file serves the experts)."""
    if precision == "fp8":
        return layout.mtp_fp8_specs(F)
    if precision == "bf16":
        return layout.mtp_bf16_specs(F)
    raise ValueError(f"MTP side files are bf16 or fp8, not {precision!r}")


def check_source(F, precision: str) -> None:
    """Writing a side file reads the checkpoint's own encoding: the fused BF16 experts for bf16, NVIDIA's per-expert
    FP8 (block 128) for fp8."""
    want = SOURCES[precision]
    if (F.mtp_experts, F.mtp_block if want[0] == "fp8_block" else 0) != want:
        raise ValueError(f"a {precision} side file is written from a checkpoint that keeps the MTP experts as "
                         f"{want[0]!r}; this one keeps {F.mtp_experts!r} (block {F.mtp_block})")


def path(directory, rank: int, precision: str) -> Path:
    return Path(directory) / f"mtp-{precision}-r{rank}of{facts.TP}.safetensors"


def write(ckpt, out, *, precision: str, source_revision: "str | None" = None) -> dict:
    started = time.monotonic()
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: the side files are immutable, write a new directory")
    F = facts.load(ckpt)
    check_source(F, precision)
    specs = side_specs(F, precision)
    ck = Checkpoint(str(ckpt))
    partial = out.with_name(out.name + ".incomplete")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    report = {"layout": LAYOUTS[precision], "precision": precision, "world": facts.TP, "source": str(ckpt), "source_revision": source_revision,
              "source_config_sha256": file_hash(Path(ckpt) / "config.json"), "files": []}
    m = "mtp.layers.0.mlp."
    for r in range(facts.TP):
        target = path(partial, r, precision)
        writer = RankWriter(target, specs, {"weight_layout": LAYOUTS[precision], "rank": r, "world": facts.TP})
        source = ck.views(sorted(layout.mtp_expert_keys(m, F, r)))
        hashes = {}
        for spec in specs:
            tensor = spec.build(source, r, facts.TP)
            if tuple(tensor.shape) != tuple(spec.shape) or tensor.dtype != spec.dtype:
                raise ValueError(f"{spec.name}: built {tuple(tensor.shape)} {tensor.dtype}, declared {spec.shape} {spec.dtype}")
            if tensor.dtype == torch.float32 and not torch.isfinite(tensor).all():
                raise ValueError(f"nonfinite scale: {spec.name}")
            hashes[spec.name] = tensor_hash(tensor)
            writer.put(spec.name, tensor)
            del tensor
        writer.close()
        del source
        loaded = RankLoader(target).load([s.name for s in specs], device="cpu", max_run=32 << 20)
        for spec in specs:
            got = loaded[spec.name]
            if tuple(got.shape) != tuple(spec.shape) or got.dtype != spec.dtype or tensor_hash(got) != hashes[spec.name]:
                raise AssertionError(("side file readback mismatch", r, spec.name))
        del loaded
        entry = dict(name=target.name, bytes=target.stat().st_size, sha256=file_hash(target), tensors=len(specs))
        report["files"].append(entry)
        print(json.dumps(dict(stage="rank", rank=r, seconds=round(time.monotonic() - started, 1), **entry)), flush=True)
    (partial / "manifest.json").write_text(json.dumps(report, indent=1) + "\n")
    partial.rename(out)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m engine.profiles.qwen38.mtp_side", description=__doc__.splitlines()[0])
    ap.add_argument("--precision", choices=tuple(LAYOUTS), required=True)
    ap.add_argument("--ckpt", required=True, help="bf16: a copy with the fused BF16 MTP experts; fp8: NVIDIA's export")
    ap.add_argument("--out", required=True, help="a new directory for the four side files and the manifest")
    ap.add_argument("--source-revision", default=None)
    a = ap.parse_args(argv)
    report = write(a.ckpt, a.out, precision=a.precision, source_revision=a.source_revision)
    print(json.dumps({"done": a.out, "files": [f["name"] for f in report["files"]]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
