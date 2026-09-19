"""The MTP head's experts in the checkpoint's own FP8, as a side file a rank loads beside its rank file (profile).

    python3 -m engine.profiles.qwen38.mtp_fp8 --ckpt /home/choiceoh/models/qwen38-flash-next-nvidia-nvfp4 \\
        --out /home/choiceoh/models/st-qwen38-mtp-fp8

The rank files keep the MTP head's experts as NVFP4 re-encoded from the export's FP8 (specs.py: quantised twice, the
activations to FP4 as well, "unmeasured"). NVIDIA's export kept that layer in FP8 where it made the target's experts
NVFP4. This writes the export's own bytes -- rank r's 128 experts, e4m3 as they are, their BF16 tile scales widened to
FP32 exactly (specs.mtp_fp8_specs) -- to `mtp-fp8-r{r}of4.safetensors` (base/preshard.RankWriter: the loader's layout)
in a directory of its own, not the rank files' (theirs is immutable, preshard.py). Each file is read back and every
tensor's bytes compared with what was written; the directory is built as `<out>.incomplete` and renamed when every
check passed, with a manifest. A boot serving them: fleet.py --mtp-experts-dir (kernels/moe_fp8_rows).
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

LAYOUT = "qwen38-mtp-fp8-v1"


def path(directory, rank: int) -> Path:
    return Path(directory) / f"mtp-fp8-r{rank}of{facts.TP}.safetensors"


def write(ckpt, out, *, source_revision: "str | None" = None) -> dict:
    started = time.monotonic()
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: the side files are immutable, write a new directory")
    F = facts.load(ckpt)
    specs = layout.mtp_fp8_specs(F)
    ck = Checkpoint(str(ckpt))
    partial = out.with_name(out.name + ".incomplete")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    report = {"layout": LAYOUT, "world": facts.TP, "source": str(ckpt), "source_revision": source_revision,
              "source_config_sha256": file_hash(Path(ckpt) / "config.json"), "files": []}
    m = "mtp.layers.0.mlp."
    for r in range(facts.TP):
        target = path(partial, r)
        writer = RankWriter(target, specs, {"weight_layout": LAYOUT, "rank": r, "world": facts.TP})
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
    ap = argparse.ArgumentParser(prog="python3 -m engine.profiles.qwen38.mtp_fp8", description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="NVIDIA's export: the MTP experts in FP8 (hf_quant_config)")
    ap.add_argument("--out", required=True, help="a new directory for the four side files and the manifest")
    ap.add_argument("--source-revision", default=None)
    a = ap.parse_args(argv)
    report = write(a.ckpt, a.out, source_revision=a.source_revision)
    print(json.dumps({"done": a.out, "files": [f["name"] for f in report["files"]]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
