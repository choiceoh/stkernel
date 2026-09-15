"""Preshard Qwen3.8-Flash-Next's ModelOpt NVFP4 checkpoint into four TEP=4 rank files (profile).

    python3 -m engine.profiles.qwen38.preshard --ckpt /home/choiceoh/models/qwen38-flash-next-nvfp4 \\
        --out /home/choiceoh/models/st-qwen38-tep4 --source-revision <hf revision> [--plan]

The layout is specs.py's (merged projections, whole experts on their rank, the PLE table in parts). One group at a time
is read (a layer's tensors once for all four ranks; the PLE table four shards at a time), every rank's tensors built
and streamed into its file (base/preshard.RankWriter: headers first, 256-byte aligned). Each file is read back and every
tensor's bytes compared with what was written. The checkpoint's metadata files are copied beside the ranks and the
shape wizard's record is written (base/kernel_shape.write_record): what a boot binds (fleet.py).

The output directory is new and immutable: built as `<out>.incomplete` and renamed when every check passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from engine.base.checkpoint import Checkpoint          # noqa: E402
from engine.base.loader import RankLoader              # noqa: E402
from engine.base.preshard import RankWriter            # noqa: E402
from engine.profiles.qwen38 import facts, specs as layout   # noqa: E402

METADATA_SUFFIXES = (".json", ".jinja", ".md")


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def file_hash(path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 << 20):
            value.update(block)
    return value.hexdigest()


def plan(ckpt):
    F = facts.load(ckpt)
    groups = list(layout.groups(F))
    index = json.loads((Path(ckpt) / "model.safetensors.index.json").read_text())["weight_map"]
    sources = {key for _, keys, _ in groups for key in keys}
    missing = sources - set(index)
    if missing:
        raise ValueError(("checkpoint tensors the layout reads are missing", sorted(missing)[:8]))
    unread = sorted(k for k in index if not k.startswith("model.visual.") and k not in sources)   # the text model, whole
    specs = [s for _, _, of in groups for s in of(0)]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("duplicate output tensor")
    report = dict(weight_layout=F.weight_layout, world=facts.TP, tensors_per_rank=len(specs),
                  payload_bytes_per_rank=sum(s.nbytes() for s in specs), source_tensors=len(sources),
                  unread_text_tensors=unread[:32], unread_text_count=len(unread),
                  source_config_sha256=file_hash(Path(ckpt) / "config.json"),
                  source_index_sha256=file_hash(Path(ckpt) / "model.safetensors.index.json"))
    return F, groups, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, default=facts.CKPT)
    ap.add_argument("--out", type=Path, default=facts.RANKS)
    ap.add_argument("--plan", action="store_true", help="read the headers and print the plan; write nothing")
    ap.add_argument("--source-revision", required=True)
    a = ap.parse_args(argv)
    F, groups, report = plan(a.ckpt)
    report["source_revision"] = a.source_revision
    print(json.dumps(report), flush=True)
    if report["unread_text_count"]:
        raise ValueError(f"{report['unread_text_count']} text tensors no spec reads: {report['unread_text_tensors'][:8]}")
    if a.plan:
        return 0
    out = a.out.resolve()
    partial = out.with_name(out.name + ".incomplete")
    if out.exists() or partial.exists():
        raise ValueError("the output must be a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    required = facts.TP * report["payload_bytes_per_rank"] + (8 << 30)
    if shutil.disk_usage(out.parent).free < required:
        raise ValueError("insufficient disk space for four ranks and a reserve")
    partial.mkdir()
    torch.set_num_threads(4)
    ck = Checkpoint(str(a.ckpt))
    all_specs = [s for _, _, of in groups for s in of(0)]
    writers = [RankWriter(partial / f"rank{r}of4.safetensors", all_specs,
                          dict(model="qwen38", world=facts.TP, rank=r, weight_layout=F.weight_layout,
                               source_revision=a.source_revision, scale_convention="modelopt-multiplier"))
               for r in range(facts.TP)]
    hashes = [{} for _ in writers]
    started = time.monotonic()
    for label, keys, specs_of in groups:
        source = ck.load(keys)
        for r, writer in enumerate(writers):
            for spec in specs_of(r):
                tensor = spec.build(source, r, facts.TP)
                if tuple(tensor.shape) != tuple(spec.shape) or tensor.dtype != spec.dtype:
                    raise ValueError(f"{spec.name}: built {tuple(tensor.shape)} {tensor.dtype}, declared {spec.shape} {spec.dtype}")
                if tensor.is_floating_point() and tensor.dtype not in (torch.float8_e4m3fn,) \
                        and not torch.isfinite(tensor.float()).all():
                    raise ValueError(f"nonfinite weight: {spec.name}")
                hashes[r][spec.name] = tensor_hash(tensor)
                writer.put(spec.name, tensor)
                del tensor
        del source
        print(json.dumps(dict(stage="write", group=label, seconds=round(time.monotonic() - started, 1))), flush=True)
    for writer in writers:
        writer.close()
    report["rank_files"] = []
    for r in range(facts.TP):
        path = partial / f"rank{r}of4.safetensors"
        reader = RankLoader(path)
        if reader.metadata["weight_layout"] != F.weight_layout or reader.metadata["rank"] != str(r):
            raise AssertionError("rank identity mismatch")
        for label, _keys, specs_of in groups:
            names = [s.name for s in specs_of(r)]
            loaded = reader.load(names, device="cpu", max_run=32 << 20)
            for spec in specs_of(r):
                got = loaded[spec.name]
                if tuple(got.shape) != tuple(spec.shape) or got.dtype != spec.dtype or tensor_hash(got) != hashes[r][spec.name]:
                    raise AssertionError(("rank readback mismatch", r, spec.name))
            del loaded
        entry = dict(name=path.name, bytes=path.stat().st_size, sha256=file_hash(path), tensors_verified=len(hashes[r]))
        report["rank_files"].append(entry)
        print(json.dumps(dict(stage="verify", rank=r, **entry)), flush=True)
    metadata = []
    for path in sorted(a.ckpt.iterdir()):
        if (path.is_file() and path.suffix in METADATA_SUFFIXES and path.name != "model.safetensors.index.json") \
                or path.name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            shutil.copyfile(path, partial / path.name)
            metadata.append(dict(name=path.name, sha256=file_hash(path)))
    from engine.base import kernel_shape
    from engine.kernels import cells
    shape = F.kernel_shape()
    verdicts = cells.admission(shape)
    record = kernel_shape.write_record(partial, shape, profile="qwen38", config_sha256=report["source_config_sha256"],
                                       admission=verdicts)
    report["kernel_shape"] = dict(name=record.name, sha256=file_hash(record), admission={v.lane: v.status for v in verdicts})
    report.update(metadata_files=metadata, seconds=round(time.monotonic() - started, 1), torch_version=torch.__version__,
                  source_script_sha256=file_hash(__file__))
    (partial / "preshard-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (partial / "SHA256SUMS").write_text("".join(row["sha256"] + "  " + row["name"] + "\n" for row in report["rank_files"]))
    os.rename(partial, out)
    print(json.dumps(dict(stage="complete", out=str(out), seconds=report["seconds"])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
