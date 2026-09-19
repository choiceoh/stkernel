"""Preshard Qwen3.8-Flash-Next's ModelOpt NVFP4 checkpoint into four TEP=4 rank files and four PLE table files (profile).

    python3 -m engine.profiles.qwen38.preshard --ckpt /home/choiceoh/models/qwen38-flash-next-nvidia-nvfp4 \\
        --out /home/choiceoh/models/st-qwen38-tep4 --source-revision fc694b54fb0174e0913e6adf86691ef85a4ead47 [--plan]

The layout is specs.py's (merged projections, whole experts on their rank, the MTP head's experts re-encoded as NVFP4
from whichever encoding the checkpoint keeps). One group at a time is read (a layer's dense tensors once for all four
ranks; its routed experts a rank at a time, so a quarter of a layer's experts is the most held), every rank's tensors
built and streamed into its file (base/preshard.RankWriter: headers first, 256-byte aligned). Each file is read back
and every tensor's bytes compared with what was written.

The PLE table goes beside the rank files, not into them (the operator's decision of 2026-09-18; ple_table.py): rank r's
`ple-r{r}of4.weight` is the checkpoint's 32 shards of its vocabulary range written back to back, one shard read at a
time, hashed as written and read back whole against that hash, and `ple-r{r}of4.json` says what the file is. The two
checkpoint copies on srv2 hold the same table bytes, so rank 0's file equals the older `qwen38-ple-ssd/ple-r0of4.weight`.

The checkpoint's metadata files are copied beside the ranks and the shape wizard's record is written
(base/kernel_shape.write_record): what a boot binds (fleet.py). The output directory is new and immutable: built as
`<out>.incomplete` and renamed when every check passed. `--layers a-b` builds a development subset (no completeness
check; a boot serves those layers).
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
from engine.profiles.qwen38.ple_table import write_sidecar  # noqa: E402

METADATA_SUFFIXES = (".json", ".jinja", ".md")
PIECE = 64 << 20


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def file_hash(path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 << 20):
            value.update(block)
    return value.hexdigest()


def parse_layers(text: "str | None", count: int) -> "list[int] | None":
    if not text:
        return None
    lo, _, hi = text.partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))
    if not layers or layers[0] < 0 or layers[-1] >= count:
        raise ValueError(f"--layers {text}: the model has layers 0..{count - 1}")
    return layers


def ple_plan(F, ck: Checkpoint) -> dict:
    """The table as the checkpoint stores it, checked against what the profile derives (D3): every shard's header
    says [rows_per_shard, width] e4m3."""
    L = F.ple_layers[0]
    rows, width = F.ple_rows_per_shard, F.ple_head_dim
    for rank in range(facts.TP):
        for name in layout.ple_shards(F, L, rank):
            entry = ck.reader(ck.weight_map[name]).header[name]
            if entry["dtype"] != "F8_E4M3" or entry["shape"] != [rows, width]:
                raise ValueError(f"{name}: {entry['dtype']} {entry['shape']}, the profile derives F8_E4M3 [{rows}, {width}]")
    return dict(layer=L, shards_per_rank=F.ngram_parts // facts.TP, rows_per_shard=rows, width=width,
                rows_per_rank=F.ple_rows_per_rank, bytes_per_rank=F.ple_rows_per_rank * width,
                scale_name=layout.ple_table_name(F, L) + ".weight_scale")


def plan(ckpt, layers=None):
    F = facts.load(ckpt)
    groups = list(layout.groups(F, layers))
    ck = Checkpoint(str(ckpt))
    index = ck.weight_map
    sources = {key for _, keys, _ in groups for key in keys}
    missing = sources - set(index)
    if missing:
        raise ValueError(("checkpoint tensors the layout reads are missing", sorted(missing)[:8]))
    ple = ple_plan(F, ck)
    table = {name for rank in range(facts.TP) for name in layout.ple_shards(F, ple["layer"], rank)} | {ple["scale_name"]}
    missing = table - set(index)
    if missing:
        raise ValueError(("PLE table tensors are missing", sorted(missing)[:8]))
    unread = sorted(k for k in index if not k.startswith("model.visual.") and k not in sources and k not in table)
    specs = [s for _, _, of in groups for s in of(0)]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("duplicate output tensor")
    report = dict(weight_layout=F.weight_layout, world=facts.TP, tensors_per_rank=len(specs),
                  payload_bytes_per_rank=sum(s.nbytes() for s in specs), source_tensors=len(sources) + len(table),
                  ple=ple, mtp_experts=F.mtp_experts, layers="all" if layers is None else layers,
                  unread_text_tensors=unread[:32], unread_text_count=len(unread),
                  source_config_sha256=file_hash(Path(ckpt) / "config.json"),
                  source_index_sha256=file_hash(Path(ckpt) / "model.safetensors.index.json"))
    return F, groups, report, ck


def write_ranks(F, groups, ck, partial: Path, report: dict, metadata: dict, started: float) -> list:
    all_specs = [s for _, _, of in groups for s in of(0)]
    writers = [RankWriter(partial / f"rank{r}of4.safetensors", all_specs, dict(metadata, rank=r))
               for r in range(facts.TP)]
    hashes = [{} for _ in writers]
    for label, keys, specs_of in groups:
        source = ck.views(keys)                   # mapped, not staged: a group costs its built tensors, not its sources
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
    return hashes


def verify_ranks(F, groups, partial: Path, hashes: list, report: dict) -> None:
    report["rank_files"] = []
    for r in range(facts.TP):
        path = partial / f"rank{r}of4.safetensors"
        reader = RankLoader(path)
        if reader.metadata["weight_layout"] != F.weight_layout or reader.metadata["rank"] != str(r):
            raise AssertionError("rank identity mismatch")
        for label, _keys, specs_of in groups:
            names = [s.name for s in specs_of(r)]
            if not names:
                continue
            loaded = reader.load(names, device="cpu", max_run=32 << 20)
            for spec in specs_of(r):
                got = loaded[spec.name]
                if tuple(got.shape) != tuple(spec.shape) or got.dtype != spec.dtype or tensor_hash(got) != hashes[r][spec.name]:
                    raise AssertionError(("rank readback mismatch", r, spec.name))
            del loaded
        entry = dict(name=path.name, bytes=path.stat().st_size, sha256=file_hash(path), tensors_verified=len(hashes[r]))
        report["rank_files"].append(entry)
        print(json.dumps(dict(stage="verify", rank=r, **entry)), flush=True)


def write_tables(F, ck: Checkpoint, partial: Path, report: dict, source_revision: str, started: float,
                 shard_limit: "int | None" = None) -> None:
    """Rank r's table file: its shards' bytes back to back, hashed as written, read back whole against that hash.
    `shard_limit` (development) writes only the first shards of each rank's range: the sidecar says so, and a boot
    refuses such a table (ple_table.PLETable.open checks the row count)."""
    ple = report["ple"]
    L, width = ple["layer"], ple["width"]
    scale = float(ck.views([ple["scale_name"]])[ple["scale_name"]].float().reshape(()))
    report["ple_files"] = []
    for r in range(facts.TP):
        shards = layout.ple_shards(F, L, r)
        if shard_limit is not None:
            shards = shards[:shard_limit]
            ple["bytes_per_rank"] = len(shards) * ple["rows_per_shard"] * width
            ple["rows_per_rank"] = len(shards) * ple["rows_per_shard"]
            ple["dev_shards"] = shard_limit
        path = partial / facts.ple_file(r)
        digest = hashlib.sha256()
        written = 0
        with path.open("wb") as out:
            for name in shards:
                t = ck.views([name])[name]
                if t.dtype != torch.float8_e4m3fn or tuple(t.shape) != (ple["rows_per_shard"], width):
                    raise ValueError(f"{name}: {t.dtype} {tuple(t.shape)}")
                raw = memoryview(t.view(torch.uint8).numpy().reshape(-1))       # the mapped bytes themselves
                out.write(raw)
                digest.update(raw)
                written += len(raw)
                del t, raw
        if written != ple["bytes_per_rank"] or path.stat().st_size != written:
            raise AssertionError(f"{path.name}: wrote {written} bytes, expected {ple['bytes_per_rank']}")
        check = hashlib.sha256()
        with path.open("rb") as back:
            while piece := back.read(PIECE):
                check.update(piece)
        if check.hexdigest() != digest.hexdigest():
            raise AssertionError(f"{path.name}: read back differs from what was written")
        entry = dict(name=path.name, bytes=written, sha256=digest.hexdigest(), rows=ple["rows_per_rank"], width=width)
        write_sidecar(partial / facts.ple_sidecar(r), layout=F.weight_layout, model="qwen38", world=facts.TP, rank=r,
                      layer=L, rows=ple["rows_per_rank"], width=width, dtype="F8_E4M3", rows_per_shard=ple["rows_per_shard"],
                      shards=shards, scale=scale, scale_name=ple["scale_name"], bytes=written, sha256=entry["sha256"],
                      source_revision=source_revision, seconds=round(time.monotonic() - started, 1),
                      **({"dev_shards": shard_limit} if shard_limit is not None else {}))
        report["ple_files"].append(entry)
        print(json.dumps(dict(stage="ple", rank=r, **entry, seconds=round(time.monotonic() - started, 1))), flush=True)
    report["ple"]["scale"] = scale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, default=facts.CKPT)
    ap.add_argument("--out", type=Path, default=facts.RANKS)
    ap.add_argument("--plan", action="store_true", help="read the headers and print the plan; write nothing")
    ap.add_argument("--source-revision", help="the checkpoint's revision, recorded in the rank files (required but with --vision)")
    ap.add_argument("--layers", help="a-b: a development subset of the layers (no completeness check)")
    ap.add_argument("--ple-shards", type=int, help="development: only the first N shards of each rank's table range")
    ap.add_argument("--threads", type=int, default=4, help="torch threads while encoding")
    ap.add_argument("--vision", action="store_true",
                    help="write only vision.safetensors into the existing rank directory: the vision tower, whole, for every rank")
    a = ap.parse_args(argv)
    if a.vision:
        from engine.profiles.qwen38 import vision
        if not a.out.is_dir():
            raise ValueError(f"--vision writes next to existing rank files: {a.out} is not a directory")
        print(f"  qwen38 preshard: vision tower -> {a.out / vision.FILE}", flush=True)
        size = vision.write_file(a.ckpt, a.out)
        print(f"  done: {size / 2**30:.2f} GiB", flush=True)
        return 0
    if not a.source_revision:
        ap.error("--source-revision is required")
    if a.ple_shards is not None and (a.layers is None or a.ple_shards <= 0):
        raise ValueError("--ple-shards is a development option that goes with --layers, and needs at least one shard")
    layers = parse_layers(a.layers, facts.load(a.ckpt).layers)
    F, groups, report, ck = plan(a.ckpt, layers)
    report["source_revision"] = a.source_revision
    print(json.dumps(report), flush=True)
    if layers is None and report["unread_text_count"]:
        raise ValueError(f"{report['unread_text_count']} text tensors no spec reads: {report['unread_text_tensors'][:8]}")
    if a.plan:
        return 0
    out = a.out.resolve()
    partial = out.with_name(out.name + ".incomplete")
    if out.exists() or partial.exists():
        raise ValueError("the output must be a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    table_bytes = report["ple"]["bytes_per_rank"] if a.ple_shards is None else \
        a.ple_shards * report["ple"]["rows_per_shard"] * report["ple"]["width"]
    required = facts.TP * (report["payload_bytes_per_rank"] + table_bytes) + (2 << 30 if layers else 8 << 30)
    if shutil.disk_usage(out.parent).free < required:
        raise ValueError(f"insufficient disk space: {required / 2**30:.1f} GiB for four ranks, four tables and a reserve")
    partial.mkdir()
    torch.set_num_threads(a.threads)
    started = time.monotonic()
    metadata = dict(model="qwen38", world=facts.TP, weight_layout=F.weight_layout, source_revision=a.source_revision,
                    scale_convention="modelopt-multiplier", mtp_experts=F.mtp_experts, ple="ssd",
                    layers="all" if layers is None else f"{layers[0]}-{layers[-1]}")
    hashes = write_ranks(F, groups, ck, partial, report, metadata, started)
    verify_ranks(F, groups, partial, hashes, report)
    write_tables(F, ck, partial, report, a.source_revision, started, shard_limit=a.ple_shards)
    copied = []
    for path in sorted(a.ckpt.iterdir()):
        if (path.is_file() and path.suffix in METADATA_SUFFIXES and path.name != "model.safetensors.index.json") \
                or path.name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            shutil.copyfile(path, partial / path.name)
            copied.append(dict(name=path.name, sha256=file_hash(path)))
    from engine.base import kernel_shape
    from engine.kernels import cells
    shape = F.kernel_shape()
    verdicts = cells.admission(shape)
    record = kernel_shape.write_record(partial, shape, profile="qwen38", config_sha256=report["source_config_sha256"],
                                       admission=verdicts)
    report["kernel_shape"] = dict(name=record.name, sha256=file_hash(record), admission={v.lane: v.status for v in verdicts})
    report.update(metadata_files=copied, seconds=round(time.monotonic() - started, 1), torch_version=torch.__version__,
                  source_script_sha256=file_hash(__file__))
    (partial / "preshard-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (partial / "SHA256SUMS").write_text("".join(row["sha256"] + "  " + row["name"] + "\n"
                                                for row in report["rank_files"] + report["ple_files"]))
    os.rename(partial, out)
    print(json.dumps(dict(stage="complete", out=str(out), seconds=report["seconds"])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
