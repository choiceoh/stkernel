#!/usr/bin/env python3
"""Split the engram tables out of the checkpoint into per-rank files.

The two tables are 188.8 GiB of the 475.2 GiB checkpoint and every rank reads
about a quarter of the rows. This writes that quarter to its own file so a node
holds 47.2 GiB instead of 188.8, which is one of the two things that gets
DeepSeek-V4.1-Flash onto srv1 at all (the other is expert parallelism; see
tools/dsv41_preshard.py).

    python3 tools/dsv41_engram_shard.py plan
    python3 tools/dsv41_engram_shard.py build  --out /home/choiceoh/models/...-engram
    python3 tools/dsv41_engram_shard.py verify --out /home/choiceoh/models/...-engram
    python3 tools/dsv41_engram_shard.py selftest      # synthetic, no checkpoint

The partition is not chosen here. `EngramConfig` in the dsv41_engram overlay
carries the reference's arithmetic -- `part = ceil(rows / world_size)`, rank r
owning `[r*part, (r+1)*part)` -- and this tool imports it rather than repeating
it, so the writer cannot drift from the reader. probes/dsv41_engram_diff.py
holds that reader to being bit-identical to DeepSeek's own module.

No safetensors dependency: a shard file is an 8-byte little-endian header
length, that many bytes of JSON, then the raw tensor bytes. A row range is a
contiguous byte range inside it, so the copy is a seek and a read rather than a
load -- which matters when one tensor is 91.5 GiB and the host has 120.

`verify` re-reads what was written and compares it against the source. That is
not the same check as the differential probe: the probe proves the READER agrees
with the reference given a correct file, and this proves the file is correct.
Both are needed, and neither implies the other.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_engram"))

from dsv41_engram import (EMB_ROW_DIM, SCALE_COLS,  # noqa: E402
                          EngramConfig)

GIB = float(1 << 30)
CHUNK = 64 << 20          # copy granularity; the source rows are contiguous


def shard_header(path: Path):
    """(header dict, byte offset where tensor data starts)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    return head, 8 + n


def locate(repo: Path):
    """Every engram embed tensor: name -> (file, dtype, shape, absolute start)."""
    index = json.loads((repo / "model.safetensors.index.json").read_text())
    want = {n: s for n, s in index["weight_map"].items()
            if ".engram.embed." in n}
    if not want:
        raise SystemExit("ABORT: the index lists no engram embed tensors")
    out = {}
    for name, shard in sorted(want.items()):
        path = repo / shard
        if not path.is_file():
            out[name] = None
            continue
        head, base = shard_header(path)
        meta = head[name]
        out[name] = (path, meta["dtype"], tuple(meta["shape"]),
                     base + meta["data_offsets"][0])
    return out


def layer_of(name: str) -> int:
    return int(name.split(".")[1])


def copy_rows(src: Path, src_start: int, row_bytes: int, first_row: int,
              n_rows: int, dst: Path) -> int:
    """Rows [first_row, first_row + n_rows) of one tensor into its own file."""
    remaining = n_rows * row_bytes
    offset = src_start + first_row * row_bytes
    written = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fin.seek(offset)
        while remaining:
            block = fin.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit(
                    f"ABORT: {src.name} ended {remaining} bytes early at "
                    f"row {first_row + written // row_bytes} -- the shard is "
                    f"truncated or the offset is wrong")
            fout.write(block)
            remaining -= len(block)
            written += len(block)
    return written


def plan_rows(tensors, world: int):
    """(layer, weight meta, scale meta, rows) per engram layer."""
    layers = {}
    for name, meta in tensors.items():
        layers.setdefault(layer_of(name), {})[
            "scale" if name.endswith(".scale") else "weight"] = (name, meta)
    for layer in sorted(layers):
        pair = layers[layer]
        if set(pair) != {"weight", "scale"}:
            raise SystemExit(f"ABORT: layer {layer} has {sorted(pair)}, "
                             f"not both weight and scale")
        yield layer, pair["weight"], pair["scale"]


def cmd_plan(args) -> int:
    repo = Path(args.repo)
    tensors = locate(repo)
    missing = [n for n, m in tensors.items() if m is None]
    total = 0.0
    print(f"world_size {args.world_size}  ->  {args.out}")
    for layer, (wn, wm), (sn, sm) in plan_rows(tensors, args.world_size):
        if wm is None or sm is None:
            print(f"  layer {layer:2d}  (shard not downloaded yet)")
            continue
        rows = wm[2][0]
        for r in range(args.world_size):
            cfg = EngramConfig(rank=r, world_size=args.world_size,
                               num_embeddings=rows)
            n = cfg.rows_on_disk()
            wb = n * EMB_ROW_DIM
            sb = n * SCALE_COLS
            total += (wb + sb) / GIB
            if r == 0:
                print(f"  layer {layer:2d}  {rows:,} rows, {wm[1]}/{sm[1]}")
            print(f"    rank {r}  rows [{cfg.vocab_start_idx:,}, "
                  f"{cfg.vocab_start_idx + n:,})  "
                  f"{wb / GIB:6.2f} GiB weight + {sb / GIB:.2f} GiB scale")
    print(f"  total written {total:.1f} GiB "
          f"({total / args.world_size:.1f} GiB per rank)")
    if missing:
        print(f"  NOT YET DOWNLOADED: {len(missing)} tensor(s) "
              f"-- {', '.join(sorted(set(m.split('.')[1] for m in missing)))}")
        return 1
    return 0


def cmd_build(args) -> int:
    repo, out = Path(args.repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors = locate(repo)
    if any(m is None for m in tensors.values()):
        raise SystemExit(
            "ABORT: an engram shard is not downloaded. Building from a partial "
            "checkpoint would write a file that reads as zeros where the rows "
            "are missing, which nothing downstream can detect.")
    done = 0
    for layer, (wn, wm), (sn, sm) in plan_rows(tensors, args.world_size):
        if args.only_layer is not None and layer != args.only_layer:
            continue
        wpath, _wd, wshape, wstart = wm
        spath, _sd, sshape, sstart = sm
        rows = wshape[0]
        if sshape[0] != rows:
            raise SystemExit(f"ABORT: {wn} has {rows} rows and {sn} has "
                             f"{sshape[0]}; they index together")
        if wshape[1] != EMB_ROW_DIM or sshape[1] != SCALE_COLS:
            raise SystemExit(
                f"ABORT: {wn} is {wshape} and {sn} is {sshape}; expected "
                f"[.., {EMB_ROW_DIM}] and [.., {SCALE_COLS}]")
        for r in range(args.world_size):
            if args.only_rank is not None and r != args.only_rank:
                continue
            cfg = EngramConfig(rank=r, world_size=args.world_size,
                               num_embeddings=rows)
            n = cfg.rows_on_disk()
            for src, start, row_bytes, dst in (
                    (wpath, wstart, EMB_ROW_DIM, Path(cfg.shard_path(layer))),
                    (spath, sstart, SCALE_COLS, Path(cfg.scale_path(layer)))):
                dst = out / Path(dst).name
                got = copy_rows(src, start, row_bytes,
                                cfg.vocab_start_idx, n, dst)
                print(f"  layer {layer:2d} rank {r}  {dst.name}  "
                      f"{got / GIB:6.2f} GiB", flush=True)
                done += 1
    print(f"  wrote {done} files")
    return 0


def cmd_verify(args) -> int:
    """Sample rows out of each shard and compare against the source bytes."""
    import random

    repo, out = Path(args.repo), Path(args.out)
    tensors = locate(repo)
    if any(m is None for m in tensors.values()):
        raise SystemExit("ABORT: an engram shard is not downloaded")
    rng = random.Random(args.seed)
    checked = bad = 0
    for layer, (wn, wm), (sn, sm) in plan_rows(tensors, args.world_size):
        if args.only_layer is not None and layer != args.only_layer:
            continue
        rows = wm[2][0]
        for r in range(args.world_size):
            if args.only_rank is not None and r != args.only_rank:
                continue
            cfg = EngramConfig(rank=r, world_size=args.world_size,
                               num_embeddings=rows)
            n = cfg.rows_on_disk()
            for (src, start, row_bytes, name) in (
                    (wm[0], wm[3], EMB_ROW_DIM, Path(cfg.shard_path(layer)).name),
                    (sm[0], sm[3], SCALE_COLS, Path(cfg.scale_path(layer)).name)):
                dst = out / name
                want_bytes = n * row_bytes
                if dst.stat().st_size != want_bytes:
                    print(f"  {name}: {dst.stat().st_size} bytes, "
                          f"expected {want_bytes}")
                    bad += 1
                    continue
                # always the boundaries, then a random interior sample: an
                # off-by-one in the block arithmetic lands on a boundary
                picks = [0, n - 1] + [rng.randrange(n)
                                      for _ in range(args.samples)]
                with open(src, "rb") as fs, open(dst, "rb") as fd:
                    for row in picks:
                        fs.seek(start + (cfg.vocab_start_idx + row) * row_bytes)
                        fd.seek(row * row_bytes)
                        if fs.read(row_bytes) != fd.read(row_bytes):
                            print(f"  {name}: row {row} (global "
                                  f"{cfg.vocab_start_idx + row}) differs")
                            bad += 1
                            break
                        checked += 1
    print(f"  {checked:,} rows compared, {bad} problem(s)")
    return 1 if bad else 0


def cmd_selftest(args) -> int:
    """Build and verify against a synthetic checkpoint. No download needed."""
    import shutil
    import tempfile

    rows, world = 100_003, args.world_size
    with tempfile.TemporaryDirectory(prefix="engram-shard-selftest-") as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        # one shard carrying both tensors, with the second offset non-zero so
        # a builder that ignored data_offsets would be caught
        wbytes = rows * EMB_ROW_DIM
        sbytes = rows * SCALE_COLS
        head = {
            "layers.1.engram.embed.weight":
                {"dtype": "F8_E4M3", "shape": [rows, EMB_ROW_DIM],
                 "data_offsets": [0, wbytes]},
            "layers.1.engram.embed.scale":
                {"dtype": "F8_E8M0", "shape": [rows, SCALE_COLS],
                 "data_offsets": [wbytes, wbytes + sbytes]},
        }
        blob = json.dumps(head).encode()
        shard = repo / "model-00001-of-00001.safetensors"
        with open(shard, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            for row in range(rows):
                fh.write(bytes([row & 0xFF, (row >> 8) & 0xFF])
                         * (EMB_ROW_DIM // 2))
            for row in range(rows):
                fh.write(bytes([(row * 7) & 0xFF]) * SCALE_COLS)
        (repo / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {n: shard.name for n in head}}))

        out = Path(tmp) / "out"
        ns = argparse.Namespace(repo=str(repo), out=str(out),
                                world_size=world, samples=64, seed=1,
                                only_layer=None, only_rank=None)
        print(f"selftest: {rows:,} rows, world_size {world}")
        if cmd_build(ns):
            return 1
        rc = cmd_verify(ns)
        # Every row of BOTH files, against values computed here rather than
        # re-read from the source. `verify` reads the source through the same
        # offset arithmetic the build used, so it agrees with itself when that
        # arithmetic is wrong -- an injected "ignore data_offsets[0]" passed it,
        # and passed an earlier version of this loop too, because that version
        # only checked the weight file (whose offset happens to be 0).
        total = 0
        for r in range(world):
            cfg = EngramConfig(rank=r, world_size=world, num_embeddings=rows)
            n = cfg.rows_on_disk()
            total += n
            wdata = (out / Path(cfg.shard_path(1)).name).read_bytes()
            sdata = (out / Path(cfg.scale_path(1)).name).read_bytes()
            for row in range(n):
                g = cfg.vocab_start_idx + row
                want_w = bytes([g & 0xFF, (g >> 8) & 0xFF]) * (EMB_ROW_DIM // 2)
                if wdata[row * EMB_ROW_DIM:(row + 1) * EMB_ROW_DIM] != want_w:
                    print(f"  FAIL weight rank {r} row {row} (global {g})")
                    return 1
                want_s = bytes([(g * 7) & 0xFF]) * SCALE_COLS
                if sdata[row * SCALE_COLS:(row + 1) * SCALE_COLS] != want_s:
                    print(f"  FAIL scale rank {r} row {row} (global {g}): "
                          f"got {sdata[row * SCALE_COLS]:#04x}, "
                          f"want {want_s[0]:#04x} -- a wrong tensor offset "
                          f"reads the weight table here")
                    return 1
        if total != rows:
            print(f"  FAIL: the blocks cover {total} rows, table has {rows}")
            return 1
        print(f"  every one of {total:,} rows exact in BOTH weight and scale, "
              f"and the blocks cover the table exactly once")
        shutil.rmtree(out, ignore_errors=True)
        return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build", "verify", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--repo",
                       default="/home/choiceoh/models/DeepSeek-V4.1-Flash")
        p.add_argument("--out",
                       default="/home/choiceoh/models/DeepSeek-V4.1-Flash-engram")
        p.add_argument("--world-size", type=int, default=4)
        if name in ("verify", "selftest"):
            p.add_argument("--samples", type=int, default=64)
            p.add_argument("--seed", type=int, default=1)
        # One rank of one layer is 22.9 GiB and proves the machinery on real
        # tables; all eight files are 183 GiB that nothing can load until an
        # image exists.
        p.add_argument("--only-layer", type=int, default=None)
        p.add_argument("--only-rank", type=int, default=None)
    args = ap.parse_args()
    return {"plan": cmd_plan, "build": cmd_build,
            "verify": cmd_verify, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
