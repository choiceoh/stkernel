#!/usr/bin/env python3
"""Split the PLE n-gram table out of the checkpoint into per-rank row blocks.

The table is 128 shard tensors `...ngram_embedding.shard_{i}.weight`, each
[2,500,012 x 160] F8_E4M3, 47.7 GiB together. `VocabParallelEmbedding` gives
rank r of W the rows `[r * per, (r + 1) * per)`, `per = rows / W`; this writes
exactly that block to `ple-r{r}of{W}.weight` so a rank can read its rows off
SSD instead of holding 11.9 GiB of them (qwen38_ple_ssd.py).

    python3 tools/qwen38_ple_shard.py plan
    python3 tools/qwen38_ple_shard.py build  --out /home/choiceoh/models/qwen38-ple-ssd
    python3 tools/qwen38_ple_shard.py verify --out /home/choiceoh/models/qwen38-ple-ssd
    python3 tools/qwen38_ple_shard.py selftest      # synthetic, no checkpoint

The partition is imported from the overlay (`PleSsdConfig`), not repeated, so
the writer cannot drift from the reader. No safetensors dependency: a shard
file is an 8-byte header length, JSON, then raw bytes, and a tensor's rows are
one contiguous byte range in it -- the copy is seeks and reads, 64 MiB at a
time, never a load. `verify` samples rows back against the source bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/qwen38_ple"))
from qwen38_ple_ssd import PLE_ROW_BYTES, PleSsdConfig  # noqa: E402

GIB = float(1 << 30)
CHUNK = 64 << 20
SHARD_KEY = ".ngram_embedding.shard_"


def shard_header(path: Path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    return head, 8 + n


def locate(repo: Path):
    """shard index -> (file, rows, absolute byte start), in index order."""
    index = json.loads((repo / "model.safetensors.index.json").read_text())
    want = {n: s for n, s in index["weight_map"].items()
            if SHARD_KEY in n and n.endswith(".weight")}
    if not want:
        raise SystemExit("ABORT: the index lists no PLE shard tensors")
    out = {}
    headers = {}
    for name, shard in want.items():
        idx = int(name.split(SHARD_KEY, 1)[1].split(".", 1)[0])
        path = repo / shard
        if not path.is_file():
            out[idx] = None
            continue
        if path not in headers:
            headers[path] = shard_header(path)
        head, base = headers[path]
        meta = head[name]
        if meta["dtype"] != "F8_E4M3" or meta["shape"][1] != PLE_ROW_BYTES:
            raise SystemExit(f"ABORT: {name} is {meta['dtype']} {meta['shape']}, "
                             f"expected F8_E4M3 [.., {PLE_ROW_BYTES}]")
        out[idx] = (path, int(meta["shape"][0]), base + meta["data_offsets"][0])
    if sorted(out) != list(range(len(out))):
        raise SystemExit(f"ABORT: shard indices are not 0..{len(out) - 1}: "
                         f"{sorted(out)[:8]}...")
    return [out[i] for i in range(len(out))]


def table_rows(shards) -> int:
    return sum(rows for _p, rows, _s in shards)


def row_ranges(shards, first: int, count: int):
    """(file, absolute byte start, n_rows) pieces covering rows [first, first+count)."""
    base = 0
    for path, rows, start in shards:
        lo, hi = max(first, base), min(first + count, base + rows)
        if lo < hi:
            yield path, start + (lo - base) * PLE_ROW_BYTES, hi - lo
        base += rows


def copy_block(shards, first: int, count: int, dst: Path) -> int:
    written = 0
    with open(dst, "wb") as fout:
        for path, start, n in row_ranges(shards, first, count):
            remaining = n * PLE_ROW_BYTES
            with open(path, "rb") as fin:
                fin.seek(start)
                while remaining:
                    block = fin.read(min(CHUNK, remaining))
                    if not block:
                        raise SystemExit(f"ABORT: {path.name} ended {remaining} "
                                         f"bytes early -- truncated shard")
                    fout.write(block)
                    remaining -= len(block)
                    written += len(block)
    return written


def cmd_plan(args) -> int:
    shards = locate(Path(args.repo))
    missing = [i for i, s in enumerate(shards) if s is None]
    if missing:
        print(f"  NOT YET DOWNLOADED: shards {missing[:8]}{'...' if len(missing) > 8 else ''}")
        return 1
    rows = table_rows(shards)
    print(f"world_size {args.world_size}  ->  {args.out}")
    print(f"  {len(shards)} shards, {rows:,} rows x {PLE_ROW_BYTES} B = {rows * PLE_ROW_BYTES / GIB:.1f} GiB")
    for r in range(args.world_size):
        cfg = PleSsdConfig(rank=r, world_size=args.world_size, num_embeddings=rows)
        print(f"    rank {r}  rows [{cfg.vocab_start_idx:,}, {cfg.vocab_end_idx:,})  "
              f"{cfg.rows_on_disk() * PLE_ROW_BYTES / GIB:6.2f} GiB  -> {Path(cfg.shard_path()).name}")
    return 0


def cmd_build(args) -> int:
    repo, out = Path(args.repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = locate(repo)
    if any(s is None for s in shards):
        raise SystemExit("ABORT: a PLE shard is not downloaded; a block built from "
                         "a partial checkpoint reads as zeros where rows are missing")
    rows = table_rows(shards)
    done = 0
    for r in range(args.world_size):
        if args.only_rank is not None and r != args.only_rank:
            continue
        cfg = PleSsdConfig(rank=r, world_size=args.world_size, num_embeddings=rows)
        dst = out / Path(cfg.shard_path()).name
        got = copy_block(shards, cfg.vocab_start_idx, cfg.rows_on_disk(), dst)
        print(f"  rank {r}  {dst.name}  {got / GIB:6.2f} GiB", flush=True)
        done += 1
    print(f"  wrote {done} file(s)")
    return 0


def cmd_verify(args) -> int:
    """Boundaries plus a random sample of each block, against the source bytes."""
    import random
    repo, out = Path(args.repo), Path(args.out)
    shards = locate(repo)
    if any(s is None for s in shards):
        raise SystemExit("ABORT: a PLE shard is not downloaded")
    rows = table_rows(shards)
    rng = random.Random(args.seed)
    checked = bad = 0
    for r in range(args.world_size):
        if args.only_rank is not None and r != args.only_rank:
            continue
        cfg = PleSsdConfig(rank=r, world_size=args.world_size, num_embeddings=rows)
        n = cfg.rows_on_disk()
        dst = out / Path(cfg.shard_path()).name
        if not dst.is_file() or dst.stat().st_size != n * PLE_ROW_BYTES:
            print(f"  {dst.name}: {'missing' if not dst.is_file() else dst.stat().st_size} "
                  f"bytes, expected {n * PLE_ROW_BYTES}")
            bad += 1
            continue
        # every shard boundary inside the block, both sides, then random rows
        picks = {0, n - 1}
        base = 0
        for _p, srows, _s in shards:
            for g in (base - 1, base):
                if cfg.vocab_start_idx <= g < cfg.vocab_end_idx:
                    picks.add(g - cfg.vocab_start_idx)
            base += srows
        picks.update(rng.randrange(n) for _ in range(args.samples))
        with open(dst, "rb") as fd:
            for row in sorted(picks):
                (path, start, _n), = row_ranges(shards, cfg.vocab_start_idx + row, 1)
                with open(path, "rb") as fs:
                    fs.seek(start)
                    want = fs.read(PLE_ROW_BYTES)
                fd.seek(row * PLE_ROW_BYTES)
                if fd.read(PLE_ROW_BYTES) != want:
                    print(f"  {dst.name}: row {row} (global {cfg.vocab_start_idx + row}) differs")
                    bad += 1
                    break
                checked += 1
    print(f"  {checked:,} rows compared, {bad} problem(s)")
    return 1 if bad else 0


def cmd_selftest(args) -> int:
    """Build and verify against a synthetic checkpoint: 6 shards, uneven files."""
    import shutil
    import tempfile
    n_shards, shard_rows, world = 6, 1_000, args.world_size
    rows = n_shards * shard_rows
    with tempfile.TemporaryDirectory(prefix="ple-shard-selftest-") as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        weight_map = {}
        # shards land in two files, out of index order, behind a decoy tensor so
        # a builder that ignored data_offsets or file boundaries is caught
        files = {"a.safetensors": [1, 4, 0], "b.safetensors": [5, 2, 3]}
        for fname, idxs in files.items():
            head, blobs, off = {}, [], 0
            decoy = b"\xee" * 777
            head["decoy"] = {"dtype": "U8", "shape": [777], "data_offsets": [0, 777]}
            blobs.append(decoy); off += 777
            for i in idxs:
                name = f"model.layers.1.ple.ple_embedding{SHARD_KEY}{i}.weight"
                data = b"".join(bytes([g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, 0x5A])
                                * (PLE_ROW_BYTES // 4)
                                for g in range(i * shard_rows, (i + 1) * shard_rows))
                head[name] = {"dtype": "F8_E4M3", "shape": [shard_rows, PLE_ROW_BYTES],
                              "data_offsets": [off, off + len(data)]}
                blobs.append(data); off += len(data)
                weight_map[name] = fname
            hb = json.dumps(head).encode()
            with open(repo / fname, "wb") as fh:
                fh.write(struct.pack("<Q", len(hb))); fh.write(hb)
                for b in blobs:
                    fh.write(b)
        (repo / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        out = Path(tmp) / "out"
        ns = argparse.Namespace(repo=str(repo), out=str(out), world_size=world,
                                samples=64, seed=1, only_rank=None)
        print(f"selftest: {rows:,} rows in {n_shards} shards over 2 files, world_size {world}")
        if cmd_build(ns):
            return 1
        rc = cmd_verify(ns)
        total = 0
        for r in range(world):
            cfg = PleSsdConfig(rank=r, world_size=world, num_embeddings=rows)
            data = (out / Path(cfg.shard_path()).name).read_bytes()
            n = cfg.rows_on_disk(); total += n
            for row in range(n):
                g = cfg.vocab_start_idx + row
                want = bytes([g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, 0x5A]) * (PLE_ROW_BYTES // 4)
                if data[row * PLE_ROW_BYTES:(row + 1) * PLE_ROW_BYTES] != want:
                    print(f"  FAIL rank {r} row {row} (global {g})")
                    return 1
        if total != rows:
            print(f"  FAIL: blocks cover {total} rows, table has {rows}")
            return 1
        print(f"  every one of {total:,} rows exact, blocks cover the table exactly once")
        shutil.rmtree(out, ignore_errors=True)
        return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build", "verify", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default="/home/choiceoh/models/qwen38-flash-next-nvfp4")
        p.add_argument("--out", default="/home/choiceoh/models/qwen38-ple-ssd")
        p.add_argument("--world-size", type=int, default=4)
        if name in ("verify", "selftest"):
            p.add_argument("--samples", type=int, default=256)
            p.add_argument("--seed", type=int, default=1)
        p.add_argument("--only-rank", type=int, default=None)
    args = ap.parse_args()
    return {"plan": cmd_plan, "build": cmd_build,
            "verify": cmd_verify, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
