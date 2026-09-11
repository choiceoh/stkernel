#!/usr/bin/env python3
"""Does the PLE SSD table hand back the rows a rank's device table would have?

CPU-only. Builds a synthetic 6-shard checkpoint through tools/qwen38_ple_shard.py
(the real writer, not a copy), splits it four ways, and for every rank drives
`PleSsdTable.gather` -- the exact call `VocabParallelEmbedding.forward` makes
through the SSD embedding method -- with the ids that path produces: local
indices, duplicates, and the 0 that a masked (foreign) id becomes. Every row is
required to be byte-identical to the table row it names, in the order asked.

    python3 probes/qwen38_ple_ssd.py            # synthetic
    python3 probes/qwen38_ple_ssd.py --out /home/choiceoh/models/qwen38-ple-ssd \\
        --repo /home/choiceoh/models/qwen38-flash-next-nvfp4     # real blocks: 512 rows/rank

The real-block mode reads rows through the O_DIRECT reader and compares them
with the checkpoint bytes at the same global row -- the reader's sector
straddling (160 does not divide 512) is exercised on the real files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/qwen38_ple"))
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_engram"))
sys.path.insert(0, str(HERE / "tools"))

from dsv41_engram_io import ShardReader  # noqa: E402
from qwen38_ple_ssd import PLE_ROW_BYTES, PleSsdConfig, PleSsdTable  # noqa: E402
import qwen38_ple_shard as tool  # noqa: E402


def synthetic_repo(root: Path, n_shards: int, shard_rows: int) -> dict:
    """Two files, shards out of order, a decoy tensor first. Returns row->bytes."""
    rows = {}
    weight_map = {}
    files = {"a.safetensors": [1, 4, 0], "b.safetensors": [5, 2, 3]}
    for fname, idxs in files.items():
        head, blobs, off = {}, [], 0
        head["decoy"] = {"dtype": "U8", "shape": [333], "data_offsets": [0, 333]}
        blobs.append(b"\xee" * 333); off += 333
        for i in idxs:
            name = f"model.layers.1.ple.ple_embedding{tool.SHARD_KEY}{i}.weight"
            data = bytearray()
            for g in range(i * shard_rows, (i + 1) * shard_rows):
                row = bytes(((g * 7 + k * 13) & 0xFF) for k in range(PLE_ROW_BYTES))
                rows[g] = row
                data += row
            head[name] = {"dtype": "F8_E4M3", "shape": [shard_rows, PLE_ROW_BYTES],
                          "data_offsets": [off, off + len(data)]}
            blobs.append(bytes(data)); off += len(data)
            weight_map[name] = fname
        hb = json.dumps(head).encode()
        with open(root / fname, "wb") as fh:
            fh.write(struct.pack("<Q", len(hb))); fh.write(hb)
            for b in blobs:
                fh.write(b)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return rows


def check_rank(cfg: PleSsdConfig, want_row, n_ids: int, rng: random.Random, label: str) -> int:
    """want_row(global) -> bytes. Returns the number of problems."""
    reader = ShardReader(cfg.shard_path(), queue_depth=cfg.queue_depth, row_bytes=cfg.row_bytes)
    table = PleSsdTable(cfg, reader)
    n_local = cfg.rows_on_disk()
    # what the masking hands the method: local ids, many 0 (foreign), duplicates
    ids = [rng.randrange(n_local) for _ in range(n_ids)]
    for i in range(0, n_ids, 4):
        ids[i] = 0
    ids[1] = ids[2] = n_local - 1
    local = torch.tensor(ids, dtype=torch.int64).reshape(-1, 4)
    got = table.gather(local)
    table.close()
    bad = 0
    if tuple(got.shape) != (*local.shape, PLE_ROW_BYTES) or got.dtype != torch.uint8:
        print(f"  {label}: shape/dtype {tuple(got.shape)} {got.dtype}")
        return 1
    flat = got.reshape(-1, PLE_ROW_BYTES)
    for k, lid in enumerate(ids):
        want = want_row(cfg.vocab_start_idx + lid)
        if bytes(flat[k].tolist()) != want:
            print(f"  {label}: slot {k} local {lid} (global {cfg.vocab_start_idx + lid}) differs")
            bad += 1
            if bad > 3:
                break
    st = table.stats
    print(f"  {label}: {st['rows']} rows asked, {st['distinct']} distinct read, "
          f"{'OK' if not bad else 'FAIL'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--out")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--rows", type=int, default=512, help="ids per rank in real mode")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bad = 0
    if args.out and args.repo:
        shards = tool.locate(Path(args.repo))
        total = tool.table_rows(shards)
        def want_row(g: int) -> bytes:
            (path, start, _n), = tool.row_ranges(shards, g, 1)
            with open(path, "rb") as fh:
                fh.seek(start)
                return fh.read(PLE_ROW_BYTES)
        os.environ["DENEB_PLE_SSD"] = "1"; os.environ["DENEB_PLE_SSD_DIR"] = args.out
        for r in range(args.world_size):
            cfg = PleSsdConfig(rank=r, world_size=args.world_size, num_embeddings=total)
            bad += check_rank(cfg, want_row, args.rows, rng, f"real rank {r}")
    else:
        with tempfile.TemporaryDirectory(prefix="ple-ssd-probe-") as tmp:
            repo = Path(tmp) / "repo"; repo.mkdir()
            rows = synthetic_repo(repo, 6, 1_000)
            out = Path(tmp) / "out"
            ns = argparse.Namespace(repo=str(repo), out=str(out), world_size=args.world_size, only_rank=None)
            if tool.cmd_build(ns):
                return 1
            os.environ["DENEB_PLE_SSD"] = "1"; os.environ["DENEB_PLE_SSD_DIR"] = str(out)
            for r in range(args.world_size):
                cfg = PleSsdConfig(rank=r, world_size=args.world_size, num_embeddings=len(rows))
                bad += check_rank(cfg, rows.__getitem__, 400, rng, f"synthetic rank {r}")
    print("PASS" if not bad else "FAIL", "qwen38 PLE SSD table")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
