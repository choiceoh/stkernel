#!/usr/bin/env python3
"""Is the SSD engram lookup bit-identical to DeepSeek's own? No GPU, no model.

Every other piece of evidence for this module is about time -- how many IOPS,
how much stall. None of it says the rows come back RIGHT. This does, and it
does it against the vendor's code rather than against a second copy of ours:
`ParallelEngramEmbedding` is extracted verbatim from the checkpoint's
`inference/model.py` and run over the same synthetic table.

    python3 probes/dsv41_engram_diff.py [--rows 200000] [--tokens 64]

What is compared, per rank and then summed:

    reference   ParallelEngramEmbedding.forward(hash_ids)   -- fp8 rows in
                memory, out-of-block ids pointed at row 0 and zeroed
    ours        ShardEmbedding.readout(...)                 -- the same rows
                read from a per-rank file with O_DIRECT, out-of-block ids
                never read at all

The reference ends in `dist.all_reduce`, which in one process is just the sum
over ranks; both sides are summed the same way and the two totals must match
BIT for bit, not within a tolerance. A tolerance would hide exactly the bug
this is looking for -- an off-by-one in the row-block arithmetic returns a
neighbouring row, and neighbouring rows of a hash table are uncorrelated, so a
wrong row is a wrong answer rather than a small one.

Extraction rather than import: `model.py` pulls in the triton kernels, which
need a GPU. Taking the one class verbatim is the same thing
probes/mk_mla_prefill32_compile.py does with the CUDA sources.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_engram"))

from dsv41_engram import (EMB_ROW_DIM, FP8_BLOCK, SCALE_COLS,  # noqa: E402
                          EngramConfig, ShardEmbedding)
from dsv41_engram_io import ShardReader  # noqa: E402


def load_reference(model_py: Path, world_size: int, rank: int):
    """`ParallelEngramEmbedding`, verbatim, with the globals it reads."""
    text = model_py.read_text()
    start = text.index("class ParallelEngramEmbedding(nn.Module):")
    end = text.index("class Engram(nn.Module):", start)
    source = text[start:end]
    calls = source.count("dist.all_reduce")
    if calls != 1:
        raise SystemExit(
            f"the reference has {calls} all_reduce calls, not 1 -- the summing "
            f"this probe does in its place may no longer be equivalent")

    class _NoDist:
        @staticmethod
        def all_reduce(_tensor):
            # one process: the sum over ranks is done by the caller instead
            return None

    ns = {"nn": nn, "torch": torch, "F": F, "dist": _NoDist,
          "world_size": world_size, "rank": rank,
          "fp8_block_size": FP8_BLOCK, "scale_dtype": torch.float8_e8m0fnu}
    exec(compile(source, str(model_py), "exec"), ns)
    return ns["ParallelEngramEmbedding"], source


def build_table(rows: int, rng: random.Random):
    """A table whose rows are all distinct, so a wrong row cannot pass."""
    gen = torch.Generator().manual_seed(20260910)
    weight = torch.randint(0, 256, (rows, EMB_ROW_DIM), generator=gen,
                           dtype=torch.uint8)
    # 0x7F and 0xFF are the NaN encodings of e4m3fn. Random bytes hit them
    # about twice per 256, and a NaN compares unequal to itself -- which would
    # fail this probe for a reason that has nothing to do with the row-block
    # arithmetic it is testing, and would equally hide a real mismatch behind
    # a NaN. Real weights are finite; make these finite too.
    weight[(weight & 0x7F) == 0x7F] = 0x3C     # 1.0 in e4m3fn
    # e8m0 byte b is 2**(b-127); keep the exponents mild so the fp32 product
    # stays in a range where bf16 rounding is not the thing being tested
    scale = torch.randint(120, 135, (rows, SCALE_COLS), generator=gen,
                          dtype=torch.uint8)
    return weight, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_003,
                    help="table rows; a prime-ish size exercises the ragged "
                         "last block, which is where ceil() partitions break")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--hash-cols", type=int, default=24)
    ap.add_argument("--model-py", default=None,
                    help="inference/model.py from the checkpoint")
    args = ap.parse_args()

    model_py = Path(args.model_py) if args.model_py else None
    if model_py is None:
        for cand in (HERE / "probes/_dsv41_model.py",
                     Path("/home/choiceoh/models/DeepSeek-V4.1-Flash/inference/model.py")):
            if cand.is_file():
                model_py = cand
                break
    if model_py is None or not model_py.is_file():
        raise SystemExit(
            "need the checkpoint's inference/model.py (--model-py). Without it "
            "this probe would compare our code against our code.")

    rng = random.Random(4241)
    rows, world = args.rows, args.world_size
    weight, scale = build_table(rows, rng)
    # ids drawn over the whole table so every rank's block is hit, including
    # the short last one
    ids = torch.randint(0, rows, (args.tokens, args.hash_cols),
                        generator=torch.Generator().manual_seed(7)).long()

    print(f"table {rows:,} rows x {EMB_ROW_DIM}  world_size {world}  "
          f"ids {tuple(ids.shape)}")
    print(f"reference: {model_py}")

    ref_total = torch.zeros(args.tokens, args.hash_cols, EMB_ROW_DIM,
                            dtype=torch.bfloat16)
    ours_total = torch.zeros_like(ref_total)
    counts = []

    with tempfile.TemporaryDirectory(prefix="engram-diff-") as tmp:
        os.environ["ENGRAM_BACKEND"] = "ssd"
        os.environ["ENGRAM_SHARD_DIR"] = tmp
        for r in range(world):
            cfg = EngramConfig(rank=r, world_size=world, num_embeddings=rows)
            lo, n_local = cfg.vocab_start_idx, cfg.rows_on_disk()

            # -- reference: the padded block, rows past the table left at zero
            Ref, _src = load_reference(model_py, world, r)
            ref = Ref(rows, EMB_ROW_DIM)
            with torch.no_grad():
                w = torch.zeros(cfg.part_num_embeddings, EMB_ROW_DIM,
                                dtype=torch.uint8)
                s = torch.zeros(cfg.part_num_embeddings, SCALE_COLS,
                                dtype=torch.uint8)
                w[:n_local] = weight[lo:lo + n_local]
                s[:n_local] = scale[lo:lo + n_local]
                ref.weight.copy_(w.view(torch.float8_e4m3fn))
                ref.scale.copy_(s.view(torch.float8_e8m0fnu))
            ref_total += ref(ids)

            # -- ours: the same rows, on disk, read with O_DIRECT
            Path(cfg.shard_path(1)).write_bytes(
                weight[lo:lo + n_local].contiguous().numpy().tobytes())
            reader = ShardReader(cfg.shard_path(1), queue_depth=8)
            emb = ShardEmbedding(
                cfg, reader, scale[lo:lo + n_local].view(torch.float8_e8m0fnu))
            flat = ids.flatten().tolist()
            owned, gather = emb.submit(flat)
            counts.append(len(owned))
            ours_total += emb.readout((owned, gather), tuple(ids.shape))
            reader.close()

    served = sum(counts)
    print(f"rows served per rank: {counts}  total {served} "
          f"of {ids.numel()} ids")
    if served != ids.numel():
        raise SystemExit(
            f"FAIL: the four blocks served {served} of {ids.numel()} ids -- a "
            f"row-block partition must cover every id exactly once")

    nan_ref = int(torch.isnan(ref_total.float()).sum())
    if nan_ref:
        raise SystemExit(
            f"FAIL: the reference produced {nan_ref} NaNs -- the synthetic "
            f"table is supposed to be finite, so equality would be testing "
            f"NaN semantics rather than the partition")
    if not torch.equal(ref_total, ours_total):
        diff = (ref_total.float() - ours_total.float()).abs()
        bad = int((diff > 0).sum())
        where = (diff > 0).nonzero()[:3].tolist()
        print(f"FAIL: {bad} of {diff.numel()} elements differ; first {where}")
        return 1
    nz = int((ref_total != 0).sum())
    print(f"MATCH: bit-identical over {ref_total.numel():,} elements "
          f"({nz:,} non-zero)")
    print("  covers: block boundaries, the ragged last block, and ids whose "
          "row this rank does not own")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
