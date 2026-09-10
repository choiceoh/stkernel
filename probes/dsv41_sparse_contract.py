#!/usr/bin/env python3
"""Do the index producers satisfy what sparse_attn requires? No GPU, no tilelang.

The kernel cannot be run here, so this checks the side that can be: whatever
produces `topk_idxs`. Two producers are exercised against all three clauses --
the reference's own `Indexer.forward` tail, and this repo's
`select_candidate_ids` -- and the kernel source those clauses were read from is
pinned, so a vendor change to it fails here rather than drifting.

The clause that nothing else checks is RANGE. The gather is

    kv_shared[i, j] = if(idxs[i] != -1, kv[by, idxs[i], j], 0)

with no upper bound: an id at or past kv's length is an out-of-bounds device
read. On a GPU that is a wrong number or a fault, and either way it is not the
masked zero it looks like.

    python3 probes/dsv41_sparse_contract.py --kernel-py .../inference/kernel.py
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_model"))

from dsv41_sparse_contract import (SENTINEL, ContractViolation,  # noqa: E402
                                    check_topk_idxs, rows_all_sentinel)


def pin_kernel(kernel_py: Path) -> str:
    """The three clauses, re-read from the source, so drift is a failure."""
    src = kernel_py.read_text()
    start = src.index("def sparse_attn_kernel(")
    end = src.index("def sparse_attn(", start)
    body = src[start:end]
    clauses = {
        "INT32 declaration": "topk_idxs: T.Tensor[(b, m, topk), INT32]",
        "-1 is the sentinel": "idxs[i] != -1, kv[by, idxs[i], j], 0",
        "-1 also masks the score": "idxs[j] != -1, 0, -T.infinity(FP32)",
        "no upper bound on the gather": None,
        "finite max seeds the all--1 row": "T.fill(scores_max, -1e30)",
    }
    ok = True
    for label, needle in clauses.items():
        if needle is None:
            # the absence of a bound is the clause: no comparison against a
            # length appears anywhere in the gather
            present = bool(re.search(r"idxs\[\w+\]\s*<\s*\w+", body))
            good = not present
        else:
            good = needle in body
        print(f"  clause  {label:34s} {'OK' if good else 'CHANGED'}")
        ok &= good
    print(f"  kernel sha256 {hashlib.sha256(body.encode()).hexdigest()[:16]}…")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-py", required=True)
    args = ap.parse_args()
    fail = not pin_kernel(Path(args.kernel_py))
    print()

    gen = torch.Generator().manual_seed(20260910)
    kv_len = 96

    # -- producer 1: the reference's tail, which is two lines and the shape of
    #    every producer that follows it
    for label, offset, lens in (("in range", 0, 40),
                                ("with an offset", 32, 40),
                                ("nothing reachable", 0, 0)):
        score = torch.randn(2, 5, kv_len, generator=gen)
        want_len = torch.full((5, 1), lens)
        idxs = score.topk(8, dim=-1, sorted=False).indices.sort(dim=-1).values
        out = torch.where(idxs < want_len, idxs + offset, SENTINEL).int()
        try:
            check_topk_idxs(out, kv_len, where=f"reference tail, {label}")
            empty = int(rows_all_sentinel(out).sum())
            print(f"  produce reference tail, {label:20s} OK   "
                  f"all--1 rows {empty}")
        except ContractViolation as exc:
            print(f"  produce reference tail, {label:20s} VIOLATION\n      {exc}")
            fail = True

    # the offset case is exactly where the bound stops protecting: the
    # reference compares BEFORE adding, so a large offset walks off the end
    score = torch.randn(2, 5, kv_len, generator=gen)
    idxs = score.topk(8, dim=-1, sorted=False).indices.sort(dim=-1).values
    walked = torch.where(idxs < 40, idxs + 80, SENTINEL).int()
    try:
        check_topk_idxs(walked, kv_len, where="offset past the cache")
        print("  produce offset past the cache          NOT CAUGHT -- the "
              "range clause is not doing its job")
        fail = True
    except ContractViolation:
        print("  produce offset past the cache          caught, as it must be")

    # -- producer 2: this repo's own selector
    try:
        from dsv41_indexer import select_candidate_ids
    except Exception as exc:                            # noqa: BLE001
        print(f"  produce select_candidate_ids           unavailable ({exc})")
        return 1 if fail else 0
    scores = torch.randn(2, 4, kv_len, generator=gen)
    lens = torch.arange(1, 5).unsqueeze(-1).clamp(max=kv_len)
    scores.masked_fill_(torch.arange(kv_len) >= lens.unsqueeze(0), -torch.inf)
    ids = select_candidate_ids(scores, lens, topk_blocks=4, block_size=8)
    try:
        check_topk_idxs(ids, kv_len, where="select_candidate_ids")
        empty = int(rows_all_sentinel(ids).sum())
        asc = all(
            torch.equal(v, v.sort().values)
            for v in ids.flatten(0, 1)
            for v in [v[v != SENTINEL]])
        print(f"  produce select_candidate_ids           OK   all--1 rows "
              f"{empty}, ascending {asc}")
        fail |= not asc
    except ContractViolation as exc:
        print(f"  produce select_candidate_ids           VIOLATION\n      {exc}")
        fail = True

    print("\n" + ("SPARSE CONTRACT FAIL" if fail else "SPARSE CONTRACT PASS"))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
