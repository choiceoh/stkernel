#!/usr/bin/env python3
"""Does the pre-shard loader read back what the builder wrote? No GPU, no model.

`tools/dsv41_preshard.py` writes a rank file and `dsv41_preshard_load` reads
one. They never run together -- one on srv4 today, the other inside a container
later -- so the only thing keeping them agreed is a test that holds both to the
same partition. That is what this is.

    1. bijection. Over every global expert id, expert_rank -> local_expert must
       be one-to-one onto [0, per_rank) for each rank, and the union over ranks
       must reconstruct the global set exactly. A partition that loses or
       doubles an expert passes every byte check ever written.
    2. name sets. For each rank, `localize` over the names the builder selects
       must give that rank's local names, with no collisions, and the ranks'
       global name sets must be disjoint and cover the index.
    3. refusals. Each of these is a file that opens, has correct dtypes and
       shapes, and is wrong:
          rank 1's file under rank 0's name
          a file built for world size 8, read by a TP=4 job
          an expert this rank does not own
          no __metadata__ at all
    4. the real rank file, if one has been built: its header's names must be
       exactly what the loader expects for that rank.

    python3 probes/dsv41_preshard_load.py [--out .../DeepSeek-V4.1-Flash-tp4]

Only headers are read, never tensors, so (4) costs a few milliseconds against
an 85 GiB file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_model"))

from dsv41_layers import expert_rank                       # noqa: E402
from dsv41_loader import wanted_by                          # noqa: E402
from dsv41_preshard_load import (                           # noqa: E402
    Layout, PreshardMismatch, local_expert, localize, read_header, read_layout,
    require,
)

DEFAULT_REPO = "/home/choiceoh/models/DeepSeek-V4.1-Flash"


def check(label: str, fn) -> bool:
    """A refusal is the pass condition."""
    try:
        fn()
    except PreshardMismatch as exc:
        print(f"  refuses {label:44s} OK  ({str(exc).split('.')[0][:60]})")
        return True
    print(f"  refuses {label:44s} FAIL -- accepted it")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default="/home/choiceoh/models/DeepSeek-V4.1-Flash-tp4")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--dspark-experts", type=int, default=128)
    ap.add_argument("--mtp", choices=("replicate", "ep"), default="replicate",
                    help="must match how the rank file was built")
    ap.add_argument("--allow-unstated", action="store_true",
                    help="accept a rank file built before __metadata__ existed")
    args = ap.parse_args()
    W, N = args.world_size, args.experts
    ok = True

    # -- 1. the partition is a bijection ----------------------------------
    per_rank = N // W
    seen = {}
    for e in range(N):
        r = expert_rank(e, N, W)
        l = local_expert(e, N, W, r)
        if not 0 <= l < per_rank:
            print(f"  FAIL: expert {e} -> rank {r} local {l}, outside "
                  f"[0, {per_rank})")
            return 1
        if (r, l) in seen:
            print(f"  FAIL: rank {r} slot {l} claimed by experts "
                  f"{seen[(r, l)]} and {e}")
            return 1
        seen[(r, l)] = e
    if len(seen) != N:
        print(f"  FAIL: {len(seen)} slots for {N} experts")
        return 1
    print(f"  bijection  {N} experts -> {W} ranks x {per_rank} slots, "
          f"one-to-one and onto")

    # -- 2. name sets against the real index ------------------------------
    index_path = Path(args.repo) / "model.safetensors.index.json"
    if index_path.is_file():
        names = list(json.loads(index_path.read_text())["weight_map"])
        globals_by_rank, local_by_rank = {}, {}
        for r in range(W):
            mine = [n for n in names
                    if wanted_by(n, r, W, N, args.dspark_experts,
                                 mtp=args.mtp)]
            globals_by_rank[r] = set(mine)
            loc = [localize(n, rank=r, world_size=W, n_routed_experts=N,
                            dspark_experts=args.dspark_experts, mtp=args.mtp)
                   for n in mine]
            if len(set(loc)) != len(loc):
                print(f"  FAIL rank {r}: renumbering collapsed "
                      f"{len(loc) - len(set(loc))} names")
                return 1
            local_by_rank[r] = set(loc)
        expert_names = {n for n in names if ".ffn.experts." in n
                        and not n.startswith("mtp.")}
        union = set().union(*(globals_by_rank[r] & expert_names
                              for r in range(W)))
        overlap = [(a, b) for a in range(W) for b in range(a + 1, W)
                   if globals_by_rank[a] & globals_by_rank[b] & expert_names]
        if union != expert_names or overlap:
            print(f"  FAIL: experts covered {len(union)}/{len(expert_names)}, "
                  f"overlapping rank pairs {overlap}")
            return 1
        widest = max(len(local_by_rank[r]) for r in range(W))
        print(f"  names      {len(names):,} in the index; each rank takes "
              f"{widest:,}, expert names partition with no overlap")
        # the renumbering has to actually DO something, or (2) is vacuous
        moved = sum(1 for n in globals_by_rank[W - 1]
                    if localize(n, rank=W - 1, world_size=W,
                                n_routed_experts=N,
                                dspark_experts=args.dspark_experts,
                                mtp=args.mtp) != n)
        print(f"  renumber   rank {W - 1} rewrites {moved:,} of "
              f"{len(globals_by_rank[W - 1]):,} names")
        if not moved:
            print("  FAIL: no name changed, so the mapping was never exercised")
            return 1
    else:
        print(f"  names      skipped: no index at {index_path}")

    # -- 3. refusals -------------------------------------------------------
    stated = Layout(rank=1, world_size=W, dense="replicate", mtp="replicate",
                    n_routed_experts=N, dspark_experts=args.dspark_experts)
    ok &= check("rank 1's file read by rank 0",
                lambda: require(stated, rank=0, world_size=W,
                                n_routed_experts=N))
    ok &= check("a world-size-8 file read by a TP=4 job",
                lambda: require(Layout(0, 8, "replicate", "replicate", N,
                                       args.dspark_experts),
                                rank=0, world_size=W, n_routed_experts=N))
    ok &= check("an --mtp ep file read as replicate",
                lambda: require(Layout(0, W, "replicate", "ep", N,
                                       args.dspark_experts),
                                rank=0, world_size=W, n_routed_experts=N))
    ok &= check("a config with a different expert count",
                lambda: require(Layout(0, W, "replicate", "replicate", 256,
                                       args.dspark_experts),
                                rank=0, world_size=W, n_routed_experts=N))
    ok &= check("no __metadata__, unstated not allowed",
                lambda: require(Layout(0, W, "replicate", "replicate", 0, 0,
                                       stated=False),
                                rank=0, world_size=W))
    ok &= check("an expert this rank does not own",
                lambda: localize("layers.5.ffn.experts.0.w1.weight", rank=1,
                                 world_size=W, n_routed_experts=N,
                                 dspark_experts=args.dspark_experts))
    ok &= check("a DSpark expert this rank does not own under --mtp ep",
                lambda: localize("mtp.0.ffn.experts.0.w1.weight", rank=1,
                                 world_size=W, n_routed_experts=N,
                                 dspark_experts=args.dspark_experts, mtp="ep"))
    # ... and under replicate the SAME name must pass through untouched: every
    # rank holds all 128, so its local index is its global one
    same = localize("mtp.0.ffn.experts.0.w1.weight", rank=1, world_size=W,
                    n_routed_experts=N, dspark_experts=args.dspark_experts,
                    mtp="replicate")
    print(f"  passes  a replicated DSpark expert through unchanged      "
          f"{'OK' if same == 'mtp.0.ffn.experts.0.w1.weight' else f'FAIL -> {same}'}")
    ok &= same == "mtp.0.ffn.experts.0.w1.weight"
    # and the positive control: the same call must SUCCEED for its real owner
    got = localize(f"layers.5.ffn.experts.{per_rank}.w1.weight", rank=1,
                   world_size=W, n_routed_experts=N,
                   dspark_experts=args.dspark_experts)
    want = "layers.5.ffn.experts.0.w1.weight"
    print(f"  accepts rank 1's own expert {per_rank:<27d} "
          f"{'OK' if got == want else f'FAIL -> {got}'}")
    ok &= got == want
    try:
        require(Layout(0, W, "replicate", "replicate", 0, 0, stated=False),
                rank=0, world_size=W, allow_unstated=True)
        print(f"  accepts an unstated layout when told to             OK")
    except PreshardMismatch as exc:
        print(f"  accepts an unstated layout when told to             FAIL "
              f"({exc})")
        ok = False

    # -- 4. a real rank file, if one exists --------------------------------
    out = Path(args.out)
    built = sorted(out.glob("rank*of*.safetensors")) if out.is_dir() else []
    if not built:
        print(f"  real file  none under {out} yet")
    for path in built:
        m = re.match(r"^rank(\d+)of(\d+)\.safetensors$", path.name)
        r, w = int(m.group(1)), int(m.group(2))
        header, _ = read_header(path)
        header.pop("__metadata__", None)
        layout = read_layout(path)
        print(f"  real file  {path.name}: {layout.describe()}, "
              f"{len(header):,} tensors, {path.stat().st_size / (1<<30):.1f} GiB")
        if index_path.is_file():
            want_names = {n for n in names
                          if wanted_by(n, r, w, N, args.dspark_experts,
                                       mtp=layout.mtp)}
            miss, extra = want_names - set(header), set(header) - want_names
            if miss or extra:
                print(f"    FAIL: {len(miss)} missing, {len(extra)} extra "
                      f"(e.g. {sorted(miss)[:1] or sorted(extra)[:1]})")
                ok = False
            else:
                print(f"    the loader's expected set for rank {r} matches "
                      f"the file exactly")
        try:
            require(layout, rank=r, world_size=w, n_routed_experts=N,
                    mtp=layout.mtp,
                    allow_unstated=args.allow_unstated)
            print(f"    require() accepts it for rank {r}")
        except PreshardMismatch as exc:
            print(f"    require() rejects it: {exc}")
            ok = False

    print("\n" + ("PRESHARD LOAD PASS" if ok else "PRESHARD LOAD FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
