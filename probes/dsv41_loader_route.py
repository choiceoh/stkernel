#!/usr/bin/env python3
"""Is every checkpoint tensor routed, once, and does the loader agree with the builder?

Three checks, and the third is the one nothing else makes:

  1. COVERAGE. All 96,085 names from the index go through `route`, which raises
     on anything unclaimed. A loader that skipped a tensor would leave a module
     at its initialization values and say nothing.
  2. The engram tables route to the SSD path and NOT to a parameter. A loader
     that materializes them needs 188.8 GiB it does not have.
  3. AGREEMENT. For every rank, the names `wanted_by` accepts must equal the
     names tools/dsv41_preshard.py writes into that rank's file. The two never
     run together -- one writes on srv4 today, the other reads in a container
     later -- so a test is the only thing holding them to the same partition.

    python3 probes/dsv41_loader_route.py [--index ...] [--config ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_model"))
sys.path.insert(0, str(HERE / "tools"))

from dsv41_loader import ENGRAM, PARAM, UnroutedTensor, route, wanted_by  # noqa: E402

HF = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/"
REPO = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash")


def load_json(path, name):
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    if (REPO / name).is_file():
        return json.loads((REPO / name).read_text())
    return json.loads(urllib.request.urlopen(HF + name, timeout=180).read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--index")
    ap.add_argument("--world-size", type=int, default=4)
    args = ap.parse_args()

    cfg = load_json(args.config, "config.json")["text_config"]
    names = sorted(load_json(args.index, "model.safetensors.index.json")["weight_map"])
    n_routed = cfg["n_routed_experts"]
    n_dspark = cfg["dspark_n_routed_experts"]

    # -- 1. coverage --------------------------------------------------------
    routes, unrouted = {}, []
    for name in names:
        try:
            routes[name] = route(name)
        except UnroutedTensor:
            unrouted.append(name)
    if unrouted:
        pats = collections.Counter(re.sub(r"\.\d+\.", ".N.", n) for n in unrouted)
        print(f"  FAIL: {len(unrouted)} unrouted, {len(pats)} patterns:")
        for pat, n in pats.most_common(8):
            print(f"      {n:6d}x {pat}")
        return 1
    kinds = collections.Counter(r.kind for r in routes.values())
    print(f"  routed {len(routes):,}/{len(names):,}  "
          f"({', '.join(f'{k} {v:,}' for k, v in kinds.most_common())})")

    # a destination collision would silently overwrite one tensor with another
    dest = collections.Counter((r.kind, r.module, r.param)
                               for r in routes.values())
    dupes = [d for d, n in dest.items() if n > 1]
    if dupes:
        print(f"  FAIL: {len(dupes)} destinations claimed twice, e.g. {dupes[:3]}")
        return 1
    print(f"  destinations distinct: {len(dest):,}")

    # -- 2. the engram tables do not become parameters ----------------------
    eng = [n for n, r in routes.items() if r.kind == ENGRAM]
    want_eng = [n for n in names if ".engram.embed." in n]
    if sorted(eng) != sorted(want_eng):
        print(f"  FAIL: engram routing is {len(eng)} names, expected "
              f"{len(want_eng)}")
        return 1
    proj = [n for n in names if ".engram." in n and n not in want_eng]
    if any(routes[n].kind != PARAM for n in proj):
        print("  FAIL: an engram projection was routed away from a parameter")
        return 1
    print(f"  engram: {len(eng)} tables to the SSD path, "
          f"{len(proj)} projections stay parameters")

    # -- 3. loader and builder agree, per rank ------------------------------
    import dsv41_preshard as builder

    ok = True
    # EVERY name, both modes. This used to drop `mtp.` from the comparison
    # because "the builder's expert filter only knows the main stack's expert
    # count" -- which was true, and was the bug: the builder replicated the
    # DSpark block's 128 experts while the loader sharded them, 1,728 tensors
    # per rank, and the exclusion is what let it sit there.
    for mtp_mode in ("replicate", "ep"):
        for rank in range(args.world_size):
            cfg = {"rank": rank, "world": args.world_size,
                   "experts": n_routed, "mtp_experts": n_dspark,
                   "dense": "replicate", "mtp": mtp_mode}
            mine = {n for n in names
                    if wanted_by(n, rank, args.world_size, n_routed, n_dspark,
                                 mtp=mtp_mode)}
            theirs = {n for n in names if builder._wanted(n, cfg)}
            if mine != theirs:
                ok = False
                only_l = sorted(mine - theirs)[:2]
                only_b = sorted(theirs - mine)[:2]
                print(f"  FAIL mtp={mtp_mode} rank {rank}: loader-only "
                      f"{len(mine - theirs)} {only_l}, builder-only "
                      f"{len(theirs - mine)} {only_b}")
            else:
                n_mtp = sum(1 for n in mine if n.startswith("mtp."))
                print(f"  mtp={mtp_mode:9s} rank {rank}: {len(mine):,} tensors "
                      f"({n_mtp:,} DSpark), loader and builder agree")
        # the two modes must actually DIFFER, or agreeing about them is
        # agreeing about nothing
        a = {n for n in names
             if wanted_by(n, 0, args.world_size, n_routed, n_dspark,
                          mtp="replicate")}
        b = {n for n in names
             if wanted_by(n, 0, args.world_size, n_routed, n_dspark,
                          mtp="ep")}
        if a == b:
            print("  FAIL: --mtp replicate and --mtp ep select the same set")
            ok = False
    if not ok:
        return 1

    total = sum(1 for n in names if not n.startswith("mtp."))
    per = [sum(1 for n in names
               if not n.startswith("mtp.")
               and wanted_by(n, r, args.world_size, n_routed, n_dspark))
           for r in range(args.world_size)]
    print(f"  main stack {total:,} tensors -> {per} per rank "
          f"(sum {sum(per):,}; replication is why it exceeds the total)")
    print("\nLOADER ROUTE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
