"""Replay an actual prefill MoE input without loading the full TP model.

The sample contains x, sel and w tensors captured immediately before a MoE
layer. Rank weights must be the original row-major ST safetensors. This probe
checks repeatability, workspace reuse and zero-weight output on the served lane.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from engine.base.loader import RankLoader
from engine.profiles.glm53.lanes import served


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--rank-file", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two repeats are required")
    torch.set_num_threads(4)
    sample = torch.load(args.sample, map_location="cuda", weights_only=True)
    x, sel, weights = (sample[k] for k in ("x", "sel", "w"))
    names = [f"L{args.layer}.moe.{k}" for k in ("w13", "w13_sf", "w2", "w2_sf")]
    params = RankLoader(args.rank_file).load(names)
    packed = [params[k] for k in names]
    lanes = served(moe_static="t,r,sf6,q0", consume_scales=True)
    lanes.moe_prepare(*packed, sel.shape[1], 10.)
    from engine.kernels.b12x import moe_dispatch as md

    results = []
    reference = None
    for rep in range(args.repeats + 1):
        # Poison the retained accumulator to catch partial zeroing and stale
        # values from another request. The final call keeps every route but
        # sets its weight to zero, so every output element must be zero.
        for owner in md.cached_workspace_owners():
            plane = getattr(owner, "ep_scatter_fp32", None)
            if plane is not None:
                plane.fill_(float("nan"))
        zero_weights = rep == args.repeats
        route_weights = torch.zeros_like(weights) if zero_weights else weights
        begin = time.perf_counter()
        out = lanes.moe(x, sel, route_weights, *packed, 10.).clone()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        if reference is None:
            reference = out
        expected = torch.zeros_like(reference) if zero_weights else reference
        diff = out.float() - expected.float()
        result = dict(repeat=rep, zero_weights=zero_weights, shape=list(out.shape),
                      finite=bool(torch.isfinite(out).all()),
                      different=int((out != expected).sum()),
                      max_abs=float(diff.abs().max()), seconds=elapsed)
        print(json.dumps(result), flush=True)
        results.append(result)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["finite"] and r["different"] == 0 for r in results):
        raise SystemExit("prefill MoE repeat/workspace regression failed")


if __name__ == "__main__":
    main()
