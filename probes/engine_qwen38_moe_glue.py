"""Qwen3.8's prefill MoE glue on one GB10, step by step (probe, single-GPU lane `qwen38_moe_glue`).

A prefill chunk's MoE layer is the router (router_fp32.router_logits_mma, #1300), the route (lanes.route: torch's softmax
and top-k -- a captured step takes moe_route.softmax_topk instead), the shared gate (a [4,096 x 2,560] by [2,560 x 1]
product and its sigmoid), this rank's pairs (local_routes, a mask and torch.nonzero, which waits for the device), their
rows gathered (x.index_select and two index gathers), the dynamic expert launch, and the pairs summed back per token
(moe_output.pair_sum). The census keeps its top 25 kernels, so it names only the gathers and the pair sum; this times each
step of the glue on synthetic routes at the served widths -- 512 experts, ten a token, a rank's 128 -- interleaved over
many rounds, to say which is worth a launch of its own. The expert launch itself is not timed (the b12x census's).

    python3 probes/engine_kernel_check.py --lanes qwen38_moe_glue --output /cache/qwen38-moe-glue.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HIDDEN, EXPERTS, TOPK, LOCAL, FIRST = 2560, 512, 10, 128, 0
ROWS = (512, 4096)
ROUNDS = 21


def run(output=None) -> dict:
    import torch
    from engine.kernels import moe_output, moe_route
    from engine.kernels.router_fp32 import router_logits_mma
    from engine.profiles.qwen38.lanes import local_routes, route_softmax_topk
    torch.manual_seed(0)
    gates = torch.randn(EXPERTS + 1, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "rows": {}}
    for m in ROWS:
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        scores = router_logits_mma(x, gates[:EXPERTS])
        ids, weights = route_softmax_topk(scores, TOPK)
        local_ids, w = local_routes(ids, weights, FIRST, LOCAL)
        token, route = ((ids >= FIRST) & (ids < FIRST + LOCAL)).nonzero(as_tuple=True)
        pairs = torch.randn(token.numel(), HIDDEN, device="cuda", dtype=torch.bfloat16)
        steps = {
            "router mma [512]": lambda: router_logits_mma(x, gates[:EXPERTS]),
            "router mma [513] (shared gate folded)": lambda: router_logits_mma(x, gates),
            "route: torch softmax + topk (reference)": lambda: route_softmax_topk(scores, TOPK),
            "route: moe_route.softmax_topk (served)": lambda: moe_route.softmax_topk(scores, TOPK, experts=EXPERTS),
            "shared gate: mm [1] + sigmoid": lambda: torch.sigmoid(torch.mm(x, gates[EXPERTS:].t()).float()),
            "remap + mask + nonzero (before)": lambda: (local_routes(ids, weights, FIRST, LOCAL),
                                                        ((ids - FIRST >= 0) & (ids - FIRST < LOCAL))
                                                        .nonzero(as_tuple=True)),
            "compact_routes + nonzero": lambda: moe_route.compact_routes(ids, weights, FIRST, LOCAL)[2]
                                                 .nonzero(as_tuple=True),
            "gather: x rows + ids + weights (before)": lambda: (x.index_select(0, token),
                                                                 local_ids[token, route][:, None],
                                                                 w[token, route][:, None]),
            "pair_rows": lambda: moe_route.pair_rows(x, local_ids, w, token, route),
            "pair_sum": lambda: moe_output.pair_sum(pairs, token, m),
        }
        times = {name: [] for name in steps}
        for fn in steps.values():
            fn()
        torch.cuda.synchronize()
        for r in range(ROUNDS):
            for name, fn in (steps.items() if r % 2 == 0 else list(steps.items())[::-1]):
                torch.cuda.synchronize()
                began = time.perf_counter()
                for _ in range(4):
                    fn()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / 4 * 1e6)
        row = {name: {"median": round(statistics.median(v), 1), "min": round(min(v), 1)} for name, v in times.items()}
        a, b = route_softmax_topk(scores, TOPK), moe_route.softmax_topk(scores, TOPK, experts=EXPERTS)
        c_ids, c_w, c_mine = moe_route.compact_routes(ids, weights, FIRST, LOCAL)
        xp, ip, wp = moe_route.pair_rows(x, local_ids, w, token, route)
        checks = {"pairs": int(token.numel()), "route_ids_equal": bool(torch.equal(a[0].to(torch.int32), b[0])),
                  "route_weights_max_diff": float((a[1].float() - b[1]).abs().max()),
                  "compact_routes_bytes": bool(torch.equal(c_ids, local_ids) and torch.equal(c_w, w) and torch.equal(
                      c_mine, (ids - FIRST >= 0) & (ids - FIRST < LOCAL))),
                  "pair_rows_bytes": bool(torch.equal(xp, x.index_select(0, token)) and torch.equal(
                      ip, local_ids[token, route][:, None]) and torch.equal(wp, w[token, route][:, None]))}
        report["rows"][m] = {"us_a_call": row, "checks": checks}
        print(json.dumps({f"moe glue rows {m}": {k: v["min"] for k, v in row.items()}, "checks": checks}), flush=True)
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
