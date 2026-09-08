#!/usr/bin/env python3
"""Where the prefill step's FIXED cost lives, per kernel category.

40차 measured it in wall time: a prefill step costs `a + b*tokens` with
a = 211 ms, and at a 1,152-token chunk that `a` is 39% of the step. This turns
`a` into kernel categories.

Input is one or more reports from tools/trace_prefill_attribution.py, which
already breaks a capture into per-chunk occupancy. Every chunk in every report
is a data point (`rows` tokens, `occupied_ms` per category), and the partial
last chunk of each request is a free extra point at a different size. Per
category:

    occupied_ms(chunk) = fixed_ms + slope_ms_per_token * rows

The intercept is that category's share of `a`. Categories whose intercept is
near zero scale with the work and are not the target; the ones that do not
shrink when the chunk shrinks are what a fix has to remove.

`idle` (span minus the union of kernel intervals -- launch glue and host stalls,
NOT comm waits, which sit inside the nccl kernels) is carried through the same
regression, because it is a prime suspect for a per-step cost.

  python3 probes/prefill_fixed_cost_attribution.py attr-1152.json attr-8192.json
      [--json OUT] [--min-points 4]
"""
from __future__ import annotations

import argparse
import json
import sys


def points(reports: list[dict], keep_first: bool = False) -> list[dict]:
    """One row per prefill chunk across every report."""
    out = []
    for rep in reports:
        chunks = rep.get("chunks", [])
        for ch in (chunks if keep_first else chunks[1:]):
            rows = ch.get("rows")
            if not rows:
                continue
            cats = {name: c["occupied_ms"] for name, c in ch.get("categories", {}).items()}
            cats["(idle)"] = ch.get("idle_ms", 0.0)
            cats["(span)"] = ch.get("span_ms", 0.0)
            out.append({"rows": rows, "cats": cats, "path": rep.get("path", "")})
    return out


def regress(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """(intercept, slope, se_intercept) by ordinary least squares."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    se = float("nan")
    if n > 2:
        ssr = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        s2 = ssr / (n - 2)
        se = (s2 * (1.0 / n + mx * mx / sxx)) ** 0.5
    return intercept, slope, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--json", default="")
    ap.add_argument("--min-points", type=int, default=4)
    # The partial last chunk of each request is a free point at a different
    # size, and a small one carries a lot of leverage on the intercept -- but a
    # 30-token chunk need not run the same kernels as a 1,152-token one. Raise
    # this to see whether the answer depends on those tails.
    ap.add_argument("--min-rows", type=int, default=0)
    # The FIRST chunk of a request is a warm-up: on the 2026-09-08 capture its
    # communication read 352 ms against 240-246 for the four after it (42%
    # spread), which alone invented a ~92 ms "fixed" communication cost. Every
    # other category is within 1-9% across the later chunks. Dropping it is the
    # default; --keep-first puts it back.
    ap.add_argument("--keep-first", action="store_true",
                    help="keep each request's first chunk (a warm-up outlier)")
    args = ap.parse_args()

    reports = [json.load(open(p)) for p in args.reports]
    pts = [p for p in points(reports, args.keep_first) if p["rows"] >= args.min_rows]
    sizes = sorted({p["rows"] for p in pts})
    print(f"{len(pts)} prefill chunks from {len(reports)} capture(s); "
          f"chunk sizes {sizes[:8]}{'...' if len(sizes) > 8 else ''}")
    if len(pts) < args.min_points or len(sizes) < 2:
        print("!! need at least two distinct chunk sizes (and a few chunks): a single size "
              "cannot separate the fixed part from the per-token part", file=sys.stderr)
        return 2

    names = sorted({name for p in pts for name in p["cats"]})
    rows = []
    for name in names:
        xs = [p["rows"] for p in pts if name in p["cats"]]
        ys = [p["cats"][name] for p in pts if name in p["cats"]]
        if len(xs) < 3:
            continue
        fixed, slope, se = regress([float(x) for x in xs], ys)
        groups: dict[int, list[float]] = {}
        for x, y in zip(xs, ys):
            groups.setdefault(int(x), []).append(y)
        spreads = [100 * (max(v) - min(v)) / (sum(v) / len(v))
                   for v in groups.values() if len(v) > 1 and sum(v)]
        rows.append({"category": name, "fixed_ms": fixed, "us_per_token": slope * 1000,
                     "se_fixed_ms": se, "n": len(xs),
                     "spread_pct": max(spreads) if spreads else float("nan"),
                     "at_1152_ms": fixed + slope * 1152, "at_8192_ms": fixed + slope * 8192})

    span = next((r for r in rows if r["category"] == "(span)"), None)
    body = [r for r in rows if not r["category"].startswith("(")]
    body.sort(key=lambda r: -r["fixed_ms"])
    idle = next((r for r in rows if r["category"] == "(idle)"), None)

    print(f"\n{'category':<26} {'fixed ms':>9} {'+-95%':>7} {'us/token':>9} "
          f"{'@1152 ms':>9} {'spread':>7}")
    print("  (spread = worst disagreement between chunks of the SAME size; a wide one "
          "means that\n   category's intercept is scatter, not a fixed cost)")
    for r in body + ([idle] if idle else []):
        pm = "" if r["se_fixed_ms"] != r["se_fixed_ms"] else f"{1.96 * r['se_fixed_ms']:7.1f}"
        sp = "" if r["spread_pct"] != r["spread_pct"] else f"{r['spread_pct']:6.1f}%"
        print(f"{r['category']:<26} {r['fixed_ms']:>9.1f} {pm:>7} {r['us_per_token']:>9.2f} "
              f"{r['at_1152_ms']:>9.1f} {sp:>7}")
    total_fixed = sum(r["fixed_ms"] for r in body if r["fixed_ms"] > 0)
    print(f"\n  sum of positive category intercepts: {total_fixed:.1f} ms")
    if idle:
        print(f"  of which idle (launch glue + host stalls): {idle['fixed_ms']:.1f} ms")
    if span:
        print(f"  the step's own intercept (span): {span['fixed_ms']:.1f} ms "
              f"+- {1.96 * span['se_fixed_ms']:.1f}" if span["se_fixed_ms"] == span["se_fixed_ms"]
              else f"  the step's own intercept (span): {span['fixed_ms']:.1f} ms")
        print("  (compare with the wall-clock a from probes/prefill_chunk_sweep.py; the trace "
              "span is GPU occupancy of an INSTRUMENTED run, not request wall time)")

    if args.json:
        json.dump({"rows": rows, "points": len(pts), "sizes": sizes,
                   "reports": args.reports}, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
