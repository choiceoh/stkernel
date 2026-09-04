# SPDX-License-Identifier: Apache-2.0
# usage: python3 tools/trace_launch_positions.py <trace.gz> <kernel-name-substring> [--csv out.csv]
"""Per-launch-POSITION view of one kernel family in a decode trace.

Steps are cut with trace_common; inside every step the launches whose name
contains the substring are numbered in order (position 0, 1, ...), and the
table prints, per position, the median duration over steps (p10/p90), the
gap from the previous kernel on the same stream, how much of the launch
overlapped the routed MoE kernel and other kernels on OTHER streams, the
stream, and the most common predecessor kernel. Position is what maps a
launch onto a linear when the trace carries no shapes: the model's layer
sequence is the same in every step.

This is the view that found 30차 (MEASUREMENTS.md): the shared expert's
down GEMM at 18 us alone and 135 us in the step, sitting under the MoE
kernel with a 650-880 us gap on its stream. A per-name sum hides it.

Also printed: the launch attributes (smem, registers, grid) of the first
occurrence of a few kernel families, and the aux-stream exposure -- for
every non-main-stream kernel, the part of its duration that no main-stream
kernel covers (main = the stream carrying the most kernels).
"""
from __future__ import annotations

import bisect
import collections
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_common import cut_steps, load_kernel_events, stream_of  # noqa: E402

ATTR_FAMILIES = ("moecute", "mk_gemm", "mk_mhc", "deep_gemm::sm120_fp8_fp4",
                 "cutlass_80", "BatchMLA", "_deneb_gate", "k_oneshot")


def _exposed(ivs: list[tuple[float, float]], s: float, t: float) -> float:
    """(t - s) minus the union of ivs clipped to [s, t)."""
    cov = 0.0
    cur_s = cur_e = None
    for ms, me in ivs:
        if me <= s or ms >= t:
            continue
        ms2, me2 = max(ms, s), min(me, t)
        if cur_e is None or ms2 > cur_e:
            if cur_e is not None:
                cov += cur_e - cur_s
            cur_s, cur_e = ms2, me2
        elif me2 > cur_e:
            cur_e = me2
    if cur_e is not None:
        cov += cur_e - cur_s
    return (t - s) - cov


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    path, pat = sys.argv[1], sys.argv[2]
    out_csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    ev = load_kernel_events(path)
    anchor, starts = cut_steps(ev)
    print(f"kernels={len(ev)} anchor={anchor} steps={len(starts) - 1}")
    seen = set()
    for e in ev:
        fam = e["name"][:28]
        if fam in seen or not any(k in e["name"] for k in ATTR_FAMILIES):
            continue
        seen.add(fam)
        a = e.get("args", {})
        keep = {k: a[k] for k in ("stream", "registers per thread", "shared memory", "grid", "block")
                if k in a}
        print(f"ATTR {e['name'][:56]:<56} {json.dumps(keep)}")
    # the main stream is the one carrying the model's forward: the stream
    # with the most kernel TIME (the routed MoE kernels alone are ~half the
    # step). A kernel-count heuristic picks the glue stream, and the step
    # anchor's stream is not it either -- the 09-01 stock trace has the prep
    # kernels and the shared-expert pair on 17 and the forward on 210, the
    # armed 09-04 trace the other way round.
    busy = collections.Counter()
    for e in ev:
        busy[stream_of(e)] += e["dur"]
    main_stream = busy.most_common(1)[0][0]
    print("streams (kernels, busy ms over the trace): "
          + ", ".join(f"{k}: {sum(1 for e in ev if stream_of(e) == k)}, {v / 1e3:.0f}"
                      for k, v in busy.most_common()))
    rows = []
    # per step, per kernel family: exposed us; a family absent from a step
    # contributes 0 to that step (a median over occurrences only would let a
    # single prefill step dominate, as it did for the MoE kernels)
    exposed_by = collections.defaultdict(dict)
    nsteps = 0
    # step 0 is the profiler's warm-up (a prefill and the first decode): skip
    for si, (a, b) in enumerate(zip(starts[1:-1], starts[2:]), start=1):
        nsteps += 1
        seg = ev[a:b]
        seg_ts = [e["ts"] for e in seg]
        lookback = max(e["dur"] for e in seg)  # any kernel still running at s0
        by_stream = collections.defaultdict(list)
        for e in seg:
            by_stream[stream_of(e)].append(e)
        prev = {}
        for lst in by_stream.values():
            lst.sort(key=lambda e: e["ts"])
            for i, e in enumerate(lst):
                prev[id(e)] = lst[i - 1] if i else None
        main_ivs = [(e["ts"], e["ts"] + e["dur"]) for e in by_stream.get(main_stream, [])]
        expo = collections.Counter()
        for e in seg:
            if stream_of(e) != main_stream:
                expo[e["name"][:30]] += _exposed(main_ivs, e["ts"], e["ts"] + e["dur"])
        for k, v in expo.items():
            exposed_by[k][si] = v
        gemms = [e for e in seg if pat in e["name"]]
        for gi, e in enumerate(gemms):
            p = prev[id(e)]
            gap = e["ts"] - (p["ts"] + p["dur"]) if p else None  # first on its stream
            s0, e0 = e["ts"], e["ts"] + e["dur"]
            lo = bisect.bisect_left(seg_ts, s0 - lookback)
            hi = bisect.bisect_right(seg_ts, e0)
            ov = ov_moe = 0.0
            for o in seg[lo:hi]:
                if o is e or stream_of(o) == stream_of(e):
                    continue
                x = min(e0, o["ts"] + o["dur"]) - max(s0, o["ts"])
                if x > 0:
                    ov += x
                    if "moecute" in o["name"] or "moe_static" in o["name"]:
                        ov_moe += x
            rows.append({"step": si, "idx": gi, "dur": e["dur"], "stream": stream_of(e),
                         "gap": round(gap, 1) if gap is not None else None,
                         "ov_moe": round(ov_moe, 1),
                         "ov_other": round(ov - ov_moe, 1),
                         "prev": p["name"][:40] if p else ""})
    if not rows:
        print(f"no launches match {pat!r}")
        return 1
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    steps = collections.defaultdict(list)
    for r in rows:
        steps[r["step"]].append(r)
    modal = collections.Counter(len(v) for v in steps.values()).most_common(1)[0][0]
    print(f"{pat}: modal launches/step = {modal} ({sum(1 for v in steps.values() if len(v) == modal)} steps)")
    pos = collections.defaultdict(list)
    for rs in steps.values():
        if len(rs) == modal:
            for r in rs:
                pos[r["idx"]].append(r)
    print(f"{'idx':>3} {'med_dur':>8} {'p10':>7} {'p90':>7} {'med_gap':>8} {'ov_moe':>7} {'ov_oth':>7} {'str':>4}  prev")
    tot = 0.0
    for i in sorted(pos):
        rs = pos[i]
        d = sorted(r["dur"] for r in rs)
        md = statistics.median(d)
        tot += md
        gaps = [r["gap"] for r in rs if r["gap"] is not None]
        mg = statistics.median(gaps) if gaps else float("nan")
        print(f"{i:3d} {md:8.1f} {d[len(d) // 10]:7.1f} {d[len(d) * 9 // 10]:7.1f} "
              f"{mg:8.1f} "
              f"{statistics.median(r['ov_moe'] for r in rs):7.1f} "
              f"{statistics.median(r['ov_other'] for r in rs):7.1f} "
              f"{collections.Counter(r['stream'] for r in rs).most_common(1)[0][0]:>4}  "
              f"{collections.Counter(r['prev'] for r in rs).most_common(1)[0][0]}")
    print(f"sum of per-position medians = {tot / 1e3:.2f} ms/step")
    print(f"\nnon-main-stream kernels (main = stream {main_stream}): exposed us/step, median over steps")
    def _med(d):
        return statistics.median(list(d.values()) + [0.0] * (nsteps - len(d)))
    for k, v in sorted(exposed_by.items(), key=lambda kv: -_med(kv[1]))[:12]:
        print(f"  {_med(v):8.1f}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
