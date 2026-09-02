# SPDX-License-Identifier: Apache-2.0
# usage: python3 tools/trace_step_composition.py <trace.gz> [<cand trace.gz>]
#        python3 tools/trace_step_composition.py base.gz --diff cand.gz
"""Per-step time composition of a decode trace by kernel category (medians
over steps), plus the one-shot all-reduce duration distribution. Idle is the
step span minus the interval union of all kernels across streams; the sum of
the per-category rows can exceed span-minus-idle by the cross-stream overlap,
which is printed separately.

With a second trace (base cand order: the first argument is the base) the
same analysis runs on both and prints side by side. The count delta is the
authoritative channel -- counts are immune to profiler distortion -- so a
time change with an unchanged count and a kernel-count change read
differently."""
from __future__ import annotations

import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_common import category, cut_steps, load_kernel_events, percentile, union_busy_us  # noqa: E402


def analyze(path: str) -> dict:
    ev = load_kernel_events(path)
    anchor, starts = cut_steps(ev)
    if len(starts) < 8:
        raise SystemExit(f"{path}: too few steps for medians (need >= 8, got {len(starts)})")
    tot = collections.defaultdict(list)
    cnt = collections.defaultdict(list)
    steplen, idle, overlap, ar_durs = [], [], [], []
    for a, b in zip(starts[5:-1], starts[6:]):
        seg = ev[a:b]
        t0 = seg[0]["ts"]
        t1 = max(e["ts"] + e["dur"] for e in seg)
        union = union_busy_us(seg)
        summed = sum(e["dur"] for e in seg)
        steplen.append(t1 - t0)
        idle.append(t1 - t0 - union)
        overlap.append(summed - union)
        s = collections.Counter()
        c = collections.Counter()
        for e in seg:
            k = category(e["name"])
            s[k] += e["dur"]
            c[k] += 1
            if k == "AR k_oneshot":
                ar_durs.append(e["dur"])
        for k in s:
            tot[k].append(s[k])
            cnt[k].append(c[k])
    cats = {k: {"ms": statistics.median(tot[k]) / 1000,
                "cnt": statistics.median(cnt[k])} for k in tot}
    out = {"path": path, "anchor": anchor, "kernels": len(ev),
           "steps": len(steplen),
           "step_ms": statistics.median(steplen) / 1000,
           "idle_ms": statistics.median(idle) / 1000,
           "overlap_ms": statistics.median(overlap) / 1000,
           "cats": cats}
    if ar_durs:
        ar_durs.sort()
        out["ar"] = (percentile(ar_durs, 0.10), percentile(ar_durs, 0.50),
                     percentile(ar_durs, 0.90), ar_durs[-1], len(ar_durs))
    return out


def report(r: dict) -> None:
    print(f"kernels {r['kernels']}, steps {r['steps']} (anchor {r['anchor']})")
    print(f"steps analysed {r['steps']}: step len median {r['step_ms']:.2f} ms, "
          f"idle (span - union busy) median {r['idle_ms']:.2f} ms, "
          f"cross-stream overlap median {r['overlap_ms']:.2f} ms")
    for k, v in sorted(r["cats"].items(), key=lambda kv: -kv[1]["ms"]):
        print(f"{v['ms']:7.2f} ms  {v['cnt']:6.0f}/step  {k}")
    if "ar" in r:
        print("k_oneshot dur percentiles (us): p10 %.1f p50 %.1f p90 %.1f max %.1f, n=%d"
              % r["ar"])
    else:
        print("no k_oneshot kernels in this trace (NCCL all-reduce or TP=1)")


def diff(base: dict, cand: dict) -> None:
    for label, r in (("base", base), ("cand", cand)):
        print(f"{label}: {r['path']}  steps {r['steps']}, step len median "
              f"{r['step_ms']:.2f} ms, idle median {r['idle_ms']:.2f} ms")
    keys = sorted(set(base["cats"]) | set(cand["cats"]),
                  key=lambda k: -base["cats"].get(k, {"ms": 0})["ms"])
    print(f"\n{'category':<32}{'base ms':>9}{'cand ms':>9}{'d_ms':>8}"
          f"{'base/step':>11}{'cand/step':>11}{'d_cnt':>7}")
    for k in keys:
        b = base["cats"].get(k, {"ms": 0.0, "cnt": 0.0})
        c = cand["cats"].get(k, {"ms": 0.0, "cnt": 0.0})
        d_cnt = c["cnt"] - b["cnt"]
        mark = "  <- kernel-count change" if d_cnt else ""
        print(f"{k:<32}{b['ms']:>9.2f}{c['ms']:>9.2f}{c['ms'] - b['ms']:>+8.2f}"
              f"{b['cnt']:>11.0f}{c['cnt']:>11.0f}{d_cnt:>+7.0f}{mark}")
    d_step = cand["step_ms"] - base["step_ms"]
    d_idle = cand["idle_ms"] - base["idle_ms"]
    print(f"\nstep len {base['step_ms']:.2f} -> {cand['step_ms']:.2f} ms "
          f"({d_step:+.2f}); idle {base['idle_ms']:.2f} -> {cand['idle_ms']:.2f} "
          f"ms ({d_idle:+.2f}). ms deltas are CUPTI-distorted; the count "
          f"column is the ground truth.")


def main(argv: list) -> int:
    paths = []
    diff_path = None
    i = 0
    while i < len(argv):
        if argv[i] == "--diff":
            diff_path = argv[i + 1]
            i += 2
        else:
            paths.append(argv[i])
            i += 1
    if not paths:
        print(__doc__)
        return 2
    base = analyze(paths[0])
    if diff_path:
        diff(base, analyze(diff_path))
    else:
        report(base)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
