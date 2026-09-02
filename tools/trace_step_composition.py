# SPDX-License-Identifier: Apache-2.0
# usage: python3 tools/trace_step_composition.py <profiler .pt.trace.json.gz>
"""Per-step time composition of a decode trace by kernel category (medians
over steps), plus the one-shot all-reduce duration distribution. Idle is the
step span minus the interval union of all kernels across streams; the sum of
the per-category rows can exceed span-minus-idle by the cross-stream overlap,
which is printed separately."""
from __future__ import annotations

import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_common import category, cut_steps, load_kernel_events, percentile, union_busy_us  # noqa: E402


def main(path: str) -> int:
    ev = load_kernel_events(path)
    anchor, starts = cut_steps(ev)
    print(f"kernels {len(ev)}, steps {len(starts)} (anchor {anchor})")
    if len(starts) < 8:
        print("too few steps for medians (need >= 8)")
        return 1
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
    print(f"steps analysed {len(steplen)}: step len median {statistics.median(steplen)/1000:.2f} ms, "
          f"idle (span - union busy) median {statistics.median(idle)/1000:.2f} ms, "
          f"cross-stream overlap median {statistics.median(overlap)/1000:.2f} ms")
    for k, v in sorted(tot.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f"{statistics.median(v)/1000:7.2f} ms  {statistics.median(cnt[k]):6.0f}/step  {k}")
    if ar_durs:
        ar_durs.sort()
        print("k_oneshot dur percentiles (us): p10 %.1f p50 %.1f p90 %.1f max %.1f, n=%d" % (
            percentile(ar_durs, 0.10), percentile(ar_durs, 0.50), percentile(ar_durs, 0.90), ar_durs[-1], len(ar_durs)))
    else:
        print("no k_oneshot kernels in this trace (NCCL all-reduce or TP=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
