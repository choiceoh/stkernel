# SPDX-License-Identifier: Apache-2.0
# usage: python3 tools/trace_step_tail.py <profiler .pt.trace.json.gz>
"""Split each decode step into the target forward and the tail, and break the
tail down by category and by kernel (medians over steps).

The step is cut at the runner's prep kernel (trace_common.cut_steps); the tail
starts after the LAST MoE expert kernel of the step -- the final MoE layer is
the last thing the target forward does, so what follows is head + sampler +
drafter. 25차 read that region off an ad-hoc script and found the drafter's
bf16 GEMMs (3.2~3.6 ms/step) on the critical path; this makes the cut
reproducible. The tail's union busy says how much of it is idle.

Time is CUPTI time: compare rows within one trace, not across traces.
"""
from __future__ import annotations

import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_common import (category, cut_steps, load_kernel_events,  # noqa: E402
                          union_busy_us)


def analyze(path: str) -> dict:
    ev = load_kernel_events(path)
    anchor, starts = cut_steps(ev)
    fwd_ms, tail_ms, tail_busy = [], [], []
    per_cat: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    per_kernel: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    for a, b in zip(starts[5:-1], starts[6:]):        # 앞쪽 스텝은 워밍업이라 뺀다
        seg = ev[a:b]
        last_moe = max((i for i, e in enumerate(seg)
                        if "moecute" in e["name"] or "moe_static" in e["name"]),
                       default=None)
        if last_moe is None:
            continue
        t0 = seg[0]["ts"]
        cut = seg[last_moe]["ts"] + seg[last_moe]["dur"]
        t1 = max(e["ts"] + e["dur"] for e in seg)
        tail = seg[last_moe + 1:]
        fwd_ms.append((cut - t0) / 1000)
        tail_ms.append((t1 - cut) / 1000)
        tail_busy.append(union_busy_us(tail) / 1000)
        cc, cd = collections.Counter(), collections.Counter()
        kc, kd = collections.Counter(), collections.Counter()
        for e in tail:
            cc[category(e["name"])] += 1
            cd[category(e["name"])] += e["dur"]
            kc[e["name"]] += 1
            kd[e["name"]] += e["dur"]
        for k in cc:
            per_cat[k].append((cc[k], cd[k]))
        for k in kc:
            per_kernel[k].append((kc[k], kd[k]))

    def med(rows):
        return {k: {"cnt": statistics.median([c for c, _ in v]),
                    "ms": statistics.median([d for _, d in v]) / 1000}
                for k, v in rows.items()}

    return {"path": path, "anchor": anchor, "steps": len(fwd_ms),
            "fwd_ms": statistics.median(fwd_ms) if fwd_ms else 0.0,
            "tail_ms": statistics.median(tail_ms) if tail_ms else 0.0,
            "tail_busy_ms": statistics.median(tail_busy) if tail_busy else 0.0,
            "cats": med(per_cat), "kernels": med(per_kernel)}


def main(path: str) -> int:
    r = analyze(path)
    if not r["steps"]:
        print("no step contained a MoE expert kernel -- is this a decode trace?")
        return 1
    busy_pct = r["tail_busy_ms"] / r["tail_ms"] * 100 if r["tail_ms"] else 0.0
    print(f"steps {r['steps']} (anchor {r['anchor']}): forward median "
          f"{r['fwd_ms']:.2f} ms, tail median {r['tail_ms']:.2f} ms, "
          f"tail union-busy {r['tail_busy_ms']:.2f} ms ({busy_pct:.0f}% busy)")
    for title, rows, top in (("tail by category (median)", r["cats"], 20),
                             ("tail by kernel (median)", r["kernels"], 14)):
        print(f"\n# {title}")
        for name, v in sorted(rows.items(), key=lambda kv: -kv[1]["ms"])[:top]:
            print(f"  {v['ms']:7.3f} ms  {v['cnt']:5.0f}/step  {name[:92]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
