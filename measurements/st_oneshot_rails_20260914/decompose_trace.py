"""Decompose one-shot collective wait from four ranks' CUPTI decode traces, without cross-node clocks.

    python3 decompose_trace.py DIR [STEP ...]     DIR holds rank{0..3}-decode{STEP}.json.gz

For every transport kernel on the main stream, the time it runs past the latest compute kernel before it is
its tail: publication plus the wait for all three peers' flags (plus the reduce for a consumer). A PDL kernel's
own CUPTI span overlaps its producer, so the raw durations overstate the transport by the producer's time.
Collective k finishes on every rank within the wire skew, so each rank's clock offset is the median over
collectives of its end time minus rank 0's. In that common base the latest publication splits each rank's wait
into straggler skew (published earlier, waited for the last rank) and the floor after the last publication.
"""
import gzip
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

TRANSPORT = (("publish", r"k_publish_packets"), ("consumer", r"k_oneshot_consumer"),
             ("moe_packets", r"k_oneshot_moe_packets"), ("max64", r"k_oneshot_max_int64"),
             ("gather64", r"k_oneshot_gather_int64"))


def family(name):
    for fam, pattern in TRANSPORT:
        if re.search(pattern, name):
            return fam
    return "compute"


def sequence(path):
    trace = json.load(gzip.open(path))
    kernels = sorted((e for e in trace["traceEvents"]
                      if e.get("cat") == "kernel" and e.get("args", {}).get("stream") == 7), key=lambda e: e["ts"])
    out, latest = [], 0
    for e in kernels:
        fam, end = family(e["name"]), e["ts"] + e["dur"]
        if fam != "compute":
            out.append((fam, max(latest, e["ts"]), end))       # (family, ready to publish, landed)
        else:
            latest = max(latest, end)
    return out


def main():
    root = Path(sys.argv[1])
    steps = sys.argv[2:] or ["2", "3", "4"]
    agg = defaultdict(lambda: defaultdict(list))
    for step in steps:
        ranks = {r: sequence(root / f"rank{r}-decode{step}.json.gz") for r in range(4)}
        n = min(len(v) for v in ranks.values())
        offset = {r: st.median(ranks[r][i][2] - ranks[0][i][2] for i in range(n)) for r in range(4)}
        print(f"step {step}: {n} transport kernels, clock offsets vs rank 0 (µs)",
              {r: round(v, 1) for r, v in offset.items()})
        for i in range(n):
            fam = ranks[0][i][0]
            if any(ranks[r][i][0] != fam for r in range(4)):
                raise SystemExit(f"step {step}: transport kernel {i} differs across ranks")
            ready = [ranks[r][i][1] - offset[r] for r in range(4)]
            landed = [ranks[r][i][2] - offset[r] for r in range(4)]
            last = max(ready)
            agg[fam]["wait"].append(st.mean(e - p for e, p in zip(landed, ready)))
            agg[fam]["floor"].append(st.median(landed) - last)
            agg[fam]["skew"].append(last - min(ready))
            agg[fam]["last"].append(ready.index(last))
    per_step = len(steps)
    for fam, m in agg.items():
        wait, floor = st.mean(m["wait"]), st.mean(m["floor"])
        print(f"{fam:12s} {len(m['wait']) / per_step:5.1f}/step | mean rank wait {wait:6.1f} µs = skew {wait - floor:5.1f}"
              f" + floor after last publication {floor:5.1f} | median last-first {st.median(m['skew']):5.1f}"
              f" | last publisher {dict(Counter(m['last']))}")


if __name__ == "__main__":
    main()
