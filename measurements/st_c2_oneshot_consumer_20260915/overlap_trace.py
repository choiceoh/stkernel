"""Per-collective producer/successor timing from four ranks' CUPTI decode traces (no cross-node clocks).

    python3 overlap_trace.py DIR STEP [STEP ...]    DIR holds rank-{0..3}/decode-{STEP}.trace.json.gz

For every transport kernel on the main stream (7):
  lead     = producer end - collective start   (>0: the collective launched while its producer still ran)
  tail     = collective end - max(producer end, collective start)  (what the collective adds past its producer)
  next_gap = next compute kernel start - collective end  (<0: its successor launched before it completed)
  floor    = median rank landing - latest rank ready, after aligning rank clocks by collective completion
"""
import gzip
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

TRANSPORT = (("publish", r"^k_publish_packets"), ("consumer", r"^k_oneshot_consumer"),
             ("moe_packets", r"^k_oneshot_moe_packets"), ("packets", r"^k_oneshot_packets"),
             ("ordinary", r"^k_oneshot\("), ("max64", r"^k_oneshot_max_int64"),
             ("gather64", r"^k_oneshot_gather_int64"), ("reserve", r"^k_reserve_packets"))


def family(name):
    for fam, pattern in TRANSPORT:
        if re.search(pattern, name):
            return fam
    return "compute"


def short(name):
    name = re.sub(r"\(anonymous namespace\)::|^void ", "", name)
    return re.split(r"[(<]", name, maxsplit=1)[0][:40]


def sequence(path):
    trace = json.load(gzip.open(path))
    kernels = sorted((e for e in trace["traceEvents"]
                      if e.get("cat") == "kernel" and e.get("args", {}).get("stream") == 7), key=lambda e: e["ts"])
    out = []
    latest, latest_name = 0, ""
    for i, e in enumerate(kernels):
        fam, start, end = family(e["name"]), e["ts"], e["ts"] + e["dur"]
        if fam == "reserve":
            continue
        if fam != "compute":
            succ = next((k for k in kernels[i + 1:] if family(k["name"]) == "compute"), None)
            out.append(dict(fam=fam, start=start, end=end, ready=max(latest, start), producer_end=latest,
                            producer=latest_name, next_start=succ["ts"] if succ else None,
                            successor=short(succ["name"]) if succ else ""))
        else:
            if end > latest:
                latest, latest_name = end, short(e["name"])
    return out


def main():
    root = Path(sys.argv[1])
    steps = sys.argv[2:]
    agg = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(lambda: defaultdict(int))
    for step in steps:
        ranks = {r: sequence(root / f"rank-{r}" / f"decode-{step}.trace.json.gz") for r in range(4)}
        n = min(len(v) for v in ranks.values())
        offset = {r: st.median(ranks[r][i]["end"] - ranks[0][i]["end"] for i in range(n)) for r in range(4)}
        for i in range(n):
            fam = ranks[0][i]["fam"]
            if any(ranks[r][i]["fam"] != fam for r in range(4)):
                raise SystemExit(f"step {step}: transport kernel {i} differs across ranks")
            ready = [ranks[r][i]["ready"] - offset[r] for r in range(4)]
            landed = [ranks[r][i]["end"] - offset[r] for r in range(4)]
            m = agg[fam]
            m["floor"].append(st.median(landed) - max(ready))
            m["skew"].append(max(ready) - min(ready))
            for r in range(4):
                c = ranks[r][i]
                m["span"].append(c["end"] - c["start"])
                m["lead"].append(c["producer_end"] - c["start"])
                m["tail"].append(c["end"] - c["ready"])
                if c["next_start"] is not None:
                    m["next_gap"].append(c["next_start"] - c["end"])
                pairs[fam][(c["producer"], c["successor"])] += 1
    per = 4 * len(steps)
    for fam, m in agg.items():
        def q(name):
            v = sorted(m[name])
            return f"{name} mean {st.mean(v):6.1f} med {st.median(v):6.1f}"
        print(f"{fam:12s} {len(m['span']) / per:5.1f}/step/rank | {q('span')} | {q('lead')} | {q('tail')} | "
              f"{q('next_gap')} | {q('floor')} | {q('skew')}")
        for (p, s), count in sorted(pairs[fam].items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {count / per:5.1f}/step  after {p:48s} before {s}")


if __name__ == "__main__":
    main()
