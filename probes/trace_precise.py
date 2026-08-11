#!/usr/bin/env python3
"""Precise trace analysis beyond bucket sums: (1) inter-kernel idle-gap
distribution on the compute stream (launch/sync stalls hidden inside the
0.9% 'idle'); (2) per-kernel per-call min/median/p99 (outliers); (3) one
steady-state step's kernel sequence with cumulative gaps."""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    doc = json.load(f)
evs = [e for e in doc.get("traceEvents", [])
       if e.get("ph") == "X" and "kernel" in e.get("cat", "")]

# group by stream (tid); find the busiest stream = main compute
by_tid = defaultdict(list)
for e in evs:
    by_tid[e.get("tid")].append((e.get("ts", 0.0), e.get("dur", 0.0),
                                 e.get("name", "?")))
for t in by_tid:
    by_tid[t].sort()
main_tid = max(by_tid, key=lambda t: sum(d for _, d, _ in by_tid[t]))
seq = by_tid[main_tid]
print(f"main compute stream tid={main_tid}, {len(seq)} kernels")

# (1) inter-kernel gap distribution (only positive gaps = GPU idle between
# consecutive kernels on the same stream)
gaps = []
for i in range(1, len(seq)):
    g = seq[i][0] - (seq[i - 1][0] + seq[i - 1][1])
    if g > 0.05:  # >50ns
        gaps.append((g, seq[i - 1][2], seq[i][2]))
gaps.sort(reverse=True)
tot_gap = sum(g for g, _, _ in gaps)
print(f"\n=== inter-kernel idle gaps (main stream) ===")
print(f"total idle-gap {tot_gap/1e3:.1f}ms across {len(gaps)} gaps")
print("largest gap types (prev->next, summed):")
agg = defaultdict(lambda: [0.0, 0])
for g, p, n in gaps:
    def sh(x):
        import re
        x = re.sub(r"^void ", "", x)
        if "deep_gemm" in x:
            m = re.search(r"impl<(\d+)u, (\d+)u, (\d+)u", x)
            return f"gemm<{m.group(2)},{m.group(3)}>" if m else "gemm"
        return re.match(r"([A-Za-z0-9_:]+)", x).group(1)[:30]
    agg[(sh(p), sh(n))][0] += g
    agg[(sh(p), sh(n))][1] += 1
for (p, n), (g, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
    print(f"  {g/1e3:6.2f}ms x{c:<5} [{p} -> {n}]")

# (2) per-kernel per-call distribution (outliers)
percall = defaultdict(list)
for ts, d, nm in seq:
    import re
    nm = re.sub(r"^void ", "", nm)
    if "deep_gemm" in nm:
        m = re.search(r"impl<(\d+)u, (\d+)u, (\d+)u", nm)
        nm = f"gemm<{m.group(2)},{m.group(3)}>" if m else "gemm"
    else:
        nm = re.match(r"([A-Za-z0-9_:]+)", nm).group(1)[:28]
    percall[nm].append(d)
print(f"\n=== per-call outliers (max/median ratio, top by total) ===")
rows = [(sum(v), nm, v) for nm, v in percall.items()]
rows.sort(reverse=True)
for tot, nm, v in rows[:12]:
    v.sort()
    med = v[len(v)//2]
    p99 = v[min(len(v)-1, int(len(v)*0.99))]
    mx = v[-1]
    ratio = mx / med if med > 0 else 0
    print(f"  {nm:<30} tot{tot/1e3:6.1f}ms n{len(v):<5} "
          f"med{med:6.1f} p99{p99:7.1f} max{mx:8.1f}us (max/med {ratio:4.1f}x)")
