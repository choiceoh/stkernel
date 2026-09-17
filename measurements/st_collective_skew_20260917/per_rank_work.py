"""Per-rank compute time inside the same decode steps: does one rank simply do less?"""
import gzip, json, re, sys
from pathlib import Path
root = Path(sys.argv[1]); steps = [int(a) for a in sys.argv[2:]] or [2,3,4,5]
TRANSPORT = re.compile(r"k_publish_packets|k_oneshot_consumer|k_oneshot_moe_packets|k_oneshot_max_int64|k_oneshot_gather_int64|nccl")
print("%-6s %12s %12s %10s %8s" % ("rank", "compute us", "transport us", "kernels", "span us"))
tot = {}
for r in range(4):
    c = t = n = span = 0.0
    for s in steps:
        p = root / ("rank%d-decode-%d.trace.json.gz" % (r, s))
        ev = json.load(gzip.open(p))["traceEvents"]
        ks = [e for e in ev if e.get("cat") == "kernel" and e.get("args", {}).get("stream") == 7]
        if not ks: continue
        lo = min(e["ts"] for e in ks); hi = max(e["ts"] + e["dur"] for e in ks)
        span += hi - lo
        for e in ks:
            n += 1
            if TRANSPORT.search(e["name"]): t += e["dur"]
            else: c += e["dur"]
    tot[r] = (c, t, n, span)
    print("%-6d %12.0f %12.0f %10d %8.0f" % (r, c/len(steps), t/len(steps), n/len(steps), span/len(steps)))
base = tot[0][0]
print()
print("compute 상대값 (rank0 = 100):", {r: round(100*tot[r][0]/base, 1) for r in tot})
print("span 상대값    (rank0 = 100):", {r: round(100*tot[r][3]/tot[0][3], 1) for r in tot})
