#!/usr/bin/env python3
"""Census of the vendor GEMM launches (cutlass / cuBLAS gemmSN / deep_gemm
mqa logits) in a decode trace: per (kernel, grid) the launches per step, the
time per step and the kernels that precede them on the same stream -- which
module they belong to. 30차 §14 found the 3.76 ms/step of "small bf16 GEMMs"
this way: the MLA q_b + indexer wq_b bf16 pair (1.48 ms, EXP-4's target),
the fp32 head gate on cuBLAS's 2-block kernel (0.96 ms, EXP-9's target) and
the absorbed-MLA split-K pair (0.53 ms).

  nice -n 19 python3 tools/trace_vendor_gemm_census.py <trace.json.gz>
"""
import sys, collections, statistics
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from trace_common import cut_steps, load_kernel_events, stream_of
ev = load_kernel_events(sys.argv[1])
anchor, starts = cut_steps(ev)
print(f"kernels={len(ev)} anchor={anchor} steps={len(starts)}")
pat = ("cutlass_80", "gemmSN", "paged_mqa", "splitKreduce", "gemvx", "gemv")
# per step: (short name, grid) -> [count, dur sum]; also the previous kernel on the same stream
agg = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
nsteps = 0
for si in range(1, len(starts) - 1):
    seg = ev[starts[si]:starts[si + 1]]
    if len(seg) < 500:
        continue
    nsteps += 1
    last = {}
    for e in seg:
        st = stream_of(e)
        n = e["name"]
        if any(p in n for p in pat):
            short = n.split("(")[0][-58:]
            grid = tuple(e.get("args", {}).get("grid", []))
            key = (short, grid)
            a = agg[key]; a[0] += 1; a[1] += e["dur"]
            a[2][last.get(st, "?")[:34]] += 1
        last[st] = n
print(f"steps used {nsteps}")
rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
tot = sum(a[1] for _, a in rows) / nsteps
print(f"total {tot:.0f} us/step over {sum(a[0] for _, a in rows) / nsteps:.0f} launches/step")
print(f"{'us/step':>8} {'n/step':>7} {'us/launch':>9}  grid           name  | prev kernels")
for (short, grid), a in rows[:24]:
    prev = ", ".join(f"{k}x{v // nsteps}" for k, v in a[2].most_common(2))
    print(f"{a[1] / nsteps:8.0f} {a[0] / nsteps:7.1f} {a[1] / a[0]:9.1f}  {str(grid):14s} {short[-44:]}  | {prev}")
