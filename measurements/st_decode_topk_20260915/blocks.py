import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import plan
from bench_graph import graph_time, build, SMEM_LIMIT, K
STATIC = 8192
kern = build()
dev = "cuda"; gen = torch.Generator(device=dev)
widths = (128, 256, 512, 1024)
print(f"{'shape':>13} " + " ".join(f"{'T=' + str(w):>8}" for w in widths) + "   best")
for rows, n in ((8,1024),(16,1024),(16,2048),(16,4096),(16,8192),(16,16384),(16,32768),(16,50688),(8,50688)):
    gen.manual_seed(1)
    v = torch.randn(rows, n, device=dev, generator=gen)
    lg = torch.where(v > 0.3, v*3.0, torch.zeros_like(v)).contiguous()
    ke = torch.full((rows,), n, device=dev, dtype=torch.int32)
    out = torch.empty(rows, K, device=dev, dtype=torch.int32)
    bb, st = plan(n, SMEM_LIMIT - STATIC)
    ts = []
    for w in widths:
        f = lambda w=w: kern((rows,1,1),(w,1,1),(lg,lg.stride(0),n,ke,out,st,bb), shared_mem=bb+4*st*4)
        ts.append(graph_time(f))
    print(f"{rows:>4}x{n:<8} " + " ".join(f"{t:8.1f}" for t in ts) + f"   T={widths[ts.index(min(ts))]}")
