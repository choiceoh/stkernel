"""Graph-replay timing: the decode step is captured, so launch cost is not the question.

Both arms are captured into a CUDA graph and replayed, which is what production does.
Capture also proves the kernel is capture-safe (no sync, no allocation inside).
"""
import os
import sys

import torch
from torch.cuda import _compile_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import plan, block_threads  # noqa: E402
STATIC = 8192

REPO = r"C:\Users\user\Downloads\github\stkernel\.claude\worktrees\heuristic-pare-a5e502"
K, THREADS = 512, 1024
SMEM_LIMIT = 96 * 1024
STATIC = 8192


def build():
    src = open(os.path.join(REPO, "engine", "kernels", "decode_topk.cu"), encoding="utf-8").read()
    src = "#define ST_DECODE_TOPK_NVRTC 1\n" + src.split("#ifndef ST_DECODE_TOPK_NVRTC\nvoid run(")[0]
    k = _compile_kernel(src, "st_dsa_select", compute_capability="120")
    k.set_shared_memory_config(SMEM_LIMIT - STATIC)
    return k


def call(kern, lg, ke, out, bin_bytes, stash):
    smem = bin_bytes + 4 * stash * 4
    kern((lg.shape[0], 1, 1), (block_threads(lg.shape[1]), 1, 1),
         (lg, lg.stride(0), lg.shape[1], ke, out, stash, bin_bytes), shared_mem=smem)


def graph_time(fn, iters=200, inner=10):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / (iters * inner) * 1000.0


def main():
    kern = build()
    dev = "cuda"
    gen = torch.Generator(device=dev)
    peak = float(os.environ.get("ST_PEAK_GBS", "320"))
    print(f"{torch.cuda.get_device_properties(0).name}  smem budget {SMEM_LIMIT//1024} KiB  "
          f"peak assumed {peak:.0f} GB/s")
    print(f"{'rows':>5} {'n_cand':>7} {'MB':>6} {'bins':>6} {'stash':>6} {'mask+topk':>10} "
          f"{'fused':>8} {'speedup':>8} {'GB/s':>6} {'vs 1-read floor':>16}")
    tot_b = tot_f = tot_floor = 0.0
    for rows in (8, 16):
        for n in (1024, 4096, 8192, 16384, 32768, 50688):
            gen.manual_seed(1)
            v = torch.randn(rows, n, device=dev, generator=gen)
            lg = torch.where(v > 0.3, v * 3.0, torch.zeros_like(v)).contiguous()
            ke = torch.full((rows,), n, device=dev, dtype=torch.int32)
            cols = torch.arange(n, device=dev)
            mask = cols[None, :] >= ke[:, None]
            vals = torch.empty(rows, K, device=dev)
            win = torch.empty(rows, K, device=dev, dtype=torch.int64)
            out = torch.empty(rows, K, device=dev, dtype=torch.int32)
            bin_bytes, stash = plan(n, SMEM_LIMIT)

            def base():
                lg.masked_fill_(mask, float("-inf"))
                torch.topk(lg, K, dim=-1, sorted=False, out=(vals, win))

            tb = graph_time(base)
            tf = graph_time(lambda: call(kern, lg, ke, out, bin_bytes, stash))
            mb = rows * n * 4 / 1e6
            floor = mb / 1e3 / peak * 1e6
            tot_b += tb
            tot_f += tf
            tot_floor += floor
            print(f"{rows:5d} {n:7d} {mb:6.2f} {bin_bytes//1024:5d}K {stash:6d} {tb:10.1f} "
                  f"{tf:8.1f} {tb/tf:7.2f}x {mb/1e3/(tf/1e6):6.0f} {tf/floor:15.1f}x")
    print(f"{'sum':>5} {'':>7} {'':>6} {'':>6} {'':>6} {tot_b:10.1f} {tot_f:8.1f} "
          f"{tot_b/tot_f:7.2f}x {'':>6} {tot_f/tot_floor:15.1f}x")


if __name__ == "__main__":
    main()
