#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""moe_gate_sm121 router gate: tile sweep on the GLM-5.3 shape, cold L2.

Copies _deneb_gate_partial_kernel verbatim (SPLIT_K=1 path) and sweeps
BLOCK_N / BLOCK_K / warps / stages. 46 distinct weights cycle inside one CUDA
graph so every launch streams from DRAM (the module's own sweep method).
Reports us/launch and whether the fp32 output is bit-identical to the shipped
config (BN16/BK512/w4/s4)."""
import argparse, itertools, sys
import torch, triton, triton.language as tl


@triton.jit
def _gate_kernel(x_ptr, w_ptr, part_ptr, M, stride_xm, N: tl.constexpr, K: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_lo = pid_k * (K // SPLIT_K)
    for k0 in range(0, K // SPLIT_K, BLOCK_K):
        rk = k_lo + k0 + tl.arange(0, BLOCK_K)
        x_tile = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=m_mask[:, None], other=0.0)
        w_tile = tl.load(w_ptr + rn[:, None] * K + rk[None, :])
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
    part = part_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :]
    tl.store(part, acc, mask=m_mask[:, None])


def run(x, w, out, BN, BK, warps, stages, SK=1):
    M = x.shape[0]; N, K = w.shape
    BM = 16 if M <= 16 else 32
    _gate_kernel[(N // BN, SK)](x, w, out, M, x.stride(0), N=N, K=K, BLOCK_M=BM, BLOCK_N=BN,
                                BLOCK_K=BK, SPLIT_K=SK, num_warps=warps, num_stages=stages)


def stock(x, w):
    return torch.nn.functional.linear(x, w).to(torch.float32)


def time_graph(fn, ws, xs, reps=20):
    # fn(x, w) over 46 distinct weights inside one graph -> DRAM-cold per launch
    for i in range(3):
        fn(xs[i], ws[i])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        with torch.cuda.graph(g, stream=st):
            for i in range(len(ws)):
                fn(xs[i], ws[i])
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / (reps * len(ws)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=256); ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--M", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--nw", type=int, default=46)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    ws = [(torch.randn(args.E, args.K, device=dev) * 0.02).to(torch.bfloat16) for _ in range(args.nw)]
    for M in args.M:
        xs = [torch.randn(M, args.K, device=dev).to(torch.bfloat16) for _ in range(args.nw)]
        out = torch.empty(M, args.E, device=dev, dtype=torch.float32)
        ref = torch.empty_like(out)
        run(xs[0], ws[0], ref, 16, 512, 4, 4); torch.cuda.synchronize()
        ref = ref.clone()
        t_stock = time_graph(lambda x, w: stock(x, w), ws, xs)
        t_base = time_graph(lambda x, w: run(x, w, out, 16, 512, 4, 4), ws, xs)
        print(f"M={M} E={args.E} K={args.K}: stock F.linear+cast {t_stock:.1f} us | shipped BN16/BK512/w4/s4 {t_base:.1f} us "
              f"({args.E//16} programs), max|stock-fused| {(stock(xs[0], ws[0]) - ref).abs().max().item():.2e}")
        rows = []
        for BN, BK, warps, stages in itertools.product((8, 16, 32), (256, 512, 1024), (2, 4, 8), (2, 3, 4)):
            if args.E % BN: continue
            BM = 16 if M <= 16 else 32
            if (BM + BN) * BK * 2 * (stages - 1) > 99 * 1024: continue
            try:
                out.zero_(); run(xs[0], ws[0], out, BN, BK, warps, stages); torch.cuda.synchronize()
                exact = torch.equal(out, ref)
                t = time_graph(lambda x, w: run(x, w, out, BN, BK, warps, stages), ws, xs)
                rows.append((t, BN, BK, warps, stages, exact))
            except Exception as e:
                rows.append((float("inf"), BN, BK, warps, stages, f"ERR {str(e)[:40]}"))
        rows.sort(key=lambda r: r[0])
        for t, BN, BK, warps, stages, exact in rows[:12]:
            print(f"   {t:6.1f} us  BN{BN:2d} BK{BK:4d} w{warps} s{stages} programs={args.E//BN:3d} bit-exact-vs-shipped={exact}")
        print("   ... slowest:", f"{rows[-1][0]:.1f} us BN{rows[-1][1]} BK{rows[-1][2]} w{rows[-1][3]} s{rows[-1][4]}")


if __name__ == "__main__":
    sys.exit(main())
