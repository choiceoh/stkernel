#!/usr/bin/env python3
"""Does fusing small M=6 projections (concat N) recover bandwidth vs separate
deep_gemm calls? Measures 3x separate N vs 1x concat-N at real shapes.
Run in a fresh GPU container (engine down)."""
import sys
import torch

sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages")
from vllm.utils.deep_gemm import fp8_gemm_nt  # noqa: E402

M, K = 6, 4096
DEV = "cuda"
GROUP = 128


def quant_act(x):
    # per-token-group (128) fp8 quant -> (fp8, scale)
    r, c = x.shape
    xg = x.view(r, c // GROUP, GROUP).to(torch.float32)
    amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    scale = amax / 448.0
    q = (xg / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(r, c), scale.squeeze(-1).to(torch.float32)


def quant_w(w):
    n, c = w.shape
    wg = w.view(n // GROUP, GROUP, c // GROUP, GROUP).to(torch.float32)
    amax = wg.abs().amax((1, 3), keepdim=True).clamp_min(1e-6)
    scale = amax / 448.0
    q = (wg / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(n, c), scale.view(n // GROUP, c // GROUP).to(torch.float32)


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(8):
            fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / (iters * 8)


def case(Ns):
    xq, xs = quant_act(torch.randn(M, K, device=DEV, dtype=torch.bfloat16))
    # separate
    seps = []
    for N in Ns:
        wq, ws = quant_w(torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02)
        out = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
        seps.append((wq, ws, out))

    def run_sep():
        for wq, ws, out in seps:
            fp8_gemm_nt((xq, xs), (wq, ws), out)
    t_sep = bench(run_sep)

    # fused: concat N
    Ntot = sum(Ns)
    wqf, wsf = quant_w(torch.randn(Ntot, K, device=DEV, dtype=torch.bfloat16) * 0.02)
    outf = torch.empty(M, Ntot, device=DEV, dtype=torch.bfloat16)

    def run_fused():
        fp8_gemm_nt((xq, xs), (wqf, wsf), outf)
    t_fused = bench(run_fused)
    wb = Ntot * K
    bw_sep = wb / (t_sep * 1e-6) / 1e9
    bw_fused = wb / (t_fused * 1e-6) / 1e9
    print(f"  N={Ns} (sum {Ntot}): separate {t_sep:6.2f}us ({bw_sep:4.0f}GB/s) "
          f"-> fused {t_fused:6.2f}us ({bw_fused:4.0f}GB/s) "
          f"= {(t_sep - t_fused) / t_sep * 100:+5.1f}%")


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name(0)}, M={M} K={K}")
    case([512, 512, 512])
    case([1024, 1024])
    case([512, 1024, 2048])
    case([2048, 2048])
