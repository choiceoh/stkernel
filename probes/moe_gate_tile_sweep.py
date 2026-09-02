#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""moe_gate_sm121 router gate: tile sweep on a given router shape, cold L2.

Loads the SHIPPED kernel from overlay/modules/moe_gate_sm121/moe_gate_sm121.py
(no copy) and sweeps BLOCK_N / BLOCK_K / warps / stages as launch-time
constexprs. 46 distinct weights cycle inside one CUDA graph so every launch
streams from DRAM (the module's own sweep method). Reports us/launch and
whether the fp32 output is bit-identical to the shipped config (BN16/BK512,
warps 4, stages 4 for BLOCK_M=16 and 3 for BLOCK_M=32 -- the module's ladder).
M is capped at 32 (the kernel's largest BLOCK_M; rows past it are not written).

    docker run --rm --gpus all --entrypoint python3 \
      --mount type=bind,src=$REPO,dst=/repo,readonly glm53:v13-b12x \
      /repo/probes/moe_gate_tile_sweep.py --E 288 --K 4096 --M 8 16 24
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")


def _load_module():
    os.environ.setdefault("VLLM_MOE_GATE_FUSED", "1")  # the kernel is defined under this gate
    spec = importlib.util.spec_from_file_location(
        "moe_gate_sm121", "/repo/overlay/modules/moe_gate_sm121/moe_gate_sm121.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def shipped_stages(block_m: int) -> int:
    return 4 if block_m == 16 else 3


def run(mod, x, w, out, BN, BK, warps, stages):
    M = x.shape[0]
    N, K = w.shape
    BM = 16 if M <= 16 else 32
    mod._deneb_gate_partial_kernel[(N // BN, 1)](
        x, w, out, M, x.stride(0), N=N, K=K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        SPLIT_K=1, num_warps=warps, num_stages=stages)


def stock(x, w):
    return torch.nn.functional.linear(x, w).to(torch.float32)


def time_graph(fn, n, reps=20):
    for i in range(3):
        fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        with torch.cuda.graph(g, stream=st):
            for i in range(n):
                fn(i)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / (reps * n) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=288, help="routed experts (GLM-5.3-Flash: 288)")
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--M", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--nw", type=int, default=46)
    args = ap.parse_args()
    if max(args.M) > 32:
        print("M must be <= 32 (kernel BLOCK_M cap)")
        return 2
    torch.manual_seed(0)
    dev = "cuda"
    mod = _load_module()
    ws = [(torch.randn(args.E, args.K, device=dev) * 0.02).to(torch.bfloat16) for _ in range(args.nw)]
    for M in args.M:
        BM = 16 if M <= 16 else 32
        xs = [torch.randn(M, args.K, device=dev).to(torch.bfloat16) for _ in range(args.nw)]
        out = torch.empty(M, args.E, device=dev, dtype=torch.float32)
        ref = torch.empty_like(out)
        run(mod, xs[0], ws[0], ref, 16, 512, 4, shipped_stages(BM))
        torch.cuda.synchronize()
        ref = ref.clone()
        t_stock = time_graph(lambda i: stock(xs[i], ws[i]), args.nw)
        print(f"M={M} E={args.E} K={args.K}: stock F.linear+cast {t_stock:.1f} us | "
              f"max|stock-fused| {(stock(xs[0], ws[0]) - ref).abs().max().item():.2e}")
        rows = []
        for BN, BK, warps, stages in itertools.product((8, 16, 32), (256, 512, 1024), (2, 4, 8), (2, 3, 4)):
            if args.E % BN or (BM + BN) * BK * 2 * (stages - 1) > 99 * 1024:
                continue
            try:
                out.zero_()
                run(mod, xs[0], ws[0], out, BN, BK, warps, stages)
                torch.cuda.synchronize()
                exact = torch.equal(out, ref)
                t = time_graph(lambda i: run(mod, xs[i], ws[i], out, BN, BK, warps, stages), args.nw)
                rows.append((t, BN, BK, warps, stages, exact))
            except Exception as e:  # a config Triton refuses (resources); CUDA faults abort below
                torch.cuda.synchronize()
                rows.append((float("inf"), BN, BK, warps, stages, f"ERR {type(e).__name__}"))
        rows.sort(key=lambda r: r[0])
        for t, BN, BK, warps, stages, exact in rows:
            tag = " <- shipped" if (BN, BK, warps, stages) == (16, 512, 4, shipped_stages(BM)) else ""
            if t != float("inf") and (rows.index((t, BN, BK, warps, stages, exact)) < 12 or tag):
                print(f"   {t:6.1f} us  BN{BN:2d} BK{BK:4d} w{warps} s{stages} programs={args.E//BN:3d} "
                      f"bit-exact-vs-shipped={exact}{tag}")
        errs = [r for r in rows if r[0] == float("inf")]
        if errs:
            print(f"   {len(errs)} configs refused:", [(r[1], r[2], r[3], r[4], r[5]) for r in errs[:4]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
