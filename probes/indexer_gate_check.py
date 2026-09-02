#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53_indexer_gate_splitk: numerics + timing of the split-K head gate.

Compares head_gate_splitk against the stock torch.mm(x.float(), w) on random
and on checkpoint-like inputs (bf16 hidden states, fp32 weights of the
indexer's scale), reports max |diff| (absolute and relative to the row's
max), top-1 / top-4 pool-ranking flips across the 16 heads, and GPU time by
CUDA-graph replay. Run inside the image (any glm53 image; the kernel is
standalone):

    docker run --rm --gpus all --entrypoint python3 \
      --mount type=bind,src=$REPO,dst=/repo,readonly glm53:v13-b12x \
      /repo/probes/indexer_gate_check.py [--trials 200]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")


def _load_kernel():
    spec = importlib.util.spec_from_file_location(
        "glm53_indexer_gate", "/repo/overlay/modules/glm53_indexer_gate_splitk/glm53_indexer_gate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _gpu_us(fn, n_cap=20, reps=10):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
        st.synchronize()
        with torch.cuda.graph(g, stream=st):
            for _ in range(n_cap):
                fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / (reps * n_cap) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    mod = _load_kernel()
    K, N = 4096, 16
    worst_abs = worst_rel = 0.0
    flips1 = flips4 = rows = 0
    for t in range(args.trials):
        M = int(torch.randint(1, mod.MAX_M + 1, (1,)).item())
        # hidden states ~ bf16 activations, weights ~ small fp32 projections
        x = (torch.randn(M, K, device=dev) * 1.5).to(torch.bfloat16)
        w = (torch.randn(N, K, device=dev) * 0.02).t().contiguous()
        ref = torch.mm(x.float(), w)
        out = mod.head_gate_splitk(x, w)
        d = (out - ref).abs()
        worst_abs = max(worst_abs, d.max().item())
        worst_rel = max(worst_rel, (d / ref.abs().amax(1, keepdim=True).clamp_min(1e-6)).max().item())
        flips1 += int((out.argmax(1) != ref.argmax(1)).sum().item())
        top4 = lambda a: a.topk(4, dim=1).indices.sort(1).values
        flips4 += int((top4(out) != top4(ref)).any(1).sum().item())
        rows += M
    print(f"numerics over {args.trials} trials / {rows} rows: max|diff| {worst_abs:.3e} abs, "
          f"{worst_rel:.3e} of row max; top-1 flips {flips1}, top-4 set changes {flips4}")
    for M in (1, 8, 16, 32):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = (torch.randn(N, K, device=dev) * 0.02).t().contiguous()
        g_mm = _gpu_us(lambda: torch.mm(x.float(), w))
        g_sk = _gpu_us(lambda: mod.head_gate_splitk(x, w))
        t0 = time.perf_counter()
        for _ in range(200):
            mod.head_gate_splitk(x, w)
        torch.cuda.synchronize()
        h_sk = (time.perf_counter() - t0) / 200 * 1e6
        print(f"M={M:2d}: torch.mm {g_mm:6.1f} us GPU | split-K {g_sk:6.1f} us GPU ({h_sk:.0f} us host) | "
              f"{'split-K used' if M <= mod.MAX_M else 'torch.mm kept'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
