"""What the drafter's normalisations cost, the torch way and the fused way (45차 §69).

Two shapes, both at what production runs -- one sequence, K=5, so six rows through a five-layer drafter with
two local KV heads and eight local query heads at TP=4:

  the observe tail   five `rope(rmsnorm(...))` over the context K, one a layer.
  a proposal block   eleven `rmsnorm` and ten `rope(rmsnorm(...))` a block, five blocks a step.

Run it beside nothing: it shares the GPU with whatever else is on it, and the torch side is dozens of launches,
which is exactly what contention inflates. The ratio is the reading; the absolute microseconds are not.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.kernels.norm_rope import norm, norm_rope
from engine.profiles.glm53.drafter import rmsnorm, rope

ROWS, LAYERS, KV, HEADS, D, HIDDEN, THETA, EPS = 6, 5, 2, 8, 128, 4096, 10000.0, 1e-5


def timed(fn, warm=20, iters=200):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this probe measures the CUDA path")
    dev = "cuda"
    ctx = torch.randn(ROWS, LAYERS, 2, KV, D, device=dev, dtype=torch.bfloat16)
    kw = torch.ones(D, device=dev, dtype=torch.bfloat16)
    hw = torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(ROWS, device=dev)
    x = torch.randn(ROWS, HIDDEN, device=dev, dtype=torch.bfloat16)
    q0 = torch.randn(ROWS, HEADS * D, device=dev, dtype=torch.bfloat16)
    k0 = torch.randn(ROWS, KV * D, device=dev, dtype=torch.bfloat16)

    def observe_torch():
        for L in range(LAYERS):
            rope(rmsnorm(ctx[:, L, 0], kw, EPS), pos, THETA)

    def observe_fused():
        for L in range(LAYERS):
            norm_rope(ctx[:, L, 0], kw, EPS, pos, THETA)

    def block_torch():
        for _ in range(LAYERS):
            rmsnorm(x, hw, EPS), rmsnorm(x, hw, EPS)
            rope(rmsnorm(q0.reshape(ROWS, HEADS, D), kw, EPS), pos, THETA)
            rope(rmsnorm(k0.reshape(ROWS, KV, D), kw, EPS), pos, THETA)
        rmsnorm(x, hw, EPS)

    def block_fused():
        for _ in range(LAYERS):
            norm(x, hw, EPS), norm(x, hw, EPS)
            norm_rope(q0.reshape(ROWS, HEADS, D), kw, EPS, pos, THETA)
            norm_rope(k0.reshape(ROWS, KV, D), kw, EPS, pos, THETA)
        norm(x, hw, EPS)

    print(f"{'':22s}{'torch':>10s}{'fused':>10s}{'':>8s}")
    for name, a, b in (("observe tail (x5)", observe_torch, observe_fused),
                       ("proposal block", block_torch, block_fused)):
        t, f = timed(a), timed(b)
        print(f"{name:22s}{t:9.1f}us{f:9.1f}us   {t / f:4.1f}x")
    t, f = timed(block_torch), timed(block_fused)
    print(f"{'a step (observe + 5 blocks)':22s}"
          f"{timed(observe_torch) + 5 * t:9.1f}us{timed(observe_fused) + 5 * f:9.1f}us")


if __name__ == "__main__":
    main()
