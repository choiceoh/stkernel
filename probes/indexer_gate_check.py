#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53_indexer_gate_splitk: numerics, determinism and timing of the head gate
through the REAL entry point (`head_gate` with the knob set), on the checkpoint's
shape (index_n_heads from --config, default 32 = the fleet checkpoint).

Checks (non-zero exit on any failure):
  * routing: M<=16 takes split-K, M>16 / strided x / K mismatch keep torch.mm
  * numerics vs torch.mm over random and activation-scale inputs: max |diff|,
    top-1 / top-4 pool-ranking flips
  * determinism: 50 repeated launches bit-identical (the stock cuBLAS path is
    deterministic; the ranks' top-k selections must agree)
  * timing by CUDA-graph replay with 46 distinct weights cycled (DRAM-cold),
    M = 1..32, printing the crossover against torch.mm

Run inside a glm53 image with the repo mounted at /repo:

    docker run --rm --gpus all --entrypoint python3 \
      --mount type=bind,src=$REPO,dst=/repo,readonly glm53:v13-b12x \
      /repo/probes/indexer_gate_check.py [--config /models/.../config.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
ENV = "VLLM_GLM53_INDEXER_GATE_SPLITK"
NW = 46  # weights cycled inside one graph so every launch streams from DRAM


def _load_kernel():
    spec = importlib.util.spec_from_file_location(
        "glm53_indexer_gate", "/repo/overlay/modules/glm53_indexer_gate_splitk/glm53_indexer_gate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _graph_us(fn, n=NW, reps=20):
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
    ap.add_argument("--config", default=None, help="checkpoint config.json (index_n_heads, hidden_size)")
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    N, K = args.N, args.K
    if args.config:
        c = json.load(open(args.config))
        c = c.get("text_config", c)
        N, K = int(c["index_n_heads"]), int(c["hidden_size"])
    torch.manual_seed(0)
    dev = "cuda"
    os.environ[ENV] = "1"
    mod = _load_kernel()
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    def mm(x, w):
        return torch.mm(x.float(), w)

    w = (torch.randn(N, K, device=dev) * 0.02).t().contiguous()
    x8 = (torch.randn(8, K, device=dev) * 1.5).to(torch.bfloat16)
    print(f"shape: M<=16 x K={K} @ K x N={N} (index_n_heads)")
    # --- routing
    check(mod.splitk_applicable(x8, w), "M=8 bf16 contiguous -> split-K")
    check(mod.splitk_applicable(x8.float(), w), "fp32 x -> split-K")
    check(not mod.splitk_applicable(torch.randn(17, K, device=dev).to(torch.bfloat16), w), "M=17 -> torch.mm")
    xt = torch.randn(K, 8, device=dev).to(torch.bfloat16).t()
    check(not mod.splitk_applicable(xt, w), "column-strided x -> torch.mm")
    check(torch.equal(mod.head_gate(xt, w), mm(xt, w)), "head_gate(strided x) == torch.mm bitwise")
    check(not mod.splitk_applicable(x8[:, : K // 2], w), "K mismatch -> torch.mm")
    check(not mod.splitk_applicable(x8, w.t()), "non-contiguous w -> torch.mm")
    check(not mod.splitk_applicable(x8, torch.randn(K, mod.MAX_N + 1, device=dev)), f"N={mod.MAX_N + 1} -> torch.mm")
    check(torch.equal(mod.head_gate(x8[:0], w), mm(x8[:0], w)), "M=0 -> torch.mm")
    # --- numerics vs torch.mm (routing on)
    worst_abs = worst_rel = 0.0
    flips1 = flips4 = rows = 0
    top4 = lambda a: a.topk(4, dim=1).indices.sort(1).values  # noqa: E731
    for t in range(args.trials):
        M = int(torch.randint(1, mod.MAX_M + 1, (1,)).item())
        x = (torch.randn(M, K, device=dev) * 1.5).to(torch.bfloat16)
        wt = (torch.randn(N, K, device=dev) * 0.02).t().contiguous()
        ref = mm(x, wt)
        out = mod.head_gate(x, wt)
        d = (out - ref).abs()
        worst_abs = max(worst_abs, d.max().item())
        worst_rel = max(worst_rel, (d / ref.abs().amax(1, keepdim=True).clamp_min(1e-6)).max().item())
        flips1 += int((out.argmax(1) != ref.argmax(1)).sum().item())
        flips4 += int((top4(out) != top4(ref)).any(1).sum().item())
        rows += M
    print(f"numerics over {args.trials} trials / {rows} rows: max|diff| {worst_abs:.3e} abs, "
          f"{worst_rel:.3e} of row max; top-1 flips {flips1}, top-4 set changes {flips4}")
    check(worst_rel < 1e-5, "relative error < 1e-5 of the row max")
    check(flips1 == 0 and flips4 == 0, "no top-1 / top-4 ranking change vs torch.mm")
    # --- determinism
    outs = [mod.head_gate(x8, w) for _ in range(50)]
    check(all(torch.equal(outs[0], o) for o in outs), "50 launches bit-identical")
    # --- timing, DRAM-cold, crossover
    ws = [(torch.randn(N, K, device=dev) * 0.02).t().contiguous() for _ in range(NW)]
    print(f"timing (CUDA-graph replay, {NW} weights cycled):")
    for M in (1, 2, 4, 8, 12, 16, 24, 32):
        xs = [torch.randn(M, K, device=dev).to(torch.bfloat16) for _ in range(NW)]
        t_mm = _graph_us(lambda i: mm(xs[i], ws[i]))
        if M <= mod.MAX_M:
            t_sk = _graph_us(lambda i: mod.head_gate_splitk(xs[i], ws[i]))
            print(f"  M={M:2d}: torch.mm {t_mm:6.1f} us | split-K {t_sk:6.1f} us | routed: split-K")
            check(t_sk < t_mm, f"M={M}: split-K faster than torch.mm")
        else:
            print(f"  M={M:2d}: torch.mm {t_mm:6.1f} us | routed: torch.mm")
    print("RESULT:", "OK" if not fails else f"{len(fails)} FAIL: " + "; ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
