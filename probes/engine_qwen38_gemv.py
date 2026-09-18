"""A skinny BF16 GEMV against cuBLAS at the decode step's own shapes, on one GB10 (probe, single-GPU lane).

The decode step census (probes/engine_qwen38_step.py, 2026-09-19) put 11.2 ms of a C=1 step's 20.2 in BF16 GEMMs of 2 to
8 rows: the hyper-connection mixers' two a site (cuBLAS picks sm80 WMMA kernels, about 201 GB/s -- 6.2 ms), the router
(cuBLAS gemv, about 105 GB/s -- 1.2 ms). Every one of them is a weight read once for a handful of rows, so its floor is
its bytes over the memory's bandwidth (273 GB/s). This times a Triton GEMV that reads each weight tile once for all the
rows (tensor-core dot over rows padded to 16, FP32 accumulation, split-K where the output is narrow) against torch.mm,
both inside CUDA graphs, interleaved A/B so production's own steps on the same GPU land on both, with the weights
rotated over more copies than the L2 holds (a step reads a mixer's weights once and 1,600 other launches between).

    python3 probes/engine_kernel_check.py --lanes qwen38_gemv --output /cache/qwen38-gemv.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHAPES = {"hc down": (324, 10240), "hc up": (10240, 320), "router": (513, 2560)}   # (outputs, K) of W [outputs, K]
ROWS = (2, 8)
COPIES_BYTES = 64 << 20        # weights rotated over at least this many bytes: more than GB10's L2
CALLS = 16                     # calls a graph (one per copy in turn)
ROUNDS = 7
CONFIGS = ((64, 128, 1), (32, 256, 1), (16, 256, 4), (16, 512, 8), (32, 128, 4), (16, 256, 8), (64, 64, 1))


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def gemv(X, W, OUT, M, N, K, sx, sw, so, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT: tl.constexpr,
             PARTIAL: tl.constexpr, FP32_DOT: tl.constexpr):
        pid_n, pid_k = tl.program_id(0), tl.program_id(1)
        rows = tl.arange(0, 16)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        span = tl.cdiv(K, SPLIT)
        k0 = pid_k * span
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for k in range(k0, tl.minimum(k0 + span, K), BLOCK_K):
            ks = k + tl.arange(0, BLOCK_K)
            kmask = ks < tl.minimum(k0 + span, K)
            x = tl.load(X + rows[:, None] * sx + ks[None, :], mask=(rows[:, None] < M) & kmask[None, :], other=0.0)
            w = tl.load(W + cols[:, None] * sw + ks[None, :], mask=(cols[:, None] < N) & kmask[None, :], other=0.0)
            if FP32_DOT:                                  # Triton's interpreter reads a BF16 dot's bits as integers
                x, w = x.to(tl.float32), w.to(tl.float32)
            acc += tl.dot(x, tl.trans(w))
        out_mask = (rows[:, None] < M) & (cols[None, :] < N)
        if PARTIAL:
            tl.store(OUT + pid_k * M * N + rows[:, None] * N + cols[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rows[:, None] * so + cols[None, :], acc.to(OUT.dtype.element_ty), mask=out_mask)

    @triton.jit
    def reduce(P, OUT, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in range(SPLIT):
            acc += tl.load(P + s * MN + i, mask=i < MN, other=0.0)
        tl.store(OUT + i, acc.to(OUT.dtype.element_ty), mask=i < MN)

    return gemv, reduce


def skinny_gemv(x, w, config, out=None, partial=None):
    """x [M <= 16, K] bf16 @ w [N, K].T -> [M, N] bf16, one read of w; `partial` fp32 [split, M, N] when split > 1."""
    import torch
    import triton
    gemv, reduce = _KERNELS
    block_n, block_k, split = config
    m, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, block_n), split)
    if split == 1:
        gemv[grid](x, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0), BLOCK_N=block_n, BLOCK_K=block_k,
                   SPLIT=1, PARTIAL=False, FP32_DOT=not x.is_cuda, num_warps=4)
        return out
    if partial is None:
        partial = torch.empty(split, m, n, dtype=torch.float32, device=x.device)
    gemv[grid](x, w, partial, m, n, k, x.stride(0), w.stride(0), n, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT=split,
               PARTIAL=True, FP32_DOT=not x.is_cuda, num_warps=4)
    reduce[(triton.cdiv(m * n, 1024),)](partial, out, m * n, SPLIT=split, BLOCK=1024)
    return out


_KERNELS = None


def run(output=None) -> dict:
    import torch
    global _KERNELS
    _KERNELS = _kernels()
    torch.manual_seed(0)
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS, "shapes": {}}
    for label, (n, k) in SHAPES.items():
        nbytes = n * k * 2
        copies = max(CALLS, -(-COPIES_BYTES // nbytes))
        weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        for m in ROWS:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = (x.float() @ weights[0].float().t())
            row = {"weight_MB": round(nbytes / 1e6, 2)}

            def graph_of(fn):
                outs = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(CALLS)]
                for i in range(CALLS):
                    fn(weights[i % copies], outs[i])
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(CALLS):
                        fn(weights[(i * 7) % copies], outs[i])
                return g

            arms = {"cublas": graph_of(lambda w, o: torch.mm(x, w.t(), out=o))}
            errors = {}
            for config in CONFIGS:
                split = config[2]
                partial = torch.empty(split, m, n, dtype=torch.float32, device="cuda") if split > 1 else None
                got = skinny_gemv(x, weights[0], config, partial=partial).float()
                errors[str(config)] = round(float((got - ref).abs().max() / ref.abs().max()), 6)
                arms[str(config)] = graph_of(lambda w, o, config=config, partial=partial:
                                             skinny_gemv(x, w, config, out=o, partial=partial))
            times = {name: [] for name in arms}
            for _ in range(ROUNDS):
                for name, g in arms.items():              # interleaved: contention lands on every arm alike
                    g.replay()
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    times[name].append((time.perf_counter() - began) / CALLS * 1e6)
            for name, samples in times.items():
                us = statistics.median(samples)
                row[name] = {"us": round(us, 2), "GBps": round(nbytes / us / 1e3, 1),
                             **({"rel_err": errors[name]} if name in errors else {})}
            best = min((name for name in arms if name != "cublas"), key=lambda name: row[name]["us"])
            row["best"] = best
            row["speedup"] = round(row["cublas"]["us"] / row[best]["us"], 3)
            report["shapes"][f"{label} rows {m}"] = row
            print(json.dumps({f"{label} rows {m}": {"cublas": row["cublas"], "best": best, **row[best],
                                                     "speedup": row["speedup"]}}), flush=True)
        del weights
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
