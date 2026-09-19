"""Qwen3.8's vocabulary head at decode rows on one GB10: deep_gemm (the lane today) against GLM-5.3's cuBLASLt reader
(probe, single-GPU lane).

A K=3 step at C=1 reads the rank's head four times -- the verify step's and each of the draft chain's three -- and
each read is 62,080 x 2,560 FP8, 159 MB: the step census (q38stepab-0919b) timed deep_gemm's sm120 FP8 GEMM at about
780 us a call, 200 GB/s, 3.1 ms of the step. GLM-5.3 serves its head through engine/kernels/dense/cublaslt_serving
(profiles/glm53/cublas.py, the operator's default there); Qwen3.8's FP8Linear never prepares one. This times, inside
CUDA graphs of 8 calls, interleaved, at 1, 4 and 16 rows:

    read only         every word of the FP8 weight read once and summed: the bandwidth a head could reach here
    deep_gemm         FP8Linear as Qwen3.8 calls it: block-128 activation quantize, deep_gemm fp8_gemm_nt
    cublas direct     the reader's MX path: 32-element activation blocks (e8m0), cuBLASLt's fixed algorithm
    triton fp8        deep_gemm's recipe (the same quantized rows, FP8 dot a 128-wide K block, power-of-two scales)
                      in one program a BLOCK_N of vocabulary rows
    triton w8a16      the BF16 rows against the weight widened in registers, no activation quantization

and each against the BF16 product (largest error over the largest magnitude), deep_gemm's output and its argmax per
row. The first run (q38head-0919a, beside a busy production) had the reader's split and block-128 paths too: no path
beat deep_gemm (878-899 us; the reader 903-1023).

    python3 probes/engine_kernel_check.py --lanes qwen38_head --output /cache/qwen38-head.json
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

VOCAB_LOCAL, HIDDEN = 62080, 2560          # 248,320 / 4 rows a rank; the checkpoint's hidden width
ROWS = (1, 4, 16)
CALLS = 8                                  # calls a graph; the weight (159 MB) is far past the L2 on its own
ROUNDS = 7
FP8_TILES = ((32, 4, 3), (64, 4, 3), (128, 4, 3), (64, 8, 4), (128, 8, 4))    # BLOCK_N, warps, stages
W8A16_TILES = ((64, 4, 3), (128, 8, 3))

try:                                       # module globals: a jit function finds what it calls among them
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _read(W, OUT, CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
        # the bandwidth this lane can reach: every int32 of the weight read once, one sum a program
        pid = tl.program_id(0)
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for c in range(CHUNKS):
            acc += tl.load(W + (pid * CHUNKS + c) * BLOCK + tl.arange(0, BLOCK))
        tl.store(OUT + pid, tl.sum(acc))

    @triton.jit
    def _head_fp8(XQ, XS, WQ, WS, OUT, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
        # deep_gemm's recipe on the tensor cores' FP8 dot: a 128-wide K block's FP32 partial, times the row's and the
        # weight block's power-of-two scales, summed over the blocks
        pid = tl.program_id(0)
        rows = tl.arange(0, 16)
        cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        live, cm = rows < M, cols < N
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for kb in range(K // 128):
            ks = kb * 128 + tl.arange(0, 128)
            xq = tl.load(XQ + rows[:, None] * K + ks[None, :], mask=live[:, None], other=0.0)
            wq = tl.load(WQ + cols[:, None] * K + ks[None, :], mask=cm[:, None], other=0.0)
            xs = tl.load(XS + rows * (K // 128) + kb, mask=live, other=0.0)
            ws = tl.load(WS + (cols // 128) * (K // 128) + kb, mask=cm, other=0.0)
            acc += tl.dot(xq, tl.trans(wq)) * xs[:, None] * ws[None, :]
        tl.store(OUT + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16), mask=live[:, None] & cm[None, :])

    @triton.jit
    def _head_w8a16(X, WQ, WS, OUT, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
        # BF16 rows against the FP8 weight widened in registers: no activation quantization
        pid = tl.program_id(0)
        rows = tl.arange(0, 16)
        cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        live, cm = rows < M, cols < N
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for kb in range(K // 128):
            ks = kb * 128 + tl.arange(0, 128)
            x = tl.load(X + rows[:, None] * K + ks[None, :], mask=live[:, None], other=0.0)
            w = tl.load(WQ + cols[:, None] * K + ks[None, :], mask=cm[:, None], other=0.0).to(tl.bfloat16)
            ws = tl.load(WS + (cols // 128) * (K // 128) + kb, mask=cm, other=0.0)
            acc += tl.dot(x, tl.trans(w)) * ws[None, :]
        tl.store(OUT + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16), mask=live[:, None] & cm[None, :])


def run(output=None) -> dict:
    import torch
    from engine.kernels.dense import FP8Linear
    from engine.kernels.dense.cublaslt_serving import Reader
    from engine.kernels.dense.fp8 import quantize
    torch.manual_seed(0)
    w = (torch.randn(VOCAB_LOCAL, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    lane = FP8Linear(w)
    direct = Reader(lane.weight)
    nbytes = lane.weight[0].numel() + lane.weight[1].numel() * 4
    report = {"device": torch.cuda.get_device_name(), "weight_MB": round(nbytes / 1e6, 1), "rounds": ROUNDS,
              "calls_a_graph": CALLS, "algorithms": {"direct": direct.report()["algorithms"]}, "rows": {}}
    wq, ws = lane.weight
    words = wq.view(torch.int32).reshape(-1)
    block, chunks = 1024, 16
    programs = words.numel() // (block * chunks)
    sums = torch.empty(programs, dtype=torch.int32, device="cuda")

    def read(x):
        _read[(programs,)](words, sums, CHUNKS=chunks, BLOCK=block, num_warps=4)
        return sums

    def fp8(tile):
        block_n, warps, stages = tile

        def go(x):
            xq, xs = quantize(x)
            out = torch.empty(x.shape[0], VOCAB_LOCAL, dtype=torch.bfloat16, device="cuda")
            _head_fp8[(triton.cdiv(VOCAB_LOCAL, block_n),)](xq, xs, wq, ws, out, x.shape[0], VOCAB_LOCAL, K=HIDDEN,
                                                            BLOCK_N=block_n, num_warps=warps, num_stages=stages)
            return out
        return go

    def w8a16(tile):
        block_n, warps, stages = tile

        def go(x):
            out = torch.empty(x.shape[0], VOCAB_LOCAL, dtype=torch.bfloat16, device="cuda")
            _head_w8a16[(triton.cdiv(VOCAB_LOCAL, block_n),)](x, wq, ws, out, x.shape[0], VOCAB_LOCAL, K=HIDDEN,
                                                              BLOCK_N=block_n, num_warps=warps, num_stages=stages)
            return out
        return go

    arms_of = {
        "deep_gemm": lambda x: lane(x),
        "cublas direct": lambda x: direct(x)[:, :VOCAB_LOCAL],
        **{f"triton fp8 {t}": fp8(t) for t in FP8_TILES},
        **{f"triton w8a16 {t}": w8a16(t) for t in W8A16_TILES},
    }
    report["read_programs"] = programs
    for m in ROWS:
        x = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        outs = {name: fn(x) for name, fn in arms_of.items()}
        torch.cuda.synchronize()
        base_pick = outs["deep_gemm"].float().argmax(-1)
        checks = {name: {"rel_err": round(float((o.float() - ref).abs().max() / ref.abs().max()), 6),
                         "vs_deep_gemm": round(float((o.float() - outs["deep_gemm"].float()).abs().max()
                                                     / outs["deep_gemm"].float().abs().max()), 6),
                         "argmax_as_bf16": round(float((o.float().argmax(-1) == ref.argmax(-1)).float().mean()), 3),
                         "argmax_as_deep_gemm": round(float((o.float().argmax(-1) == base_pick).float().mean()), 3)}
                  for name, o in outs.items()}
        keep = []

        def graph_of(fn):
            keep.append(fn(x))
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(CALLS):
                    keep.append(fn(x))
            return g

        graphs = {"read only": graph_of(read), **{name: graph_of(fn) for name, fn in arms_of.items()}}
        times = {name: [] for name in graphs}
        for _ in range(ROUNDS):
            for name, g in graphs.items():
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        del graphs, keep
        row = {name: {"us": round(statistics.median(v), 1), "GBps": round(nbytes / statistics.median(v) / 1e3, 1),
                      **checks.get(name, {})} for name, v in times.items()}
        report["rows"][m] = row
        print(json.dumps({f"head rows {m}": row}), flush=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
