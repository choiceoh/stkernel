"""Qwen3.8's vocabulary head at decode rows on one GB10: deep_gemm against engine/kernels/dense/fp8_rows (probe,
single-GPU lane).

A K=3 step at C=1 reads the rank's head four times -- the verify step's and each of the draft chain's three -- and
each read is 62,080 x 2,560 FP8, 159 MB: the step census (q38stepab-0919b) timed deep_gemm's sm120 FP8 GEMM at about
780 us a call, 3.1 ms of the step. This times, inside CUDA graphs of 8 calls, interleaved, at 1, 2, 4, 8 and 16 rows:

    read only         every word of the FP8 weight read once and summed: the bandwidth a head could reach here
    deep_gemm         FP8Linear as Qwen3.8 called it: block-128 activation quantize, deep_gemm fp8_gemm_nt
    served            FP8Linear(decode_rows=True): the same quantize, then fp8_rows at its tile for the rows
    fp8_rows (N, ...) the kernel at each tile of the sweep (BLOCK_N, warps, stages)

and each against the BF16 product (largest error over the largest magnitude), deep_gemm's output and its argmax per
row, after fp8_rows.qualify (the boot's check). Earlier runs: q38head-0919a (beside a busy production) had GLM's
cuBLASLt reader -- direct MX, five-way split, block-128 inputs -- none beat deep_gemm (878-899 against 903-1023 us);
q38head-0919b timed the prototype fp8_rows came from (742-836 us, deep_gemm 949-1013, the read 695-747).

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
ROWS = (1, 2, 4, 8, 16)
CALLS = 8                                  # calls a graph; the weight (159 MB) is far past the L2 on its own
ROUNDS = 9
TILES = ((32, 4, 3), (64, 4, 3), (128, 4, 3), (128, 8, 3), (32, 4, 4))    # BLOCK_N, warps, stages

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


def run(output=None) -> dict:
    import torch
    from engine.kernels.dense import FP8Linear, fp8_rows
    from engine.kernels.dense.fp8 import quantize
    torch.manual_seed(0)
    qualified = fp8_rows.qualify(torch.device("cuda"))
    w = (torch.randn(VOCAB_LOCAL, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    lane = FP8Linear(w)
    served = FP8Linear(w, quantized=lane.weight, decode_rows=True)
    nbytes = lane.weight[0].numel() + lane.weight[1].numel() * 4
    report = {"device": torch.cuda.get_device_name(), "weight_MB": round(nbytes / 1e6, 1), "rounds": ROUNDS,
              "calls_a_graph": CALLS, "qualify": qualified, "rows": {}}
    print(json.dumps({"qualify": qualified}), flush=True)
    wq, ws = lane.weight
    words = wq.view(torch.int32).reshape(-1)
    block, chunks = 1024, 16
    programs = words.numel() // (block * chunks)
    sums = torch.empty(programs, dtype=torch.int32, device="cuda")

    def read(x):
        _read[(programs,)](words, sums, CHUNKS=chunks, BLOCK=block, num_warps=4)
        return sums

    def at(tile_):
        block_n, warps, stages = tile_

        def go(x):
            q, s = quantize(x)
            out = torch.empty(x.shape[0], VOCAB_LOCAL, dtype=torch.bfloat16, device="cuda")
            fp8_rows._fp8_rows[(triton.cdiv(VOCAB_LOCAL, block_n),)](q, s, wq, ws, out, x.shape[0], VOCAB_LOCAL,
                                                                      out.stride(0), K=HIDDEN, BLOCK_N=block_n,
                                                                      num_warps=warps, num_stages=stages)
            return out
        return go

    arms_of = {"deep_gemm": lambda x: lane(x), "served": lambda x: served(x),
               "draft w8a16": lambda x: fp8_rows.project_bf16(x, served.weight),       # net.draft_logits
               **{f"fp8_rows {t}": at(t) for t in TILES}}
    for m in ROWS:
        x = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        outs = {name: fn(x) for name, fn in arms_of.items()}
        torch.cuda.synchronize()
        base = outs["deep_gemm"].float()
        checks = {name: {"rel_err": round(float((o.float() - ref).abs().max() / ref.abs().max()), 6),
                         "vs_deep_gemm": round(float((o.float() - base).abs().max() / base.abs().max()), 6),
                         "argmax_as_bf16": round(float((o.float().argmax(-1) == ref.argmax(-1)).float().mean()), 3),
                         "argmax_as_deep_gemm": round(float((o.float().argmax(-1) == base.argmax(-1)).float().mean()), 3)}
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
        row["served_tile"] = list(fp8_rows.tile(m))
        report["rows"][m] = row
        print(json.dumps({f"head rows {m}": row}), flush=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
