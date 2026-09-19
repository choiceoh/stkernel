"""Qwen3.8's vocabulary head at decode rows on one GB10: deep_gemm (the lane today) against GLM-5.3's cuBLASLt reader
(probe, single-GPU lane).

A K=3 step at C=1 reads the rank's head four times -- the verify step's and each of the draft chain's three -- and
each read is 62,080 x 2,560 FP8, 159 MB: the step census (q38stepab-0919b) timed deep_gemm's sm120 FP8 GEMM at about
780 us a call, 200 GB/s, 3.1 ms of the step. GLM-5.3 serves its head through engine/kernels/dense/cublaslt_serving
(profiles/glm53/cublas.py, the operator's default there); Qwen3.8's FP8Linear never prepares one. This times, inside
CUDA graphs of 8 calls, interleaved, at 1, 4 and 16 rows:

    deep_gemm         FP8Linear as Qwen3.8 calls it: block-128 activation quantize, deep_gemm fp8_gemm_nt
    cublas direct     the reader's MX path: 32-element activation blocks (e8m0), cuBLASLt's fixed algorithm
    cublas split      the reader's five-way split decode (GLM's drafter fc), partials reduced in a second launch
    cublas block128   the reader fed deep_gemm's own block-128 activations (project_quantized): the same inputs

and each against the BF16 product (largest error over the largest magnitude) and deep_gemm's argmax per row.

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


def run(output=None) -> dict:
    import torch
    from engine.kernels.dense import FP8Linear
    from engine.kernels.dense.cublaslt_serving import Reader
    from engine.kernels.dense.fp8 import quantize
    torch.manual_seed(0)
    w = (torch.randn(VOCAB_LOCAL, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    lane = FP8Linear(w)
    direct = Reader(lane.weight)
    split = Reader(lane.weight, split_decode=True)
    nbytes = lane.weight[0].numel() + lane.weight[1].numel() * 4
    report = {"device": torch.cuda.get_device_name(), "weight_MB": round(nbytes / 1e6, 1), "rounds": ROUNDS,
              "calls_a_graph": CALLS, "algorithms": {"direct": direct.report()["algorithms"],
                                                      "split": split.report()["algorithms"]}, "rows": {}}
    arms_of = {
        "deep_gemm": lambda x: lane(x),
        "cublas direct": lambda x: direct(x)[:, :VOCAB_LOCAL],
        "cublas split": lambda x: split(x, decode=True)[:, :VOCAB_LOCAL],
        "cublas block128": lambda x: direct.project_quantized(*quantize(x))[:, :VOCAB_LOCAL],
    }
    for m in ROWS:
        x = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        outs = {name: fn(x) for name, fn in arms_of.items()}
        torch.cuda.synchronize()
        base_pick = outs["deep_gemm"].float().argmax(-1)
        checks = {name: {"rel_err": round(float((o.float() - ref).abs().max() / ref.abs().max()), 6),
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

        graphs = {name: graph_of(fn) for name, fn in arms_of.items()}
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
                      **checks[name]} for name, v in times.items()}
        report["rows"][m] = row
        print(json.dumps({f"head rows {m}": row}), flush=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
