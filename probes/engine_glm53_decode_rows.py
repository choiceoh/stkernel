"""GLM-5.3's decode-row GEMMs against the kernels Qwen3.8's campaign built for the same work, on one GB10 (probe,
single-GPU lane).

Qwen3.8's campaign built two kernels for weights read once for a handful of rows: engine/kernels/dense/fp8_rows (the
vocabulary head, #1217: 890-908 -> 693-716 us a call on Qwen's head) and engine/kernels/common/skinny_gemv (BF16 x@W.T
for 2..16 rows, #1207: the router 27.9-35.7 -> 14.1-14.3 us). GLM-5.3 serves the same kind of work through other
readers, and whether either kernel pays there is a GB10's answer, not a reading of the code. This lane asks it at
GLM's shapes and GLM's rows (SPEC_K = 7: a verify step is 8 rows a request, a draft pass 7; C = 1 and 2 are the rows
both kernels admit), with synthetic weights of the served shape and format:

    head   the rank's vocabulary head, 38,720 x 4,096 e4m3 with block-128 scales (159 MB, far past the L2 on its own):
           verify  cublaslt  the served reader (glm53/cublas.prepare): MX32 activation quantize + cuBLASLt, direct
                   fp8_rows  FP8Linear(decode_rows=True): block-128 activation quantize + fp8_rows.project
                   w8a16     fp8_rows.project_bf16: the rows NOT quantised (when the kernel exists on this checkout)
           draft   cublaslt  Reader.project_mx on MX32 rows, as the drafter's fused add_norm_head hands them over
                   fp8_rows  fp8_rows.project on block-128 rows (the quantize outside the timing, like the producer's)
           deep_gemm         FP8Linear without a reader: what the head was before cuBLASLt
           read only         every word of the weight read once: the floor
           each against the BF16 product (largest error over the largest magnitude) and its argmax per row, and the
           served cuBLASLt output's argmax

    gemv   engine/kernels/common/skinny_gemv's configurations against torch.mm (probes/engine_qwen38_gemv.run) at the
           GLM step's BF16 GEMMs still on cuBLAS: the indexer's wk+gate pair [256, 4096] (decode_projection.IndexerPair,
           eleven DSA layers a step) and the drafter's context K/V [2560, 4096] (once a step), weights rotated past the L2

    python3 probes/engine_kernel_check.py --lanes glm53_head --output /cache/glm53-head.json
    python3 probes/engine_kernel_check.py --lanes glm53_gemv --output /cache/glm53-gemv.json

Numbers, not a verdict: a kernel that wins here is wired into GLM's serving path by a pull request that says so, and a
speed claim on the served step is the fleet's (D17).
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

# engine/profiles/glm53/facts: vocab_size 154,880 over TP = 4, hidden_size 4,096 (the checkpoint's config)
VOCAB_LOCAL, HIDDEN = 38720, 4096
HEAD_ROWS = (7, 8, 14, 16)                 # draft and verify rows at C = 1 and C = 2 (SPEC_K = 7)
GEMV_ROWS = (8, 16)                        # a verify step's rows at C = 1 and C = 2
CALLS = 8                                  # head calls a graph
ROUNDS = 9
TILES = ((32, 4, 3), (32, 4, 4), (64, 4, 3), (64, 4, 4), (128, 8, 3))    # fp8_rows (BLOCK_N, warps, stages) at K = 4096

# (outputs, K) -> skinny_gemv configurations (BLOCK_N, BLOCK_K, SPLIT, warps, stages). The pair is 256 outputs over a
# 4,096-wide K: 16 programs at BLOCK_N 16 without a split, so the split is most of the sweep.
GEMV_SHAPES = {
    "glm53 indexer wk+gate": ((256, 4096), ((16, 256, 1, 4, 3), (16, 256, 2, 4, 3), (16, 256, 4, 4, 3),
                                            (16, 256, 8, 4, 3), (32, 256, 4, 4, 3), (16, 128, 8, 4, 3),
                                            (16, 512, 4, 4, 2), (32, 128, 8, 4, 3))),
    "glm53 drafter ctx kv": ((2560, 4096), ((16, 256, 1, 4, 3), (32, 256, 1, 4, 3), (32, 128, 1, 4, 3),
                                            (64, 128, 1, 4, 3), (16, 256, 2, 4, 3), (32, 256, 2, 4, 3),
                                            (64, 256, 1, 8, 3), (128, 128, 1, 8, 3))),
}


def block_quantize(w):
    """(q [N', K] e4m3, s [N'/128, K/128] fp32) with N' = N padded to 128: deep_gemm's per_block_cast_to_fp8 recipe
    (block-128 amax, UE8M0 power-of-two scales), for a box without deep_gemm. The ST image has it, and there the
    probe packs through FP8Linear itself; this is what lets the lane be run end to end off a GB10 first."""
    import torch
    n, k = w.shape
    padded = torch.nn.functional.pad(w.float(), (0, 0, 0, (n + 127) // 128 * 128 - n))
    blocks = padded.view(padded.shape[0] // 128, 128, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks / scale).to(torch.float8_e4m3fn).view(padded.shape)
    return q.contiguous(), scale.view(padded.shape[0] // 128, k // 128).contiguous()


def run_head(output=None) -> dict:
    import torch
    import triton
    from engine.kernels.dense import FP8Linear, fp8_rows, mxfp8
    from engine.kernels.dense.fp8 import quantize
    from probes.engine_qwen38_head import _read
    torch.manual_seed(0)
    device = torch.device("cuda")
    w = (torch.randn(VOCAB_LOCAL, HIDDEN, device=device) * 0.02).to(torch.bfloat16)
    try:
        packed = FP8Linear(w).weight                               # block-128 weight, as the pack store holds it
        packer = "deep_gemm"
    except ImportError:
        packed, packer = block_quantize(w), "probe (no deep_gemm on this box)"
    reference = FP8Linear(w, quantized=packed)
    rows_kernel = FP8Linear(w, quantized=packed, decode_rows=True)
    served = FP8Linear(w, quantized=packed)                        # + the cuBLASLt reader, as glm53/cublas.prepare
    unavailable = {}
    try:
        served.prepare_cublas(split_decode=False)
    except Exception as exc:                                       # noqa: BLE001 -- recorded, the other arms still run
        unavailable["cublaslt"] = f"{type(exc).__name__}: {exc}"[:200]
        served = None
    wq, ws = packed
    bf16_rows = getattr(fp8_rows, "project_bf16", None)            # #1246's W8A16 kernel, when this checkout has it
    nbytes = wq.numel() + ws.numel() * 4
    try:
        qualified = fp8_rows.qualify(device)                       # the boot's own D3 check of the kernel
    except ImportError as exc:
        qualified = f"not run: {exc}"
    report = {"device": torch.cuda.get_device_name(), "weight_MB": round(nbytes / 1e6, 1), "shape": [VOCAB_LOCAL, HIDDEN],
              "packer": packer, "rounds": ROUNDS, "calls_a_graph": CALLS, "qualify": qualified,
              "w8a16": bf16_rows is not None, "unavailable": unavailable, "rows": {}}
    print(json.dumps({"qualify": report["qualify"], "packer": packer, "w8a16": report["w8a16"],
                      "unavailable": unavailable}), flush=True)
    words = wq.view(torch.int32).reshape(-1)
    block, chunks = 1024, 16
    programs = words.numel() // (block * chunks)
    sums = torch.empty(programs, dtype=torch.int32, device=device)

    def read(_x):
        _read[(programs,)](words, sums, CHUNKS=chunks, BLOCK=block, num_warps=4)
        return sums

    def tiled(tile_):
        block_n, warps, stages = tile_

        def go(x):
            q, s = quantize(x)
            out = torch.empty(x.shape[0], wq.shape[0], dtype=torch.bfloat16, device=device)
            fp8_rows._fp8_rows[(triton.cdiv(wq.shape[0], block_n),)](q, s, wq, ws, out, x.shape[0], wq.shape[0],
                                                                      out.stride(0), K=HIDDEN, BLOCK_N=block_n,
                                                                      num_warps=warps, num_stages=stages)
            return out
        return go

    for m in HEAD_ROWS:
        x = torch.randn(m, HIDDEN, device=device).to(torch.bfloat16)
        q_mx, s_mx = mxfp8.quantize(x, num_warps=1)                # what the drafter's producer hands the reader
        q_128, s_128 = quantize(x)
        arms = {"verify fp8_rows": lambda x_: rows_kernel(x_),
                "draft fp8_rows block-128 rows": lambda _x: fp8_rows.project(q_128, s_128, packed),
                "deep_gemm": lambda x_: reference(x_),
                **{f"fp8_rows tile {t}": tiled(t) for t in TILES}}
        if served is not None:
            arms = {"verify cublaslt (served)": lambda x_: served(x_),
                    "draft cublaslt mx rows (served)": lambda _x: served.cublas.project_mx(q_mx, s_mx), **arms}
        if bf16_rows is not None:
            arms["w8a16 (verify or draft)"] = lambda x_: bf16_rows(x_, packed)
        ref = x.float() @ w.float().t()
        outs = {}
        for name, fn in list(arms.items()):
            try:
                outs[name] = fn(x)[:, :VOCAB_LOCAL].float()
                torch.cuda.synchronize()
            except Exception as exc:                               # noqa: BLE001 -- one arm's failure is its result
                unavailable[f"{name} rows {m}"] = f"{type(exc).__name__}: {exc}"[:200]
                del arms[name]
        base = "verify cublaslt (served)" if "verify cublaslt (served)" in outs else "deep_gemm" if "deep_gemm" in outs \
            else next(iter(outs))
        served_argmax = outs[base].argmax(-1)
        checks = {name: {"rel_err": round(float((o - ref).abs().max() / ref.abs().max()), 6),
                         "argmax_as_bf16": round(float((o.argmax(-1) == ref.argmax(-1)).float().mean()), 3),
                         f"argmax_as_{base.split(' (')[0].replace(' ', '_')}": round(float((o.argmax(-1) == served_argmax)
                                                                                           .float().mean()), 3)}
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

        graphs = {"read only": graph_of(read), **{name: graph_of(fn) for name, fn in arms.items()}}
        times = {name: [] for name in graphs}
        for _ in range(ROUNDS):
            for name, g in graphs.items():                         # interleaved: production's steps land on every arm
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        del graphs, keep
        row = {name: {"us": round(statistics.median(v), 1), "GBps": round(nbytes / statistics.median(v) / 1e3, 1),
                      **checks.get(name, {})} for name, v in times.items()}
        row["fp8_rows_tile"] = list(fp8_rows.tile(m))
        row["argmax_reference"] = base
        report["rows"][m] = row
        print(json.dumps({f"head rows {m}": row}), flush=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


def run_gemv(output=None) -> dict:
    from probes.engine_qwen38_gemv import run
    return run(output, shapes=GEMV_SHAPES, rows=GEMV_ROWS)


if __name__ == "__main__":
    {"head": run_head, "gemv": run_gemv}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else None)
