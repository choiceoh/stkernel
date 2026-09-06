#!/usr/bin/env python3
"""Exact-byte and same-source A/B for packed FP8 and transposed M<=8 GEMM.

Run through fleet.sh run --gpu --probe and run_mk_probe.sh. Builds both
variants from the same composed .cu; no model load or serving restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


def pack_source(source: str) -> str:
    start = source.index("// BEGIN MK_FP8_PACK4")
    end = source.index("// END MK_FP8_PACK4", start)
    helper = source[start:end]
    # Both variants in one tiny extension, so the byte oracle and candidate
    # receive identical inputs on the same GPU and stream.
    helpers = []
    for flag in (0, 1):
        helpers.append(
            f"#undef MK_FP8_PACK2_DEF\n#define MK_FP8_PACK2_DEF {flag}\n"
            + helper.replace("mk_f32x4_to_e4m3", f"pack{flag}")
        )
    return "\n".join([
        "#include <torch/extension.h>",
        "#include <c10/cuda/CUDAStream.h>",
        "#include <cuda_fp8.h>",
        *helpers,
        r"""
__global__ void compare_pack(const float* x, int* a, int* b, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float4 v = reinterpret_cast<const float4*>(x)[i];
  a[i] = pack0(v.x, v.y, v.z, v.w);
  b[i] = pack1(v.x, v.y, v.z, v.w);
}
void run_pack(torch::Tensor x, torch::Tensor a, torch::Tensor b) {
  const int n = x.numel() / 4;
  compare_pack<<<(n + 255) / 256, 256, 0,
                 c10::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), a.data_ptr<int>(), b.data_ptr<int>(), n);
}
""",
    ])


def byte_gate(ext, torch):
    # All bf16 encodings; every finite e4m3 midpoint and its two FP32
    # neighbours; random raw FP32 bits, including NaNs/subnormals/infinities.
    bf16 = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mid = (values[:-1] + values[1:]) * 0.5
    mid = torch.cat((mid, -mid))
    near = torch.cat((mid, torch.nextafter(mid, torch.full_like(mid, float("inf"))),
                      torch.nextafter(mid, torch.full_like(mid, -float("inf")))))
    gen = torch.Generator().manual_seed(5306)
    raw = torch.randint(-(2**31), 2**31 - 1, (1 << 20,), generator=gen,
                        dtype=torch.int32).view(torch.float32)
    special = torch.tensor([0., -0., 448., -448., 464., -464.,
                            float("inf"), -float("inf"), float("nan")])
    x = torch.cat((bf16.view(torch.bfloat16).float(), near, raw, special))
    x = torch.cat((x, torch.zeros((-x.numel()) % 4))).cuda()
    a = torch.empty(x.numel() // 4, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    ext.run_pack(x, a, b)
    torch.cuda.synchronize()
    mismatches = int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
    print(json.dumps({"gate": "FP8 bytes", "values": x.numel(),
                      "mismatches": mismatches}), flush=True)
    if mismatches:
        raise RuntimeError("packed converter changed FP8 bytes")


def capture(ext, torch, x, pack, n, calls, cold=False):
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    def run():
        ext.run_gemm(x, pack[0], pack[1], out, n, float(pack[2]), 0,
                     0 if len(pack) < 4 or pack[3] is None else pack[3].data_ptr(),
                     0, 0, 0)
    for _ in range(3):
        run()
    if cold:
        from megakernel_glm53_bench import _l2_flush
        _l2_flush((x,))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
    with torch.cuda.graph(graph):
        if cold:
            # Reuse the campaign's write + compute spacer + clean read drain.
            # Captured events exclude that flush and any host launch gap.
            _l2_flush((x,))
        start.record()
        for _ in range(calls):
            run()
        end.record()
    return graph, out, start, end


def graph_time(captured, calls):
    graph, _, start, end = captured
    graph.replay()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / calls


def compact_smlp_gate(mk, torch, replays):
    # Exercise the compact kernel both as the pair_act producer (3072)
    # and as the a_ready consumer (2048), at the admitted M=6 geometry.
    for n_int in (3072, 2048):
        x = torch.randn(6, 4096, device="cuda", dtype=torch.bfloat16)
        gu = mk.build_mk_weight_w4(torch.randn(2 * n_int, 4096, device="cuda",
                                             dtype=torch.bfloat16) * 0.05)
        down = mk.build_mk_weight_w4(torch.randn(4096, n_int, device="cuda",
                                               dtype=torch.bfloat16) * 0.05)
        ref = mk._smlp_ref(x, gu, down, 2 * n_int, n_int, 4096, 10.0)
        def run():
            return mk._smlp2_call(x, gu, down, 2 * n_int, n_int, 4096, 10.0)
        initial = run().clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        worst = 0.0
        for _ in range(replays):
            graph.replay()
            torch.cuda.synchronize()
            ok, error, over_ulp = mk._smlp_gate(out, ref)
            if not ok or not torch.equal(out, initial):
                raise RuntimeError(f"compact SMLP2 n_int={n_int}: error={error}, ulps={over_ulp}")
            worst = max(worst, error)
        print(json.dumps({"gate": "compact SMLP2 graph", "n_int": n_int,
                          "relative_error": worst, "replays": replays}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--compact", action="store_true", help="compare the compact M<=8 allocation")
    args = ap.parse_args()
    if args.reps < 2 or args.calls < 1:
        ap.error("need reps >= 2 and calls >= 1")
    import torch
    from torch.utils.cpp_extension import load_inline
    from vllm.model_executor.layers import glm53_megakernel as mk
    os.environ["VLLM_GLM53_MK_PDL"] = "1"
    torch.set_float32_matmul_precision("highest")

    src = Path(mk._SRC)
    repo_src = Path(__file__).resolve().parents[1] / "build/glm53/glm53_megakernel.cu"
    if src.read_bytes() != repo_src.read_bytes():
        raise RuntimeError("imported CUDA source differs from the composed build")
    print(json.dumps({"device": None if args.compile_only else torch.cuda.get_device_name(), "source": str(src),
                      "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                      "torch": torch.__version__, "cuda": torch.version.cuda}), flush=True)
    gate = load_inline(
        name="mk_fp8_pack_gate", cpp_sources="void run_pack(torch::Tensor, torch::Tensor, torch::Tensor);",
        cuda_sources=pack_source(src.read_text()), functions=["run_pack"],
        extra_cuda_cflags=["-O2", "-gencode", "arch=compute_121a,code=sm_121a"],
        verbose=False,
    )
    if not args.compile_only:
        byte_gate(gate, torch)
    variants = []
    modes = ((0, 0), (1, 1), (1, 2)) if args.compact else ((0, 0), (1, 0), (1, 1))
    names = (["baseline", "pack2+transpose_m8", "pack2+compact_m8"] if args.compact
             else ["baseline", "pack2", "pack2+transpose_m8"])
    for packing, transpose in modes:
        os.environ["VLLM_GLM53_MK_FP8_PACK2"] = str(packing)
        os.environ["VLLM_GLM53_MK_GEMM_TRANSPOSE_M8"] = str(transpose)
        mk._EXT = None
        ext = mk._build()
        if not args.compile_only:
            ext.set_gemm2(0)
        variants.append(ext)
        print(f"compiled pack2={packing} transpose_m8={transpose}", flush=True)
    if args.compile_only:
        print("PASS: all three same-source variants compiled (no GPU validation)", flush=True)
        return
    # Boot gate against the independent dequantized-weight FP32 oracle.
    for ext in variants[1:]:
        mk._EXT = ext
        mk._selftest_gemm_exact()
        if not mk._selftest_smlp2():
            raise RuntimeError("SMLP2 gate failed")
    if args.compact:
        compact_smlp_gate(mk, torch, args.reps)
    torch.manual_seed(53)
    for m, n, k in ((6, 6416, 4096), (6, 4096, 2048), (6, 1024, 4096),
                    (6, 4096, 512), (6, 6144, 4096), (8, 6416, 4096),
                    (12, 6416, 4096), (24, 6416, 4096), (32, 1024, 4096),
                    (1, 129, 128), (2, 1024, 512), (7, 6416, 4096)):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
        pack = mk.build_mk_weight_w4(w)
        ref = (mk._mk_quant_x_ref(x) @ mk.mk_pack_dequant(pack, n).float().T)
        plans = [list(ext.gemm2_plan(m, n, k)) for ext in variants]
        use_compact = args.compact and m == 6 and (n, k) in ((4096, 2048), (6144, 4096))
        if plans[2][2] != (3 if use_compact else 2):
            raise RuntimeError(f"unexpected selected-kernel occupancy: {plans[2]}")
        for cold in (False, True):
            calls = 1 if cold else args.calls
            graphs = [capture(ext, torch, x, pack, n, calls, cold) for ext in variants]
            snapshots = [graph[1].clone() for graph in graphs]
            times = [[], [], []]
            orders = ((0, 1, 2), (2, 1, 0), (1, 2, 0),
                      (0, 2, 1), (2, 0, 1), (1, 0, 2))
            for rep in range(args.reps):
                for flag in orders[rep % len(orders)]:
                    times[flag].append(graph_time(graphs[flag], calls))
                for flag in (1, 2):
                    if not torch.equal(graphs[flag][1], snapshots[flag]):
                        raise RuntimeError(f"{names[flag]} replay drift at {(m, n, k)}, rep={rep}")
                    if not (use_compact and flag == 2) and not torch.equal(graphs[0][1], graphs[flag][1]):
                        raise RuntimeError(f"{names[flag]} output differs at {(m, n, k)}, rep={rep}")
            errors = [mk._exact_gate(graph[1], ref) for graph in graphs]
            if any(not e <= 1e-3 or n_ulp for e, n_ulp in errors):
                raise RuntimeError(f"GEMM oracle failed at {(m, n, k)}: {errors}")
            med = [statistics.median(t) for t in times]
            print(json.dumps({"shape": [m, n, k],
                              "exact": all(torch.equal(graphs[0][1], g[1]) for g in graphs[1:]),
                              "oracle_errors": errors, "plans_ksr_units_bps": plans,
                              "regime": "cold weights, hot x" if cold else "warm graph replay",
                              "variants": names, "median_us": med,
                              "delta_pct": [(t / med[0] - 1) * 100 for t in med],
                              "samples_us": times}), flush=True)
    print("PASS: bytes, GEMM oracle, SMLP2, alternating graph replay", flush=True)


if __name__ == "__main__":
    main()
