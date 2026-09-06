#!/usr/bin/env python3
"""Exact BF16 storage for checkpoint MHC FP32 weights, same-build GPU A/B."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
from mhc_reuse_bench import source_block, mathematical_reference


def packed_source(source):
    p1 = source_block(source, "__device__ void mk_mhc_p1(")
    old = """#pragma unroll
    for (int m = 0; m < NOUT; ++m)
#pragma unroll
      for (int j = 0; j < HC; ++j)
        fnr[m][j] = a.fn[(size_t)m * HC * HIDDEN + j * HIDDEN + h];"""
    assert old in p1
    new = """if constexpr (PACK == 1) {
#pragma unroll
      for (int m = 0; m < NOUT; ++m)
#pragma unroll
        for (int j = 0; j < HC; ++j)
          fnr[m][j] = __bfloat162float(
              ((const __nv_bfloat16*)a.fn)[(size_t)m * HC * HIDDEN + j * HIDDEN + h]);
    } else {
#pragma unroll
      for (int m = 0; m < NOUT; m += 2)
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          unsigned int bits = ((const unsigned int*)a.fn)[(size_t)(m/2) * HC * HIDDEN + j * HIDDEN + h];
          fnr[m][j] = __uint_as_float(bits << 16);
          fnr[m+1][j] = __uint_as_float(bits & 0xffff0000u);
        }
    }"""
    p1 = "template <int PACK>\n" + p1.replace("mk_mhc_p1(", "mk_mhc_packed_p1(", 1).replace(old, new, 1)
    kernel = source_block(source, "__global__ void mk_mhc_kernel(")
    clone = "template <int PACK>\n" + kernel.replace("mk_mhc_kernel(", "mk_mhc_packed_kernel(", 1).replace("mk_mhc_p1(a,", "mk_mhc_packed_p1<PACK>(a,")
    source = source.replace(kernel, kernel + "\n" + p1 + "\n" + clone, 1)
    source = source.replace("void mk_run_mhc(", "int g_probe_mhc_pack = 0;\nvoid mk_run_mhc(", 1)
    launches = []
    for mode in (1, 2):
        launches.append(f"""
  if (g_probe_mhc_pack == {mode}) {{
    static int grid = 0;
    if (!grid) {{
      int blocks = 0, sms = 0;
      MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, mk_mhc_packed_kernel<{mode}>, MK_THREADS, 0));
      MK_CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
      grid = std::min(blocks * sms, MK_MHC_GRID_CAP);
      TORCH_CHECK(grid > 0, "packed MHC has no resident blocks");
    }}
    a.grid = grid;
    mk_launch(mk_mhc_packed_kernel<{mode}>, grid, 0, stream, a);
    return;
  }}
""")
    source = source.replace("  static int mhc_grid = 0;", "".join(launches) + "  static int mhc_grid = 0;", 1)
    return source.replace('  m.def("run_mhc",',
        '  m.def("set_mhc_pack", [](int v) { TORCH_CHECK(v >= 0 && v <= 2); g_probe_mhc_pack = v; });\n  m.def("run_mhc",', 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=18)
    ap.add_argument("--fixtures", default="/cache/mhc-fixtures")
    args = ap.parse_args()
    assert args.reps >= 6 and args.reps % 6 == 0
    os.environ["VLLM_GLM53_MK_PDL"] = "1"
    import torch
    from torch.utils.cpp_extension import load
    from vllm.model_executor.layers import glm53_megakernel as mk
    from megakernel_glm53_bench import _l2_flush
    root = Path(__file__).resolve().parents[1]
    original = Path(mk._SRC).read_bytes()
    assert original == (root / "build/glm53/glm53_megakernel.cu").read_bytes()
    source = packed_source(original.decode())
    sha = hashlib.sha256(source.encode()).hexdigest()
    directory = Path(os.environ["VLLM_GLM53_MK_BUILD_DIR"]) / ("prototype-" + sha[:12])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prototype.cu"
    path.write_text(source)
    print(json.dumps({"source_sha256": hashlib.sha256(original).hexdigest(), "prototype_sha256": sha,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "projection": "exact-bf16-storage"}), flush=True)
    ext = load(name="mhc_pack_" + sha[:12], sources=[str(path)],
        extra_cuda_cflags=["-O2", "-gencode", "arch=compute_121a,code=sm_121a"],
        build_directory=str(directory), verbose=False)
    mk._EXT = ext
    mk._WS = None
    assert mk._selftest_mhc(), "baseline stock oracle"
    assert mk._selftest_mhc_pre(), "baseline pre oracle"
    fixtures = Path(args.fixtures)
    manifest = json.loads((fixtures / "manifest.json").read_text())
    def tensor(meta):
        raw = bytearray((fixtures / meta["file"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == meta["sha256"]
        return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(meta["shape"]).float().cuda()
    weights = [(item["name"], *(tensor(item[k]) for k in ("fn", "scale", "base"))) for item in manifest]
    print(json.dumps({"fixtures": manifest}), flush=True)
    def inputs(tokens, seed, weight):
        torch.manual_seed(seed)
        name, fn, scale, base = weight
        assert torch.equal(fn, fn.bfloat16().float())
        vals = (torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16) * .1,
            torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16) * .1,
            torch.rand(tokens, 4, device="cuda"), torch.rand(tokens, 16, device="cuda"),
            fn, scale, base, torch.randn(4096, device="cuda", dtype=torch.bfloat16))
        if seed == 0:
            vals[0].zero_(); vals[1].zero_()
        bf = fn.bfloat16()
        paired = bf.reshape(12, 2, 4, 4096).permute(0, 2, 3, 1).contiguous().view(torch.int32)
        # All conversions happen before graph capture and outside GPU timing.
        return vals, (fn, bf, paired)
    def call(vals, packs, mode):
        return mk._mhc_call(*vals[:4], packs[mode], *vals[5:], vals[0].shape[0],
                            1e-6, 1e-6, 1e-6, 1., 1e-6, 20)
    for wi, weight in enumerate(weights):
        for tokens in (1, 2, 6, 8, 12, 32):
            for seed in (53, 0):
                vals, packs = inputs(tokens, seed, weight)
                ext.set_mhc_pack(0)
                ref = tuple(v.clone() for v in call(vals, packs, 0))
                oracle = mathematical_reference(vals, torch)
                errors = [mk._rel_err(a, b) for a, b in zip(ref, oracle)]
                assert max(errors) <= mk._TOL_MHC, (weight[0], tokens, seed, errors)
                for mode in (1, 2):
                    ext.set_mhc_pack(mode)
                    got = call(vals, packs, mode)
                    assert all(torch.equal(a, b) for a, b in zip(ref, got)), (weight[0], tokens, seed, mode, "bits changed")
                    assert all(torch.isfinite(v).all() for v in got)
                print(json.dumps({"gate": "all four outputs bit equal", "weight": weight[0],
                    "T": tokens, "seed": seed, "fp64_errors": errors}), flush=True)
    for tokens in (2, 6, 8):
        vals, packs = inputs(tokens, 71, weights[0])
        for cold in (False, True):
            calls = 1 if cold else 32
            graphs = []
            for mode in (0, 1, 2):
                ext.set_mhc_pack(mode)
                for _ in range(4): call(vals, packs, mode)
                if cold: _l2_flush(vals[:4])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                begin, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                with torch.cuda.graph(graph):
                    if cold: _l2_flush(vals[:4])
                    begin.record()
                    for _ in range(calls): got = call(vals, packs, mode)
                    end.record()
                graph.replay(); end.synchronize()
                graphs.append((graph, begin, end, got, tuple(v.clone() for v in got)))
            times = [[], [], []]
            orders = list(itertools.permutations(range(3)))
            for rep in range(args.reps):
                for mode in orders[rep % 6]:
                    graph, begin, end, got, snapshot = graphs[mode]
                    graph.replay(); end.synchronize()
                    times[mode].append(begin.elapsed_time(end) * 1000 / calls)
                    assert all(torch.equal(a, b) for a, b in zip(got, snapshot)), (tokens, mode, "graph drift")
            med = [statistics.median(t) for t in times]
            print(json.dumps({"T": tokens, "regime": "cold" if cold else "warm",
                "modes": ["baseline", "bf16", "bf16pair"], "median_us": med,
                "samples_us": times, "changes_pct": [100*(v/med[0]-1) for v in med]}), flush=True)
    print("PASS: checkpoint fixtures, FP64 oracle, all four outputs bit equal and repeated CUDA graphs", flush=True)


if __name__ == "__main__":
    main()
