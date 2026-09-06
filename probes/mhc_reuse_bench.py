#!/usr/bin/env python3
"""Same-source MHC weight reuse prototype, oracle and balanced graph timings.

Run through fleet.sh run --gpu --probe and run_mk_probe.sh. The experiment
injects only its own two kernels and selector into a private source copy;
the serving source/defaults are unchanged until this gate establishes a win.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics


def candidate_source(source, tensorcore=False):
    marker = "__device__ void mk_mhc_p2_token("
    start = source.index(marker)
    end = source.index("// p3 + p4 for ONE token", start)
    p2 = source[start:end].replace("c < NCHUNK", "c < CHUNKS")
    p2 = p2.replace("float* s_pmix) {", "float* s_pmix, float ready = 0.0f) {")
    p2 = p2.replace("  float mine = 0.0f;\n  if (lane < NOUT)",
                    "  float mine = ready;\n  if constexpr (!READY) {\n  if (lane < NOUT)")
    p2 = p2.replace("  MK_MHC_PROBE(1);", "  }\n  MK_MHC_PROBE(1);")
    source = source[:start] + "template <int CHUNKS = NCHUNK, bool READY = false>\n" + p2 + source[end:]
    candidate = Path(__file__).with_name("mhc_reuse_candidate.cuh").read_text()
    if tensorcore:
        candidate += "\n" + Path(__file__).with_name("mhc_tf32x3_candidate.cuh").read_text()
    source = source.replace("__device__ void mk_mhc_p1(", candidate + "\n__device__ void mk_mhc_p1(", 1)
    source = source.replace("void mk_run_mhc(", "int g_probe_mhc_reuse = 0;\nvoid mk_run_mhc(", 1)
    marker = "  static int mhc_grid = 0;"
    launch = []
    for mode, chunk in ((1, 64), (2, 128)):
        for tokens in (2, 6, 8):
            producer = (f"mk_mhc_tf32x3_p1<{tokens}, {chunk}><<<HIDDEN/{chunk}, MK_THREADS, 0, stream>>>(a);"
                        if tensorcore else f"mk_mhc_reuse_p1<2, {chunk}><<<(HIDDEN/{chunk})*({tokens}/2), MK_THREADS, 0, stream>>>(a);")
            launch.append(f"""
  if (g_probe_mhc_reuse == {mode} && a.num_tokens == {tokens}) {{
    {producer}
    MK_CHECK_CUDA(cudaGetLastError());
    mk_mhc_reuse_tail<HIDDEN/{chunk}><<<{tokens}, MK_THREADS, 0, stream>>>(a);
    MK_CHECK_CUDA(cudaGetLastError());
    return;
  }}
""")
    source = source.replace(marker, "".join(launch) + marker, 1)
    source = source.replace('  m.def("run_mhc",',
                            '  m.def("set_mhc_reuse", [](int v) { TORCH_CHECK(v >= 0 && v <= 2); g_probe_mhc_reuse = v; });\n  m.def("run_mhc",', 1)
    return source


def mathematical_reference(values, torch):
    """Independent FP64 formula, with the two required BF16 rounding points.

    This is an accuracy oracle, not a bitwise FP32 execution emulator.
    It intentionally shares neither CUDA partials nor either kernel's tree.
    """
    x, res, pm, cm, fn, scale, base, nw = (v.double() for v in values)
    tokens = x.shape[0]
    cm = cm.reshape(tokens, 4, 4)
    r = pm[:, :, None] * x[:, None, :]
    for k in range(4):
        r = r + cm[:, k, :, None] * res[:, k, None, :]
    y = r.reshape(tokens, -1) @ fn.T
    y = y * torch.rsqrt(r.square().mean((1, 2)) + 1e-6)[:, None]
    pre = torch.sigmoid(y[:, :4] * scale[0] + base[:4]) + 1e-6
    post = torch.sigmoid(y[:, 4:8] * scale[1] + base[4:8])
    comb = torch.softmax((y[:, 8:] * scale[2] + base[8:]).reshape(tokens, 4, 4), -1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    residual = r.to(torch.bfloat16)
    layer = (pre[:, :, None] * residual.double()).sum(1)
    norm = torch.rsqrt(layer.square().mean(-1) + 1e-6)
    layer = (layer.to(torch.bfloat16).double() * norm[:, None] * nw).to(torch.bfloat16)
    return residual, post.float(), comb.reshape(tokens, 16).float(), layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=18)
    ap.add_argument("--tensorcore", action="store_true", help="compensated TF32 projection; separate numeric gate")
    args = ap.parse_args()
    if args.reps < 6 or args.reps % 6:
        ap.error("reps must be a positive multiple of six for balanced orders")
    os.environ["VLLM_GLM53_MK_PDL"] = "1"
    import torch
    from vllm.model_executor.layers import glm53_megakernel as mk
    from torch.utils.cpp_extension import load
    from megakernel_glm53_bench import _l2_flush

    root = Path(__file__).resolve().parents[1]
    original = Path(mk._SRC).read_bytes()
    assert original == (root / "build/glm53/glm53_megakernel.cu").read_bytes()
    source = candidate_source(original.decode(), args.tensorcore)
    sha = hashlib.sha256(source.encode()).hexdigest()
    directory = Path(os.environ.get("VLLM_GLM53_MK_BUILD_DIR", "/tmp/mhc-reuse"))
    directory = directory / ("prototype-" + sha[:12])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prototype.cu"
    path.write_text(source)
    print(json.dumps({"source_sha256": hashlib.sha256(original).hexdigest(),
                      "prototype_sha256": sha, "device": torch.cuda.get_device_name(),
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "projection": "tf32x3" if args.tensorcore else "simt-fp32"}), flush=True)
    ext = load(name="mhc_reuse_" + sha[:12], sources=[str(path)],
               extra_cuda_cflags=["-O2", "-gencode", "arch=compute_121a,code=sm_121a"],
               build_directory=str(directory), verbose=False)
    mk._EXT = ext
    mk.NCHUNK = 64  # ample scratch for both prototype split geometries
    mk._WS = None
    assert mk._selftest_mhc(), "unmodified MHC failed the stock oracle"
    for mode in (1, 2):
        ext.set_mhc_reuse(mode)
        assert mk._selftest_mhc(), f"prototype {mode} failed the stock oracle"
        assert mk._selftest_mhc_pre(), f"prototype {mode} failed the pre-only oracle"

    def inputs(tokens, seed):
        torch.manual_seed(seed)
        return (torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16) * .1,
                torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16) * .1,
                torch.rand(tokens, 4, device="cuda"), torch.rand(tokens, 16, device="cuda"),
                torch.randn(24, 16384, device="cuda") * .02,
                torch.ones(3, device="cuda"), torch.zeros(24, device="cuda"),
                torch.randn(4096, device="cuda", dtype=torch.bfloat16))

    def call(values):
        return mk._mhc_call(*values, values[0].shape[0], 1e-6, 1e-6, 1e-6, 1., 1e-6, 20)

    # Different input seeds, zero residuals, served shapes and fallback sizes.
    for tokens in (1, 2, 6, 8, 12, 32):
        for seed in (53, 97, 0):
            values = inputs(tokens, seed)
            if seed == 0:
                values[0].zero_(); values[1].zero_()
            ext.set_mhc_reuse(0)
            ref = tuple(v.clone() for v in call(values))
            oracle = mathematical_reference(values, torch)
            baseline_errors = [mk._rel_err(a, b) for a, b in zip(ref, oracle)]
            assert max(baseline_errors) <= mk._TOL_MHC, (tokens, seed, "baseline vs FP64", baseline_errors)
            for mode in (1, 2):
                ext.set_mhc_reuse(mode)
                got = call(values)
                torch.cuda.synchronize()
                errors = [mk._rel_err(a, b) for a, b in zip(got, ref)]
                assert torch.equal(got[0], ref[0]), (tokens, seed, mode, "residual rounding")
                assert max(errors) <= mk._TOL_MHC, (tokens, seed, mode, errors)
                oracle_errors = [mk._rel_err(a, b) for a, b in zip(got, oracle)]
                assert max(oracle_errors) <= mk._TOL_MHC, (tokens, seed, mode, "FP64", oracle_errors)
                assert all(torch.isfinite(v).all() for v in got)
                print(json.dumps({"gate": "baseline differential", "T": tokens,
                                  "seed": seed, "mode": mode, "errors": errors,
                                  "fp64_errors": oracle_errors,
                                  "baseline_fp64_errors": baseline_errors}), flush=True)

    orders = list(itertools.permutations(range(3)))
    for tokens in (2, 6, 8):
        values = inputs(tokens, 71)
        for cold in (False, True):
            calls = 1 if cold else 32
            graphs = []
            snapshots = []
            for mode in (0, 1, 2):
                ext.set_mhc_reuse(mode)
                for _ in range(4): call(values)
                if cold: _l2_flush(values[:4])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                begin, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                with torch.cuda.graph(graph):
                    if cold: _l2_flush(values[:4])
                    begin.record()
                    for _ in range(calls): got = call(values)
                    end.record()
                graph.replay(); end.synchronize()
                snapshots.append(tuple(v.clone() for v in got))
                graphs.append((graph, begin, end, got))
            times = [[], [], []]
            for rep in range(args.reps):
                for mode in orders[rep % 6]:
                    graph, begin, end, got = graphs[mode]
                    graph.replay(); end.synchronize()
                    times[mode].append(begin.elapsed_time(end) * 1000 / calls)
                    assert all(torch.equal(a, b) for a, b in zip(got, snapshots[mode])), (tokens, mode, "graph drift")
            med = [statistics.median(t) for t in times]
            print(json.dumps({"T": tokens, "regime": "cold" if cold else "warm",
                              "modes": ["baseline", "reuse64", "reuse128"],
                              "median_us": med, "samples_us": times,
                              "changes_pct": [100 * (v / med[0] - 1) for v in med]}), flush=True)
    print("PASS: stock and baseline oracles, exact residuals and repeated CUDA graphs", flush=True)


if __name__ == "__main__":
    main()
