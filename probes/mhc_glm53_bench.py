#!/usr/bin/env python3
"""mhc_fused_post_pre small-M launch-config sweep for GLM-5.3 (GB10, sm_121a).

Attacks the exact heuristic the image's dispatcher marks
"TODO(gnovack): investigate autotuning" (tilelang.py, use_small_fma branch):
stock runs tile_n=2, n_splits=8 at num_tokens<8 -- the C=1 decode regime
(45 layers x pair per step, 185 kernels/step per the #108 census). The dsv4
lane swept the same TODO and adopted (6,4) at M<8 for +16 pct kernel-time
(dsv4_mhc_tilelang R1); this harness answers whether GLM's shapes win too.

Runs the REAL kernel pair the dispatcher runs per layer in decode:
  1. mhc_fused_tilelang          (post-map + prenorm GEMM, FMA path)
  2. mhc_pre_big_fuse_with_norm  (mix/sinkhorn + normed layer_input)
with the dispatcher's allocation and argument order, CUDA-graph captured
(kernels are 10-100 us; event timing without a graph is launch-noise).

Run in a FRESH GPU container (never docker-exec CUDA in the serving one):
  cd /home/choiceoh/stkernel
  probes/run_mhc_glm53_bench.sh [--quick|--onepass|--prefill|
                                 --passes|--hcweight]

The wrapper composes GLM53, bind-mounts both MHC sources over their exact vLLM
targets, and gives this process the composed files for SHA-256 verification.
Direct docker invocation is refused so a stock-image import cannot pass as an
overlay verdict.

Verdicts are kernel-time only -- adoption still needs the bracket boot
(VLLM_GLM53_MHC_SMALLM="tile_n,n_splits", quality 9/9 + Korean 0/16 + C=1).
"""
import argparse
import hashlib
import importlib
import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

mhc_fused_tilelang = None
mhc_pre_big_fuse_with_norm_tilelang = None
_dispatcher_mod = None

# GLM-5.3-Flash: config.json + glm5next_model.py call sites. hc_post_mult only
# scales the post mix -- it moves no timing; the numerics reference uses it
# exactly as the model would.
HC, HIDDEN, N_OUT = 4, 4096, 24  # n_out = hc_mult*(hc_mult+2) = hc_mult3
RMS_EPS, HC_EPS, SINKHORN, POST_MULT, NORM_EPS = 1e-5, 1e-6, 20, 2.0, 1e-5
STOCK_LT8 = (2, 8)  # the TODO heuristic's num_tokens<8 arm
STOCK_GE8 = (3, 4)  # the TODO heuristic's 8<=num_tokens<=16 arm


def _stock_config(num_tokens):
    return STOCK_LT8 if num_tokens < 8 else STOCK_GE8


def _parse_ms(raw):
    parts = raw.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError("--ms must be a non-empty comma-separated list")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--ms values must be integers") from exc
    if any(value < 1 or value > 16 for value in values):
        raise ValueError("--ms values must stay in the small-FMA range 1..16")
    return values


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_composed_source(module, source):
    """Fail unless the imported module is the exact composed repo source."""
    build = os.environ.get("STKERNEL_MHC_OVERLAY_BUILD", "").strip()
    if not build:
        raise RuntimeError(
            "STKERNEL_MHC_OVERLAY_BUILD is unset; run "
            "probes/run_mhc_glm53_bench.sh so the composed MHC sources are "
            "mounted into vLLM"
        )
    expected = os.path.join(build, source)
    actual = getattr(module, "__file__", "")
    if not os.path.isfile(expected) or not actual or not os.path.isfile(actual):
        raise RuntimeError(
            f"cannot verify imported {source}: expected={expected!r} "
            f"actual={actual!r}"
        )
    expected_hash = _sha256(expected)
    actual_hash = _sha256(actual)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"imported {source} is not the composed overlay: "
            f"{actual} sha256={actual_hash}, expected {expected_hash}"
        )
    print(f"source {source}: {actual} sha256={actual_hash}", flush=True)


def _load_mhc_overlay(require_onepass=False):
    """Import vLLM only after mode env is set, then prove source identity."""
    global mhc_fused_tilelang, mhc_pre_big_fuse_with_norm_tilelang
    global _dispatcher_mod

    dispatcher = importlib.import_module(
        "vllm.model_executor.kernels.mhc.tilelang"
    )
    kernels = importlib.import_module(
        "vllm.model_executor.kernels.mhc.tilelang_kernels"
    )
    _dispatcher_mod = dispatcher
    _require_composed_source(dispatcher, "tilelang.py")
    _require_composed_source(kernels, "tilelang_kernels.py")
    if require_onepass:
        if getattr(dispatcher, "_DENEB_ONEPASS", None) is not True:
            raise RuntimeError(
                "ONEPASS was not frozen ON before the dispatcher import"
            )
        if not hasattr(kernels, "mhc_onepass_tilelang"):
            raise RuntimeError("the imported kernels lack mhc_onepass_tilelang")
    mhc_fused_tilelang = kernels.mhc_fused_tilelang
    mhc_pre_big_fuse_with_norm_tilelang = (
        kernels.mhc_pre_big_fuse_with_norm_tilelang
    )


def make_inputs(m, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    comb = torch.randn(m, HC, HC, generator=g, device="cuda", dtype=torch.float32)
    residual = torch.randn(
        m, HC, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16
    )
    post = torch.randn(m, HC, generator=g, device="cuda", dtype=torch.float32)
    x = torch.randn(m, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(N_OUT, HC, HIDDEN, generator=g, device="cuda",
                     dtype=torch.float32) * 0.02
    norm_w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.0, 1.0, 1.0], device="cuda", dtype=torch.float32)
    base = torch.zeros(N_OUT, device="cuda", dtype=torch.float32)
    return comb, residual, post, x, fn, norm_w, scale, base


def run_pair(tensors, tile_n, n_splits, outs=None):
    """The dispatcher's use_small_fma path, argument for argument."""
    comb, residual, post, x, fn, norm_w, scale, base = tensors
    if outs is None:
        outs = alloc_outs(len(residual), n_splits)
    gemm_mul, gemm_sqr, res_cur, post_c, comb_c, layer_c = outs
    mhc_fused_tilelang(
        comb, residual, post, x, fn,
        gemm_mul, gemm_sqr, res_cur,
        HC, HIDDEN, N_OUT,
        tile_n=tile_n, n_splits=n_splits,
    )
    mhc_pre_big_fuse_with_norm_tilelang(
        gemm_mul, gemm_sqr, scale, base, res_cur,
        post_c, comb_c, layer_c, norm_w,
        HIDDEN, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, NORM_EPS,
        n_splits=n_splits, hc_mult=HC,
    )
    return outs


def alloc_outs(m, n_splits):
    return (
        torch.empty(n_splits, m, N_OUT, device="cuda", dtype=torch.float32),
        torch.empty(n_splits, m, device="cuda", dtype=torch.float32),
        torch.empty(m, HC, HIDDEN, device="cuda", dtype=torch.bfloat16),
        torch.empty(m, HC, device="cuda", dtype=torch.float32),
        torch.empty(m, HC * HC, device="cuda", dtype=torch.float32),
        torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16),
    )


def bench_us(fn, iters=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def rel_err(a, b):
    d = (a.float() - b.float()).norm()
    den = b.float().norm()
    return float(d / den) if den > 0 else float("inf")


def _onepass_rel_errors(out, ref):
    """Compare op outputs after matching the stock harness' flat buffers."""
    aligned = (
        out[0],
        out[1].reshape_as(ref[3]),
        out[2].reshape_as(ref[4]),
        out[3],
    )
    return [rel_err(actual, expected)
            for actual, expected in zip(aligned, ref[2:])]


def _pair_rel_errors(out, ref):
    """Compare only the pair's config-independent final outputs."""
    return [rel_err(out[index], ref[index]) for index in (2, 3, 4, 5)]


def onepass_check():
    """Fused single-launch (VLLM_GLM53_MHC_ONEPASS) vs the stock pair.

    Must run with --onepass ALONE: the gate is frozen at the dispatcher's
    import, so this mode arms it before importing the op module and compares
    torch.ops.vllm.mhc_fused_post_pre_tilelang (which then routes to
    mhc_onepass_tilelang) against the direct stock kernel pair."""
    print("onepass vs stock pair (gate frozen ON for the op path):")
    for m in (1, 2, 4, 8, 16):
        tensors = make_inputs(m)
        comb, residual, post, x, fn, norm_w, scale, base = tensors
        stock_config = _stock_config(m)
        ref = run_pair(tensors, *stock_config)
        torch.cuda.synchronize()
        out = torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x, residual, post, comb, fn.view(N_OUT, HC, HIDDEN),
            scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
            1, 1, norm_w, NORM_EPS,
        )
        torch.cuda.synchronize()
        errs = _onepass_rel_errors(out, ref)
        t_stock = bench_us(lambda: run_pair(tensors, *stock_config))
        t_one = bench_us(lambda: torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x, residual, post, comb, fn.view(N_OUT, HC, HIDDEN),
            scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
            1, 1, norm_w, NORM_EPS))
        flag = "" if max(errs) <= 1e-4 else "  ! MISMATCH"
        print(f"M={m:<3d} stock{stock_config}={t_stock:7.1f}us "
              f"onepass={t_one:7.1f}us "
              f"({100 * (t_stock - t_one) / t_stock:+5.1f}%) "
              f"rel_err(max)={max(errs):.2e}{flag}", flush=True)
    print("adopt only if rel_err stays <=1e-4 and the bracket confirms "
          "end-to-end (C=1 step/s + quality 9/9 + Korean 0/16).")


def prefill_check():
    """h_blk sweep for the prefill big_fuse (dsv4 R3 precedent: h_blk=4096
    single pipelined block won +5.6% at M=4096 on this kernel family;
    GLM candidate value feeds VLLM_GLM53_MHC_BIGFUSE)."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.math_utils import cdiv

    print("prefill big_fuse_with_norm h_blk sweep "
          "(n_splits from the stock compute_num_split):")
    for m in (512, 2048, 8192):
        ns_split = compute_num_split(64, HC * HIDDEN, cdiv(m, 64))
        g = torch.Generator(device="cuda").manual_seed(0)
        gemm_mul = torch.randn(ns_split, m, N_OUT, generator=g,
                               device="cuda", dtype=torch.float32) * 0.1
        gemm_sqr = torch.rand(ns_split, m, generator=g,
                              device="cuda", dtype=torch.float32) + 16.0
        residual = torch.randn(m, HC, HIDDEN, generator=g,
                               device="cuda", dtype=torch.bfloat16)
        norm_w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([1.0, 1.0, 1.0], device="cuda",
                             dtype=torch.float32)
        base = torch.zeros(N_OUT, device="cuda", dtype=torch.float32)
        ref = None
        for h_blk in (1024, 2048, 4096):
            post = torch.empty(m, HC, device="cuda", dtype=torch.float32)
            comb = torch.empty(m, HC * HC, device="cuda", dtype=torch.float32)
            layer = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
            def call():
                mhc_pre_big_fuse_with_norm_tilelang(
                    gemm_mul, gemm_sqr, scale, base, residual,
                    post, comb, layer, norm_w,
                    HIDDEN, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                    NORM_EPS, ns_split, HC, -1, h_blk,
                )
            call()
            torch.cuda.synchronize()
            err = rel_err(layer, ref[2]) if ref is not None else 0.0
            us = bench_us(call)
            print(f"M={m:<5d} h_blk={h_blk:<5d} {us:9.1f}us"
                  f"{'' if ref is None else f'  rel_err(layer)={err:.2e}'}",
                  flush=True)
            if ref is None:
                ref = (post, comb, layer)
            del post, comb, layer
        del gemm_mul, gemm_sqr, residual, ref


def passes_check(ref_path=None, load_path=None):
    """Time the stock pair (and onepass, when its gate froze ON) under the
    VLLM_GLM53_MHC_PASSES combo the wrapper set for THIS process, and check
    the pair's final outputs against a stock-passes reference.

    The pass knob compiles into every mhc kernel at import, so the A/B is
    ACROSS processes: run_mhc_glm53_bench.sh --passes loops the four combos,
    the stock combo saving the reference (--ref-save) and the others comparing
    against it (--ref-load). Within a process the harness (graph-captured,
    real dispatcher argument order) is identical, so cross-process timings are
    comparable. Gate: rel_err <= 1e-4 vs the stock-passes reference; a pass
    rewrite that cannot hold sum order that tight is not a candidate."""
    import tilelang
    from vllm.model_executor.kernels.mhc import tilelang_kernels as kernels

    frozen = getattr(kernels, "_DENEB_MHC_PASSES", None)
    if frozen is None:
        raise RuntimeError(
            "VLLM_GLM53_MHC_PASSES was not set at import; the wrapper must "
            "freeze a combo per process (stock combo saves the reference)"
        )
    print(f"frozen pass combo: tma={frozen[0]} ws={frozen[1]}")
    for key in (tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER,
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED):
        print(f"  {key.name}={kernels.pass_configs[key]}")

    onepass_on = getattr(_dispatcher_mod, "_DENEB_ONEPASS", False) is True
    saved = None
    if load_path:
        saved = torch.load(load_path, map_location="cuda")
        print(f"reference loaded: {load_path}")
    to_save = {} if ref_path else None

    for m in (1, 2, 4, 8, 16):
        tensors = make_inputs(m)
        comb, residual, post, x, fn, norm_w, scale, base = tensors
        stock_config = _stock_config(m)
        outs = run_pair(tensors, *stock_config)
        torch.cuda.synchronize()
        t_pair = bench_us(lambda: run_pair(tensors, *stock_config))
        line = f"M={m:<3d} pair{stock_config}={t_pair:7.1f}us"
        if onepass_on:
            op = lambda: torch.ops.vllm.mhc_fused_post_pre_tilelang(  # noqa: E731
                x, residual, post, comb, fn.view(N_OUT, HC, HIDDEN),
                scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
                1, 1, norm_w, NORM_EPS)
            op()
            torch.cuda.synchronize()
            line += f" onepass={bench_us(op):7.1f}us"
        if saved is not None:
            errs = [rel_err(outs[i], saved[m][j])
                    for j, i in enumerate((2, 3, 4, 5))]
            line += (f"  rel_err(max vs stock-passes ref)={max(errs):.2e}"
                     + ("" if max(errs) <= 1e-4 else "  ! MISMATCH"))
        if to_save is not None:
            to_save[m] = [outs[i].detach().clone() for i in (2, 3, 4, 5)]
        print(line, flush=True)
        del tensors, outs
    if ref_path:
        torch.save(to_save, ref_path)
        print(f"reference saved: {ref_path}")


def _alloc_onepass_outs(m):
    return (
        torch.empty(m, HC, HIDDEN, device="cuda", dtype=torch.bfloat16),
        torch.empty(m, HC, device="cuda", dtype=torch.float32),
        torch.empty(m, HC * HC, device="cuda", dtype=torch.float32),
        torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16),
    )


def _call_onepass(kern, tensors, fn3d, outs=None):
    comb, residual, post, x, fn, norm_w, scale, base = tensors
    if outs is None:
        outs = _alloc_onepass_outs(len(x))
    kern(comb, residual, post, x, fn3d, scale, base, norm_w, *outs,
         HC, HIDDEN, N_OUT, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
         NORM_EPS)
    return outs


def hcweight_check():
    """bf16 hc-weight onepass probe (numerics experiment, NOT a <=1e-4 axis).

    The onepass kernel's dominant bytes are weight_t [n_out,hc,h] read as
    fp32 (~2MB/call at GLM shapes); bf16 halves that. "hc weights are fp32 by
    design" (P1) makes this a precision trade, so the probe separates the two
    error sources:
      control  = stock onepass fed the SAME rounded weights (bf16->fp32)
      variant  = bf16-weight kernel (transcription below, 2-line delta)
    Gate: variant vs control <= 1e-4 (that is only transcription fidelity);
    variant vs stock-fp32 is REPORTED, not gated -- it is the quantization
    cost the bracket's quality gates (9/9 + Korean 0/16) would have to
    absorb."""
    import math

    import tilelang
    import tilelang.language as T
    from vllm.model_executor.kernels.mhc import tilelang_kernels as kernels

    pass_configs = kernels.pass_configs
    ENABLE_PDL = kernels.ENABLE_PDL

    # Transcription of mhc_onepass_tilelang from the composed kernels module.
    # The ONLY deltas: weight_t is declared bfloat16, and the phase-1 FMA
    # casts it to fp32 before use (so all arithmetic stays fp32 exactly as
    # stock). Everything else is line-for-line the overlay kernel.
    @tilelang.jit(pass_configs=pass_configs)
    def mhc_onepass_bf16w(
        comb_mix, residual_in, post_mix, x_in, weight_t, hc_scale, hc_base,
        norm_weight, residual_out, post_mix_out, comb_mix_out, layer_input,
        hc: int, hidden: int, n_out: int,
        rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float,
        hc_post_mult_value: float, sinkhorn_repeat: int, norm_eps: float,
        n_thr: int = 256,
    ) -> tilelang.JITKernel:
        m = T.dynamic("num_tokens")
        h = hidden
        h_blk = math.gcd(1024, h)

        comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
        residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
        post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
        x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
        # DELTA 1/2: fp32 -> bfloat16 weight
        weight_t: T.Tensor((n_out, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
        hc_scale: T.Tensor((3,), T.float32)  # type: ignore[no-redef, valid-type]
        hc_base: T.Tensor((n_out,), T.float32)  # type: ignore[no-redef, valid-type]
        norm_weight: T.Tensor((h,), T.bfloat16)  # type: ignore[no-redef, valid-type]
        residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
        post_mix_out: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
        comb_mix_out: T.Tensor((m, hc * hc), T.float32)  # type: ignore[no-redef, valid-type]
        layer_input: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]

        h_iters = h // n_thr
        num_warps = n_thr // 32

        with T.Kernel(m, threads=n_thr) as i_n:
            tid = T.get_thread_binding()
            warp_id = tid // 32
            lane = tid % 32

            s_post = T.alloc_shared((hc,), T.float32)
            s_comb = T.alloc_shared((hc, hc), T.float32)
            pm = T.alloc_local((hc,), T.float32)
            cm = T.alloc_local((hc, hc), T.float32)
            new_r = T.alloc_local((hc,), T.float32)
            acc = T.alloc_local((n_out,), T.float32)
            sqr = T.alloc_local((1,), T.float32)

            s_warp = T.alloc_shared((num_warps, n_out + 1), T.float32)

            if ENABLE_PDL:
                T.pdl_sync()

            T.copy(post_mix[i_n, 0], s_post)
            T.copy(comb_mix[i_n, 0, 0], s_comb)
            for j in T.unroll(hc):
                pm[j] = s_post[j]
            for j in T.unroll(hc):
                for k in T.unroll(hc):
                    cm[k, j] = s_comb[k, j]

            T.clear(acc)
            T.clear(sqr)
            for it in T.serial(h_iters):
                h_idx = it * n_thr + tid
                for j in T.unroll(hc):
                    new_r[j] = pm[j] * x_in[i_n, h_idx]
                    for k in T.unroll(hc):
                        new_r[j] += cm[k, j] * residual_in[i_n, k, h_idx]
                for j in T.unroll(hc):
                    residual_out[i_n, j, h_idx] = new_r[j]
                    sqr[0] += new_r[j] * new_r[j]
                for n in T.unroll(n_out):
                    for j in T.unroll(hc):
                        # DELTA 2/2: cast the bf16 weight to fp32 before the
                        # FMA -- arithmetic dtype unchanged vs stock
                        acc[n] += T.float32(weight_t[n, j, h_idx]) * new_r[j]

            for n in T.unroll(n_out):
                acc[n] = T.warp_reduce_sum(acc[n])
            sqr[0] = T.warp_reduce_sum(sqr[0])

            if lane == 0:
                for n in T.unroll(n_out):
                    s_warp[warp_id, n] = acc[n]
                s_warp[warp_id, n_out] = sqr[0]
            T.sync_threads()

            rp = T.alloc_fragment((1,), T.float32)
            rms = T.alloc_fragment((1,), T.float32)
            mixes = T.alloc_fragment(n_out, T.float32)
            rp[0] = 0
            rms[0] = 0
            for w in T.unroll(num_warps):
                rp[0] += s_warp[w, n_out]
            rms[0] = T.rsqrt(rp[0] / (hc * h) + rms_eps)
            for n in T.Parallel(n_out):
                mixes[n] = 0
                for w in T.unroll(num_warps):
                    mixes[n] += s_warp[w, n]
                mixes[n] *= rms[0]
            s_mixes = T.alloc_shared(n_out, T.float32)
            T.copy(mixes, s_mixes)

            if tid < 32:
                for j in T.Parallel(hc):
                    post_mix_out[i_n, j] = (
                        T.sigmoid(
                            s_mixes[j + hc] * hc_scale[1] + hc_base[j + hc]
                        )
                        * hc_post_mult_value
                    )
                cm_f = T.alloc_fragment((hc, hc), T.float32)
                for j, k in T.Parallel(hc, hc):
                    cm_f[j, k] = (
                        s_mixes[j * hc + k + hc * 2] * hc_scale[2]
                        + hc_base[j * hc + k + hc * 2]
                    )

                row_sum = T.alloc_fragment(hc, T.float32)
                col_sum = T.alloc_fragment(hc, T.float32)
                row_max = T.alloc_fragment(hc, T.float32)
                T.reduce_max(cm_f, row_max, dim=1)
                for j, k in T.Parallel(hc, hc):
                    cm_f[j, k] = T.exp(cm_f[j, k] - row_max[j])
                T.reduce_sum(cm_f, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    cm_f[j, k] = cm_f[j, k] / row_sum[j] + hc_sinkhorn_eps

                T.reduce_sum(cm_f, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    cm_f[j, k] = cm_f[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm_f, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        cm_f[j, k] = cm_f[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    T.reduce_sum(cm_f, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        cm_f[j, k] = cm_f[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for j, k in T.Parallel(hc, hc):
                    comb_mix_out[i_n, j * hc + k] = cm_f[j, k]

            pre_mix_shared = T.alloc_shared(hc, T.float32)
            for j in T.Parallel(hc):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        s_mixes[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            output_shared = T.alloc_shared(h, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(h_blk, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                xs = T.alloc_shared((hc, h_blk), T.bfloat16)
                xl = T.alloc_fragment((hc, h_blk), T.float32)
                T.copy(residual_out[i_n, 0, i0_h * h_blk], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(h_blk, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(h_blk):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                for i1_h in T.Parallel(h_blk):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * h_blk + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / h + norm_eps)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                w_shared = T.alloc_shared(h_blk, T.bfloat16)
                w_local = T.alloc_fragment(h_blk, T.float32)
                T.copy(norm_weight[i0_h * h_blk], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(h_blk, T.float32)
                for i1_h in T.Parallel(h_blk):
                    ol[i1_h] = (
                        output_shared[i0_h * h_blk + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i_n, i0_h * h_blk])

            if ENABLE_PDL:
                T.pdl_trigger()

    onepass = kernels.mhc_onepass_tilelang
    print("bf16 hc-weight onepass vs stock (control isolates transcription):")
    for m in (1, 2, 4, 8, 16):
        tensors = make_inputs(m)
        fn = tensors[4]
        fn_bf = fn.to(torch.bfloat16)
        fn_ctl = fn_bf.to(torch.float32)

        ref = _call_onepass(onepass, tensors, fn)
        ctl = _call_onepass(onepass, tensors, fn_ctl)
        var = _call_onepass(mhc_onepass_bf16w, tensors, fn_bf)
        torch.cuda.synchronize()
        e_trans = max(rel_err(var[i], ctl[i]) for i in range(4))
        e_quant = max(rel_err(var[i], ref[i]) for i in range(4))

        t_stock = bench_us(lambda: _call_onepass(onepass, tensors, fn, ref))
        t_bf = bench_us(lambda: _call_onepass(mhc_onepass_bf16w, tensors,
                                              fn_bf, var))
        flag = "" if e_trans <= 1e-4 else "  ! TRANSCRIPTION MISMATCH"
        print(f"M={m:<3d} stock={t_stock:7.1f}us bf16w={t_bf:7.1f}us "
              f"({100 * (t_stock - t_bf) / t_stock:+5.1f}%) "
              f"rel_err(trans)={e_trans:.2e} rel_err(quant)={e_quant:.2e}"
              f"{flag}", flush=True)
        del tensors, ref, ctl, var
    print("transcription gate <=1e-4 vs control; quant error is REPORTED -- "
          "adoption would need bind-time weight casting in the dispatcher "
          "plus the full quality gates (9/9 + Korean 0/16 + C=1 bracket).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="stock neighborhood only (fewer JIT compiles)")
    ap.add_argument("--ms", default="1,2,4,6,8,12,16",
                    help="num_tokens values (comma separated)")
    ap.add_argument("--onepass", action="store_true",
                    help="verify the fused single-launch path against the "
                         "stock pair, then time both (VLLM_GLM53_MHC_ONEPASS)")
    ap.add_argument("--passes", action="store_true",
                    help="time the pair (+onepass if its env is on) under "
                         "the VLLM_GLM53_MHC_PASSES combo frozen for THIS "
                         "process; the wrapper loops the four combos and "
                         "carries the numerics reference across them")
    ap.add_argument("--ref-save", default=None, metavar="PATH",
                    help="with --passes (stock combo): save the pair's final "
                         "outputs as the cross-combo numerics reference")
    ap.add_argument("--ref-load", default=None, metavar="PATH",
                    help="with --passes (non-stock combos): rel_err the pair "
                         "against the saved reference (gate <=1e-4)")
    ap.add_argument("--hcweight", action="store_true",
                    help="bf16 hc-weight onepass numerics experiment: time "
                         "and error vs a control that isolates transcription "
                         "from quantization (no adoption gate here)")
    ap.add_argument("--prefill", action="store_true",
                    help="time the prefill big_fuse_with_norm across h_blk "
                         "(the VLLM_GLM53_MHC_BIGFUSE candidate set)")
    args = ap.parse_args()
    if args.onepass:
        # vllm.model_executor.kernels.mhc.__init__ imports the dispatcher, so
        # this must happen before importing either MHC submodule.
        os.environ["VLLM_GLM53_MHC_ONEPASS"] = "1"
    if args.hcweight:
        _load_mhc_overlay()
        hcweight_check()
        return
    if args.passes:
        if not os.environ.get("VLLM_GLM53_MHC_PASSES", "").strip():
            ap.error("--passes needs VLLM_GLM53_MHC_PASSES in the env "
                     "(run via probes/run_mhc_glm53_bench.sh --passes)")
        _load_mhc_overlay()
        passes_check(args.ref_save, args.ref_load)
        return
    _load_mhc_overlay(require_onepass=args.onepass)
    if args.onepass:
        onepass_check()
        return
    if args.prefill:
        prefill_check()
        return
    try:
        ms = _parse_ms(args.ms)
    except ValueError as exc:
        ap.error(str(exc))

    tiles = [1, 2, 3, 4, 6, 8, 12] if not args.quick else [1, 2, 3, 4, 6]
    splits = [1, 2, 4, 8] if not args.quick else [4, 8]

    print(f"HC={HC} HIDDEN={HIDDEN} N_OUT={N_OUT} "
          f"stock(<8)={STOCK_LT8} stock(8..16)={STOCK_GE8}")
    print(f"{'config':>10} " + " ".join(f"M={m:<9d}" for m in ms))
    rows = {}
    ref = {}
    for tn in tiles:
        for ns in splits:
            key = f"({tn},{ns})"
            cells, times = [], []
            for m in ms:
                tensors = make_inputs(m)
                if m not in ref:
                    ref[m] = run_pair(tensors, *_stock_config(m))
                    torch.cuda.synchronize()
                outs = alloc_outs(m, ns)
                run_pair(tensors, tn, ns, outs=outs)
                torch.cuda.synchronize()
                err = max(_pair_rel_errors(outs, ref[m]))
                us = bench_us(lambda: run_pair(tensors, tn, ns, outs=outs))
                cells.append(f"{us:8.1f}{'' if err <= 1e-4 else ' !'}")
                times.append(us)
                del tensors, outs
            rows[key] = times
            print(f"{key:>10} " + " ".join(cells), flush=True)

    print("\nvs stock, per M (positive = faster than stock):")
    stock = [
        rows[f"({_stock_config(m)[0]},{_stock_config(m)[1]})"][i]
        for i, m in enumerate(ms)
    ]
    for key, times in sorted(rows.items()):
        if all(key == f"({_stock_config(m)[0]},{_stock_config(m)[1]})"
               for m in ms):
            continue
        deltas = [100 * (s - t) / s for t, s in zip(times, stock)]
        print(f"{key:>10} " + " ".join(f"{d:+8.1f}%" for d in deltas))

    lt8 = [i for i, m in enumerate(ms) if m < 8]
    if lt8:
        best = min(
            (k for k in rows if k != f"({STOCK_LT8[0]},{STOCK_LT8[1]})"),
            key=lambda k: sum(rows[k][i] for i in lt8),
        )
        print(f"\nsuggested VLLM_GLM53_MHC_SMALLM={best[1:-1]} "
              f"(best summed M<8 kernel-time; bracket boot decides adoption)")
    else:
        print("\nno M<8 rows; no SMALLM suggestion", flush=True)


if __name__ == "__main__":
    main()
