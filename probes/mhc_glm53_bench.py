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
  probes/run_mhc_glm53_bench.sh [--quick|--prefill|--passes]

The wrapper composes GLM53, bind-mounts both MHC sources over their exact vLLM
targets, and gives this process the composed files for SHA-256 verification.
Direct docker invocation is refused so a stock-image import cannot pass as an
overlay verdict.

Verdicts are kernel-time only -- adoption still needs the bracket boot
(quality 9/9 + Korean 0/16 + C=1). The decode-side knobs this bench once fed
(MHC_SMALLM's tile pair, MHC_ONEPASS) were sunset in 34차 §8; the stock-pair
sweep stays as a kernel-level reference, --passes and --prefill are the live
arms.
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


def _load_mhc_overlay():
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


def _pair_rel_errors(out, ref):
    """Compare only the pair's config-independent final outputs."""
    return [rel_err(out[index], ref[index]) for index in (2, 3, 4, 5)]


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
    """Time the stock pair under the VLLM_GLM53_MHC_PASSES combo the wrapper set for THIS process, and check
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="stock neighborhood only (fewer JIT compiles)")
    ap.add_argument("--ms", default="1,2,4,6,8,12,16",
                    help="num_tokens values (comma separated)")
    ap.add_argument("--passes", action="store_true",
                    help="time the pair under "
                         "the VLLM_GLM53_MHC_PASSES combo frozen for THIS "
                         "process; the wrapper loops the four combos and "
                         "carries the numerics reference across them")
    ap.add_argument("--ref-save", default=None, metavar="PATH",
                    help="with --passes (stock combo): save the pair's final "
                         "outputs as the cross-combo numerics reference")
    ap.add_argument("--ref-load", default=None, metavar="PATH",
                    help="with --passes (non-stock combos): rel_err the pair "
                         "against the saved reference (gate <=1e-4)")
    ap.add_argument("--prefill", action="store_true",
                    help="time the prefill big_fuse_with_norm across h_blk "
                         "(the VLLM_GLM53_MHC_BIGFUSE candidate set)")
    args = ap.parse_args()
    if args.passes:
        if not os.environ.get("VLLM_GLM53_MHC_PASSES", "").strip():
            ap.error("--passes needs VLLM_GLM53_MHC_PASSES in the env "
                     "(run via probes/run_mhc_glm53_bench.sh --passes)")
        _load_mhc_overlay()
        passes_check(args.ref_save, args.ref_load)
        return
    _load_mhc_overlay()
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

    # (the sweep once suggested a VLLM_GLM53_MHC_SMALLM pair; that knob was
    # sunset in 34차 §8 -- the megakernel's mk_mhc serves decode -- so this
    # table is a kernel-level reference of the stock pair only)


if __name__ == "__main__":
    main()
