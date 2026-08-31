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
  docker run --rm --gpus all -v /home/choiceoh/overlays:/mnt:ro \
    glm53:v13-b12x python3 /mnt/probes/mhc_glm53_bench.py [--quick]

Verdicts are kernel-time only -- adoption still needs the bracket boot
(VLLM_GLM53_MHC_SMALLM="tile_n,n_splits", quality 9/9 + Korean 0/16 + C=1).
"""
import argparse
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.model_executor.kernels.mhc.tilelang_kernels import (  # noqa: E402
    mhc_fused_tilelang,
    mhc_pre_big_fuse_with_norm_tilelang,
)

# GLM-5.3-Flash: config.json + glm5next_model.py call sites. hc_post_mult only
# scales the post mix -- it moves no timing; the numerics reference uses it
# exactly as the model would.
HC, HIDDEN, N_OUT = 4, 4096, 24  # n_out = hc_mult*(hc_mult+2) = hc_mult3
RMS_EPS, HC_EPS, SINKHORN, POST_MULT, NORM_EPS = 1e-5, 1e-6, 20, 2.0, 1e-5
STOCK_LT8 = (2, 8)  # the TODO heuristic's num_tokens<8 arm


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


def onepass_check():
    """Fused single-launch (VLLM_GLM53_MHC_ONEPASS) vs the stock pair.

    Must run with --onepass ALONE: the gate is frozen at the dispatcher's
    import, so this mode arms it before importing the op module and compares
    torch.ops.vllm.mhc_fused_post_pre_tilelang (which then routes to
    mhc_onepass_tilelang) against the direct stock kernel pair."""
    import os
    os.environ["VLLM_GLM53_MHC_ONEPASS"] = "1"
    import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401 registers ops

    print("onepass vs stock pair (gate frozen ON for the op path):")
    for m in (1, 2, 4, 8, 16):
        tensors = make_inputs(m)
        comb, residual, post, x, fn, norm_w, scale, base = tensors
        ref = run_pair(tensors, *STOCK_LT8)
        torch.cuda.synchronize()
        out = torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x, residual, post, comb, fn.view(N_OUT, HC, HIDDEN),
            scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
            1, 1, norm_w, NORM_EPS,
        )
        torch.cuda.synchronize()
        errs = [rel_err(o, r) for o, r in zip(
            out, (ref[2], ref[3], ref[4], ref[5]))]
        t_stock = bench_us(lambda: run_pair(tensors, *STOCK_LT8))
        t_one = bench_us(lambda: torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x, residual, post, comb, fn.view(N_OUT, HC, HIDDEN),
            scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
            1, 1, norm_w, NORM_EPS))
        flag = "" if max(errs) <= 1e-4 else "  ! MISMATCH"
        print(f"M={m:<3d} stock={t_stock:7.1f}us onepass={t_one:7.1f}us "
              f"({100 * (t_stock - t_one) / t_stock:+5.1f}%) "
              f"rel_err(max)={max(errs):.2e}{flag}", flush=True)
    print("adopt only if rel_err stays <=1e-4 and the bracket confirms "
          "end-to-end (C=1 step/s + quality 9/9 + Korean 0/16).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="stock neighborhood only (fewer JIT compiles)")
    ap.add_argument("--ms", default="1,2,4,6,8,12,16",
                    help="num_tokens values (comma separated)")
    ap.add_argument("--onepass", action="store_true",
                    help="verify the fused single-launch path against the "
                         "stock pair, then time both (VLLM_GLM53_MHC_ONEPASS)")
    args = ap.parse_args()
    if args.onepass:
        onepass_check()
        return
    ms = [int(v) for v in args.ms.split(",")]

    tiles = [1, 2, 3, 4, 6, 8, 12] if not args.quick else [1, 2, 3, 4, 6]
    splits = [1, 2, 4, 8] if not args.quick else [4, 8]

    print(f"HC={HC} HIDDEN={HIDDEN} N_OUT={N_OUT} stock(<8)={STOCK_LT8}")
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
                    ref[m] = run_pair(tensors, *STOCK_LT8)
                    torch.cuda.synchronize()
                outs = alloc_outs(m, ns)
                run_pair(tensors, tn, ns, outs=outs)
                torch.cuda.synchronize()
                err = max(
                    rel_err(outs[0], ref[m][0]),
                    rel_err(outs[1], ref[m][1]),
                    rel_err(outs[2], ref[m][2]),
                    rel_err(outs[5], ref[m][5]),
                )
                us = bench_us(lambda: run_pair(tensors, tn, ns, outs=outs))
                cells.append(f"{us:8.1f}{'' if err <= 1e-4 else ' !'}")
                times.append(us)
                del tensors, outs
            rows[key] = times
            print(f"{key:>10} " + " ".join(cells), flush=True)

    print("\nvs stock, per M (positive = faster than stock):")
    stock = rows[f"({STOCK_LT8[0]},{STOCK_LT8[1]})"]
    for key, times in sorted(rows.items()):
        if key == f"({STOCK_LT8[0]},{STOCK_LT8[1]})":
            continue
        deltas = [100 * (s - t) / s for t, s in zip(times, stock)]
        print(f"{key:>10} " + " ".join(f"{d:+8.1f}%" for d in deltas))

    lt8 = [i for i, m in enumerate(ms) if m < 8]
    best = min(
        (k for k in rows if k != f"({STOCK_LT8[0]},{STOCK_LT8[1]})"),
        key=lambda k: sum(rows[k][i] for i in lt8),
    )
    print(f"\nsuggested VLLM_GLM53_MHC_SMALLM={best[1:-1]} "
          f"(best summed M<8 kernel-time; bracket boot decides adoption)")


if __name__ == "__main__":
    main()
