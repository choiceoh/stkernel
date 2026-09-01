#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53_megakernel numerics + timing probe (fresh container, never serving).

Gate ladder step 2 of overlay/modules/glm53_megakernel/README.md: this is
the pre-boot proof. It diffs each persistent segment against the stock path
it replaces on production shapes and times both arms with CUDA events.

    /repo/probes/megakernel_glm53_bench.py [--iters 30] [--skip-kda]

or, on srv4, via the wrapper that bind-mounts the composed overlay at its
real image paths first:

    bash /repo/probes/run_megakernel_bench.sh [--iters 30]

Conventions follow the repo's probe rules:
  * rel err > gate marks the cell with `!` and fails the run
  * KDA compares outputs AND the rolled conv/recurrent states (the
    state-index contract is the open item; this probe is where it closes)
  * timing is CUDA-event, 10 warmup + N reps, medians

Run inside a container that has the composed overlay mounted (srv4 scratch
container, NOT the serving one -- serving containers are CUDA-quiescent).
"""
from __future__ import annotations

import argparse
import sys

# arm every segment for the probe process only; the serving knobs are the
# profile's business
import os

os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
os.environ.setdefault("VLLM_GLM53_MK_MHC", "1")
os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
os.environ.setdefault("VLLM_GLM53_MK_KDA", "1")
os.environ.setdefault("VLLM_GLM53_MK_W4", "1")

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import torch  # noqa: E402

TOL = {"mhc": 1e-3, "gemm": 2e-2, "kda": 2e-2}
DEV = "cuda"


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).norm()
    den = b.float().norm()
    return float(d / den) if den > 0 else float(d)


# GB10 has 24 MB of L2, and the weights these shapes read are 8-26 MB. A
# timing loop that re-reads the same weight therefore measures an L2-RESIDENT
# GEMM, not the DRAM-bound one a serving step does -- and which arm gets that
# gift depends on allocation order, so the same comparison swung 3-5x between
# process runs (stock m=16 n=4096 measured 45 us in isolation and 148 us with
# a 48 MB buffer merely allocated beforehand). Flush L2 between iterations so
# the number is the one production sees and is reproducible.
_L2_BYTES = 24 << 20
_L2_FLUSH = None


def _l2_flush() -> None:
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(2 * _L2_BYTES, dtype=torch.int8, device=DEV)
    _L2_FLUSH.zero_()


def _time(fn, iters: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        _l2_flush()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)  # us
    times.sort()
    return times[len(times) // 2]


def probe_gemm(iters: int) -> bool:
    from vllm.model_executor.layers.glm53_fp8_dense import _fp8_dense_gemm
    from vllm.model_executor.layers import glm53_megakernel as mk

    ok = True
    print(f"{'shape':<22}{'rel_err':>10}{'gate':>8}{'stock_us':>10}{'mk_us':>9}"
          f"{'mk_GBps':>9}{'st_GBps':>9}")
    for m, n in ((8, 6416), (16, 4096), (32, 2048), (32, 1024)):
        torch.manual_seed(0)
        w = torch.randn(n, 4096, dtype=torch.bfloat16, device=DEV) * 0.05
        x = torch.randn(m, 4096, dtype=torch.bfloat16, device=DEV)
        sq, sws, srows, scols = mk._stock_fp8_pair(w)
        mkq, mkws = mk.build_mk_weight(w)
        ref = _fp8_dense_gemm(x, sq, sws, srows, scols)
        got = mk._gemm_call(x, (mkq, mkws), n)
        torch.cuda.synchronize()
        r = _rel(got, ref)
        t_ref = _time(lambda: _fp8_dense_gemm(x, sq, sws, srows, scols), iters)
        t_mk = _time(lambda: mk._gemm_call(x, (mkq, mkws), n), iters)
        # replay stability: the timing loop just relaunched the same kernel
        # dozens of times over ONE workspace -- the monotonic-barrier
        # contract. The result must be unchanged.
        again = mk._gemm_call(x, (mkq, mkws), n)
        torch.cuda.synchronize()
        rep = _rel(got, again)
        mark = "!" if (r > TOL["gemm"] or rep > 1e-6) else " "
        ok &= r <= TOL["gemm"] and rep <= 1e-6
        # effective DRAM rate: the bytes each arm must move once. This is
        # the column that prices the cp.async pipeline against deepgemm
        # (GB10 peak ~223 GB/s/rank; the W8A8 dense step floor sits here).
        nbytes = sq.numel() + sws.numel() * 4 + x.numel() * 2 + m * n * 2
        print(f"{mark}gemm m={m:<4}n={n:<8}{r:>10.2e}{TOL['gemm']:>8.0e}"
              f"{t_ref:>10.1f}{t_mk:>9.1f}"
              f"{nbytes / t_mk / 1e3:>9.0f}{nbytes / t_ref / 1e3:>9.0f}")
    return ok


def probe_w4(iters: int) -> bool:
    """W4 arm: exact expansion gate + by-design error + timing. The exact
    gate is the load-bearing one -- it proves nibble/scale/expansion
    bit-for-bit, the by-design number just confirms e2m1's expected loss."""
    from vllm.model_executor.layers import glm53_megakernel as mk

    print(f"{'case':<24}{'rel_err':>10}{'gate':>8}{'w8_us':>8}{'w4_us':>8}"
          f"{'w4_GBps':>9}")
    ok = True
    torch.manual_seed(0)
    n, k, m = 1024, 4096, 8

    def run4(pack, out_n):
        out = torch.empty(m, out_n, dtype=torch.bfloat16, device=DEV)
        mk._EXT.run_gemm_w4(x.contiguous(), pack[2], pack[3], out, out_n)
        return out

    # exact fixture: weights ON the e2m1 grid
    code = torch.randint(0, 8, (n, k // 16, 16), device=DEV)
    sexp = torch.randint(-5, 7, (n, k // 16, 1), device=DEV)
    grid = torch.tensor(mk._E2M1_GRID, device=DEV)
    w_exact = (grid[code] * torch.exp2(sexp.float())) *         torch.where(torch.randn_like(code.float()) < 0, -1.0, 1.0)
    w_exact = w_exact.view(n, k).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
    p8, p4 = mk.build_mk_weight(w_exact), mk.build_mk_weight_w4(w_exact)
    o8 = torch.empty(m, n, dtype=torch.bfloat16, device=DEV)
    mk._EXT.run_gemm(x.contiguous(), p8[0], p8[1], o8, n)
    o4 = run4((None, None) + p4, n)
    torch.cuda.synchronize()
    e_exact = _rel(o4, o8)
    mark = "!" if e_exact > 1e-5 else " "
    ok &= e_exact <= 1e-5
    print(f"{mark}w4 exact grid{e_exact:>13.2e}{1e-5:>8.0e}{'-':>8}{'-':>8}"
          f"{'-':>9}")

    # by-design + timing on random weights
    w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.05
    p4, p8r = mk.build_mk_weight_w4(w), mk.build_mk_weight(w)
    sq, sws, srows, scols = mk._stock_fp8_pair(w)
    from vllm.model_executor.layers.glm53_fp8_dense import _fp8_dense_gemm

    ref = _fp8_dense_gemm(x, sq, sws, srows, scols)
    got = run4((None, None) + p4, n)
    torch.cuda.synchronize()
    e = _rel(got, ref)
    t8 = _time(lambda: mk._EXT.run_gemm(
        x.contiguous(), p8r[0], p8r[1],
        torch.empty(m, n, dtype=torch.bfloat16, device=DEV), n), iters)
    t4 = _time(lambda: run4((None, None) + p4, n), iters)
    nbytes4 = p4[0].numel() + p4[1].numel() + x.numel() * 2 + m * n * 2
    mark = "!" if e > 0.15 else " "
    ok &= e <= 0.15
    print(f"{mark}w4 by-design{'':<8}{e:>13.2e}{0.15:>8.2f}{t8:>8.1f}"
          f"{t4:>8.1f}{nbytes4 / t4 / 1e3:>9.0f}")
    return ok


def probe_mhc(iters: int) -> bool:
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tlk
    from vllm.model_executor.layers import glm53_megakernel as mk

    ok = True
    print(f"{'shape':<22}{'rel_err':>10}{'gate':>8}{'stock_us':>10}{'mk_us':>9}")

    def stock(x, res, pm, cm, fn, nw):
        T = res.shape[0]
        yp = torch.empty(4, T, 24, dtype=torch.float32, device=DEV)
        rp = torch.empty(4, T, dtype=torch.float32, device=DEV)
        res_ref = torch.empty_like(res)
        tlk.mhc_fused_tilelang(cm, res, pm, x, fn.view(24, 4, 4096), yp, rp,
                               res_ref, 4, 4096, 24, tile_n=2, n_splits=4)
        pm_ref = torch.empty(T, 4, dtype=torch.float32, device=DEV)
        cm_ref = torch.empty(T, 16, dtype=torch.float32, device=DEV)
        li_ref = torch.empty(T, 4096, dtype=torch.bfloat16, device=DEV)
        tlk.mhc_pre_big_fuse_with_norm_tilelang(
            yp, rp, mk.hc_scale_ones(), mk.hc_base_zeros(), res_ref, pm_ref,
            cm_ref, li_ref, nw, 4096, 1e-6, 1e-6, 1e-6, 1.0, 4, 1e-6,
            n_splits=4, hc_mult=4)
        return res_ref, pm_ref, cm_ref, li_ref

    for T in (8, 32):
        torch.manual_seed(0)
        x = torch.randn(T, 4096, dtype=torch.bfloat16, device=DEV) * 0.1
        res = torch.randn(T, 4, 4096, dtype=torch.bfloat16, device=DEV) * 0.1
        pm = torch.rand(T, 4, dtype=torch.float32, device=DEV)
        cm = torch.rand(T, 4, 4, dtype=torch.float32, device=DEV)
        fn = torch.randn(24, 16384, dtype=torch.float32, device=DEV) * 0.02
        nw = torch.randn(4096, dtype=torch.bfloat16, device=DEV)
        got = mk._mhc_call(x, res, pm, cm.reshape(T, 16).contiguous(), fn,
                           mk.hc_scale_ones(), mk.hc_base_zeros(), nw, T,
                           1e-6, 1e-6, 1e-6, 1.0, 1e-6, 4)
        ref = stock(x, res, pm, cm, fn, nw)
        torch.cuda.synchronize()
        r = max(_rel(g, rr) for g, rr in zip(got, ref))
        got0 = tuple(g.clone() for g in got)
        t_ref = _time(lambda: stock(x, res, pm, cm, fn, nw), iters)
        t_mk = _time(lambda: mk._mhc_call(
            x, res, pm, cm.reshape(T, 16).contiguous(), fn,
            mk.hc_scale_ones(), mk.hc_base_zeros(), nw, T, 1e-6, 1e-6, 1e-6,
            1.0, 1e-6, 4), iters)
        got2 = mk._mhc_call(x, res, pm, cm.reshape(T, 16).contiguous(), fn,
                            mk.hc_scale_ones(), mk.hc_base_zeros(), nw, T,
                            1e-6, 1e-6, 1e-6, 1.0, 1e-6, 4)
        torch.cuda.synchronize()
        rep = max(_rel(g, g0) for g, g0 in zip(got2, got0))
        mark = "!" if (r > TOL["mhc"] or rep > 1e-6) else " "
        ok &= r <= TOL["mhc"] and rep <= 1e-6
        print(f"{mark}mhc  T={T:<14}{r:>10.2e}{TOL['mhc']:>8.0e}"
              f"{t_ref:>10.1f}{t_mk:>9.1f}")
    return ok


def probe_kda(iters: int) -> bool:
    """Numerics for the state contract + timing of the whole block vs the
    stock chain, on the fixture the boot self-test itself uses (shared via
    the driver, so gate and probe cannot drift apart)."""
    from vllm.model_executor.layers import glm53_megakernel as mk

    ok = True
    print(f"{'case':<22}{'rel_err':>10}{'gate':>8}{'stock_us':>10}{'mk_us':>9}"
          "  details")
    for acc in (1, 3, 8):
        fx = mk._KdaFixture(acc=acc)
        got, ref = fx.mk_run(), fx.stock_run()
        torch.cuda.synchronize()
        errs = {k: _rel(got[k], ref[k]) for k in ref}
        r = max(errs.values())
        got0 = {k: v.clone() for k, v in got.items()}
        t_ref = _time(fx.stock_run, iters)
        t_mk = _time(fx.mk_run, iters)
        # fx.mk_run() clones fresh states per call, so replay drift here
        # means the launch itself is nondeterministic (barrier state), not
        # that buffers were reused.
        again = fx.mk_run()
        rep = max(_rel(again[k], got0[k]) for k in got0)
        mark = "!" if (r > TOL["kda"] or rep > 1e-6) else " "
        ok &= r <= TOL["kda"] and rep <= 1e-6
        print(f"{mark}kda  acc={acc:<10}{r:>10.2e}{TOL['kda']:>8.0e}"
              f"{t_ref:>10.1f}{t_mk:>9.1f}  "
              + " ".join(f"{k}={v:.1e}" for k, v in errs.items()))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--skip-kda", action="store_true")
    args = ap.parse_args()

    from vllm.model_executor.layers import glm53_megakernel as mk

    torch.cuda.init()
    ext = mk._build()
    major, minor, sms, smem = ext.probe_device()
    print(f"device cc={major}.{minor} sms={sms} smem_optin={smem}")
    assert (major, minor, sms) == (12, 1, 48), "not a GB10"

    ok = True
    ok &= probe_gemm(args.iters)
    if mk.ENABLE_W4:
        ok &= probe_w4(args.iters)
    ok &= probe_mhc(args.iters)
    if not args.skip_kda:
        ok &= probe_kda(args.iters)
    print("VERDICT:", "PASS" if ok else "FAIL (a ! cell disqualifies)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
