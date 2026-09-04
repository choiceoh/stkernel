#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53_megakernel numerics + timing probe (fresh container, never serving).

Gate ladder step 2 of overlay/modules/glm53_megakernel/README.md: this is
the pre-boot proof. It diffs each persistent segment against the stock path
it replaces on production shapes and times both arms with CUDA events.

    /repo/probes/megakernel_glm53_bench.py [--iters 30] [--skip-kda]
        [--segments mhc] [--stock raw|dispatch|both] [--sinkhorn 20]

BASIS NOTE (2026-09-03): --sinkhorn defaults to the driver's
`SINKHORN_SERVED` (20), the value both models serve (`hc_sinkhorn_iters`,
passed at the call site as `sinkhorn_repeat`) AND the value the boot
self-test gates on -- one number, so the arming gate and this probe cannot
disagree about what the segment is.
It used to be hard-coded to 4, and the MHC rows recorded in MEASUREMENTS
(4/9/10차) were measured that way -- the sinkhorn is a runtime loop in BOTH
arms (`for it < sinkhorn_repeat - 1` in the .cu, `T.serial(...)` in the
TileLang pair) and the two implementations do not scale with it alike, so a
ratio taken at 4 does not transfer to serving. Pass --sinkhorn 4 to reproduce
the older rows; the header prints whichever value was used.

or, on srv4, via the wrapper that bind-mounts the composed overlay at its
real image paths first:

    bash /repo/probes/run_megakernel_bench.sh [--profile glm53|dsv4] [--iters 30]

The wrapper picks the profile's image, composed build tree, mount list and
package root; the profile decides which segments can run at all (dsv4
reaches MHC alone).

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

import os

# The driver reads its knobs at ITS import, which main() does after argparse --
# so arming is a function of the selected segments, not an import-time
# constant. It used to setdefault every segment here: on dsv4, whose profile
# reaches MHC alone, arm() then built W4 packs for the GEMM self-test and the
# two KDA fixture packs (6416x4096, 4096x2048 -- a 16-candidate search per row
# chunk plus an empty_cache() per pack since #268) before an MHC-only
# measurement, and any DISARM those self-tests logged landed in the MHC run's
# log as if the lane under test had failed.
_SEG_KNOB = {"mhc": "VLLM_GLM53_MK_MHC", "gemm": "VLLM_GLM53_MK_GEMM",
             "exact": "VLLM_GLM53_MK_GEMM", "kda": "VLLM_GLM53_MK_KDA"}


def _arm_env(segs) -> None:
    """Arm ONLY the selected segments, for this process; a caller's explicit
    value still wins (setdefault) so a sweep can hold a knob at 0."""
    os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
    for seg in segs:
        os.environ.setdefault(_SEG_KNOB[seg], "1")
    # programmatic launches: the mk_x2 column below is where they show
    os.environ.setdefault("VLLM_GLM53_MK_PDL", "1")


# The image's package root. GLM's image installs to dist-packages, dsv4's to
# the venv site-packages -- the wrapper passes the profile's TARGET_PREFIX so
# a probe run can never import the wrong tree (it would import nothing at all
# and read as "vllm missing" instead of "wrong profile").
sys.path.insert(0, os.environ.get("MK_PKG_PATH",
                                  "/usr/local/lib/python3.12/dist-packages"))

import torch  # noqa: E402

TOL = {"mhc": 1e-3, "gemm": 0.15, "kda": 2e-2}  # gemm: e2m1 by-design class
TOL_SPLIT = 1e-3  # sweep rows vs the same lane at another split: summation order only
DEV = "cuda"


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).norm()
    den = b.float().norm()
    return float(d / den) if den > 0 else float(d)


# GB10 has 24 MB of L2, and the weights these shapes read are 4-26 MB. A
# timing loop that re-reads the same weight therefore measures an L2-RESIDENT
# GEMM, not the DRAM-bound one a serving step does -- and which arm gets that
# gift depends on allocation order, so the same comparison swung 3-5x between
# process runs (stock m=16 n=4096 measured 45 us in isolation and 148 us with
# a 48 MB buffer merely allocated beforehand). Flush L2 between iterations so
# the number is the one production sees and is reproducible.
#
# The flush must be a WRITE pass: a read-only pass over 2x L2 does not evict
# lines that were re-read (a second sum() over 48 MB left the 8 MB weight
# resident and the kernel 20% faster than cold). And the write pass must be
# followed by a SPACER: the fill completes when its stores reach L2, not
# DRAM, and the write-back of up to 24 MB of dirty lines then drains UNDER
# the timed kernel -- +15% at n=6416, +25% at n=2048/1024, measured against
# the same flush followed by a ~300 us compute-bound matmul (which lets the
# drain finish while the GPU stays busy). zero_ and add_ flushes, with or
# without an extra 5 ms sleep, all land on that same quiescent number, so the
# spacer is what matters, not the store flavour. The spacer also keeps the
# stream busy while Python issues the timed call, so the start event never
# lands on an idle stream: a synchronize-then-sleep variant exposed the
# stock arm's launch latency as 200-450 us of pure gap. Never sync here.
_L2_BYTES = 24 << 20
_L2_FLUSH = None
_SPACER = None
_DRAIN = None


def _l2_flush(hot=()) -> None:
    """Cold weights, HOT activations: after the flush the kernel's small
    inputs are touched again (read+write, so they sit in L2 dirty exactly
    as the previous kernel of a decode step leaves them). Without this the
    flush also evicted x, and the mk gemm's A-quant prologue -- 8 KB of x per
    block -- paid a DRAM round trip that production never pays, and paid it
    unevenly: blocks whose x loads queued behind other blocks' W fill took
    13 us where the rest took 2, and the publishing barrier then held every
    block for the slowest (bar1 median 8 us in the phase stamps)."""
    global _L2_FLUSH, _SPACER, _DRAIN
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(2 * _L2_BYTES, dtype=torch.int8, device=DEV)
        a = torch.randn(2048, 2048, dtype=torch.bfloat16, device=DEV)
        b = torch.randn(2048, 2048, dtype=torch.bfloat16, device=DEV)
        _SPACER = (a, b, torch.empty(2048, 2048, dtype=torch.bfloat16,
                                     device=DEV))
        _DRAIN = torch.zeros(16 << 20, dtype=torch.float32, device=DEV)  # 64 MB
    _L2_FLUSH.zero_()
    # The fill leaves up to 24 MB of DIRTY lines in L2; the compute spacer
    # lets the fill's own drain finish but does not evict those, so the
    # timed kernel evicts them and their write-back runs under it: a pure
    # 24 MB stream measured 148 GB/s after zero_ + matmul and 232 GB/s
    # after zero_ + a 128 MB read stream (srv2, sustained state, both arms
    # affected; the pair's SECOND launch was already at 228). A read pass
    # evicts once-written lines (they carry no reuse) and forces the
    # write-back out before the start event. Two passes.
    torch.mm(_SPACER[0], _SPACER[1], out=_SPACER[2])
    # the drain goes LAST: the matmul spacer's own 8 MB output is dirty too
    _DRAIN.sum()
    _DRAIN.sum()
    for t in hot:
        t.add_(0)


def _time_stats(fn, iters: int, hot=()) -> tuple[float, float, float]:
    """(median, min, max) in us over iters cold-L2 launches; `hot` lists
    the activation tensors re-touched after each flush (see _l2_flush)."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        _l2_flush(hot)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)  # us
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def _time(fn, iters: int, hot=()) -> float:
    return _time_stats(fn, iters, hot)[0]


GEMM_SHAPES = [(8, 6416, 4096), (16, 4096, 4096), (32, 2048, 4096), (32, 1024, 4096),
               (8, 1024, 4096), (8, 4096, 512), (8, 4096, 4096)]


def probe_gemm(iters: int) -> bool:
    """MK-GEMM (the W4 lane) against the stock quant+deepgemm pair on the
    decode shapes: by-design error (e2m1, gated at 0.15), single-launch
    and back-to-back timings, replay stability."""
    from vllm.model_executor.layers.glm53_fp8_dense import _fp8_dense_gemm
    from vllm.model_executor.layers import glm53_megakernel as mk

    ok = True
    print(f"{'shape':<24}{'rel_err':>8}{'gate':>8}{'stock_us':>10}{'mk_us':>9}"
          f"{'mk_GBps':>9}{'st_GBps':>9}{'mk_spread':>10}{'mk_x2':>7}"
          f"{'st_x2':>7}")
    # The original four sweep n at k = 4096; the three after them are the
    # per-rank production shapes whose launches sit in the 30-45 us class
    # (86/step, 3.5 ms, 2.4 ms of it exposed on the critical path -- 28차):
    # the shared expert's gate_up [1024 x 4096] and down [4096 x 512], and
    # o_proj [4096 x 4096]. Override with --gemm-shapes m:n:k,...
    for m, n, k in GEMM_SHAPES:
        torch.manual_seed(0)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.05
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
        sq, sws, srows, scols = mk._stock_fp8_pair(w)
        p4 = mk.build_mk_weight_w4(w)
        # a SECOND weight for the back-to-back pair: two launches on the
        # same weight leave the second one reading L2 (4 MB at n=1024 is
        # entirely resident after the first), which is not what a decode
        # step's run of different projections sees. Pair = w then w2.
        w2 = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.05
        sq2, sws2, srows2, scols2 = mk._stock_fp8_pair(w2)
        p4b = mk.build_mk_weight_w4(w2)
        del w2
        ref = _fp8_dense_gemm(x, sq, sws, srows, scols)
        got = mk._gemm_call(x, p4, n)
        torch.cuda.synchronize()
        r = _rel(got, ref)
        t_ref = _time(lambda: _fp8_dense_gemm(x, sq, sws, srows, scols),
                      iters, hot=(x,))
        t_mk, t_lo, t_hi = _time_stats(
            lambda: mk._gemm_call(x, p4, n), iters, hot=(x,))
        # two launches back to back on two DIFFERENT weights, per launch:
        # what a decode step's run of GEMMs sees. With VLLM_GLM53_MK_PDL=1
        # the second one starts on the SMs the first frees and prefetches
        # its W during the first's tail; a single launch cannot show that.
        t_x2 = _time(lambda: (mk._gemm_call(x, p4, n),
                              mk._gemm_call(x, p4b, n)),
                     iters, hot=(x,)) / 2
        # the stock pair too: back-to-back launches amortise launch latency
        # for either arm, so single-launch columns flatter neither and the
        # x2 pair is the like-for-like comparison.
        t_sx2 = _time(lambda: (_fp8_dense_gemm(x, sq, sws, srows, scols),
                               _fp8_dense_gemm(x, sq2, sws2, srows2, scols2)),
                      iters, hot=(x,)) / 2
        # replay stability: the timing loop just relaunched the same kernel
        # dozens of times over ONE workspace -- the monotonic-barrier
        # contract. The result must be unchanged.
        again = mk._gemm_call(x, p4, n)
        torch.cuda.synchronize()
        rep = _rel(got, again)
        mark = "!" if (r > TOL["gemm"] or rep > 1e-6) else " "
        ok &= r <= TOL["gemm"] and rep <= 1e-6
        # effective DRAM rate: the bytes each arm must move once (the MK
        # lane streams nibbles + group exponents, 0.56x the fp8 bytes).
        nb_mk = p4[0].numel() + p4[1].numel() + x.numel() * 2 + m * n * 2
        nb_st = sq.numel() + sws.numel() * 4 + x.numel() * 2 + m * n * 2
        # spread = (max - min) / median over the timed launches. A bimodal
        # cell (two clusters inside one process) shows up here where a
        # median alone would hide it behind a plausible number.
        print(f"{mark}gemm m={m:<3}n={n:<5}k={k:<5}{r:>8.2e}{TOL['gemm']:>8.2f}"
              f"{t_ref:>10.1f}{t_mk:>9.1f}"
              f"{nb_mk / t_mk / 1e3:>9.0f}{nb_st / t_ref / 1e3:>9.0f}"
              f"{100 * (t_hi - t_lo) / t_mk:>9.1f}%{t_x2:>7.1f}{t_sx2:>7.1f}")
    return ok


# The one-unit-per-block shapes (the 30-45 us class) plus one of each
# neighbour: n=2048 (48 units at the model's ksr=3) and the o_proj [4096 x
# 4096] (96 units: the global path, a control row that must not move).
SWEEP_SHAPES = [(8, 1024, 4096), (8, 4096, 512), (8, 2048, 4096),
                (16, 1024, 4096), (32, 1024, 4096), (32, 4096, 512),
                (8, 4096, 4096)]


def _stamps(ext, grid: int) -> str:
    """One line from the last launch's phase stamps (MK_PHASE_TS builds):
    medians over the blocks that finished, ns -> us, relative to entry.
    Slots: 1 = A quant done (global: before the barrier; local: first
    k-block), 2 = prologue done (global: barrier released), 6 = first unit
    done, 4 = block done; 3/5/7 = accumulated mma / pipeline-wait / expand.
    Idle blocks never stamp 6 (either path) and are left out."""
    ts = ext.read_ts()
    if not ts:
        return "(no stamps: build with VLLM_GLM53_MK_PHASE_TS=1)"
    rows = [ts[b * 8:(b + 1) * 8] for b in range(grid)]
    # active blocks only: a block that ran a unit stamped 6. Global-path
    # idle blocks pass the barrier and stamp 4 too (never 6); local-path
    # idle blocks return before either.
    act = [r for r in rows if r[6] > 0 and r[4] > 0 and r[0] > 0]
    if not act:
        return "(no active blocks stamped)"

    def med(vals):
        vals = sorted(vals)
        return vals[len(vals) // 2] / 1e3

    t0 = min(r[0] for r in act)
    skew = (max(r[0] for r in act) - t0) / 1e3
    rel = lambda s: med([r[s] - r[0] for r in act])  # noqa: E731
    span = max(r[4] for r in act) - t0
    return (f"act={len(act):<2} skew={skew:4.1f} q={rel(1):5.1f} pro={rel(2):5.1f}"
            f" u1={rel(6):5.1f} end={rel(4):5.1f} span={span / 1e3:5.1f}"
            f" | mma={med([r[3] for r in act]):4.1f}"
            f" wait={med([r[5] for r in act]):4.1f}"
            f" exp={med([r[7] for r in act]):4.1f}")


def probe_gemm_sweep(iters: int, stamps: bool, shapes=None) -> bool:
    """The standalone lane under its probe knobs, per shape: the
    barrier-free local-quant path (VLLM_GLM53_MK_LOCALQ=2: every launch
    with units <= grid, launched on `lgrid` blocks) against the global path
    (0), over the split (VLLM_GLM53_MK_KSR, 0 = the cost model; a forced
    value equal to the model's pick is not timed twice). `same` = the local
    path's output is BITWISE the global path's at the same split (it must
    be: same bytes into the same mma); rel = against the global path at the
    model's split (a different split changes the fp32 summation order, so
    that one is a tolerance, not equality). MK against MK only -- the stock
    comparison is probe_gemm's table, which a sweep run keeps."""
    from vllm.model_executor.layers import glm53_megakernel as mk

    ext = mk._EXT
    ok = True
    print(f"{'shape':<22}{'lq':>3}{'ksr':>4}{'units':>6}{'lgrid':>6}{'mk_us':>8}"
          f"{'spread':>8}{'mk_x2':>7}{'same':>5}{'rel':>9}"
          f"{'  stamps (us)' if stamps else ''}")
    for m, n, k in (shapes or SWEEP_SHAPES):
        torch.manual_seed(0)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.05
        w2 = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.05
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
        p4 = mk.build_mk_weight_w4(w)
        p4b = mk.build_mk_weight_w4(w2)
        del w, w2
        kblk = k // 128
        ext.set_probe(0, 0, -1)
        ref = mk._gemm_call(x, p4, n)  # global path, the model's split
        torch.cuda.synchronize()
        model_ksr = ext.gemm_plan(m, n, k, 0)[1]
        by_ksr = {}
        for lq in (0, 2):
            for ksr in [0] + [r for r in (2, 3, 4, 6, 8, 12, 16)
                              if r <= kblk and r != model_ksr]:
                ext.set_probe(ksr, lq, -1)
                grid, ksr_eff, units, localq, lgrid = ext.gemm_plan(m, n, k, 0)
                if lq and not localq:
                    continue  # the plan refuses the local path here
                if ksr and units == n // 128 and ksr_eff > 1:
                    continue  # the split gate refused this force: an unsplit run
                got = mk._gemm_call(x, p4, n)
                torch.cuda.synchronize()
                key = ksr_eff
                if lq == 0:
                    by_ksr[key] = got
                    same = "-"
                else:
                    g0 = by_ksr.get(key)
                    same_b = g0 is not None and torch.equal(got, g0)
                    same = "yes" if same_b else "NO"
                    ok &= same_b
                # the reference is the MK global path at the model's split:
                # a different split reorders the fp32 sums (~1e-5 here), a
                # dropped or double-counted k-block moves it by percent --
                # the summation-noise class, not the e2m1 by-design class
                r = _rel(got, ref)
                ok &= r <= TOL_SPLIT
                t_mk, t_lo, t_hi = _time_stats(
                    lambda: mk._gemm_call(x, p4, n), iters, hot=(x,))
                t_x2 = _time(lambda: (mk._gemm_call(x, p4, n),
                                      mk._gemm_call(x, p4b, n)),
                             iters, hot=(x,)) / 2
                # replay stability on this knob setting
                again = mk._gemm_call(x, p4, n)
                torch.cuda.synchronize()
                rep_ok = torch.equal(again, got)
                ok &= rep_ok
                line = ""
                if stamps:
                    _l2_flush(hot=(x,))
                    mk._gemm_call(x, p4, n)
                    torch.cuda.synchronize()
                    line = "  " + _stamps(ext, grid)
                mark = " " if (same != "NO" and r <= TOL["gemm"] and rep_ok) else "!"
                print(f"{mark}m={m:<3}n={n:<5}k={k:<5}{1 if lq else 0:>3}{ksr_eff:>4}"
                      f"{units:>6}{lgrid:>6}"
                      f"{t_mk:>8.1f}{100 * (t_hi - t_lo) / t_mk:>7.1f}%{t_x2:>7.1f}"
                      f"{same:>5}{r:>9.2e}{line}")
        del p4, p4b, by_ksr
    ext.set_probe(-1, -1, -1)  # back to the env's knobs for the segments after
    return ok


def probe_exact() -> bool:
    """The load-bearing W4 gate: weights ON the e2m1 grid pack losslessly,
    so the kernel must reproduce a torch fp32 matmul of the kernel-
    quantized activations against the dequantized pack to accumulation
    noise. It proves nibble / exponent / swizzle / expansion bit-for-bit
    (there is no fp8 MK arm left to diff against; the pure-torch twins
    mk_w4_dequant and _mk_quant_x_ref are the reference)."""
    from vllm.model_executor.layers import glm53_megakernel as mk

    print(f"{'case':<24}{'rel_err':>10}{'gate':>8}")
    torch.manual_seed(0)
    n, k, m = 1024, 4096, 8
    code = torch.randint(0, 8, (n, k // 16, 16), device=DEV)
    # PRODUCTION magnitudes, the same draw as the boot self-test
    # (_selftest_gemm): GLM-5.3's dense projections need group exponents
    # around 2^-7 (median), 2^-16 (p1). The pack normalises the tensor by
    # its median pow2 and stores an e4m3 scale whose exponent spans
    # [-5, 5] around it, so a fixture spanning 11 octaves round-trips
    # bit-exactly -- and one that draws the O(1) range [-5, 6] this probe
    # used to draw no longer fits: its 2^6 groups clamp (8.3% of groups),
    # the roundtrip reads 6.6e-2, and the run FAILs on the fixture, not
    # on the kernel (2026-09-04, srv2, the first on-device run of #268).
    sexp = torch.randint(-12, -2, (n, k // 16, 1), device=DEV)
    grid = torch.tensor(mk._E2M1_GRID, device=DEV)
    w_exact = (grid[code] * torch.exp2(sexp.float())) * torch.where(
        torch.randn_like(code.float()) < 0, -1.0, 1.0)
    w_exact = w_exact.view(n, k).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
    p4 = mk.build_mk_weight_w4(w_exact)
    w_back = mk.mk_w4_dequant(p4[0], p4[1], n, p4[2])  # p4[2]: 2^-shift
    e_pack = _rel(w_back, w_exact)  # the pack itself must round-trip
    ref = mk._mk_quant_x_ref(x) @ w_back.float().T
    # BOTH kernels: this fixture is a one-unit-per-block launch, so it is
    # the one shape where the local-quant kernel can be exact-gated -- and
    # the global kernel must not lose its own gate to that.
    ext = mk._EXT
    got = {}
    for lq in (0, 2):
        ext.set_probe(-1, lq, -1)
        got[lq] = mk._gemm_call(x, p4, n)
    ext.set_probe(-1, -1, -1)
    torch.cuda.synchronize()
    # the kernel writes bf16: judge against the bf16-rounded reference, no
    # element more than one bf16 ulp off (a different fp32 summation order
    # flips a few by one ulp; a layout bug moves whole rows)
    ok = e_pack == 0.0
    print(f"{'!' if e_pack else ' '}w4 pack roundtrip{e_pack:>17.2e}{0:>8.0e}")
    for lq, name in ((0, "global"), (2, "local")):
        e_exact, n_ulp = mk._exact_gate(got[lq], ref)
        good = e_exact <= 1e-3 and n_ulp == 0
        ok &= good
        print(f"{' ' if good else '!'}w4 exact grid ({name}){e_exact:>12.2e}"
              f"{1e-3:>8.0e}  over-ulp={n_ulp}")
    same = torch.equal(got[0], got[2])
    ok &= same
    print(f"{' ' if same else '!'}w4 local == global (bitwise){'yes' if same else 'NO':>10}")
    return ok


def probe_mhc(iters: int, sk: int) -> bool:
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tlk
    from vllm.model_executor.layers import glm53_megakernel as mk

    ok = True
    # The reference here is a HAND-ROLLED pair at tile_n=2/n_splits=4: the
    # kernel instrument, not the arm a boot takes. No profile's dispatcher
    # picks that config (GLM's heuristic takes tile_n=3 at T=8; dsv4's swept
    # pair takes 6/4 below T=8 and 4/4 with n_thr=128 to T=10), so reading
    # this row as "the win over stock" overstates it -- on dsv4 by exactly
    # the R1/R2 sweep that was already adopted. `--stock dispatch` is the row
    # that answers the serving question.
    print(f"raw stock arm: mhc_fused(tile_n=2, n_splits=4) + "
          f"big_fuse_with_norm(n_splits=4), sinkhorn_repeat={sk}")
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
            cm_ref, li_ref, nw, 4096, 1e-6, 1e-6, 1e-6, 1.0, sk, 1e-6,
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
                           1e-6, 1e-6, 1e-6, 1.0, 1e-6, sk)
        ref = stock(x, res, pm, cm, fn, nw)
        torch.cuda.synchronize()
        r = max(_rel(g, rr) for g, rr in zip(got, ref))
        got0 = tuple(g.clone() for g in got)
        hot = (x, res, pm, cm)
        t_ref = _time(lambda: stock(x, res, pm, cm, fn, nw), iters, hot=hot)
        t_mk = _time(lambda: mk._mhc_call(
            x, res, pm, cm.reshape(T, 16).contiguous(), fn,
            mk.hc_scale_ones(), mk.hc_base_zeros(), nw, T, 1e-6, 1e-6, 1e-6,
            1.0, 1e-6, sk), iters, hot=hot)
        # Three replays, not one. A single got0-vs-got2 comparison samples a
        # random variable once: p3's sumsq used to drift 0 / 4.1e-05 / 0 /
        # 3.3e-04 call to call, and this gate passed or failed on luck --
        # it waved the defect through for weeks. probes/mhc_replay.py is the
        # instrument for diagnosing one of these; this is the gate.
        rep = 0.0
        for _ in range(3):
            got2 = mk._mhc_call(x, res, pm, cm.reshape(T, 16).contiguous(),
                                fn, mk.hc_scale_ones(), mk.hc_base_zeros(),
                                nw, T, 1e-6, 1e-6, 1e-6, 1.0, 1e-6, sk)
            torch.cuda.synchronize()
            rep = max(rep, max(_rel(g, g0) for g, g0 in zip(got2, got0)))
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


def probe_mhc_dispatch(iters: int, sk: int) -> bool:
    """MHC through the image's OWN wrapper -- the arm a boot actually takes.

    `probe_mhc` above times the MK kernel against a hand-rolled stock pair at
    tile_n=2 / n_splits=4. That is the right instrument for the KERNEL and it
    is the basis every recorded MHC number was measured on, but it is NOT
    what a boot runs:

      * each profile's wrapper picks its own tile config -- dsv4's small-M
        pair is swept (R1/R2/R3, per-call 15.6 -> 13.1 us), so measuring MK
        against the unswept config would manufacture the sweep's win twice;
      * the serving hook sits under `use_small_fma` (T <= 16), even though
        the kernel's own gate is T <= 32. Above 16 the wrapper takes
        mhc_post + big_fuse and MK is never offered the call.

    So this arm calls the wrapper twice -- MK disarmed, then armed -- and
    prints a `hit` column that says whether MK served it. `hit=no` with a
    0.0 rel err is the receipt for the window, not a passing gate.
    """
    from vllm.model_executor.kernels.mhc import tilelang as tl
    from vllm.model_executor.layers import glm53_megakernel as mk

    call = getattr(tl, "_mhc_fused_post_pre_tilelang_impl", None) or \
        tl.mhc_fused_post_pre_tilelang

    mk.maybe_arm()  # so _ARMED reflects the self-test, not the first call
    armed0 = mk._ARMED["mhc"]
    if not armed0:
        print("!mhc dispatch: the MHC segment did not arm -- nothing to compare")
        return False

    # `hit` comes from the core's own served-call counter, not from patching
    # a private of it: the receipt has to survive a refactor of how mhc_hook
    # calls through, or every run would FAIL with "never offered a call" and
    # send the reader after a wiring bug that is not there.
    served = lambda: mk._HOOK_SERVED[0]   # noqa: E731
    ok = True
    any_hit = False
    try:
        print(f"{'shape':<22}{'rel_err':>10}{'gate':>8}{'stock_us':>10}"
              f"{'mk_us':>9}{'hit':>6}")
        for T in (8, 16, 32):
            torch.manual_seed(0)
            x = torch.randn(T, 4096, dtype=torch.bfloat16, device=DEV) * 0.1
            res = torch.randn(T, 4, 4096, dtype=torch.bfloat16, device=DEV) * 0.1
            pm = torch.rand(T, 4, dtype=torch.float32, device=DEV)
            cm = torch.rand(T, 4, 4, dtype=torch.float32, device=DEV)
            fn = torch.randn(24, 16384, dtype=torch.float32, device=DEV) * 0.02
            nw = torch.randn(4096, dtype=torch.bfloat16, device=DEV)
            args = (x, res, pm, cm, fn, mk.hc_scale_ones(), mk.hc_base_zeros(),
                    1e-6, 1e-6, 1e-6, 1.0, sk)
            kw = {"norm_weight": nw, "norm_eps": 1e-6}
            hot = (x, res, pm, cm)

            mk._ARMED["mhc"] = False
            ref = call(*args, **kw)
            t_ref = _time(lambda: call(*args, **kw), iters, hot=hot)

            mk._ARMED["mhc"] = True
            before = served()
            got = call(*args, **kw)
            torch.cuda.synchronize()
            hit = served() > before
            any_hit |= hit
            # timing the armed arm when MK was never offered the call would
            # measure the stock path twice and print it under mk_us
            t_mk = _time(lambda: call(*args, **kw), iters, hot=hot) if hit \
                else None

            r = max(_rel(g, rr) for g, rr in zip(got, ref))
            bad = r > TOL["mhc"]
            ok &= not bad
            print(f"{'!' if bad else ' '}mhc T={T:<17}{r:>10.2e}"
                  f"{TOL['mhc']:>8.0e}{t_ref:>10.1f}"
                  f"{(f'{t_mk:.1f}' if hit else '-'):>9}"
                  f"{('yes' if hit else 'no'):>6}")
    finally:
        mk._ARMED["mhc"] = armed0
    if not any_hit:
        # every row compared stock against stock: rel == 0 everywhere and the
        # gate would have passed without the kernel running once
        print("!mhc dispatch: MK was never offered a call (hit=no at every T) "
              "-- nothing was measured, so this is a FAIL, not a PASS")
        return False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--gemm-shapes", default=None,
                    help="m:n:k,... for the gemm table AND --gemm-sweep (default: "
                         "the decode sweep + the production small shapes / "
                         "SWEEP_SHAPES)")
    ap.add_argument("--skip-kda", action="store_true")
    # the standalone lane's knob sweep (local-quant path x split) on the
    # one-unit-per-block shapes; --stamps adds the phase stamps of a build
    # with VLLM_GLM53_MK_PHASE_TS=1 (a plain build prints a note instead)
    ap.add_argument("--gemm-sweep", action="store_true")
    ap.add_argument("--stamps", action="store_true")
    # Which segments this profile can even run. dsv4 reaches MHC alone: it has
    # no linear-attention layer (kda), no e2m1 dense pack (gemm/exact) and a
    # different MLA geometry. Naming a segment a profile cannot serve would
    # measure an arm nothing will ever run.
    ap.add_argument("--segments", default="gemm,exact,mhc,kda",
                    help="comma list of gemm,exact,mhc,kda (default: all)")
    # raw   = the hand-rolled stock pair (kernel instrument; the recorded basis)
    # both  = raw plus the wrapper's real arm (see probe_mhc_dispatch)
    ap.add_argument("--stock", choices=("raw", "dispatch", "both"),
                    default="raw")
    # what the models pass as sinkhorn_repeat (hc_sinkhorn_iters); unset
    # means the driver's SINKHORN_SERVED, which is also what the boot
    # self-test gates on. See the BASIS NOTE in the module docstring.
    ap.add_argument("--sinkhorn", type=int, default=None)
    args = ap.parse_args()
    if args.gemm_shapes:
        GEMM_SHAPES[:] = [tuple(int(v) for v in t.split(":")) for t in args.gemm_shapes.split(",")]
        SWEEP_SHAPES[:] = list(GEMM_SHAPES)
    if args.stamps and not args.gemm_sweep:
        print("--stamps applies to --gemm-sweep only; ignored")
    if args.sinkhorn is not None and args.sinkhorn < 1:
        print("--sinkhorn must be >= 1")
        return 2
    segs = [s.strip() for s in args.segments.split(",") if s.strip()]
    if args.gemm_sweep and "gemm" not in segs:
        segs.append("gemm")
    unknown = [s for s in segs if s not in ("gemm", "exact", "mhc", "kda")]
    if unknown:
        print(f"unknown segment(s): {unknown}")
        return 2
    if args.skip_kda and "kda" in segs:
        segs.remove("kda")
    if not segs:
        # an empty selection used to run nothing and print PASS -- the one
        # output this probe must never produce, since a green VERDICT is what
        # authorises the boot bracket
        print("--segments selected nothing to run")
        return 2
    _arm_env(segs)   # before the driver import: it reads the knobs there

    from vllm.model_executor.layers import glm53_megakernel as mk

    sk = mk.SINKHORN_SERVED if args.sinkhorn is None else args.sinkhorn
    torch.cuda.init()
    ext = mk._build()
    major, minor, sms, smem = ext.probe_device()
    print(f"device cc={major}.{minor} sms={sms} smem_optin={smem}")
    assert (major, minor, sms) == (12, 1, 48), "not a GB10"

    print(f"segments={','.join(segs)} stock={args.stock} "
          f"sinkhorn_repeat={sk}"
          f"{' (driver default)' if args.sinkhorn is None else ''}")
    ok = True
    if "gemm" in segs:
        ok &= probe_gemm(args.iters)
    if args.gemm_sweep:
        ok &= probe_gemm_sweep(args.iters, args.stamps)
    if "exact" in segs:
        ok &= probe_exact()
    if "mhc" in segs:
        if args.stock in ("raw", "both"):
            ok &= probe_mhc(args.iters, sk)
        if args.stock in ("dispatch", "both"):
            ok &= probe_mhc_dispatch(args.iters, sk)
    if "kda" in segs:
        ok &= probe_kda(args.iters)
    print("VERDICT:", "PASS" if ok else "FAIL (a ! cell disqualifies)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
