#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch-count bundle 2 (RUNBOOK EXP-20) -- offline numerics + timing gate.

Three self-owned micro-fusions of the GLM-5.3 decode step, each checked on
the fleet shapes against the stock chain it replaces, inside the glm53 image
with the composed overlay mounted (bash probes/run_micro_fusion_check.sh):

  dual   glm53_kda_onepass dual f_b/g_b GEMM vs two F.linear
         (fp32-accumulated bf16 either way; reports bit-exact or the ulp class,
         and sweeps the N tile)
  kda    glm53_kda_onepass one-pass conv+recurrence+norm vs the stock three
         kernels + copies (states and output expected bit-exact; both cache
         dtypes and layouts, acceptance 1/3/5/8, varlen, a two-step chain,
         and the module's own boot self-test)
  kpool  batched kpool update with int64 positions vs the int32 cast copy
         (expected bit-exact caches)

Timing is CUDA-graph replay of one decode step's worth of layers (34 KDA
layers, 11 indexer layers) with DISTINCT weights/states per layer so nothing
is L2-hot across layers. Two numbers per arm: ``gpu`` times the second of two
back-to-back replays (the host submission of graph N+1 overlaps graph N, so
only device time is measured -- the megakernel bench's x2 discipline) and
``+launch`` times a single replay from an idle stream (device time plus the
cudaGraphLaunch submission, which grows with the node count and is also what
a decode step pays). Stock and fused are captured the same way in the same
process; read the ratios. The stock KDA chain writes its conv output into
the input in place, so its timing loop gets pre-cloned inputs and no copy
node. Exit status is non-zero on any numerics failure.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Arm the bundle for this process (a caller's explicit value wins).
os.environ.setdefault("VLLM_GLM53_KDA_DUAL_GEMM", "1")
os.environ.setdefault("VLLM_GLM53_KDA_ONEPASS", "1")
os.environ.setdefault("VLLM_GLM53_KPOOL_UPDATE_DIRECT_POS", "1")

sys.path.insert(0, os.environ.get("MK_PKG_PATH",
                                  "/usr/local/lib/python3.12/dist-packages"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# the repo's relative-error metric (probes dir is sys.path[0] when run as a file)
from megakernel_glm53_bench import _rel as rel  # noqa: E402

DEV = "cuda"
FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  !! FAIL: {msg}")


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16)
    if t.dtype == torch.float32:
        return t.contiguous().view(torch.int32)
    return t


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(_bits(a), _bits(b)))


def bf16_mismatch(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """(# elements whose bf16 bits differ, max ulp distance)."""
    ai = _bits(a).to(torch.int32)
    bi = _bits(b).to(torch.int32)
    d = (ai - bi).abs()
    return int((d != 0).sum()), int(d.max()) if d.numel() else 0


def graph_time(fn, iters: int, warm: int = 5) -> tuple[float, float]:
    """(gpu_ms, launch_ms) per replay of a CUDA graph holding one call of fn().

    gpu_ms: the second of two back-to-back replays -- its host submission is
    hidden behind the first replay's execution, so the events bracket device
    time only. launch_ms: a single replay recorded from an idle stream, i.e.
    device time plus cudaGraphLaunch submission (node-count proportional).
    Medians over ``iters``.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    gpu, launch = [], []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)
        # single replay from idle: submission + device
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        launch.append(e0.elapsed_time(e1))
        # pair: the second replay's submission overlaps the first's execution
        g.replay()
        e2.record()
        g.replay()
        e3.record()
        torch.cuda.synchronize()
        gpu.append(e2.elapsed_time(e3))
    gpu.sort()
    launch.sort()
    return gpu[len(gpu) // 2], launch[len(launch) // 2]


def _fmt(tag: str, layers: int, stock: tuple[float, float], fused: tuple[float, float]) -> str:
    sg, sl = stock
    fg, fl = fused
    return (f"  timing {tag}: {layers} layers  gpu stock {sg*1000/layers:7.1f} -> fused "
            f"{fg*1000/layers:7.1f} us/layer (step {fg-sg:+.3f} ms)  |  +launch stock "
            f"{sl*1000/layers:7.1f} -> fused {fl*1000/layers:7.1f} us/layer (step {fl-sl:+.3f} ms)")


# ---------------------------------------------------------------------------
# dual f_b/g_b GEMM
# ---------------------------------------------------------------------------
def section_dual(iters: int) -> None:
    print("== dual: f_b + g_b one launch vs two F.linear ==")
    from vllm.model_executor.layers import glm53_kda_onepass as ko

    H, D, XS = 16, 128, 6416
    OFF = 3 * H * D + H
    g = torch.Generator(device=DEV).manual_seed(77)
    for bn in (128, 64, 32):
        exact = True
        worst = (0, 0)
        for m in (1, 8, 16, 32):
            x = torch.randn((m, XS), generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)
            wf = (torch.randn((H * D, D), generator=g, device=DEV, dtype=torch.float32) * 0.05).to(torch.bfloat16)
            wg = (torch.randn((H * D, D), generator=g, device=DEV, dtype=torch.float32) * 0.05).to(torch.bfloat16)
            x_fg = x[:, OFF:OFF + 2 * D]
            if not ko.dual_gemm_applicable(x_fg, wf, wg):
                fail(f"dual GEMM not applicable at M={m} on the fleet shape")
                continue
            f0 = F.linear(x[:, OFF:OFF + D], wf)
            g0 = F.linear(x[:, OFF + D:OFF + 2 * D], wg)
            f1, g1 = ko.dual_gate_gemm(x_fg, wf, wg, block_n=bn)
            torch.cuda.synchronize()
            for name, a, b in (("f_b", f0, f1), ("g_b", g0, g1)):
                n, ulp = bf16_mismatch(a, b)
                r = rel(b, a)
                if n:
                    exact = False
                    worst = max(worst, (ulp, n))
                if r > 1e-2:
                    fail(f"dual GEMM BLOCK_N={bn} {name} M={m}: rel {r:.3e} (mismatch {n}, max ulp {ulp})")
                if bn == ko._DUAL_BLOCK_N:
                    print(f"  M={m:2d} {name}: rel {r:.2e}  bf16 mismatches {n}/{a.numel()}  max ulp {ulp}")
        print(f"  BLOCK_N={bn:3d} numerics class: "
              f"{'bit-exact with cuBLAS' if exact else f'bf16 reduce-order (max ulp {worst[0]}, {worst[1]} elems)'}")
    L = 34
    wfs = [(torch.randn((H * D, D), device=DEV, dtype=torch.float32) * 0.05).to(torch.bfloat16) for _ in range(L)]
    wgs = [(torch.randn((H * D, D), device=DEV, dtype=torch.float32) * 0.05).to(torch.bfloat16) for _ in range(L)]
    for m in (8, 16, 32):
        x = torch.randn((m, XS), device=DEV, dtype=torch.float32).to(torch.bfloat16)

        def run_stock():
            for i in range(L):
                F.linear(x[:, OFF:OFF + D], wfs[i])
                F.linear(x[:, OFF + D:OFF + 2 * D], wgs[i])

        t0 = graph_time(run_stock, iters)
        for bn in (128, 64, 32):
            def run_fused(bn=bn):
                for i in range(L):
                    ko.dual_gate_gemm(x[:, OFF:OFF + 2 * D], wfs[i], wgs[i], block_n=bn)
            t1 = graph_time(run_fused, iters)
            print(_fmt(f"M={m:2d} BLOCK_N={bn:3d}", L, t0, t1))


# ---------------------------------------------------------------------------
# KDA one-pass
# ---------------------------------------------------------------------------
def section_kda(iters: int) -> None:
    print("== kda: one-pass conv+recurrence+norm vs stock chain ==")
    from vllm.model_executor.layers import glm53_kda_onepass as ko

    # the boot gate itself
    ok, detail = ko.selftest(torch.device(DEV))
    print(f"  module self-test: {'PASS' if ok else 'FAIL'} ({detail})")
    if not ok:
        fail(f"selftest: {detail}")

    cases = [
        ("C=1 uniform T=8 acc=8", 1, [8], [8], "SD", torch.bfloat16),
        ("C=1 uniform T=8 acc=1", 1, [8], [1], "SD", torch.bfloat16),
        ("C=1 uniform T=8 acc=3 DS", 1, [8], [3], "DS", torch.bfloat16),
        ("C=1 uniform T=8 acc=8 fp32 conv", 1, [8], [8], "SD", torch.float32),
        ("C=2 uniform T=8 acc=[1,3]", 2, [8, 8], [1, 3], "SD", torch.bfloat16),
        ("C=4 uniform T=8 acc=[8,3,1,5]", 4, [8, 8, 8, 8], [8, 3, 1, 5], "DS", torch.bfloat16),
        ("C=4 uniform acc=[8,3,1,5] fp32 DS", 4, [8, 8, 8, 8], [8, 3, 1, 5], "DS", torch.float32),
        ("C=4 varlen T=[8,3,8,1] acc=[2,3,8,1]", 4, [8, 3, 8, 1], [2, 3, 8, 1], "SD", torch.bfloat16),
    ]
    seed = 4242
    for name, n, lens, accs, layout, cdt in cases:
        fx = ko.make_fixture(n, lens, accs, layout, seed, conv_dtype=cdt, lines=48)
        seed += 1
        H, D, T_all = fx["H"], fx["D"], fx["T_all"]
        cs0, rec0 = fx["conv_state"].clone(), fx["rec"].clone()
        cs1, rec1 = fx["conv_state"].clone(), fx["rec"].clone()
        out0 = torch.zeros((T_all, H, D), dtype=torch.bfloat16, device=DEV)
        out1 = torch.zeros((T_all, H, D), dtype=torch.bfloat16, device=DEV)
        y0 = ko.run_stock_chain(fx, cs0, rec0, out0)
        y1 = ko.run_onepass(fx, cs1, rec1, out1)
        torch.cuda.synchronize()
        cs_ok = bits_equal(cs0, cs1)
        rec_ok = bits_equal(rec0, rec1)
        n_mis, ulp = bf16_mismatch(y0, y1)
        r = rel(y1, y0)
        if not cs_ok:
            d = (cs0.float() - cs1.float()).abs()
            fail(f"kda {name}: conv state differs ({int((d != 0).sum())} elems, max {d.max():.3e})")
        if not rec_ok:
            d = (rec0 - rec1).abs()
            fail(f"kda {name}: recurrent state differs ({int((d != 0).sum())} elems, max {d.max():.3e})")
        if r > 2e-3:
            fail(f"kda {name}: output rel {r:.3e} above the reduce-order gate 2e-3")
        if torch.isnan(y1.float()).any():
            fail(f"kda {name}: NaN in output")
        print(f"  {name:40s} conv state {'bit-exact' if cs_ok else 'DIFF'}  rec state "
              f"{'bit-exact' if rec_ok else 'DIFF'}  out rel {r:.2e}  bf16 mismatch {n_mis}/{y0.numel()} (max ulp {ulp})")
        # a second launch on the same counters must behave the same (monotonic counters)
        cs2, rec2 = fx["conv_state"].clone(), fx["rec"].clone()
        out2 = torch.zeros_like(out1)
        ko.run_onepass(fx, cs2, rec2, out2)
        torch.cuda.synchronize()
        if not (bits_equal(cs2, cs1) and bits_equal(rec2, rec1) and bits_equal(out2, out1)):
            fail(f"kda {name}: second launch differs from the first (counter arithmetic?)")

    # two consecutive verify steps on the same requests: step 2 reads the
    # conv/recurrent state step 1 rolled, with fresh tokens and acceptances
    print("  two-step chain (state rolled by step 1 feeds step 2):")
    fx1 = ko.make_fixture(2, [8, 8], [8, 3], "SD", 777, lines=32)
    fx2 = ko.make_fixture(2, [8, 8], [2, 6], "SD", 778, lines=32)
    for key in ("conv_w", "a_log", "dt_bias", "nw", "idx"):
        fx2[key] = fx1[key]  # same layer, same slots
    cs_s, rec_s = fx1["conv_state"].clone(), fx1["rec"].clone()
    cs_f, rec_f = fx1["conv_state"].clone(), fx1["rec"].clone()
    for step, fx in ((1, fx1), (2, fx2)):
        out_s = torch.zeros((fx["T_all"], fx["H"], fx["D"]), dtype=torch.bfloat16, device=DEV)
        out_f = torch.zeros_like(out_s)
        y_s = ko.run_stock_chain(fx, cs_s, rec_s, out_s)
        y_f = ko.run_onepass(fx, cs_f, rec_f, out_f)
        torch.cuda.synchronize()
        ok = bits_equal(cs_s, cs_f) and bits_equal(rec_s, rec_f) and bits_equal(y_s, y_f)
        print(f"    step {step}: conv state {'bit-exact' if bits_equal(cs_s, cs_f) else 'DIFF'}  rec state "
              f"{'bit-exact' if bits_equal(rec_s, rec_f) else 'DIFF'}  out {'bit-exact' if bits_equal(y_s, y_f) else 'DIFF'}")
        if not ok:
            fail(f"kda two-step chain: step {step} differs from stock")

    # applicability: what the guard must decline / admit
    fx = ko.make_fixture(1, [8], [8], "SD", 999, lines=16)
    cs = ko.dim_first(fx["conv_state"], fx["proj"])
    out = torch.zeros((8, 16, 128), dtype=torch.bfloat16, device=DEV)

    def _strided_like(t):
        # the same [rows, S] values in a view whose inner stride is 2
        wide = torch.zeros((t.shape[0], 2 * t.shape[1]), dtype=t.dtype, device=t.device)
        view = wide[:, ::2]
        view.copy_(t)
        return view

    def adm(**kw):
        args = dict(num_actual_tokens=8, num_spec_decodes=1, local_num_heads=16, head_dim=128,
                    conv_size=4, max_query_len=8)
        args.update(kw)
        return ko.onepass_applicable(fx["x"], fx["g1"], fx["g2"], fx["conv_w"], None, cs, fx["rec"],
                                     fx["cu"], fx["idx"], fx["acc"], out, **args)

    checks = [
        ("fleet shape admitted", adm(), True),
        ("padded projection rows admitted (actual < rows)", ko.onepass_applicable(
            torch.cat([fx["x"], fx["x"]]), fx["g1"], fx["g2"], fx["conv_w"], None, cs, fx["rec"],
            fx["cu"], fx["idx"], fx["acc"], out, num_actual_tokens=8, num_spec_decodes=1,
            local_num_heads=16, head_dim=128, conv_size=4, max_query_len=8), True),
        ("non-power-of-two V blocks declined", adm(block_v=24), False),
        ("too many request-heads declined", adm(num_spec_decodes=257), False),
        ("conv width != 4 declined", adm(conv_size=5), False),
        ("strided state indices declined", ko.onepass_applicable(
            fx["x"], fx["g1"], fx["g2"], fx["conv_w"], None, cs, fx["rec"], fx["cu"],
            _strided_like(fx["idx"]), fx["acc"], out, num_actual_tokens=8, num_spec_decodes=1,
            local_num_heads=16, head_dim=128, conv_size=4, max_query_len=8), False),
        ("fp16 conv state declined", ko.onepass_applicable(
            fx["x"], fx["g1"], fx["g2"], fx["conv_w"], None, cs.to(torch.float16), fx["rec"], fx["cu"],
            fx["idx"], fx["acc"], out, num_actual_tokens=8, num_spec_decodes=1,
            local_num_heads=16, head_dim=128, conv_size=4, max_query_len=8), False),
    ]
    for name, got, want in checks:
        if got != want:
            fail(f"applicability: {name} (got {got})")
        print(f"  applicability: {name} .. {'OK' if got == want else 'FAIL'}")

    # timing: 34 layers, distinct weights/states, C=1 T=8 (pre-cloned stock inputs)
    L = 34
    fxs = [ko.make_fixture(1, [8], [8], "SD", 1000 + i, lines=16) for i in range(L)]
    outs = [torch.zeros((8, 16, 128), dtype=torch.bfloat16, device=DEV) for _ in range(L)]
    xs = [f["x"].clone() for f in fxs]

    def run_stock():
        for i in range(L):
            ko.run_stock_chain(fxs[i], fxs[i]["conv_state"], fxs[i]["rec"], outs[i], x=xs[i])

    t0 = graph_time(run_stock, iters)
    for bv in (8, 16, 32):
        def run_fused(bv=bv):
            for i in range(L):
                ko.run_onepass(fxs[i], fxs[i]["conv_state"], fxs[i]["rec"], outs[i], block_v=bv)
        t1 = graph_time(run_fused, iters)
        print(_fmt(f"C=1 T=8 one-pass BV={bv:2d}", L, t0, t1))
    del fxs, outs, xs
    torch.cuda.empty_cache()
    fx4 = [ko.make_fixture(4, [8] * 4, [8, 3, 1, 5], "SD", 2000 + i, lines=40) for i in range(L)]
    outs4 = [torch.zeros((32, 16, 128), dtype=torch.bfloat16, device=DEV) for _ in range(L)]
    xs4 = [f["x"].clone() for f in fx4]

    def run_stock4():
        for i in range(L):
            ko.run_stock_chain(fx4[i], fx4[i]["conv_state"], fx4[i]["rec"], outs4[i], x=xs4[i])

    def run_fused4():
        for i in range(L):
            ko.run_onepass(fx4[i], fx4[i]["conv_state"], fx4[i]["rec"], outs4[i])

    t0 = graph_time(run_stock4, iters)
    t1 = graph_time(run_fused4, iters)
    print(_fmt("C=4 T=8 one-pass BV= 8", L, t0, t1))
    del fx4, outs4, xs4
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# kpool update with int64 positions
# ---------------------------------------------------------------------------
def section_kpool(iters: int) -> None:
    print("== kpool: batched update with int64 positions vs int32 cast ==")
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_decode_update_and_maybe_write_cache_batched,
    )
    g = torch.Generator(device=DEV).manual_seed(99)
    B, NEXT_N, HD, POOL, PAGE = 4, 8, 128, 4, 64
    blocks, tblocks = 64, 32
    kv = torch.randint(0, 255, (blocks, PAGE, HD + 4), generator=g, dtype=torch.uint8, device=DEV)
    tail = (torch.randn((tblocks, 2, POOL, HD), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    key = (torch.randn((B, NEXT_N, HD), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    score = (torch.randn((B, NEXT_N, HD), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    ape = torch.randn((POOL, HD), generator=g, device=DEV, dtype=torch.float32) * 0.1
    base_pos = torch.tensor([37, 1000, 3, 4094], device=DEV)
    pos64 = (base_pos[:, None] + torch.arange(NEXT_N, device=DEV)[None, :]).to(torch.int64)
    tail_block = torch.arange(B, device=DEV)[:, None] * 3 + 1
    tail_slot = (tail_block * POOL + pos64 % POOL).to(torch.int32)
    slot = torch.where(pos64 % POOL == POOL - 1,
                       (pos64 // POOL + 7 * torch.arange(B, device=DEV)[:, None]) % (blocks * PAGE),
                       torch.full_like(pos64, -1)).to(torch.int32)
    kv0, tail0 = kv.clone(), tail.clone()
    kv1, tail1 = kv.clone(), tail.clone()
    kpool_decode_update_and_maybe_write_cache_batched(
        kv0, tail0, tail_slot, key, score, ape, slot, pos64.to(torch.int32), POOL, HD, round_scale=True)
    kpool_decode_update_and_maybe_write_cache_batched(
        kv1, tail1, tail_slot, key, score, ape, slot, pos64, POOL, HD, round_scale=True)
    torch.cuda.synchronize()
    same = bool(torch.equal(kv0, kv1)) and bits_equal(tail0, tail1)
    if not same:
        fail("kpool int64 positions: cache writes differ from the int32 path")
    touched = int((kv0 != kv).any(dim=-1).sum()) + int((tail0 != tail).any(dim=-1).sum())
    print(f"  caches after update: {'bit-exact' if same else 'DIFF'} ({touched} rows written)")

    def run_stock():
        for _ in range(11):
            p = pos64.to(torch.int32)
            kpool_decode_update_and_maybe_write_cache_batched(
                kv0, tail0, tail_slot, key, score, ape, slot, p, POOL, HD, round_scale=True)

    def run_fused():
        for _ in range(11):
            kpool_decode_update_and_maybe_write_cache_batched(
                kv1, tail1, tail_slot, key, score, ape, slot, pos64, POOL, HD, round_scale=True)

    t0 = graph_time(run_stock, iters)
    t1 = graph_time(run_fused, iters)
    print(_fmt("11 layers cast+update -> direct", 11, t0, t1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="dual,kda,kpool")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    print(f"device {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"sections {args.sections}  iters {args.iters}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    secs = {"dual": section_dual, "kda": section_kda, "kpool": section_kpool}
    for name in args.sections.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            secs[name](args.iters)
        except Exception as e:  # keep going, report at the end
            import traceback
            traceback.print_exc()
            fail(f"section {name} raised {type(e).__name__}: {e}")
    print("== VERDICT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
