#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch-count bundle 2 (RUNBOOK EXP-20) -- offline numerics + timing gate.

Three self-owned micro-fusions of the GLM-5.3 decode step, each checked on
the fleet shapes against the stock chain it replaces, inside the glm53 image
with the composed overlay mounted (bash probes/run_micro_fusion_check.sh):

  dual   glm53_kda_onepass dual f_b/g_b GEMM vs two F.linear
         (fp32-accumulated bf16 either way; reports bit-exact or the ulp class)
  kda    glm53_kda_onepass one-pass conv+recurrence+norm vs the stock three
         kernels + copies (states expected bit-exact; output = norm reduce
         order class, reported as rel err + bf16 mismatch count)
  kpool  batched kpool update with int64 positions vs the int32 cast copy
         (expected bit-exact caches)

Timing is CUDA-graph replay of one decode step's worth of layers (34 KDA
layers, 11 indexer layers) with DISTINCT weights/states per layer so nothing is L2-hot
across layers; stock and fused are captured the same way in the same
process, so the ratio is what to read. Exit status is non-zero on any
numerics failure. Conventions: probes/megakernel_glm53_bench.py.
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

DEV = "cuda"
FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  !! FAIL: {msg}")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).norm()
    n = b.float().norm()
    return float(d / n) if n > 0 else float(d)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.view(torch.int32) if a.dtype == torch.float32
                            else a.view(torch.int16) if a.dtype == torch.bfloat16
                            else a, b.view(torch.int32) if b.dtype == torch.float32
                            else b.view(torch.int16) if b.dtype == torch.bfloat16
                            else b))


def bf16_mismatch(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """(# elements whose bf16 bits differ, max ulp distance)."""
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    d = (ai - bi).abs()
    return int((d != 0).sum()), int(d.max()) if d.numel() else 0


def graph_time(fn, iters: int, warm: int = 5) -> float:
    """Median ms per replay of a CUDA graph holding one call of fn()."""
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
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


# ---------------------------------------------------------------------------
# dual f_b/g_b GEMM
# ---------------------------------------------------------------------------
def section_dual(iters: int) -> None:
    print("== dual: f_b + g_b one launch vs two F.linear ==")
    from vllm.model_executor.layers import glm53_kda_onepass as ko

    H, D, XS = 16, 128, 6416
    OFF = 3 * H * D + H
    g = torch.Generator(device=DEV).manual_seed(77)
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
        f1, g1 = ko.dual_gate_gemm(x_fg, wf, wg)
        torch.cuda.synchronize()
        for name, a, b in (("f_b", f0, f1), ("g_b", g0, g1)):
            n, ulp = bf16_mismatch(a, b)
            r = rel(b, a)
            if n:
                exact = False
                worst = max(worst, (ulp, n))
            if r > 1e-2:
                fail(f"dual GEMM {name} M={m}: rel {r:.3e} (mismatch {n}, max ulp {ulp})")
            print(f"  M={m:2d} {name}: rel {r:.2e}  bf16 mismatches {n}/{a.numel()}  max ulp {ulp}")
    print(f"  numerics class: {'bit-exact with cuBLAS' if exact else f'bf16 reduce-order (max ulp {worst[0]})'}")
    L = 34
    wfs = [(torch.randn((H * D, D), device=DEV, dtype=torch.float32) * 0.05).to(torch.bfloat16) for _ in range(L)]
    wgs = [(torch.randn((H * D, D), device=DEV, dtype=torch.float32) * 0.05).to(torch.bfloat16) for _ in range(L)]
    for m in (8, 16, 32):
        x = torch.randn((m, XS), device=DEV, dtype=torch.float32).to(torch.bfloat16)

        def run_stock():
            for i in range(L):
                F.linear(x[:, OFF:OFF + D], wfs[i])
                F.linear(x[:, OFF + D:OFF + 2 * D], wgs[i])

        def run_fused():
            for i in range(L):
                ko.dual_gate_gemm(x[:, OFF:OFF + 2 * D], wfs[i], wgs[i])

        t0 = graph_time(run_stock, iters)
        t1 = graph_time(run_fused, iters)
        print(f"  timing M={m:2d}: {L} layers  stock {t0*1000/L:7.1f} us/layer  fused {t1*1000/L:7.1f} us/layer"
              f"  step delta {t1-t0:+.3f} ms")


# ---------------------------------------------------------------------------
# KDA one-pass
# ---------------------------------------------------------------------------
def _kda_fixture(n_req, lens, accs, layout, g, lines=48, H=16, D=128, W=4, S=8):
    proj = H * D
    XS = 3 * proj + H + 2 * D
    T_all = sum(lens)
    x = (torch.randn((T_all, XS), generator=g, device=DEV, dtype=torch.float32) * 0.7).to(torch.bfloat16)
    g1 = (torch.randn((T_all, proj), generator=g, device=DEV, dtype=torch.float32) * 1.0).to(torch.bfloat16)
    g2 = (torch.randn((T_all, proj), generator=g, device=DEV, dtype=torch.float32) * 1.0).to(torch.bfloat16)
    conv_w = torch.randn((3 * proj, W), generator=g, device=DEV, dtype=torch.float32) * 0.4
    state_len = W - 1 + (S - 1)
    if layout == "DS":
        conv_state = (torch.randn((lines, 3 * proj, state_len), generator=g, device=DEV) * 0.7).to(torch.bfloat16)
    else:
        conv_state = (torch.randn((lines, state_len, 3 * proj), generator=g, device=DEV) * 0.7).to(torch.bfloat16)
    rec = torch.randn((lines, H, D, D), generator=g, device=DEV, dtype=torch.float32) * 0.05
    # slots: request n owns lines [1 + n*S, 1 + (n+1)*S)  (0 = null block)
    idx = torch.zeros((n_req, S), dtype=torch.int32, device=DEV)
    for n in range(n_req):
        idx[n] = torch.arange(1 + n * S, 1 + (n + 1) * S, dtype=torch.int32)
    acc = torch.tensor(accs, dtype=torch.int32, device=DEV)
    cu = torch.zeros((n_req + 1,), dtype=torch.int32, device=DEV)
    cu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=DEV), 0)
    a_log = torch.randn((1, 1, H, 1), generator=g, device=DEV, dtype=torch.float32) * 0.5
    dt_bias = torch.randn((proj,), generator=g, device=DEV, dtype=torch.float32) * 0.5
    nw = (torch.rand((D,), generator=g, device=DEV, dtype=torch.float32) + 0.5).to(torch.bfloat16)
    return dict(x=x, g1=g1, g2=g2, conv_w=conv_w, conv_state=conv_state, rec=rec, idx=idx,
                acc=acc, cu=cu, a_log=a_log, dt_bias=dt_bias, nw=nw, H=H, D=D, S=S, W=W,
                n_req=n_req, T_all=T_all, proj=proj)


def _kda_stock(fx, conv_state, rec, out):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.kda import (
        fused_recurrent_kda,
        rms_norm_gated,
    )
    H, D, S, proj = fx["H"], fx["D"], fx["S"], fx["proj"]
    n = fx["n_req"]
    # causal_conv1d_update writes its output INTO the qkv columns (out=x),
    # exactly like the layer does to the merged projection; the fused kernel
    # reads the untouched rows, so the stock chain gets its own copy.
    x = fx["x"].clone()
    qkv = x[:, : 3 * proj]
    beta = x[:, 3 * proj: 3 * proj + H].unsqueeze(0)
    g1 = fx["g1"].reshape(1, -1, H, D)
    cs = conv_state if conv_state.shape[1] == 3 * proj else conv_state.transpose(-1, -2)
    qkv = causal_conv1d_update(
        qkv, cs, fx["conv_w"], None, activation="silu",
        conv_state_indices=fx["idx"][:, 0][:n], num_accepted_tokens=fx["acc"],
        query_start_loc=fx["cu"], max_query_len=S,
    )
    q, k, v = qkv.split(proj, dim=-1)

    def _r(t):
        return t.reshape(1, -1, H, D)

    fused_recurrent_kda(
        q=_r(q), k=_r(k), v=_r(v), g=g1, beta=beta, initial_state=rec,
        use_qk_l2norm_in_kernel=True, cu_seqlens=fx["cu"][: n + 1],
        ssm_state_indices=fx["idx"], num_accepted_tokens=fx["acc"],
        out=out.unsqueeze(0), sigmoid_beta=True, a_log=fx["a_log"],
        g_bias=fx["dt_bias"], compute_gate=True, lower_bound=-5.0,
    )
    y = rms_norm_gated(out, fx["g2"].reshape(-1, H, D), fx["nw"], None, "sigmoid", eps=1e-5)
    return y


def _kda_fused(fx, conv_state, rec, out, block_v=8):
    from vllm.model_executor.layers import glm53_kda_onepass as ko
    H, D, S = fx["H"], fx["D"], fx["S"]
    n = fx["n_req"]
    cs = conv_state if conv_state.shape[1] == 3 * fx["proj"] else conv_state.transpose(-1, -2)
    ok = ko.onepass_applicable(
        fx["x"], fx["g1"], fx["g2"], fx["conv_w"], None, cs, rec, fx["cu"], fx["idx"],
        fx["acc"], out, local_num_heads=H, head_dim=D, conv_size=fx["W"], max_query_len=S)
    if not ok:
        fail("one-pass not applicable on the fleet fixture")
        return out
    ko.kda_onepass_spec(
        fx["x"], fx["g1"], fx["g2"], fx["conv_w"], cs, rec, fx["cu"][: n + 1], fx["idx"],
        fx["acc"], fx["a_log"], fx["dt_bias"], fx["nw"], out,
        num_spec_decodes=n, local_num_heads=H, head_dim=D, max_query_len=S,
        lower_bound=-5.0, eps=1e-5, block_v=block_v)
    return out


def section_kda(iters: int) -> None:
    print("== kda: one-pass conv+recurrence+norm vs stock chain ==")
    g = torch.Generator(device=DEV).manual_seed(4242)
    cases = [
        ("C=1 uniform T=8 acc=8", 1, [8], [8], "SD"),
        ("C=1 uniform T=8 acc=1", 1, [8], [1], "SD"),
        ("C=1 uniform T=8 acc=3 DS", 1, [8], [3], "DS"),
        ("C=2 uniform T=8 acc=[1,3]", 2, [8, 8], [1, 3], "SD"),
        ("C=4 uniform T=8 acc=[8,3,1,5]", 4, [8, 8, 8, 8], [8, 3, 1, 5], "DS"),
        ("C=4 varlen T=[8,3,8,1] acc=[2,3,8,1]", 4, [8, 3, 8, 1], [2, 3, 8, 1], "SD"),
    ]
    for name, n, lens, accs, layout in cases:
        fx = _kda_fixture(n, lens, accs, layout, g)
        H, D, T_all = fx["H"], fx["D"], fx["T_all"]
        cs0, rec0 = fx["conv_state"].clone(), fx["rec"].clone()
        cs1, rec1 = fx["conv_state"].clone(), fx["rec"].clone()
        out0 = torch.zeros((T_all, H, D), dtype=torch.bfloat16, device=DEV)
        out1 = torch.zeros((T_all, H, D), dtype=torch.bfloat16, device=DEV)
        y0 = _kda_stock(fx, cs0, rec0, out0)
        y1 = _kda_fused(fx, cs1, rec1, out1)
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
        # a second launch on the same counters must behave the same (reset check)
        cs2, rec2 = fx["conv_state"].clone(), fx["rec"].clone()
        out2 = torch.zeros_like(out1)
        _kda_fused(fx, cs2, rec2, out2)
        torch.cuda.synchronize()
        if not (bits_equal(cs2, cs1) and bits_equal(rec2, rec1) and bits_equal(out2, out1)):
            fail(f"kda {name}: second launch differs from the first (counter reset?)")

    # timing: 34 layers, distinct weights/states, C=1 T=8
    L = 34
    fxs = [_kda_fixture(1, [8], [8], "SD", g, lines=16) for _ in range(L)]
    outs = [torch.zeros((8, 16, 128), dtype=torch.bfloat16, device=DEV) for _ in range(L)]

    def run_stock():
        for i in range(L):
            _kda_stock(fxs[i], fxs[i]["conv_state"], fxs[i]["rec"], outs[i])

    t0 = graph_time(run_stock, iters)
    print(f"  timing C=1 T=8: {L} layers  stock chain {t0*1000/L:7.1f} us/layer")
    for bv in (8, 16, 32):
        def run_fused(bv=bv):
            for i in range(L):
                _kda_fused(fxs[i], fxs[i]["conv_state"], fxs[i]["rec"], outs[i], block_v=bv)
        t1 = graph_time(run_fused, iters)
        print(f"  timing C=1 T=8: one-pass BV={bv:2d} {t1*1000/L:7.1f} us/layer  step delta {t1-t0:+.3f} ms")
    fx4 = [_kda_fixture(4, [8] * 4, [8, 3, 1, 5], "SD", g, lines=40) for _ in range(L)]
    outs4 = [torch.zeros((32, 16, 128), dtype=torch.bfloat16, device=DEV) for _ in range(L)]

    def run_stock4():
        for i in range(L):
            _kda_stock(fx4[i], fx4[i]["conv_state"], fx4[i]["rec"], outs4[i])

    def run_fused4():
        for i in range(L):
            _kda_fused(fx4[i], fx4[i]["conv_state"], fx4[i]["rec"], outs4[i])

    t0 = graph_time(run_stock4, iters)
    t1 = graph_time(run_fused4, iters)
    print(f"  timing C=4 T=8: {L} layers  stock {t0*1000/L:7.1f} us/layer  one-pass BV=8 {t1*1000/L:7.1f} us/layer"
          f"  step delta {t1-t0:+.3f} ms")


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
    if not torch.equal(kv0, kv1) or not bits_equal(tail0, tail1):
        fail("kpool int64 positions: cache writes differ from the int32 path")
    touched = int((kv0 != kv).any(dim=-1).sum()) + int((tail0 != tail).any(dim=-1).sum())
    print(f"  caches after update: {'bit-exact' if not FAILS or 'kpool' not in FAILS[-1] else 'DIFF'}"
          f" ({touched} rows written)")

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
    print(f"  timing 11 layers: cast+update {t0*1000/11:6.1f} us/layer  direct {t1*1000/11:6.1f} us/layer"
          f"  step delta {t1-t0:+.3f} ms")


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
