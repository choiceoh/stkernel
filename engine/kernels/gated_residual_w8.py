"""Experimental two-launch W8A16 mixer for Qwen's 1..16-row sites.

Only weights are block-128 FP8; inputs and intermediate rounding remain BF16.
This changes model arithmetic and stays opt-in until TP4 output-quality proof.
The GB10 component record is measurements/qwen38_mix_w8_20260920.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels import gated_residual as hcr
from engine.kernels.common import skinny_gemv as sg

@triton.jit
def _dot(X, W, WS, sx, sw, ss, rows, cols, M, N, k0, k1,
         BN: tl.constexpr, BK: tl.constexpr, FP32_DOT: tl.constexpr):
    acc = tl.zeros((16, BN), dtype=tl.float32)
    for k in range(k0, k1, BK):
        ks = k + tl.arange(0, BK)
        km = ks < k1
        x = tl.load(X + rows[:, None] * sx + ks[None, :],
                    mask=(rows[:, None] < M) & km[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * sw + ks[None, :],
                    mask=(cols[:, None] < N) & km[None, :], other=0.0).to(tl.bfloat16)
        # BK=128 never crosses a scale block. Scale the dot, rather than each
        # weight element: the first prototype's elementwise expansion was 2x slower.
        scale = tl.load(WS + (cols // 128) * ss + k // 128, mask=cols < N, other=0.0)
        if FP32_DOT:
            x, w = x.to(tl.float32), w.to(tl.float32)
        acc += tl.dot(x, tl.trans(w)) * scale[None, :]
    return acc


@triton.jit
def _down(X, W, WS, MIX, INJ, PART, LOCKS, M, N, K, sx, sw, ss, sm, si, HC_F,
          R: tl.constexpr, HC: tl.constexpr, INJECT: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, FP32_DOT: tl.constexpr):
    pn, pk = tl.program_id(0), tl.program_id(1)
    rows, cols = tl.arange(0, 16), pn * BN + tl.arange(0, BN)
    k0, k1 = sg.split_span(K, pk, SPLIT, BK)
    acc = _dot(X, W, WS, sx, sw, ss, rows, cols, M, N, k0, k1, BN, BK, FP32_DOT)
    if SPLIT > 1:
        total, last = sg.split_sum(acc, PART, LOCKS, pn, pk, rows, cols, M, N, SPLIT)
        if last:
            hcr._gate_store(total, rows, cols, M, MIX, INJ, sm, si, HC_F, R, HC, INJECT)
    else:
        hcr._gate_store(acc, rows, cols, M, MIX, INJ, sm, si, HC_F, R, HC, INJECT)


@triton.jit
def _up(G, W, WS, NORMED, OUT, M, sg_, sw, ss, sn, so, HC_F,
        HID: tl.constexpr, R: tl.constexpr, HC: tl.constexpr,
        BD: tl.constexpr, BK: tl.constexpr, FP32_DOT: tl.constexpr):
    rows = tl.arange(0, 16)
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    live = (rows[:, None] < M) & (d[None, :] < HID)
    acc = tl.zeros((16, BD), dtype=tl.float32)
    for s in tl.static_range(HC):
        u = _dot(G, W, WS, sg_, sw, ss, rows, s * HID + d, M, (s + 1) * HID,
                 0, R, BD, BK, FP32_DOT)
        g = tl.sigmoid(u.to(OUT.dtype.element_ty).to(tl.float32)).to(OUT.dtype.element_ty).to(tl.float32)
        n = tl.load(NORMED + rows[:, None] * sn + (s * HID + d)[None, :], mask=live, other=0.0).to(tl.float32)
        acc += (g * n).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + rows[:, None] * so + d[None, :], (acc / HC_F).to(OUT.dtype.element_ty), mask=live)


def mix(x, down, up, hc, rank, *, inject=True):
    """Packed weights (e4m3 matrix, block-128 scales); all row values remain BF16."""
    rows, width = x.shape
    if not 1 <= rows <= 16 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("requires 1..16 contiguous BF16 rows")
    if (width, hc, rank) != (10240, 4, 320):
        raise ValueError("W8 mixer is qualified only for Qwen hc=4, hidden=2560, rank=320")
    for (w, s), shape in zip((down, up), ((384, 10240), (10240, 384))):
        if (tuple(w.shape) != shape or tuple(s.shape) != (shape[0] // 128, shape[1] // 128)
                or w.dtype != torch.float8_e4m3fn or s.dtype != torch.float32
                or w.device != x.device or s.device != x.device or not w.is_contiguous() or not s.is_contiguous()):
            raise ValueError("W8 mixer weights must be packed e4m3 with contiguous FP32 block scales")
    n = rank + (hc if inject else 0)
    bn, bk, split, warps, stages = (16, 128, 8, 4, 3)
    w, ws = down
    gates = torch.empty(rows, rank, device=x.device, dtype=x.dtype)
    inj = torch.empty(rows, hc, device=x.device, dtype=x.dtype) if inject else gates
    partial = torch.empty(split, rows, n, device=x.device, dtype=torch.float32)
    locks = sg.prepare(x.device)
    _down[(triton.cdiv(n, bn), split)](
        x, w, ws, gates, inj, partial, locks, rows, n, width, x.stride(0), w.stride(0), ws.stride(0),
        gates.stride(0), inj.stride(0), float(hc), R=rank, HC=hc, INJECT=inject,
        BN=bn, BK=bk, SPLIT=split, FP32_DOT=not x.is_cuda, num_warps=warps, num_stages=stages)
    out = torch.empty(rows, width // hc, device=x.device, dtype=x.dtype)
    w, ws = up
    bd, bk, warps, stages = (32, 128, 4, 3)
    _up[(triton.cdiv(width // hc, bd),)](
        gates, w, ws, x, out, rows, gates.stride(0), w.stride(0), ws.stride(0), x.stride(0), out.stride(0),
        float(hc), HID=width // hc, R=rank, HC=hc, BD=bd, BK=bk, FP32_DOT=not x.is_cuda,
        num_warps=warps, num_stages=stages)
    return out, inj if inject else None



def pack(w):
    """Pack outside capture; pad only stored weights, never the live input rows."""
    from deep_gemm import per_block_cast_to_fp8
    n, k = w.shape
    if w.dtype != torch.bfloat16 or (n, k) not in ((320, 10240), (324, 10240), (10240, 320)):
        raise ValueError("W8 mixer requires Qwen BF16 down(+inject) or up weights")
    padded = torch.nn.functional.pad(w, (0, -k % 128, 0, -n % 128))
    q, s = per_block_cast_to_fp8(padded.float(), use_ue8m0=True)
    return q.contiguous(), s.contiguous()


def qualify(down, up, *, hc=4, rank=320, inject=True):
    """Boot check against the FP32-scaled weights widened to BF16, not a quality verdict."""
    def widen(weight, n, k):
        q, s = weight
        return (q.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()[:n, :k].contiguous()
    df = widen(down, rank + (hc if inject else 0), 10240)
    uf = widen(up, 10240, rank)
    gen = torch.Generator().manual_seed(7)
    worst = 0.0
    for rows in (1, 4, 16):
        x = torch.randn(rows, 10240, generator=gen).bfloat16().to(df.device)
        got = mix(x, down, up, hc, rank, inject=inject)
        ref = hcr.mix_rows(x, df, uf, hc, inject=inject)
        for a, b in zip(got, ref):
            if a is None:
                continue
            error = hcr.drift(a, b)[0]
            if not bool(torch.isfinite(a).all()) or not error <= 2 ** -7:
                raise RuntimeError(f"W8 mixer recipe error at {rows} rows: {error}")
            worst = max(worst, error)
    return worst
