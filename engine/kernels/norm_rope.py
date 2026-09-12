"""RMS norm, and RMS norm followed by rotary, in one launch each.

Every rotary call in the drafter is `rope(rmsnorm(x, w, eps), positions, theta)` -- ten sites, one shape family
[N, heads, D] -- and every one of them issued about twenty-one device operations on tensors holding a thousand
numbers. Worse, `rope` rebuilt its inverse-frequency table on every call: an arange, a divide, a pow and a
reciprocal for sixty-four constants of the model. A decode step makes fifty-five of these calls (five in the
drafter's observe, ten a block through the five layers and five blocks a proposal), so the step carried about
eleven hundred launches of arithmetic that fits in one.

The table is now built once per (device, dim, theta) and passed in, which also makes the angle bit-identical to
the torch path: the same fp32 `position * inv` multiply. The norm's reduction order is the kernel's own, so the
last bit of the normalised value can differ from torch's -- measured below one bf16 ulp on every element.

Under capture the table must already exist. Building it inside a graph would put a model constant in that
graph's private pool, to be freed when the graph closes; `warm` is called while the packs are prepared, and a
missing table during capture is an error rather than a quiet allocation (D3).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_TABLES: "dict[tuple, torch.Tensor]" = {}


def warm(device, dim: int, theta: float) -> torch.Tensor:
    """The inverse frequencies for a `dim`-wide head at `theta`, built once. Call before capture.

    The key carries the device index a tensor would report, so a caller that warms "cuda" and a tensor that
    lives on "cuda:0" share one table rather than each building its own."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (str(device), int(dim), float(theta))
    table = _TABLES.get(key)
    if table is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("rotary table must be warmed before capture, not allocated in a graph's pool")
        table = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        _TABLES[key] = table
    return table


@triton.jit
def _add_norm(A, B, W, SUM, OUT, sA, sB, sS, sO, EPS, D: tl.constexpr, BD: tl.constexpr):
    """The residual join and the norm that reads it, in one launch.

    A block writes `res = res + x` and then normalises `res`, twice a layer. Split, that is two launches over
    the same row and one of them exists only to hand the other its input."""
    r = tl.program_id(0)
    d = tl.arange(0, BD)
    m = d < D
    total = (tl.load(A + r * sA + d, mask=m, other=0.0).to(tl.float32)
             + tl.load(B + r * sB + d, mask=m, other=0.0).to(tl.float32)).to(SUM.dtype.element_ty)
    tl.store(SUM + r * sS + d, total, mask=m)
    x = total.to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / D + EPS)
    w = tl.load(W + d, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r * sO + d, (x * scale).to(OUT.dtype.element_ty) * w.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _norm(X, W, OUT, sX, sO, EPS, D: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BD)
    m = d < D
    x = tl.load(X + r * sX + d, mask=m, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / D + EPS)
    w = tl.load(W + d, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r * sO + d, (x * scale).to(OUT.dtype.element_ty) * w.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _norm_rope(X, W, POS, INV, OUT, sXr, sXh, sO, EPS, D: tl.constexpr, H: tl.constexpr, BH: tl.constexpr):
    r, h = tl.program_id(0), tl.program_id(1)
    base, out = X + r * sXr + h * sXh, OUT + r * sO + h * D
    head = tl.load(base + tl.arange(0, D)).to(tl.float32)
    scale = tl.rsqrt(tl.sum(head * head) / D + EPS)

    i = tl.arange(0, BH)
    m = i < H
    # the halves separately, each normalised and weighted exactly as the torch form does it: the bf16 round
    # happens after the scale and before the weight, and the rotation reads the rounded value
    lo = tl.load(base + i, mask=m, other=0.0).to(tl.float32) * scale
    hi = tl.load(base + H + i, mask=m, other=0.0).to(tl.float32) * scale
    lo = (lo.to(OUT.dtype.element_ty) * tl.load(W + i, mask=m, other=0.0)).to(tl.float32)
    hi = (hi.to(OUT.dtype.element_ty) * tl.load(W + H + i, mask=m, other=0.0)).to(tl.float32)

    angle = tl.load(POS + r).to(tl.float32) * tl.load(INV + i, mask=m, other=0.0)
    cos, sin = tl.cos(angle), tl.sin(angle)
    tl.store(out + i, (lo * cos - hi * sin).to(OUT.dtype.element_ty), mask=m)
    tl.store(out + H + i, (lo * sin + hi * cos).to(OUT.dtype.element_ty), mask=m)


def norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm over the last dimension, `w` applied in the input's dtype exactly as the torch form does."""
    if w.ndim != 1 or x.shape[-1] != w.shape[0]:
        raise ValueError("rms norm weight must be one row matching the input's last dimension")
    if not x.is_cuda:
        return _norm_by_torch(x, w, eps)
    flat = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(flat)
    D = flat.shape[1]
    if flat.shape[0]:
        _norm[(flat.shape[0],)](flat, w, out, flat.stride(0), out.stride(0), eps,
                                D=D, BD=triton.next_power_of_2(D), num_warps=4 if D <= 1024 else 8)
    return out.view_as(x)


def add_norm(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, eps: float):
    """`total = a + b` and `norm(total, w, eps)`, returned together in one launch."""
    if a.shape != b.shape or w.ndim != 1 or a.shape[-1] != w.shape[0]:
        raise ValueError("the residual join takes two tensors of one shape and a weight for their last dimension")
    if not a.is_cuda:
        total = a + b
        return total, _norm_by_torch(total, w, eps)
    flat_a, flat_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    total, out = torch.empty_like(flat_a), torch.empty_like(flat_a)
    D = flat_a.shape[1]
    if flat_a.shape[0]:
        _add_norm[(flat_a.shape[0],)](flat_a, flat_b, w, total, out, flat_a.stride(0), flat_b.stride(0),
                                      total.stride(0), out.stride(0), eps,
                                      D=D, BD=triton.next_power_of_2(D), num_warps=4 if D <= 1024 else 8)
    return total.view_as(a), out.view_as(a)


def norm_rope(x: torch.Tensor, w: torch.Tensor, eps: float, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """`rope(rmsnorm(x, w, eps), positions, theta)` for x [N, heads, D] at absolute `positions` [N]."""
    if x.ndim != 3 or w.ndim != 1 or w.shape[0] != x.shape[-1] or positions.shape[0] != x.shape[0]:
        raise ValueError("norm_rope takes x [N, heads, D], w [D] and positions [N]")
    if x.shape[-1] % 2 or x.shape[-1] & (x.shape[-1] - 1):
        raise ValueError("rotary needs a power-of-two head dimension")
    if not x.is_cuda:
        return _norm_rope_by_torch(x, w, eps, positions, theta)
    rows, heads, D = x.shape
    # The heads arrive as a slice of a fused projection (`qkv.split(...)` reshaped), whose rows are wider than
    # the heads read here. The kernel takes both strides, so the step does not copy them into place first.
    src = x if x.stride(2) == 1 else x.contiguous()
    out = torch.empty(rows, heads, D, device=x.device, dtype=x.dtype)
    inv = warm(x.device, D, theta)
    pos = positions.contiguous()
    if rows and heads:
        _norm_rope[(rows, heads)](src, w, pos, inv, out, src.stride(0), src.stride(1), out.stride(0), eps,
                                  D=D, H=D // 2, BH=triton.next_power_of_2(D // 2), num_warps=4)
    return out


# -- the torch forms, kept for the CPU tests and as the reference the kernel is judged against ------------
def _norm_by_torch(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def _norm_rope_by_torch(x, w, eps, positions, theta):
    h = _norm_by_torch(x, w, eps)
    d = h.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=h.device, dtype=torch.float32) / d))
    ang = positions.float()[:, None] * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = h[..., : d // 2].float(), h[..., d // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(h.dtype)
