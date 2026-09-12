"""Normalize, rotate and file every draft context layer in one launch.

The merged context projection already contains every layer. Read it where it
lies, preserve norm_rope's BF16 rounding, and write only committed positions.
No intermediate normalized key planes or per-layer reshape copies are needed.
"""
import torch
import triton as tr
import triton.language as tl

from .norm_rope import warm


@tr.jit
def _write_context(C, Weights, Inv, Positions, Slots, Valid, Field,
                   T: tl.constexpr, L: tl.constexpr, HK: tl.constexpr,
                   D: tl.constexpr, W: tl.constexpr, FIELD_HK: tl.constexpr,
                   SLOT_STRIDE: tl.constexpr, LAYER_STRIDE: tl.constexpr,
                   EPS: tl.constexpr):
    flat, lane = tl.program_id(0), tl.program_id(1)
    row, token = flat // T, flat % T
    layer, head = lane // HK, lane % HK
    if token < tl.load(Valid + row):
        offset = ((flat * L + layer) * 2 * HK + head) * D
        d = tl.arange(0, D)
        key = tl.load(C + offset + d).to(tl.float32)
        scale = tl.rsqrt(tl.sum(key * key) / D + EPS)
        half = tl.arange(0, D // 2)
        lo = tl.load(C + offset + half).to(tl.float32) * scale
        hi = tl.load(C + offset + D // 2 + half).to(tl.float32) * scale
        lo = (lo.to(Field.dtype.element_ty) * tl.load(Weights + layer * D + half)).to(tl.float32)
        hi = (hi.to(Field.dtype.element_ty) * tl.load(Weights + layer * D + D // 2 + half)).to(tl.float32)
        position = tl.load(Positions + flat)
        angle = position.to(tl.float32) * tl.load(Inv + half)
        cos, sin = tl.cos(angle), tl.sin(angle)
        slot = tl.load(Slots + row).to(tl.int64)
        dest = Field + slot * SLOT_STRIDE + layer * LAYER_STRIDE + (position % W * FIELD_HK + head) * D
        tl.store(dest + half, lo * cos - hi * sin)
        tl.store(dest + D // 2 + half, lo * sin + hi * cos)
        value = tl.load(C + offset + HK * D + d)
        tl.store(dest + W * FIELD_HK * D + d, value)


def write_context(field, slots, positions, context, weights, valid, eps, theta):
    """context [n,t,layers,2,local_kv,128] -> field [slots,layers,2,window,kv,128]."""
    if context.ndim != 6 or positions.ndim != 2 or field.ndim != 6:
        raise ValueError('draft context and field must be six-dimensional with [n,t] positions')
    n, t, layers, planes, heads, dim = context.shape
    if (planes != 2 or dim != 128 or field.shape[1:3] != (layers, 2) or field.shape[-1] != dim
            or heads > field.shape[-2] or t > field.shape[3] or positions.shape != (n, t)
            or slots.shape != (n,) or valid.shape != (n,) or weights.shape != (layers, dim)
            or any(x.dtype != torch.int64 for x in (slots, positions, valid))
            or any(x.dtype != torch.bfloat16 for x in (field, context, weights))
            or not field[0].is_contiguous()
            or any(not x.is_contiguous() for x in (context, weights, slots, positions, valid))
            or any(not x.is_cuda or x.device != context.device for x in (field, context, weights, slots, positions, valid))):
        raise ValueError('invalid contiguous CUDA BF16 draft context write')
    inverse = warm(context.device, dim, theta)
    if n and t:
        _write_context[(n * t, layers * heads)](context, weights, inverse, positions, slots, valid, field,
            t, layers, heads, dim, field.shape[3], field.shape[-2], field.stride(0), field.stride(1),
            float(eps), num_warps=4)
