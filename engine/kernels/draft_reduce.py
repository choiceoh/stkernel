"""Consume one-shot rank packets at the drafter's post-convolution boundary.

The exchange still publishes and waits on the existing stream. This reader
uses canonical rank order and rounds the sum to BF16 before any convolution.
Remote ring loads bypass L1, including when a captured graph reuses addresses.
No transport state, flags, or collectives are changed here.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _packet_mix(PACKETS, DELTA, BASE, r, c, live,
                WIDTH: tl.constexpr, BLOCK: tl.constexpr, GROUP: tl.constexpr, T: tl.constexpr,
                DR: tl.constexpr, DT: tl.constexpr, DG: tl.constexpr):
    p0 = tl.load(PACKETS + 0, cache_modifier='.cg').to(tl.pointer_type(tl.bfloat16))
    p1 = tl.load(PACKETS + 1, cache_modifier='.cg').to(tl.pointer_type(tl.bfloat16))
    p2 = tl.load(PACKETS + 2, cache_modifier='.cg').to(tl.pointer_type(tl.bfloat16))
    p3 = tl.load(PACKETS + 3, cache_modifier='.cg').to(tl.pointer_type(tl.bfloat16))
    acc = tl.full(c.shape, 0, tl.float32)
    for tap in tl.static_range(T):
        offset = (r - tap) * WIDTH + c
        mask = live & (r % BLOCK >= tap)
        a = tl.load(p0 + offset, mask, 0, cache_modifier='.cg').to(tl.float32)
        b = tl.load(p1 + offset, mask, 0, cache_modifier='.cg').to(tl.float32)
        d = tl.load(p2 + offset, mask, 0, cache_modifier='.cg').to(tl.float32)
        e = tl.load(p3 + offset, mask, 0, cache_modifier='.cg').to(tl.float32)
        value = tl.inline_asm_elementwise(
            '{ .reg .f32 sum; add.rn.f32 sum, $1, $2; add.rn.f32 sum, sum, $3; '
            'add.rn.f32 sum, sum, $4; mov.f32 $0, sum; }',
            constraints='=f,f,f,f,f', args=[a, b, d, e], dtype=tl.float32, is_pure=True, pack=1)
        # This is the store rounding of the ordinary all-reduce, not a
        # reassociation of a rank sum with the coefficient multiplication.
        value = value.to(tl.bfloat16).to(tl.float32)
        coeff = (tl.load(BASE + tap * WIDTH + c, live, 0).to(tl.float32)
                 + tl.load(DELTA + r * DR + tap * DT + c // GROUP * DG, live, 0).to(tl.float32)
                 ).to(tl.bfloat16).to(tl.float32)
        acc += coeff * value
    return acc.to(tl.bfloat16)


@triton.jit
def _mix(PACKETS, DELTA, BASE, OUT, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
         GROUP: tl.constexpr, T: tl.constexpr, DR: tl.constexpr, DT: tl.constexpr, DG: tl.constexpr,
         BC: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mixed = _packet_mix(PACKETS, DELTA, BASE, r, c, c < WIDTH, WIDTH, BLOCK, GROUP, T, DR, DT, DG)
    tl.store(OUT + r * WIDTH + c, mixed, c < WIDTH)


@triton.jit
def _mix_add_norm(PACKETS, DELTA, BASE, RES, W, TOTAL, OUT,
                  WIDTH: tl.constexpr, BLOCK: tl.constexpr, GROUP: tl.constexpr, T: tl.constexpr,
                  DR: tl.constexpr, DT: tl.constexpr, DG: tl.constexpr, SR: tl.constexpr,
                  EPS: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BC)
    live = c < WIDTH
    mixed = _packet_mix(PACKETS, DELTA, BASE, r, c, live, WIDTH, BLOCK, GROUP, T, DR, DT, DG)
    total = (tl.load(RES + r * SR + c, live, 0).to(tl.float32) + mixed.to(tl.float32)).to(tl.bfloat16)
    tl.store(TOTAL + r * WIDTH + c, total, live)
    value = total.to(tl.float32)
    scale = tl.rsqrt(tl.sum(value * value) / WIDTH + EPS)
    weight = tl.load(W + c, live, 0)
    tl.store(OUT + r * WIDTH + c, (value * scale).to(tl.bfloat16) * weight, live)


def _check(source, packets, delta, base, group, block):
    if (source.ndim != 2 or source.shape[1] != 4096 or not 1 <= source.shape[0] <= 32
            or not source.is_cuda or source.dtype != torch.bfloat16 or not source.is_contiguous()
            or source.data_ptr() % 16 or packets.shape != (4,) or packets.dtype != torch.int64
            or packets.device != source.device or not packets.is_contiguous()):
        raise ValueError('draft reduction requires TP4 CUDA BF16 [1..32,4096] and four rank pointers')
    rows, width = source.shape
    if (type(group) is not int or group <= 0 or width % group or type(block) is not int
            or block <= 0 or rows % block or delta.ndim != 3 or delta.shape[0] != rows
            or delta.shape[1] < 1 or delta.shape[2] != width // group
            or base.shape != (delta.shape[1], width) or not base.is_contiguous()
            or any(x.dtype != source.dtype or x.device != source.device for x in (delta, base))):
        raise ValueError('draft reduction needs whole tap blocks and same-device BF16 coefficients')
    return rows, width, delta.shape[1]


def packet_tap_mix(source, packets, delta, base, group, block):
    """Terminal MLP: leave the existing head's residual/norm/MX packing intact."""
    rows, width, taps = _check(source, packets, delta, base, group, block)
    out = torch.empty_like(source)
    _mix[(rows, triton.cdiv(width, 512))](packets, delta, base, out, width, block, group,
                                       taps, *delta.stride(), 512, num_warps=4)
    return out


def packet_tap_add_norm(source, packets, delta, base, residual, weight, eps, group, block):
    """Nine nonterminal boundaries: rank sum, taps, residual and RMS together."""
    rows, width, taps = _check(source, packets, delta, base, group, block)
    if (residual.shape != source.shape or residual.stride(1) != 1 or weight.shape != (width,)
            or not weight.is_contiguous() or not 0 < eps < float('inf')
            or any(x.dtype != source.dtype or x.device != source.device for x in (residual, weight))):
        raise ValueError('draft reduction norm requires matching residual, weight and finite epsilon')
    total, out = torch.empty_like(source), torch.empty_like(source)
    _mix_add_norm[(rows,)](packets, delta, base, residual, weight, total, out, width, block, group,
                           taps, *delta.stride(), residual.stride(0), eps,
                           triton.next_power_of_2(width), num_warps=8)
    return total, out
