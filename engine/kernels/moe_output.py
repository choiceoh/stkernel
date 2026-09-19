"""Decode MoE finalization, retaining both established BF16 boundaries.

The caller owns the FP32 scatter until this stream-ordered consumer finishes.
Packet finalization is fused separately into OneShot's existing exchange grid.
The captured TP4 decode path selects this finalizer by default.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _finish(Acc, Shared, Destination, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(Acc + i, i < COUNT, 0)
    # BF16(FP32 scatter) is a model boundary, even when no tensor stores it.
    a = a.to(tl.bfloat16).to(tl.float32)
    b = tl.load(Shared + i, i < COUNT, 0).to(tl.float32)
    value = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
        dtype=tl.float32, is_pure=True, pack=1).to(tl.bfloat16)
    tl.store(Destination + i, value, i < COUNT)


def _inputs(acc, shared):
    if (acc.ndim != 2 or not 1 <= acc.shape[0] <= 32 or acc.shape[1] != 4096
            or acc.dtype != torch.float32 or not acc.is_cuda or not acc.is_contiguous()
            or shared.shape != acc.shape or shared.dtype != torch.bfloat16
            or shared.device != acc.device or not shared.is_contiguous()):
        raise ValueError('MoE output requires matching contiguous CUDA FP32/BF16 [1..32,4096]')


def combine(acc, shared, *, out=None):
    """BF16(BF16(acc) + shared), in one launch and one BF16 destination."""
    _inputs(acc, shared)
    if out is None:
        out = torch.empty_like(shared)
    if (out.shape != shared.shape or out.dtype != shared.dtype
            or out.device != shared.device or not out.is_contiguous()
            or torch._C._overlaps(out, acc) or torch._C._overlaps(out, shared)):
        raise ValueError('MoE output needs a distinct contiguous BF16 destination')
    _finish[(tr.cdiv(acc.numel(), 256),)](acc, shared, out, acc.numel(), 256,
                                        num_warps=4, enable_fp_fusion=False)
    return out


@tr.jit
def _gated(Routed, Shared, Gate, Destination, sR, sS, sG, sD, H: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < H
    a = tl.load(Routed + r * sR + c, m, 0).to(tl.float32)
    b = tl.load(Shared + r * sS + c, m, 0).to(tl.float32)
    g = tl.load(Gate + r * sG)
    tl.store(Destination + r * sD + c, (a + b * g).to(Destination.dtype.element_ty), m)


def gated_sum(routed, shared, gate, *, out=None):
    """BF16(FP32(routed) + FP32(shared) * gate): Qwen3.8's MoE output, the routed partial plus the sigmoid-gated shared
    expert, in one launch instead of five (two widenings, the product, the sum, the rounding). routed and shared are
    BF16 [N, H] with packed columns, gate FP32 [N, 1]. The launch compiles without fused multiply-add, so the sum is the
    torch composition's; the served destination is BF16, and an FP32 `out` keeps the unrounded sum (the tests hold the
    arithmetic and the rounding apart: Triton's CPU interpreter does not round BF16 the way a GPU does)."""
    if (routed.ndim != 2 or shared.shape != routed.shape or routed.dtype != torch.bfloat16
            or shared.dtype != torch.bfloat16 or gate.shape != (routed.shape[0], 1) or gate.dtype != torch.float32
            or not (routed.device == shared.device == gate.device) or not routed.is_cuda
            or routed.stride(1) != 1 or shared.stride(1) != 1):
        raise ValueError('the gated MoE output takes BF16 routed and shared [N, H] with packed columns and an FP32 '
                         'gate [N, 1] on one CUDA device')
    if out is None:
        out = torch.empty_like(routed, memory_format=torch.contiguous_format)
    if (out.shape != routed.shape or out.dtype not in (torch.bfloat16, torch.float32) or out.device != routed.device
            or out.stride(1) != 1
            or torch._C._overlaps(out, routed) or torch._C._overlaps(out, shared) or torch._C._overlaps(out, gate)):
        raise ValueError('the gated MoE output needs a distinct BF16 (or FP32) destination with packed columns')
    rows, hidden = routed.shape
    if rows and hidden:
        block = 512
        _gated[(rows, tr.cdiv(hidden, block))](routed, shared, gate, out, routed.stride(0), shared.stride(0),
                                               gate.stride(0), out.stride(0), H=hidden, BLOCK=block, num_warps=4,
                                               enable_fp_fusion=False)
    return out


@tr.jit
def _lower_bound(Token, P, key):
    """The first index of `Token` [P] (ascending) not below `key`: a branchless binary search, 32 halvings."""
    lo = P * 0
    hi = P + 0
    for _ in tl.static_range(32):
        live = lo < hi
        mid = (lo + hi) // 2
        t = tl.load(Token + mid, mask=live, other=0)
        lo = tl.where(live & (t < key), mid + 1, lo)
        hi = tl.where(live & (t >= key), mid, hi)
    return lo


@tr.jit
def _pair_sum(Pairs, Token, Out, P, sP, sO, H: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < H
    lo = _lower_bound(Token, P, r)
    hi = _lower_bound(Token, P, r + 1)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for p in range(lo, hi):                                       # the row's pairs in their order: one fp32 sum
        acc += tl.load(Pairs + p * sP + c, mask=m, other=0.0).to(tl.float32)   # 32-bit offsets: the entry bounds them
    tl.store(Out + r * sO + c, acc.to(Out.dtype.element_ty), mask=m)


def pair_sum(pairs, token, rows: int, *, out=None):
    """Each row's pairs summed in FP32 in their order and rounded once: out[r] = BF16(sum of pairs[p] with token[p] == r)
    -- the compact MoE's combine (engine/profiles/qwen38/lanes: an eager step dispatches only this rank's (token,
    route) pairs, one route a pair), in one launch instead of an FP32 zero fill, the pairs widened to FP32, an atomic
    `index_add_` and the rounding. `token` is ascending int64 [P] (torch.nonzero's row-major order), `pairs` BF16 [P, H]
    with packed columns; a row without pairs is zeros. The order is the pairs' own, so the result is one value
    whatever the scheduling -- the atomic sum it replaces could differ in its last place from run to run -- and it is
    the sequential `index_add_` of the CPU's. An FP32 `out` keeps the sum unrounded (the tests hold the arithmetic
    apart: Triton's CPU interpreter does not round BF16 the way a GPU does)."""
    if (pairs.ndim != 2 or token.ndim != 1 or token.shape[0] != pairs.shape[0] or token.dtype != torch.int64
            or pairs.dtype != torch.bfloat16 or pairs.device != token.device or not pairs.is_cuda
            or pairs.stride(1) != 1 or not token.is_contiguous() or type(rows) is not int or rows < 0
            or max(pairs.shape[0] * pairs.stride(0), rows * pairs.shape[1]) >= 2 ** 31):
        raise ValueError('the pair sum takes BF16 pairs [P, H] with packed columns, their ascending int64 rows [P] '
                         'on the same CUDA device, and a row count, every offset within 32 bits (a 32K-token chunk, '
                         'ten routes a token at hidden 2560, is 0.84e9)')
    hidden = pairs.shape[1]
    if out is None:
        out = torch.empty(rows, hidden, device=pairs.device, dtype=pairs.dtype)
    if (out.shape != (rows, hidden) or out.dtype not in (torch.bfloat16, torch.float32) or out.device != pairs.device
            or out.stride(1) != 1 or torch._C._overlaps(out, pairs)):
        raise ValueError('the pair sum needs a distinct BF16 (or FP32) [rows, H] destination with packed columns')
    if rows and hidden:
        block = 512
        _pair_sum[(rows, tr.cdiv(hidden, block))](pairs, token, out, pairs.shape[0], pairs.stride(0), out.stride(0),
                                                 H=hidden, BLOCK=block, num_warps=4)
    return out


def validate_finalizer(finalize, *, rows, experts, local_experts, hidden,
                       intermediate, topk, quant_mode, activation, limit,
                       alpha, beta, tiled):
    """Refuse an incompatible path before any kernel can touch its destination."""
    if finalize is None:
        return
    if (not callable(finalize) or type(rows) is not int or not 1 <= rows <= 32
            or (experts, local_experts, hidden, intermediate, topk) != (288, 288, 4096, 512, 8)
            or (quant_mode, activation, limit, alpha, beta)
                != ('nvfp4', 'swigluoai_uninterleave', 10., 1., 0.) or not tiled):
        raise ValueError('MoE finalizer requires the explicit tiled TP4 decode FP32 scatter contract')
