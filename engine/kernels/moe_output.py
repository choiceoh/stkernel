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
