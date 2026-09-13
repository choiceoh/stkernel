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
