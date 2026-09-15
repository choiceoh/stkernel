"""FP32 prefill router adapters, including the TP4 FP8 transport boundary.

Projection always uses the same IEEE FP32 operator as decode. Packet inputs
retain the transport's BF16 rounding before promotion; FP32 routing does not
change the values sent to the experts.
"""
import torch
import triton
import triton.language as tl

from engine.kernels.router_fp32 import router_logits as project


@triton.jit(do_not_specialize=['M', 'LOCAL_ROWS', 'PACKET_BYTES'])
def _packet_input(X, Scales, Out, M, LOCAL_ROWS, PACKET_BYTES, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // 4096
    rank, local_row = row // LOCAL_ROWS, row % LOCAL_ROWS
    local_offset = local_row * 4096 + offset % 4096
    q = tl.load(X + rank * PACKET_BYTES + local_offset, row < M, other=0.).to(tl.float32)
    scale = tl.load(Scales + rank * (PACKET_BYTES // 4) + LOCAL_ROWS * 1024
                    + local_offset // 2048, row < M, other=0.)
    # The ordinary gather stores BF16. Preserve that exact boundary, then
    # write FP32 operands directly without a full received BF16 allocation.
    value = (q * scale).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + offset, value, row < M)


def router_logits(x, weight):
    """Eligible long-prefill adapter; the network uses the common FP32 entry."""
    if (x.ndim != 2 or weight.ndim != 2 or not 8192 < x.shape[0] <= 32768
            or x.shape[1] != 4096 or tuple(weight.shape) != (288,4096)
            or not x.is_cuda or not weight.is_cuda or x.device != weight.device
            or x.dtype != torch.bfloat16 or weight.dtype != torch.float32
            or not x.is_contiguous() or not weight.is_contiguous()):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    return project(x, weight)


def router_shard_logits(x, weight):
    """Project one sender's BF16 transport roundtrip using resident FP32 gates."""
    if (x.ndim != 2 or weight.ndim != 2 or not 2049 <= x.shape[0] <= 8192 or x.shape[1] != 4096
            or tuple(weight.shape) != (288,4096) or not x.is_cuda or not weight.is_cuda
            or x.device != weight.device or x.dtype != torch.bfloat16
            or weight.dtype != torch.float32 or not x.is_contiguous()
            or not weight.is_contiguous() or torch.cuda.is_current_stream_capturing()):
        raise ValueError('sender router requires a BF16 TP4 prefill roundtrip shard and FP32 gate')
    return project(x, weight)


def router_packet_logits(batch, weight):
    from engine.modules.prefill_packets import PacketBatch, ffn_packet_rows
    if (not isinstance(batch, PacketBatch) or not ffn_packet_rows(batch.geometry.rows)
            or tuple(weight.shape) != (288,4096) or not weight.is_cuda
            or weight.device != batch.received.device or weight.dtype != torch.float32
            or not weight.is_contiguous()):
        raise ValueError('packet router requires the long-prefill TP4/H4096/E288 contract and FP32 gate')
    g, x = batch.geometry, batch.received
    promoted = torch.empty((g.rows,4096), device=x.device, dtype=torch.float32)
    _packet_input[(triton.cdiv(promoted.numel(),2048),)](
        x.view(torch.float8_e4m3fn), x.view(torch.float32), promoted, g.rows,
        g.local_rows, g.stride, BLOCK=2048, enable_fp_fusion=False)
    return project(promoted, weight)
