"""Experimental FP8 all-gather packet to DeepGEMM activation conversion.

The ordinary consumer unpacks to BF16, then quantizes 128-column groups.
This kernel keeps that BF16 rounding in registers and emits the same FP8
values and power-of-two scales without a full BF16 intermediate. It does
not change the packet or communicate. The tiled KDA prefill consumer uses it
when no calibration observer needs the BF16 activation.
"""
import torch
import triton
import triton.language as tl

from engine.kernels.prefill_collectives import BLOCK
from engine.kernels.dense.mxfp8 import _publish, _rows, row_programs


@triton.jit(do_not_specialize=["LOCAL_N", "PAYLOAD_BYTES"])
def _quantize_gather(Packed, Scales, Q, S, LOCAL_N, PAYLOAD_BYTES,
                     K: tl.constexpr, G: tl.constexpr, PACK_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)*4 + tl.arange(0, 4)
    col = group[:, None]*128 + tl.arange(0, 128)[None, :]
    local_rows = LOCAL_N // K
    rank, local_row = row // local_rows, row % local_rows
    values = tl.load(Packed + rank*PAYLOAD_BYTES + local_row*K + col,
                     col < K, other=0.0).to(tl.float32)
    scale = tl.load(Scales + rank*(PAYLOAD_BYTES//4) + LOCAL_N//4
                    + (local_row*K + group*128)//PACK_BLOCK, group < G, other=0)
    # Preserve the baseline unpack's BF16 store/reload, including rounding
    # near underflow. A transport scale cannot replace the 128-column scale.
    x = (values*scale[:, None]).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    output_scale = tl.exp2(tl.ceil(tl.log2(amax/448.)))
    tl.store(Q + row*K + col, (x/output_scale[:, None]).to(tl.float8e4nv), col < K)
    tl.store(S + row*G + group, output_scale, group < G)


@triton.jit(do_not_specialize=['M', 'LOCAL_N', 'PAYLOAD_BYTES'])
def _quantize_gather_mx(Packed, Scales, Q, S, M, LOCAL_N, PAYLOAD_BYTES,
                        K: tl.constexpr, G: tl.constexpr, PACK_BLOCK: tl.constexpr, TILED: tl.constexpr):
    row = _rows(TILED)
    group = tl.program_id(1)
    col = group*128 + tl.arange(0, 128)
    rank, local_row = row // (LOCAL_N // K), row % (LOCAL_N // K)
    values = tl.load(Packed + rank[:, None]*PAYLOAD_BYTES + local_row[:, None]*K + col[None, :],
                     row[:, None] < M, other=0.).to(tl.float32)
    scale = tl.load(Scales + rank*(PAYLOAD_BYTES//4) + LOCAL_N//4
                    + (local_row*K + group*128)//PACK_BLOCK, row < M, other=1.)
    x = (values*scale[:, None]).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    output_scale = tl.exp2(tl.ceil(tl.log2(amax/448.)))
    tl.store(Q + row[:, None]*K + col[None, :],
             (x/output_scale[:, None]).to(tl.float8e4nv), row[:, None] < M)
    _publish(S, output_scale, row, group, M, G)


def quantize_gather(received, local_rows, *, real_rows=None, routed=False, mx=False, out=None):
    """Convert four rank-ordered packets to contiguous FP8 rows and FP32 scales.

`received` is the byte output of the existing FP8-v3 all-gather. Its owner
guarantees the packet values and scales obey that transport's contract.
"""
    if (type(local_rows) is not int or local_rows < 32 or received.ndim != 1
            or not received.is_cuda or received.dtype != torch.uint8 or not received.is_contiguous()):
        raise ValueError("consumer requires CUDA byte packets and at least 32 local rows")
    k, peers = 4096, 4
    local = local_rows*k
    stride = ((local + 4*(local//BLOCK) + 127)//128)*128
    rows = local_rows*peers if real_rows is None else real_rows
    if (type(rows) is not int or not (local_rows-1)*peers < rows <= local_rows*peers):
        raise ValueError('real rows must match the exact equal transport padding')
    if type(routed) is not bool:
        raise ValueError('routed packet ABI selection must be a boolean')
    if routed:
        from engine.modules.prefill_packets import PacketGeometry
        stride = PacketGeometry(rows, local_rows, routed=True).stride
    if received.numel() != peers*stride:
        raise ValueError("consumer packet length does not match four rank-ordered packets")
    from engine.kernels.dense.mxfp8 import buffers, scale_bytes
    if type(mx) is not bool:
        raise ValueError('MX output ABI selection must be a boolean')
    if out is not None:
        q, scales = out
    elif mx:
        q, scales = buffers(rows, k, received.device)
    else:
        q = torch.empty((rows, k), device=received.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty((rows, k//128), device=received.device, dtype=torch.float32)
    if (q.shape != (rows, k) or q.dtype != torch.float8_e4m3fn or q.device != received.device
            or scales.shape != ((scale_bytes(rows, k),) if mx else (rows, k//128))
            or scales.dtype != (torch.uint8 if mx else torch.float32) or scales.device != received.device
            or not q.is_contiguous() or not scales.is_contiguous()):
        raise ValueError('packet producer output buffers do not match their ABI')
    if out is not None:
        from engine.kernels.dense.fp8 import require_disjoint
        require_disjoint(received, q, scales)
    if mx:
        _quantize_gather_mx[(row_programs(rows), k//128)](
            received.view(torch.float8_e4m3fn), received.view(torch.float32), q, scales.view(torch.int32),
            rows, local, stride, k, k//128, BLOCK, rows >= 128, num_warps=4)
    else:
        _quantize_gather[(rows, triton.cdiv(k//128, 4))](
            received.view(torch.float8_e4m3fn), received.view(torch.float32), q, scales,
            local, stride, k, k//128, BLOCK, num_warps=4)
    return q, scales
