"""Experimental FP8 all-gather packet to DeepGEMM activation conversion.

The ordinary consumer unpacks to BF16, then quantizes 128-column groups.
This kernel keeps that BF16 rounding in registers and emits the same FP8
values and power-of-two scales without a full BF16 intermediate. It does
not change the packet or communicate, and is not connected to serving yet.
"""
import torch
import triton
import triton.language as tl

from . import BLOCK


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


def quantize_gather(received, local_rows):
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
    if received.numel() != peers*stride:
        raise ValueError("consumer packet length does not match four rank-ordered packets")
    q = torch.empty((local_rows*peers, k), device=received.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((local_rows*peers, k//128), device=received.device, dtype=torch.float32)
    _quantize_gather[(local_rows*peers, triton.cdiv(k//128, 4))](
        received.view(torch.float8_e4m3fn), received.view(torch.float32), q, scales,
        local, stride, k, k//128, BLOCK, num_warps=4)
    return q, scales
