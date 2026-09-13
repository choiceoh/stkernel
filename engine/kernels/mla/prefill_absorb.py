# SPDX-License-Identifier: Apache-2.0
"""Experimental MLA contractions that write token-major outputs directly.

The existing einsums produce head-major storage, then both consumers copy it
to token-major storage. These GEMMs keep BF16 operands, FP32 accumulation and
the two BF16 output boundaries. GPU rounding and throughput remain unqualified.
Weights are read from the existing kv_b slices without a persistent repack.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['ROWS'])
def _absorb(X, W, Y, ROWS, HEADS: tl.constexpr, INPUT: tl.constexpr,
            OUTPUT: tl.constexpr, WH: tl.constexpr, WR: tl.constexpr,
            TRANSPOSE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr):
    m = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.program_id(2)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(INPUT, BK)):
        ki = start * BK + k
        x = tl.load(X + (m[:, None] * HEADS + h) * INPUT + ki[None, :],
                    (m[:, None] < ROWS) & (ki[None, :] < INPUT), other=0)
        if TRANSPOSE:
            wi = h * WH + n[None, :] * WR + ki[:, None]
        else:
            wi = h * WH + ki[:, None] * WR + n[None, :]
        w = tl.load(W + wi, (ki[:, None] < INPUT) & (n[None, :] < OUTPUT), other=0)
        acc = tl.dot(x, w, acc)
    tl.store(Y + (m[:, None] * HEADS + h) * OUTPUT + n[None, :],
             acc.to(tl.bfloat16), (m[:, None] < ROWS) & (n[None, :] < OUTPUT))


def mla_prefill_absorb(x, weight, *, transpose=False):
    """[T,16,256] @ W or [T,16,512] @ W.T -> fresh contiguous BF16.

    This explicit opt-in lane is limited to eager prefill. The two weight
    slices have shape [16,256,512], including the output slice's storage offset.
    """
    if type(transpose) is not bool:
        raise ValueError('absorb transpose must be boolean')
    inner, outer = (512, 256) if transpose else (256, 512)
    if (x.ndim != 3 or not 128 <= x.shape[0] <= 32768
            or x.shape[1:] != (16, inner) or weight.shape != (16, 256, 512)
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not x.is_contiguous() or weight.stride(2) != 1
            or weight.stride(1) != 512 or weight.stride(0) < 256 * 512
            or x.device != weight.device):
        raise ValueError('absorb requires bounded contiguous GLM BF16 inputs and kv_b slices')
    if not x.is_cuda or torch.cuda.is_current_stream_capturing():
        raise ValueError('unqualified absorb tiles are eager CUDA only')
    out = torch.empty((x.shape[0], 16, outer), dtype=x.dtype, device=x.device)
    _absorb[(triton.cdiv(x.shape[0], 64), triton.cdiv(outer, 64), 16)](
        x, weight, out, x.shape[0], 16, inner, outer, weight.stride(0), weight.stride(1),
        transpose, 64, 64, 32, num_warps=4, num_stages=2)
    return out
