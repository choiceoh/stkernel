# SPDX-License-Identifier: Apache-2.0
"""Experimental causal-prefix MLA: adjacent queries share each FP8 KV tile.

The profile proves every visible pool is selected before calling this lane.
There is no union discovery, membership scratch, cache rewrite or collective.
BF16 queries, exact FP8-to-BF16 keys and BF16 probability operands follow the
served attention precision. MMA and online-softmax ordering still need GPU
numerical and consumer qualification. Serving enables this lane by explicit
operator choice; the experimental boot retains an off override.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['ROWS', 'CONTEXT'])
def _dense_prefix(Q, KV, Blocks, Out, ROWS, CONTEXT, SCALE: tl.constexpr,
                  KV_SCALE: tl.constexpr, BLOCK: tl.constexpr, STRIDE: tl.constexpr,
                  OFFSET: tl.constexpr, IDENTITY: tl.constexpr,
                  HEADS: tl.constexpr, DIM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    query = m // HEADS
    d = tl.arange(0, DIM)
    q = tl.load(Q + m[:, None] * DIM + d[None, :], query[:, None] < ROWS, other=0)
    group_end = CONTEXT + tl.minimum((tl.program_id(0) + 1) * (BM // HEADS), ROWS)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    result = tl.zeros((BM, DIM), tl.float32)
    for tile in range(tl.cdiv(group_end, BN)):
        # One descending KV tile shared by every head of 2/4 adjacent queries.
        # Each query masks the group's future positions independently.
        position = group_end - 1 - tile * BN - tl.arange(0, BN)
        safe = tl.maximum(position, 0).to(tl.int64)
        if IDENTITY:
            slot = safe
        else:
            block = tl.load(Blocks + safe // BLOCK, position >= 0, other=0).to(tl.int64)
            slot = block * STRIDE + OFFSET + safe % BLOCK
        key = tl.load(KV + slot[:, None] * DIM + d[None, :], position[:, None] >= 0, other=0.0).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(key)) * (SCALE * KV_SCALE)
        allowed = (position[None, :] >= 0) & (position[None, :] <= CONTEXT + query[:, None])
        scores = tl.where(allowed, scores, -float('inf'))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - next_maximum)
        probability = tl.exp(scores - next_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        result = result * correction[:, None]
        result = tl.dot((probability * KV_SCALE).to(tl.bfloat16), key, result)
        maximum = next_maximum
    result = result / denominator[:, None]
    tl.store(Out + m[:, None] * DIM + d[None, :], result.to(tl.bfloat16), query[:, None] < ROWS)


def mla_dense_prefix(q, latent, block_table, block_size, block_stride,
                     layer_offset, context, scale, ckv_scale, *, out=None):
    """Explicit eager GLM TP4 lane; the caller establishes top-k coverage."""
    rows, heads, dim = q.shape
    if (not 1 <= rows <= 2051 or context < 0 or context + rows > 2051
            or (heads, dim) != (16, 512) or q.dtype != torch.bfloat16
            or latent.ndim != 2 or latent.shape[1] != dim
            or latent.dtype != torch.float8_e4m3fn
            or block_size <= 0 or block_stride <= 0 or layer_offset < 0
            or not math.isfinite(scale) or scale <= 0
            or not math.isfinite(ckv_scale) or ckv_scale <= 0):
        raise ValueError('dense prefix requires bounded GLM BF16/FP8 geometry')
    tensors = (q, latent) + (() if block_table is None else (block_table,))
    if any(t.device != q.device or not t.is_contiguous() for t in tensors):
        raise ValueError('dense prefix requires contiguous tensors on one device')
    if block_table is not None and (block_table.ndim != 1 or block_table.dtype != torch.int32
            or block_table.numel() < triton.cdiv(context + rows, block_size)):
        raise ValueError('dense prefix requires a complete int32 page table')
    if block_table is None and latent.shape[0] < context + rows:
        raise ValueError('identity cache does not contain the visible prefix')
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype
            or out.device != q.device or not out.is_contiguous()):
        raise ValueError('dense prefix output must match the contiguous query geometry')
    if not q.is_cuda or torch.cuda.is_current_stream_capturing():
        raise ValueError('unqualified dense prefix is eager CUDA only')
    if out is None:
        out = torch.empty_like(q)
    _dense_prefix[(triton.cdiv(rows, 2),)](
        q, latent, q if block_table is None else block_table, out, rows, context,
        scale, ckv_scale, block_size, block_stride, layer_offset, block_table is None,
        heads, dim, 32, 32, num_warps=8, num_stages=1)
    return out
