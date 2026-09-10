"""Sparse attention over separate window and compressed BF16 KV pools.

The reference's merged index domain is retained: [0, window_width) addresses
the window, and subsequent positions address compressed KV. Only 64 selected
rows per query tile are gathered; neither pool is concatenated or copied in
full. The inputs are the reference's already quantized/dequantized BF16 values.

This preserves the reference's 64-slot online-softmax recurrence, including
BF16 probability rounding before the PV product and the final sink term.
Torch and Triton GEMM/reduction implementations need not be bit-identical to
TileLang. The optional Triton backend requires separate GPU numerical tests.
Invalid -1 slots are zero KV with an initial -inf QK accumulator, not a mask
applied after QK. Other out-of-range IDs are safely treated the same way, but
lie outside the original reference's defined input domain.
"""
from __future__ import annotations

import math
import struct

import torch


_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1
_HEADS = (8, 16, 32, 64)
_SLOTS = 64
_MAX_QUERY_CHUNK = 32


def _integer(value, name, *, minimum=0, maximum=_INT32_MAX):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _contract(q, window_kv, compressed_kv, attn_sink, topk_idxs, softmax_scale, query_chunk_size):
    tensors = (q, window_kv, attn_sink, topk_idxs)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("q, window KV, sink and indices must be tensors")
    if compressed_kv is not None and not isinstance(compressed_kv, torch.Tensor):
        raise TypeError("compressed KV must be a tensor or None")
    if q.ndim != 4 or window_kv.ndim != 3 or topk_idxs.ndim != 3:
        raise ValueError("expected q[B,Q,H,512], KV[B,S,512] and indices[B,Q,K]")
    batch, queries, heads, dim = q.shape
    if batch < 1 or queries < 1 or heads not in _HEADS or dim != 512:
        raise ValueError("require positive B/Q, H in {8,16,32,64}, D=512")
    if attn_sink.shape != (heads,) or topk_idxs.shape[:2] != (batch, queries):
        raise ValueError("sink/indices must match q geometry")
    pools = (window_kv,) if compressed_kv is None else (window_kv, compressed_kv)
    for pool in pools:
        if pool.ndim != 3 or pool.shape[0] != batch or pool.shape[2] != dim:
            raise ValueError("KV pools must match q batch and D=512")
        if pool.dtype != torch.bfloat16:
            raise TypeError("KV pools must contain reference BF16 values")
    if q.dtype != torch.bfloat16 or attn_sink.dtype != torch.float32 or topk_idxs.dtype != torch.int32:
        raise TypeError("require BF16 q, FP32 sink and int32 indices")
    # The device loop advances an i32 slot counter by 64. Reserve room for
    # its final increment, including a partially filled last tile.
    _integer(topk_idxs.shape[-1], "selected slot count", maximum=_INT32_MAX - (_SLOTS - 1))
    for tensor in (*tensors, *pools):
        if tensor.device != q.device or tensor.layout != torch.strided:
            raise ValueError("all inputs must be strided tensors on the same device")
        for size in tensor.shape:
            _integer(size, "tensor dimension")
        for stride in tensor.stride():
            _integer(stride, "tensor stride", maximum=_INT64_MAX)
    width_window = window_kv.shape[1]
    width_comp = 0 if compressed_kv is None else compressed_kv.shape[1]
    _integer(width_window + width_comp, "merged KV width")
    _integer(query_chunk_size, "query_chunk_size", minimum=1, maximum=_MAX_QUERY_CHUNK)
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a finite positive Python scalar")
    try:
        scale = struct.unpack("f", struct.pack("f", softmax_scale))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("softmax_scale must be representable as finite positive FP32") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be representable as finite positive FP32")
    return batch, queries, heads, width_window, width_comp, scale


def _gather_tile(window_kv, compressed_kv, batch_ids, selected):
    """Gather at most query_chunk_size*64 rows; never index an invalid address."""
    width_window = window_kv.shape[1]
    width_comp = 0 if compressed_kv is None else compressed_kv.shape[1]
    valid_window = (selected >= 0) & (selected < width_window)
    valid_comp = (selected >= width_window) & (selected < width_window + width_comp)
    keys = torch.zeros((*selected.shape, 512), dtype=torch.bfloat16, device=selected.device)
    if width_window:
        gathered = window_kv[batch_ids[:, None], selected.clamp(0, width_window - 1)]
        keys = torch.where(valid_window[..., None], gathered, keys)
    if width_comp:
        local = (selected - width_window).clamp(0, width_comp - 1)
        gathered = compressed_kv[batch_ids[:, None], local]
        keys = torch.where(valid_comp[..., None], gathered, keys)
    return keys, valid_window | valid_comp


def dual_sparse_attn(
    q, window_kv, compressed_kv, attn_sink, topk_idxs, softmax_scale,
    *, backend="torch", query_chunk_size=1,
):
    """Return contiguous BF16 [B,Q,H,512] with no full-prefix KV temporary.

    Pools, q, sink and IDs may have non-contiguous, nonnegative strides.
    Inputs must not be mutated concurrently. Duplicate valid indices retain
    multiplicity and ordering. K=0 and all-invalid rows still execute the sink
    denominator; they are not unconditionally replaced with zero. NaN payload
    identity and backend floating reduction bit identity are not promised.
    """
    batch, queries, heads, _, _, scale = _contract(
        q, window_kv, compressed_kv, attn_sink, topk_idxs, softmax_scale, query_chunk_size,
    )
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or explicit opt-in 'triton'")
    if backend == "triton":
        try:
            from .dsv41_dual_sparse_triton import dual_sparse_triton
        except ImportError:
            from dsv41_dual_sparse_triton import dual_sparse_triton
        return dual_sparse_triton(q, window_kv, compressed_kv, attn_sink, topk_idxs, scale)

    result = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    slots = topk_idxs.shape[-1]
    for first in range(0, batch * queries, query_chunk_size):
        rows = torch.arange(first, min(first + query_chunk_size, batch * queries), device=q.device)
        batch_ids, query_ids = rows // queries, rows % queries
        q_float = q[batch_ids, query_ids].float()
        maximum = torch.full((rows.numel(), heads), -1e30, dtype=torch.float32, device=q.device)
        denominator = torch.zeros_like(maximum)
        accumulated = torch.zeros_like(q_float)
        for start in range(0, slots, _SLOTS):
            selected = torch.full((rows.numel(), _SLOTS), -1, dtype=torch.int64, device=q.device)
            count = min(_SLOTS, slots - start)
            selected[:, :count] = topk_idxs[batch_ids, query_ids, start:start + count]
            keys, valid = _gather_tile(window_kv, compressed_kv, batch_ids, selected)
            keys_float = keys.float()
            initial = torch.full((rows.numel(), heads, _SLOTS), -torch.inf,
                                 dtype=torch.float32, device=q.device)
            initial.masked_fill_(valid[:, None, :], 0.0)
            # Seed QK with -inf before GEMM, preserving NaN*0 behavior of
            # invalid slots. A post-GEMM masked_fill would not be equivalent.
            scores = torch.baddbmm(initial, q_float, keys_float.transpose(1, 2)) * scale
            previous = maximum
            maximum = torch.maximum(previous, scores.amax(dim=-1))
            correction = torch.exp(previous - maximum)
            probability = torch.exp(scores - maximum[..., None])
            denominator = denominator * correction + probability.sum(dim=-1)
            accumulated = torch.baddbmm(
                accumulated * correction[..., None],
                probability.to(torch.bfloat16).float(), keys_float,
            )
        denominator = denominator + torch.exp(attn_sink[None, :] - maximum)
        result[batch_ids, query_ids] = (accumulated / denominator[..., None]).to(torch.bfloat16)
    return result
