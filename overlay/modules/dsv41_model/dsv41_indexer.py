"""Candidate-position scoring for the DeepSeek-V4.1 CED indexer.

The source indexer remains dense. Its original block top-k selection supplies
both the dense reference mask and compact, ascending position IDs. Consumers
score those positions, reduce the BF16 scores across head shards, and restore
the original full-width top-k domain. In particular, compact top-k must NOT
replace the final reference top-k: doing so changes its handling of ties.

Inputs q/key are the BF16 values *after* reference FP4 quantize/dequantize and
RoPE. This module neither changes the quantizer nor owns a KV cache. The Torch
backend is the default; the optional Triton backend requires separate device
numerical validation. Neither backend promises bit identity across different
GEMM or collective algorithms.

Gather and full-width restoration temporaries are bounded by query_chunk_size
flattened (batch, query) rows, not by the total prefill query count. Returned
compact IDs/scores still occupy B*Q*C elements. The reference adapter initially
uses this component only for long-context, one-query decode.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_INT32_MAX = 2**31 - 1


def _integer(value, name, *, minimum=0, maximum=_INT32_MAX):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _lengths(compress_lens, shape, device):
    """Use the reference's broadcasting: scalar or [..., queries, 1]."""
    if type(compress_lens) is int:
        _integer(compress_lens, "compress_lens")
        return torch.full((*shape, 1), compress_lens, dtype=torch.int64, device=device)
    if not isinstance(compress_lens, torch.Tensor):
        raise TypeError("compress_lens must be an integer or integer tensor")
    if compress_lens.device != device or compress_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("compress_lens must be an int32/int64 tensor on the input device")
    try:
        return torch.broadcast_to(compress_lens, (*shape, 1))
    except RuntimeError as exc:
        raise ValueError("compress_lens must broadcast to [batch, queries, 1]") from exc


def select_candidate_ids(
    full_scores, compress_lens, topk_blocks=2048, block_size=8, *, return_mask=False,
):
    """Return int32 [B,Q,min(S,K*block_size)] IDs, valid ascending then -1.

    Scores must already have the reference's causal -inf mask. Block selection
    is exactly pad -> amax -> newest-block +inf -> torch.topk. With return_mask,
    return (ids, original_dense_mask) from that SINGLE selection, so equal block
    scores cannot produce different source and consumer choices.
    """
    if not isinstance(full_scores, torch.Tensor) or full_scores.ndim != 3:
        raise ValueError("full_scores must have shape [batch, queries, positions]")
    if not full_scores.is_floating_point():
        raise TypeError("full_scores must have floating dtype")
    _integer(topk_blocks, "topk_blocks", minimum=1)
    _integer(block_size, "block_size", minimum=1)
    if type(return_mask) is not bool:
        raise TypeError("return_mask must be bool")
    width = _integer(full_scores.shape[-1], "positions")
    lengths = _lengths(compress_lens, full_scores.shape[:-1], full_scores.device)
    if width == 0:
        ids = torch.empty_like(full_scores, dtype=torch.int32)
        return (ids, torch.empty_like(full_scores, dtype=torch.bool)) if return_mask else ids

    scores = F.pad(full_scores, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    last = (lengths - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=full_scores.device) == last, torch.inf,
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    reachable = top.values > -torch.inf
    positions = top.indices.unsqueeze(-1) * block_size + torch.arange(
        block_size, device=full_scores.device,
    )
    valid = reachable.unsqueeze(-1) & (positions < width)
    positions = positions.masked_fill(~valid, width).flatten(-2).sort(dim=-1).values
    positions = positions[..., : min(width, topk_blocks * block_size)]
    ids = positions.masked_fill(positions == width, -1).to(torch.int32).contiguous()
    if not return_mask:
        return ids
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, reachable)
    return ids, keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _score_contract(q, index_k, weights, ids, query_chunk_size):
    if not all(isinstance(t, torch.Tensor) for t in (q, index_k, weights, ids)):
        raise TypeError("q, index_k, weights and ids must be tensors")
    if q.ndim != 4 or index_k.ndim != 3 or weights.ndim != 3 or ids.ndim != 3:
        raise ValueError("expected q[B,Q,H,D], key[B,S,D], weights[B,Q,H], ids[B,Q,C]")
    batch, queries, heads, dim = q.shape
    if batch < 1 or queries < 1 or heads < 1 or dim < 1:
        raise ValueError("q batch, query, head and feature dimensions must be positive")
    if index_k.shape[0] != batch or index_k.shape[2] != dim:
        raise ValueError("key batch/features do not match q")
    if weights.shape != (batch, queries, heads) or ids.shape[:2] != (batch, queries):
        raise ValueError("weights/ids do not match q batch/query/head dimensions")
    if any(t.dtype != torch.bfloat16 for t in (q, index_k, weights)):
        raise TypeError("q, key and weights must be reference post-quantization BF16")
    if ids.dtype != torch.int32 or any(t.device != q.device for t in (index_k, weights, ids)):
        raise ValueError("ids must be int32 and all inputs must share a device")
    _integer(index_k.shape[1], "positions")
    if ids.shape[-1] > index_k.shape[1]:
        raise ValueError("candidate capacity cannot exceed full position width")
    _integer(query_chunk_size, "query_chunk_size", minimum=1)
    return batch, queries, heads, dim


def compact_index_scores(
    q, index_k, weights, ids, reduce_fn=None, *, query_chunk_size=1, backend="torch",
):
    """Compute BF16 [B,Q,C] scores with reference rounding boundaries.

    IDs should come from select_candidate_ids. Invalid IDs never index memory
    and contribute zero to the collective. reduce_fn is called exactly once,
    on contiguous BF16 [B,Q,C], including a zero-length C. It must synchronously
    enqueue an in-place sum on the current stream and return None or that SAME
    tensor (not a Work handle or a replacement). Rank-local head partitions and
    identical IDs on every rank are the caller's responsibility.
    """
    batch, queries, heads, dim = _score_contract(q, index_k, weights, ids, query_chunk_size)
    if reduce_fn is not None and not callable(reduce_fn):
        raise TypeError("reduce_fn must be callable or None")
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or explicit opt-in 'triton'")
    width, capacity = index_k.shape[1], ids.shape[-1]
    if backend == "triton":
        try:
            from .dsv41_indexer_triton import compact_scores_triton
        except ImportError:
            from dsv41_indexer_triton import compact_scores_triton
        result = compact_scores_triton(q, index_k, weights, ids)
    else:
        result = torch.zeros((batch, queries, capacity), dtype=q.dtype, device=q.device)
        if width and capacity:
            result_rows = result.view(batch * queries, capacity)
            # Advanced indexing gathers only this flattened-query chunk. It
            # also supports strided batch views of a preallocated KV cache.
            for start in range(0, batch * queries, query_chunk_size):
                row = torch.arange(start, min(start + query_chunk_size, batch * queries), device=q.device)
                b, query = row // queries, row % queries
                selected = ids[b, query].to(torch.int64)
                valid = (selected >= 0) & (selected < width)
                keys = index_k[b[:, None], selected.clamp(0, width - 1)]
                dots = torch.einsum("nhd,ncd->nhc", q[b, query], keys)
                # einsum produces BF16. The separate product likewise rounds
                # to BF16 BEFORE sum, matching the reference eager expression.
                products = dots.relu() * weights[b, query].unsqueeze(-1)
                values = products.sum(dim=1)
                result_rows[start : start + row.numel()] = values.masked_fill(~valid, 0)
    if reduce_fn is not None:
        storage_pointer, strides = result.data_ptr(), result.stride()
        returned = reduce_fn(result)
        if returned is not None and returned is not result:
            raise TypeError("reduce_fn must modify scores in place and return None or the same tensor")
        if result.shape != (batch, queries, capacity) or result.dtype != torch.bfloat16 or result.device != q.device:
            raise ValueError("reduce_fn changed score metadata")
        if not result.is_contiguous():
            raise ValueError("reduce_fn must preserve contiguous score storage")
        if result.data_ptr() != storage_pointer or result.stride() != strides:
            raise ValueError("reduce_fn must preserve score storage and strides")
    return result


def compact_topk(
    scores, ids, compress_lens, offset, k, *, full_width, query_chunk_size=1,
):
    """Restore the original S-wide domain, then run the reference final top-k.

    Each chunk uses [query_chunk_size, S+1] scratch; the last column is a sink
    for invalid IDs. The admitted B=Q=1 decode keeps the original [1,1,S] top-k
    call shape, as well as its full position domain. Multi-query chunks retain
    the mathematical domain but do NOT promise GPU tie identity: row count can
    affect Torch's top-k kernel selection. Candidate IDs are unique by construction.
    Causal masking is applied before top-k, including future positions in the
    source's newest partly-filled block. The final where matches the reference:
    selected positions < compress_lens get offset, and all others become -1.
    """
    if not isinstance(scores, torch.Tensor) or not isinstance(ids, torch.Tensor):
        raise TypeError("scores and ids must be tensors")
    if scores.ndim != 3 or ids.shape != scores.shape or ids.dtype != torch.int32:
        raise ValueError("scores/ids must have matching [batch, queries, candidates] shapes")
    if scores.dtype != torch.bfloat16 or ids.device != scores.device:
        raise ValueError("scores must be BF16 and ids must share its device")
    _integer(full_width, "full_width")
    _integer(offset, "offset", maximum=_INT32_MAX - full_width)
    _integer(k, "k", minimum=1)
    _integer(query_chunk_size, "query_chunk_size", minimum=1)
    batch, queries, capacity = scores.shape
    lengths = _lengths(compress_lens, (batch, queries), scores.device)
    result = torch.empty((batch, queries, min(k, full_width)), dtype=torch.int32, device=scores.device)
    if full_width == 0:
        return result
    positions = torch.arange(full_width, device=scores.device)
    for start in range(0, batch * queries, query_chunk_size):
        row = torch.arange(start, min(start + query_chunk_size, batch * queries), device=scores.device)
        b, query = row // queries, row % queries
        selected = ids[b, query].to(torch.int64)
        valid = (selected >= 0) & (selected < full_width)
        restored = torch.full((row.numel(), full_width + 1), -torch.inf, dtype=scores.dtype, device=scores.device)
        restored.scatter_(1, selected.masked_fill(~valid, full_width), scores[b, query].masked_fill(~valid, -torch.inf))
        full = restored[:, :full_width].contiguous()
        lens = lengths[b, query]
        full.masked_fill_(positions >= lens, -torch.inf)
        topk_input = full.view(1, 1, full_width) if batch == queries == 1 else full
        chosen = topk_input.topk(min(k, full_width), dim=-1, sorted=False).indices.sort(dim=-1).values
        chosen = chosen.reshape(row.numel(), min(k, full_width))
        result[b, query] = torch.where(chosen < lens, chosen + offset, -1).to(torch.int32)
    return result
