# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8's QSA: key compression, block scoring and selection, and sparse paged GQA attention (kernels).

The Triton kernels below are vLLM's (vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py, the file that served this model
on this fleet in the vLLM stack; SOURCES.json pins the ported copy, overlay/modules/qwen38_qsa/ops_qsa.py). Their bodies
are unchanged. What changed around them (engine/kernels/SOURCES.json lists it):

- the imports are Triton's own, not vllm.triton_utils / vllm.platforms;
- block selection uses the engine's top-k instead of vLLM's persistent_topk C++ op: engine/kernels/prefill_topk (a radix
  select over the valid prefix, ties to the lower block) where it admits the shape, torch.topk otherwise (decode and
  capture). Columns past a row's visible blocks are never written by the scorer; they are masked before torch.topk;
- the split-K profile of the sparse attention is upstream's, without the DENEB_QSA_MAX_SPLITS environment cap (D11:
  the kernel package reads no knobs). Whether GB10 wants fewer splits is the wizard's measurement, not a default;
- `norm_rope_partial` is new: Qwen3.8 normalises query and key heads with a unit-offset weight and rotates only the
  first `rotary_dim` channels (64 of 256, 64 of the indexer's 128) as neox halves -- engine/kernels/common/norm_rope
  rotates the whole head and weights plainly;
- the sparse attention can read the chosen blocks themselves (FROM_BLOCKS: `qsa_select_paged_blocks` then
  `qsa_sparse_paged_attention_blocks`): each tile computes its columns' positions with the expansion kernel's
  arithmetic instead of loading them from an expanded buffer -- the int32 positions of the row's blocks sorted
  ascending (-1 last; `tl.sort` in the program), so the attention's sums are the same for every selector's order of
  the same set, without the expansion launch and its [rows, top-k + ratio - 1] buffer;
- a layer's inputs take two launches instead of nine: `qsa_index_keys` runs the compression, the norm and rotation of
  the pooled keys and their store in one program a row, `qsa_inputs` the query, key and index query norms and rotations
  with the K, V and raw-key ring stores in one program a (row, head) -- the ported programs' arithmetic line for line
  (`_norm_rope_into` is `_norm_rope_partial`'s), in the order that keeps the ring read before it is written;
- the sparse attention can apply the layer's output gate in its final store (GATED: the one-split launch or the merge):
  the attention rounded to BF16, times sigmoid(gate) in fp32, rounded once -- the layer's gate launches without them.

The references are engine/modules: attention.Attention(select=QSA) and sparse_indexer.qsa_select. Paged caches here are
[pages, page_size, heads, dim] with a page table of physical pages per request -- the served caches present their
block regions in that form (a strided view per layer), so the kernels never learn the engine's block layout.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    # Pad the small head axis to a tensor-core-compatible N dimension.
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(page_valid, score, -float("inf")),
            mask=live & (columns < num_columns),
        )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    query_positions_ptr=None,
    sequence_lengths_ptr=None,
    num_lengths=0,
    FROM_BLOCKS: tl.constexpr = False,
    BLOCK_TOPK: tl.constexpr = 0,
    COMPRESS_RATIO: tl.constexpr = 1,
    gate_ptr=None,
    stride_gate_row=0,
    stride_gate_head=0,
    GATED: tl.constexpr = False,
    BLOCK_SORT: tl.constexpr = 1,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    if FROM_BLOCKS:
        # ST: indices_ptr holds the row's chosen blocks [rows, BLOCK_TOPK]; the positions they expand to are computed
        # tile by tile below with _expand_qsa_indices_kernel's arithmetic (TOPK is its output width)
        query_position = tl.load(query_positions_ptr + row)
        sequence_length = tl.load(
            sequence_lengths_ptr + tl.minimum(tl.maximum(request, 0), num_lengths - 1),
            mask=(request >= 0) & (request < num_lengths),
            other=0,
        )
        complete_blocks = tl.minimum(
            tl.minimum(
                (query_position + 1) // COMPRESS_RATIO,
                sequence_length // COMPRESS_RATIO,
            ),
            BLOCK_TOPK,
        )
        expanded_count = complete_blocks * COMPRESS_RATIO
        tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
        tail_count = (query_position + 1) - tail_start
        # ST (carry Q6): the chosen blocks in ascending order, -1 last (BLOCK_SORT, a power of two, holds BLOCK_TOPK):
        # every selector's set -- torch.topk's value order, prefill_topk's, st_dsa_select's -- then puts the same
        # positions on the same tiles, so the attention's sums round alike whichever one ran
        sort_ranks = tl.arange(0, BLOCK_SORT)
        chosen = tl.load(
            indices_ptr + row * stride_indices_row + sort_ranks,
            mask=sort_ranks < BLOCK_TOPK,
            other=2147483647,
        )
        ordered = tl.sort(tl.where(chosen < 0, 2147483647, chosen))

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        if FROM_BLOCKS:
            is_expanded = columns < expanded_count
            picked = tl.gather(ordered, tl.minimum(columns // COMPRESS_RATIO, BLOCK_TOPK - 1), 0)
            block = tl.where(is_expanded & (picked != 2147483647), picked, -1)
            expanded = block * COMPRESS_RATIO + columns % COMPRESS_RATIO
            tail_offset = columns - expanded_count
            is_tail = (
                (columns >= expanded_count)
                & (tail_offset < tail_count)
                & (tail_offset < COMPRESS_RATIO - 1)
            )
            token = tl.where(is_expanded, expanded, tail_start + tail_offset)
            logical_token = tl.where(
                (columns < TOPK)
                & (is_expanded | is_tail)
                & (token >= 0)
                & (token < sequence_length),
                token,
                -1,
            )
        else:
            logical_token = tl.load(
                indices_ptr + row * stride_indices_row + columns,
                mask=columns < TOPK,
                other=-1,
            )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        value = normalized_output
        if GATED:
            # ST (carry Q4): the output gate in the store -- the attention rounded to the output's dtype, times
            # sigmoid(gate) in fp32 (1 / (1 + exp(-g)), torch's), rounded once: the layer's
            # (attended.float() * torch.sigmoid(gate.float())).to(bf16) without its launches and fp32 temporaries
            gate = tl.load(
                gate_ptr
                + row * stride_gate_row
                + (first_head + head_offsets[:, None]) * stride_gate_head
                + dim_offsets[None, :],
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
            value = normalized_output.to(output_ptr.dtype.element_ty).to(tl.float32) * tl.sigmoid(gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            value,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    gate_ptr=None,
    stride_gate_row=0,
    stride_gate_head=0,
    GATED: tl.constexpr = False,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    if GATED:
        # ST (carry Q4): the output gate in the merged store, as in the one-split store
        gate = tl.load(gate_ptr + row * stride_gate_row + head * stride_gate_head + dim_offsets).to(tl.float32)
        merged = merged.to(output_ptr.dtype.element_ty).to(tl.float32) * tl.sigmoid(gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    # int64 before the page stride: an int32 slot mapping times a block-major page stride (5,431,296 bf16 rows for
    # Qwen3.8's 13 attention layers) wraps from page 396 on
    block = (tl.maximum(slot, 0) // PAGE_SIZE).to(tl.int64)
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


@triton.jit
def _norm_rope_partial(X, W, POS, INV, OUT, sXr, sXh, sO, sP, EPS, D: tl.constexpr, R2: tl.constexpr,
                       BD: tl.constexpr, BR: tl.constexpr):
    # One program owns one head: the unit-offset RMS norm over the whole head, rounded once to the output's dtype
    # (engine/modules/norm.rmsnorm_unit_offset), then the neox rotation of the first 2 * R2 channels reading the
    # rounded values (engine/modules/rotary.apply_rope). The rest of the head passes through as the norm left it.
    r = tl.program_id(0)
    h = tl.program_id(1)
    base, out = X + r * sXr + h * sXh, OUT + r * sO + h * D
    d = tl.arange(0, BD)
    m = d < D
    x = tl.load(base + d, mask=m, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / D + EPS)
    w = tl.load(W + d, mask=m, other=0.0).to(tl.float32)
    tl.store(out + d, ((x * scale) * (1.0 + w)).to(OUT.dtype.element_ty), mask=m)
    i = tl.arange(0, BR)
    mi = i < R2
    xl = tl.load(base + i, mask=mi, other=0.0).to(tl.float32)
    xh = tl.load(base + R2 + i, mask=mi, other=0.0).to(tl.float32)
    wl = tl.load(W + i, mask=mi, other=0.0).to(tl.float32)
    wh = tl.load(W + R2 + i, mask=mi, other=0.0).to(tl.float32)
    lo = ((xl * scale) * (1.0 + wl)).to(OUT.dtype.element_ty).to(tl.float32)
    hi = ((xh * scale) * (1.0 + wh)).to(OUT.dtype.element_ty).to(tl.float32)
    angle = tl.load(POS + r * sP).to(tl.float32) * tl.load(INV + i, mask=mi, other=0.0)
    cos, sin = tl.cos(angle), tl.sin(angle)
    tl.store(out + i, (lo * cos - hi * sin).to(OUT.dtype.element_ty), mask=mi)
    tl.store(out + R2 + i, (lo * sin + hi * cos).to(OUT.dtype.element_ty), mask=mi)


def norm_rope_partial(x: torch.Tensor, w: torch.Tensor, eps: float, positions: torch.Tensor, theta: float,
                      rotary_dim: int) -> torch.Tensor:
    """rope(rmsnorm_unit_offset(x, w, eps)) with the rotation over the first `rotary_dim` channels, for heads
    x [N, heads, D] at absolute `positions` [N] int64."""
    if (x.ndim != 3 or w.shape != (x.shape[-1],) or positions.shape != (x.shape[0],) or rotary_dim <= 0
            or rotary_dim % 2 or rotary_dim > x.shape[-1] or positions.dtype != torch.int64):
        raise ValueError("norm_rope_partial takes x [N, heads, D], w [D], int64 positions [N] and an even rotary "
                         "width no wider than the head")
    if not x.is_cuda:
        from engine.modules.norm import rmsnorm_unit_offset
        from engine.modules.rotary import apply_rope, rope_tables
        cos, sin = rope_tables(positions, rotary_dim, theta, dtype=x.dtype)
        return apply_rope(rmsnorm_unit_offset(x, w, eps), cos, sin)
    from engine.kernels.common.norm_rope import warm
    rows, heads, D = x.shape
    src = x if x.stride(2) == 1 else x.contiguous()
    out = torch.empty(rows, heads, D, device=x.device, dtype=x.dtype)
    inv = warm(x.device, rotary_dim, theta)
    if rows and heads:
        # positions are read through their stride: a group's first positions arrive as a column of [rows, 3]
        _norm_rope_partial[(rows, heads)](src, w, positions, inv, out, src.stride(0), src.stride(1), out.stride(0),
                                          positions.stride(0), eps,
                                          D=D, R2=rotary_dim // 2, BD=triton.next_power_of_2(D),
                                          BR=triton.next_power_of_2(rotary_dim // 2), num_warps=4)
    return out


@triton.jit
def _norm_rope_into(base, W, pos, INV, out, live, EPS, D: tl.constexpr, R2: tl.constexpr, BD: tl.constexpr,
                    BR: tl.constexpr):
    # _norm_rope_partial's program, line for line, for one head at `base` into `out` (both unit-stride along the head)
    # when `live`: the fused QSA input launches below run it for several heads and caches in one program
    d = tl.arange(0, BD)
    m = (d < D) & live
    x = tl.load(base + d, mask=m, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / D + EPS)
    w = tl.load(W + d, mask=m, other=0.0).to(tl.float32)
    tl.store(out + d, ((x * scale) * (1.0 + w)).to(out.dtype.element_ty), mask=m)
    i = tl.arange(0, BR)
    mi = (i < R2) & live
    xl = tl.load(base + i, mask=mi, other=0.0).to(tl.float32)
    xh = tl.load(base + R2 + i, mask=mi, other=0.0).to(tl.float32)
    wl = tl.load(W + i, mask=mi, other=0.0).to(tl.float32)
    wh = tl.load(W + R2 + i, mask=mi, other=0.0).to(tl.float32)
    lo = ((xl * scale) * (1.0 + wl)).to(out.dtype.element_ty).to(tl.float32)
    hi = ((xh * scale) * (1.0 + wh)).to(out.dtype.element_ty).to(tl.float32)
    angle = pos.to(tl.float32) * tl.load(INV + i, mask=mi, other=0.0)
    cos, sin = tl.cos(angle), tl.sin(angle)
    tl.store(out + i, (lo * cos - hi * sin).to(out.dtype.element_ty), mask=mi)
    tl.store(out + R2 + i, (lo * sin + hi * cos).to(out.dtype.element_ty), mask=mi)


@triton.jit
def _qsa_inputs_kernel(QG, KX, VX, IQX, IKX, POS, INV, WQ, WK, WIQ, QOUT, IQOUT, KC, VC, RC, KV_SLOTS, RING_SLOTS,
                       sQGr, sQGh, sKr, sVr, sIQr, sIQh, sIKr, sQOr, sIQOr, sP, sKCb, sKCt, sVCb, sVCt, sRCb, sRCt,
                       kv_pages, ring_pages, EPS, HQ: tl.constexpr, HI: tl.constexpr, D: tl.constexpr,
                       DI: tl.constexpr, R2: tl.constexpr, KV_PAGE: tl.constexpr, RING_PAGE: tl.constexpr,
                       BD: tl.constexpr, BDI: tl.constexpr, BR: tl.constexpr):
    # One program a (row, query head): the query head's norm and rotation, the index query head's (h < HI), and at
    # h == 0 the key head's straight into the K cache, the value row into V and the raw index key into the key ring --
    # norm_rope_partial x 3 and qsa_store_cache_rows x 3, the same arithmetic and addresses (one KV head a rank)
    r = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(POS + r * sP)
    _norm_rope_into(QG + r * sQGr + h * sQGh, WQ, pos, INV, QOUT + r * sQOr + h * D, h < HQ, EPS, D, R2, BD, BR)
    _norm_rope_into(IQX + r * sIQr + h * sIQh, WIQ, pos, INV, IQOUT + r * sIQOr + h * DI, h < HI, EPS, DI, R2, BDI,
                    BR)
    # qsa_store_cache_rows' address and validity: int64 before the page stride
    kv_slot = tl.load(KV_SLOTS + r)
    kv_live = (h == 0) & (kv_slot >= 0) & (kv_slot < kv_pages * KV_PAGE)
    kv_block = (tl.maximum(kv_slot, 0) // KV_PAGE).to(tl.int64)
    kv_token = tl.maximum(kv_slot, 0) % KV_PAGE
    _norm_rope_into(KX + r * sKr, WK, pos, INV, KC + kv_block * sKCb + kv_token * sKCt, kv_live, EPS, D, R2, BD, BR)
    dims = tl.arange(0, BD)
    tl.store(VC + kv_block * sVCb + kv_token * sVCt + dims,
             tl.load(VX + r * sVr + dims, mask=kv_live & (dims < D), other=0), mask=kv_live & (dims < D))
    ring_slot = tl.load(RING_SLOTS + r)
    ring_live = (h == 0) & (ring_slot >= 0) & (ring_slot < ring_pages * RING_PAGE)
    ring_block = (tl.maximum(ring_slot, 0) // RING_PAGE).to(tl.int64)
    ring_token = tl.maximum(ring_slot, 0) % RING_PAGE
    idims = tl.arange(0, BDI)
    tl.store(RC + ring_block * sRCb + ring_token * sRCt + idims,
             tl.load(IKX + r * sIKr + idims, mask=ring_live & (idims < DI), other=0), mask=ring_live & (idims < DI))


@triton.jit
def _qsa_index_keys_kernel(raw_keys_ptr, compressor_state_cache_ptr, compressor_state_table_ptr, token_to_req_ptr,
                           query_start_loc_ptr, logical_positions_ptr, compressed_slots_ptr, pooled_ptr, W, INV,
                           KEYS, stride_raw_row, stride_compressor_state_block, stride_compressor_state_token,
                           stride_compressor_state_table_req, stride_pooled_row, sKb, sKt, num_rows,
                           num_compressor_state_blocks, num_requests, key_pages, EPS,
                           COMPRESSOR_STATE_SIZE: tl.constexpr, COMPRESS_RATIO: tl.constexpr, HEAD_DIM: tl.constexpr,
                           BLOCK_D: tl.constexpr, R2: tl.constexpr, BR: tl.constexpr, KEY_PAGE: tl.constexpr):
    # _compress_qsa_groups_kernel's program (the pooled mean of the group a row closes, from the ring and this step's
    # rows; unit strides along the head, no rope cache), then _norm_rope_partial's at the group's first position and
    # qsa_store_cache_rows' write of the key at the row's slot: the three launches of a QSA layer's index keys in one
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)
    # the pooled row as the compression stored it (rounded to the keys' dtype), read back by the norm below
    pooled = pooled_ptr + row * stride_pooled_row
    tl.store(pooled + dims, accumulator / COMPRESS_RATIO, mask=(row < num_rows) & (dims < HEAD_DIM))
    first_position = tl.where(valid_row, end_position - COMPRESS_RATIO + 1, 0)
    key_live = (compressed_slot >= 0) & (compressed_slot < key_pages * KEY_PAGE)
    key_block = (tl.maximum(compressed_slot, 0) // KEY_PAGE).to(tl.int64)
    key_token = tl.maximum(compressed_slot, 0) % KEY_PAGE
    _norm_rope_into(pooled, W, first_position, INV, KEYS + key_block * sKb + key_token * sKt, key_live, EPS,
                    HEAD_DIM, R2, BLOCK_D, BR)


def _paged_rows(cache: torch.Tensor, what: str, width: int) -> None:
    if (cache.ndim != 4 or cache.shape[2] != 1 or cache.shape[3] != width or not all(cache.shape)
            or cache.stride(3) != 1):
        raise ValueError(f"{what} must be [pages, page_size, 1, {width}] with a unit stride along the row")


def qsa_index_keys(raw_keys, compressor_state_cache, compressor_state_block_table, token_to_req, query_start_loc,
                   logical_positions, compressed_slots, compress_ratio, weight, eps, theta, rotary_dim, key_cache):
    """The index keys a step's rows close, in one launch: qsa_compress_groups_with_ratio (no rope cache), then
    norm_rope_partial at each group's first position, then qsa_store_cache_rows at `compressed_slots` (-1 skipped) --
    the same arithmetic and bytes. `raw_keys` is [rows, head_dim] with a unit stride along the head (the in_proj
    columns). Launch it before this step's raw keys go into the ring: the pooled groups read the ring's older members,
    and a long prefill's ring writes cover every ring cell."""
    if not raw_keys.is_cuda:
        raise RuntimeError("QSA index keys run on CUDA")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 2 or raw_keys.shape[0] != rows or raw_keys.stride(1) != 1:
        raise ValueError("QSA raw keys must be [rows, head_dim] with a unit stride along the head")
    head_dim = raw_keys.shape[1]
    if (logical_positions.shape != (rows,) or compressed_slots.shape != (rows,)
            or logical_positions.dtype != torch.int64):
        raise ValueError("QSA compression metadata must match token rows (int64 positions)")
    _packed_rows("QSA index keys", token_to_req, logical_positions, compressed_slots)
    _paged_rows(compressor_state_cache, "QSA compressor-state cache", head_dim)
    _paged_rows(key_cache, "QSA index key cache", head_dim)
    if compressor_state_cache.shape[1] < compress_ratio or compressor_state_cache.dtype != raw_keys.dtype:
        raise ValueError("QSA compressor-state cache does not match the compression layout")
    if key_cache.dtype != raw_keys.dtype or weight.shape != (head_dim,):
        raise ValueError("QSA index keys take the raw keys' dtype and a [head_dim] norm weight")
    if compressor_state_block_table.ndim != 2 or compressor_state_block_table.shape[1] < 1:
        raise ValueError("QSA compressor-state block table must contain one block per request")
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_dim:
        raise ValueError("QSA index keys rotate an even width no wider than the head")
    if not rows:
        return
    from engine.kernels.common.norm_rope import warm
    pooled = torch.empty((rows, head_dim), dtype=raw_keys.dtype, device=raw_keys.device)
    inv = warm(raw_keys.device, rotary_dim, theta)
    _qsa_index_keys_kernel[(rows,)](
        raw_keys, compressor_state_cache, compressor_state_block_table, token_to_req, query_start_loc,
        logical_positions, compressed_slots, pooled, weight, inv, key_cache,
        raw_keys.stride(0), compressor_state_cache.stride(0), compressor_state_cache.stride(1),
        compressor_state_block_table.stride(0), pooled.stride(0), key_cache.stride(0), key_cache.stride(1),
        rows, compressor_state_cache.shape[0], num_requests, key_cache.shape[0], eps,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1], COMPRESS_RATIO=compress_ratio, HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim), R2=rotary_dim // 2, BR=triton.next_power_of_2(rotary_dim // 2),
        KEY_PAGE=key_cache.shape[1], num_warps=4,
    )


def qsa_inputs(q, k, v, iq, ik, positions, q_norm, k_norm, iq_norm, eps, theta, rotary_dim, k_cache, v_cache, kv_slots,
               ring, ring_slots):
    """A QSA layer's inputs in one launch: norm_rope_partial of the query heads q [N, Hq, D] and of the index query
    heads iq [N, Hi, Di] (both returned), of the key head k [N, 1, D] straight into k_cache, and qsa_store_cache_rows of
    v [N, 1, D] into v_cache at `kv_slots` and of the raw index keys ik [N, Di] into the key ring at `ring_slots` (-1
    skipped) -- the same arithmetic and bytes as those six launches. Views are read through their strides, unit along
    the head. The ring write belongs after qsa_index_keys, which reads the ring."""
    if not q.is_cuda:
        raise RuntimeError("QSA inputs run on CUDA")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or iq.ndim != 3 or ik.ndim != 2:
        raise ValueError("QSA inputs take q [N, Hq, D], k and v [N, 1, D], iq [N, Hi, Di] and ik [N, Di]")
    rows, hq, dim = q.shape
    hi, di = iq.shape[1:]
    if k.shape != (rows, 1, dim) or iq.shape[0] != rows or ik.shape != (rows, di) or hi > hq:
        raise ValueError("QSA inputs: one KV head a rank, as many rows everywhere, no more index heads than query heads")
    if any(x.stride(-1) != 1 for x in (q, k, v, iq, ik)):
        raise ValueError("QSA inputs are read with a unit stride along the head")
    if positions.shape != (rows,) or positions.dtype != torch.int64:
        raise ValueError("QSA inputs take int64 positions [N]")
    if q_norm.shape != (dim,) or k_norm.shape != (dim,) or iq_norm.shape != (di,):
        raise ValueError("QSA input norm weights must match their heads")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > di:
        raise ValueError("QSA inputs rotate an even width no wider than either head")
    _paged_rows(k_cache, "QSA K cache", dim)
    _paged_rows(v_cache, "QSA V cache", dim)
    _paged_rows(ring, "QSA key ring", di)
    if k_cache.shape[:2] != v_cache.shape[:2] or kv_slots.shape != (rows,) or ring_slots.shape != (rows,):
        raise ValueError("QSA inputs: K and V pages match and every row has a K/V and a ring slot")
    if not (q.dtype == k.dtype == v.dtype == iq.dtype == ik.dtype == k_cache.dtype == v_cache.dtype == ring.dtype):
        raise ValueError("QSA inputs and caches share one dtype")
    _packed_rows("QSA inputs", positions, kv_slots, ring_slots)
    q_out = torch.empty(rows, hq, dim, device=q.device, dtype=q.dtype)
    iq_out = torch.empty(rows, hi, di, device=q.device, dtype=q.dtype)
    if not rows:
        return q_out, iq_out
    from engine.kernels.common.norm_rope import warm
    inv = warm(q.device, rotary_dim, theta)
    _qsa_inputs_kernel[(rows, hq)](
        q, k, v, iq, ik, positions, inv, q_norm, k_norm, iq_norm, q_out, iq_out, k_cache, v_cache, ring, kv_slots,
        ring_slots, q.stride(0), q.stride(1), k.stride(0), v.stride(0), iq.stride(0), iq.stride(1), ik.stride(0),
        q_out.stride(0), iq_out.stride(0), positions.stride(0), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1), ring.stride(0), ring.stride(1), k_cache.shape[0], ring.shape[0], eps,
        HQ=hq, HI=hi, D=dim, DI=di, R2=rotary_dim // 2, KV_PAGE=k_cache.shape[1], RING_PAGE=ring.shape[1],
        BD=triton.next_power_of_2(dim), BDI=triton.next_power_of_2(di), BR=triton.next_power_of_2(rotary_dim // 2),
        num_warps=4,
    )
    return q_out, iq_out


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def _packed_rows(what: str, *rows: torch.Tensor) -> None:
    """The kernels load one-dimensional row metadata at `ptr + row`, without a stride: a strided view -- an expanded
    one-row mapping has stride 0 -- would be read past its storage."""
    if any(t.numel() > 1 and t.stride(0) != 1 for t in rows):
        raise ValueError(f"{what} needs packed row metadata (stride 1)")


def qsa_mqa_paged(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, compress_ratio,
                  num_columns=None, score_scale=None):
    """Block scores sum_heads relu(q . key) / score_scale straight from a paged compressed-key cache
    [pages, page_size, 1, head_dim]: (logits f32 [rows, columns], visible blocks int32 [rows]). Columns at or past a
    row's visible count are not written."""
    _validate_mqa(q)
    if not q.is_cuda:
        raise RuntimeError("paged QSA scoring runs on CUDA")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    _packed_rows("paged QSA scoring", token_to_req, query_positions)
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    BLOCK_N = 64
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    MAX_N = max(16, triton.next_power_of_2(q.shape[1]))
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    _qsa_mqa_paged_kernel[(q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))](
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, visible_blocks, logits,
        q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
        page_table.stride(0), page_table.stride(1), logits.stride(0), q.shape[0], columns, k_cache.shape[0],
        page_table.shape[0], float(score_divisor),
        PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=page_table.shape[1], NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2],
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, TILES_PER_PROG=tiles_per_program, STAGES=2, MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio, num_warps=2,
    )
    return logits, visible_blocks


def expand_qsa_block_indices_cuda(block_indices, query_positions, sequence_lengths, token_to_req, compress_ratio,
                                  token_topk, out=None):
    """Expand the chosen blocks to their positions and append the causal tail of the open group:
    int32 [rows, token_topk + compress_ratio - 1], -1 where nothing is attended."""
    if not block_indices.is_cuda:
        raise RuntimeError("QSA expansion runs on CUDA")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    _packed_rows("QSA expansion", token_to_req, query_positions)
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if out is None:
        out = torch.empty((block_indices.shape[0], output_width), dtype=torch.int32, device=block_indices.device)
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[(block_indices.shape[0], triton.cdiv(output_width, column_block))](
        block_indices, query_positions, sequence_lengths, token_to_req, out,
        block_indices.stride(0), block_indices.stride(1), out.stride(0), out.stride(1),
        block_indices.shape[0], sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk, COMPRESS_RATIO=compress_ratio, TOKEN_TOPK=token_topk, OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block, num_warps=4,
    )
    return out


def select_blocks(logits: torch.Tensor, visible_blocks: torch.Tensor, block_topk: int, out: torch.Tensor) -> torch.Tensor:
    """The `block_topk` best blocks of each row among its first `visible_blocks` columns, into out int32
    [rows, block_topk]. Only a row's first min(visible, block_topk) picks are read downstream (the expansion), in any
    order. Where engine/kernels/prefill_topk admits the shape (k 512, 65..32768 rows, eager) it selects over the
    valid prefix with ties to the lower block; otherwise torch.topk over the masked logits."""
    rows, columns = logits.shape
    if out.shape != (rows, block_topk) or out.dtype != torch.int32:
        raise ValueError("block selection writes int32 [rows, block_topk]")
    if not rows:
        return out
    from engine.kernels import prefill_topk
    picked = prefill_topk.select(logits, visible_blocks, block_topk)
    if picked is not None:
        out.copy_(picked)
        return out
    k = min(block_topk, columns)
    if k == 0:
        out.fill_(-1)
        return out
    live = torch.arange(columns, device=logits.device).unsqueeze(0) < visible_blocks.unsqueeze(1)
    values, index = torch.topk(logits.masked_fill(~live, float("-inf")), k, dim=1, largest=True, sorted=True)
    out[:, :k] = torch.where(values > float("-inf"), index.to(torch.int32), -1)
    if k < block_topk:
        out[:, k:] = -1
    return out


def shards_select_alike(rows: int, shards, columns: int, block_topk: int) -> bool:
    """Whether `qsa_select_paged_blocks` over each of `shards` -- the row counts of disjoint row ranges of one step --
    chooses for every row the set one call over the step's `rows` rows chooses (carry Q11: the ranks score a quarter
    of a prefill step's index queries each and gather the ids).

    A row's scores are its own whatever rows share its launch. Its selector is not: `select_blocks` takes the radix
    select (ties to the lower block) where engine/kernels/prefill_topk admits the rows it was handed, a chunk of
    `_LOGITS_WORKSPACE_BYTES` of logits at a time, and torch.topk (ties as its candidates fall) elsewhere -- and just
    past the budget's reach relu leaves enough equal zeros for the two to part at the budget's edge. So a split is the
    whole only where every chunk on both sides takes the radix select, whose choice is a row's alone; nothing else is
    known to order a row's ties alike in launches of different sizes. The tensors' half of the radix rule (CUDA, eager,
    fp32 logits) is the same on both sides and is not asked here."""
    from engine.kernels import prefill_topk
    per = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))          # qsa_select_paged_blocks' rows a scoring call
    return (prefill_topk.admits_calls(rows, per, columns, block_topk)
            and all(prefill_topk.admits_calls(count, per, columns, block_topk) for count in shards if count))


def qsa_select_paged_tokens(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, token_topk,
                            compress_ratio, out=None):
    """Score, select and expand QSA positions without a host synchronization: int32 [rows, token_topk + ratio - 1]."""
    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out
    columns = page_table.shape[1] * k_cache.shape[1]
    block_topk = token_topk // compress_ratio
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    blocks_buffer = torch.empty((min(rows, rows_per_chunk), block_topk), dtype=torch.int32, device=q.device)
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_blocks = qsa_mqa_paged(q[row_slice], k_cache, page_table, token_to_req[row_slice],
                                               query_positions[row_slice], sequence_lengths, compress_ratio)
        blocks = select_blocks(logits, visible_blocks, block_topk, blocks_buffer[: row_end - row_start])
        expand_qsa_block_indices_cuda(blocks, query_positions[row_slice], sequence_lengths, token_to_req[row_slice],
                                      compress_ratio, token_topk, out[row_slice])
    return out


def qsa_select_paged_blocks(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, token_topk,
                            compress_ratio, out=None):
    """`qsa_select_paged_tokens` without its expansion: the chosen blocks, int32 [rows, token_topk // compress_ratio]
    (-1 past a row's visible blocks), for `qsa_sparse_paged_attention_blocks` to expand inside its tiles. The scoring
    chunks and the selection are `qsa_select_paged_tokens`' own, writing each chunk's rows in place."""
    if token_topk <= 0 or compress_ratio <= 0 or token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    rows = q.shape[0]
    block_topk = token_topk // compress_ratio
    if out is None:
        out = torch.empty((rows, block_topk), dtype=torch.int32, device=q.device)
    if out.shape != (rows, block_topk):
        raise ValueError("QSA block selection output has an invalid shape")
    if not rows:
        return out
    columns = page_table.shape[1] * k_cache.shape[1]
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_slice = slice(row_start, min(row_start + rows_per_chunk, rows))
        logits, visible_blocks = qsa_mqa_paged(q[row_slice], k_cache, page_table, token_to_req[row_slice],
                                               query_positions[row_slice], sequence_lengths, compress_ratio)
        select_blocks(logits, visible_blocks, block_topk, out[row_slice])
    return out


def qsa_sparse_paged_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out=None, *, gate=None):
    """Sparse GQA over paged BF16 K/V caches [blocks, page_size, kv_heads, head_dim] at the selected positions
    (int32 [rows, width], -1 skipped): softmax(q . k / sqrt(head_dim)) v per query head, [rows, heads, head_dim]. With
    `gate` (BF16, q's shape, unit stride along the head) the output is BF16(attention * sigmoid(gate)), applied in the
    final store."""
    width = logical_indices.shape[1] if logical_indices.ndim == 2 else 0
    return _sparse_paged_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, width,
                                   gate=gate)


def qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, block_indices, query_positions, sequence_lengths,
                                      compress_ratio, token_topk, block_table, token_to_req, out=None, *, gate=None):
    """`qsa_sparse_paged_attention` at the positions `expand_qsa_block_indices_cuda` expands the chosen blocks to --
    int32 [rows, token_topk // compress_ratio] as `qsa_select_paged_blocks` writes them -- computed tile by tile in the
    attention's own launch (FROM_BLOCKS), without the expansion launch and its [rows, token_topk + compress_ratio - 1]
    buffer. The blocks are read in ascending order, -1 last, whatever order a row holds them in: the output is the
    expanded attention over the row's blocks sorted so, and the same for every order of the same set."""
    if token_topk <= 0 or compress_ratio <= 0 or token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    rows = q.shape[0] if q.ndim == 3 else -1
    if block_indices.shape != (rows, block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if query_positions.shape != (rows,):
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if not (block_indices.dtype == query_positions.dtype == sequence_lengths.dtype == torch.int32):
        raise ValueError("QSA sparse attention metadata is int32")
    if query_positions.device != q.device or sequence_lengths.device != q.device:
        raise ValueError("QSA sparse attention tensors share one device")
    _packed_rows("QSA sparse attention", query_positions, sequence_lengths)
    return _sparse_paged_attention(q, k_cache, v_cache, block_indices, block_table, token_to_req, out,
                                   token_topk + compress_ratio - 1,
                                   blocks=(query_positions, sequence_lengths, compress_ratio, block_topk), gate=gate)


def _sparse_paged_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, width, blocks=None,
                            gate=None):
    """The launch of both entries: `logical_indices` holds the positions at `width` columns, or (with `blocks` --
    query positions, sequence lengths, the compression ratio and the block top-k) the chosen blocks whose expansion
    is `width` columns wide. `gate`, when given, is applied in the final store (the one-split launch or the merge)."""
    if not q.is_cuda:
        raise RuntimeError("paged QSA sparse attention runs on CUDA")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if width <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("QSA sparse attention takes a power-of-two head dimension of at least 16")
    if not (q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16):
        raise ValueError("QSA sparse attention reads BF16 queries and caches")
    if not (logical_indices.dtype == block_table.dtype == token_to_req.dtype == torch.int32):
        raise ValueError("QSA sparse attention metadata is int32")
    if not (q.device == k_cache.device == v_cache.device == logical_indices.device == block_table.device
            == token_to_req.device):
        raise ValueError("QSA sparse attention tensors share one device")
    if (q.stride(2) != 1 or k_cache.stride(3) != 1 or v_cache.stride(3) != 1 or logical_indices.stride(1) != 1
            or block_table.stride(1) != 1 or token_to_req.stride(0) != 1):
        raise ValueError("QSA sparse attention needs packed head dimensions and packed metadata rows")
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device or out.stride(2) != 1:
        raise ValueError("QSA sparse output must match its query")
    if gate is not None and (gate.shape != q.shape or gate.dtype != torch.bfloat16 or gate.device != q.device
                             or gate.stride(2) != 1):
        raise ValueError("the QSA output gate is BF16 in the query's shape with a unit stride along the head")
    gated = {} if gate is None else dict(gate_ptr=gate, stride_gate_row=gate.stride(0), stride_gate_head=gate.stride(1),
                                         GATED=True)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4
    # upstream's split-K profile (tuned on GB300 for the Qwen-Air TP1/2/4 attention shapes): narrow tiles for decode,
    # wide tiles for prefill
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
    num_tiles = triton.cdiv(width, block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    expansion = {}
    if blocks is not None:
        query_positions, sequence_lengths, compress_ratio, block_topk = blocks
        expansion = dict(query_positions_ptr=query_positions, sequence_lengths_ptr=sequence_lengths,
                         num_lengths=sequence_lengths.shape[0], FROM_BLOCKS=True, BLOCK_TOPK=block_topk,
                         BLOCK_SORT=triton.next_power_of_2(block_topk),
                         COMPRESS_RATIO=compress_ratio)
    _qsa_sparse_paged_gqa_splitk_kernel[(q.shape[0], k_cache.shape[2], num_splits)](
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, partial_output, partial_lse, out,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), logical_indices.stride(0), block_table.stride(0),
        out.stride(0), out.stride(1), q.shape[0], k_cache.shape[0], block_table.shape[0],
        TOPK=width, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=partial_warps, num_stages=2, **expansion,
        **(gated if num_splits == 1 else {}),
    )
    if num_splits == 1:
        return out
    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits), num_warps=2, num_stages=1, **gated,
    )
    return out


def qsa_store_cache_rows(cache, slot_mapping, rows):
    """Store fixed-width rows at flat slots (page * page_size + offset; -1 skipped) of a cache
    [pages, page_size, 1, width]."""
    if not cache.is_cuda:
        raise RuntimeError("QSA cache stores run on CUDA")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    _packed_rows("QSA cache stores", slot_mapping)
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache, slot_mapping, rows, cache.stride(0), cache.stride(1), cache.stride(3), rows.stride(0), rows.stride(1),
        rows.shape[0], cache.shape[0], PAGE_SIZE=cache.shape[1], WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]), num_warps=4,
    )


def qsa_compress_groups_with_ratio(raw_keys, raw_positions, compressor_state_cache, compressor_state_block_table,
                                   token_to_req, query_start_loc, logical_positions, compressed_slots, compress_ratio,
                                   rope_cache=None):
    """Pool every group completed at a row's position from the per-request ring of earlier raw keys and this step's
    raw rows: (pooled mean [rows, 1, head_dim] in the keys' dtype, first positions int64 [rows, 3]). No norm and no
    rotation: the caller applies k_layernorm and the rotation at the group's first position (norm_rope_partial)."""
    if not raw_keys.is_cuda:
        raise RuntimeError("QSA compression runs on CUDA")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    _packed_rows("QSA compression", token_to_req, logical_positions, compressed_slots)
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (compressor_state_cache.shape[1] < compress_ratio or compressor_state_cache.shape[3] != raw_keys.shape[2]
            or compressor_state_cache.dtype != raw_keys.dtype):
        raise ValueError("QSA compressor-state cache does not match the compression layout")
    if compressor_state_block_table.ndim != 2 or compressor_state_block_table.shape[1] < 1:
        raise ValueError("QSA compressor-state block table must contain one block per request")
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (rope_cache.ndim != 4 or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
                                   or rope_cache.shape[3] != 3 or rope_cache.dtype != torch.int64):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (not all(compressor_state_cache.shape) or not all(compressor_state_block_table.shape)):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty((rows, 1, raw_keys.shape[2]), dtype=raw_keys.dtype, device=raw_keys.device)
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys, raw_positions, compressor_state_cache, rope_cache, compressor_state_block_table, token_to_req,
        query_start_loc, logical_positions, compressed_slots, pooled, first_positions,
        raw_keys.stride(0), raw_keys.stride(2), raw_positions.stride(0), raw_positions.stride(2),
        compressor_state_cache.stride(0), compressor_state_cache.stride(1), compressor_state_cache.stride(3),
        rope_cache.stride(0), rope_cache.stride(1), rope_cache.stride(3), compressor_state_block_table.stride(0),
        pooled.stride(0), pooled.stride(2), first_positions.stride(0), first_positions.stride(1),
        rows, compressor_state_cache.shape[0], num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1], COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2], LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]), num_warps=4,
    )
    return pooled, first_positions


def qualify(device, *, heads=((6, 256), (4, 128)), rotary_dim: int, theta: float, eps: float, dtype=torch.bfloat16,
            rows=(1, 7, 300), max_position: int = 262144, band_max: float = 5e-2, band_rms: float = 2e-2,
            seed: int = 0) -> dict:
    """Hold `norm_rope_partial` to engine/modules (rmsnorm_unit_offset, then rope_tables + apply_rope over the first
    `rotary_dim` channels) on `device`, at each (heads, head_dim) the model normalises and rotates (the query heads and
    the indexer's), within a few BF16 steps (engine/kernels/gated_residual.drift). The ported vLLM kernels are not
    held here: their bodies are the ones that served this model (SOURCES.json)."""
    from engine.kernels.gated_residual import drift
    from engine.modules.norm import rmsnorm_unit_offset
    from engine.modules.rotary import apply_rope, rope_tables
    gen = torch.Generator(device="cpu").manual_seed(seed)
    worst = {}
    for h, d in heads:
        w = (torch.randn(d, generator=gen) * 0.1).to(device=device, dtype=dtype)
        key, most = f"norm_rope_{h}x{d}", (0.0, 0.0)
        for n in rows:
            x = torch.randn(n, h, d, generator=gen).to(device=device, dtype=dtype)
            positions = torch.randint(0, max_position, (n,), generator=gen).to(device)
            cos, sin = rope_tables(positions, rotary_dim, theta, dtype=dtype)
            m, r = drift(norm_rope_partial(x, w, eps, positions, theta, rotary_dim),
                         apply_rope(rmsnorm_unit_offset(x, w, eps), cos, sin))
            most = (max(most[0], m), max(most[1], r))
        worst[key] = most
    bad = {k: v for k, v in worst.items() if v[0] > band_max or v[1] > band_rms}
    if bad:
        raise RuntimeError(f"QSA norm and partial rotation drift from engine/modules beyond max {band_max:g} / "
                           f"rms {band_rms:g}: {bad}")
    return worst


__all__ = ["norm_rope_partial", "qsa_mqa_paged", "expand_qsa_block_indices_cuda", "select_blocks", "shards_select_alike",
           "qsa_index_keys", "qsa_inputs", "qsa_select_paged_tokens", "qsa_select_paged_blocks",
           "qsa_sparse_paged_attention",
           "qsa_sparse_paged_attention_blocks", "qsa_store_cache_rows", "qsa_compress_groups_with_ratio", "qualify"]
