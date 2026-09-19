# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8's QSA: key compression, block scoring and selection, and sparse paged GQA attention (kernels).

The Triton kernels below are vLLM's (vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py, the file that served this model
on this fleet in the vLLM stack; SOURCES.json pins the ported copy, overlay/modules/qwen38_qsa/ops_qsa.py). Their bodies
are unchanged. What changed around them (engine/kernels/SOURCES.json lists it):

- the imports are Triton's own, not vllm.triton_utils / vllm.platforms;
- block selection uses the engine's top-k instead of vLLM's persistent_topk C++ op: engine/kernels/prefill_topk (a radix
  select over the valid prefix, ties to the lower block) where it admits the shape, engine/kernels/qsa_select (one
  launch a step, the same rule) for every other device step -- decode and capture -- and torch.topk on the CPU. Columns
  past a row's visible blocks are never written by the scorer; every selector reads a row's visible prefix only;
- the split-K profile of the sparse attention is a GB10's, not upstream's GB300 one, and there is no
  DENEB_QSA_MAX_SPLITS environment cap (D11: the kernel package reads no knobs): the lane probe
  probes/engine_qwen38_qsa_geometry.py forces a launch's geometry through the `_OVERRIDE` probe hooks below and its
  record (carry Q9, measurements/qwen38_qsa_geometry_20260919) is `_split_profile`'s table;
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

# Probe hooks (probes/engine_qwen38_qsa_geometry.py, carry Q9): a launch's geometry forced when set, the rule when None.
# A launcher reads its hook when it runs, so a captured graph keeps the geometry it was captured with. Nothing served
# sets them (D11: the kernel package reads no knobs).
_SPLIT_PROFILE_OVERRIDE = None      # (tile width, target splits, warps): the sparse and the covered attention
_SCORE_PROFILE_OVERRIDE = None      # (tile width, tiles a program, warps): qsa_mqa_paged, the row and the run kernel
_INPUT_WARPS_OVERRIDE = None        # warps of qsa_index_keys' and of qsa_inputs' launch
_STACK_OVERRIDE = None              # (tile width, stacked M, warps): qsa_covered_paged_attention's stacked launch

# The stacked covered launch (`_qsa_covered_stacked_kernel`): a program's M rows are STACK_M // group_size query rows'
# every head (60 of 64 at Qwen3.8's six heads a KV head), from STACK_MIN_ROWS rows of one request -- below, the run
# kernel's splits fill the device better than a handful of stacked programs. On a GB10 (q38qsastack-0919a, minima of an
# eager sweep beside production) a fresh prompt's covered attention went 1,610 -> 795 us at 2,048 rows, 474 -> 222 at
# 1,024, 171 -> 124 at 512, the sparse launch's bytes at 16-wide tiles and 4 warps; 32-wide tiles were 20% faster
# again (637 us) but round the softmax elsewhere, and 64-wide ones or M 128 at 32 ask more than the shared memory.
STACK_M, STACK_WARPS, STACK_MIN_ROWS = 64, 4, 256
_RUNS_OVERRIDE = None               # (run rows, warps): qsa_sparse_paged_attention_blocks' run launch

# The sparse run launch (`_qsa_sparse_runs_kernel`): RUN consecutive rows of one request a program over the union of
# their chosen blocks, from RUNS_MIN_ROWS rows (a prefill segment's; the split launch serves fewer). Two rows' six heads
# are twelve of the MMA's sixteen rows that one row's six already pay for, so the pair's tiles cost what the two rows'
# own cost and read each block the two share once.
RUN_ROWS, RUN_WARPS, RUNS_MIN_ROWS = 2, 4, 256


def _forced_geometry(name: str, value, fields: int) -> tuple:
    """A probe hook's value: `fields` powers of two, a tile width no narrower than upstream's 16 first."""
    if (type(value) is not tuple or len(value) != fields
            or not all(type(n) is int and n > 0 and not n & (n - 1) for n in value) or (fields > 1 and value[0] < 16)):
        raise ValueError(f"{name} is {fields} powers of two" + (", the tile width 16 or wider" if fields > 1 else ""))
    return value


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
def _qsa_visible(query_positions_ptr, row, num_rows, reach, COMPRESS_RATIO: tl.constexpr):
    """A row's visible blocks as `_qsa_mqa_paged_kernel` counts them; a row past the launch (position -1) sees none."""
    position = tl.load(query_positions_ptr + row, mask=row < num_rows, other=-1)
    return tl.minimum((position + 1) // COMPRESS_RATIO, reach)


@triton.jit
def _qsa_load_query(q_ptr, row, num_rows, stride_q_row, stride_q_head, stride_q_dim, heads, dims,
                    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr):
    return tl.load(
        q_ptr + row * stride_q_row + heads[None, :] * stride_q_head + dims[:, None] * stride_q_dim,
        mask=(row < num_rows) & (heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )


@triton.jit
def _qsa_store_scores(keys, query, heads, columns, tile_valid, visible, row, num_rows, num_columns, score_divisor,
                      logits_ptr, stride_logits_row, NUM_HEADS: tl.constexpr):
    """One row's scores of a key tile by `_qsa_mqa_paged_kernel`'s own arithmetic -- the same dot of the same shapes,
    each column's sum its own -- stored under the row's own horizon."""
    scores = tl.dot(keys, query, out_dtype=tl.float32)
    scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
    score = tl.sum(scores, axis=1) / score_divisor
    live = columns < visible
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(tile_valid & live, score, -float("inf")),
        mask=live & (columns < num_columns) & (row < num_rows),
    )


@triton.jit
def _qsa_mqa_paged_group_kernel(
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
    GROUP: tl.constexpr,
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
    """ST (carry Q8): `_qsa_mqa_paged_kernel` with GROUP (2..4) consecutive rows of ONE request a program. A verify
    step's rows of a sequence, and a prefill segment's, score the same paged keys, and a program a row read every key
    tile once a row. Here a tile is read once, as far as the group's furthest row sees, and each row takes its scores
    from it with the row kernel's own dot -- the same shapes, a column's sum its own -- and stores them under its own
    horizon. So a row's logits are the row kernel's bytes: a key past a nearer row's horizon enters no value that row
    stores. The caller answers for the grouping -- the kernel reads the request of a group's first row only."""
    first = tl.program_id(0) * GROUP
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + first)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    reach = sequence_length // COMPRESS_RATIO
    visible0 = _qsa_visible(query_positions_ptr, first, num_rows, reach, COMPRESS_RATIO)
    visible1 = _qsa_visible(query_positions_ptr, first + 1, num_rows, reach, COMPRESS_RATIO)
    horizon = tl.maximum(visible0, visible1)
    if GROUP > 2:
        visible2 = _qsa_visible(query_positions_ptr, first + 2, num_rows, reach, COMPRESS_RATIO)
        horizon = tl.maximum(horizon, visible2)
    if GROUP > 3:
        visible3 = _qsa_visible(query_positions_ptr, first + 3, num_rows, reach, COMPRESS_RATIO)
        horizon = tl.maximum(horizon, visible3)
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + first, visible0)
        tl.store(visible_blocks_ptr + first + 1, visible1, mask=first + 1 < num_rows)
        if GROUP > 2:
            tl.store(visible_blocks_ptr + first + 2, visible2, mask=first + 2 < num_rows)
        if GROUP > 3:
            tl.store(visible_blocks_ptr + first + 3, visible3, mask=first + 3 < num_rows)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by each row's visible blocks, so columns past the furthest row's need no value.
    if tile_start * BLOCK_N >= horizon:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(horizon, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    query0 = _qsa_load_query(q_ptr, first + 0, num_rows, stride_q_row, stride_q_head, stride_q_dim, heads, dims,
                             NUM_HEADS, HEAD_DIM)
    query1 = _qsa_load_query(q_ptr, first + 1, num_rows, stride_q_row, stride_q_head, stride_q_dim, heads, dims,
                             NUM_HEADS, HEAD_DIM)
    if GROUP > 2:
        query2 = _qsa_load_query(q_ptr, first + 2, num_rows, stride_q_row, stride_q_head, stride_q_dim, heads, dims,
                                 NUM_HEADS, HEAD_DIM)
    if GROUP > 3:
        query3 = _qsa_load_query(q_ptr, first + 3, num_rows, stride_q_row, stride_q_head, stride_q_dim, heads, dims,
                                 NUM_HEADS, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < horizon
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        tile_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=tile_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        _qsa_store_scores(keys, query0, heads, columns, tile_valid, visible0, first + 0, num_rows, num_columns,
                          score_divisor, logits_ptr, stride_logits_row, NUM_HEADS)
        _qsa_store_scores(keys, query1, heads, columns, tile_valid, visible1, first + 1, num_rows, num_columns,
                          score_divisor, logits_ptr, stride_logits_row, NUM_HEADS)
        if GROUP > 2:
            _qsa_store_scores(keys, query2, heads, columns, tile_valid, visible2, first + 2, num_rows, num_columns,
                              score_divisor, logits_ptr, stride_logits_row, NUM_HEADS)
        if GROUP > 3:
            _qsa_store_scores(keys, query3, heads, columns, tile_valid, visible3, first + 3, num_rows, num_columns,
                              score_divisor, logits_ptr, stride_logits_row, NUM_HEADS)


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
    SORTED: tl.constexpr = False,
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
        if SORTED:
            # the rows arrive ascending, -1 last (`sorted_blocks`): the order the sort below makes of any selection
            ordered = tl.where(chosen < 0, 2147483647, chosen)
        else:
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
def _qsa_covered_query(q_ptr, row, num_rows, first_head, head_offsets, dim_offsets, stride_q_row, stride_q_head,
                       GROUP_SIZE: tl.constexpr):
    return tl.load(
        q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head + dim_offsets[None, :],
        mask=(row < num_rows) & (head_offsets[:, None] < GROUP_SIZE),
        other=0.0,
    )


@triton.jit
def _qsa_covered_tile(query, keys, values, valid, max_value, normalizer, accumulator, HEAD_DIM: tl.constexpr):
    """One row's online-softmax step over a K/V tile: `_qsa_sparse_paged_gqa_splitk_kernel`'s, operation for operation.
    A column that is not the row's -- past its position, though another row of the run reads it -- scores -1e20 and
    weighs an exact zero, as a masked load's did."""
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
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
    return next_max, normalizer, accumulator


@triton.jit
def _qsa_covered_store(row, num_rows, split_id, first_head, head_offsets, dim_offsets, max_value, normalizer,
                       accumulator, partial_output_ptr, partial_lse_ptr, output_ptr, stride_output_row,
                       stride_output_head, gate_ptr, stride_gate_row, stride_gate_head, GROUP_SIZE: tl.constexpr,
                       HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr,
                       GATED: tl.constexpr):
    """A row's result as `_qsa_sparse_paged_gqa_splitk_kernel` leaves it: the final store (the output gate applied in
    it) of a one-split launch, or the split's partial output and log-sum-exp for `_qsa_merge_splitk_kernel`."""
    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = (row < num_rows) & (head_offsets[:, None] < GROUP_SIZE)
    if NUM_SPLITS == 1:
        value = normalized_output
        if GATED:
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
            mask=(row < num_rows) & (head_offsets < GROUP_SIZE),
        )


@triton.jit
def _qsa_covered_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_gate_row,
    stride_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    num_lengths,
    GROUP: tl.constexpr,
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
    GATED: tl.constexpr,
) -> None:
    """ST (carry Q10): `_qsa_sparse_paged_gqa_splitk_kernel` for a step the budget covers, GROUP (1..4) consecutive
    rows of ONE request a program. A covered row attends every position up to its own -- its chosen blocks expand to
    column c holding position c -- so nothing is chosen, loaded, sorted or gathered: a tile's positions are its
    columns. A run reads each K/V tile once, as far as its furthest row sees, and stops there instead of walking the
    budget's 2,051 columns; each row steps its own softmax over the tile with the sparse kernel's operations on the
    sparse launch's tiles and splits, a column past its position masked as a missing one was. So a row's output is the
    sparse launch's bytes. The caller answers for the step being covered and for the runs."""
    first = tl.program_id(0) * GROUP
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + first)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + tl.minimum(tl.maximum(request, 0), num_lengths - 1),
        mask=(request >= 0) & (request < num_lengths),
        other=0,
    )
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE

    position0 = tl.load(query_positions_ptr + first + 0, mask=first + 0 < num_rows, other=-1)
    query0 = _qsa_covered_query(q_ptr, first + 0, num_rows, first_head, head_offsets, dim_offsets, stride_q_row,
                                 stride_q_head, GROUP_SIZE)
    max0 = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    norm0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    if GROUP > 1:
        position1 = tl.load(query_positions_ptr + first + 1, mask=first + 1 < num_rows, other=-1)
        query1 = _qsa_covered_query(q_ptr, first + 1, num_rows, first_head, head_offsets, dim_offsets, stride_q_row,
                                     stride_q_head, GROUP_SIZE)
        max1 = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
        norm1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    if GROUP > 2:
        position2 = tl.load(query_positions_ptr + first + 2, mask=first + 2 < num_rows, other=-1)
        query2 = _qsa_covered_query(q_ptr, first + 2, num_rows, first_head, head_offsets, dim_offsets, stride_q_row,
                                     stride_q_head, GROUP_SIZE)
        max2 = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
        norm2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    if GROUP > 3:
        position3 = tl.load(query_positions_ptr + first + 3, mask=first + 3 < num_rows, other=-1)
        query3 = _qsa_covered_query(q_ptr, first + 3, num_rows, first_head, head_offsets, dim_offsets, stride_q_row,
                                     stride_q_head, GROUP_SIZE)
        max3 = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
        norm3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    horizon = position0 + 1
    if GROUP > 1:
        horizon = tl.maximum(horizon, position1 + 1)
    if GROUP > 2:
        horizon = tl.maximum(horizon, position2 + 1)
    if GROUP > 3:
        horizon = tl.maximum(horizon, position3 + 1)
    horizon = tl.minimum(tl.minimum(horizon, sequence_length), TOPK)

    # The sparse launch's tiles and splits; a split stops where the run's furthest row does (the tiles past it hold
    # nothing of any row, and a tile of nothing leaves a softmax's state as it was).
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = tl.minimum((split_id + 1) * NUM_TILES // NUM_SPLITS, tl.cdiv(horizon, BLOCK_N))
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        tile_valid = (
            (request >= 0)
            & (request < num_requests)
            & (columns < horizon)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=tile_valid,
            other=-1,
        )
        tile_valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=tile_valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=tile_valid[:, None],
            other=0.0,
        )
        max0, norm0, acc0 = _qsa_covered_tile(query0, keys, values, tile_valid & (columns <= position0),
                                                   max0, norm0, acc0, HEAD_DIM)
        if GROUP > 1:
            max1, norm1, acc1 = _qsa_covered_tile(query1, keys, values, tile_valid & (columns <= position1),
                                                       max1, norm1, acc1, HEAD_DIM)
        if GROUP > 2:
            max2, norm2, acc2 = _qsa_covered_tile(query2, keys, values, tile_valid & (columns <= position2),
                                                       max2, norm2, acc2, HEAD_DIM)
        if GROUP > 3:
            max3, norm3, acc3 = _qsa_covered_tile(query3, keys, values, tile_valid & (columns <= position3),
                                                       max3, norm3, acc3, HEAD_DIM)

    _qsa_covered_store(first + 0, num_rows, split_id, first_head, head_offsets, dim_offsets, max0, norm0, acc0,
                           partial_output_ptr, partial_lse_ptr, output_ptr, stride_output_row, stride_output_head,
                           gate_ptr, stride_gate_row, stride_gate_head, GROUP_SIZE, HEAD_DIM, NUM_QUERY_HEADS,
                           NUM_SPLITS, GATED)
    if GROUP > 1:
        _qsa_covered_store(first + 1, num_rows, split_id, first_head, head_offsets, dim_offsets, max1, norm1, acc1,
                               partial_output_ptr, partial_lse_ptr, output_ptr, stride_output_row, stride_output_head,
                               gate_ptr, stride_gate_row, stride_gate_head, GROUP_SIZE, HEAD_DIM, NUM_QUERY_HEADS,
                               NUM_SPLITS, GATED)
    if GROUP > 2:
        _qsa_covered_store(first + 2, num_rows, split_id, first_head, head_offsets, dim_offsets, max2, norm2, acc2,
                               partial_output_ptr, partial_lse_ptr, output_ptr, stride_output_row, stride_output_head,
                               gate_ptr, stride_gate_row, stride_gate_head, GROUP_SIZE, HEAD_DIM, NUM_QUERY_HEADS,
                               NUM_SPLITS, GATED)
    if GROUP > 3:
        _qsa_covered_store(first + 3, num_rows, split_id, first_head, head_offsets, dim_offsets, max3, norm3, acc3,
                               partial_output_ptr, partial_lse_ptr, output_ptr, stride_output_row, stride_output_head,
                               gate_ptr, stride_gate_row, stride_gate_head, GROUP_SIZE, HEAD_DIM, NUM_QUERY_HEADS,
                               NUM_SPLITS, GATED)


@triton.jit
def _qsa_covered_stacked_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    output_ptr,
    gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_gate_row,
    stride_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    num_lengths,
    ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GATED: tl.constexpr,
) -> None:
    """`_qsa_covered_paged_gqa_kernel` with a program's rows stacked into one dot: ROWS consecutive rows of ONE request,
    every head of the KV head's group, as the M rows of a [BLOCK_M, HEAD_DIM] query tile (M row m: the run's row
    m // GROUP_SIZE, its head m % GROUP_SIZE) -- the run kernel keeps a row's six heads in a 16-row MMA of its own. Each
    M row steps the sparse kernel's softmax, operation for operation, on the sparse launch's tiles, a column past its
    row's position masked as a missing one was, so a row's output is the sparse launch's bytes. One split: the stacked
    launch serves a prefill segment's many rows. The programs run from the last run back: the latest rows see the most
    columns, so they start first."""
    first = (tl.num_programs(0) - 1 - tl.program_id(0)) * ROWS
    kv_head = tl.program_id(1)
    request = tl.load(token_to_req_ptr + first)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + tl.minimum(tl.maximum(request, 0), num_lengths - 1),
        mask=(request >= 0) & (request < num_lengths),
        other=0,
    )
    stacked = tl.arange(0, BLOCK_M)
    member = stacked // GROUP_SIZE
    head = kv_head * GROUP_SIZE + stacked % GROUP_SIZE
    row = first + member
    live = (member < ROWS) & (row < num_rows)
    position = tl.load(query_positions_ptr + row, mask=live, other=-1)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    query = tl.load(
        q_ptr + row[:, None] * stride_q_row + head[:, None] * stride_q_head + dim_offsets[None, :],
        mask=live[:, None],
        other=0.0,
    )
    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    horizon = tl.minimum(tl.minimum(tl.max(position) + 1, sequence_length), TOPK)
    for tile in range(0, tl.cdiv(horizon, BLOCK_N)):
        columns = tile * BLOCK_N + column_offsets
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        tile_valid = (
            (request >= 0)
            & (request < num_requests)
            & (columns < horizon)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=tile_valid,
            other=-1,
        )
        tile_valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=tile_valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=tile_valid[:, None],
            other=0.0,
        )
        # `_qsa_covered_tile`'s step with each M row's own mask
        valid = tile_valid[None, :] & (columns[None, :] <= position[:, None])
        scores = tl.dot(query, keys)
        scores *= softmax_scale_log2
        scores = tl.where(valid, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid, tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    # `_qsa_covered_store`'s one-split store
    normalized_output = tl.where(
        (normalizer > 0)[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    value = normalized_output
    if GATED:
        gate = tl.load(
            gate_ptr + row[:, None] * stride_gate_row + head[:, None] * stride_gate_head + dim_offsets[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        value = normalized_output.to(output_ptr.dtype.element_ty).to(tl.float32) * tl.sigmoid(gate)
    tl.store(
        output_ptr + row[:, None] * stride_output_row + head[:, None] * stride_output_head + dim_offsets[None, :],
        value,
        mask=live[:, None],
    )


@triton.jit
def _qsa_runs_tile(query, token, mine, request, safe_request, sequence_length, num_requests, num_cache_blocks,
                   k_cache_ptr, v_cache_ptr, block_table_ptr, stride_k_block, stride_k_token, stride_k_head,
                   stride_v_block, stride_v_token, stride_v_head, stride_table_req, kv_head, dim_offsets, max_value,
                   normalizer, accumulator, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
                   HEAD_DIM: tl.constexpr):
    """One tile of `_qsa_sparse_runs_kernel`: the positions `token` [BLOCK_N] (-1: none) gathered through the page
    table, and each M row's softmax stepped over the columns `mine` [BLOCK_M, BLOCK_N] gives it -- the sparse kernel's
    operations, a column that is not the row's scoring -1e20 and weighing an exact zero as a missing one does."""
    safe_token = tl.maximum(token, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token % PAGE_SIZE
    valid = (
        (request >= 0)
        & (request < num_requests)
        & (token >= 0)
        & (token < sequence_length)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
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
    take = valid[None, :] & mine
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    scores = tl.dot(query, keys)
    # Scaling scores avoids re-quantizing a scaled query to BF16.
    scores *= softmax_scale_log2
    scores = tl.where(take, scores, -1.0e20)
    next_max = tl.maximum(max_value, tl.max(scores, axis=1))
    alpha = tl.math.exp2(max_value - next_max)
    probabilities = tl.where(take, tl.math.exp2(scores - next_max[:, None]), 0.0)
    accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
    normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
    return next_max, normalizer, accumulator


@triton.jit
def _qsa_sparse_runs_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    output_ptr,
    gate_ptr,
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
    stride_gate_row,
    stride_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    num_lengths,
    RUN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    UNION: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    GATED: tl.constexpr,
) -> None:
    """`_qsa_sparse_paged_gqa_splitk_kernel` (FROM_BLOCKS, one split) for RUN consecutive rows of ONE request a program:
    the rows' chosen blocks merged into one ascending union -- each block once, tagged with the rows that chose it --
    read tile by tile, every row's heads stacked into the M rows of one dot (M row m: the run's row m // GROUP_SIZE, its
    head m % GROUP_SIZE), a column weighing in only for the rows whose block it is; then the rows' open groups (the
    positions after their complete ones, which no selection holds) in one last tile. Each M row steps the sparse
    kernel's softmax over its own columns in ascending order, so it attends what the sparse launch attends; its sums
    round at the union's tile boundaries, not its own, so its bytes are the band's. A selection holds complete groups
    the row sees, as every selector's does (so the sparse kernel's first `complete` slots are all of its ids); the
    caller answers for the run being one request's consecutive positions."""
    first = tl.program_id(0) * RUN
    kv_head = tl.program_id(1)
    request = tl.load(token_to_req_ptr + first)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + tl.minimum(tl.maximum(request, 0), num_lengths - 1),
        mask=(request >= 0) & (request < num_lengths),
        other=0,
    )
    MAXI: tl.constexpr = 2147483647
    OWN: tl.constexpr = 1 << RUN                                   # a union entry: block * OWN + the rows that chose it

    # the run's selections, one row after another (row i's slot j at i * BLOCK_TOPK + j), keyed (block, row) so a
    # block's choosers sort side by side
    entry = tl.arange(0, UNION)
    member = entry // BLOCK_TOPK
    row_live = (first + member) < num_rows
    position = tl.load(query_positions_ptr + first + member, mask=row_live, other=-1)
    sees = tl.minimum((position + 1) // COMPRESS_RATIO, sequence_length // COMPRESS_RATIO)
    chosen = tl.load(indices_ptr + (first + member) * stride_indices_row + entry % BLOCK_TOPK, mask=row_live, other=-1)
    keyed = tl.sort(tl.where(row_live & (chosen >= 0) & (chosen < sees), chosen * RUN + member, MAXI), dim=0)
    live = keyed != MAXI
    block = keyed // RUN
    owners = tl.where(live, 1 << (keyed % RUN), 0)
    for step in tl.static_range(1, RUN):
        ahead = tl.gather(keyed, tl.minimum(entry + step, UNION - 1), 0)
        same = live & (entry + step < UNION) & (ahead != MAXI) & (ahead // RUN == block)
        owners = owners | tl.where(same, 1 << (ahead % RUN), 0)
    behind = tl.gather(keyed, tl.maximum(entry - 1, 0), 0)
    repeat = (entry > 0) & (behind != MAXI) & (behind // RUN == block)
    union = tl.sort(tl.where(live & ~repeat, block * OWN + owners, MAXI), dim=0)
    in_union = tl.sum((union != MAXI).to(tl.int32), axis=0)

    stacked = tl.arange(0, BLOCK_M)
    row_of = stacked // GROUP_SIZE
    head = kv_head * GROUP_SIZE + stacked % GROUP_SIZE
    row = first + row_of
    live_m = (row_of < RUN) & (row < num_rows)
    row_position = tl.load(query_positions_ptr + row, mask=live_m, other=-1)
    tail_start = ((row_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    query = tl.load(
        q_ptr + row[:, None] * stride_q_row + head[:, None] * stride_q_head + dim_offsets[None, :],
        mask=live_m[:, None],
        other=0.0,
    )
    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    for tile in range(0, tl.cdiv(in_union * COMPRESS_RATIO, BLOCK_N)):
        columns = tile * BLOCK_N + column_offsets
        picked = tl.gather(union, tl.minimum(columns // COMPRESS_RATIO, UNION - 1), 0)
        present = (columns < in_union * COMPRESS_RATIO) & (picked != MAXI)
        token = tl.where(present, (picked // OWN) * COMPRESS_RATIO + columns % COMPRESS_RATIO, -1)
        mine = ((tl.where(present, picked % OWN, 0)[None, :] >> row_of[:, None]) & 1) == 1
        max_value, normalizer, accumulator = _qsa_runs_tile(
            query, token, mine & live_m[:, None], request, safe_request, sequence_length, num_requests,
            num_cache_blocks, k_cache_ptr, v_cache_ptr, block_table_ptr, stride_k_block, stride_k_token, stride_k_head,
            stride_v_block, stride_v_token, stride_v_head, stride_table_req, kv_head, dim_offsets, max_value, normalizer,
            accumulator, PAGE_SIZE, PAGE_TABLE_WIDTH, HEAD_DIM)
    # the open groups, one tile from the run's lowest tail start to its furthest row: a row's own from its tail start
    # up to its position (the sparse kernel's tail columns)
    lowest_tail = tl.min(tl.where(live_m, tail_start, MAXI), axis=0)
    furthest = tl.max(tl.where(live_m, row_position, -1), axis=0)
    token = tl.where(lowest_tail + column_offsets <= furthest, lowest_tail + column_offsets, -1)
    mine = ((token[None, :] >= tail_start[:, None]) & (token[None, :] <= row_position[:, None])
            & (token[None, :] - tail_start[:, None] < COMPRESS_RATIO - 1) & live_m[:, None])
    max_value, normalizer, accumulator = _qsa_runs_tile(
        query, token, mine, request, safe_request, sequence_length, num_requests, num_cache_blocks, k_cache_ptr,
        v_cache_ptr, block_table_ptr, stride_k_block, stride_k_token, stride_k_head, stride_v_block, stride_v_token,
        stride_v_head, stride_table_req, kv_head, dim_offsets, max_value, normalizer, accumulator, PAGE_SIZE,
        PAGE_TABLE_WIDTH, HEAD_DIM)

    normalized_output = tl.where(
        (normalizer > 0)[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    value = normalized_output
    if GATED:
        gate = tl.load(
            gate_ptr + row[:, None] * stride_gate_row + head[:, None] * stride_gate_head + dim_offsets[None, :],
            mask=live_m[:, None],
            other=0.0,
        ).to(tl.float32)
        value = normalized_output.to(output_ptr.dtype.element_ty).to(tl.float32) * tl.sigmoid(gate)
    tl.store(
        output_ptr + row[:, None] * stride_output_row + head[:, None] * stride_output_head + dim_offsets[None, :],
        value,
        mask=live_m[:, None],
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
    # ONE store an address. The norm used to be stored over the whole head and the rotation over its first 2 * R2
    # channels afterwards: two stores to one address are ordered only when the same GPU thread makes both, and at
    # D 128 under four warps channels 32..63 are another thread's in the wide store than in the narrow one. On a GB10
    # 7 of 2000 boot qualifies lost that race -- one head's second rotated half left as the norm, not repeating
    # (measurements/qwen38_qualify_soak_20260919). The rotation is computed from the inputs, never read back, so
    # leaving those channels out of this store changes no byte.
    tl.store(out + d, ((x * scale) * (1.0 + w)).to(OUT.dtype.element_ty), mask=m & (d >= 2 * R2))
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
                    BR: tl.constexpr, pos_h=0, pos_w=0, MROPE: tl.constexpr = False):
    # _norm_rope_partial's program, line for line, for one head at `base` into `out` (both unit-stride along the head)
    # when `live`: the fused QSA input launches below run it for several heads and caches in one program. `MROPE`: the
    # text model's interleaved multimodal rotary -- rotary pair i turns at the position on axis i % 3 (t, h, w: the
    # sections 11/11/10 of 32 pairs are exactly that interleave), `pos` being the t axis; a text token's three axes are
    # one position, and the text path compiles without this branch
    d = tl.arange(0, BD)
    m = (d < D) & live
    x = tl.load(base + d, mask=m, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / D + EPS)
    w = tl.load(W + d, mask=m, other=0.0).to(tl.float32)
    # one store an address, as in _norm_rope_partial: here a lost race would stay in the K cache and the index keys
    tl.store(out + d, ((x * scale) * (1.0 + w)).to(out.dtype.element_ty), mask=m & (d >= 2 * R2))
    i = tl.arange(0, BR)
    mi = (i < R2) & live
    xl = tl.load(base + i, mask=mi, other=0.0).to(tl.float32)
    xh = tl.load(base + R2 + i, mask=mi, other=0.0).to(tl.float32)
    wl = tl.load(W + i, mask=mi, other=0.0).to(tl.float32)
    wh = tl.load(W + R2 + i, mask=mi, other=0.0).to(tl.float32)
    lo = ((xl * scale) * (1.0 + wl)).to(out.dtype.element_ty).to(tl.float32)
    hi = ((xh * scale) * (1.0 + wh)).to(out.dtype.element_ty).to(tl.float32)
    if MROPE:
        axis = i % 3
        angle = tl.where(axis == 0, pos, tl.where(axis == 1, pos_h, pos_w)).to(tl.float32) * tl.load(INV + i, mask=mi, other=0.0)
    else:
        angle = pos.to(tl.float32) * tl.load(INV + i, mask=mi, other=0.0)
    cos, sin = tl.cos(angle), tl.sin(angle)
    tl.store(out + i, (lo * cos - hi * sin).to(out.dtype.element_ty), mask=mi)
    tl.store(out + R2 + i, (lo * sin + hi * cos).to(out.dtype.element_ty), mask=mi)


@triton.jit
def _qsa_inputs_kernel(QG, KX, VX, IQX, IKX, POS, INV, WQ, WK, WIQ, QOUT, IQOUT, KC, VC, RC, KV_SLOTS, RING_SLOTS,
                       sQGr, sQGh, sKr, sVr, sIQr, sIQh, sIKr, sQOr, sIQOr, sP, sPA, sKCb, sKCt, sVCb, sVCt, sRCb, sRCt,
                       kv_pages, ring_pages, EPS, HQ: tl.constexpr, HI: tl.constexpr, D: tl.constexpr,
                       DI: tl.constexpr, R2: tl.constexpr, KV_PAGE: tl.constexpr, RING_PAGE: tl.constexpr,
                       BD: tl.constexpr, BDI: tl.constexpr, BR: tl.constexpr, MROPE: tl.constexpr = False):
    # One program a (row, query head): the query head's norm and rotation, the index query head's (h < HI), and at
    # h == 0 the key head's straight into the K cache, the value row into V and the raw index key into the key ring --
    # norm_rope_partial x 3 and qsa_store_cache_rows x 3, the same arithmetic and addresses (one KV head a rank)
    r = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(POS + r * sP)
    pos_h, pos_w = pos, pos
    if MROPE:                                        # positions [3, N]: the h and w axes a row below
        pos_h = tl.load(POS + sPA + r * sP)
        pos_w = tl.load(POS + 2 * sPA + r * sP)
    _norm_rope_into(QG + r * sQGr + h * sQGh, WQ, pos, INV, QOUT + r * sQOr + h * D, h < HQ, EPS, D, R2, BD, BR,
                    pos_h, pos_w, MROPE)
    _norm_rope_into(IQX + r * sIQr + h * sIQh, WIQ, pos, INV, IQOUT + r * sIQOr + h * DI, h < HI, EPS, DI, R2, BDI,
                    BR, pos_h, pos_w, MROPE)
    # qsa_store_cache_rows' address and validity: int64 before the page stride
    kv_slot = tl.load(KV_SLOTS + r)
    kv_live = (h == 0) & (kv_slot >= 0) & (kv_slot < kv_pages * KV_PAGE)
    kv_block = (tl.maximum(kv_slot, 0) // KV_PAGE).to(tl.int64)
    kv_token = tl.maximum(kv_slot, 0) % KV_PAGE
    _norm_rope_into(KX + r * sKr, WK, pos, INV, KC + kv_block * sKCb + kv_token * sKCt, kv_live, EPS, D, R2, BD, BR,
                    pos_h, pos_w, MROPE)
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
                           BLOCK_D: tl.constexpr, R2: tl.constexpr, BR: tl.constexpr, KEY_PAGE: tl.constexpr,
                           ROPE=None, sRR=0, sRA=0, HAS_ROPE: tl.constexpr = False, MROPE: tl.constexpr = False):
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
    first_h, first_w = first_position, first_position
    if HAS_ROPE:
        # the rotary position of the group's first member, as the caller computed it (a picture's positions are not
        # the cache's): [N] for text, [3, N] with MROPE
        first_position = tl.where(valid_row, tl.load(ROPE + row * sRR, mask=valid_row, other=0), 0)
        first_h, first_w = first_position, first_position
        if MROPE:
            first_h = tl.where(valid_row, tl.load(ROPE + sRA + row * sRR, mask=valid_row, other=0), 0)
            first_w = tl.where(valid_row, tl.load(ROPE + 2 * sRA + row * sRR, mask=valid_row, other=0), 0)
    key_live = (compressed_slot >= 0) & (compressed_slot < key_pages * KEY_PAGE)
    key_block = (tl.maximum(compressed_slot, 0) // KEY_PAGE).to(tl.int64)
    key_token = tl.maximum(compressed_slot, 0) % KEY_PAGE
    _norm_rope_into(pooled, W, first_position, INV, KEYS + key_block * sKb + key_token * sKt, key_live, EPS,
                    HEAD_DIM, R2, BLOCK_D, BR, first_h, first_w, MROPE)


def _paged_rows(cache: torch.Tensor, what: str, width: int) -> None:
    if (cache.ndim != 4 or cache.shape[2] != 1 or cache.shape[3] != width or not all(cache.shape)
            or cache.stride(3) != 1):
        raise ValueError(f"{what} must be [pages, page_size, 1, {width}] with a unit stride along the row")


def qsa_index_keys(raw_keys, compressor_state_cache, compressor_state_block_table, token_to_req, query_start_loc,
                   logical_positions, compressed_slots, compress_ratio, weight, eps, theta, rotary_dim, key_cache,
                   rope_first=None):
    """The index keys a step's rows close, in one launch: qsa_compress_groups_with_ratio (no rope cache), then
    norm_rope_partial at each group's first position, then qsa_store_cache_rows at `compressed_slots` (-1 skipped) --
    the same arithmetic and bytes. `raw_keys` is [rows, head_dim] with a unit stride along the head (the in_proj
    columns). Launch it before this step's raw keys go into the ring: the pooled groups read the ring's older members,
    and a long prefill's ring writes cover every ring cell.

    `rope_first`: the rotary position of each row's group's first member when it is not the cache's -- int64 [rows],
    or [3, rows] (t, h, w) for the text model's interleaved multimodal rotary; read on the rows that close a group.
    None rotates at the cache position (`logical_positions` - ratio + 1), a text-only sequence's."""
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
    rope, s_rr, s_ra, mrope = _rope_axes(rope_first, rows, "QSA index keys' first-member positions")
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
        KEY_PAGE=key_cache.shape[1], ROPE=rope, sRR=s_rr, sRA=s_ra, HAS_ROPE=rope is not None, MROPE=mrope,
        num_warps=_input_warps(),
    )


def qsa_inputs(q, k, v, iq, ik, positions, q_norm, k_norm, iq_norm, eps, theta, rotary_dim, k_cache, v_cache, kv_slots,
               ring, ring_slots):
    """A QSA layer's inputs in one launch: norm_rope_partial of the query heads q [N, Hq, D] and of the index query
    heads iq [N, Hi, Di] (both returned), of the key head k [N, 1, D] straight into k_cache, and qsa_store_cache_rows of
    v [N, 1, D] into v_cache at `kv_slots` and of the raw index keys ik [N, Di] into the key ring at `ring_slots` (-1
    skipped) -- the same arithmetic and bytes as those six launches. Views are read through their strides, unit along
    the head. The ring write belongs after qsa_index_keys, which reads the ring.

    `positions` are the rotary positions: int64 [N], or [3, N] (t, h, w) for a step whose rows include a picture's
    (the text model's interleaved multimodal rotary; engine/profiles/qwen38/vision.rope_positions). The caches are
    addressed by `kv_slots` and `ring_slots`, never by these."""
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
    if positions.dtype != torch.int64 or positions.shape not in ((rows,), (3, rows)):
        raise ValueError("QSA inputs take int64 positions [N] or [3, N]")
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
    _, s_p, s_pa, mrope = _rope_axes(positions, rows, "QSA input positions")
    _packed_rows("QSA inputs", positions if positions.ndim == 1 else positions[0], kv_slots, ring_slots)
    q_out = torch.empty(rows, hq, dim, device=q.device, dtype=q.dtype)
    iq_out = torch.empty(rows, hi, di, device=q.device, dtype=q.dtype)
    if not rows:
        return q_out, iq_out
    from engine.kernels.common.norm_rope import warm
    inv = warm(q.device, rotary_dim, theta)
    _qsa_inputs_kernel[(rows, hq)](
        q, k, v, iq, ik, positions, inv, q_norm, k_norm, iq_norm, q_out, iq_out, k_cache, v_cache, ring, kv_slots,
        ring_slots, q.stride(0), q.stride(1), k.stride(0), v.stride(0), iq.stride(0), iq.stride(1), ik.stride(0),
        q_out.stride(0), iq_out.stride(0), s_p, s_pa, k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1), ring.stride(0), ring.stride(1), k_cache.shape[0], ring.shape[0], eps,
        HQ=hq, HI=hi, D=dim, DI=di, R2=rotary_dim // 2, KV_PAGE=k_cache.shape[1], RING_PAGE=ring.shape[1],
        BD=triton.next_power_of_2(dim), BDI=triton.next_power_of_2(di), BR=triton.next_power_of_2(rotary_dim // 2),
        MROPE=mrope, num_warps=_input_warps(),
    )
    return q_out, iq_out


def _rope_axes(positions, rows: int, what: str):
    """(tensor, row stride, axis stride, mrope) for rotary positions given as int64 [rows] or [3, rows] (t, h, w);
    None passes through as (None, 0, 0, False). Rows are read at `ptr + row * stride`, the axes `axis_stride` apart."""
    if positions is None:
        return None, 0, 0, False
    if positions.dtype != torch.int64 or positions.shape not in ((rows,), (3, rows)):
        raise ValueError(f"{what} must be int64 [{rows}] or [3, {rows}]")
    if positions.ndim == 1:
        return positions, positions.stride(0), 0, False
    return positions, positions.stride(1), positions.stride(0), True


def _input_warps() -> int:
    """The warps of a layer's two input launches (qsa_index_keys, qsa_inputs): 4, or the probe hook's."""
    if _INPUT_WARPS_OVERRIDE is None:
        return 4
    return _forced_geometry("_INPUT_WARPS_OVERRIDE", _INPUT_WARPS_OVERRIDE, 1)[0]


def _score_profile(rows: int):
    """(tile width, tiles a program, warps) of a scoring launch over `rows` rows. A decode step's few rows keep
    upstream's 64-wide tile a program at 2 warps: on a GB10 nothing in the grid beat it by more than the noise beside
    production (carry Q9, measurements/qwen38_qsa_geometry_20260919). Prefill's many take 32 tiles of 128 columns a
    program at 4 warps in place of upstream's eight of 64 at 2: 6%, 12% and 16% less time at the 4K, 32K and 256K
    buckets. The scores are the same bytes at every geometry of that record -- a tile is a loop bound, a column's
    dot and its sum over the heads are its own."""
    if _SCORE_PROFILE_OVERRIDE is not None:
        return _forced_geometry("_SCORE_PROFILE_OVERRIDE", _SCORE_PROFILE_OVERRIDE, 3)
    return (64, 1, 2) if rows <= 32 else (128, 32, 4)


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def _packed_rows(what: str, *rows: torch.Tensor) -> None:
    """The kernels load one-dimensional row metadata at `ptr + row`, without a stride: a strided view -- an expanded
    one-row mapping has stride 0 -- would be read past its storage."""
    if any(t.numel() > 1 and t.stride(0) != 1 for t in rows):
        raise ValueError(f"{what} needs packed row metadata (stride 1)")


def qsa_mqa_paged(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, compress_ratio,
                  num_columns=None, score_scale=None, *, group: int = 1):
    """Block scores sum_heads relu(q . key) / score_scale straight from a paged compressed-key cache
    [pages, page_size, 1, head_dim]: (logits f32 [rows, columns], visible blocks int32 [rows]). Columns at or past a
    row's visible count are not written.

    `group` (1..4, carry Q8): the rows come in runs of `group` of one request -- a captured step's tokens a row, any
    rows of one prefill segment; the last run may be short -- and a program scores a run from one read of each key
    tile (`_qsa_mqa_paged_group_kernel`). The logits and the visible counts are the row launch's bytes. The caller
    answers for the runs: the request mapping is on the device, and nothing here reads it back to check."""
    _validate_mqa(q)
    if type(group) is not int or not 1 <= group <= 4:
        raise ValueError("QSA scoring groups 1..4 rows of a request a program")
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
    BLOCK_N, tiles_per_program, warps = _score_profile(q.shape[0])
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    MAX_N = max(16, triton.next_power_of_2(q.shape[1]))
    args = (q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, visible_blocks, logits,
            q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
            page_table.stride(0), page_table.stride(1), logits.stride(0), q.shape[0], columns, k_cache.shape[0],
            page_table.shape[0], float(score_divisor))
    shape = dict(PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=page_table.shape[1], NUM_HEADS=q.shape[1],
                 HEAD_DIM=q.shape[2], BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, TILES_PER_PROG=tiles_per_program, STAGES=2,
                 MAX_N=MAX_N, COMPRESS_RATIO=compress_ratio, num_warps=warps)
    tile_programs = triton.cdiv(columns, BLOCK_N * tiles_per_program)
    if group == 1 or q.shape[0] == 1:
        _qsa_mqa_paged_kernel[(q.shape[0], tile_programs)](*args, **shape)
    else:
        _qsa_mqa_paged_group_kernel[(triton.cdiv(q.shape[0], group), tile_programs)](*args, GROUP=group, **shape)
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
    valid prefix with ties to the lower block; any other step on a device -- every captured one -- takes
    engine/kernels/qsa_select's one launch, the same rule; the CPU keeps torch.topk over the masked logits."""
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
    from engine.kernels import qsa_select
    if qsa_select.admits(logits, block_topk) and visible_blocks.dtype == torch.int32 and out.stride(1) == 1:
        return qsa_select.select(logits, visible_blocks.contiguous(), block_topk, out)
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


def shards_select_alike(rows: int, shards, columns: int, block_topk: int, group: int = 1) -> bool:
    """Whether `qsa_select_paged_blocks` over each of `shards` -- the row counts of disjoint row ranges of one step --
    chooses for every row the set one call over the step's `rows` rows chooses (carry Q11: the ranks score a quarter
    of a prefill step's index queries each and gather the ids).

    A row's scores are its own whatever rows share its launch. Its selector is not: `select_blocks` takes the radix
    select (ties to the lower block) where engine/kernels/prefill_topk admits the rows it was handed, a chunk of
    `_LOGITS_WORKSPACE_BYTES` of logits at a time, and torch.topk (ties as its candidates fall) elsewhere -- and just
    past the budget's reach relu leaves enough equal zeros for the two to part at the budget's edge. So a split is the
    whole only where every chunk on both sides takes the radix select, whose choice is a row's alone; nothing else is
    known to order a row's ties alike in launches of different sizes. The tensors' half of the radix rule (CUDA, eager,
    fp32 logits) is the same on both sides and is not asked here. `group`: the runs both sides score in
    (qsa_mqa_paged's), which round a call's rows."""
    from engine.kernels import prefill_topk
    per = _rows_a_scoring_call(columns, group)                            # qsa_select_paged_blocks' rows a scoring call
    return (prefill_topk.admits_calls(rows, per, columns, block_topk)
            and all(prefill_topk.admits_calls(count, per, columns, block_topk) for count in shards if count))


def _rows_a_scoring_call(columns: int, group: int) -> int:
    """The rows one scoring call takes under the logits workspace, whole runs of `group` rows."""
    rows = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    return max(group, rows // group * group)


def qsa_select_paged_tokens(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, token_topk,
                            compress_ratio, out=None, *, group: int = 1):
    """Score, select and expand QSA positions without a host synchronization: int32 [rows, token_topk + ratio - 1].
    `group`: qsa_mqa_paged's."""
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
    rows_per_chunk = _rows_a_scoring_call(columns, group)
    blocks_buffer = torch.empty((min(rows, rows_per_chunk), block_topk), dtype=torch.int32, device=q.device)
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_blocks = qsa_mqa_paged(q[row_slice], k_cache, page_table, token_to_req[row_slice],
                                               query_positions[row_slice], sequence_lengths, compress_ratio,
                                               group=group)
        blocks = select_blocks(logits, visible_blocks, block_topk, blocks_buffer[: row_end - row_start])
        expand_qsa_block_indices_cuda(blocks, query_positions[row_slice], sequence_lengths, token_to_req[row_slice],
                                      compress_ratio, token_topk, out[row_slice])
    return out


def qsa_select_paged_blocks(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, token_topk,
                            compress_ratio, out=None, *, group: int = 1):
    """`qsa_select_paged_tokens` without its expansion: the chosen blocks, int32 [rows, token_topk // compress_ratio]
    (-1 past a row's visible blocks), for `qsa_sparse_paged_attention_blocks` to expand inside its tiles. The scoring
    chunks and the selection are `qsa_select_paged_tokens`' own, writing each chunk's rows in place. `group`:
    qsa_mqa_paged's."""
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
    rows_per_chunk = _rows_a_scoring_call(columns, group)
    for row_start in range(0, rows, rows_per_chunk):
        row_slice = slice(row_start, min(row_start + rows_per_chunk, rows))
        logits, visible_blocks = qsa_mqa_paged(q[row_slice], k_cache, page_table, token_to_req[row_slice],
                                               query_positions[row_slice], sequence_lengths, compress_ratio,
                                               group=group)
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


def sorted_blocks(block_indices: torch.Tensor) -> torch.Tensor:
    """A selection's rows ascending, -1 last -- the order the sparse kernel sorts every row into before it reads it --
    in one launch for all rows, so the attention need not sort a row in each of its programs (`presorted`)."""
    big = torch.iinfo(torch.int32).max
    ordered = torch.where(block_indices < 0, big, block_indices).sort(dim=1).values
    return torch.where(ordered == big, -1, ordered).to(torch.int32)


def qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, block_indices, query_positions, sequence_lengths,
                                      compress_ratio, token_topk, block_table, token_to_req, out=None, *, gate=None,
                                      one_request: bool = False, presorted: bool = False):
    """`qsa_sparse_paged_attention` at the positions `expand_qsa_block_indices_cuda` expands the chosen blocks to --
    int32 [rows, token_topk // compress_ratio] as `qsa_select_paged_blocks` writes them -- computed tile by tile in the
    attention's own launch (FROM_BLOCKS), without the expansion launch and its [rows, token_topk + compress_ratio - 1]
    buffer. The blocks are read in ascending order, -1 last, whatever order a row holds them in: the output is the
    expanded attention over the row's blocks sorted so, and the same for every order of the same set. `one_request`:
    every row is one request's, at consecutive positions (a prefill segment) -- from RUNS_MIN_ROWS rows the run launch
    serves it, RUN_ROWS rows a program over the union of their blocks (`_qsa_sparse_runs_kernel`; the band's bytes)."""
    if token_topk <= 0 or compress_ratio <= 0 or token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    if type(one_request) is not bool:
        raise ValueError("one_request is a declared boolean")
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
                                   blocks=(query_positions, sequence_lengths, compress_ratio, block_topk), gate=gate,
                                   runs=one_request and rows >= RUNS_MIN_ROWS, presorted=presorted)


def _split_profile(rows: int, kv_heads: int, block_m: int, width: int):
    """(tile width, tiles, splits, warps of a split launch) of a sparse attention over `width` columns, as a GB10
    measured it at Qwen3.8's per-rank cell (carry Q9, measurements/qwen38_qsa_geometry_20260919): 16-wide tiles at 4
    warps throughout, and the splits fall as the programs (rows x KV heads) rise -- 64 for a step of one or two rows,
    16 to eight, 4 to 256, and one launch without a merge above. Upstream's profile (tuned on GB300 for the Qwen-Air
    attention shapes) kept 64 and 32 splits to 31 programs and went to 64-wide tiles at 2 warps above: on a GB10 that
    was 3-9% slower from four rows to sixteen, 18-29% at an eager step's 64 to 1,024 rows, 25% over a 4,096-row chunk,
    and half the covered launch's time. Every geometry held the oracle band there; the bytes of a step change with its
    tier, as they always have with its rows. The covered launch takes the profile from the same rows, so its tiles
    and splits are the sparse launch's. `block_m` is kept for the callers: the tiers were measured at one group width."""
    base_programs = rows * kv_heads
    if _SPLIT_PROFILE_OVERRIDE is not None:
        block_n, target_splits, partial_warps = _forced_geometry("_SPLIT_PROFILE_OVERRIDE", _SPLIT_PROFILE_OVERRIDE, 3)
    else:
        block_n, partial_warps = 16, 4
        target_splits = 64 if base_programs <= 2 else 16 if base_programs <= 8 else 4 if base_programs <= 256 else 1
    num_tiles = triton.cdiv(width, block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    return block_n, num_tiles, min(max_useful_splits, target_splits), partial_warps


def _covered_stacked(q, k_cache, v_cache, query_positions, sequence_lengths, block_table, token_to_req, out, gate,
                     width, group_size, block_n):
    """`_qsa_covered_stacked_kernel` over the rows (checked by qsa_covered_paged_attention): STACK_M // group_size rows
    a program, on the sparse launch's tile width (`block_n`, the split profile's) unless the probe hook forces one."""
    stack_m, warps = STACK_M, STACK_WARPS
    if _STACK_OVERRIDE is not None:
        block_n, stack_m, warps = _forced_geometry("_STACK_OVERRIDE", _STACK_OVERRIDE, 3)
    run = stack_m // group_size
    if not run:
        raise ValueError(f"a stacked covered program holds {stack_m} M rows; a KV head's group is {group_size}")
    gated = gate is not None
    gate_rows = gate if gated else out                                    # never read without GATED
    _qsa_covered_stacked_kernel[(triton.cdiv(q.shape[0], run), k_cache.shape[2])](
        q, k_cache, v_cache, block_table, token_to_req, query_positions, sequence_lengths, out, gate_rows,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), block_table.stride(0), out.stride(0), out.stride(1),
        gate_rows.stride(0), gate_rows.stride(1), q.shape[0], k_cache.shape[0], block_table.shape[0],
        sequence_lengths.shape[0],
        ROWS=run, TOPK=width, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], BLOCK_M=stack_m, BLOCK_N=block_n, GATED=gated,
        num_warps=warps, num_stages=2,
    )
    return out


def qsa_covered_paged_attention(q, k_cache, v_cache, query_positions, sequence_lengths, compress_ratio, token_topk,
                                block_table, token_to_req, out=None, *, gate=None, group: int = 1,
                                one_request: bool = False):
    """`qsa_sparse_paged_attention_blocks` for a step the budget covers -- every row sees no more complete groups than
    token_topk // compress_ratio, so it attends every position up to its own -- without any blocks (carry Q10): the
    same bytes from a dense causal launch that reads a run's K/V tiles once and stops where the run's furthest row
    does. `group` (1..4): the rows come in runs of that many of one request (qsa_mqa_paged's). `one_request`: every row
    is one request's (a prefill segment) -- from STACK_MIN_ROWS rows the stacked launch serves it, a run's rows and
    heads the M rows of one dot (`_qsa_covered_stacked_kernel`), the same bytes. The caller answers for all three: a
    row that is not covered would attend only the first token_topk + compress_ratio - 1 positions."""
    if token_topk <= 0 or compress_ratio <= 0 or token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    if type(group) is not int or not 1 <= group <= 4:
        raise ValueError("QSA covered attention groups 1..4 rows of a request a program")
    if type(one_request) is not bool:
        raise ValueError("one_request is a declared boolean")
    if not q.is_cuda:
        raise RuntimeError("paged QSA covered attention runs on CUDA")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA covered attention received invalid Q/K/V shapes")
    rows = q.shape[0]
    if query_positions.shape != (rows,) or token_to_req.shape != (rows,) or block_table.ndim != 2:
        raise ValueError("QSA covered attention metadata has invalid shapes")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if not (query_positions.dtype == sequence_lengths.dtype == torch.int32):
        raise ValueError("QSA covered attention metadata is int32")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA covered attention cache and block table must be nonempty")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA covered attention heads must divide over the cache's")
    if not (q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16):
        raise ValueError("QSA covered attention is BF16")
    if not (q.device == k_cache.device == v_cache.device == block_table.device == token_to_req.device
            == query_positions.device == sequence_lengths.device):
        raise ValueError("QSA covered attention tensors share one device")
    if (q.stride(2) != 1 or k_cache.stride(3) != 1 or v_cache.stride(3) != 1 or block_table.stride(1) != 1):
        raise ValueError("QSA covered attention needs packed head dimensions and packed metadata rows")
    _packed_rows("QSA covered attention", token_to_req, query_positions, sequence_lengths)
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device or out.stride(2) != 1:
        raise ValueError("QSA covered output must match its query")
    if gate is not None and (gate.shape != q.shape or gate.dtype != torch.bfloat16 or gate.device != q.device
                             or gate.stride(2) != 1):
        raise ValueError("the QSA output gate is BF16 in the query's shape with a unit stride along the head")
    if not rows:
        return out
    width = token_topk + compress_ratio - 1
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    block_n, num_tiles, num_splits, partial_warps = _split_profile(rows, k_cache.shape[2], block_m, width)
    if one_request and rows >= STACK_MIN_ROWS:
        return _covered_stacked(q, k_cache, v_cache, query_positions, sequence_lengths, block_table, token_to_req, out,
                                gate, width, group_size, block_n)
    if num_splits == 1:
        partial_output = partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, rows, q.shape[1]), dtype=torch.float32, device=q.device)
    gated = gate is not None
    gate_rows = gate if gated else out                                    # never read without GATED
    _qsa_covered_paged_gqa_kernel[(triton.cdiv(rows, group), k_cache.shape[2], num_splits)](
        q, k_cache, v_cache, block_table, token_to_req, query_positions, sequence_lengths, partial_output, partial_lse,
        out, gate_rows, q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), block_table.stride(0), out.stride(0), out.stride(1),
        gate_rows.stride(0), gate_rows.stride(1), rows, k_cache.shape[0], block_table.shape[0],
        sequence_lengths.shape[0],
        GROUP=group, TOPK=width, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=block_n, GATED=gated and num_splits == 1,
        num_warps=partial_warps, num_stages=2,
    )
    if num_splits == 1:
        return out
    merge_gate = dict(gate_ptr=gate, stride_gate_row=gate.stride(0), stride_gate_head=gate.stride(1), GATED=True) if gated else {}
    _qsa_merge_splitk_kernel[(rows, q.shape[1])](
        partial_output, partial_lse, out, out.stride(0), out.stride(1), rows,
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits), num_warps=2, num_stages=1, **merge_gate,
    )
    return out


def _sparse_runs(q, k_cache, v_cache, block_indices, block_table, token_to_req, out, gate, blocks, group_size,
                 block_n):
    """`_qsa_sparse_runs_kernel` over the rows (checked by qsa_sparse_paged_attention_blocks): RUN_ROWS rows a program
    on the sparse launch's tile width (the split profile's) unless the probe hook forces (run rows, warps)."""
    query_positions, sequence_lengths, compress_ratio, block_topk = blocks
    run, warps = RUN_ROWS, RUN_WARPS
    if _RUNS_OVERRIDE is not None:
        forced = _RUNS_OVERRIDE
        if (type(forced) is not tuple or len(forced) != 2
                or not all(type(n) is int and n > 0 and not n & (n - 1) for n in forced) or forced[0] > 8):
            raise ValueError("_RUNS_OVERRIDE is (run rows 1..8, warps), powers of two")
        run, warps = forced
    if block_topk & (block_topk - 1):
        raise ValueError(f"the run launch merges power-of-two selections; this one holds {block_topk} blocks")
    gated = gate is not None
    gate_rows = gate if gated else out                                    # never read without GATED
    _qsa_sparse_runs_kernel[(triton.cdiv(q.shape[0], run), k_cache.shape[2])](
        q, k_cache, v_cache, block_indices, block_table, token_to_req, query_positions, sequence_lengths, out,
        gate_rows, q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), v_cache.stride(0),
        v_cache.stride(1), v_cache.stride(2), block_indices.stride(0), block_table.stride(0), out.stride(0),
        out.stride(1), gate_rows.stride(0), gate_rows.stride(1), q.shape[0], k_cache.shape[0], block_table.shape[0],
        sequence_lengths.shape[0],
        RUN=run, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2], BLOCK_M=max(16, triton.next_power_of_2(run * group_size)), BLOCK_N=block_n,
        BLOCK_TOPK=block_topk, UNION=run * block_topk, COMPRESS_RATIO=compress_ratio, GATED=gated,
        num_warps=warps, num_stages=2,
    )
    return out


def _sparse_paged_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, width, blocks=None,
                            gate=None, runs=False, presorted=False):
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
    block_n, num_tiles, num_splits, partial_warps = _split_profile(q.shape[0], k_cache.shape[2], block_m, width)
    if runs and blocks is not None:
        return _sparse_runs(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, gate, blocks,
                            group_size, block_n)
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
                         COMPRESS_RATIO=compress_ratio, SORTED=presorted)
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
    from engine.kernels.gated_residual import blame, drift
    from engine.modules.norm import rmsnorm_unit_offset
    from engine.modules.rotary import apply_rope, rope_tables
    gen = torch.Generator(device="cpu").manual_seed(seed)
    worst, found = {}, []
    for h, d in heads:
        w = (torch.randn(d, generator=gen) * 0.1).to(device=device, dtype=dtype)
        key, most = f"norm_rope_{h}x{d}", (0.0, 0.0)
        for n in rows:
            x = torch.randn(n, h, d, generator=gen).to(device=device, dtype=dtype)
            positions = torch.randint(0, max_position, (n,), generator=gen).to(device)

            def ours(x=x, w=w, positions=positions):
                return norm_rope_partial(x, w, eps, positions, theta, rotary_dim)

            def reference(x=x, w=w, positions=positions):
                cos, sin = rope_tables(positions, rotary_dim, theta, dtype=dtype)
                return apply_rope(rmsnorm_unit_offset(x, w, eps), cos, sin)

            got, want = ours(), reference()
            m, r = drift(got, want)
            if m > band_max or r > band_rms:               # (rows, heads, channels): the failure says where and whose
                found.append(f"{key} at {n} rows: " + blame(
                    got, want, ours, reference, lambda: reference(x.cpu(), w.cpu(), positions.cpu()), band_max=band_max))
            most = (max(most[0], m), max(most[1], r))
        worst[key] = most
    bad = {k: v for k, v in worst.items() if v[0] > band_max or v[1] > band_rms}
    if bad:
        raise RuntimeError(f"QSA norm and partial rotation drift from engine/modules beyond max {band_max:g} / "
                           f"rms {band_rms:g}: {bad}. " + " | ".join(found))
    return worst


__all__ = ["norm_rope_partial", "qsa_mqa_paged", "expand_qsa_block_indices_cuda", "select_blocks", "shards_select_alike",
           "qsa_index_keys", "qsa_inputs", "qsa_select_paged_tokens", "qsa_select_paged_blocks",
           "qsa_sparse_paged_attention",
           "qsa_sparse_paged_attention_blocks", "qsa_covered_paged_attention", "qsa_store_cache_rows", "qsa_compress_groups_with_ratio", "qualify"]
