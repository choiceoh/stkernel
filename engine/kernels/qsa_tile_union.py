# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8's QSA sparse attention for a prefill step, a tile of consecutive rows reading the union of their chosen
blocks once (sm_121a intake U12: vLLM PR 55430, RFC vllm#55394).

In prefill, neighbouring rows choose nearly the same compressed blocks (Jaccard ~0.9 at 8k context on
Qwen3.8-Flash-Next, the RFC's count), yet `qsa.qsa_sparse_paged_attention_blocks` gathers every row's selection on its
own and runs the GQA dot at M = one head group. Here R consecutive rows of ONE request form a tile: the kernel walks
the UNION of the tile's chosen blocks, gathers each block once, and applies a per-row membership mask inside the
online softmax, so every row still attends exactly its own selection and its causal tail. The output is the split-K
launch's up to summation order -- the tile walks the union's blocks eight at a time and the tails in a last step,
where the split-K launch walks a row's positions sixteen columns at a time -- so the two are held to each other in a
band, not byte for byte. Measured on GB10 by the PR's author: 1.42-1.5x at the kernel, TTFT -1.7..-2.8%.

    rows -> tiles (`tiles`)              from the segments' row offsets alone: small torch ops, int32, no sync
    pack kernel                          each tile's rows' block ids read once into packed sort keys
    torch.sort over each tile's keys
    build kernel (in place over them)    the union as physical token bases, the [R, N] membership, the count, and
                                         each row's causal tail resolved to physical tokens
    attention kernel                     no page-table reads

The kernels are vLLM's (vllm/models/qwen4_exp/nvidia/ops/qsa_tile_union.py at jschmied/vllm c5d7eba3, PR 55430, open
when taken; SOURCES.json pins it). What changed around them:

- the imports are Triton's own; there is no VLLM_QSA_TILE_UNION environment switch, no device-capability table and
  no "R,BNB,warps,min_rows" override (D11: the kernel package reads no knobs). The engine runs on a GB10 only (D5), so
  the PR's SM121 tile is the one tile, a constant (`TILE`);
- the inputs are the engine's (net._qsa's): the chosen blocks as `qsa.qsa_select_paged_blocks` (or the covered,
  windowed and rank-sharded selections) leave them -- int32 [rows, token_topk // ratio], each row's distinct blocks
  among those it sees in any order, -1 anywhere else -- the positions as StepMeta.positions32 (int32; the PR reads
  vLLM's int64 buffer), the segments' row offsets as StepMeta.starts (vLLM's query_start_loc), the row -> segment map
  as StepMeta.rows_req and the page table as StepMeta.page_table; K and V are the paged regions caches.Qwen38Caches
  presents ([pages, page, kv heads, head_dim], strided views);
- the output gate is optional (GATED), as it is in the split-K launch; with it, the PR's order -- the attention
  rounded to BF16, times sigmoid(gate) in fp32, rounded once -- which is the split-K launch's (carry Q4);
- eligibility is `admits`, host integers only, and `attention` refuses what it does not admit (D3): the caller
  chooses this launch by the step's shape -- a prefill step of at least 1,024 rows and 64 a segment on average
  (`TILE`'s gates) -- as `qsa.select_blocks` chooses its selector by row count. There is no fallback inside;
- left out: the PR's warmup (the engine compiles at a launch; the served lanes' warm pass is its own), its selection
  workspace (the engine already keeps the compact selection; carry Q5) and its dispatch inside the split-K entry.

Not wired into serving: no lane calls `attention`. Where it would go -- net._qsa's scored branch, with the tile layout
built once a step beside StepMeta and shared by every QSA layer (`layout=`) -- and whether it is on by default are the
operator's decision after a GB10 judgment (engine/QWEN38_CARRY.md; CHARTER D17 for the speed).

The caller answers for the selection being a selector's, which nothing here reads back to check: every id a row holds
is attended, where the split-K launch reads a row's smallest min(seen, budget) ids -- the same set whenever the row
holds only blocks it sees ((position + 1) // ratio of them), which is all any selector writes -- and a block repeated
in a row, which no selector writes either, would count once here and twice there.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class Tile:
    """A tile's geometry and the step shapes it pays at."""
    rows: int                       # R: consecutive rows of one request sharing one gather
    blocks: int                     # BNB: union blocks an inner step (BN = blocks * ratio tokens)
    warps: int
    min_rows: int                   # below this many rows a step, the split-K launch is faster
    min_rows_per_request: int = 64  # below this many rows a segment on average the tiles share little and the union
                                    # build is overhead (fragmented batches)

    def __post_init__(self) -> None:
        # The kernels use tl.arange over R * GP, BNB and TAIL_COLS, and the packed key keeps 3 bits for the row within
        # its tile.
        if self.rows not in (1, 2, 4, 8):
            raise ValueError("a tile's rows are 1, 2, 4 or 8")
        if not (0 < self.blocks <= 32) or (self.blocks & (self.blocks - 1)):
            raise ValueError("a tile's blocks an inner step are a power of two <= 32")
        if self.warps not in (1, 2, 4, 8):
            raise ValueError("a tile's warps are 1, 2, 4 or 8")
        if self.min_rows < 0 or self.min_rows_per_request < 0:
            raise ValueError("a tile's row gates are >= 0")


# GB10 (the PR's measurement): R=2 / BN=32 / 4 warps beat R=4 at both 8k and 30k context; R=4 at BN=32 spills (an M=64
# accumulator), BN=128 exceeds the 99 KiB of shared memory a block has.
TILE = Tile(rows=2, blocks=8, warps=4, min_rows=1024, min_rows_per_request=64)

_TILE_UNION_ROW_BITS = 3  # packed key = block_id << 3 | row_in_tile
_SENTINEL_BLOCK_VALUE = 1 << 27  # sorts after every real block id
_SENTINEL_VALUE = (_SENTINEL_BLOCK_VALUE << _TILE_UNION_ROW_BITS) | ((1 << _TILE_UNION_ROW_BITS) - 1)
# Triton kernels may only read module globals declared as constexpr.
_TILE_UNION_SENTINEL_BLOCK = tl.constexpr(_SENTINEL_BLOCK_VALUE)
_TILE_UNION_SENTINEL = tl.constexpr(_SENTINEL_VALUE)


def fits(compress_ratio: int, token_topk: int, page_size: int, table_width: int, cache_pages: int) -> bool:
    """The model and cache constants the kernels handle (the PR's static contract, and its int32 cache bound)."""
    if compress_ratio < 2 or compress_ratio & (compress_ratio - 1):
        return False  # tl.arange(0, CR)
    if token_topk <= 0 or token_topk % compress_ratio:
        return False
    if page_size <= 0 or page_size % compress_ratio:
        return False  # a compressed block must not straddle a page
    if cache_pages <= 0 or cache_pages * page_size >= 2 ** 31:
        return False  # the union stores page * PAGE_SIZE + offset as int32
    # The packed key encodes block ids below the sentinel block.
    return 0 < table_width and table_width * (page_size // compress_ratio) < _SENTINEL_BLOCK_VALUE


def admits(rows: int, requests: int, *, compress_ratio: int, token_topk: int, page_size: int, table_width: int,
           cache_pages: int) -> bool:
    """Whether `attention` takes a step: the kernels' constants (`fits`) and the tile's gates -- at least
    `TILE.min_rows` rows, and `TILE.min_rows_per_request` a segment on average. Host integers only, no device read: the
    caller asks it of the step's shape before any launch and chooses this attention or the split-K one by the answer.
    A captured (decode) step's few rows never pass; a pure-prefill step is the only kind that does (D9)."""
    if requests <= 0 or rows < TILE.min_rows or rows < TILE.min_rows_per_request * requests:
        return False
    return fits(compress_ratio, token_topk, page_size, table_width, cache_pages)


def _tile_union_tail_cols(tile: Tile, compress_ratio: int) -> int:
    # tl.dot needs N >= 16.
    return max(16, triton.next_power_of_2(tile.rows * (compress_ratio - 1)))


# ---------------------------------------------------------------------------
# Kernels (vLLM PR 55430's; ST changes are marked)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_tile_union_pack_kernel(
    block_indices_ptr,
    packed_ptr,
    tile_row0_ptr,
    tile_request_ptr,
    query_start_loc_ptr,
    token_to_req_ptr,
    stride_blocks_row,
    stride_packed,
    num_rows,
    num_requests,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
):
    """One program per tile: read each of the tile's rows' block ids once and
    write the sort input (block_id << 3 | slot, or the sentinel for -1 ids,
    padding rows and rows whose request id is invalid); the N - R * E pad
    columns get the sentinel too."""
    tile = tl.program_id(0)
    row0 = tl.load(tile_row0_ptr + tile)
    has_rows = row0 >= 0
    request = tl.load(tile_request_ptr + tile)
    request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    request_end = tl.load(query_start_loc_ptr + request + 1, mask=has_rows, other=0)
    request_end = tl.where(request == num_requests - 1, num_rows, request_end)
    n_rows = tl.where(has_rows, tl.minimum(R, request_end - row0), 0)
    e = tl.arange(0, E_PAD)
    for r in tl.static_range(R):
        row = tl.maximum(row0, 0) + r
        structural = r < n_rows
        row_request = tl.load(
            token_to_req_ptr + tl.minimum(row, num_rows - 1), mask=structural, other=-1
        )
        live = structural & (row_request == request)
        ids = tl.load(
            block_indices_ptr + row * stride_blocks_row + e,
            mask=live & (e < E),
            other=-1,
        )
        keys = tl.where(ids >= 0, (ids << 3) | r, _TILE_UNION_SENTINEL)
        tl.store(packed_ptr + tile * stride_packed + r * E + e, keys, mask=e < E)
    if N > R * E:
        # N - R * E need not be a power of two (tl.arange requires one).
        pad = tl.arange(0, N)
        tl.store(
            packed_ptr + tile * stride_packed + R * E + pad,
            tl.full((N,), _TILE_UNION_SENTINEL, tl.int32),
            mask=pad < N - R * E,
        )


@triton.jit(
    do_not_specialize=["num_rows", "num_requests", "table_width", "num_cache_blocks"]
)
def _qsa_tile_union_build_kernel(
    keys_ptr,
    mem_ptr,
    cnt_ptr,
    tail_ptr,
    tile_row0_ptr,
    tile_request_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    positions_ptr,
    stride_keys,
    stride_mem_tile,
    stride_mem_row,
    stride_tail,
    stride_table_req,
    num_rows,
    num_requests,
    table_width,
    num_cache_blocks,
    N: tl.constexpr,
    R: tl.constexpr,
    CR: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TAIL_COLS: tl.constexpr,
):
    """Per tile, from its sorted packed keys (block_id << 3 | row,
    sentinel-padded to exactly N), IN PLACE: keys[0:count] become the union's
    physical token bases (page * PAGE_SIZE + offset of the block's first token,
    -1 if the page is invalid); plus the int8 [R, N] membership matrix, the
    union count, and each row's causal-tail tokens resolved the same way."""
    BPP: tl.constexpr = PAGE_SIZE // CR
    tile = tl.program_id(0)
    i = tl.arange(0, N)
    packed = tl.load(keys_ptr + tile * stride_keys + i)
    prev = tl.load(keys_ptr + tile * stride_keys + i - 1, mask=i > 0, other=-8)
    blk = packed // 8
    r = packed % 8
    valid = blk < _TILE_UNION_SENTINEL_BLOCK
    first = (blk != prev // 8) & valid
    pos = tl.cumsum(first.to(tl.int32)) - 1
    row0 = tl.load(tile_row0_ptr + tile)
    has_rows = row0 >= 0
    request = tl.load(tile_request_ptr + tile)
    request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    # Page lookup only for first occurrences (the stored ones).
    logical_page = blk // BPP
    page_ok = first & has_rows & (logical_page < table_width)
    physical_page = tl.load(
        block_table_ptr
        + request * stride_table_req
        + tl.minimum(logical_page, table_width - 1),
        mask=page_ok,
        other=-1,
    )
    page_ok &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    phys = tl.where(page_ok, physical_page * PAGE_SIZE + (blk % BPP) * CR, -1)
    # All loads above precede these stores: the in-place rewrite is safe.
    tl.store(keys_ptr + tile * stride_keys + pos, phys, mask=first)
    tl.store(
        mem_ptr + tile * stride_mem_tile + r * stride_mem_row + pos,
        tl.full((N,), 1, tl.int8),
        mask=valid,
    )
    tl.store(cnt_ptr + tile, tl.sum(first.to(tl.int32)))
    # Causal tails (the expansion kernel's rule: start = ((q + 1) // CR) * CR,
    # count = q + 1 - start < CR), resolved to physical tokens here.
    request_end = tl.load(query_start_loc_ptr + request + 1, mask=has_rows, other=0)
    request_end = tl.where(request == num_requests - 1, num_rows, request_end)
    n_rows = tl.where(has_rows, tl.minimum(R, request_end - row0), 0)
    tt = tl.arange(0, TAIL_COLS)
    r_t = tt // (CR - 1)
    j_t = tt % (CR - 1)
    tmask = r_t < n_rows
    # ST: the engine's positions are int32 (StepMeta.positions32; the PR reads vLLM's int64 buffer) -- the same
    # arithmetic; only the final physical location widens to int64 and narrows, bounded by the int32 cache check in
    # `fits`.
    position = tl.load(
        positions_ptr + tl.minimum(tl.maximum(row0, 0) + r_t, num_rows - 1),
        mask=tmask,
        other=-1,
    )
    tail_start = ((position + 1) // CR) * CR
    tail_count = position + 1 - tail_start
    tail_token = tail_start + j_t
    tail_ok = tmask & (position >= 0) & (j_t < tail_count)
    tail_page = tail_token // PAGE_SIZE
    tail_ok &= tail_page < table_width
    tail_phys_page = tl.load(
        block_table_ptr
        + request * stride_table_req
        + tl.minimum(tail_page, table_width - 1),
        mask=tail_ok,
        other=-1,
    )
    tail_ok &= (tail_phys_page >= 0) & (tail_phys_page < num_cache_blocks)
    tail_phys = tl.where(
        tail_ok, tail_phys_page.to(tl.int64) * PAGE_SIZE + tail_token % PAGE_SIZE, -1
    ).to(tl.int32)
    tl.store(tail_ptr + tile * stride_tail + tt, tail_phys)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_tile_union_attn_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    uni_ptr,
    mem_ptr,
    cnt_ptr,
    tail_ptr,
    tile_row0_ptr,
    tile_request_ptr,
    query_start_loc_ptr,
    token_to_req_ptr,
    out_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_uni,
    stride_mem_tile,
    stride_mem_row,
    stride_tail,
    stride_out_row,
    stride_out_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_requests,
    R: tl.constexpr,
    GP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BNB: tl.constexpr,
    CR: tl.constexpr,
    TAIL_COLS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GATED: tl.constexpr,
):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    M: tl.constexpr = R * GP
    BN: tl.constexpr = BNB * CR
    TAIL_PER_ROW: tl.constexpr = CR - 1
    m_off = tl.arange(0, M)
    r_of_m = m_off // GP
    h_of_m = m_off % GP
    dim_offsets = tl.arange(0, HEAD_DIM)
    b_off = tl.arange(0, BNB)
    j_off = tl.arange(0, CR)
    row0 = tl.load(tile_row0_ptr + tile)
    has_rows = row0 >= 0
    request = tl.load(tile_request_ptr + tile)
    request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    request_end = tl.load(query_start_loc_ptr + request + 1, mask=has_rows, other=0)
    request_end = tl.where(request == num_requests - 1, num_rows, request_end)
    n_rows = tl.where(has_rows, tl.minimum(R, request_end - row0), 0)
    row = tl.maximum(row0, 0) + r_of_m
    rmask = r_of_m < n_rows
    # rmask: the tile's structural rows (always written, zeros if masked);
    # live: rows whose request id is valid. Split-K contract: a row whose
    # request id is invalid (padding) is masked and written as zeros.
    row_request = tl.load(
        token_to_req_ptr + tl.minimum(row, num_rows - 1), mask=rmask, other=-1
    )
    live = rmask & (row_request == request)
    qmask = live & (h_of_m < GROUP_SIZE)
    store_mask = rmask & (h_of_m < GROUP_SIZE)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row[:, None] * stride_q_row
        + (first_head + h_of_m[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=qmask[:, None],
        other=0.0,
    )
    max_value = tl.full((M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((M,), dtype=tl.float32)
    accumulator = tl.zeros((M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    # Pass 1: the tile's union of whole compressed blocks, gathered once,
    # per-row membership from the int8 matrix.
    ubound = tl.load(cnt_ptr + tile)
    for t in range(0, ubound, BNB):
        emask = (t + b_off) < ubound
        phys = tl.load(uni_ptr + tile * stride_uni + t + b_off, mask=emask, other=-1)
        tok2 = tl.where((phys >= 0)[:, None], phys[:, None] + j_off[None, :], -1)
        physical_token = tl.reshape(tok2, (BN,))
        valid = physical_token >= 0
        safe_token = tl.maximum(physical_token, 0)
        # A block never straddles a page (PAGE_SIZE % CR == 0), so page and
        # offset come back from the base without a table lookup.
        safe_page = (safe_token // PAGE_SIZE).to(tl.int64)
        page_offset = safe_token % PAGE_SIZE
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
        memb = tl.load(
            mem_ptr
            + tile * stride_mem_tile
            + r_of_m[:, None] * stride_mem_row
            + t
            + b_off[None, :],
            mask=emask[None, :],
            other=0,
        )
        memt = tl.reshape(tl.broadcast_to(memb[:, :, None], (M, BNB, CR)), (M, BN))
        active = (memt > 0) & valid[None, :] & live[:, None]
        scores = tl.dot(query, keys) * softmax_scale_log2
        scores = tl.where(active, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max
    # Pass 2: each row's causal tail (< CR tokens of its open block), already
    # physical.
    tt = tl.arange(0, TAIL_COLS)
    slot_row = tt // TAIL_PER_ROW
    tail_phys = tl.load(
        tail_ptr + tile * stride_tail + tt, mask=tt < R * TAIL_PER_ROW, other=-1
    )
    valid = tail_phys >= 0
    safe_token = tl.maximum(tail_phys, 0)
    safe_page = (safe_token // PAGE_SIZE).to(tl.int64)
    page_offset = safe_token % PAGE_SIZE
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
    active = (r_of_m[:, None] == slot_row[None, :]) & valid[None, :] & live[:, None]
    scores = tl.dot(query, keys) * softmax_scale_log2
    scores = tl.where(active, scores, -1.0e20)
    next_max = tl.maximum(max_value, tl.max(scores, axis=1))
    alpha = tl.math.exp2(max_value - next_max)
    probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
    accumulator = tl.dot(
        probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None]
    )
    normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
    has_values = normalizer > 0
    result = tl.where(
        has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0
    )
    # Same rounding order as the split-K kernel (ops/qsa.py): round the
    # attention output to BF16 first, then apply the gate in FP32. Gating
    # before the round would make the two kernels differ by a rounding step,
    # not by their selection, and the kernel-vs-kernel test could not tell
    # the two apart.
    result = result.to(tl.bfloat16)
    if GATED:
        # ST: the gate is optional, as in the split-K launch (the PR always gates); ungated, the store is the rounded
        # attention
        output_gate = tl.load(
            output_gate_ptr
            + row[:, None] * stride_output_gate_row
            + (first_head + h_of_m[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=store_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        result = result.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        out_ptr
        + row[:, None] * stride_out_row
        + (first_head + h_of_m[:, None]) * stride_out_head
        + dim_offsets[None, :],
        result,
        mask=store_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def tiles(starts: torch.Tensor, rows: int, requests: int) -> "tuple[torch.Tensor, torch.Tensor, int]":
    """Row -> (tile, slot) from the step's segment row offsets alone (StepMeta.starts, vLLM's query_start_loc): rows
    are contiguous per segment and each segment starts a fresh tile, so tiles never straddle segments. Rows past
    starts[-1] (padding) belong structurally to the last segment and are masked by the row map in the kernels.
    Returns (tile_row0 [T] int32, -1 = unused tile; tile_request [T] int32; T). Everything is int32 and sync-free: a
    step's layout, the same for each of its QSA layers (`attention(layout=)`)."""
    R = TILE.rows
    device = starts.device
    qsl = starts[: requests + 1]
    row = torch.arange(rows, device=device, dtype=torch.int32)
    request = torch.searchsorted(qsl[1:], row, right=True, out_int32=True).clamp_(max=requests - 1)
    lengths = qsl[1:] - qsl[:-1]
    tiles_per_request = (lengths + R - 1) // R
    tile_base = torch.cumsum(tiles_per_request, 0, dtype=torch.int32) - tiles_per_request
    offset = row - qsl[request]
    tile = tile_base[request] + offset // R
    slot = offset % R
    num_tiles = (rows + R - 1) // R + requests  # >= sum(ceil(len / R))
    # One junk entry at index num_tiles absorbs the non-start rows.
    scatter_index = torch.where(slot == 0, tile, num_tiles).to(torch.int64)
    tile_row0 = torch.full((num_tiles + 1,), -1, dtype=torch.int32, device=device)
    tile_row0.scatter_(0, scatter_index, row)
    tile_request = torch.zeros(num_tiles + 1, dtype=torch.int32, device=device)
    tile_request.scatter_(0, scatter_index, request)
    return tile_row0[:num_tiles], tile_request[:num_tiles], num_tiles


def attention(q, k_cache, v_cache, block_indices, query_positions, starts, compress_ratio, token_topk, block_table,
              token_to_req, out=None, *, gate=None, layout=None):
    """`qsa.qsa_sparse_paged_attention_blocks` for a prefill step `admits` takes: each row attends its chosen blocks
    (int32 [rows, token_topk // compress_ratio], -1 anywhere, as the selection leaves them) and its open group's causal
    tail, softmax(q . k / sqrt(head_dim)) v per query head over paged BF16 K/V [pages, page_size, kv_heads, head_dim],
    [rows, heads, head_dim] -- the split-K launch's result up to summation order. `starts` [segments + 1] int32 are the
    segments' row offsets, `token_to_req` [rows] int32 each row's segment, `block_table` [segments, pages] int32. With
    `gate` (BF16, q's shape, unit stride along the head) the output is BF16(attention * sigmoid(gate)), applied in the
    store. `layout`: `tiles(starts, rows, segments)`, when the step's QSA layers share one. Refuses a step `admits`
    does not take (D3: no fallback -- the caller chooses by shape)."""
    if token_topk <= 0 or compress_ratio <= 0 or token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    if not q.is_cuda:
        raise RuntimeError("the QSA tile-union attention runs on CUDA")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape or block_table.ndim != 2:
        raise ValueError("QSA tile-union attention received invalid Q/K/V or page-table shapes")
    rows, requests = q.shape[0], block_table.shape[0]
    block_topk = token_topk // compress_ratio
    if block_indices.shape != (rows, block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if query_positions.shape != (rows,) or token_to_req.shape != (rows,) or starts.shape != (requests + 1,):
        raise ValueError("QSA tile-union metadata has invalid shapes: a position and a segment a row, a row offset a "
                         "segment and one past the last")
    if not (block_indices.dtype == query_positions.dtype == starts.dtype == token_to_req.dtype == block_table.dtype
            == torch.int32):
        raise ValueError("QSA tile-union metadata is int32")
    if not (q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16):
        raise ValueError("QSA tile-union attention reads BF16 queries and caches")
    head_dim = q.shape[2]
    if head_dim != k_cache.shape[3] or q.shape[1] % k_cache.shape[2] or head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("QSA tile-union attention takes grouped query heads over the cache's, a power-of-two head "
                         "dimension of at least 16")
    if not (q.device == k_cache.device == v_cache.device == block_indices.device == query_positions.device
            == starts.device == token_to_req.device == block_table.device):
        raise ValueError("QSA tile-union attention tensors share one device")
    if (q.stride(2) != 1 or k_cache.stride(3) != 1 or v_cache.stride(3) != 1 or block_indices.stride(1) != 1
            or block_table.stride(1) != 1):
        raise ValueError("QSA tile-union attention needs packed head dimensions and packed block ids and page rows")
    from engine.kernels.qsa import _packed_rows
    _packed_rows("QSA tile-union attention", token_to_req, query_positions, starts)
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device or out.stride(2) != 1:
        raise ValueError("QSA tile-union output must match its query")
    if gate is not None and (gate.shape != q.shape or gate.dtype != torch.bfloat16 or gate.device != q.device
                             or gate.stride(2) != 1):
        raise ValueError("the QSA output gate is BF16 in the query's shape with a unit stride along the head")
    if not admits(rows, requests, compress_ratio=compress_ratio, token_topk=token_topk, page_size=k_cache.shape[1],
                  table_width=block_table.shape[1], cache_pages=k_cache.shape[0]):
        raise ValueError(f"the QSA tile-union attention does not take this step ({rows} rows, {requests} segments, ratio "
                         f"{compress_ratio}, top-k {token_topk}, pages of {k_cache.shape[1]}, a table "
                         f"{block_table.shape[1]} wide over {k_cache.shape[0]} pages): the caller chooses by `admits`")
    R = TILE.rows
    device = q.device
    if layout is not None:
        tile_row0, tile_request, num_tiles = layout
        if (tile_row0.shape != (num_tiles,) or tile_request.shape != (num_tiles,) or tile_row0.dtype != torch.int32
                or tile_request.dtype != torch.int32 or num_tiles < triton.cdiv(rows, R)):
            raise ValueError("a QSA tile layout is `tiles(starts, rows, segments)` of this step")
    else:
        tile_row0, tile_request, num_tiles = tiles(starts, rows, requests)
    N = triton.next_power_of_2(R * block_topk)
    keys = torch.empty((num_tiles, N), dtype=torch.int32, device=device)
    _qsa_tile_union_pack_kernel[(num_tiles,)](
        block_indices, keys, tile_row0, tile_request, starts, token_to_req,
        block_indices.stride(0), keys.stride(0), rows, requests,
        E=block_topk, E_PAD=triton.next_power_of_2(block_topk), N=N, R=R, num_warps=4,
    )
    # Keys-only would do; torch.sort also materialises the index tensor.
    keys = torch.sort(keys, dim=1).values
    tail_cols = _tile_union_tail_cols(TILE, compress_ratio)
    mem = torch.zeros((num_tiles, R, N), dtype=torch.int8, device=device)
    cnt = torch.empty(num_tiles, dtype=torch.int32, device=device)
    tails = torch.empty((num_tiles, tail_cols), dtype=torch.int32, device=device)
    _qsa_tile_union_build_kernel[(num_tiles,)](
        keys, mem, cnt, tails, tile_row0, tile_request, starts, block_table, query_positions,
        keys.stride(0), mem.stride(0), mem.stride(1), tails.stride(0), block_table.stride(0),
        rows, requests, block_table.shape[1], k_cache.shape[0],
        N=N, R=R, CR=compress_ratio, PAGE_SIZE=k_cache.shape[1], TAIL_COLS=tail_cols, num_warps=4,
    )
    group_size = q.shape[1] // k_cache.shape[2]
    gate_rows = gate if gate is not None else out                        # never read without GATED
    _qsa_tile_union_attn_kernel[(num_tiles, k_cache.shape[2])](
        q, k_cache, v_cache, keys, mem, cnt, tails, tile_row0, tile_request, starts, token_to_req, out, gate_rows,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), keys.stride(0), mem.stride(0), mem.stride(1),
        tails.stride(0), out.stride(0), out.stride(1), gate_rows.stride(0), gate_rows.stride(1), rows, requests,
        R=R, GP=triton.next_power_of_2(group_size), GROUP_SIZE=group_size, HEAD_DIM=head_dim, BNB=TILE.blocks,
        CR=compress_ratio, TAIL_COLS=tail_cols, PAGE_SIZE=k_cache.shape[1], GATED=gate is not None,
        num_warps=TILE.warps, num_stages=1,
    )
    return out


QUALIFY_CONTEXT = 12000             # a qualifying segment starts past the budget: every row chooses among more groups
BAND = (2 * 2.0 ** -7, 2.0 ** -7)   # the served sparse attention's band: two BF16 steps at the largest value, one in rms


def qualify(device, *, heads: int, head_dim: int, ratio: int, budget: int, page_size: int, seed: int = 0) -> dict:
    """The launch a boot serves held to the one it replaces (D3): one step `admits` takes -- TILE.min_rows rows of one
    segment QUALIFY_CONTEXT positions deep, a prefill's selection (a segment-wide score plus a little of each row's own,
    so neighbours share most of their groups), BF16 K/V on shuffled pages, the output gate -- through `attention` and
    through `qsa.qsa_sparse_paged_attention_blocks`, within BAND. Raises RuntimeError outside it; returns the drift."""
    from engine.kernels import qsa
    gen = torch.Generator().manual_seed(seed)
    rows, ctx = TILE.min_rows, QUALIFY_CONTEXT
    total = ctx + rows
    need = -(-total // page_size)
    pages = need + 3
    page_table = torch.randperm(pages, generator=gen)[:need].to(torch.int32).view(1, need).to(device)
    positions = torch.arange(ctx, total, dtype=torch.int32)
    seen = (positions + 1) // ratio
    groups, keep = int(seen.max()), budget // ratio
    scores = torch.rand(1, groups, generator=gen) + 0.05 * torch.rand(rows, groups, generator=gen)
    scores = scores.masked_fill(torch.arange(groups).unsqueeze(0) >= seen.unsqueeze(1), -1.0)
    blocks = scores.topk(keep, dim=1).indices.to(torch.int32).contiguous().to(device)

    def bf16(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16).to(device)
    k_cache, v_cache = bf16(pages, page_size, 1, head_dim), bf16(pages, page_size, 1, head_dim)
    q, gate = bf16(rows, heads, head_dim, scale=2.0), bf16(rows, heads, head_dim)
    positions32 = positions.to(device)
    lengths = torch.tensor([total], dtype=torch.int32, device=device)
    starts = torch.tensor([0, rows], dtype=torch.int32, device=device)
    rows_req = torch.zeros(rows, dtype=torch.int32, device=device)
    if not admits(rows, 1, compress_ratio=ratio, token_topk=budget, page_size=page_size, table_width=need,
                  cache_pages=pages):
        raise RuntimeError(f"qsa_tile_union.qualify: the model's widths (ratio {ratio}, budget {budget}, pages of "
                           f"{page_size}) are not ones the tile-union launch takes")
    want = qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, blocks, positions32, lengths, ratio, budget,
                                                 page_table, rows_req, gate=gate).float()
    got = attention(q, k_cache, v_cache, blocks, positions32, starts, ratio, budget, page_table, rows_req,
                    gate=gate).float()
    difference = got - want
    largest = float(difference.abs().max() / want.abs().max().clamp_min(1e-30))
    rms = float(difference.pow(2).mean().sqrt() / want.pow(2).mean().sqrt().clamp_min(1e-30))
    if not bool(torch.isfinite(got).all()) or largest > BAND[0] or rms > BAND[1]:
        raise RuntimeError(f"qsa_tile_union.qualify: {rows} rows against the split-K launch -- largest {largest:.3g} "
                           f"(band {BAND[0]:.3g}), rms {rms:.3g} (band {BAND[1]:.3g}), finite "
                           f"{bool(torch.isfinite(got).all())}")
    return {"rows": rows, "largest": round(largest, 6), "rms": round(rms, 6)}
