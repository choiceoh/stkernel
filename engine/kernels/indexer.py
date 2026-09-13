"""Integer-only finalization of sparse indexer positions into latent slots."""
import torch
import triton
import triton.language as tl


@triton.jit
def _map_positions(pos, table, table_s0, block_size: tl.constexpr,
                   block_stride, layer_offset, MAPPED: tl.constexpr):
    if MAPPED:
        page = pos.to(tl.int64) // block_size
        physical = tl.load(table + page * table_s0, pos >= 0, other=0)
        pos = tl.where(pos >= 0, physical * block_stride + layer_offset + pos % block_size, -1)
    return pos


@triton.jit
def _pool_slots(ids, lengths, table, out, counts, groups: tl.constexpr,
                id_s0, id_s1, len_s0, table_s0, out_s0, out_s1, count_s0,
                block_size: tl.constexpr, block_stride, layer_offset,
                POOL: tl.constexpr, MAPPED: tl.constexpr, BLOCK: tl.constexpr,
                table_s1=0, TOKENS: tl.constexpr = 1):
    row = tl.program_id(0)
    # A captured decode step's rows come TOKENS to a sequence, each sequence with its own block row at
    # table_s1 apart; a one-sequence launch has table_s1 0, and reads the one row it was given.
    table += (row // TOKENS) * table_s1
    g = tl.arange(0, BLOCK)
    seq = tl.load(lengths + row * len_s0)
    tail = seq % POOL
    pool = tl.load(ids + row * id_s0 + g * id_s1, g < groups, other=-1)
    pool = tl.sort(tl.where((pool >= 0) & (pool < seq // POOL), pool, -1), descending=True)
    # Sorting compressed pool IDs is 8x smaller than a padded 2051-token sort.
    # Runs also preserve the token-level order if a pool is selected twice.
    prev = tl.gather(pool, tl.maximum(g - 1, 0), 0)
    nxt = tl.gather(pool, tl.minimum(g + 1, BLOCK - 1), 0)
    first = tl.associative_scan(tl.where((g == 0) | (prev != pool), g, 0), 0, _maximum)
    last = tl.associative_scan(tl.where((g == BLOCK - 1) | (nxt != pool), g + 1, BLOCK), 0, _minimum, reverse=True)
    off = tl.arange(0, POOL)
    within = ((g - first)[:, None] * POOL + off[None, :]) // (last - first)[:, None]
    # A pool never straddles a KV block. Load one page address per pool,
    # then broadcast it to its tokens instead of gathering four copies.
    base = _map_positions(pool * POOL, table, table_s0, block_size, block_stride, layer_offset, MAPPED)
    mapped = tl.where(pool[:, None] >= 0, base[:, None] + POOL - 1 - within, -1)
    cols = tail + g[:, None] * POOL + off[None, :]
    tl.store(out + row * out_s0 + cols * out_s1, mapped, g[:, None] < groups)
    # Complete the disjoint tail prefix and padding suffix. Every output is written.
    extra_cols = tl.where(off < tail, off, groups * POOL + off)
    tail_base = _map_positions(tl.where(tail > 0, seq - tail, -1), table, table_s0,
                               block_size, block_stride, layer_offset, MAPPED)
    extra = tl.where(off < tail, tail_base + tail - 1 - off, -1)
    tl.store(out + row * out_s0 + extra_cols * out_s1, extra, off < POOL - 1)
    tl.store(counts + row * count_s0, tl.sum((pool >= 0).to(tl.int32), 0) * POOL + tail)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _minimum(a, b):
    return tl.minimum(a, b)


def pool_slots(pool_ids, seq_lens, pool_size, block_table, block_size, block_stride,
               layer_offset, out, counts, tokens: int = 1):
    """Expand selected complete pools directly into descending-position slots.

    Integer-only, no scratch allocation or device-to-host reads. Invalid pools
    are masked before sorting or addressing. Duplicate pools retain multiplicity.
    Inputs/outputs may be strided but must not overlap. Sequence lengths are
    nonnegative int32; valid token positions must fit int32 and the block row.
    Mapped KV blocks must contain a whole number of pools.

    A 2-D block table [sequences, blocks] serves a captured decode step: rows come
    `tokens` to a sequence in order, and row r reads block row r // tokens -- one
    launch whose program r is what a one-sequence launch's program does for that row.
    """
    assert pool_ids.ndim == 2 and pool_ids.dtype == torch.int32
    rows, groups = pool_ids.shape
    assert pool_size > 0 and pool_size & (pool_size - 1) == 0
    assert seq_lens.shape == (rows,) and seq_lens.dtype == torch.int32
    assert out.shape == (rows, groups * pool_size + pool_size - 1) and out.dtype == torch.int32
    assert counts.shape == (rows,) and counts.dtype == torch.int32 and block_size > 0
    table_s0 = table_s1 = 0
    if block_table is not None:
        assert block_table.ndim in (1, 2) and block_table.dtype == torch.int32
        assert block_size % pool_size == 0
        if block_table.ndim == 2:
            assert tokens > 0 and block_table.shape[0] * tokens == rows, "one block row per `tokens` query rows"
            table_s0, table_s1 = block_table.stride(1), block_table.stride(0)
        else:
            table_s0 = block_table.stride(0)
    if rows == 0:
        return
    _pool_slots[(rows,)](
        pool_ids, seq_lens, block_table if block_table is not None else pool_ids, out, counts, groups,
        *pool_ids.stride(), seq_lens.stride(0), table_s0,
        *out.stride(), counts.stride(0), block_size, block_stride, layer_offset,
        POOL=pool_size, MAPPED=block_table is not None, BLOCK=triton.next_power_of_2(max(1, groups)), num_warps=4,
        table_s1=table_s1, TOKENS=tokens if block_table is not None and block_table.ndim == 2 else 1)


@triton.jit
def _indexer_slots(tokens, table, out, counts, width: tl.constexpr,
                   token_s0, token_s1, table_s0, out_s0, out_s1, count_s0,
                   block_size: tl.constexpr, block_stride, layer_offset,
                   MAPPED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    positions = tl.load(tokens + row * token_s0 + cols * token_s1,
                        cols < width, other=-1)
    # Positions are already sorted. Mask padding before any block-table read.
    valid = (cols < width) & (positions >= 0)
    if MAPPED:
        page = positions.to(tl.int64) // block_size
        physical = tl.load(table + page * table_s0, valid, other=0)
        slots = physical * block_stride + layer_offset + positions % block_size
    else:
        slots = positions
    tl.store(out + row * out_s0 + cols * out_s1, tl.where(valid, slots, -1), cols < width)
    tl.store(counts + row * count_s0, tl.sum(valid.to(tl.int32), 0))


def indexer_slots(tokens, block_table, block_size, block_stride, layer_offset, out, counts):
    """Sort int32 token positions descending, map and write slots/counts in place.

    Negative positions are padding. Nonnegative positions must belong to the
    reserved block row. A None table means the identity map used by chain
    checks. Input and output views may be strided; they must not overlap.
    """
    assert tokens.ndim == 2 and tokens.dtype == torch.int32
    assert out.shape == tokens.shape and out.dtype == torch.int32
    assert counts.shape == (tokens.shape[0],) and counts.dtype == torch.int32
    assert block_size > 0
    if block_table is not None:
        assert block_table.ndim == 1 and block_table.dtype == torch.int32
    rows, width = tokens.shape
    if rows == 0:
        return
    # Keep PyTorch's integer sort: a single-program bitonic sort over the
    # padded 4096-column vector costs more than the launches it eliminates.
    # Only the subsequent count/map/mask/write chain is fused.
    tokens = tokens.sort(dim=1, descending=True).values
    size = triton.next_power_of_2(max(1, width))
    _indexer_slots[(rows,)](
        tokens, block_table if block_table is not None else tokens, out, counts, width,
        *tokens.stride(), block_table.stride(0) if block_table is not None else 0,
        *out.stride(), counts.stride(0), block_size, block_stride, layer_offset,
        MAPPED=block_table is not None, BLOCK=size, num_warps=8 if size > 1024 else 4)
