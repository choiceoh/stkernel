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


# -- a captured decode step's DSA-layer glue, one launch apiece (45차, the C=4 question, third fold) -------------------
# Every kernel here moves bytes or computes integers: the same values the torch composition in
# engine/modules/sparse_indexer.py (the reference lane) produces, byte for byte.

@triton.jit
def _row_lengths(CTX, SEQ, KE, T: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr):
    seg = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    seq = (tl.load(CTX + seg) + j + 1).to(tl.int32)
    tl.store(SEQ + seg * T + j, seq, j < T)
    tl.store(KE + seg * T + j, seq // KP, j < T)


def row_lengths(contexts, tokens: int, pool_size: int):
    """For every row's tokens: the sequence length at each query (position + 1) and the complete pools before
    it (length // pool_size), int32 [rows * tokens] each -- the segment loop's `seq_lens` and `ke`."""
    rows = contexts.shape[0]
    assert contexts.ndim == 1 and contexts.stride(0) == 1 and tokens > 0
    seq = torch.empty(rows * tokens, dtype=torch.int32, device=contexts.device)
    ke = torch.empty_like(seq)
    if rows:
        _row_lengths[(rows,)](contexts, seq, ke, tokens, pool_size, triton.next_power_of_2(tokens))
    return seq, ke


@triton.jit
def _latent_write_rows(SRC, LAT, TABLE, CTX, T: tl.constexpr, BLOCK_TOKENS: tl.constexpr, table_s0, table_s1,
                       block_stride, layer_offset, src_s0, lat_s0, DB: tl.constexpr):
    r = tl.program_id(0)
    seg = r // T
    pos = tl.load(CTX + seg) + r % T
    page = tl.load(TABLE + seg * table_s0 + (pos // BLOCK_TOKENS) * table_s1)
    slot = page.to(tl.int64) * block_stride + layer_offset + pos % BLOCK_TOKENS
    d = tl.arange(0, DB)
    tl.store(LAT + slot * lat_s0 + d, tl.load(SRC + r * src_s0 + d))


def latent_write_rows(values, latent, block_table, block_size, block_stride, layer_offset, contexts, tokens: int):
    """latent[slot(row i, contexts[i] + j)] = values[i * tokens + j] for every row at once: `token_rows`' slot
    arithmetic (row i's block row, `block_stride` latent rows per block, this layer at `layer_offset`) and the
    scatter in one launch. Bytes are copied as they are; the caller converts to the cache dtype first."""
    rows = block_table.shape[0]
    assert values.ndim == 2 and values.shape[0] == rows * tokens and values.stride(1) == 1
    assert latent.ndim == 2 and latent.stride(1) == 1 and values.element_size() == latent.element_size()
    assert values.shape[1] == latent.shape[1] and contexts.shape == (rows,) and contexts.stride(0) == 1
    width = values.shape[1] * values.element_size()
    assert width & (width - 1) == 0, "a latent row is a power-of-two byte count"
    src, lat = values.view(torch.uint8), latent.view(torch.uint8)
    if rows:
        _latent_write_rows[(rows * tokens,)](src, lat, block_table, contexts, tokens, block_size, block_table.stride(0),
                                             block_table.stride(1), block_stride, layer_offset, src.stride(0), lat.stride(0), width)


@triton.jit
def _gather_candidates(KEYS, SCALES, TABLE, OUT_K, OUT_S, n_cand, key_s0, scale_s0, table_s0, table_s1,
                       PER: tl.constexpr, block_stride, layer_offset, DB: tl.constexpr, BLOCK_P: tl.constexpr):
    row = tl.program_id(0)
    p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    live = p < n_cand
    page = tl.load(TABLE + row * table_s0 + (p // PER) * table_s1, live, other=0)
    rec = page.to(tl.int64) * block_stride + layer_offset + p % PER
    d = tl.arange(0, DB)
    keys = tl.load(KEYS + rec[:, None] * key_s0 + d[None, :], live[:, None], other=0)
    tl.store(OUT_K + (row * n_cand + p)[:, None] * DB + d[None, :], keys, live[:, None])
    tl.store(OUT_S + row * n_cand + p, tl.load(SCALES + rec * scale_s0, live, other=0.), live)


def gather_candidates(keys, scales, block_table, per, block_stride, layer_offset, n_cand: int):
    """Every row's candidate pool keys and scales, contiguous: keys [rows, n_cand, d] and scales [rows, n_cand],
    pool p of row i read from record `block_row[p // per] * block_stride + layer_offset + p % per` -- the slot
    arithmetic of `pool_rows` and the two gathers `keys[cand]`, `scales[cand]` in one launch."""
    rows = block_table.shape[0]
    assert keys.ndim == 2 and keys.stride(1) == 1 and keys.element_size() == 1 and scales.ndim == 1
    out_k = torch.empty((rows, n_cand, keys.shape[1]), dtype=keys.dtype, device=keys.device)
    out_s = torch.empty((rows, n_cand), dtype=scales.dtype, device=keys.device)
    width = keys.shape[1]
    assert width & (width - 1) == 0, "a key record is a power-of-two byte count"
    if rows and n_cand:
        block_p = 64
        _gather_candidates[(rows, triton.cdiv(n_cand, block_p))](
            keys.view(torch.uint8), scales, block_table, out_k.view(torch.uint8), out_s, n_cand, keys.stride(0), scales.stride(0),
            block_table.stride(0), block_table.stride(1), per, block_stride, layer_offset, width, block_p)
    return out_k, out_s


@triton.jit
def _pool_window(TAILS, K, GATE, CTX, OUT_K, OUT_G, T: tl.constexpr, KP: tl.constexpr, W: tl.constexpr, NPOS: tl.constexpr,
                 tail_s0, tail_s1, tail_s2, k_s0, k_s1, gate_s0, gate_s1, D: tl.constexpr):
    seg = tl.program_id(0)
    i = tl.program_id(1)
    ctx = tl.load(CTX + seg)
    rel = i - ctx % KP                                        # this window position relative to the step's first token
    earlier = rel < 0                                         # before this step: from the tail ring
    prev = (ctx + rel) % W
    cur = tl.minimum(tl.maximum(rel, 0), T - 1)
    d = tl.arange(0, D)
    ring = TAILS + seg * tail_s0 + prev * tail_s1
    k_ring = tl.load(ring + d, earlier & (d < D), other=0)
    g_ring = tl.load(ring + tail_s2 + d, earlier & (d < D), other=0)
    k_cur = tl.load(K + seg * k_s0 + cur * k_s1 + d)
    g_cur = tl.load(GATE + seg * gate_s0 + cur * gate_s1 + d)
    out = (seg * NPOS + i) * D + d
    tl.store(OUT_K + out, tl.where(earlier, k_ring, k_cur))
    tl.store(OUT_G + out, tl.where(earlier, g_ring, g_cur))


def pool_window(tails, keys, gates, contexts, pool_size: int, max_pools: int):
    """The window each row pools this step -- the tail ring's earlier tokens of the half-built pool, then this
    step's -- as (keys, gates) [rows * max_pools, pool_size, d]: `complete_pools`' window in one launch."""
    n, w, two, d = tails.shape
    t = keys.shape[1]
    assert two == 2 and keys.shape == (n, t, d) == gates.shape and tails.stride(3) == 1
    assert keys.stride(2) == 1 and gates.stride(2) == 1 and contexts.shape == (n,) and contexts.stride(0) == 1
    assert d & (d - 1) == 0
    npos = max_pools * pool_size
    out_k = torch.empty((n * npos, d), dtype=keys.dtype, device=keys.device)
    out_g = torch.empty((n * npos, d), dtype=gates.dtype, device=keys.device)
    if n and npos:
        _pool_window[(n, npos)](tails, keys, gates, contexts, out_k, out_g, t, pool_size, w, npos,
                                tails.stride(0), tails.stride(1), tails.stride(2), keys.stride(0), keys.stride(1),
                                gates.stride(0), gates.stride(1), d)
    return out_k.view(n * max_pools, pool_size, d), out_g.view(n * max_pools, pool_size, d)


@triton.jit
def _pool_addresses(CTX, TABLE, COUNTS, SLOTS, T: tl.constexpr, KP: tl.constexpr, MAXP: tl.constexpr, PER: tl.constexpr,
                    table_s0, table_s1, block_stride, layer_offset, cap, BLOCK: tl.constexpr):
    seg = tl.program_id(0)
    ctx = tl.load(CTX + seg)
    tl.store(COUNTS + seg, (ctx % KP + T) // KP)
    j = tl.arange(0, BLOCK)
    pid = tl.minimum(ctx // KP + j, cap - 1)
    page = tl.load(TABLE + seg * table_s0 + (pid // PER) * table_s1, j < MAXP, other=0)
    tl.store(SLOTS + seg * MAXP + j, page.to(tl.int64) * block_stride + layer_offset + pid % PER, j < MAXP)


def pool_addresses(contexts, block_table, per, block_stride, layer_offset, pool_size: int, tokens: int, max_pools: int, capacity: int):
    """Per row: how many pools this step completes ((context % pool_size + tokens) // pool_size) and the record
    slots of its `max_pools` pools from context // pool_size on, ids clamped to the candidate capacity -- the
    `counts` and `pool_rows(pids)` of `complete_pools` in one launch. int64 [rows] and [rows, max_pools]."""
    n = contexts.shape[0]
    assert contexts.ndim == 1 and contexts.stride(0) == 1 and block_table.shape[0] == n and max_pools > 0
    counts = torch.empty(n, dtype=torch.int64, device=contexts.device)
    slots = torch.empty((n, max_pools), dtype=torch.int64, device=contexts.device)
    if n:
        _pool_addresses[(n,)](contexts, block_table, counts, slots, tokens, pool_size, max_pools, per,
                              block_table.stride(0), block_table.stride(1), block_stride, layer_offset, capacity,
                              triton.next_power_of_2(max_pools))
    return counts, slots
