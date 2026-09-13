"""Candidate keys, and the selection over them (45차 §87).

A key is `(ordered_score << 32) | (0xffffffff - id)`: one int64 that sorts exactly as the score does and breaks
every tie toward the lower vocabulary id. Keys are therefore UNIQUE -- no two columns can carry the same one --
so the top-k over them is a total order with one answer.

Selection used to be torch's. That cost twice: `key.topk(16)` over a rank's 38,720-wide shard was 169 us against
a 5 us read, and the merge after the exchange restored the candidates into a DENSE [rows, 154,880] fp32 tensor
filled with -inf -- 3.1 MiB written and scanned to choose sixteen of at most sixty-four -- for another 110 us.
The restoration existed to give torch's topk the vocabulary positions so its tie order would be the pinned one.
`select` replaces the first of those. It does not replace the merge: `modules/vocab.topk` is pinned to torch's
dense CUDA topk down to which of two equal scores comes first, and that order is torch's, not the keys'. What
`select` has to be is the same SET -- and since the keys are unique, the k largest are one set. The local step
already says so: it asks torch for `sorted=False`.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_partials(X, OUT, ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr,
                     VALID: tl.constexpr, START: tl.constexpr, PARTS: tl.constexpr,
                     BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * ROW_STRIDE + col * COL_STRIDE, col < VALID, other=0).to(tl.float32)
    # torch.argmax ties signed zeros and chooses the first NaN, independently
    # of its sign/payload. Canonicalize before the int64 MAX tie breaker.
    value = tl.where(value == 0, 0.0, value)
    bits = value.to(tl.int32, bitcast=True).to(tl.int64)
    ordered = tl.where(bits < 0, bits ^ 0x7fffffff, bits)
    ordered = tl.where(value != value, 0x7fc00000, ordered)
    key = (ordered << 32) | (0xffffffff - (START + col.to(tl.int64)))
    key = tl.where(col < VALID, key, -9223372036854775808)
    tl.store(OUT + row * PARTS + part, tl.max(key, 0))


@triton.jit
def _argmax_finish(PARTIALS, OUT, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    key = tl.load(PARTIALS + row * PARTS + part, part < PARTS, other=-9223372036854775808)
    tl.store(OUT + row, tl.max(key, 0))


def argmax_key(local_logits, start, valid):
    """One exact MAX packet per row, without a full FP32 vocabulary temporary."""
    rows = local_logits.shape[0]
    if not valid:
        return torch.full((rows,), -(2**63), dtype=torch.int64, device=local_logits.device)
    parts = triton.cdiv(valid, 1024)
    partials = torch.empty((rows, parts), dtype=torch.int64, device=local_logits.device)
    _argmax_partials[(rows, parts)](local_logits, partials, local_logits.stride(0),
                                  local_logits.stride(1), valid, start, parts, 1024)
    if parts == 1:
        return partials.view(rows)
    out = torch.empty(rows, dtype=torch.int64, device=local_logits.device)
    _argmax_finish[(rows,)](partials, out, parts, triton.next_power_of_2(parts))
    return out


@triton.jit
def _key(value, ids):
    bits = value.to(tl.float32).to(tl.int32, bitcast=True).to(tl.int64)
    ordered = tl.where(bits < 0, bits ^ 0x7fffffff, bits)
    ordered = tl.where(value != value, 0x7fffffff, ordered)
    return (ordered << 32) | (0xffffffff - ids.to(tl.int64))


@triton.jit
def _pack(X, OUT, ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr,
          VALID: tl.constexpr, START: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * ROW_STRIDE + col * COL_STRIDE, col < VALID, other=0).to(tl.float32)
    key = _key(value, START + col.to(tl.int64))
    tl.store(OUT + row * VALID + col, key, col < VALID)


@triton.jit
def _restore(PACKET, OUT, COUNT: tl.constexpr, VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    key = tl.load(PACKET + row * COUNT + col, col < COUNT, other=-9223372036854775808)
    ids = 0xffffffff - (key & 0xffffffff)
    ordered = key >> 32
    bits = tl.where(ordered < 0, ordered ^ 0x7fffffff, ordered).to(tl.int32)
    values = bits.to(tl.float32, bitcast=True)
    # Padding never writes. Every real id appears once in the gathered packet.
    tl.store(OUT + row * VOCAB + ids, values, (col < COUNT) & (key != -9223372036854775808))


def pack(local_logits, start, valid):
    out = torch.empty((local_logits.shape[0], valid), dtype=torch.int64, device=local_logits.device)
    _pack[(out.shape[0], triton.cdiv(valid, 256))](
        local_logits, out, local_logits.stride(0), local_logits.stride(1), valid, start, 256)
    return out


MIN_KEY = tl.constexpr(-9223372036854775808)   # annotation form is rejected by the JIT


@triton.jit
def _select(SRC, OUT, width, sS, sO, K: tl.constexpr, SEGS: tl.constexpr, BLOCK: tl.constexpr):
    """One segment's K largest keys, descending. The keys are unique, so "largest below the last one taken"
    is exact and the round needs no mask of what it has already used."""
    row, seg = tl.program_id(0), tl.program_id(1)
    col = seg * BLOCK + tl.arange(0, BLOCK)
    keys = tl.load(SRC + row * sS + col, col < width, other=MIN_KEY)
    limit = 0x7fffffffffffffff
    for i in tl.static_range(K):
        live = keys <= limit if i == 0 else keys < limit
        limit = tl.max(tl.where(live, keys, MIN_KEY), 0)
        tl.store(OUT + row * sO + seg * K + i, limit)


def select(keys, k):
    """The k largest keys of every row, descending: [rows, width] int64 -> [rows, k] int64."""
    rows, width = keys.shape
    block = min(2048, max(16, triton.next_power_of_2(width)))
    segs = triton.cdiv(width, block)
    out = torch.empty((rows, segs * k), dtype=torch.int64, device=keys.device)
    _select[(rows, segs)](keys, out, width, keys.stride(0), out.stride(0), K=k, SEGS=segs, BLOCK=block)
    return out if segs == 1 else select(out, k)


@triton.jit
def _select_logits(X, OUT, ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr,
                   VALID: tl.constexpr, START: tl.constexpr, K: tl.constexpr,
                   SEGS: tl.constexpr, BLOCK: tl.constexpr):
    row, seg = tl.program_id(0), tl.program_id(1)
    col = seg * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * ROW_STRIDE + col * COL_STRIDE, col < VALID, other=0).to(tl.float32)
    keys = tl.where(col < VALID, _key(value, START + col.to(tl.int64)), MIN_KEY)
    limit = 0x7fffffffffffffff
    for i in tl.static_range(K):
        live = keys <= limit if i == 0 else keys < limit
        limit = tl.max(tl.where(live, keys, MIN_KEY), 0)
        tl.store(OUT + row * SEGS * K + seg * K + i, limit)


def select_logits(local_logits, start, valid, k):
    """The exact packet from select(pack(...)), without writing a vocabulary of keys.

    Encoding stays in the first selection's registers. Only each segment's k
    winners leave the kernel; subsequent reductions and the final dense merge
    retain their existing order, including tied logits and nonfinite values.
    """
    rows = local_logits.shape[0]
    block = min(2048, max(16, triton.next_power_of_2(valid)))
    segs = triton.cdiv(valid, block)
    out = torch.empty((rows, segs * k), dtype=torch.int64, device=local_logits.device)
    _select_logits[(rows, segs)](local_logits, out, local_logits.stride(0), local_logits.stride(1),
                                valid, start, k, segs, block)
    return out if segs == 1 else select(out, k)


def restore(gathered, vocab):
    dense = torch.full((gathered.shape[0], vocab), float('-inf'), dtype=torch.float32, device=gathered.device)
    _restore[(gathered.shape[0], triton.cdiv(gathered.shape[1], 128))](gathered, dense, gathered.shape[1], vocab, 128)
    return dense


@triton.jit
def _restore_reuse(PACKET, PREVIOUS, OUT, COUNT: tl.constexpr, VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    old = tl.load(PREVIOUS + row * COUNT + col, col < COUNT, other=MIN_KEY)
    old_id = 0xffffffff - (old & 0xffffffff)
    tl.store(OUT + row * VOCAB + old_id, -float('inf'), (col < COUNT) & (old != MIN_KEY))
    # A token may occur in both packets, on different lanes. All old entries
    # must be cleared before any current entry is restored (one CTA per row).
    tl.debug_barrier()
    key = tl.load(PACKET + row * COUNT + col, col < COUNT, other=MIN_KEY)
    ids = 0xffffffff - (key & 0xffffffff)
    ordered = key >> 32
    bits = tl.where(ordered < 0, ordered ^ 0x7fffffff, ordered).to(tl.int32)
    tl.store(OUT + row * VOCAB + ids, bits.to(tl.float32, bitcast=True),
             (col < COUNT) & (key != MIN_KEY))
    tl.store(PREVIOUS + row * COUNT + col, key, col < COUNT)


def restore_reuse(gathered, dense, previous):
    rows, count = gathered.shape
    _restore_reuse[(rows,)](gathered, previous, dense, count, dense.shape[1],
                            triton.next_power_of_2(count), num_warps=4)
    return dense[:rows]
