"""Fuse candidate key encoding and sparse restoration; selection stays in CUDA topk."""
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
def _pack(X, OUT, ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr,
          VALID: tl.constexpr, START: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * ROW_STRIDE + col * COL_STRIDE, col < VALID, other=0).to(tl.float32)
    bits = value.to(tl.int32, bitcast=True).to(tl.int64)
    ordered = tl.where(bits < 0, bits ^ 0x7fffffff, bits)
    ordered = tl.where(value != value, 0x7fffffff, ordered)
    key = (ordered << 32) | (0xffffffff - (START + col.to(tl.int64)))
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


def restore(gathered, vocab):
    dense = torch.full((gathered.shape[0], vocab), float('-inf'), dtype=torch.float32, device=gathered.device)
    _restore[(gathered.shape[0], triton.cdiv(gathered.shape[1], 128))](gathered, dense, gathered.shape[1], vocab, 128)
    return dense
