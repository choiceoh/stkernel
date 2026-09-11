"""Fuse candidate key encoding and sparse restoration; selection stays in CUDA topk."""
import torch
import triton
import triton.language as tl


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
