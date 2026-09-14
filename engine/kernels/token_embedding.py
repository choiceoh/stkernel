"""Fused rank-local embedding lookup: IDs, ownership mask and exact BF16 byte reads."""
import torch
import triton
import triton.language as tl


@triton.jit
def _lookup(IDS, W, OUT, IS: tl.constexpr, WS: tl.constexpr, V: tl.constexpr, H: tl.constexpr,
            START: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    local = tl.load(IDS + row * IS) - tl.full((), START, tl.int64)
    owned = (local >= 0) & (local < V)
    safe = tl.where(owned, local, 0)
    # Move payload bits, including signed zeros and NaNs, without a float conversion.
    bits = tl.load(W.to(tl.pointer_type(tl.uint16)) + safe * WS + col, owned & (col < H), other=0)
    tl.store(OUT.to(tl.pointer_type(tl.uint16)) + row * H + col, bits, col < H)


def lookup(ids, weight, start):
    rows, hidden = ids.numel(), weight.shape[1]
    out = torch.empty((rows, hidden), dtype=weight.dtype, device=weight.device)
    if rows:
        _lookup[(rows, triton.cdiv(hidden, 256))](ids, weight, out, ids.stride(0), weight.stride(0),
                                                weight.shape[0], hidden, start, 256, num_warps=4)
    return out
