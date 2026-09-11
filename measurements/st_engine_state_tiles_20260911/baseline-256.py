"""Address recurrent/conv rings by device slot id without copying whole slots."""
import torch
import triton
import triton.language as tl


@triton.jit
def _read_conv(SRC, SLOT, CTX, OUT, SS: tl.constexpr, CS: tl.constexpr,
               RING: tl.constexpr, CHANNELS: tl.constexpr, HISTORY: tl.constexpr,
               BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel, back = i // HISTORY, i % HISTORY
    pos = tl.load(CTX) - HISTORY + back
    slot = tl.load(SLOT)
    value = tl.load(SRC + slot * SS + channel * CS + (tl.maximum(pos, 0) % RING),
                    (channel < CHANNELS) & (pos >= 0), other=0)
    tl.store(OUT + i, value, channel < CHANNELS)


@triton.jit
def _read_rec(SRC, SLOT, CTX, OUT, SS: tl.constexpr, RS: tl.constexpr,
              RING: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ctx, slot = tl.load(CTX), tl.load(SLOT)
    row = tl.maximum(ctx - 1, 0) % RING
    value = tl.load(SRC + slot * SS + row * RS + i, (i < WIDTH) & (ctx > 0), other=0)
    tl.store(OUT + i, value, i < WIDTH)


@triton.jit
def _write_conv(SRC, DST, SLOT, CTX, SS: tl.constexpr, DS: tl.constexpr,
                CS: tl.constexpr, CHANNELS: tl.constexpr, RING: tl.constexpr,
                FIRST: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot, ctx = tl.load(SLOT), tl.load(CTX)
    value = tl.load(SRC + (FIRST + row) * SS + col, col < CHANNELS, other=0)
    tl.store(DST + slot * DS + col * CS + (ctx + FIRST + row) % RING, value, col < CHANNELS)


@triton.jit
def _write_ring(SRC, DST, SLOT, CTX, SS: tl.constexpr, DS: tl.constexpr,
                RS: tl.constexpr, WIDTH: tl.constexpr, RING: tl.constexpr,
                FIRST: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot, ctx = tl.load(SLOT), tl.load(CTX)
    value = tl.load(SRC + (FIRST + row) * SS + col, col < WIDTH, other=0)
    tl.store(DST + slot * DS + ((ctx + FIRST + row) % RING) * RS + col, value, col < WIDTH)


def kda_history(conv, rec, slot, context, history):
    """Only K-1 conv rows and one recurrent state; context zero reads zeros."""
    channels = conv.shape[1]
    hist = torch.empty((channels, history), dtype=conv.dtype, device=conv.device)
    state = torch.empty((1, *rec.shape[2:]), dtype=rec.dtype, device=rec.device)
    width = state.numel()
    _read_conv[(triton.cdiv(hist.numel(), 256),)](
        conv, slot, context, hist, conv.stride(0), conv.stride(1), conv.shape[2], channels, history, 256)
    _read_rec[(triton.cdiv(width, 256),)](
        rec, slot, context, state, rec.stride(0), rec.stride(1), rec.shape[1], width, 256)
    return hist, state


def write_conv(src, dst, slot, context):
    length, channels = src.shape
    first = max(0, length - dst.shape[2])
    _write_conv[(length-first, triton.cdiv(channels, 256))](
        src, dst, slot, context, src.stride(0), dst.stride(0), dst.stride(1),
        channels, dst.shape[2], first, 256)


def write_ring(src, dst, slot, context):
    """Write each changed position once, preserving other slots and ring cells."""
    flat = src.reshape(src.shape[0], -1)
    length, width = flat.shape
    first = max(0, length - dst.shape[1])
    _write_ring[(length-first, triton.cdiv(width, 256))](
        flat, dst, slot, context, flat.stride(0), dst.stride(0), dst.stride(1),
        width, dst.shape[1], first, 256)
