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


def conv_history(conv, slot, context, history):
    """Gather only K-1 convolution rows, masking positions before context zero."""
    channels = conv.shape[1]
    hist = torch.empty((channels, history), dtype=conv.dtype, device=conv.device)
    _read_conv[(triton.cdiv(hist.numel(), 256),)](
        conv, slot, context, hist, conv.stride(0), conv.stride(1), conv.shape[2], channels, history, 256)
    return hist


def kda_history(conv, rec, slot, context, history):
    """Only K-1 conv rows and one recurrent state; context zero reads zeros."""
    hist = conv_history(conv, slot, context, history)
    state = torch.empty((1, *rec.shape[2:]), dtype=rec.dtype, device=rec.device)
    width = state.numel()
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


# -- boundaries crossed by a decode step ahead of the host (45차 §23; profiles/glm53/caches.stage_boundaries) --------------
@triton.jit
def _stage_rec(RING, STAGE, ROFF, SOFF, SLOT, BEFORE, COUNT, BLOCK_TOKENS: tl.constexpr, CELLS: tl.constexpr,
               CELL: tl.constexpr, RS: tl.constexpr, SS: tl.constexpr, BLOCK: tl.constexpr):
    i, L, c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot, before, count = tl.load(SLOT + i), tl.load(BEFORE + i), tl.load(COUNT + i)
    after = before + count
    boundary = (after // BLOCK_TOKENS) * BLOCK_TOKENS
    crossed = (count > 0) & (boundary > before)
    cell = (boundary - 1) % CELLS
    col = c * BLOCK + tl.arange(0, BLOCK)
    mask = (col < CELL) & crossed
    value = tl.load(RING + slot * RS + tl.load(ROFF + L) + cell * CELL + col, mask, other=0.0)
    tl.store(STAGE + slot * SS + tl.load(SOFF + L) + col, value, mask)


@triton.jit
def _stage_conv(RING, STAGE, ROFF, SOFF, SLOT, BEFORE, COUNT, BLOCK_TOKENS: tl.constexpr, WIDTH: tl.constexpr,
                TAPS: tl.constexpr, CHANNELS: tl.constexpr, RS: tl.constexpr, SS: tl.constexpr, BLOCK: tl.constexpr):
    i, L, c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot, before, count = tl.load(SLOT + i), tl.load(BEFORE + i), tl.load(COUNT + i)
    after = before + count
    boundary = (after // BLOCK_TOKENS) * BLOCK_TOKENS
    crossed = (count > 0) & (boundary > before)
    ch = c * BLOCK + tl.arange(0, BLOCK)
    mask = (ch < CHANNELS) & crossed
    for j in tl.static_range(TAPS):
        cell = (boundary - TAPS + j) % WIDTH
        value = tl.load(RING + slot * RS + tl.load(ROFF + L) + ch * WIDTH + cell, mask, other=0.0)
        tl.store(STAGE + slot * SS + tl.load(SOFF + L) + ch * TAPS + j, value, mask)


def stage_boundaries(caches, slots, ctx_before, counts):
    """caches.stage_boundaries on the device: one launch for every KDA layer's recurrent cell, one for the conv taps."""
    F = caches.F
    kda = [L for L in caches.layers if not F.is_dsa(L)]
    if not kda:
        return
    state_f32, state_bf16 = caches.state.view(torch.float32), caches.state.view(torch.bfloat16)
    stage_f32, stage_bf16 = caches.stage_store.view(torch.float32), caches.stage_store.view(torch.bfloat16)
    tables = getattr(caches, "_stage_tables", None)
    if tables is None:
        # the layout's offsets are constants: built once, not four host-to-device copies (each a stream wait) per step
        dev = slots.device
        tables = caches._stage_tables = (
            torch.tensor([caches._fields["rec", L].storage_offset() - state_f32.storage_offset() for L in kda], device=dev),
            torch.tensor([caches._stage["rec", L].storage_offset() - stage_f32.storage_offset() for L in kda], device=dev),
            torch.tensor([caches._fields["conv", L].storage_offset() - state_bf16.storage_offset() for L in kda], device=dev),
            torch.tensor([caches._stage["conv", L].storage_offset() - stage_bf16.storage_offset() for L in kda], device=dev))
    rec_off, rec_stage_off, conv_off, conv_stage_off = tables
    n, cells = int(slots.numel()), F.spec_k + 1
    cell = F.kda_heads_local * F.kda_dim * F.kda_dim
    _stage_rec[(n, len(kda), triton.cdiv(cell, 1024))](
        state_f32, stage_f32, rec_off, rec_stage_off, slots, ctx_before, counts, F.block, cells, cell,
        caches.layout.slot_bytes // 4, caches.stage_bytes // 4, 1024)
    channels, width, taps = 3 * F.kda_heads_local * F.kda_dim, F.conv - 1 + F.spec_k, F.conv - 1
    _stage_conv[(n, len(kda), triton.cdiv(channels, 256))](
        state_bf16, stage_bf16, conv_off, conv_stage_off, slots, ctx_before, counts, F.block, width, taps, channels,
        caches.layout.slot_bytes // 2, caches.stage_bytes // 2, 256)
