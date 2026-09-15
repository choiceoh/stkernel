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
                FIRST: tl.constexpr, BLOCK: tl.constexpr, SEG: tl.constexpr = 0):
    # Grid axis 2 is the row of a captured decode step (write_ring_rows): program `seg` reads its own slot
    # and context and its own source rows at SEG. A one-row launch has one program there, at seg 0 and
    # SEG 0 -- the same loads and stores as before the axis existed.
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    seg = tl.program_id(2)
    slot, ctx = tl.load(SLOT + seg), tl.load(CTX + seg)
    value = tl.load(SRC + seg * SEG + (FIRST + row) * SS + col, col < WIDTH, other=0)
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


def write_ring_rows(src, dst, slots, contexts):
    """write_ring for every row of a captured decode step in one launch: row i of `src` [rows, length, ...]
    goes to ring slot `slots[i]` from position `contexts[i]`. Grid axis 2 is the row; each program does what
    the one-row launch does for that row, so the ring bytes are the same as `rows` one-row launches."""
    rows = src.shape[0]
    flat = src.reshape(rows, src.shape[1], -1)
    length, width = flat.shape[1], flat.shape[2]
    first = max(0, length - dst.shape[1])
    if slots.shape != (rows,) or contexts.shape != (rows,) or slots.stride(0) != 1 or contexts.stride(0) != 1:
        raise ValueError("write_ring_rows takes one contiguous slot and context per source row")
    _write_ring[(length-first, triton.cdiv(width, 256), rows)](
        flat, dst, slots, contexts, flat.stride(1), dst.stride(0), dst.stride(1),
        width, dst.shape[1], first, 256, SEG=flat.stride(0))


# -- boundaries crossed by a decode step ahead of the host (45차 §23; profiles/glm53/caches.stage_boundaries) --------------
@triton.jit
def _stage_kda(REC, CONV, REC_STAGE, CONV_STAGE, ROFF, RSOFF, COFF, CSOFF, SLOT, BEFORE, COUNT,
               BLOCK_TOKENS: tl.constexpr, CELLS: tl.constexpr, CELL: tl.constexpr,
               WIDTH: tl.constexpr, TAPS: tl.constexpr, CHANNELS: tl.constexpr,
               RS: tl.constexpr, RSS: tl.constexpr, CS: tl.constexpr, CSS: tl.constexpr,
               RB: tl.constexpr, CB: tl.constexpr):
    """A bounded grid parks both KDA histories only when a row crosses a prefix boundary."""
    i, L, c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    before, count = tl.load(BEFORE + i), tl.load(COUNT + i)
    after = before + count
    boundary = (after // BLOCK_TOKENS) * BLOCK_TOKENS
    crossed = (count > 0) & (boundary > before)
    if not crossed:
        return
    slot = tl.load(SLOT + i)
    cell = (boundary - 1) % CELLS
    src = REC + slot * RS + tl.load(ROFF + L) + cell * CELL
    dst = REC_STAGE + slot * RSS + tl.load(RSOFF + L)
    # A small fixed grid checks the boundary. Only crossing rows walk the
    # full state, instead of launching a CTA for every dormant 1024-cell tile.
    for tile in range(c, triton.cdiv(CELL, RB), tl.num_programs(2)):
        col = tile * RB + tl.arange(0, RB)
        value = tl.load(src + col, col < CELL, other=0.0)
        tl.store(dst + col, value, col < CELL)
    src = CONV + slot * CS + tl.load(COFF + L)
    dst = CONV_STAGE + slot * CSS + tl.load(CSOFF + L)
    for tile in range(c, triton.cdiv(CHANNELS, CB), tl.num_programs(2)):
        ch = tile * CB + tl.arange(0, CB)
        for j in tl.static_range(TAPS):
            cell = (boundary - TAPS + j) % WIDTH
            value = tl.load(src + ch * WIDTH + cell, ch < CHANNELS, other=0.0)
            tl.store(dst + ch * TAPS + j, value, ch < CHANNELS)


@triton.jit
def _stage_draft(RING, STAGE, ROFF, SOFF, SLOT, BEFORE, COUNT, BLOCK_TOKENS: tl.constexpr, WINDOW: tl.constexpr,
                 CELLS: tl.constexpr, CELL: tl.constexpr, RS: tl.constexpr, SS: tl.constexpr, BLOCK: tl.constexpr):
    """A crossing row's drafter ring cells for positions boundary .. boundary + CELLS - 1, into its stage before the
    step's observe writes past the boundary: one program a (row, layer and half, cell)."""
    i, q = tl.program_id(0), tl.program_id(1)
    slot, before, count = tl.load(SLOT + i), tl.load(BEFORE + i), tl.load(COUNT + i)
    after = before + count
    boundary = (after // BLOCK_TOKENS) * BLOCK_TOKENS
    crossed = (count > 0) & (boundary > before)
    plane, j = q // CELLS, q % CELLS                               # plane: layer * 2 + half
    cell = (boundary + j) % WINDOW
    col = tl.arange(0, BLOCK)
    mask = (col < CELL) & crossed
    value = tl.load(RING + slot * RS + ROFF + (plane * WINDOW + cell) * CELL + col, mask, other=0.0)
    tl.store(STAGE + slot * SS + SOFF + (plane * CELLS + j) * CELL + col, value, mask)


def stage_boundaries(caches, slots, ctx_before, counts):
    """Park every KDA layer's recurrent cell and conv taps in one bounded launch,
    plus one for the drafter ring cells past the boundary."""
    F = caches.F
    draft = caches._stage.get(("draft", -1))
    if draft is not None and slots.numel():
        ring = caches._fields["draft", -1]
        state_bf16, stage_bf16 = caches.state.view(torch.bfloat16), caches.stage_store.view(torch.bfloat16)
        layers, _, window, heads, dim = ring.shape[1:]
        cells = draft.shape[3]
        _stage_draft[(int(slots.numel()), layers * 2 * cells)](
            state_bf16, stage_bf16, ring.storage_offset() - state_bf16.storage_offset(),
            draft.storage_offset() - stage_bf16.storage_offset(), slots, ctx_before, counts, F.block, window, cells,
            heads * dim, caches.layout.slot_bytes // 2, caches.stage_bytes // 2, triton.next_power_of_2(heads * dim))
    kda = [L for L in caches.layers if not F.is_dsa(L)]
    if not kda:
        return
    recurrent = caches._fields["rec", kda[0]]
    state_rec, state_bf16 = caches.state.view(recurrent.dtype), caches.state.view(torch.bfloat16)
    stage_rec, stage_bf16 = caches.stage_store.view(recurrent.dtype), caches.stage_store.view(torch.bfloat16)
    rec_size = recurrent.element_size()
    tables = getattr(caches, "_stage_tables", None)
    if tables is None:
        # the layout's offsets are constants: built once, not four host-to-device copies (each a stream wait) per step
        dev = slots.device
        tables = caches._stage_tables = (
            torch.tensor([caches._fields["rec", L].storage_offset() - state_rec.storage_offset() for L in kda], device=dev),
            torch.tensor([caches._stage["rec", L].storage_offset() - stage_rec.storage_offset() for L in kda], device=dev),
            torch.tensor([caches._fields["conv", L].storage_offset() - state_bf16.storage_offset() for L in kda], device=dev),
            torch.tensor([caches._stage["conv", L].storage_offset() - stage_bf16.storage_offset() for L in kda], device=dev))
    rec_off, rec_stage_off, conv_off, conv_stage_off = tables
    n, cells = int(slots.numel()), F.spec_k + 1
    cell = F.kda_heads_local * F.kda_dim * F.kda_dim
    channels, width, taps = 3 * F.kda_heads_local * F.kda_dim, F.conv - 1 + F.spec_k, F.conv - 1
    tiles = min(8, max(triton.cdiv(cell, 1024), triton.cdiv(channels, 256)))
    _stage_kda[(n, len(kda), tiles)](
        state_rec, state_bf16, stage_rec, stage_bf16, rec_off, rec_stage_off, conv_off, conv_stage_off,
        slots, ctx_before, counts, F.block, cells, cell, width, taps, channels,
        caches.layout.slot_bytes // rec_size, caches.stage_bytes // rec_size,
        caches.layout.slot_bytes // 2, caches.stage_bytes // 2, 1024, 256)
