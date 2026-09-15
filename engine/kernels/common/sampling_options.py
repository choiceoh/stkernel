"""Per-position sampling transforms and committed history updates, without CPU decisions."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "E", "ES", "START", "VALID"])
def _process(X, OUT, DRAFT, GENERATED, ENDS, SEEN, COUNTS, BIAS, PENALTIES, MINIMUM, FORCE,
             WIDTH: tl.constexpr, VOCAB: tl.constexpr, START, VALID,
             T, K: tl.constexpr, E, XS: tl.constexpr, XC: tl.constexpr,
             OS: tl.constexpr, OC: tl.constexpr, DS: tl.constexpr, DC: tl.constexpr,
             ES, EC: tl.constexpr, HAS_FORCE: tl.constexpr,
             BLOCK: tl.constexpr):
    flat, tile = tl.program_id(0), tl.program_id(1)
    row, pos = flat // T, flat % T
    col = tile * BLOCK + tl.arange(0, BLOCK)
    ids = START + col
    live = col < WIDTH
    value = tl.load(X + flat * XS + col * XC, live, 0).to(tl.float32)
    value += tl.load(BIAS + row * VOCAB + ids, live, 0)
    count = tl.load(COUNTS + row * VOCAB + ids, live, 0)
    seen = tl.load(SEEN + row * VOCAB + ids, live, 0) | (count > 0)
    for j in tl.static_range(K):
        draft = tl.load(DRAFT + row * DS + j * DC)
        hit = (j < pos) & (draft == ids)
        seen |= hit
        count += hit.to(tl.float32)
    repetition = tl.load(PENALTIES + row * 3)
    presence = tl.load(PENALTIES + row * 3 + 1)
    frequency = tl.load(PENALTIES + row * 3 + 2)
    value = tl.where(seen, tl.where(value > 0, tl.div_rn(value, repetition), value * repetition), value)
    value -= frequency * count + presence * (count > 0).to(tl.float32)
    under_minimum = tl.load(GENERATED + row) + pos < tl.load(MINIMUM + row)
    forbidden = tl.full((BLOCK,), False, tl.int1)
    force = tl.load(FORCE + flat) if HAS_FORCE else -1
    force_forbidden = False
    for j in range(E):
        end = tl.load(ENDS + row * ES + j * EC)
        forbidden |= under_minimum & (end >= 0) & (ids == end)
        force_forbidden |= under_minimum & (end >= 0) & (force == end)
    value = tl.where((ids < VALID) & ~forbidden, value, -float('inf'))
    if HAS_FORCE:
        forcing = (force >= 0) & ~force_forbidden
        value = tl.where(forcing, tl.where(ids == force, tl.where(value > -float('inf'), value, 0.),
                                          -float('inf')), value)
    tl.store(OUT + flat * OS + col * OC, value, live)


def process(logits, drafts, generated, ends, state, out, start, decodable, forces):
    n, vocab = state.counts.shape
    _process[(logits.shape[0], triton.cdiv(logits.shape[1], 512))](
        logits, out, drafts, generated, ends, state.seen, state.counts, state.bias, state.penalties,
        state.minimum, generated if forces is None else forces,
        logits.shape[1], vocab, start, vocab if decodable is None else decodable,
        logits.shape[0] // n, drafts.shape[1], ends.shape[1], logits.stride(0), logits.stride(1),
        out.stride(0), out.stride(1), drafts.stride(0), drafts.stride(1), ends.stride(0), ends.stride(1),
        forces is not None, 512,
        num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _commit(TOKENS, COUNT, COUNTS, T: tl.constexpr, TS: tl.constexpr, TC: tl.constexpr,
            VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.arange(0, BLOCK)
    count = tl.load(COUNT + row)
    active = (pos < T) & (pos < count)
    token = tl.load(TOKENS + row * TS + pos * TC, active, 0)
    # Duplicate tokens add independently; exact integer counts are stored in FP32.
    tl.atomic_add(COUNTS + row * VOCAB + token, 1., active, sem='relaxed')


def commit(tokens, count, counts):
    _commit[(tokens.shape[0],)](tokens, count, counts, tokens.shape[1], tokens.stride(0), tokens.stride(1), counts.shape[1],
                              triton.next_power_of_2(tokens.shape[1]), num_warps=4)
