"""Commit speculative tokens and advance a decode row in one device launch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _advance(PICKS, DRAFTS, ACCEPTED, ENDS, ALIVE, GENERATED, LIMIT, CTX, ANCHOR,
             REAL_SLOT, SLOT, COUNT, DONE, KEPT, BEFORE,
             K: tl.constexpr, E: tl.constexpr, PICK_STRIDE: tl.constexpr,
             DRAFT_STRIDE: tl.constexpr, HAS_ACCEPTED: tl.constexpr,
             BT: tl.constexpr, BE: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.arange(0, BT)
    token = tl.load(PICKS + row * PICK_STRIDE + pos, pos <= K, other=-2)
    if HAS_ACCEPTED:
        accepted = tl.load(ACCEPTED + row)
    else:
        draft = tl.load(DRAFTS + row * DRAFT_STRIDE + pos, pos < K, other=-2)
        accepted = tl.min(tl.where((pos < K) & (token != draft), pos, K), 0).to(tl.int64)
    alive = tl.load(ALIVE + row)
    generated, limit = tl.load(GENERATED + row), tl.load(LIMIT + row)
    count = tl.minimum(accepted + 1, tl.maximum(limit - generated, 0))
    end_pos = tl.arange(0, BE)
    ends = tl.load(ENDS + row * E + end_pos, end_pos < E, other=-1)
    is_end = tl.sum(((token[:, None] == ends[None, :]) & (end_pos[None, :] < E)).to(tl.int32), 1) > 0
    first = tl.min(tl.where(is_end & (pos < count) & (pos <= K), pos, K + 1), 0)
    count = tl.where(alive, tl.minimum(count, first + 1), 0)
    done = alive & ((first < K + 1) | (generated + count >= limit))
    ctx = tl.load(CTX + row)
    anchor = tl.load(ANCHOR + row)
    last = tl.load(PICKS + row * PICK_STRIDE + tl.maximum(count - 1, 0))
    real_slot = tl.load(REAL_SLOT + row)
    tl.store(BEFORE + row, ctx)
    tl.store(COUNT + row, count)
    tl.store(DONE + row, done)
    tl.store(KEPT + row, tl.minimum(accepted, tl.maximum(count - 1, 0)))
    tl.store(CTX + row, ctx + count)
    tl.store(GENERATED + row, generated + count)
    tl.store(ANCHOR + row, tl.where(count > 0, last, anchor))
    tl.store(ALIVE + row, alive & ~done)
    tl.store(SLOT + row, tl.where(alive & ~done, real_slot, 0))


def advance(picks, state, accepted=None):
    """Return readback results and old contexts; retain picks without copying."""
    n, t = picks.shape
    count = torch.empty(n, dtype=torch.int64, device=picks.device)
    done = torch.empty(n, dtype=torch.bool, device=picks.device)
    kept, before = torch.empty_like(count), torch.empty_like(count)
    _advance[(n,)](
        picks, state['drafts'], accepted if accepted is not None else count,
        state['ends'], state['alive'], state['generated'], state['limit'],
        state['ctx'], state['anchor'], state['real_slot'], state['slot'],
        count, done, kept, before, t - 1, state['ends'].shape[1], picks.stride(0),
        state['drafts'].stride(0), accepted is not None,
        triton.next_power_of_2(t), triton.next_power_of_2(state['ends'].shape[1]), num_warps=4)
    return count, done, kept, picks, before
