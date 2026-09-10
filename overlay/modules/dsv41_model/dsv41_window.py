"""The sliding-window KV ring and the slot ids a query may read. No GPU needed.

Every V4.1 attention layer attends over a window of raw KV, and the layers with
`compress_ratio > 0` concatenate compressed positions after it. This module is
the window half: where a token's KV is written, and which slots a query is
allowed to read. Both are pure index arithmetic, which is why they can be held
to the reference exactly rather than within a tolerance.

The ring invariant is one line, and everything here follows from it:

    absolute position p lives at slot p % window_size

Decode writes `cache[start_pos % window]` and satisfies it directly. Prefill
does not: it seeds the ring from the LAST `window` tokens of the chunk with a
rotated two-part copy, and it is that rotation which makes the invariant hold
for the decode steps that follow. Get the rotation wrong and the cache still
holds every token exactly once -- so nothing about it looks wrong -- while each
one is offset by `seqlen % window` slots, and the first decode step reads a
window whose tokens are real, plausible and in the wrong places.

The second thing worth writing down is that `window_topk_idxs` returns ids in
TWO DIFFERENT INDEX SPACES, chosen by `start_pos`:

    prefill (start_pos == 0)  the caller passes the chunk itself as KV, so the
                              ids are positions WITHIN THE CHUNK, one causal
                              row per query
    decode  (start_pos > 0)   the caller passes the ring, so the ids are RING
                              SLOTS, one row shared by the single query, listed
                              oldest first

A caller that hands `sparse_attn` the ring while asking for prefill ids reads
`window - seqlen % window` slots of stale KV without an error anywhere.

`-1` is the only sentinel: a slot that holds nothing yet, either because the
query is near the start of the sequence or because the ring has not filled.
`dsv41_sparse_contract.check_topk_idxs` is what enforces that downstream.
"""

from __future__ import annotations

import torch


def ring_slot(position: int, window_size: int) -> int:
    """Where absolute position `position` lives. The whole invariant."""
    return position % window_size


def window_kv_len(window_size: int, seqlen: int, start_pos: int) -> int:
    """How many KV rows the ids returned by `window_topk_idxs` index into.

    Prefill indexes the chunk (`seqlen` rows); decode indexes the ring
    (`window_size` rows). Passing this to `check_topk_idxs` is what turns the
    two index spaces from a comment into a checked contract.
    """
    return seqlen if start_pos == 0 else window_size


def prefill_ring_writes(seqlen: int, window_size: int):
    """The (source slice, destination slice) pairs that seed the ring.

    Source slices index the last `window_size` tokens of the chunk; destination
    slices index ring slots. Returned rather than applied so that a test can
    compare the MAP -- which is where the rotation lives -- instead of only the
    bytes that result from it.

    A chunk no longer than the window needs no rotation: position p is at slot
    p already.
    """
    if seqlen <= window_size:
        return [(slice(0, seqlen), slice(0, seqlen))]
    cutoff = seqlen % window_size
    tail = slice(seqlen - window_size, seqlen)      # the surviving window
    head_len = window_size - cutoff
    return [
        (slice(tail.start, tail.start + head_len), slice(cutoff, window_size)),
        (slice(tail.start + head_len, tail.stop), slice(0, cutoff)),
    ]


def apply_prefill_ring(cache: torch.Tensor, kv: torch.Tensor,
                       window_size: int) -> None:
    """Seed the ring from a prefill chunk, in place. `cache` is [b, window, d]."""
    bsz, seqlen = kv.shape[0], kv.shape[1]
    for src, dst in prefill_ring_writes(seqlen, window_size):
        cache[:bsz, dst] = kv[:, src]


def apply_decode_ring(cache: torch.Tensor, kv: torch.Tensor, window_size: int,
                      start_pos: int) -> None:
    """Write one decode token into its slot, in place."""
    bsz = kv.shape[0]
    cache[:bsz, ring_slot(start_pos, window_size)] = kv.squeeze(1)


def window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int,
                     device=None) -> torch.Tensor:
    """[bsz, queries, topk] int32 slot ids, -1 where the slot holds nothing.

    Order within a row does not matter to `sparse_attn`, which treats every
    slot independently -- but it is kept identical to the reference anyway, so
    that a mismatch is a mismatch rather than a permutation to argue about.
    """
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = ((end - window_size + 1).clamp(0)
                + torch.arange(min(seqlen, window_size), device=device))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size, device=device),
                          torch.arange(oldest, device=device)])
        # A slot id is compared against an absolute position on purpose: while
        # the ring is still filling, slot j holds exactly position j, so the
        # two spaces coincide and the comparison is the fill test. Once
        # start_pos >= window_size no slot is masked and the coincidence stops
        # mattering.
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
