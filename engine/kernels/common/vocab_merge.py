"""Compact equivalent of the pinned CUDA radix gather followed by small sort.

The CUDA gather writes winners above the cutoff in vocabulary order, then
cutoff ties in vocabulary order. Its final unstable 16-element sort depends on
that input order. Sorting packed keys directly is therefore not equivalent.
This module constructs the same input to the same Torch small sort, without
materializing the vocabulary. See measurements/atlas_candidate_tree_review_20260917.
"""
import torch
import triton
import triton.language as tl

from engine.kernels.common.vocab_candidates import _key, MIN_KEY


@triton.jit
def _preorder(PACKET, VALUES, IDS, COUNT: tl.constexpr, K: tl.constexpr,
              B: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    packet = tl.load(PACKET + row * COUNT + col, col < COUNT, other=MIN_KEY)
    token = 0xffffffff - (packet & 0xffffffff)
    # At a -inf cutoff, the dense background participates too. Its lowest K
    # IDs suffice to fill K winners, but an explicit packet entry overrides it.
    present = tl.sum(((token[:, None] == col[None, :]) &
                      (packet[:, None] != MIN_KEY)).to(tl.int32), 0) > 0
    background = tl.where((col < K) & ~present,
                          _key(tl.full((B,), -float('inf'), tl.float32), col), MIN_KEY)
    keys = tl.cat(packet, background)
    outcol = tl.arange(0, BK)
    chosen = tl.full((BK,), MIN_KEY, tl.int64)
    limit = 0x7fffffffffffffff
    for i in tl.static_range(K):
        live = keys <= limit if i == 0 else keys < limit
        limit = tl.max(tl.where(live, keys, MIN_KEY), 0)
        chosen = tl.where(outcol == i, limit, chosen)
    cutoff = limit >> 32
    ordered = chosen >> 32
    ids = 0xffffffff - (chosen & 0xffffffff)
    group = (ordered == cutoff).to(tl.int32)
    # Rank the selected keys by (strictly-above-cutoff first, original ID).
    before = ((group[:, None] > group[None, :]) |
              ((group[:, None] == group[None, :]) & (ids[:, None] > ids[None, :])))
    slot = tl.sum((before & (outcol[None, :] < K)).to(tl.int32), 1)
    bits = tl.where(ordered < 0, ordered ^ 0x7fffffff, ordered).to(tl.int32)
    tl.store(VALUES + row * K + slot, bits.to(tl.float32, bitcast=True), outcol < K)
    tl.store(IDS + row * K + slot, ids, outcol < K)


def preorder(gathered, vocab, k):
    """Return radix-gather order; also executable with Triton's CPU interpreter.

Packets are the unique valid token IDs/sentinel padding produced by vocab.topk.
The caller owns this invariant, exactly as for the old restore scatter kernel.
"""
    if (gathered.ndim != 2 or gathered.dtype != torch.int64 or not gathered.is_contiguous()
            or type(k) is not int or not 1 <= k <= 32
            or type(vocab) is not int or not k <= vocab < 2**31
            or not 1 <= gathered.shape[1] <= 128):
        raise ValueError('compact merge requires contiguous int64 packets, K<=32 and count<=128')
    rows, count = gathered.shape
    values = torch.empty((rows, k), dtype=torch.float32, device=gathered.device)
    ids = torch.empty((rows, k), dtype=torch.int64, device=gathered.device)
    if rows:
        _preorder[(rows,)](gathered, values, ids, count, k,
                          triton.next_power_of_2(max(count, k)), triton.next_power_of_2(k),
                          num_warps=4)
    return values, ids


def topk(gathered, vocab, k):
    """The pinned dense CUDA top-k ordering, using only bounded candidate arrays."""
    if not gathered.is_cuda:
        raise ValueError('exact compact top-k requires CUDA small-sort semantics')
    values, ids = preorder(gathered, vocab, k)
    values, order = values.sort(dim=-1, descending=True, stable=False)
    return torch.return_types.topk((values, ids.gather(-1, order)))
