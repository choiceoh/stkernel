"""A captured Qwen3.8 decode step's addressing in one launch (engine/profiles/qwen38/net.Qwen38Net.step_meta).

A captured step's StepMeta was about forty torch launches -- casts, floor divisions, remainders, two advanced-index
gathers, `where`s and `full_like`s over the step's rows and tokens -- built twice a step (the verify graph and the MTP
draft graph). Every value is integer arithmetic on the rows' contexts, slots and sequences and the block table, so one
program a row computes the same bytes: the positions (int64 and int32), each token's row, the row's page-table entries
at the bucket's width with unreserved (-1) entries read as page 0, the lengths, the row offsets, the slot table, and
the KV / index-key / raw-key-ring slots. GLM's #819 and #821 folded its DSA layer's addressing the same way.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _addresses(CTX, SLOTS, SEQS, TABLE, POS, POS32, ROWS, KV, KEY, RING, PAGES, LENGTHS, STARTS, SLOT_TABLE,
               sC, sS, sQ, sT0, sT1, sP0, T: tl.constexpr, BT: tl.constexpr, BLOCKS: tl.constexpr, BB: tl.constexpr,
               BLOCK: tl.constexpr, RATIO: tl.constexpr, PER_GROUP: tl.constexpr, RING_SIZE: tl.constexpr):
    r = tl.program_id(0)
    ctx = tl.load(CTX + r * sC)
    slot = tl.load(SLOTS + r * sS)
    seq = tl.load(SEQS + r * sQ)
    row = TABLE + seq * sT0

    b = tl.arange(0, BB)
    bm = b < BLOCKS
    tl.store(PAGES + r * sP0 + b, tl.maximum(tl.load(row + b * sT1, mask=bm, other=0), 0), mask=bm)
    length = ctx + T
    tl.store(LENGTHS + r, length.to(tl.int32))
    tl.store(STARTS + r + 1, ((r + 1) * T).to(tl.int32))
    tl.store(STARTS, tl.zeros_like(r).to(tl.int32), mask=r == 0)
    tl.store(SLOT_TABLE + r, slot.to(tl.int32))

    j = tl.arange(0, BT)
    jm = j < T
    pos = ctx + j
    at = r * T + j
    tl.store(POS + at, pos, mask=jm)
    tl.store(POS32 + at, pos.to(tl.int32), mask=jm)
    tl.store(ROWS + at, (tl.zeros_like(j) + r).to(tl.int32), mask=jm)
    page = tl.maximum(tl.load(row + (pos // BLOCK) * sT1, mask=jm, other=0), 0)
    tl.store(KV + at, (page.to(tl.int64) * BLOCK + pos % BLOCK).to(tl.int32), mask=jm)
    group = pos // RATIO
    key_page = tl.maximum(tl.load(row + (group // PER_GROUP) * sT1, mask=jm, other=0), 0)
    key = tl.where((pos + 1) % RATIO == 0, key_page.to(tl.int64) * PER_GROUP + group % PER_GROUP, -1)
    tl.store(KEY + at, key.to(tl.int32), mask=jm)
    ring = tl.where(pos >= length - RING_SIZE, slot * RING_SIZE + pos % RING_SIZE, -1)
    tl.store(RING + at, ring.to(tl.int32), mask=jm)


def captured(contexts: torch.Tensor, slots: torch.Tensor, seqs: torch.Tensor, block_table: torch.Tensor, *,
             tokens: int, blocks: int, block: int, ratio: int, ring: int):
    """(positions i64, positions i32, rows_req i32 [N], page_table i32 [n, blocks], lengths i32 [n], starts i32 [n+1],
    slot_table i32 [n, 1], kv_slots, key_slots, ring_slots i32 [N]) for n rows of `tokens` each, N = n * tokens."""
    n = contexts.numel()
    if (contexts.shape != (n,) or slots.shape != (n,) or seqs.shape != (n,)
            or not (contexts.dtype == slots.dtype == seqs.dtype == torch.int64)
            or block_table.ndim != 2 or block_table.dtype != torch.int32 or not 0 < blocks <= block_table.shape[1]
            or tokens <= 0 or block % ratio or ring <= 0
            or not (contexts.device == slots.device == seqs.device == block_table.device)):
        raise ValueError("captured addressing takes int64 contexts, slots, seqs [n] and an int32 block table on one "
                         "device, a positive token count and a block of whole groups")
    dev, N = contexts.device, n * tokens
    out = dict(positions=torch.empty(N, dtype=torch.int64, device=dev),
               positions32=torch.empty(N, dtype=torch.int32, device=dev),
               rows_req=torch.empty(N, dtype=torch.int32, device=dev),
               page_table=torch.empty(n, blocks, dtype=torch.int32, device=dev),
               lengths=torch.empty(n, dtype=torch.int32, device=dev),
               starts=torch.empty(n + 1, dtype=torch.int32, device=dev),
               slot_table=torch.empty(n, 1, dtype=torch.int32, device=dev),
               kv_slots=torch.empty(N, dtype=torch.int32, device=dev),
               key_slots=torch.empty(N, dtype=torch.int32, device=dev),
               ring_slots=torch.empty(N, dtype=torch.int32, device=dev))
    if n:
        o = out
        _addresses[(n,)](contexts, slots, seqs, block_table, o["positions"], o["positions32"], o["rows_req"],
                         o["kv_slots"], o["key_slots"], o["ring_slots"], o["page_table"], o["lengths"], o["starts"],
                         o["slot_table"], contexts.stride(0), slots.stride(0), seqs.stride(0), block_table.stride(0),
                         block_table.stride(1), o["page_table"].stride(0),
                         T=tokens, BT=triton.next_power_of_2(tokens), BLOCKS=blocks, BB=triton.next_power_of_2(blocks),
                         BLOCK=block, RATIO=ratio, PER_GROUP=block // ratio, RING_SIZE=ring, num_warps=1)
    else:
        out["starts"].zero_()
    return (out["positions"], out["positions32"], out["rows_req"], out["page_table"], out["lengths"], out["starts"],
            out["slot_table"], out["kv_slots"], out["key_slots"], out["ring_slots"])


__all__ = ["captured"]
