"""GLM's paged KV and position rings, owned by one arena.

Each physical block contains every DSA layer's latent rows and pooled keys
with scales. That makes a block an NVMe transfer unit without repacking.
Kernel slot ids are absolute rows in typed views of the same byte storage;
latent rows remain contiguous, including for the served MLA pointer API.
KDA and indexer rings live together in a second, slot-major arena region.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lcm, prod

from engine.base.arena import ALIGN
from engine.base.kv import BlockPool, SlotPool
from engine.base.slot_caches import SlotCaches, StateField, aligned, typed_view


def state_dtype(value: str) -> str:
    if value not in ("fp32", "fp16"):
        raise ValueError("KDA state storage must be fp32 or fp16")
    return value


def recurrent_field_dtype(F, override=None) -> str:
    return {"fp32": "f32", "fp16": "f16"}[state_dtype(
        getattr(F, "kda_state_dtype", "fp32") if override is None else override)]


@dataclass(frozen=True)
class CacheLayout:
    block_bytes: int
    slot_bytes: int
    token_offsets: dict
    pool_offsets: dict
    fields: tuple

    def nbytes(self, num_blocks: int, max_seqs: int) -> int:
        # The block table has one row per request id; state slot 0 is null.
        return (num_blocks * self.block_bytes + (max_seqs + 1) * self.slot_bytes
                + max_seqs * num_blocks * 4)


def layout(F, layers, draft=None, *, state_storage=None) -> CacheLayout:
    """Declared byte offsets; no CUDA allocation or model execution."""
    layers = tuple(layers)
    recurrent_dtype = recurrent_field_dtype(F, state_storage)
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= L < F.layers for L in layers):
        raise ValueError("cache layers must be nonempty, unique and inside the model")
    if F.block <= 0 or F.kpool <= 0 or F.block % F.kpool:
        raise ValueError("cache blocks must contain whole indexer pools")
    record = F.idx_dim + 4                       # fp8 key plus its fp32 scale
    quantum = lcm(4096, F.kv_lora, record)
    token_offsets, pool_offsets, fields = {}, {}, []
    paged = state = 0

    def field(name, L, shape, dtype):
        nonlocal state
        state = aligned(state, ALIGN)
        fields.append(StateField(name, L, shape, dtype, state))
        state += prod(shape) * (4 if dtype == "f32" else 2)

    for L in layers:
        if F.is_dsa(L):
            paged = aligned(paged, F.kv_lora)
            token_offsets[L] = paged
            paged += F.block * F.kv_lora
            paged = aligned(paged, record)
            pool_offsets[L] = paged
            paged += (F.block // F.kpool) * record
            field("tail", L, (F.kpool - 1 + F.spec_k, 2, F.idx_dim), "bf16")
        else:
            field("conv", L, (3 * F.kda_heads_local * F.kda_dim, F.conv - 1 + F.spec_k), "bf16")
            field("rec", L, (F.spec_k + 1, F.kda_heads_local, F.kda_dim, F.kda_dim), recurrent_dtype)
    if draft is not None:
        if len(draft) != 4 or any(not isinstance(n, int) or n <= 0 for n in draft):
            raise ValueError("draft cache shape must contain four positive dimensions")
        dl, dw, dkv, dd = draft
        field("draft", -1, (dl, 2, dw, dkv, dd), "bf16")
    # A KDA-only slice still has logical blocks for the runner's token ledger.
    return CacheLayout(aligned(max(1, paged), quantum), aligned(state, ALIGN),
                       token_offsets, pool_offsets, tuple(fields))


def snapshot_layout(F, layers, draft=None, *, state_storage=None):
    """What a prefix checkpoint at a chunk boundary P must keep (base/prefix.py), and its byte size:
    per KDA layer the conv ring's last conv-1 inputs (positions P-conv+1..P-1) and the recurrent
    state at P-1; the drafter's context ring whole (its window is position-addressed the same way).
    The indexer tail ring keeps nothing: a chunk is whole pools, so at P the tail is empty and the
    next pool starts fresh. Returns (nbytes, fields) with fields (name, layer, shape, dtype, offset)."""
    fields, at = [], 0
    recurrent_dtype = recurrent_field_dtype(F, state_storage)

    def field(name, L, shape, dtype):
        nonlocal at
        at = aligned(at, ALIGN)
        fields.append(StateField(name, L, shape, dtype, at))
        at += prod(shape) * (4 if dtype == "f32" else 2)

    for L in layers:
        if not F.is_dsa(L):
            field("conv", L, (3 * F.kda_heads_local * F.kda_dim, F.conv - 1), "bf16")
            field("rec", L, (F.kda_heads_local, F.kda_dim, F.kda_dim), recurrent_dtype)
    if draft is not None:
        dl, dw, dkv, dd = draft
        field("draft", -1, (dl, 2, dw, dkv, dd), "bf16")
    return aligned(max(at, 1), ALIGN), tuple(fields)


def cache_capacity(F, layers, draft, kv_gib: float, max_seqs: int, snapshot_gib: float):
    """Keep FP32's KV blocks and snapshot count when reducing state precision.

    The budgets describe the baseline capacity, not a target to fill with more
    KV or snapshots. Only the actual typed regions are allocated by the caller.
    """
    layers = tuple(layers)
    baseline = layout(F, layers, draft, state_storage="fp32")
    reference_snapshot = snapshot_layout(F, layers, draft, state_storage="fp32")[0]
    blocks = int((kv_gib * (1 << 30) - (max_seqs + 1) * baseline.slot_bytes)
                 // (baseline.block_bytes + max_seqs * 4))
    snapshots = max(9, int(snapshot_gib * (1 << 30)) // reference_snapshot)
    return blocks, snapshots


def draft_stash_cells(F) -> int:
    """How many drafter ring cells past a block boundary the stage keeps. The ring files a position's key in the cell
    of the position a window before it, so every position written past boundary P destroys a cell a snapshot at P
    needs. The async chain runs up to two decode steps ahead of the host (pipeline.AsyncDecode's depth), each
    committing at most spec_k + 1 tokens: the crossing step and the two after it write fewer than three blocks' worth
    before the host takes the snapshot (a burst stops at the boundary, a synchronous step waits for it)."""
    return 3 * (F.spec_k + 1)


def stage_layout(F, layers, draft=None):
    """The boundary stage's per-slot layout: a snapshot's KDA state and conv taps, and -- with a drafter -- its ring's
    cells for positions P .. P + draft_stash_cells - 1 as they were before the steps past P wrote them."""
    stash = None if draft is None else (draft[0], draft_stash_cells(F), draft[2], draft[3])
    return snapshot_layout(F, layers, stash)


def stage_bytes(F, layers, max_seqs: int, draft=None) -> int:
    """The boundary stage: per state slot, one KDA state and conv taps where a decode step ahead of the host parks
    the state of a block boundary it crossed (45차 §23: boundaries while generating), and the drafter ring cells
    that step and the ones after it overwrite (`stage_layout`)."""
    return (max_seqs + 1) * stage_layout(F, layers, draft)[0]


class Glm53Caches(SlotCaches):
    def __init__(self, arena, F, layers, num_blocks: int, max_seqs: int, draft=None, snapshots: int = 0, stage: bool = False):
        import torch

        self.F, self.layers = F, tuple(layers)
        if stage and draft is not None and draft_stash_cells(F) >= draft[1]:
            raise ValueError("the boundary stage keeps a drafter window's worth of cells or more: the ring is too short")
        self.layout = layout(F, self.layers, draft)
        self.snapshot_bytes_n, self._snapshot_fields = snapshot_layout(F, self.layers, draft)
        self.snapshots = snapshots
        self.pool = BlockPool(num_blocks, F.block, max_seqs, num_blocks)
        self.slots = SlotPool(max_seqs + 1)
        p = self.layout
        staged = stage_bytes(F, self.layers, max_seqs, draft) if stage else 0
        # Preflight all regions, including alignment at an existing arena cursor.
        if aligned(arena.used, ALIGN) + p.nbytes(num_blocks, max_seqs) + snapshots * self.snapshot_bytes_n + staged > arena.nbytes:
            raise MemoryError("arena cannot hold the declared GLM caches, block table, prefix snapshots and boundary stage")
        self.paged = arena.carve(num_blocks * p.block_bytes, "glm53 paged KV")
        self.device = self.paged.device
        self.pool.attach_storage(self.paged, p.block_bytes)
        self.state = arena.carve((max_seqs + 1) * p.slot_bytes, "glm53 state slots")
        self.block_table = arena.carve(max_seqs * num_blocks * 4, "glm53 block table").view(torch.int32).view(max_seqs, num_blocks)
        self._fields = {}
        for f in p.fields:
            self._fields[f.name, f.layer] = typed_view(self.state, max_seqs + 1, p.slot_bytes, f)
        self._snap = {}
        if snapshots:
            self.snapshot_store = arena.carve(snapshots * self.snapshot_bytes_n, "glm53 prefix snapshots")
            for f in self._snapshot_fields:
                self._snap[f.name, f.layer] = typed_view(self.snapshot_store, snapshots, self.snapshot_bytes_n, f)
        self._stage = {}
        if stage:
            self.stage_bytes, self._stage_fields = stage_layout(F, self.layers, draft)
            self.stage_store = arena.carve((max_seqs + 1) * self.stage_bytes, "glm53 boundary stage")
            for f in self._stage_fields:
                self._stage[f.name, f.layer] = typed_view(self.stage_store, max_seqs + 1, self.stage_bytes, f)
        record = F.idx_dim + 4
        self._latent = self.paged.view(torch.float8_e4m3fn).view(-1, F.kv_lora)
        self._keys = self.paged.as_strided((self.paged.numel() // record, F.idx_dim),
                                         (record, 1)).view(torch.float8_e4m3fn)
        base = self.paged.view(torch.float32)
        self._scales = base.as_strided((self.paged.numel() // record,), (record // 4,),
                                      base.storage_offset() + F.idx_dim // 4)
        self.reset()

    def checkpoint(self, slot: int, position: int, snap: int, past: int = 0) -> None:
        """Copy the rings' state at chunk boundary `position` out of `slot` into snapshot `snap`. `past`: how many
        positions after the boundary the rings already hold -- a synchronous decode step that crossed it, whose
        drafter cells were put aside first (`stash_draft`)."""
        F = self.F
        if not 0 <= snap < self.snapshots or not 0 < slot < self.slots.num_slots:
            raise IndexError("checkpoint needs a real state slot and a declared snapshot")
        if position <= 0 or position % F.block:
            raise ValueError("a checkpoint sits at a block boundary")
        conv_cells = self._ring_cells(position, F.conv - 1, F.conv - 1 + F.spec_k)
        rec_cell = (position - 1) % (F.spec_k + 1)
        for L in self.layers:
            if F.is_dsa(L):
                continue
            conv_ring, rec_ring = self.kda(L, slot)
            self._snap["conv", L][snap].copy_(conv_ring.index_select(1, conv_cells))
            self._snap["rec", L][snap].copy_(rec_ring[rec_cell])
        if ("draft", -1) in self._snap:
            self._snap["draft", -1][snap].copy_(self.draft_ring(slot))
            if past:
                if not 0 < past <= draft_stash_cells(F) or ("draft", -1) not in self._stage:
                    raise ValueError("a checkpoint past its boundary takes one decode step's drafter cells from the stage")
                self._put_back_draft(slot, position, snap)

    def mark_kda(self, layer: int, snap: int, state, taps) -> None:
        """A block boundary inside a prefill step: the layer's recurrent state there [H, K, V] and the conv inputs of the
        conv-1 positions before it [conv-1, C], straight into snapshot `snap` (net._kda cuts the recurrence at the mark)."""
        if not 0 <= snap < self.snapshots:
            raise IndexError("a mark needs a declared snapshot")
        self._snap["rec", layer][snap].copy_(state)
        self._snap["conv", layer][snap].copy_(taps.T)

    def mark_draft(self, snap: int, slot: int) -> None:
        """Copy the drafter's current context ring into a block-boundary snapshot."""
        if not 0 <= snap < self.snapshots or not 0 < slot < self.slots.num_slots:
            raise IndexError("a mark needs a declared snapshot and a real state slot")
        if ("draft", -1) in self._snap:
            self._snap["draft", -1][snap].copy_(self.draft_ring(slot))

    def snapshot_draft_ring(self, snap: int):
        return self._snap["draft", -1][snap]

    # -- boundaries crossed while generating (45차 §23) ----------------------------------------------------------
    def stage_boundaries(self, slots, ctx_before, counts) -> None:
        """For every row of a decode step (device tensors [n]: state slot, context before the step, tokens committed):
        if the step crossed a block boundary P (ctx_before < P <= ctx_before + count), park the KDA state at P-1 and
        the conv inputs before P in the slot's stage, with the drafter ring cells the positions past P will overwrite
        (called before the step's observe). The rings hold them now; a step ahead of the host would have overwritten
        them by the time the host asks. Nothing moves for rows that did not cross."""
        import torch
        if not self._stage:
            raise RuntimeError("this cache has no boundary stage")
        F = self.F
        n = int(slots.numel())
        if self.device.type == "cuda":
            from engine.kernels.state import stage_boundaries
            stage_boundaries(self, slots, ctx_before, counts)
            return
        for i in range(n):                                          # the eager form: what the kernel does, for tests
            slot, before, count = int(slots[i]), int(ctx_before[i]), int(counts[i])
            after = before + count
            P = (after // F.block) * F.block
            if count <= 0 or P <= before:
                continue
            conv_cells = self._ring_cells(P, F.conv - 1, F.conv - 1 + F.spec_k)
            rec_cell = (P - 1) % (F.spec_k + 1)
            for L in self.layers:
                if F.is_dsa(L):
                    continue
                conv_ring, rec_ring = self.kda(L, slot)
                self._stage["conv", L][slot].copy_(conv_ring.index_select(1, conv_cells))
                self._stage["rec", L][slot].copy_(rec_ring[rec_cell])
            self.stash_draft(slot, P)                               # before this step's observe writes past P

    def stash_draft(self, slot: int, position: int) -> None:
        """Put aside the drafter ring cells for positions `position` .. + draft_stash_cells - 1 before a decode step
        that crosses block boundary `position` writes them: what `stage_boundaries` does for the steps ahead of the
        host, for a synchronous step."""
        if ("draft", -1) not in self._stage:
            return
        cells = draft_stash_cells(self.F)
        ring = self.draft_ring(slot)
        self._stage["draft", -1][slot].copy_(ring.index_select(2, self._ring_cells(position + cells, cells, ring.shape[2])))

    def _put_back_draft(self, slot: int, position: int, snap: int) -> None:
        """Snapshot `snap` holds `slot`'s live drafter ring: put back the cells of positions `position` .. from the
        stage, as they were before the positions past the boundary were written over them."""
        cells = draft_stash_cells(self.F)
        ring = self._snap["draft", -1][snap]
        ring.index_copy_(2, self._ring_cells(position + cells, cells, ring.shape[2]), self._stage["draft", -1][slot])

    def checkpoint_from_stage(self, slot: int, snap: int, position: "int | None" = None) -> None:
        """The staged boundary `position` of `slot` into snapshot `snap`. The drafter's ring is taken live with the
        cells the steps past the boundary overwrote put back from the stage: each position past it lands on the cell
        of the position a window before, which the snapshot still needs (left there, the restored row reads a key
        rotated to a position after the boundary, from another request, as its oldest context)."""
        if not 0 <= snap < self.snapshots or not 0 < slot < self.slots.num_slots:
            raise IndexError("checkpoint needs a real state slot and a declared snapshot")
        if not self._stage:
            raise RuntimeError("this cache has no boundary stage")
        for L in self.layers:
            if self.F.is_dsa(L):
                continue
            self._snap["conv", L][snap].copy_(self._stage["conv", L][slot])
            self._snap["rec", L][snap].copy_(self._stage["rec", L][slot])
        if ("draft", -1) in self._snap:
            if position is None or position <= 0 or position % self.F.block:
                raise ValueError("a staged drafter ring is put back at its block boundary")
            self._snap["draft", -1][snap].copy_(self.draft_ring(slot))
            if ("draft", -1) in self._stage:
                self._put_back_draft(slot, position, snap)

    def restore(self, slot: int, position: int, snap: int) -> None:
        """The inverse: `slot` continues from `position` with the snapshot's state."""
        F = self.F
        if not 0 <= snap < self.snapshots or not 0 < slot < self.slots.num_slots:
            raise IndexError("restore needs a real state slot and a declared snapshot")
        if position <= 0 or position % F.block:
            raise ValueError("a restore sits at a block boundary")
        conv_cells = self._ring_cells(position, F.conv - 1, F.conv - 1 + F.spec_k)
        rec_cell = (position - 1) % (F.spec_k + 1)
        for L in self.layers:
            if F.is_dsa(L):
                continue
            conv_ring, rec_ring = self.kda(L, slot)
            conv_ring.index_copy_(1, conv_cells, self._snap["conv", L][snap])
            rec_ring[rec_cell].copy_(self._snap["rec", L][snap])
        if ("draft", -1) in self._snap:
            self.draft_ring(slot).copy_(self._snap["draft", -1][snap])

    def kda(self, layer, slot):
        return self._fields["conv", layer][slot], self._fields["rec", layer][slot]

    def tail(self, layer, slot):
        return self._fields["tail", layer][slot]

    def draft_ring(self, slot):
        return self._fields["draft", -1][slot]

    def draft_field(self):
        """Every slot's drafter ring, [slots, layers, 2, cells, kv, D]: the batched drafter paths address rows by slot."""
        return self._fields["draft", -1]

    def latent(self, layer):
        return self._latent

    def pool_keys(self, layer):
        return self._keys

    def pool_scales(self, layer):
        return self._scales

    def token_slots(self, layer, seq, positions):
        F, p = self.F, self.layout
        blocks = self.block_table[seq][(positions // F.block).long()]
        return (blocks * (p.block_bytes // F.kv_lora)
                + p.token_offsets[layer] // F.kv_lora + positions % F.block).to(blocks.dtype)

    def token_map(self, layer, seq):
        """Block row and scalar strides, measured in latent rows, for a lane."""
        F, p = self.F, self.layout
        return self.block_table[seq], F.block, p.block_bytes // F.kv_lora, p.token_offsets[layer] // F.kv_lora

    def pool_slots(self, layer, seq, pool_ids):
        F, p = self.F, self.layout
        per, record = F.block // F.kpool, F.idx_dim + 4
        blocks = self.block_table[seq][(pool_ids // per).long()]
        return (blocks * (p.block_bytes // record)
                + p.pool_offsets[layer] // record + pool_ids % per).to(blocks.dtype)

    def pool_map(self, layer, seq):
        """Block row and scalar strides, measured in key/scale records."""
        F, p = self.F, self.layout
        record = F.idx_dim+4
        return self.block_table[seq], F.block//F.kpool, p.block_bytes//record, p.pool_offsets[layer]//record
