"""GLM's paged KV and position rings, owned by one arena.

Each physical block contains every DSA layer's latent rows and pooled keys
with scales. That makes a block an NVMe transfer unit without repacking.
Kernel slot ids are absolute rows in typed views of the same byte storage;
latent rows remain contiguous, including for the served MLA pointer API.
KDA and indexer rings live together in a second, slot-major arena region.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import lcm, prod

from engine.base.arena import ALIGN
from engine.base.kv import BlockPool, SlotPool


def aligned(n: int, unit: int) -> int:
    return -(-n // unit) * unit


@dataclass(frozen=True)
class StateField:
    name: str
    layer: int
    shape: tuple
    dtype: str
    offset: int


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


def layout(F, layers, draft=None) -> CacheLayout:
    """Declared byte offsets; no CUDA allocation or model execution."""
    layers = tuple(layers)
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
            field("rec", L, (F.spec_k + 1, F.kda_heads_local, F.kda_dim, F.kda_dim), "f32")
    if draft is not None:
        if len(draft) != 4 or any(not isinstance(n, int) or n <= 0 for n in draft):
            raise ValueError("draft cache shape must contain four positive dimensions")
        dl, dw, dkv, dd = draft
        field("draft", -1, (dl, 2, dw, dkv, dd), "bf16")
    # A KDA-only slice still has logical blocks for the runner's token ledger.
    return CacheLayout(aligned(max(1, paged), quantum), aligned(state, ALIGN),
                       token_offsets, pool_offsets, tuple(fields))


def snapshot_layout(F, layers, draft=None):
    """What a prefix checkpoint at a chunk boundary P must keep (base/prefix.py), and its byte size:
    per KDA layer the conv ring's last conv-1 inputs (positions P-conv+1..P-1) and the recurrent
    state at P-1; the drafter's context ring whole (its window is position-addressed the same way).
    The indexer tail ring keeps nothing: a chunk is whole pools, so at P the tail is empty and the
    next pool starts fresh. Returns (nbytes, fields) with fields (name, layer, shape, dtype, offset)."""
    fields, at = [], 0

    def field(name, L, shape, dtype):
        nonlocal at
        at = aligned(at, ALIGN)
        fields.append(StateField(name, L, shape, dtype, at))
        at += prod(shape) * (4 if dtype == "f32" else 2)

    for L in layers:
        if not F.is_dsa(L):
            field("conv", L, (3 * F.kda_heads_local * F.kda_dim, F.conv - 1), "bf16")
            field("rec", L, (F.kda_heads_local, F.kda_dim, F.kda_dim), "f32")
    if draft is not None:
        dl, dw, dkv, dd = draft
        field("draft", -1, (dl, 2, dw, dkv, dd), "bf16")
    return aligned(max(at, 1), ALIGN), tuple(fields)


def stage_bytes(F, layers, max_seqs: int) -> int:
    """The boundary stage: per state slot, one KDA state and conv taps (a snapshot without the drafter ring) where a
    decode step ahead of the host parks the state of a block boundary it crossed (45차 §23: boundaries while generating)."""
    return (max_seqs + 1) * snapshot_layout(F, layers, None)[0]


class Glm53Caches:
    def __init__(self, arena, F, layers, num_blocks: int, max_seqs: int, draft=None, snapshots: int = 0, stage: bool = False):
        import torch

        self.F, self.layers = F, tuple(layers)
        self.layout = layout(F, self.layers, draft)
        self.snapshot_bytes_n, self._snapshot_fields = snapshot_layout(F, self.layers, draft)
        self.snapshots = snapshots
        self.pool = BlockPool(num_blocks, F.block, max_seqs, num_blocks)
        self.slots = SlotPool(max_seqs + 1)
        p = self.layout
        staged = stage_bytes(F, self.layers, max_seqs) if stage else 0
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
            dtype = torch.float32 if f.dtype == "f32" else torch.bfloat16
            size = 4 if f.dtype == "f32" else 2
            strides = tuple(prod(f.shape[i + 1:]) for i in range(len(f.shape)))
            base = self.state.view(dtype)
            self._fields[f.name, f.layer] = base.as_strided(
                (max_seqs + 1, *f.shape), (p.slot_bytes // size, *strides),
                base.storage_offset() + f.offset // size)
        self._snap = {}
        if snapshots:
            self.snapshot_store = arena.carve(snapshots * self.snapshot_bytes_n, "glm53 prefix snapshots")
            for f in self._snapshot_fields:
                dtype = torch.float32 if f.dtype == "f32" else torch.bfloat16
                size = 4 if f.dtype == "f32" else 2
                strides = tuple(prod(f.shape[i + 1:]) for i in range(len(f.shape)))
                base = self.snapshot_store.view(dtype)
                self._snap[f.name, f.layer] = base.as_strided((snapshots, *f.shape), (self.snapshot_bytes_n // size, *strides),
                                                              base.storage_offset() + f.offset // size)
        self._stage = {}
        if stage:
            self.stage_bytes, self._stage_fields = snapshot_layout(F, self.layers, None)
            self.stage_store = arena.carve((max_seqs + 1) * self.stage_bytes, "glm53 boundary stage")
            for f in self._stage_fields:
                dtype = torch.float32 if f.dtype == "f32" else torch.bfloat16
                size = 4 if f.dtype == "f32" else 2
                strides = tuple(prod(f.shape[i + 1:]) for i in range(len(f.shape)))
                base = self.stage_store.view(dtype)
                self._stage[f.name, f.layer] = base.as_strided((max_seqs + 1, *f.shape), (self.stage_bytes // size, *strides),
                                                               base.storage_offset() + f.offset // size)
        record = F.idx_dim + 4
        self._latent = self.paged.view(torch.float8_e4m3fn).view(-1, F.kv_lora)
        self._keys = self.paged.as_strided((self.paged.numel() // record, F.idx_dim),
                                         (record, 1)).view(torch.float8_e4m3fn)
        base = self.paged.view(torch.float32)
        self._scales = base.as_strided((self.paged.numel() // record,), (record // 4,),
                                      base.storage_offset() + F.idx_dim // 4)
        self.reset()

    def reset(self):
        """Clear contents at boot/check boundaries; does not change ownership."""
        self.paged.zero_()
        self.state.zero_()
        self.block_table.fill_(-1)
        self._table_blocks = array("i", [0]) * self.pool.max_seqs
        self._table_epochs = array("Q", self.pool.epochs)

    def slot_bytes(self, slot: int):
        """A real slot's bytes as one contiguous uint8 arena view: what the tier parks and restores."""
        if not 0 < slot < self.slots.num_slots:
            raise IndexError("only a real state slot has bytes to move")
        n = self.layout.slot_bytes
        return self.state[slot * n:(slot + 1) * n]

    def reset_slot(self, slot: int):
        self.slot_bytes(slot).zero_()

    def prepare(self, step):
        """Publish changed block mappings before a step, after its reservation.

        Segment bounds and slot ownership are checked without device reads.
        Rows append within an allocator epoch: upload only their new suffix.
        Release/reuse or NVMe resume changes the epoch, requiring a fresh prefix
        and clearing any stale suffix. Unchanged rows perform no CUDA work.
        """
        import torch

        for s in step.segments:
            row = self.pool.row(s.seq)
            if not 0 < s.slot < self.slots.num_slots or self.slots.owner[s.slot] != s.seq:
                raise ValueError(f"seq {s.seq} does not own state slot {s.slot}")
            if s.ctx < 0 or s.length <= 0 or s.ctx + s.length > self.pool.tokens[s.seq]:
                raise ValueError(f"seq {s.seq} step exceeds its reserved context")
            count = self.pool.blocks_for(self.pool.tokens[s.seq])
            previous = self._table_blocks[s.seq]
            epoch = self.pool.epochs[s.seq]
            if epoch == self._table_epochs[s.seq] and count == previous:
                continue
            start = previous if epoch == self._table_epochs[s.seq] else 0
            # A released row already contains -1 after its active prefix. Send
            # that padding with the new ids to clear stale entries in one copy.
            end = max(count, previous)
            self.block_table[s.seq, start:end].copy_(torch.tensor(row[start:end], dtype=torch.int32))
            # Commit only after the copy succeeds, so a failed update retries.
            self._table_blocks[s.seq] = count
            self._table_epochs[s.seq] = epoch

    def _ring_cells(self, position: int, count: int, width: int):
        import torch
        return torch.tensor([(position - count + i) % width for i in range(count)], device=self.device)

    def checkpoint(self, slot: int, position: int, snap: int) -> None:
        """Copy the rings' state at chunk boundary `position` out of `slot` into snapshot `snap`."""
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

    def mark_kda(self, layer: int, snap: int, state, taps) -> None:
        """A block boundary inside a prefill step: the layer's recurrent state there [H, K, V] and the conv inputs of the
        conv-1 positions before it [conv-1, C], straight into snapshot `snap` (net._kda cuts the recurrence at the mark)."""
        if not 0 <= snap < self.snapshots:
            raise IndexError("a mark needs a declared snapshot")
        self._snap["rec", layer][snap].copy_(state)
        self._snap["conv", layer][snap].copy_(taps.T)

    def mark_draft(self, snap: int, slot: int) -> None:
        """The drafter's context ring as it stands before the step's observations: the mark's ring is this plus the
        step's positions before the mark (the adapter observes them into the snapshot itself)."""
        if not 0 <= snap < self.snapshots or not 0 < slot < self.slots.num_slots:
            raise IndexError("a mark needs a declared snapshot and a real state slot")
        if ("draft", -1) in self._snap:
            self._snap["draft", -1][snap].copy_(self.draft_ring(slot))

    def snapshot_draft_ring(self, snap: int):
        return self._snap["draft", -1][snap]

    def snapshot_bytes(self, snap: int):
        """A snapshot's bytes as one contiguous uint8 arena view: what the prefix tier writes and reads back."""
        if not 0 <= snap < self.snapshots:
            raise IndexError("only a declared snapshot has bytes to move")
        n = self.snapshot_bytes_n
        return self.snapshot_store[snap * n:(snap + 1) * n]

    # -- boundaries crossed while generating (45차 §23) ----------------------------------------------------------
    def stage_boundaries(self, slots, ctx_before, counts) -> None:
        """For every row of a decode step (device tensors [n]: state slot, context before the step, tokens committed):
        if the step crossed a block boundary P (ctx_before < P <= ctx_before + count), park the KDA state at P-1 and
        the conv inputs before P in the slot's stage. The rings hold them now; a step ahead of the host would have
        overwritten them by the time the host asks. Nothing moves for rows that did not cross."""
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

    def checkpoint_from_stage(self, slot: int, snap: int) -> None:
        """The staged boundary of `slot` into snapshot `snap`. The drafter's ring is taken live: the steps since the
        boundary wrote at most a dozen positions past it, which land on the oldest cells of its 2,048 window."""
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
            self._snap["draft", -1][snap].copy_(self.draft_ring(slot))

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
