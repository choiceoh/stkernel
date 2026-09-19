"""A served model's caches in one arena: paged blocks, per-sequence state slots, prefix snapshots (base).

Every profile that serves carves the same four regions and moves the same bytes between them: a region of paged
blocks the `BlockPool` hands out, a region of state slots the `SlotPool` hands out (slot 0 is the null slot), a
[max_seqs, num_blocks] int32 block table the kernels read, and optionally a region of prefix snapshots the tier writes
and reads back. What differs between models is which fields a slot or a block holds and how a boundary is cut -- the
profile's layout and its checkpoint/restore. What does not differ is written here once:

    reset / reset_slot            clear contents at boot and check boundaries; ownership is unchanged
    slot_bytes / snapshot_bytes   one contiguous uint8 view of a slot or a snapshot: what the tier parks and restores
    prepare                       publish a step's changed block-table rows, and only their new suffix
    typed_view                    a field of a slot-major (or snapshot-major) region as a strided tensor

`prepare` uploads through a ring of pinned host buffers fenced by events, so the copy that reads a slot has landed
before the slot is reused, and a step's table update does not wait for the device. It was one profile's until
2026-09-19; the other published with a synchronous copy per row.

A subclass sets, before calling `reset`: `pool` (BlockPool), `slots` (SlotPool), `device`, `paged`, `state`,
`block_table`, `layout.slot_bytes`, `snapshots`, and when `snapshots` is nonzero `snapshot_store` and
`snapshot_bytes_n`.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import prod

SIZES = {"f32": 4, "f16": 2, "bf16": 2, "i64": 8}
ID_RING = 16                    # pinned upload buffers in flight before the oldest must have landed


def aligned(n: int, unit: int) -> int:
    return -(-n // unit) * unit


def field_dtype(name):
    import torch
    return {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16, "i64": torch.int64}[name]


@dataclass(frozen=True)
class StateField:
    name: str
    layer: int
    shape: tuple
    dtype: str
    offset: int


def typed_view(storage, count: int, stride_bytes: int, f: StateField):
    """`f` in each of `count` records of `stride_bytes` in `storage` (uint8): [count, *f.shape] of f's dtype."""
    dtype = field_dtype(f.dtype)
    size = SIZES[f.dtype]
    strides = tuple(prod(f.shape[i + 1:]) for i in range(len(f.shape)))
    base = storage.view(dtype)
    return base.as_strided((count, *f.shape), (stride_bytes // size, *strides), base.storage_offset() + f.offset // size)


class SlotCaches:
    """The regions' shared moves. A profile's caches subclass it and add their fields, layout and boundaries."""

    def reset(self):
        """Clear contents at boot and check boundaries; ownership is unchanged."""
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

    def snapshot_bytes(self, snap: int):
        """A snapshot's bytes as one contiguous uint8 arena view: what the prefix tier writes and reads back."""
        if not 0 <= snap < self.snapshots:
            raise IndexError("only a declared snapshot has bytes to move")
        n = self.snapshot_bytes_n
        return self.snapshot_store[snap * n:(snap + 1) * n]

    def prepare(self, step):
        """Publish changed block mappings before a step, after its reservation.

        Segment bounds and slot ownership are checked without device reads.
        Rows append within an allocator epoch: upload only their new suffix.
        Release/reuse or NVMe resume changes the epoch, requiring a fresh prefix
        and clearing any stale suffix. Unchanged rows perform no CUDA work.
        """
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
            self._upload_ids(self.block_table[s.seq, start:end], row[start:end])
            # Commit only after the copy succeeds, so a failed update retries.
            self._table_blocks[s.seq] = count
            self._table_epochs[s.seq] = epoch

    def _upload_ids(self, destination, ids):
        """Upload through a pinned ring; fence reuse after the copy that reads the slot."""
        import torch
        if self.device.type != "cuda":
            destination.copy_(torch.tensor(ids, dtype=torch.int32))
            return
        ring = getattr(self, "_id_ring", None)
        if ring is None:
            width = self.block_table.shape[1]
            ring = self._id_ring = [(torch.empty(width, dtype=torch.int32, pin_memory=True), torch.cuda.Event())
                                    for _ in range(ID_RING)]
            self._id_ring_next = 0
        host, event = ring[self._id_ring_next]
        self._id_ring_next = (self._id_ring_next + 1) % len(ring)
        event.synchronize()                                       # the slot's previous copy has landed (almost always already)
        n = len(ids)
        host[:n].copy_(torch.tensor(ids, dtype=torch.int32))
        destination.copy_(host[:n], non_blocking=True)
        event.record()

    def _ring_cells(self, position: int, count: int, width: int):
        """The `count` ring cells before `position` in a ring `width` long, oldest first, on the caches' device."""
        import torch
        return torch.tensor([(position - count + i) % width for i in range(count)], device=self.device)
