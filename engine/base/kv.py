"""Block tables and state slots, as flat arrays (base).

Two kinds of per-sequence memory exist across the three models and they do
not behave alike, so they get two allocators:

  paged blocks    full-attention KV grows with context; a sequence holds a
                  list of fixed-size blocks and a kernel reads a block table.
  state slots     linear-attention (KDA, GDN) and conv state do NOT grow with
                  context -- one fixed-size slot per sequence for its whole
                  life (Qwen3.8: 27.5 MiB, DSv4.1's compressor tails likewise).

Both are represented the way a kernel wants them (CHARTER I2): a block table
is a [max_seqs, max_blocks] int32 array filled with -1, a free list is an
int32 stack. No Python object graph sits between the scheduler and the launch.

Running out is an error, not a fallback (D3): the budget declared how many
blocks exist before the model loaded, so exhaustion means the scheduler broke
its contract, and hiding that behind an eviction would hide the bug.
"""
from __future__ import annotations

from array import array

EMPTY = -1


class BlockPool:
    """Fixed-size blocks handed out from a stack; a sequence owns a row."""

    def __init__(self, num_blocks: int, block_size: int, max_seqs: int, max_blocks_per_seq: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("a pool needs at least one block of at least one token")
        self.block_size = block_size
        self.num_blocks = num_blocks
        # free stack: block ids, top at the end. int32 so it can be handed to a kernel.
        self.free = array("i", range(num_blocks - 1, -1, -1))
        self.table = array("i", [EMPTY]) * (max_seqs * max_blocks_per_seq)
        self.max_blocks_per_seq = max_blocks_per_seq
        self.max_seqs = max_seqs
        self.tokens = array("i", [0]) * max_seqs           # tokens held per row
        self.rows_in_use = 0
        self.storage = None                                # arena view, once attached
        self.block_bytes = 0

    def attach_storage(self, view, block_bytes: int) -> None:
        """Bind the pool to `num_blocks * block_bytes` of arena (D16)."""
        if view.numel() < self.num_blocks * block_bytes:
            raise ValueError(f"storage holds {view.numel()} B, pool needs "
                             f"{self.num_blocks} x {block_bytes}")
        self.storage, self.block_bytes = view, block_bytes

    def block(self, block_id: int):
        """The bytes of one block -- a view, never a copy."""
        return self.storage[block_id * self.block_bytes:(block_id + 1) * self.block_bytes]

    def blocks_of(self, seq: int) -> "list":
        """A sequence's blocks in order, as views (what a tier demotes)."""
        return [self.block(b) for b in self.row(seq) if b != EMPTY]

    @property
    def available(self) -> int:
        return len(self.free)

    def blocks_for(self, tokens: int) -> int:
        return -(-tokens // self.block_size)

    def row(self, seq: int) -> memoryview:
        base = seq * self.max_blocks_per_seq
        return memoryview(self.table)[base:base + self.max_blocks_per_seq]

    def reserve(self, seq: int, tokens: int) -> int:
        """Grow `seq` to hold `tokens` more. Returns blocks newly taken."""
        if not 0 <= seq < self.max_seqs:
            raise IndexError(f"row {seq} outside {self.max_seqs}")
        have = self.blocks_for(self.tokens[seq])
        need = self.blocks_for(self.tokens[seq] + tokens)
        grow = need - have
        if grow > len(self.free):
            raise MemoryError(
                f"seq {seq} needs {grow} more blocks, {len(self.free)} free: the "
                "scheduler admitted more than the budget declared")
        if need > self.max_blocks_per_seq:
            raise MemoryError(f"seq {seq} would exceed {self.max_blocks_per_seq} blocks")
        row = self.row(seq)
        for i in range(have, need):
            row[i] = self.free.pop()
        if self.tokens[seq] == 0 and tokens:
            self.rows_in_use += 1
        self.tokens[seq] += tokens
        return grow

    def release(self, seq: int) -> int:
        """Give every block of `seq` back. Returns how many."""
        row = self.row(seq)
        n = 0
        for i in range(self.max_blocks_per_seq):
            if row[i] == EMPTY:
                break
            self.free.append(row[i])
            row[i] = EMPTY
            n += 1
        if self.tokens[seq]:
            self.rows_in_use -= 1
        self.tokens[seq] = 0
        return n


class SlotPool:
    """One fixed slot per live sequence, for state that does not grow.

    Slot 0 is never issued. The served conv/state kernels (vLLM's
    causal_conv1d and the KDA state kernels) treat index 0 as the NULL block
    and SKIP a sequence that names it -- no error, no output, state untouched.
    Three diagnostics in the 44th ledger read that silence as a layout bug.
    Reserving 0 here makes the mistake unrepresentable (D3).
    """

    NULL = 0

    def __init__(self, num_slots: int):
        if num_slots < 2:
            raise ValueError("a slot pool needs slot 0 (null) plus at least one real slot")
        self.free = array("i", range(num_slots - 1, 0, -1))      # 1..n-1, never 0
        self.num_slots = num_slots
        self.owner = array("i", [EMPTY]) * num_slots

    @property
    def available(self) -> int:
        return len(self.free)

    def take(self, seq: int) -> int:
        if not self.free:
            raise MemoryError("no state slot left: more live sequences than the budget declared")
        slot = self.free.pop()
        self.owner[slot] = seq
        return slot

    def give(self, slot: int) -> None:
        if slot == self.NULL:
            raise ValueError("slot 0 is the null slot and is never taken")
        if self.owner[slot] == EMPTY:
            raise ValueError(f"slot {slot} is not taken")
        self.owner[slot] = EMPTY
        self.free.append(slot)


def _selfcheck() -> None:
    pool = BlockPool(num_blocks=10, block_size=16, max_seqs=4, max_blocks_per_seq=8)
    assert pool.reserve(0, 17) == 2 and pool.tokens[0] == 17 and pool.available == 8
    assert pool.reserve(0, 15) == 0                     # fits the second block exactly
    assert pool.reserve(0, 1) == 1 and pool.available == 7
    assert list(pool.row(0))[:3] != [EMPTY] * 3 and pool.row(0)[3] == EMPTY
    assert pool.reserve(1, 16 * 7) == 7 and pool.available == 0
    try:
        pool.reserve(2, 1); raise AssertionError("exhaustion must raise")
    except MemoryError:
        pass
    assert pool.release(1) == 7 and pool.available == 7 and pool.tokens[1] == 0
    assert pool.release(0) == 3 and pool.available == 10 and pool.rows_in_use == 0
    # storage: attach a fake arena and read a sequence's blocks back as views
    import array as _a
    fake = memoryview(bytearray(10 * 64))
    class _View:                    # duck-typed like a uint8 torch view for the check
        def __init__(s, mv): s.mv = mv
        def numel(s): return len(s.mv)
        def __getitem__(s, sl): return _View(s.mv[sl])
    pool2 = BlockPool(10, 16, 2, 4); pool2.attach_storage(_View(fake), 64)
    pool2.reserve(0, 40)
    assert len(pool2.blocks_of(0)) == 3 and all(b.numel() == 64 for b in pool2.blocks_of(0))
    slots = SlotPool(3)                                     # slot 0 reserved: two usable
    a, b = slots.take(7), slots.take(9)
    assert a != 0 and b != 0 and slots.owner[a] == 7 and slots.available == 0
    try:
        slots.take(11); raise AssertionError("slot exhaustion must raise")
    except MemoryError:
        pass
    slots.give(a); assert slots.available == 1
    print("  kv: block pool + slot pool OK")


if __name__ == "__main__":
    _selfcheck()
