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

Blocks are owned by count: a row references the blocks in its table, a prefix
cache pins the blocks of a boundary it keeps (base/prefix.py), and a block
returns to the free stack when its count reaches zero. A row may adopt a
cached prefix (read-only, already complete blocks) ahead of its own
reservation. When the free stack cannot cover a reservation, the pool first
asks the cache to reclaim -- that is the one place an eviction happens, and it
frees blocks nobody but the cache holds; a shortage after that is still the
scheduler's error.
"""
from __future__ import annotations

from array import array

EMPTY = -1


class BlockPool:
    """Fixed-size blocks handed out from a stack; a sequence owns a row."""

    def __init__(self, num_blocks: int, block_size: int, max_seqs: int, max_blocks_per_seq: int):
        if any(not isinstance(n, int) or n <= 0
               for n in (num_blocks, block_size, max_seqs, max_blocks_per_seq)):
            raise ValueError("block count, block size, row count and row width must be positive integers")
        self.block_size = block_size
        self.num_blocks = num_blocks
        # free stack: block ids, top at the end. int32 so it can be handed to a kernel.
        self.free = array("i", range(num_blocks - 1, -1, -1))
        self._table = array("i", [EMPTY]) * (max_seqs * max_blocks_per_seq)
        self.table = memoryview(self._table).toreadonly()
        # Within one epoch a row only appends blocks. Release starts a new
        # epoch, even if its next owner reserves the same number of blocks.
        self._epochs = array("Q", [0]) * max_seqs
        self.epochs = memoryview(self._epochs).toreadonly()
        self.max_blocks_per_seq = max_blocks_per_seq
        self.max_seqs = max_seqs
        self.tokens = array("i", [0]) * max_seqs           # tokens held per row
        self.rows_in_use = 0
        self.storage = None                                # arena view, once attached
        self.block_bytes = 0
        self.refs = array("i", [0]) * num_blocks           # owners per block: rows + cache pins
        self.reclaim = None                                # cache callback: free at least n blocks, returns how many
        self.reclaimable = None                            # cache callback: blocks only the cache holds

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
        """Free now plus what the cache would give back on demand."""
        return len(self.free) + (self.reclaimable() if self.reclaimable is not None else 0)

    def blocks_for(self, tokens: int) -> int:
        return -(-tokens // self.block_size)

    def row(self, seq: int) -> memoryview:
        """Read-only block ids; only reserve/release may change the mapping."""
        if not 0 <= seq < self.max_seqs:
            raise IndexError(f"row {seq} outside {self.max_seqs}")
        base = seq * self.max_blocks_per_seq
        return self.table[base:base + self.max_blocks_per_seq]

    def reserve(self, seq: int, tokens: int) -> int:
        """Grow `seq` to hold `tokens` more. Returns blocks newly taken."""
        return self.reserve_many((seq,), tokens)

    def reserve_many(self, seqs, tokens: int) -> int:
        """Grow a batch by `tokens` per row, all or nothing.

        Validate every row and the combined demand before touching the free
        stack. A decoder never sees a batch whose earlier rows were reserved
        but whose later rows ran out. Called on the runner's owning thread.
        """
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("reserved token count must be a nonnegative integer")
        seqs = tuple(seqs)
        return self._reserve_counts(seqs, (tokens,) * len(seqs))

    def reserve_to(self, seqs, horizons) -> int:
        """Atomically cover absolute write ends, reusing rejected draft space."""
        seqs, horizons = tuple(seqs), tuple(horizons)
        if len(seqs) != len(horizons):
            raise ValueError("one write horizon is required per sequence")
        counts = []
        for seq, end in zip(seqs, horizons):
            self.row(seq)
            if not isinstance(end, int) or not 0 <= end < 2**31:
                raise ValueError("write horizons must be nonnegative int32 positions")
            counts.append(max(0, end - self.tokens[seq]))
        return self._reserve_counts(seqs, counts)

    def _reserve_counts(self, seqs, counts):
        seqs = tuple(seqs)
        if len(set(seqs)) != len(seqs):
            raise ValueError("a reservation batch must contain each sequence once")
        growth = []
        for seq, tokens in zip(seqs, counts):
            self.row(seq)                              # bounds before indexing tokens
            total = self.tokens[seq] + tokens
            if total >= 2**31:
                raise ValueError(f"seq {seq} token count exceeds int32")
            have, need = self.blocks_for(self.tokens[seq]), self.blocks_for(total)
            if need > self.max_blocks_per_seq:
                raise MemoryError(f"seq {seq} would exceed {self.max_blocks_per_seq} blocks")
            growth.append((seq, have, need, tokens))
        grow = sum(need - have for _, have, need, _ in growth)
        if grow > len(self.free) and self.reclaim is not None:
            self.reclaim(grow - len(self.free))            # the cache gives back least recently used boundaries
        if grow > len(self.free):
            raise MemoryError(
                f"batch needs {grow} more blocks, {len(self.free)} free: the "
                "scheduler admitted more than the budget declared")
        for seq, have, need, tokens in growth:
            base = seq * self.max_blocks_per_seq
            for i in range(have, need):
                block = self.free.pop()
                self._table[base + i] = block
                self.refs[block] = 1
            if self.tokens[seq] == 0 and tokens:
                self.rows_in_use += 1
            self.tokens[seq] += tokens
        return grow

    def adopt(self, seq: int, blocks, tokens: int) -> None:
        """Seat a cached prefix -- `tokens` whole blocks that are complete and shared -- at the head of
        an empty row; the row's own reservation continues after them."""
        blocks = tuple(blocks)
        self.row(seq)
        if self.tokens[seq] or self.row(seq)[0] != EMPTY:
            raise ValueError(f"seq {seq} already holds blocks; a prefix is adopted into an empty row")
        if type(tokens) is not int or tokens <= 0 or tokens % self.block_size or len(blocks) != tokens // self.block_size:
            raise ValueError("an adopted prefix is whole blocks with exactly their token count")
        if len(blocks) > self.max_blocks_per_seq or any(not 0 <= b < self.num_blocks or self.refs[b] <= 0 for b in blocks):
            raise ValueError("adopted blocks must be live cached blocks that fit the row")
        base = seq * self.max_blocks_per_seq
        for i, block in enumerate(blocks):
            self._table[base + i] = block
            self.refs[block] += 1
        self.tokens[seq] = tokens
        self.rows_in_use += 1

    def pin(self, blocks) -> None:
        """One more owner for each block (a cache entry)."""
        for b in blocks:
            if not 0 <= b < self.num_blocks or self.refs[b] <= 0:
                raise ValueError("only a live block can be pinned")
        for b in blocks:
            self.refs[b] += 1

    def unpin(self, blocks) -> int:
        """Drop the cache's ownership; blocks nobody else holds go back to the free stack. Returns how many did."""
        freed = 0
        for b in blocks:
            if self.refs[b] <= 0:
                raise ValueError("unpin of a block with no owner")
            self.refs[b] -= 1
            if self.refs[b] == 0:
                self.free.append(b)
                freed += 1
        return freed

    def release(self, seq: int) -> int:
        """Give every block of `seq` back (to the free stack when the row was its last owner). Returns how many."""
        row = self.row(seq)
        base = seq * self.max_blocks_per_seq
        n = 0
        for i in range(self.max_blocks_per_seq):
            if row[i] == EMPTY:
                break
            block = row[i]
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.append(block)
            self._table[base + i] = EMPTY
            n += 1
        if n:
            self._epochs[seq] += 1
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
        if not isinstance(seq, int) or not 0 <= seq < 2**31:
            raise ValueError("slot owner must be a nonnegative int32 sequence id")
        if seq in self.owner:
            raise ValueError(f"seq {seq} already owns a state slot")
        if not self.free:
            raise MemoryError("no state slot left: more live sequences than the budget declared")
        slot = self.free.pop()
        self.owner[slot] = seq
        return slot

    def give(self, slot: int) -> None:
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} outside {self.num_slots}")
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
