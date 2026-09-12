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
int32 linked list. No Python object graph sits between the scheduler and the
launch.

Running out is an error, not a fallback (D3): the budget declared how many
blocks exist before the model loaded, so exhaustion means the scheduler broke
its contract, and hiding that behind an eviction would hide the bug.

Blocks are owned by rows and remembered by boundaries. A row references the
blocks in its table and a block is free when no row does -- but free is not
blank. The prefix cache (base/prefix.py) CLAIMS the blocks of a boundary it
keeps: a claimed block waits in the free list under that boundary's name, and
the next prompt that begins the same way takes it back without a copy and
without anyone having pinned it. A claim costs nothing and blocks nobody; the
block leaves the moment a row needs one, and the boundary stops being one
then, not before.

Free blocks leave in three grades, and never out of order:

  anonymous   no boundary remembers it                  -- handed out first
  faded       a boundary that lost its snapshot claims  -- only its state can
              still come back (from the prefix tier), so the blocks are worth
              less than a whole boundary's and more than nobody's
  cached      a boundary that can be adopted right now
  pinned      a boundary an operator warmed by hand     -- handed out last of
              all, which is the whole promise of pinning one

Within a grade the front leaves first, and a row gives its blocks back TAIL
FIRST, so the head of a prompt -- the part the next prompt is most likely to
share -- is the last thing anybody takes.

`pin` remains for the one owner that is not a row: bytes being read off the
device (a prefix-tier write) must not move while the read is in flight.
"""
from __future__ import annotations

from array import array

EMPTY = -1
HELD = -1                                   # in no free list: a row or a transfer holds it
ANON, FADED, CACHED, PINNED = 0, 1, 2, 3    # the grades of a free block, taken in this order


class BlockPool:
    """Fixed-size blocks handed out from a graded free list; a sequence owns a row."""

    ANON, FADED, CACHED, PINNED = ANON, FADED, CACHED, PINNED

    def __init__(self, num_blocks: int, block_size: int, max_seqs: int, max_blocks_per_seq: int):
        if any(not isinstance(n, int) or n <= 0
               for n in (num_blocks, block_size, max_seqs, max_blocks_per_seq)):
            raise ValueError("block count, block size, row count and row width must be positive integers")
        self.block_size = block_size
        self.num_blocks = num_blocks
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
        self.refs = array("i", [0]) * num_blocks           # owners per block: rows, plus a transfer's pin
        self.claims = array("i", [0]) * num_blocks         # cached boundaries that remember it (blocks and snapshot)
        self.fades = array("i", [0]) * num_blocks          # boundaries that remember only its blocks
        self.pins = array("i", [0]) * num_blocks           # boundaries an operator pinned
        self.forget = None                                 # cache callback: this block is leaving, drop every boundary on it
        # the free list: one doubly linked chain per grade, threaded through the block ids themselves
        self._next = array("i", [EMPTY]) * num_blocks
        self._prev = array("i", [EMPTY]) * num_blocks
        self._grade = array("b", [ANON]) * num_blocks
        self._head, self._tail, self._sizes = [EMPTY] * 4, [EMPTY] * 4, [0, 0, 0, 0]
        for block in range(num_blocks):
            self._grade[block] = HELD
            self._append(block, ANON)

    # -- the graded free list ---------------------------------------------------------------
    def _append(self, block: int, grade: int) -> None:
        """`block` joins the back of `grade`; the front is what leaves first."""
        self._prev[block], self._next[block] = self._tail[grade], EMPTY
        if self._tail[grade] == EMPTY:
            self._head[grade] = block
        else:
            self._next[self._tail[grade]] = block
        self._tail[grade] = block
        self._grade[block] = grade
        self._sizes[grade] += 1

    def _unlink(self, block: int) -> None:
        grade = self._grade[block]
        if grade == HELD:
            return
        before, after = self._prev[block], self._next[block]
        if before == EMPTY:
            self._head[grade] = after
        else:
            self._next[before] = after
        if after == EMPTY:
            self._tail[grade] = before
        else:
            self._prev[after] = before
        self._prev[block] = self._next[block] = EMPTY
        self._grade[block] = HELD
        self._sizes[grade] -= 1

    def _regrade(self, block: int) -> None:
        """Put `block` where its owners say it belongs, at the back of that grade."""
        if self.refs[block]:
            want = HELD
        elif self.pins[block]:
            want = PINNED
        elif self.claims[block]:
            want = CACHED
        elif self.fades[block]:
            want = FADED
        else:
            want = ANON
        if self._grade[block] == want:
            return
        self._unlink(block)
        if want != HELD:
            self._append(block, want)

    def _take(self) -> int:
        """The next block to hand out: anonymous first, a whole boundary's last."""
        for grade in (ANON, FADED, CACHED, PINNED):
            block = self._head[grade]
            if block == EMPTY:
                continue
            if grade != ANON:
                self.forget(block)                         # every boundary on it stops being one; the block turns anonymous
            self._unlink(block)
            self.refs[block] = 1
            return block
        raise MemoryError("no block is free")

    @property
    def available(self) -> int:
        """Blocks no row holds: free now, counting the ones a boundary would give up."""
        return self._sizes[ANON] + self._sizes[FADED] + self._sizes[CACHED] + self._sizes[PINNED]

    @property
    def anonymous(self) -> int:
        """Free blocks nothing remembers: what a reservation spends before any boundary pays."""
        return self._sizes[ANON]

    @property
    def cached(self) -> int:
        """Free blocks a cached boundary still holds -- the reuse a reservation spends after everything anonymous."""
        return self._sizes[CACHED] + self._sizes[PINNED]

    @property
    def faded(self) -> int:
        """Free blocks held by a boundary whose snapshot is gone (its state lives on the prefix tier)."""
        return self._sizes[FADED]

    def _counts(self, grade: int):
        return {FADED: self.fades, CACHED: self.claims, PINNED: self.pins}[grade]

    def claim(self, blocks, grade: int) -> None:
        """A boundary remembers these blocks: they stay where they are, behind everything of a lower grade."""
        if self.forget is None:
            raise ValueError("a pool only takes claims from a bound cache: nothing would answer for the blocks")
        if grade not in (FADED, CACHED, PINNED):
            raise ValueError("a boundary claims blocks as pinned, cached or faded")
        counts = self._counts(grade)
        for block in blocks:
            if not 0 <= block < self.num_blocks:
                raise ValueError("a claim names a block of this pool")
        for block in blocks:
            counts[block] += 1
            self._regrade(block)

    def disclaim(self, blocks, grade: int) -> None:
        """That boundary is gone (or changed grade): the blocks fall back to what is left holding them."""
        counts = self._counts(grade)
        for block in blocks:
            if counts[block] <= 0:
                raise ValueError("disclaim of a block that boundary never claimed")
        for block in blocks:
            counts[block] -= 1
            self._regrade(block)

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
        list. A decoder never sees a batch whose earlier rows were reserved
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
        if grow > self.available:
            raise MemoryError(
                f"batch needs {grow} more blocks, {self.available} free: the "
                "scheduler admitted more than the budget declared")
        for seq, have, need, tokens in growth:
            base = seq * self.max_blocks_per_seq
            for i in range(have, need):
                self._table[base + i] = self._take()   # anonymous blocks first; a boundary pays only when they run out
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
        if len(blocks) > self.max_blocks_per_seq or any(
                not 0 <= b < self.num_blocks
                or not (self.claims[b] or self.fades[b] or self.pins[b] or self.refs[b]) for b in blocks):
            raise ValueError("adopted blocks must be blocks a boundary still holds")
        base = seq * self.max_blocks_per_seq
        for i, block in enumerate(blocks):
            self._table[base + i] = block
            self.refs[block] += 1
            self._regrade(block)                       # a row holds it now: out of the free list (vLLM's touch)
        self.tokens[seq] = tokens
        self.rows_in_use += 1

    def pin(self, blocks) -> None:
        """One more owner for each block, for an owner that is not a row: bytes a transfer is reading."""
        for b in blocks:
            if not 0 <= b < self.num_blocks:
                raise ValueError("only a block of this pool can be pinned")
        for b in blocks:
            self.refs[b] += 1
            self._regrade(b)

    def unpin(self, blocks) -> int:
        """Drop that owner; blocks nobody else holds go back to the free list. Returns how many did."""
        freed = 0
        for b in blocks:
            if self.refs[b] <= 0:
                raise ValueError("unpin of a block with no owner")
            self.refs[b] -= 1
            if self.refs[b] == 0:
                freed += 1
            self._regrade(b)
        return freed

    def release(self, seq: int) -> int:
        """Give every block of `seq` back, TAIL FIRST -- the head of a prompt is what the next prompt shares,
        so it must be the last block anyone takes. Returns how many blocks the row held."""
        row = self.row(seq)
        base = seq * self.max_blocks_per_seq
        n = 0
        while n < self.max_blocks_per_seq and row[n] != EMPTY:
            n += 1
        for i in range(n - 1, -1, -1):
            block = row[i]
            self.refs[block] -= 1
            self._table[base + i] = EMPTY
            self._regrade(block)
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
    row0 = list(pool.row(0))[:3]
    assert pool.release(0) == 3 and pool.available == 10 and pool.rows_in_use == 0
    # tail first: a row gives its last block back first, so its head is the last thing taken
    tail = BlockPool(3, 16, 2, 3)
    tail.reserve(0, 16 * 3)
    row = list(tail.row(0))[:3]
    tail.release(0)
    tail.reserve(1, 16 * 3)
    assert list(tail.row(1))[:3] == row[::-1], "the tail of a released row leaves before its head"
    # grades: a claimed block waits behind every anonymous one, and pays one at a time
    graded = BlockPool(10, 16, 3, 10)
    graded.forget = lambda b: graded.disclaim((b,), CACHED)
    graded.reserve(0, 16 * 10)
    kept = tuple(graded.row(0))[:2]
    graded.claim(kept, CACHED)
    graded.release(0)
    assert graded.available == 10 and graded.anonymous == 8 and graded.cached == 2
    assert graded.reserve(1, 16 * 8) == 8 and graded.cached == 2, "eight anonymous blocks came first"
    assert graded.reserve(2, 16) == 1 and graded.cached == 1, "then one boundary's block, not the rest"
    assert graded.row(2)[0] == kept[1], "and the tail of the boundary before its head"
    # storage: attach a fake arena and read a sequence's blocks back as views
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
