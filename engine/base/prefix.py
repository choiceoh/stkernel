"""Prefix reuse: a prompt that begins the way an earlier one did is not prefilled again (base).

The unit is a chunk -- the prefill chunk the contract already uses, a multiple of the block size --
because a linear-attention model's state exists only where a prefill stopped: at every chunk boundary
the runner asks the model for a checkpoint (its position rings at that boundary, a fixed-size snapshot
in the arena) and pins the blocks before it. A later prompt is hashed chunk by chunk along the same
chain; the longest cached boundary below its own length gives it those blocks (adopted, read-only:
every position in them is already written, and nothing writes there again) and its state (restored
into its own slot), and it prefills from there. Only whole chunks are shared: the boundary is never
the prompt's end, so at least one token is always computed and the first token's logits are real.

Every rank runs the same admissions in the same order, so the cache's state is the same on every
rank without a message (the hash is of token ids; the clock is a counter). Ownership is by count in
the block pool: a row references its blocks, an entry pins its blocks; a block is free when nobody
does. Eviction is least recently used, and it happens when a reservation needs blocks the free stack
does not have (the pool calls back), or when the snapshots run out; an evicted entry's blocks stay
alive for the rows still using them.
"""
from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass


@dataclass
class Entry:
    blocks: tuple
    tokens: int
    snap: int
    used: int
    hits: int = 0                   # adoptions: a boundary that served once is worth more than one nobody asked for
    pinned: bool = False            # an operator's warm prompt: leaves only when nothing else can (45차 §23 C)
    spilled: bool = False           # a copy is on the prefix tier: dropping it from memory loses nothing (A)
    spilling: bool = False          # the copy is being written: the snapshot and blocks must stay as they are
    spill_failed: bool = False      # the tier refused it for good (not room: an error)
    children: int = 0               # cached boundaries one block longer that extend this one: zero means a leaf


class PrefixCache:
    def __init__(self, block_size: int, chunk: int, snapshots: int):
        if any(type(n) is not int or n <= 0 for n in (block_size, chunk, snapshots)) or chunk % block_size:
            raise ValueError("prefix cache needs a positive block size, a chunk that is whole blocks, and snapshots")
        self.block_size, self.chunk, self.snapshots = block_size, chunk, snapshots
        self.entries: "dict[bytes, Entry]" = {}
        self.free_snaps = list(range(snapshots - 1, -1, -1))
        self.tick = 0
        self.hits = self.misses = self.evictions = 0
        self.pool = None
        self.tier_keys: "dict[bytes, int]" = {}   # boundaries whose blocks + snapshot the prefix tier holds (base/runner spills them)
        self._by_blocks: "dict[tuple, bytes]" = {}   # block list -> hash: an entry's parent is the one a block shorter
        self.version = 0                          # bumped by every insert/evict/pin: what changed since someone last looked

    def bind(self, pool) -> None:
        """The pool this cache pins blocks in; the pool reclaims through it when its free stack runs short."""
        self.pool = pool
        pool.reclaim = self.reclaim
        pool.reclaimable = self.reclaimable

    # -- the chain ---------------------------------------------------------------------------
    def chain(self, ids, salts=()) -> "dict[int, bytes]":
        """boundary tokens -> hash of the prompt up to there, for every whole BLOCK (45차 §23: the pool's block, as
        production's APC -- a prefill chunk is three of them, and the two boundaries inside a chunk are taken during the
        chunk's step, base/runner). `salts`: (position, bytes) pairs folded into the block that holds the position -- the
        digests of the media whose rows stand at those placeholder ids (the same <|image|> run, another picture, must
        never share a boundary: A7)."""
        out, h = {}, b""
        salted = sorted((int(p), bytes(d)) for p, d in salts)
        j = 0
        unit = self.block_size
        for end in range(unit, len(ids) + 1, unit):
            h = hashlib.sha1(h + array("i", ids[end - unit:end]).tobytes()).digest()
            while j < len(salted) and salted[j][0] < end:
                h = hashlib.sha1(h + salted[j][1]).digest()
                j += 1
            out[end] = h
        return out

    def extend_chain(self, chain: dict, ids, salts=(), start: int = 0) -> dict:
        """`chain` continued over `ids[start:]` -- the tokens after the last boundary it knows. `ids` may be the whole
        history or just its tail from `start`; only the new blocks are hashed (a 128K conversation crossing a boundary
        while generating must not re-hash 170 blocks of it)."""
        out = dict(chain)
        last = max(out) if out else 0
        if start > last:
            raise ValueError("the tail must begin at or before the chain's last boundary")
        h = out.get(last, b"")
        unit = self.block_size
        salted = sorted((int(p), bytes(d)) for p, d in salts if int(p) >= last)
        j = 0
        end = last + unit
        while end - start <= len(ids):
            h = hashlib.sha1(h + array("i", ids[end - unit - start:end - start]).tobytes()).digest()
            while j < len(salted) and salted[j][0] < end:
                h = hashlib.sha1(h + salted[j][1]).digest()
                j += 1
            out[end] = h
            end += unit
        return out

    def peek_chain(self, chain: dict, n: int) -> int:
        return max((t for t in chain if t < n and chain[t] in self.entries), default=0)

    def lookup_chain(self, chain: dict, n: int):
        """`lookup` over a chain computed once by the caller."""
        for tokens in sorted(chain, reverse=True):
            if tokens < n and chain[tokens] in self.entries:
                entry = self.entries[chain[tokens]]
                self.tick += 1
                entry.used = self.tick
                entry.hits += 1
                self.hits += 1
                return tokens, entry, chain[tokens]
        self.misses += 1
        return 0, None, None

    def tier_lookup_chain(self, chain: dict, n: int, above: int = 0) -> "tuple[int, bytes] | None":
        for tokens in sorted(chain, reverse=True):
            if tokens <= above or tokens >= n:
                continue
            h = chain[tokens]
            if h in self.tier_keys and h not in self.entries:
                return tokens, h
        return None

    def lookup(self, ids, salts=()):
        """(tokens, entry, hash) of the longest cached boundary strictly inside the prompt, or (0, None, None)."""
        chain = self.chain(ids, salts)
        for tokens in sorted(chain, reverse=True):
            if tokens < len(ids) and chain[tokens] in self.entries:
                entry = self.entries[chain[tokens]]
                self.tick += 1
                entry.used = self.tick
                entry.hits += 1
                self.hits += 1
                return tokens, entry, chain[tokens]
        self.misses += 1
        return 0, None, None

    def has(self, h: bytes) -> bool:
        return h in self.entries

    def peek(self, ids, salts=()) -> int:
        """The tokens `lookup` would reuse, without counting a query (admission asks before it commits)."""
        chain = self.chain(ids, salts)
        return max((t for t in chain if t < len(ids) and chain[t] in self.entries), default=0)

    def tier_lookup(self, ids, salts=(), above: int = 0) -> "tuple[int, bytes] | None":
        """(tokens, hash) of the longest boundary strictly inside the prompt that the prefix TIER holds and memory
        does not, longer than `above` (what memory can already give): a restore candidate, or None."""
        chain = self.chain(ids, salts)
        for tokens in sorted(chain, reverse=True):
            if tokens <= above or tokens >= len(ids):
                continue
            h = chain[tokens]
            if h in self.tier_keys and h not in self.entries:
                return tokens, h
        return None

    def is_leaf(self, h: bytes) -> bool:
        """No cached boundary one block longer extends this one. Only leaves go to the tier -- a restored leaf brings
        every block of its chain back, an inner boundary would bring the same ones. Kept as a count at insert/evict:
        the runner asks every step, and a scan of 96 entries' block lists per step is a decode-step's worth of host time."""
        return self.entries[h].children == 0

    def spill_candidates(self, count: int) -> "list[bytes]":
        """Up to `count` leaves in eviction order that have no copy on the tier yet: what to write ahead of need."""
        order = sorted(self.entries, key=lambda h: (self.entries[h].pinned, self.entries[h].hits > 0, self.entries[h].used))
        out = []
        for h in order:
            e = self.entries[h]
            if e.spilled or e.spilling or e.spill_failed or not self.is_leaf(h):
                continue
            out.append(h)
            if len(out) >= count:
                break
        return out

    def pin(self, hashes) -> int:
        n = 0
        for h in hashes:
            if h in self.entries and not self.entries[h].pinned:
                self.entries[h].pinned = True
                n += 1
        self.version += n > 0
        return n

    def unpin_all(self) -> int:
        n = 0
        for e in self.entries.values():
            if e.pinned:
                e.pinned = False
                n += 1
        self.version += n > 0
        return n

    # -- snapshots and entries ----------------------------------------------------------------
    def _victim(self) -> "bytes | None":
        """Who leaves when room is needed: among the boundaries nobody adopted yet, the oldest; only when every
        boundary has served, the least recently used -- so one long prompt's forty fresh boundaries cannot flush the
        system prompt every conversation shares (45차 §23: the cache is tolerant of churn, not just of size).
        Within a class, one whose copy is already on the tier goes first (nothing is lost); a pinned boundary goes
        last of all; one whose copy is being written cannot go at all (the write reads its snapshot)."""
        movable = [h for h, e in self.entries.items() if not e.spilling]
        if not movable:
            return None
        return min(movable, key=lambda h: (self.entries[h].pinned, self.entries[h].hits > 0, not self.entries[h].spilled,
                                           self.entries[h].used))

    def take_snapshot(self) -> "int | None":
        """A free snapshot slot, evicting a boundary for it if none is free (`_victim`)."""
        if not self.free_snaps and self.entries:
            victim = self._victim()
            if victim is not None:
                self._evict(victim)
        return self.free_snaps.pop() if self.free_snaps else None

    def give_snapshot(self, snap: int) -> None:
        self.free_snaps.append(snap)

    def insert(self, h: bytes, blocks, tokens: int, snap: int) -> None:
        if h in self.entries or tokens % self.block_size or len(blocks) != tokens // self.block_size:
            raise ValueError("a prefix entry is one whole-block boundary with exactly its blocks")
        self.tick += 1
        self.version += 1
        self.pool.pin(blocks)
        blocks = tuple(blocks)
        self.entries[h] = Entry(blocks, tokens, snap, self.tick)
        self._by_blocks[blocks] = h
        parent = self._by_blocks.get(blocks[:-1])
        if parent is not None:
            self.entries[parent].children += 1
        for longer, hh in self._by_blocks.items():                      # a child inserted before its parent (a tier restore) counts too
            if len(longer) == len(blocks) + 1 and longer[:-1] == blocks:
                self.entries[h].children += 1

    def _evict(self, h: bytes) -> int:
        entry = self.entries.pop(h)
        self.version += 1
        self._by_blocks.pop(entry.blocks, None)
        parent = self._by_blocks.get(entry.blocks[:-1])
        if parent is not None and self.entries[parent].children > 0:
            self.entries[parent].children -= 1
        freed = self.pool.unpin(entry.blocks)
        self.free_snaps.append(entry.snap)
        self.evictions += 1
        return freed

    def reclaim(self, blocks: int) -> int:
        """Free at least `blocks` by evicting least recently used entries; returns how many were freed."""
        freed = 0
        while freed < blocks and self.entries:
            victim = self._victim()
            if victim is None:
                break
            freed += self._evict(victim)
        return freed

    def reclaimable(self) -> int:
        """Blocks held only by the cache (by one entry or several nested boundaries): what reclaim() could
        free if it evicted everything."""
        pins = {}
        for e in self.entries.values():
            for b in e.blocks:
                pins[b] = pins.get(b, 0) + 1
        return sum(1 for b, n in pins.items() if self.pool.refs[b] == n)

    def clear(self) -> None:
        for h in list(self.entries):
            self._evict(h)
