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

    def lookup(self, ids, salts=()):
        """(tokens, entry, hash) of the longest cached boundary strictly inside the prompt, or (0, None, None)."""
        chain = self.chain(ids, salts)
        for tokens in sorted(chain, reverse=True):
            if tokens < len(ids) and chain[tokens] in self.entries:
                entry = self.entries[chain[tokens]]
                self.tick += 1
                entry.used = self.tick
                self.hits += 1
                return tokens, entry, chain[tokens]
        self.misses += 1
        return 0, None, None

    def has(self, h: bytes) -> bool:
        return h in self.entries

    # -- snapshots and entries ----------------------------------------------------------------
    def take_snapshot(self) -> "int | None":
        """A free snapshot slot, evicting the least recently used entry for it if none is free."""
        if not self.free_snaps and self.entries:
            self._evict(min(self.entries, key=lambda h: self.entries[h].used))
        return self.free_snaps.pop() if self.free_snaps else None

    def give_snapshot(self, snap: int) -> None:
        self.free_snaps.append(snap)

    def insert(self, h: bytes, blocks, tokens: int, snap: int) -> None:
        if h in self.entries or tokens % self.block_size or len(blocks) != tokens // self.block_size:
            raise ValueError("a prefix entry is one whole-block boundary with exactly its blocks")
        self.tick += 1
        self.pool.pin(blocks)
        self.entries[h] = Entry(tuple(blocks), tokens, snap, self.tick)

    def _evict(self, h: bytes) -> int:
        entry = self.entries.pop(h)
        freed = self.pool.unpin(entry.blocks)
        self.free_snaps.append(entry.snap)
        self.evictions += 1
        return freed

    def reclaim(self, blocks: int) -> int:
        """Free at least `blocks` by evicting least recently used entries; returns how many were freed."""
        freed = 0
        while freed < blocks and self.entries:
            freed += self._evict(min(self.entries, key=lambda h: self.entries[h].used))
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
