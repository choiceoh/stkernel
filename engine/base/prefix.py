"""Prefix reuse: a prompt that begins the way an earlier one did is not prefilled again (base).

The unit is a chunk -- the prefill chunk the contract already uses, a multiple of the block size --
because a linear-attention model's state exists only where a prefill stopped: at every chunk boundary
the runner asks the model for a checkpoint (its position rings at that boundary, a fixed-size snapshot
in the arena) and claims the blocks before it. A later prompt is hashed chunk by chunk along the same
chain; the longest cached boundary below its own length gives it those blocks (adopted, read-only:
every position in them is already written, and nothing writes there again) and its state (restored
into its own slot), and it prefills from there. Only whole chunks are shared: the boundary is never
the prompt's end, so at least one token is always computed and the first token's logits are real.

Every rank runs the same admissions in the same order, so the cache's state is the same on every
rank without a message (the hash is of token ids; the clock is a counter). Nothing is pinned: a
boundary CLAIMS its blocks in the pool (base/kv.py) and a claimed block waits in the free list under
its name. A reservation spends anonymous blocks first and a boundary's last, and the boundary stops
being one exactly when a block of it is handed out -- not a moment earlier, so a prompt that comes
back while its blocks are still there pays nothing at all.

The scarce thing here is not the block, it is the SNAPSHOT: 96 of them against thousands of blocks
(profiles/glm53/boot.PREFIX_SNAPSHOTS), and a snapshot is only ever freed because something else
wants it that instant. So the two resources part ways, and a boundary has three lives, not two:

  entry   blocks and snapshot in memory       -- adopt and restore, free
  faded   blocks in memory, snapshot given    -- the blocks were never handed out, so a prompt that
          away to a newer boundary               wants this boundary back reads the 77 MiB snapshot
                                                 off the prefix tier and NOT the KV under it
  gone    a block of it was handed out        -- the tier's copy whole, or a prefill

A boundary can only fade if the tier has its state; with no tier, giving up the snapshot is the end
of it, the way it always was.

`salts` are folded into the block that holds their position, so a media digest separates every
boundary from the picture onward, and a TENANT SALT at position 0 (vLLM's `cache_salt`, which it
also puts on the first block only) separates one tenant's boundaries from another's for the whole
prompt. Ranks agree because the hash is of bytes every rank has; there is no random root.
"""
from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass

from engine.base.kv import CACHED, FADED, PINNED

NO_SNAPSHOT = -1


TENANT_TAG = b"tenant:"


def is_tenant_salt(salt) -> bool:
    """Whether a (position, bytes) salt is a tenant's, not a picture's. The runner rebuilds a row's media salts from
    the model every time it extends a chain; the tenant's is not the model's to remember, so it is carried across."""
    return salt[0] == 0 and bytes(salt[1]).startswith(TENANT_TAG)


def tenant_salt(tenant) -> "tuple[int, bytes]":
    """The salt entry that separates one tenant's boundaries from every other tenant's.

    Position 0, so it is folded into the first block and inherited by every boundary after it --
    the same place vLLM puts `cache_salt` (`kv_cache_utils.py`: extra keys on the first block only).
    The tag keeps it from ever colliding with a media digest standing at position 0."""
    raw = tenant.encode() if isinstance(tenant, str) else bytes(tenant)
    return (0, TENANT_TAG + raw)


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
        self.faded: "dict[bytes, Entry]" = {}     # boundaries that kept their blocks after giving up their snapshot, oldest first
        self.max_faded = 4 * snapshots            # holding blocks last in line delays nobody, but it is still not unbounded
        self.free_snaps = list(range(snapshots - 1, -1, -1))
        self.tick = 0
        self.hits = self.misses = self.evictions = 0
        self.fades = 0                            # boundaries that gave a snapshot away and kept their blocks
        self.pool = None
        self.tier_keys: "dict[bytes, int]" = {}   # boundaries whose blocks + snapshot the prefix tier holds (base/runner spills them)
        self.tier_owner: "dict[int, bytes]" = {}  # and the inverse, which is what makes the tier's short key safe (below)
        self._by_blocks: "dict[tuple, bytes]" = {}   # block list -> hash: an entry's parent is the one a block shorter
        self._on_block: "dict[int, list]" = {}    # block -> the boundaries that hold it, in the order they took it: who stops
        # being one when it leaves. A LIST, not a set: `bytes` hash randomly per process, so a set would be walked in a
        # different order on every rank, and the order decides which snapshot slot comes free first.
        self.version = 0                          # bumped by every insert/evict/pin: what changed since someone last looked

    def bind(self, pool) -> None:
        """The pool this cache claims blocks in; the pool tells it when a claimed block has to leave."""
        self.pool = pool
        pool.forget = self.forget_block

    # -- the chain ---------------------------------------------------------------------------
    def chain(self, ids, salts=()) -> "dict[int, bytes]":
        """boundary tokens -> hash of the prompt up to there, for every whole BLOCK (45차 §23: the pool's block, as
        production's APC -- a prefill chunk is three of them, and the two boundaries inside a chunk are taken during the
        chunk's step, base/runner). `salts`: (position, bytes) pairs folded into the block that holds the position -- the
        digests of the media whose rows stand at those placeholder ids (the same <|image|> run, another picture, must
        never share a boundary: A7), and at position 0 the tenant salt (`tenant_salt`)."""
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
        while generating must not re-hash 170 blocks of it). A salt before the last boundary is already inside it."""
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
        return max((t for t in chain if t < n and self.has(chain[t])), default=0)

    def lookup_chain(self, chain: dict, n: int):
        """`lookup` over a chain computed once by the caller."""
        for tokens in sorted(chain, reverse=True):
            if tokens >= n:
                continue
            entry = self.entries.get(chain[tokens])
            if entry is not None:
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
            if h in self.tier_keys and not self.has(h):
                return tokens, h
        return None

    def lookup(self, ids, salts=()):
        """(tokens, entry, hash) of the longest cached boundary strictly inside the prompt, or (0, None, None)."""
        return self.lookup_chain(self.chain(ids, salts), len(ids))

    def has(self, h: bytes) -> bool:
        """Memory can serve this boundary right now. A faded one cannot: its state has to be read back first."""
        return h in self.entries

    def blocks_of(self, h: bytes) -> "tuple | None":
        """The blocks of a boundary memory still holds, entry or faded: where a tier read may land instead of fresh ones."""
        entry = self.entries.get(h) or self.faded.get(h)
        return None if entry is None else entry.blocks

    def peek(self, ids, salts=()) -> int:
        """The tokens `lookup` would reuse, without counting a query (admission asks before it commits)."""
        return self.peek_chain(self.chain(ids, salts), len(ids))

    def tier_lookup(self, ids, salts=(), above: int = 0) -> "tuple[int, bytes] | None":
        """(tokens, hash) of the longest boundary strictly inside the prompt that the prefix TIER holds and memory
        cannot serve, longer than `above` (what memory can already give): a restore candidate, or None."""
        return self.tier_lookup_chain(self.chain(ids, salts), len(ids), above)

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

    def spill_begin(self, h: bytes) -> None:
        """A tier write is about to read this boundary's blocks off the device: nobody may take them until it lands.
        This is the one pin left in the engine, and it is an owner the way a row is -- `available` says so."""
        e = self.entries[h]
        if not e.spilling:
            e.spilling = True
            self.pool.pin(e.blocks)

    def spill_end(self, h: bytes, *, spilled: bool = False, failed: bool = False) -> None:
        e = self.entries.get(h) or self.faded.get(h)
        if e is None or not e.spilling:
            return
        e.spilling = False
        self.pool.unpin(e.blocks)
        if spilled:
            e.spilled = True
        if failed:
            e.spill_failed = True

    def pin(self, hashes) -> int:
        """An operator's warm prompt: these boundaries go last, of every kind of pressure. Their blocks move to the
        grade behind every other free block, so a reservation reaches them only when there is nothing else at all."""
        n = 0
        for h in hashes:
            if h in self.entries and not self.entries[h].pinned:
                e = self.entries[h]
                e.pinned = True
                self.pool.disclaim(e.blocks, CACHED)
                self.pool.claim(e.blocks, PINNED)
                n += 1
        self.version += n > 0
        return n

    def unpin_all(self) -> int:
        n = 0
        for e in self.entries.values():
            if e.pinned:
                e.pinned = False
                self.pool.disclaim(e.blocks, PINNED)
                self.pool.claim(e.blocks, CACHED)
                n += 1
        self.version += n > 0
        return n

    # -- snapshots and entries ----------------------------------------------------------------
    def _victim(self) -> "bytes | None":
        """Who gives up its snapshot when one is needed: among the boundaries nobody adopted yet, the oldest; only when
        every boundary has served, the least recently used -- so one long prompt's forty fresh boundaries cannot flush
        the system prompt every conversation shares (45차 §23: the cache is tolerant of churn, not just of size).
        Within a class, one whose copy is already on the tier goes first (its state can come back); a pinned boundary
        goes last of all; one whose copy is being written cannot go at all (the write reads its snapshot)."""
        movable = [h for h, e in self.entries.items() if not e.spilling]
        if not movable:
            return None
        return min(movable, key=lambda h: (self.entries[h].pinned, self.entries[h].hits > 0, not self.entries[h].spilled,
                                           self.entries[h].used))

    def take_snapshot(self) -> "int | None":
        """A free snapshot slot, taking one from a boundary if none is free (`_victim`). The boundary keeps its blocks
        when the tier has its state (`_fade`) and is gone when it does not."""
        if not self.free_snaps and self.entries:
            victim = self._victim()
            if victim is not None:
                self._fade(victim)
        return self.free_snaps.pop() if self.free_snaps else None

    def give_snapshot(self, snap: int) -> None:
        """A slot whose contents belong to nobody (an aborted checkpoint, a boundary that is gone)."""
        self.free_snaps.append(snap)

    def insert(self, h: bytes, blocks, tokens: int, snap: int) -> None:
        if h in self.entries or tokens % self.block_size or len(blocks) != tokens // self.block_size:
            raise ValueError("a prefix entry is one whole-block boundary with exactly its blocks")
        if h in self.faded:
            self._forget(h)                             # the same boundary, computed again: the fresh one is whole
        self.tick += 1
        self.version += 1
        blocks = tuple(blocks)
        self.pool.claim(blocks, CACHED)
        self.entries[h] = Entry(blocks, tokens, snap, self.tick)
        self._index(h, blocks)

    def _watch(self, h: bytes, blocks: tuple) -> None:
        """`h` holds these blocks: it stops being a boundary when one of them is handed out."""
        for b in blocks:
            holders = self._on_block.setdefault(b, [])
            if h not in holders:
                holders.append(h)

    def _unwatch(self, h: bytes, blocks: tuple) -> None:
        for b in blocks:
            holders = self._on_block.get(b)
            if holders is not None and h in holders:
                holders.remove(h)
                if not holders:
                    self._on_block.pop(b, None)

    def _index(self, h: bytes, blocks: tuple) -> None:
        """An ENTRY joins the chain index too: who extends whom, so `is_leaf` costs nothing to ask."""
        self._watch(h, blocks)
        self._by_blocks[blocks] = h
        parent = self._by_blocks.get(blocks[:-1]) if blocks else None
        if parent is not None and parent in self.entries:
            self.entries[parent].children += 1
        for longer, hh in self._by_blocks.items():                      # a child indexed before its parent (a tier restore) counts too
            if len(longer) == len(blocks) + 1 and longer[:-1] == blocks:
                self.entries[h].children += 1

    def _unindex(self, h: bytes, blocks: tuple) -> None:
        self._unwatch(h, blocks)
        if self._by_blocks.get(blocks) == h:
            self._by_blocks.pop(blocks)
        parent = self._by_blocks.get(blocks[:-1]) if blocks else None
        if parent is not None and parent in self.entries and self.entries[parent].children > 0:
            self.entries[parent].children -= 1

    def _leave(self, h: bytes) -> Entry:
        """Out of the entry table: the blocks are no longer a whole boundary's."""
        entry = self.entries.pop(h)
        if entry.spilling:
            entry.spilling = False
            self.pool.unpin(entry.blocks)
        self._unindex(h, entry.blocks)
        self.pool.disclaim(entry.blocks, PINNED if entry.pinned else CACHED)
        self.evictions += 1
        self.version += 1
        return entry

    def _fade(self, h: bytes) -> None:
        """The boundary gives its snapshot to a newer one. If the tier holds its state it keeps its blocks -- named,
        last in line for a reservation -- so coming back costs the snapshot read alone and never the KV under it.
        With no copy anywhere, giving up the snapshot is the end of the boundary."""
        spilled = h in self.tier_keys
        entry = self._leave(h)
        entry.pinned = False                            # a boundary with no state is not what an operator warmed
        self.free_snaps.append(entry.snap)
        entry.snap = NO_SNAPSHOT
        if not spilled:
            return
        self.pool.claim(entry.blocks, FADED)
        self._watch(h, entry.blocks)                    # still the boundary on those blocks: it leaves when one of them does
        self.faded[h] = entry
        self.fades += 1
        while len(self.faded) > self.max_faded:
            oldest = next(iter(self.faded))
            if oldest == h:
                break
            self._forget(oldest)

    def _drop(self, h: bytes) -> None:
        """A block of it was handed out, so it is not a boundary any more; its slot holds nobody's state."""
        entry = self._leave(h)
        if entry.snap != NO_SNAPSHOT:
            self.free_snaps.append(entry.snap)

    def _forget(self, h: bytes) -> None:
        """A faded boundary stops being one: nothing is left to bring its state back to these blocks."""
        entry = self.faded.pop(h)
        if entry.spilling:
            entry.spilling = False
            self.pool.unpin(entry.blocks)
        self._unwatch(h, entry.blocks)
        self.pool.disclaim(entry.blocks, FADED)
        self.version += 1

    def drop(self, h: bytes) -> None:
        """Drop a boundary by name: a tier read that failed, a copy the tier threw away."""
        if h in self.entries:
            self._drop(h)
        elif h in self.faded:
            self._forget(h)

    def tier_holder(self, key: int) -> "bytes | None":
        """Which boundary the tier's slot `key` belongs to, or None.

        The tier indexes by an int -- the first seven bytes of a boundary's hash (base/runner.tier_key) -- so two
        boundaries could name the same slot. 56 bits puts that past 2^28 entries and the tier holds thousands, but the
        failure is handing one tenant the other's KV, quietly, which is exactly what the tenant salt exists to stop.
        So the slot has ONE owner: a spill onto a taken slot does not happen (base/runner), and a restore asks here
        first. Mooncake solves the same shape -- short physical keys for long logical ones -- with store-if-not-exists
        and a byte comparison on conflict; at our size one owner is enough (engine/MOONCAKE_COMPARISON_20260912.md).
        """
        return self.tier_owner.get(key)

    def hold_tier(self, h: bytes, key: int) -> None:
        """Record that the tier's slot `key` now holds boundary `h`. The caller has checked the slot was free."""
        held = self.tier_owner.get(key)
        if held is not None and held != h:
            raise ValueError(f"prefix tier slot {key} already holds {held.hex()[:8]}")
        self.tier_keys[h] = key
        self.tier_owner[key] = h

    def forget_tier(self, h: bytes) -> "int | None":
        """The tier no longer holds this boundary. Returns the key it had. A faded boundary whose slot is gone too
        has nothing left anywhere, so it stops holding its blocks."""
        key = self.tier_keys.pop(h, None)
        if key is not None and self.tier_owner.get(key) == h:
            del self.tier_owner[key]
        entry = self.entries.get(h)
        if entry is not None:
            entry.spilled = False                       # it is memory-only again, and leaves before one that is not
        elif h in self.faded:
            self._forget(h)                             # its state was the tier's copy: there is nothing to come back to
        return key

    def forget_block(self, block: int) -> None:
        """The pool is handing `block` out: every boundary that claimed it stops being one. This is the only eviction
        there is -- nothing leaves the cache until its blocks are actually needed (vLLM's free queue, per block)."""
        for h in list(self._on_block.get(block, ())):
            if h in self.entries:
                self._drop(h)
            elif h in self.faded:
                self._forget(h)

    def reclaimable(self) -> int:
        """Blocks a boundary holds and no row does: what a reservation would spend last."""
        return self.pool.cached + self.pool.faded

    def check(self) -> None:
        """The invariants a boundary's two resources have to keep, as SGLang's `mamba_radix_cache.sanity_check`
        keeps `full_lock_ref >= mamba_lock_ref`: a boundary that cannot name its blocks is not a boundary, and a
        faded one with its state nowhere is worse -- it holds blocks back for nothing. Called by the tests and by
        anyone diagnosing a pool; never on the step path."""
        for h, e in self.entries.items():
            counts = self.pool.pins if e.pinned else self.pool.claims
            if any(counts[b] <= 0 for b in e.blocks):
                raise AssertionError(f"entry {h.hex()[:8]} holds blocks it never claimed at its grade")
            if any(h not in self._on_block.get(b, ()) for b in e.blocks):
                raise AssertionError(f"entry {h.hex()[:8]} holds a block that would not tell it when it leaves")
        for h, e in self.faded.items():
            if h not in self.tier_keys:
                raise AssertionError(f"faded {h.hex()[:8]} has its state nowhere: it holds blocks for nothing")
            if any(self.pool.fades[b] <= 0 for b in e.blocks):
                raise AssertionError(f"faded {h.hex()[:8]} holds blocks it never claimed as faded")
            if any(h not in self._on_block.get(b, ()) for b in e.blocks):
                raise AssertionError(f"faded {h.hex()[:8]} holds a block that would not tell it when it leaves")
        for b, holders in self._on_block.items():
            for h in holders:
                if h not in self.entries and h not in self.faded:
                    raise AssertionError(f"block {b} still names {h.hex()[:8]}, which is neither")

    def clear(self) -> None:
        for h in list(self.entries):
            self._drop(h)
        for h in list(self.faded):
            self._forget(h)
