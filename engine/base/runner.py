"""The step loop (base): scheduler -> kv -> model -> record, instrumented.

This is the one place the pieces meet, and it knows nothing about a model
beyond a small protocol. A step is homogeneous by construction (D9):
`Model.prefill` and `Model.decode` are different methods and the runner
never calls both in one step.

Graph capture (I1) is the model's business behind `decode`: the runner only
promises that a decode step's inputs arrive as flat arrays whose shapes are
from `shapes`, so a captured graph can be replayed against them. That promise
is what `record` writes down per step -- a step is replayable from its
record because everything the scheduler decided is in it.

Conversations outlive rows (D16). A finished turn is `idle`: it keeps its
row, blocks and state slot and can `wake`, `extend` or be `evict`ed. With a
tier it can also park under a conversation KEY: blocks and slot bytes go
to NVMe with the model's host record, and the row AND slot are returned --
so the number of retained conversations is the disk's, not `max_seqs`.
Resuming brings one back into any free row and any free slot.

Both run on the tier's thread in two halves (D10: a step never waits on the
disk): `park_begin` / `park_finish` and `resume_begin` / `resume_finish`,
with `transfer_done(row)` in between. A row in flight is `retiring` or
`resuming`: neither idle nor free. `park`/`resume` are the halves back to back.
"""
from __future__ import annotations

import struct
import time
from collections import OrderedDict
from typing import Protocol

from engine.base import prefix as prefix_mod
from engine.base import scheduler as sched
from engine.base.instruments import Recorder
from engine.base.kv import BlockPool, SlotPool
from engine.base.kv_tier import TierFull
from engine.base.record import Ring
from engine.base.latency import record_step as _record_step

STEP_RECORD = struct.Struct("<QdBIIi")     # count, wall, kind, n_seqs, tokens, first seq
KIND = {sched.PREFILL: 1, sched.DECODE: 2}


class Pending(Protocol):
    """A decode step launched ahead of its result (45차 §23 B3): `resolve` waits for the device, applies the tokens to the
    model's host view and says which sequences finished. Rows the runner finished meanwhile (from an earlier step's
    result) are reported again and ignored: the model's device side made their extra step inert."""
    def resolve(self) -> "list[bool]": ...


class Model(Protocol):
    def prefill(self, seq: int, start: int, tokens: int, blocks, slot: int) -> "bool | None": ...  # finished on the prompt's first sample?
    def decode(self, seqs, blocks, slots) -> "list[bool]": ...   # per seq: finished?
    # optional: asynchronous decode -- `async_ready(seqs)` says whether these rows can run ahead of the host
    # (captured graphs, plain sampling); `decode_async` launches and returns a Pending. `horizon` must then cover
    # the growth of every step in flight for the row, since reservations precede results.
    def horizon(self, seq: int) -> int: ...  # exclusive end of the next decode writes
    def context(self, seq: int) -> int: ...  # tokens computed so far (a new turn prefills from here)
    def open(self, seq: int, slot: int) -> None: ...
    def close(self, seq: int) -> None: ...           # must also clean up a partially failed open
    # with a prefix cache (base/prefix.py): the position state at a block boundary, out and back in. The boundary at a
    # prefill step's end is copied out of the rings afterwards (`checkpoint`); boundaries INSIDE a step are named before
    # it (`prefill(..., marks={position: snapshot})`) and taken while the step runs, since the rings only hold its end.
    def checkpoint(self, seq: int, position: int, snap: int) -> None: ...   # copy seq's state at `position` into snapshot `snap`
    def restore(self, seq: int, position: int, snap: int) -> None: ...      # seq starts at `position` with that state
    # -- parking (only with a tier) --
    def park(self, seq: int) -> dict: ...            # close the row and return its host record: JSON-serialisable, with
                                                     # "context" (tokens computed) and "pending" (tokens held but not fed)
    def resume(self, seq: int, slot: int, record: dict) -> None: ...   # reopen the row in `slot` from a record; the slot's bytes are restored by the tier
    def state_bytes(self, slot: int): ...            # the slot's bytes as a contiguous uint8 device view (what a tier moves)


class Runner:
    def __init__(self, model: Model, contract: sched.Contract, kv: BlockPool,
                 slots: SlotPool, ring: Ring, recorder: "Recorder | None" = None, tiered=None,
                 keep_idle: bool = False, prefix=None):
        self.model, self.c, self.kv, self.slots, self.ring = model, contract, kv, slots, ring
        self.tiered = tiered                                # base.tiered_kv.TieredKV, optional
        self.keep_idle = keep_idle                          # a finished turn keeps its blocks and slot: the conversation lives (D16)
        self.prefix = prefix                                # base/prefix.py, or None: no reuse across requests
        if prefix is not None:
            if prefix.block_size != kv.block_size or prefix.chunk % contract.chunk_align or contract.chunk_align % prefix.block_size:
                raise ValueError("the prefix cache must share the pool's block size, which must divide the contract's chunk alignment")
            prefix.bind(kv)
        self._chain = {}                                    # seq -> boundary tokens -> hash (live prompts with a cache)
        self.idle = {}                                      # seq -> True: finished, not released, parkable
        self.parked = OrderedDict()                         # key -> record, MRU last and bounded (PARKED_RECORDS_KEPT)
        self.digests = {}                                   # key -> the three numbers a continuation scan actually needs
        self.retiring = {}                                  # row -> (key, record, slot): its park is on the tier's thread
        self.resuming = {}                                  # row -> (key, record, slot): its resume is on the tier's thread
        self.state = sched.State()
        self.slot_of = {}
        self.rec = recorder or Recorder("runner")
        self.steps = 0
        self.inflight = []                                  # [(step, pending, launched_at)]: decode steps ahead of their results
        self._salts = {}                                    # seq -> the prompt's media salts (base/prefix.chain), for boundaries after it
        # -- the prefix tier (45차 §23 A): evicted leaf boundaries survive on NVMe and come back by reading --
        self.prefix_tier = None                             # base.tiered_kv.TieredKV over the prefix tier, or None
        self._spills = {}                                   # hash -> Future: leaf boundaries being written ahead of eviction
        self._restores = {}                                 # row -> (hash, tokens, snap, Future): a boundary being read into the row
        self.spill_low_water = 8                            # keep this many snapshots free by writing leaves out ahead
        self._maintained = -1                               # prefix.version the last candidate scan saw
        self.reused_tokens = 0                              # prompt tokens served from the cache (memory or tier)
        self.prefix_spills = self.prefix_restores = self.dedup_waits = 0
        self.snapshot_self_evicts = 0                        # checkpoints a prefill threw away to make room for its own later ones
        self.depth = 2                                      # steps the device may hold before the host reads the oldest back
        self.async_steps = 0
        self.decode_batches = [0] * (contract.max_running + 1)
        self.sync_drain_steps = 0

    def submit(self, seq: int, prompt_len: int, now: float | None = None, ids=None, salts=(), prepared=None, chain=None) -> None:
        """Publish a request only after its blocks, slot and model state exist.

        Admission failures return everything acquired here; an existing live
        or parked sequence is never released by a failed duplicate submit.
        `ids`: the prompt, when a prefix cache may reuse its beginning; `salts`:
        (position, digest) of the media standing at placeholder ids (base/prefix.chain).
        `prepared`: (tokens, snap) when the row already holds a boundary read back from
        the prefix tier (`restore_finish`): adopted as is, not looked up again.
        """
        now = time.monotonic() if now is None else now
        sched.validate_arrival(self.state, seq, prompt_len, now)
        self.kv.row(seq)                                   # reject invalid row before indexing tokens
        if seq in self.slot_of or (self.kv.tokens[seq] and prepared is None):
            raise ValueError(f"seq {seq} already owns resident resources")
        self._drain_row(seq)                              # a reused row may have an old, inert readback
        reused, snap = 0, None
        if self.prefix is None or ids is None:
            chain = None
        else:
            if len(ids) != prompt_len:
                raise ValueError("the prompt ids must be the prompt")
            if chain is None:
                chain = self.prefix.chain(ids, salts)              # the door hands its chain in; a bare caller computes one
        if prepared is not None:
            reused, snap = prepared
            if self.kv.tokens[seq] != reused or reused >= prompt_len:
                raise ValueError("a prepared row holds exactly its restored boundary, shorter than the prompt")
        elif chain is not None:
            reused, entry, _ = self.prefix.lookup_chain(chain, prompt_len)
            if reused:
                snap = entry.snap
                self.kv.adopt(seq, entry.blocks, reused)   # the shared, complete prefix; the row's own blocks follow
        try:
            self.kv.reserve(seq, prompt_len - reused)      # the whole prompt is admitted or nothing (D3)
        except BaseException:
            if reused:
                self.kv.release(seq)
            raise
        slot = None
        try:
            slot = self.slots.take(seq)
            try:
                self.model.open(seq, slot)
                if reused:
                    self.model.restore(seq, reused, snap)
            except BaseException:
                self.model.close(seq)
                raise
        except BaseException:
            if slot is not None:
                self.slots.give(slot)
            self.kv.release(seq)
            raise
        self.slot_of[seq] = slot
        if chain is not None:
            # An EMPTY chain is kept too: a prompt shorter than one block has no boundary of its own, but what it
            # generates crosses them, and `_rechain` is the only thing that ever sees those. Dropping it here cost
            # every sub-block prompt -- most first turns -- every boundary of its whole answer.
            self._chain[seq] = chain
            self._salts[seq] = tuple(salts)
        self.reused_tokens += reused
        sched.arrive(self.state, seq, prompt_len, now, reused)

    # -- the prefix tier (45차 §23 A) ------------------------------------------------------------
    @staticmethod
    def tier_key(h: bytes) -> int:
        return int.from_bytes(h[:7], "big")                  # the tier indexes by int; 56 bits of the boundary's hash

    def load_prefix_tier(self) -> None:
        """What the prefix tier already holds (from an earlier boot): every record with this cache's hash form."""
        if self.prefix is None or self.prefix_tier is None:
            return
        for key in self.prefix_tier.keys():
            record = self.prefix_tier.record(key)
            if not record or not isinstance(record.get("hash"), str):
                continue
            h = bytes.fromhex(record["hash"])
            if self.tier_key(h) != key:                      # a record that does not name its own slot is not ours
                continue
            self.prefix.hold_tier(h, key)

    def prefix_tier_keys(self) -> "list[int]":
        return sorted(self.prefix.tier_keys.values()) if self.prefix is not None else []

    def maintain_prefix(self) -> None:
        """Once a step: land finished spills, and write the next leaves out while snapshots are still free -- so
        an eviction never waits on the disk (D10) and never loses a boundary the tier could have kept."""
        if self.prefix is None or self.prefix_tier is None:
            return
        prefix, tier = self.prefix, self.prefix_tier
        for h, future in list(self._spills.items()):
            if not future.done():
                continue
            self._spills.pop(h)
            try:
                future.result()
            except TierFull:
                oldest = tier.oldest()
                if oldest is not None and oldest not in self._restoring_keys():
                    tier.forget(oldest)                     # room for the next attempt: the least recently written boundary goes
                    for hh, key in list(prefix.tier_keys.items()):
                        if key == oldest:
                            prefix.forget_tier(hh)          # a faded boundary loses its last copy with it
                prefix.spill_end(h)
                continue
            except Exception:                               # noqa: BLE001 -- a bad write: this boundary stays memory-only
                prefix.spill_end(h, failed=True)
                continue
            key = self.tier_key(h)
            if prefix.tier_holder(key) not in (None, h):
                # somebody took the slot between the write being issued and it landing: the bytes on the
                # tier are this boundary's now, so the OWNER's copy is the one that is gone, not ours
                prefix.forget_tier(prefix.tier_holder(key))
            self.prefix_spills += 1
            prefix.hold_tier(h, key)
            prefix.spill_end(h, spilled=True)
        if len(prefix.free_snaps) >= self.spill_low_water or self._spills:
            return
        if self._maintained == prefix.version and not any(
                s in self.state.prompt_len and self.state.computed.get(s, 0) >= self.state.prompt_len[s] for s in self._chain):
            return                                          # nothing changed since the last scan and no prompt just finished
        self._maintained = prefix.version
        # a prompt still being prefilled extends its own leaf every step: its boundaries wait until it is in
        growing = {h for s, chain in self._chain.items()
                   if s in self.state.prompt_len and self.state.computed.get(s, 0) < self.state.prompt_len[s]
                   for h in chain.values()}
        for h in prefix.spill_candidates(4):
            if h in growing:
                continue
            e = prefix.entries[h]
            held = prefix.tier_holder(self.tier_key(h))
            if held is not None and held != h:
                continue                                     # that slot is another boundary's: this one stays in memory
            record = {"hash": h.hex(), "tokens": e.tokens}
            prefix.spill_begin(h)                           # the blocks are held until the read lands: `available` says so
            try:
                future = tier.tier.run_async(tier.tier.demote, self.tier_key(h), self.kv.storage, list(e.blocks), e.tokens,
                                             self.model.snapshot_bytes(e.snap), record)
            except Exception:                               # noqa: BLE001 -- could not even hand it over
                prefix.spill_end(h, failed=True)
                continue
            self._spills[h] = future
            break                                           # one write at a time: the tier has one staging buffer

    def _restoring_keys(self) -> set:
        return {self.tier_key(h) for h, _, _, _ in self._restores.values()}

    def restore_begin(self, seq: int, h: bytes, tokens: int) -> None:
        """Read the tier's copy of boundary `h` into the empty row `seq` and a free snapshot, on the tier's thread.

        A FADED boundary (base/prefix.py) still holds its blocks here: they were never handed out, so the row adopts
        them where they lie and the read is the snapshot alone -- 77 MiB instead of the boundary's whole KV."""
        if self.prefix is None or self.prefix_tier is None:
            raise ValueError("this runner has no prefix tier")
        if h not in self.prefix.tier_keys:
            raise ValueError("that boundary is not on the prefix tier")
        if self.prefix.tier_holder(self.tier_key(h)) != h:
            # its slot belongs to another boundary: reading it would hand this row the other one's KV
            self.prefix.forget_tier(h)
            raise ValueError("that boundary's tier slot is another boundary's")
        if self.prefix.has(h):
            raise ValueError("that boundary is already in memory")
        self.kv.row(seq)
        if self.kv.tokens[seq] or seq in self.slot_of or seq in self._restores:
            raise ValueError(f"row {seq} is not free")
        if tokens <= 0 or tokens % self.kv.block_size:
            raise ValueError("a boundary is whole blocks")
        snap = self.prefix.take_snapshot()
        if snap is None:
            raise MemoryError("no snapshot is free for the restored boundary")
        held = self.prefix.blocks_of(h)                      # asked after the snapshot: taking one can drop a faded hope
        if held is not None and len(held) != tokens // self.kv.block_size:
            held = None
        try:
            if held is not None:
                self.kv.adopt(seq, held, tokens)
            else:
                self.kv.reserve(seq, tokens)
        except BaseException:
            self.prefix.give_snapshot(snap)
            raise
        blocks = None if held is not None else [b for b in self.kv.row(seq)][: tokens // self.kv.block_size]
        try:
            future = self.prefix_tier.tier.run_async(self.prefix_tier.tier.promote, self.tier_key(h), self.kv.storage, blocks,
                                                     self.model.snapshot_bytes(snap))
        except BaseException:
            self.kv.release(seq)
            self.prefix.give_snapshot(snap)
            raise
        self._restores[seq] = (h, tokens, snap, future)

    def restore_finish(self, seq: int) -> "tuple[int, int]":
        """The read landed: the boundary is a memory entry again (pinned by the cache, held by the row) -- returns
        (tokens, snap) for `submit(prepared=...)`. A failed read frees the row and drops the tier's copy, and raises."""
        h, tokens, snap, future = self._restores.pop(seq)
        try:
            future.result()
        except BaseException:
            self.kv.release(seq)
            self.prefix.give_snapshot(snap)
            key = self.prefix.forget_tier(h)
            if key is not None:
                try:
                    self.prefix_tier.forget(key)
                except Exception:                           # noqa: BLE001 -- the copy is unreadable either way
                    pass
            raise
        blocks = tuple(self.kv.row(seq)[: tokens // self.kv.block_size])
        self.prefix.insert(h, blocks, tokens, snap)
        e = self.prefix.entries[h]
        e.spilled, e.hits = True, 1
        self.prefix.hits += 1
        self.prefix_restores += 1
        return tokens, snap

    def restore_undo(self, seq: int) -> None:
        """A restore that landed here but not on every rank: the row goes back, the memory entry stays (harmless)."""
        if self.kv.tokens[seq]:
            self.kv.release(seq)

    def restore_cancel(self, seq: int) -> None:
        """Wait for the row's read and drop everything it took (shutdown / a cancelled request)."""
        if seq not in self._restores:
            return
        try:
            self.restore_finish(seq)
        except Exception:                                   # noqa: BLE001
            return
        self.restore_undo(seq)

    # -- the same prompt, twice at once (45차 §23 B) ------------------------------------------------
    def shared_ahead(self, ids, salts=(), above: int = 0, chain=None) -> "bytes | None":
        """The hash of the longest boundary of `ids` (past `above`) that a prefill now running will cache when it gets
        there -- a request that waits for it adopts it instead of computing the same tokens beside it. None if no
        live prefill shares that much. `chain`: the prompt's, when the caller has it."""
        if self.prefix is None:
            return None
        if chain is None:
            chain = self.prefix.chain(ids, salts)
        live = [s for s in self._chain if s in self.state.prompt_len and self.state.computed.get(s, 0) < self.state.prompt_len[s]]
        for tokens in sorted(chain, reverse=True):
            if tokens <= above or tokens >= len(ids) or chain[tokens] in self.prefix.entries:
                continue
            for s in live:
                if self._chain[s].get(tokens) == chain[tokens] and self.state.computed.get(s, 0) < tokens <= self.state.prompt_len[s]:
                    return chain[tokens]
        return None

    def _finish(self, seq: int) -> None:
        sched.finish(self.state, seq)
        if self.keep_idle:
            self.idle[seq] = True
            return
        self._release(seq)

    def _release(self, seq: int) -> None:
        self._chain.pop(seq, None)
        self._salts.pop(seq, None)
        if self.kv.tokens[seq]:
            self.kv.release(seq)
        self.slots.give(self.slot_of.pop(seq))
        self.model.close(seq)

    def cancel(self, seq: int) -> None:
        """Release a live or idle row (a transfer in flight is settled first, which waits).
        Parked conversations are not rows: see `forget_parked`."""
        if seq not in self.slot_of:
            return
        self._drain_row(seq)                                  # unrelated later steps may remain in flight
        self._settle_row(seq)
        if seq not in self.slot_of:                          # its park finished: the row is free already
            return
        if seq in self.state.running or seq in self.state.waiting:
            sched.finish(self.state, seq)
        self.idle.pop(seq, None)
        self._release(seq)

    def evict(self, seq: int) -> None:
        """The idle conversation is over: blocks and slot go."""
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle")
        self.cancel(seq)

    # -- the tier: conversations leave their rows (D16), off the step path (D10) -----------------
    def park_begin(self, seq: int, key: "int | None" = None) -> None:
        """An idle conversation starts leaving the arena: the model's host record is taken and the
        blocks + slot bytes are handed to the tier's thread under `key` (the row id by default).
        The row is `retiring` until `park_finish`: not idle, not free. Only an idle one: parking a
        live one would make the next decode wait on disk, which D10 forbids."""
        if self.tiered is None:
            raise ValueError("this runner has no tier to park on")
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle; only idle conversations park")
        key = seq if key is None else key
        if self.is_parked(key) or any(k == key for k, _, _ in self.retiring.values()):
            raise ValueError(f"conversation {key} is already parked")
        slot = self.slot_of[seq]
        record = self.model.park(seq)
        if not isinstance(record, dict) or not all(isinstance(record.get(k), int) for k in ("context", "pending")):
            self.model.resume(seq, slot, record)
            raise ValueError("a park record must be a dict with integer 'context' and 'pending'")
        try:
            self.tiered.park_begin(seq, key, extra=self.model.state_bytes(slot), record=record)
        except BaseException:
            self.model.resume(seq, slot, record)           # nothing left the arena: the row stays idle and resident
            raise
        self.idle.pop(seq)
        self.retiring[seq] = (key, record, slot)

    def park_finish(self, seq: int) -> int:
        """The write is done: the blocks, the slot and the row are free. A failed write (TierFull,
        OSError) leaves the row idle and resident again, and raises."""
        key, record, slot = self.retiring.pop(seq)
        try:
            wrote = self.tiered.park_finish(seq)
        except BaseException:
            self.model.resume(seq, slot, record)
            self.idle[seq] = True
            raise
        self.slots.give(self.slot_of.pop(seq))
        self.parked[key] = record
        return wrote

    def park(self, seq: int, key: "int | None" = None) -> int:
        """Both halves, waiting in between: for callers that may wait (a local check, shutdown)."""
        self.park_begin(seq, key)
        return self.park_finish(seq)

    def resume_begin(self, seq: int, key: "int | None" = None) -> None:
        """A parked conversation starts coming back into the free row `seq`: blocks reserved, a slot
        taken, the read handed to the tier's thread. The row is `resuming` until `resume_finish`."""
        if self.tiered is None:
            raise ValueError("this runner has no tier to resume from")
        key = seq if key is None else key
        if not self.is_parked(key):
            raise ValueError(f"conversation {key} is not parked")
        if any(k == key for k, _, _ in self.resuming.values()):
            raise ValueError(f"conversation {key} is already resuming")
        self.kv.row(seq)
        if self.kv.tokens[seq] or seq in self.slot_of or seq in self.state.prompt_len:
            raise ValueError(f"row {seq} is not free")
        record = self.parked.get(key)
        if record is None:
            record = self.tiered.record(key)
        if record is None:
            raise ValueError(f"conversation {key} has no record on the tier: it cannot be reopened")
        slot = self.slots.take(seq)
        try:
            self.tiered.resume_begin(seq, key, extra=self.model.state_bytes(slot))
        except BaseException:
            self.slots.give(slot)
            raise
        self.slot_of[seq] = slot
        self.resuming[seq] = (key, record, slot)

    def resume_finish(self, seq: int) -> int:
        """The read is done: the row is idle with the conversation's state. A failed read frees
        the row, the slot and the blocks it took, keeps the disk copy, and raises."""
        key, record, slot = self.resuming.pop(seq)
        try:
            got = self.tiered.resume_finish(seq)
        except BaseException:
            self.slots.give(self.slot_of.pop(seq))
            raise
        self.model.resume(seq, slot, record)
        self.idle[seq] = True
        self.parked.pop(key, None)
        self.digests.pop(key, None)
        return got

    def resume(self, seq: int, key: "int | None" = None) -> int:
        self.resume_begin(seq, key)
        return self.resume_finish(seq)

    def transfer_done(self, seq: int) -> bool:
        """Whether the row's park/resume/restore has finished on the tier's thread (never blocks)."""
        if seq in self._restores:
            return self._restores[seq][3].done()
        return self.tiered.done(seq)

    def _settle_row(self, seq: int) -> None:
        """Wait for the row's transfer and finish it, swallowing its failure (shutdown only)."""
        try:
            if seq in self._restores:
                self.restore_cancel(seq)
            elif seq in self.retiring:
                self.park_finish(seq)
            elif seq in self.resuming:
                self.resume_finish(seq)
        except Exception:                                    # noqa: BLE001 -- the row is idle/free either way
            pass

    def settle(self) -> None:
        """Every transfer and step in flight, finished (shutdown/abort path: this waits)."""
        try:
            self.drain()
        except Exception:                                    # noqa: BLE001 -- shutdown: the engine's failure is reported elsewhere
            self.inflight.clear()
        for seq in list(self.retiring) + list(self.resuming) + list(self._restores):
            self._settle_row(seq)
        for future in list(self._spills.values()):
            try:
                future.result()
            except Exception:                                # noqa: BLE001 -- shutdown
                pass
        self._spills.clear()

    # -- steps ahead of their results (45차 §23 B3) ------------------------------------------------
    def resolve_oldest(self) -> None:
        """Read the oldest launched decode step back and apply it: finished rows leave; rows that already left
        (finished by the step before, then run once more as ghosts) are ignored."""
        step, pending, launched = self.inflight.pop(0)
        before = {s: self.model.context(s) for s in self._tracked(step.seqs)}
        resolve_start = time.perf_counter()
        done = pending.resolve()
        latency = getattr(self, 'latency', None)
        if latency is not None:
            latency.row(kind='host_wait', operation='resolve', phase='decode', rows=list(step.seqs),
                        duration_us=(time.perf_counter() - resolve_start) * 1e6)
        if len(done) != len(step.seqs):
            raise ValueError("decode must return one completion flag per sequence")
        for seq, finished in zip(step.seqs, done):
            if seq in before and seq in self.slot_of and seq in self.state.running:
                self._generated_boundaries(seq, before[seq], self.model.context(seq))
            if finished and seq in self.state.running:
                self._finish(seq)
        self.ring.push(STEP_RECORD.pack(self.steps, time.perf_counter() - launched, KIND[step.kind],
                                        len(step.seqs), step.tokens, step.seqs[0]))

    def drain(self) -> None:
        while self.inflight:
            self.resolve_oldest()

    def _drain_row(self, seq: int) -> None:
        """Settle only the prefix of the queue which still references this row."""
        while any(seq in step.seqs for step, _, _ in self.inflight):
            self.resolve_oldest()

    def _async_ok(self, step) -> bool:
        ready = getattr(self.model, "async_ready", None)
        return (step.kind == sched.DECODE and ready is not None and hasattr(self.model, "decode_async")
                and bool(ready(step.seqs)))

    def is_parked(self, key: int) -> bool:
        return self.tiered is not None and (key in self.parked or self.tiered.is_parked(key))

    PARKED_RECORDS_KEPT = 8
    """How many whole records stay in memory.

    A record carries the conversation's ENTIRE token list -- 3.8 MiB of Python ints for a 100K
    token turn -- and the door's continuation scan used to pull one for every parked
    conversation on every request. With 280 parked, which is what this fleet actually holds,
    that is **1.04 GiB** of host memory resident for a scan whose per-candidate question is
    three numbers, on a box whose OOM floor is an absolute 6 GiB (45차 §62).

    So the records are an LRU and the three numbers are `parked_digest`, kept for everybody.
    Eight is the rows this engine can hold plus slack: a record is read when a candidate
    actually passes the cheap test, or when a conversation resumes, and both are rare.
    """

    def parked_record(self, key: int) -> "dict | None":
        """The whole record, read from the tier when it is not one of the few held."""
        if not self.is_parked(key):
            return None
        record = self.parked.get(key)
        if record is not None:
            self.parked.move_to_end(key)
            return record
        record = self.tiered.record(key)
        if record is not None:
            self._hold_record(key, record)
        return record

    def _hold_record(self, key: int, record: dict) -> None:
        self.parked[key] = record
        self.parked.move_to_end(key)
        while len(self.parked) > self.PARKED_RECORDS_KEPT:
            self.parked.popitem(last=False)

    def parked_digest(self, key: int) -> "dict | None":
        """What a continuation scan needs to reject a candidate: its length, its last two ids, its pictures.

        Kept for every parked conversation because it is a handful of bytes; the token list behind
        it is read only when these three say the candidate could match.
        """
        digest = self.digests.get(key)
        if digest is not None:
            return digest
        record = self.parked_record(key)
        if record is None or "tokens" not in record:
            return None
        tokens = record["tokens"]
        if len(tokens) < 2:
            return None
        digest = {"tokens": len(tokens), "last": tokens[-1], "prev": tokens[-2],
                  "media": [(r[2], r[1]) for r in record.get("media", [])]}
        self.digests[key] = digest
        return digest

    def parked_blocks(self, key: int) -> int:
        return self.tiered.blocks(key)

    def parked_keys(self) -> "list[int]":
        return self.tiered.keys() if self.tiered is not None else []

    def forget_parked(self, key: int) -> None:
        """A parked conversation is over: its disk copy and record go."""
        if self.tiered is not None:
            self.tiered.forget(key)
        self.parked.pop(key, None)
        self.digests.pop(key, None)

    def forget_oldest_parked(self) -> "int | None":
        """Make room on the tier: forget the least recently parked conversation. Returns its key."""
        key = self.tiered.oldest() if self.tiered is not None else None
        if key is not None:
            self.forget_parked(key)
        return key

    def transfers(self) -> "list[int]":
        """Rows with a park, resume or prefix restore in flight, in a fixed order (the same on every rank)."""
        return sorted(self.retiring) + sorted(self.resuming) + sorted(self._restores)

    def wake(self, seq: int) -> None:
        """An idle conversation decodes again (its next token is pending in the model)."""
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle")
        self._drain_row(seq)
        if len(self.state.running) + int(self.state.in_prefill is not None) >= self.c.max_running:
            raise ValueError("decode width is full; wake it later")
        self.idle.pop(seq)
        self.state.running.append(seq)

    def extend(self, seq: int, tokens: int, now: "float | None" = None) -> None:
        """A new turn on an idle conversation: `tokens` more prompt tokens to
        prefill on top of what the caches already hold (from the model's
        context, not kv.tokens: reservations overshoot by the last horizon)."""
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle")
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("a turn adds at least one token to prefill")
        self._drain_row(seq)
        held = self.model.context(seq)
        now = time.monotonic() if now is None else now
        sched.validate_arrival(self.state, seq, held + tokens, now)
        self.kv.reserve_to((seq,), (held + tokens,))
        self.idle.pop(seq)
        sched.arrive(self.state, seq, held + tokens, now)
        self.state.computed[seq] = held
        self._rechain(seq)                                  # the new turn's boundaries can be marked and cached like a prompt's

    def _checkpoint(self, seq: int, position: int) -> None:
        """A prefill just reached `position`: if it is a block boundary nobody cached yet, keep the
        model's state there and pin the blocks before it."""
        chain = self._chain[seq]
        h = chain.get(position)
        if h is None or self.prefix.has(h):
            return
        snap = self.prefix.take_snapshot()
        self._note_fade(chain)
        if snap is None:
            return                                          # every snapshot is in use by a live boundary: this one goes uncached
        try:
            self.model.checkpoint(seq, position, snap)
        except BaseException:
            self.prefix.give_snapshot(snap)
            raise
        self._insert(seq, position, h, snap)

    def _rechain(self, seq: int) -> None:
        """The boundary hashes continued over what the row holds past its last known boundary -- generated tokens, a
        new turn -- when the model can tell its history (and the pictures standing in it). Only the new blocks are hashed."""
        if self.prefix is None or seq not in self._chain:
            return
        history = getattr(self.model, "history", None)
        if history is None:
            return
        marks = getattr(self.model, "media_marks", None)
        held = list(self._salts.get(seq, ()))
        if marks is None:
            salts = held
        else:                                               # the model knows the pictures; the tenant is not its business
            salts = [s for s in held if prefix_mod.is_tenant_salt(s)] + [(p, bytes.fromhex(d)) for p, d in marks(seq)]
        chain = self._chain[seq]
        last = max(chain) if chain else 0
        tail = getattr(self.model, "history_from", None)
        ids = tail(seq, last) if tail is not None else list(history(seq))[last:]
        self._chain[seq] = self.prefix.extend_chain(chain, ids, salts, start=last)

    def reset_prefix(self) -> dict:
        """Forget every cached boundary, on this rank, between two steps.

        The cache's own bookkeeping is `PrefixCache.reset`; the tier's blobs are the runner's,
        because the runner is what owns the disk. A boundary a row is restoring right now is left
        alone -- its blocks are being written into as this runs -- and so is one being spilled.

        Rows already running keep everything they adopted: a boundary that has been handed to a
        row is that row's KV now, and forgetting the NAME does not take the blocks back. What the
        reset buys is that nothing NEW adopts a stale prefix (45차 §55).
        """
        if self.prefix is None:
            return {"entries": 0, "faded": 0, "kept_spilling": 0, "tier_keys": [], "tier_forgotten": 0}
        restoring = self._restoring_keys()
        report = self.prefix.reset()
        forgotten = 0
        tier = self.prefix_tier
        for key in report["tier_keys"]:
            if tier is None or key in restoring:
                continue
            try:
                tier.forget(key)
                forgotten += 1
            except Exception as exc:                    # noqa: BLE001 -- a disk that refuses must not stop the reset
                print(f"  prefix reset: tier slot {key} stayed: {type(exc).__name__}: {exc}", flush=True)
        report["tier_forgotten"] = forgotten
        return report

    def _tracked(self, seqs) -> "list[int]":
        """Rows whose generated boundaries can enter the prefix cache: a chain exists and the model tells its history."""
        if self.prefix is None or getattr(self.model, "history", None) is None:
            return []
        return [s for s in seqs if s in self._chain and s in self.slot_of]

    def _generated_boundaries(self, seq: int, before: int, after: int) -> None:
        """A decode step moved `seq` from `before` to `after`: the block boundaries it crossed become prefix entries
        (45차 §23: a conversation the tier forgot, resent, still finds its answer's blocks). The state at a boundary
        is in the rings right after the step, or in the caches' stage when the step ran ahead of the host."""
        if self.prefix is None or seq not in self._chain:
            return
        block = self.kv.block_size
        first = (before // block + 1) * block
        if first > after:
            return
        self._rechain(seq)
        for position in range(first, after + 1, block):
            self._checkpoint(seq, position)

    def _insert(self, seq: int, position: int, h: bytes, snap: int) -> None:
        blocks = tuple(self.kv.row(seq)[: position // self.kv.block_size])
        self.prefix.insert(h, blocks, position, snap)

    def _marks(self, seq: int, start: int, end: int) -> dict:
        """{position: snapshot} for the uncached block boundaries strictly inside a prefill step [start, end): the
        rings will not hold those states after the step, so the model takes them on the way (45차 §23: block-level
        reuse -- a 5,000-token prompt shares its first 4,608 tokens, not nothing)."""
        marks = {}
        if seq not in self._chain:
            return marks
        chain = self._chain[seq]
        for position in range(start + self.kv.block_size - start % self.kv.block_size, end, self.kv.block_size):
            h = chain.get(position)
            if h is None or self.prefix.has(h):
                continue
            snap = self.prefix.take_snapshot()
            self._note_fade(chain)
            if snap is None:
                break
            marks[position] = snap
        return marks

    def _note_fade(self, chain: dict) -> None:
        """A `take_snapshot` just displaced a boundary. Count it when the boundary was this row's own: with no minimum
        spacing between checkpoints, a prompt longer than the resident snapshot count allows (which follows
        boot.PREFIX_SNAPSHOT_GIB and the shape) evicts its own earlier ones as it goes, and the state copy that
        made each of them was device work spent for nothing. This is the meter that says
        whether that is happening; it does not change who gets evicted (`prefix._victim`)."""
        fade = self.prefix.last_fade
        if fade is not None and fade in chain.values():
            self.snapshot_self_evicts += 1

    def step(self, now: float | None = None) -> "sched.Step | None":
        now = time.monotonic() if now is None else now
        self.maintain_prefix()
        while True:
            if len(self.inflight) >= self.depth:
                self.resolve_oldest()
            step = sched.plan(self.state, self.c, now)
            if step is None:
                if self.inflight:
                    self.resolve_oldest()
                    continue
                return None
            if self._async_ok(step):
                return self._launch(step)
            if self.inflight:                                # a prefill or a synchronous decode needs every step ahead landed
                self.sync_drain_steps += 1
                self.resolve_oldest()
                continue
            return self._run(step)


    @_record_step
    def _launch(self, step) -> "sched.Step":
        """A decode step the device runs while the host goes on: its result is read back at `resolve_oldest`."""
        with self.rec.phase(step.kind, aggregate=True):
            self.kv.reserve_to(step.seqs, [self.model.horizon(s) for s in step.seqs])
            pending = self.model.decode_async(step.seqs, [self.kv.row(s) for s in step.seqs],
                                              [self.slot_of[s] for s in step.seqs])
            sched.advance(self.state, step)
        self.steps += 1
        self.async_steps += 1
        self.decode_batches[len(step.seqs)] += 1
        self.inflight.append((step, pending, time.perf_counter()))
        self.rec.count(f"{step.kind}_steps")
        self.rec.count(f"{step.kind}_tokens", step.tokens)
        return step

    @_record_step
    def _run(self, step) -> "sched.Step":
        t0 = time.perf_counter()
        with self.rec.phase(step.kind, aggregate=True):
            if step.kind == sched.PREFILL:
                (seq,) = step.seqs
                start = self.state.computed[seq]
                marks = self._marks(seq, start, start + step.tokens)
                try:
                    finished = self.model.prefill(seq, start, step.tokens, self.kv.row(seq), self.slot_of[seq],
                                                  **({"marks": marks} if marks else {}))
                except BaseException:
                    for snap in marks.values():
                        self.prefix.give_snapshot(snap)
                    raise
                if finished and start + step.tokens != self.state.prompt_len[seq]:
                    raise ValueError("prefill may finish only at the end of the prompt")
                for position, snap in marks.items():
                    self._insert(seq, position, self._chain[seq][position], snap)
                if seq in self._chain:
                    self._checkpoint(seq, start + step.tokens)
            else:
                self.kv.reserve_to(step.seqs, [self.model.horizon(s) for s in step.seqs])
                before = {s: self.model.context(s) for s in self._tracked(step.seqs)}
                done = self.model.decode(step.seqs, [self.kv.row(s) for s in step.seqs],
                                         [self.slot_of[s] for s in step.seqs])
                if len(done) != len(step.seqs):
                    raise ValueError("decode must return one completion flag per sequence")
                for seq in before:
                    self._generated_boundaries(seq, before[seq], self.model.context(seq))
            sched.advance(self.state, step)
            if step.kind == sched.PREFILL and finished:
                self._finish(seq)
            if step.kind == sched.DECODE:
                for seq, finished in zip(step.seqs, done):
                    if finished:
                        self._finish(seq)
        self.steps += 1
        if step.kind == sched.DECODE:
            self.decode_batches[len(step.seqs)] += 1
        self.ring.push(STEP_RECORD.pack(self.steps, time.perf_counter() - t0, KIND[step.kind],
                                        len(step.seqs), step.tokens, step.seqs[0]))
        self.rec.count(f"{step.kind}_steps")
        self.rec.count(f"{step.kind}_tokens", step.tokens)
        return step


def _selfcheck() -> None:
    class Fake:
        def __init__(self): self.calls = []; self.left = {}; self.ctx = {}
        def open(self, seq, slot): self.left[seq] = 3; assert slot != 0
        def close(self, seq): self.left.pop(seq)
        def horizon(self, seq): return self.ctx[seq] + 1
        def context(self, seq): return self.ctx[seq]
        def prefill(self, seq, start, tokens, blocks, slot):
            assert all(b != -1 for b in list(blocks)[: -(-(start + tokens) // 16)]), "prefill must see its blocks"
            self.calls.append(("prefill", seq, start, tokens)); self.ctx[seq] = start + tokens
        def decode(self, seqs, blocks, slots):
            self.calls.append(("decode", tuple(seqs)))
            out = []
            for s in seqs:
                self.left[s] -= 1; self.ctx[s] += 1; out.append(self.left[s] == 0)
            return out
    c = sched.Contract(chunk_align=16, token_budget=64, draft_slots=0, max_wait_s=20.0, max_running=8)
    r = Runner(Fake(), c, BlockPool(64, 16, 8, 32), SlotPool(9), Ring(16, STEP_RECORD.size))   # 9 = null + 8
    r.submit(1, 100, now=0.0); r.submit(2, 20, now=0.0)
    kinds = []
    t = 0.0
    while (s := r.step(now=t)) is not None:
        kinds.append(s.kind); t += 0.01
        assert len({s.kind}) == 1
    # D10 sequential, exactly: seq 1 prefills (64 + 36 tail), then DECODES TO
    # THE END while seq 2 waits -- its prefill runs only once no one is
    # decoding -- then seq 2 decodes. No prefill ever lands beside a decoder.
    assert kinds == ["prefill", "prefill", "decode", "decode", "decode",
                     "prefill", "decode", "decode", "decode"], kinds
    assert r.state.running == [] and r.kv.available == 64 and r.slots.available == 8   # 8 usable of 9
    assert r.ring.count == len(kinds)
    # conversations that live on: a finished turn keeps blocks and slot, wakes to decode more, extends, is evicted
    r2 = Runner(Fake(), c, BlockPool(64, 16, 8, 32), SlotPool(9), Ring(16, STEP_RECORD.size), keep_idle=True)
    r2.submit(5, 40, now=0.0)
    while r2.step(now=0.0) is not None:
        pass
    assert 5 in r2.idle and r2.kv.available < 64 and r2.slots.available == 7
    r2.model.left[5] = 2; r2.wake(5)
    assert r2.step(now=0.0).kind == "decode" and r2.step(now=0.0).kind == "decode" and 5 in r2.idle
    r2.extend(5, 20, now=0.0); r2.model.left[5] = 1
    assert r2.step(now=0.0).kind == "prefill" and r2.state.computed[5] == r2.state.prompt_len[5]
    assert r2.step(now=0.0).kind == "decode" and 5 in r2.idle
    r2.evict(5); assert r2.kv.available == 64 and r2.slots.available == 8 and 5 not in r2.idle
    last = STEP_RECORD.unpack(r.ring.ordered()[-1])
    assert last[2] == KIND[sched.DECODE]
    print(f"  runner: {len(kinds)} steps ({kinds.count('prefill')} prefill, {kinds.count('decode')} decode), all homogeneous, kv/slots returned, ring recorded; keep_idle: finish keeps blocks, wake/extend/evict OK")


if __name__ == "__main__":
    _selfcheck()
