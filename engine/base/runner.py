"""The step loop (base): scheduler -> kv -> model -> record, instrumented.

This is the one place the pieces meet, and it knows nothing about a model
beyond a four-method protocol. A step is homogeneous by construction (D9):
`Model.prefill` and `Model.decode` are different methods and the runner
never calls both in one step.

Graph capture (I1) is the model's business behind `decode`: the runner only
promises that a decode step's inputs arrive as flat arrays whose shapes are
from `shapes`, so a captured graph can be replayed against them. That promise
is what `record` writes down per step -- a step is replayable from its
record because everything the scheduler decided is in it.
"""
from __future__ import annotations

import struct
import time
from typing import Protocol

from engine.base import scheduler as sched
from engine.base.instruments import Recorder
from engine.base.kv import BlockPool, SlotPool
from engine.base.record import Ring

STEP_RECORD = struct.Struct("<QdBIIi")     # count, wall, kind, n_seqs, tokens, first seq
KIND = {sched.PREFILL: 1, sched.DECODE: 2}


class Model(Protocol):
    def prefill(self, seq: int, start: int, tokens: int, blocks, slot: int) -> "bool | None": ...  # finished on the prompt's first sample?
    def decode(self, seqs, blocks, slots) -> "list[bool]": ...   # per seq: finished?
    def horizon(self, seq: int) -> int: ...  # exclusive end of the next decode writes
    def context(self, seq: int) -> int: ...  # tokens computed so far (a new turn prefills from here)
    def open(self, seq: int, slot: int) -> None: ...
    def close(self, seq: int) -> None: ...           # must also clean up a partially failed open
    # with a prefix cache (base/prefix.py): the position state at a chunk boundary, out and back in
    def checkpoint(self, seq: int, position: int, snap: int) -> None: ...   # copy seq's state at `position` into snapshot `snap`
    def restore(self, seq: int, position: int, snap: int) -> None: ...      # seq starts at `position` with that state


class Runner:
    def __init__(self, model: Model, contract: sched.Contract, kv: BlockPool,
                 slots: SlotPool, ring: Ring, recorder: "Recorder | None" = None, tiered=None,
                 keep_idle: bool = False, prefix=None):
        self.model, self.c, self.kv, self.slots, self.ring = model, contract, kv, slots, ring
        self.tiered = tiered                                # base.tiered_kv.TieredKV, optional
        self.keep_idle = keep_idle                          # a finished turn keeps its blocks and slot: the conversation lives (D16)
        self.prefix = prefix                                # base/prefix.py, or None: no reuse across requests
        if prefix is not None:
            if prefix.block_size != kv.block_size or prefix.chunk % contract.chunk_align:
                raise ValueError("the prefix cache must share the pool's block size and align with the contract's chunks")
            prefix.bind(kv)
        self._chain = {}                                    # seq -> boundary tokens -> hash (live prompts with a cache)
        self.idle = {}                                      # seq -> True: finished, not released, parkable
        self.state = sched.State()
        self.slot_of = {}
        self.rec = recorder or Recorder("runner")
        self.steps = 0

    def submit(self, seq: int, prompt_len: int, now: float | None = None, ids=None) -> None:
        """Publish a request only after its blocks, slot and model state exist.

        Admission failures return everything acquired here; an existing live
        or parked sequence is never released by a failed duplicate submit.
        `ids`: the prompt, when a prefix cache may reuse its beginning.
        """
        now = time.monotonic() if now is None else now
        sched.validate_arrival(self.state, seq, prompt_len, now)
        self.kv.row(seq)                                   # reject invalid row before indexing tokens
        if self.kv.tokens[seq] or seq in self.slot_of:
            raise ValueError(f"seq {seq} already owns resident resources")
        if self.tiered is not None and self.tiered.is_parked(seq):
            raise ValueError(f"seq {seq} is parked; resume it before reusing its id")
        reused, entry, chain = 0, None, None
        if self.prefix is not None and ids is not None:
            if len(ids) != prompt_len:
                raise ValueError("the prompt ids must be the prompt")
            reused, entry, _ = self.prefix.lookup(ids)
            chain = self.prefix.chain(ids)
        if reused:
            self.kv.adopt(seq, entry.blocks, reused)       # the shared, complete prefix; the row's own blocks follow
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
                    self.model.restore(seq, reused, entry.snap)
            except BaseException:
                self.model.close(seq)
                raise
        except BaseException:
            if slot is not None:
                self.slots.give(slot)
            self.kv.release(seq)
            raise
        self.slot_of[seq] = slot
        if chain:
            self._chain[seq] = chain
        sched.arrive(self.state, seq, prompt_len, now, reused)

    def _finish(self, seq: int) -> None:
        sched.finish(self.state, seq)
        if self.keep_idle:
            self.idle[seq] = True
            return
        self._release(seq)

    def _release(self, seq: int) -> None:
        self._chain.pop(seq, None)
        if self.kv.tokens[seq]:
            self.kv.release(seq)
        self.slots.give(self.slot_of.pop(seq))
        self.model.close(seq)

    def cancel(self, seq: int) -> None:
        """Release a live or idle conversation, including its parked disk copy."""
        if seq not in self.slot_of:
            return
        if seq in self.state.running or seq in self.state.waiting:
            sched.finish(self.state, seq)
        self.idle.pop(seq, None)
        try:
            if self.tiered is not None:
                if self.tiered.is_parked(seq) or str(seq) in self.tiered.tier.index:
                    self.tiered.tier.forget(seq)
        finally:
            if self.tiered is not None:
                self.tiered.parked.pop(seq, None)
            self._release(seq)

    def evict(self, seq: int) -> None:
        """The idle conversation is over: blocks, disk copy and slot go."""
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle")
        self.cancel(seq)


    def park(self, seq: int) -> int:
        """An idle conversation leaves the arena but keeps its KV (D16).
        Only an idle one: parking a live one would make the next decode wait
        on disk, which D10 forbids."""
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle; only idle conversations park")
        return self.tiered.park(seq)

    def resume(self, seq: int) -> int:
        """Bring a parked conversation back into fresh blocks, off the step path."""
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle")
        return self.tiered.resume(seq)

    def wake(self, seq: int) -> None:
        """An idle conversation decodes again (its next token is pending in the
        model); a parked one must be resumed first."""
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle")
        if self.tiered is not None and self.tiered.is_parked(seq):
            raise ValueError(f"seq {seq} is parked: resume it first")
        if len(self.state.running) >= self.c.max_running:
            raise ValueError("decode width is full; wake it later")
        self.idle.pop(seq)
        self.state.running.append(seq)

    def extend(self, seq: int, tokens: int, now: "float | None" = None) -> None:
        """A new turn on an idle conversation: `tokens` more prompt tokens to
        prefill on top of what the caches already hold (from the model's
        context, not kv.tokens: reservations overshoot by the last horizon)."""
        if seq not in self.idle:
            raise ValueError(f"seq {seq} is not idle")
        if self.tiered is not None and self.tiered.is_parked(seq):
            raise ValueError(f"seq {seq} is parked: resume it first")
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("a turn adds at least one token to prefill")
        held = self.model.context(seq)
        now = time.monotonic() if now is None else now
        sched.validate_arrival(self.state, seq, held + tokens, now)
        self.kv.reserve_to((seq,), (held + tokens,))
        self.idle.pop(seq)
        sched.arrive(self.state, seq, held + tokens, now)
        self.state.computed[seq] = held

    def _checkpoint(self, seq: int, position: int) -> None:
        """A prefill just reached `position`: if it is a chunk boundary nobody cached yet, keep the
        model's state there and pin the blocks before it."""
        h = self._chain[seq].get(position)
        if h is None or self.prefix.has(h):
            return
        snap = self.prefix.take_snapshot()
        if snap is None:
            return                                          # every snapshot is in use by a live boundary: this one goes uncached
        try:
            self.model.checkpoint(seq, position, snap)
        except BaseException:
            self.prefix.give_snapshot(snap)
            raise
        blocks = tuple(self.kv.row(seq)[: position // self.kv.block_size])
        self.prefix.insert(h, blocks, position, snap)

    def step(self, now: float | None = None) -> "sched.Step | None":
        now = time.monotonic() if now is None else now
        step = sched.plan(self.state, self.c, now)
        if step is None:
            return None
        t0 = time.perf_counter()
        with self.rec.phase(step.kind, aggregate=True):
            if step.kind == sched.PREFILL:
                (seq,) = step.seqs
                start = self.state.computed[seq]
                finished = self.model.prefill(seq, start, step.tokens, self.kv.row(seq), self.slot_of[seq])
                if finished and start + step.tokens != self.state.prompt_len[seq]:
                    raise ValueError("prefill may finish only at the end of the prompt")
                if seq in self._chain:
                    self._checkpoint(seq, start + step.tokens)
            else:
                self.kv.reserve_to(step.seqs, [self.model.horizon(s) for s in step.seqs])
                done = self.model.decode(step.seqs, [self.kv.row(s) for s in step.seqs],
                                         [self.slot_of[s] for s in step.seqs])
                if len(done) != len(step.seqs):
                    raise ValueError("decode must return one completion flag per sequence")
            sched.advance(self.state, step)
            if step.kind == sched.PREFILL and finished:
                self._finish(seq)
            if step.kind == sched.DECODE:
                for seq, finished in zip(step.seqs, done):
                    if finished:
                        self._finish(seq)
        self.steps += 1
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
