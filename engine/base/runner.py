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
    def open(self, seq: int, slot: int) -> None: ...
    def close(self, seq: int) -> None: ...           # must also clean up a partially failed open


class Runner:
    def __init__(self, model: Model, contract: sched.Contract, kv: BlockPool,
                 slots: SlotPool, ring: Ring, recorder: "Recorder | None" = None, tiered=None):
        self.model, self.c, self.kv, self.slots, self.ring = model, contract, kv, slots, ring
        self.tiered = tiered                                # base.tiered_kv.TieredKV, optional
        self.state = sched.State()
        self.slot_of = {}
        self.rec = recorder or Recorder("runner")
        self.steps = 0

    def submit(self, seq: int, prompt_len: int, now: float | None = None) -> None:
        """Publish a request only after its blocks, slot and model state exist.

        Admission failures return everything acquired here; an existing live
        or parked sequence is never released by a failed duplicate submit.
        """
        now = time.monotonic() if now is None else now
        sched.validate_arrival(self.state, seq, prompt_len, now)
        self.kv.row(seq)                                   # reject invalid row before indexing tokens
        if self.kv.tokens[seq] or seq in self.slot_of:
            raise ValueError(f"seq {seq} already owns resident resources")
        if self.tiered is not None and self.tiered.is_parked(seq):
            raise ValueError(f"seq {seq} is parked; resume it before reusing its id")
        self.kv.reserve(seq, prompt_len)                   # the whole prompt is admitted or nothing (D3)
        slot = None
        try:
            slot = self.slots.take(seq)
            try:
                self.model.open(seq, slot)
            except BaseException:
                self.model.close(seq)
                raise
        except BaseException:
            if slot is not None:
                self.slots.give(slot)
            self.kv.release(seq)
            raise
        self.slot_of[seq] = slot
        sched.arrive(self.state, seq, prompt_len, now)

    def _finish(self, seq: int) -> None:
        sched.finish(self.state, seq)
        self.kv.release(seq)
        self.slots.give(self.slot_of.pop(seq))
        self.model.close(seq)

    def park(self, seq: int) -> int:
        """An idle conversation leaves the arena but keeps its KV (D16).
        Only a sequence that is not running: parking a live one would make the
        next decode wait on disk, which D10 forbids."""
        if seq in self.state.running or seq in self.state.waiting:
            raise ValueError(f"seq {seq} is live; only idle sequences park")
        return self.tiered.park(seq)

    def resume(self, seq: int) -> int:
        """Bring a parked conversation back into fresh blocks, off the step path."""
        return self.tiered.resume(seq)

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
    last = STEP_RECORD.unpack(r.ring.ordered()[-1])
    assert last[2] == KIND[sched.DECODE]
    print(f"  runner: {len(kinds)} steps ({kinds.count('prefill')} prefill, {kinds.count('decode')} decode), all homogeneous, kv/slots returned, ring recorded OK")


if __name__ == "__main__":
    _selfcheck()
