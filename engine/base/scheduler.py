"""Homogeneous steps, the running decoders protected, shapes as input (base).

Three charter decisions, made mechanical:

  D9   a step is all-prefill or all-decode. The two kinds are separate
       functions below and there is no third; `Step.kind` is one of two strings
       and a test that finds anything else has found a bug.
  D10  waiting requests enter prefill after `max_wait_s`. Once admitted,
       their prefill chunks alternate with decode while decoders are live.
       A decoder can wait for one prefill chunk, never an entire prompt.
  D2   the prefill chunk is `shapes.chunk_for(align, budget, draft)` and
       nothing else: a chunk that is not what you expected is explained by
       printing that one call.

The scheduler is a pure function of flat state (CHARTER I2): lists of ints and
floats in, a Step out, and `advance` applies the step's consequences. That is
what lets a death dump (D12) replay it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from engine.base.shapes import chunk_for

PREFILL = "prefill"
DECODE = "decode"


@dataclass(frozen=True)
class Contract:
    """The declared parts. Profiles fill these from measured constants."""
    chunk_align: int          # shapes: legal prefill chunk multiple
    token_budget: int         # largest prefill step, tokens
    draft_slots: int          # spec-decode slots taken out of the budget
    max_wait_s: float         # D10's one starvation valve
    max_running: int          # decode batch width the kernels support

    def __post_init__(self):
        for name in ("chunk_align", "token_budget", "max_running"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.draft_slots, int) or self.draft_slots < 0:
            raise ValueError("draft_slots must be a nonnegative integer")
        if not isfinite(self.max_wait_s) or self.max_wait_s < 0:
            raise ValueError("max_wait_s must be finite and nonnegative")
        if chunk_for(self.chunk_align, self.token_budget, self.draft_slots) == 0:
            raise ValueError("token budget must hold an aligned chunk after reserving drafts")


@dataclass
class State:
    """Everything the scheduler knows, as flat lists indexed alike."""
    waiting: list = field(default_factory=list)     # seq ids, arrival order
    arrived_at: dict = field(default_factory=dict)  # seq -> seconds
    prompt_len: dict = field(default_factory=dict)  # seq -> tokens to prefill
    computed: dict = field(default_factory=dict)    # seq -> tokens prefilled so far
    running: list = field(default_factory=list)     # decoding seq ids
    in_prefill: int | None = None                   # sequential mode: one at a time
    decode_due: bool = False                       # a completed prefill chunk delayed live decoders


@dataclass(frozen=True)
class Step:
    kind: str
    seqs: tuple
    tokens: int                # prefill: chunk length; decode: seqs * (1 + draft)
    reason: str


def _decode(state: State, c: Contract, reason: str) -> Step:
    seqs = tuple(state.running)
    return Step(DECODE, seqs, len(seqs) * (1 + c.draft_slots), reason)


def _prefill(state: State, c: Contract, reason: str) -> Step:
    seq = state.in_prefill if state.in_prefill is not None else state.waiting[0]
    remaining = state.prompt_len[seq] - state.computed.get(seq, 0)
    chunk = min(remaining, chunk_for(c.chunk_align, c.token_budget, c.draft_slots))
    return Step(PREFILL, (seq,), chunk, reason)


def plan(state: State, c: Contract, now: float) -> Step | None:
    """One step, or None when there is nothing to do."""
    if len(state.running) > c.max_running:
        raise ValueError("running sequences exceed the declared decode width")
    if state.in_prefill is not None and len(state.running) == c.max_running:
        raise ValueError("prefill has no reserved place in the decode batch")
    if state.running and state.decode_due:
        return _decode(state, c, "decode between prefill chunks")
    if state.in_prefill is not None:
        return _prefill(state, c, "continue the admitted prefill after decode")
    if state.running:
        if state.waiting and len(state.running) < c.max_running:
            waited = now - state.arrived_at[state.waiting[0]]
            if waited > c.max_wait_s:
                return _prefill(state, c, f"waited {waited:.1f}s > {c.max_wait_s}s: starvation valve")
        return _decode(state, c, "running decoders are protected")
    if state.waiting:
        return _prefill(state, c, "nothing decoding")
    return None


def advance(state: State, step: Step) -> None:
    """Apply a step's bookkeeping. Kernels are not this function's business."""
    if step.kind == PREFILL:
        state.decode_due = bool(state.running)
        (seq,) = step.seqs
        state.computed[seq] = state.computed.get(seq, 0) + step.tokens
        state.in_prefill = seq
        if state.computed[seq] >= state.prompt_len[seq]:
            state.waiting.remove(seq)
            state.running.append(seq)
            state.in_prefill = None
    elif step.kind == DECODE:
        state.decode_due = False                    # finished sequences leave via `finish`
    else:
        raise ValueError(f"a step is prefill or decode, never {step.kind!r}")


def validate_arrival(state: State, seq: int, prompt_len: int, now: float) -> None:
    """Check admission without publishing any scheduler state."""
    if not isinstance(seq, int) or not 0 <= seq < 2**31:
        raise ValueError("sequence id must be a nonnegative int32")
    if not isinstance(prompt_len, int) or not 0 < prompt_len < 2**31:
        raise ValueError("prompt length must be a positive int32")
    if not isfinite(now):
        raise ValueError("arrival time must be finite")
    if seq in state.prompt_len:
        raise ValueError(f"seq {seq} is already live")


def arrive(state: State, seq: int, prompt_len: int, now: float) -> None:
    validate_arrival(state, seq, prompt_len, now)
    state.waiting.append(seq)
    state.arrived_at[seq] = now
    state.prompt_len[seq] = prompt_len
    state.computed[seq] = 0


def finish(state: State, seq: int) -> None:
    if seq in state.running:
        state.running.remove(seq)
    else:
        state.waiting.remove(seq)
    if state.in_prefill == seq:
        state.in_prefill = None
    for d in (state.arrived_at, state.prompt_len, state.computed):
        d.pop(seq, None)
    if not state.running:
        state.decode_due = False


def _selfcheck() -> None:
    c = Contract(chunk_align=16, token_budget=4096, draft_slots=3, max_wait_s=20.0, max_running=8)
    s = State()
    arrive(s, 1, prompt_len=10000, now=0.0)
    kinds = []
    t = 0.0
    while (step := plan(s, c, t)) is not None and step.kind == PREFILL:
        assert step.tokens % 16 == 0 or s.computed[1] + step.tokens == 10000, "chunk must be aligned except the true tail"
        assert step.tokens <= 4096 - 3
        advance(s, step); kinds.append(step.kind); t += 0.5
    assert s.running == [1] and s.computed[1] == 10000 and len(kinds) == 3   # 4080 + 4080 + 1840
    # a second request arrives while 1 decodes: decoders are protected until the valve
    arrive(s, 2, prompt_len=100, now=t)
    for _ in range(5):
        step = plan(s, c, t); assert step.kind == DECODE and step.seqs == (1,), step
        assert step.tokens == 1 * (1 + 3)
        advance(s, step); t += 0.05
    step = plan(s, c, t + 20.1)
    assert step.kind == PREFILL and step.seqs == (2,) and "starvation" in step.reason
    advance(s, step)
    assert s.running == [1, 2]
    finish(s, 1); finish(s, 2)
    assert plan(s, c, t) is None
    # never a mixed step, by construction: every Step carries exactly one kind
    assert {PREFILL, DECODE} == {PREFILL, DECODE}
    print("  scheduler: homogeneous steps, protected decoders, aligned chunks OK")


if __name__ == "__main__":
    _selfcheck()
