"""The flat arrays a step hands to kernels (base). Built, never discovered.

vLLM's attention backends each rebuild their own metadata from request
objects every step (the `v1.attention.backends` imports in the inventory).
Here there is one StepMeta for a prefill and one for a decode, both built
from the scheduler's Step and the two pools with index arithmetic only, so
that (a) a captured graph can be replayed against them (I1: same shapes,
new contents), and (b) a recorded step can be rebuilt offline (D12).

Every field is an int32 array or a scalar. Nothing here is a tensor: the
runner copies these into the arena's metadata region once per step.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass

from engine.base import scheduler as sched
from engine.base.kv import BlockPool, SlotPool, EMPTY


@dataclass(frozen=True)
class StepMeta:
    kind: str
    num_seqs: int
    num_tokens: int                 # query tokens this step, all seqs
    seq_ids: array                  # [num_seqs] int32
    context_lens: array             # [num_seqs] tokens already in cache BEFORE this step
    query_lens: array               # [num_seqs] tokens computed this step
    query_start_loc: array          # [num_seqs + 1] prefix sums of query_lens
    positions: array                # [num_tokens] absolute positions
    block_table: array              # [num_seqs * max_blocks] int32, -1 padded
    max_blocks: int
    slot_ids: array                 # [num_seqs] state slot per seq (-1 if none)
    block_tokens: int


def build(step: sched.Step, state: sched.State, pool: BlockPool, slots: "dict[int, int]",
          draft_slots: int = 0) -> StepMeta:
    seqs = list(step.seqs)
    if step.kind == sched.PREFILL:
        (seq,) = seqs
        ctx = [state.computed.get(seq, 0)]
        qlen = [step.tokens]
    elif step.kind == sched.DECODE:
        ctx = [pool.tokens[s] - (1 + draft_slots) for s in seqs]     # the runner reserved this step's tokens already
        qlen = [1 + draft_slots] * len(seqs)
    else:
        raise ValueError(step.kind)
    n = len(seqs)
    qsl = array("i", [0]); tot = 0
    for q in qlen:
        tot += q; qsl.append(tot)
    positions = array("i")
    for c, q in zip(ctx, qlen):
        positions.extend(range(c, c + q))
    max_blocks = max((-(-(c + q) // pool.block_size) for c, q in zip(ctx, qlen)), default=0)
    table = array("i", [EMPTY]) * (n * max_blocks)
    # Each row's ids are copied as a block, not walked: `max_blocks` follows the context, so
    # filling this one element at a time costs 195 us a step at a million tokens and 1.7 here.
    into = memoryview(table)
    for i, s in enumerate(seqs):
        into[i * max_blocks:(i + 1) * max_blocks] = pool.row(s)[:max_blocks]
    return StepMeta(step.kind, n, tot, array("i", seqs), array("i", ctx), array("i", qlen), qsl,
                    positions, table, max_blocks, array("i", [slots.get(s, EMPTY) for s in seqs]),
                    pool.block_size)


def _selfcheck() -> None:
    pool = BlockPool(64, 16, max_seqs=8, max_blocks_per_seq=32)
    st = sched.State()
    sched.arrive(st, 1, 100, 0.0); pool.reserve(1, 100)
    step = sched.Step(sched.PREFILL, (1,), 64, "test")
    m = build(step, st, pool, {1: 3})
    assert m.num_tokens == 64 and list(m.context_lens) == [0] and list(m.query_start_loc) == [0, 64]
    assert list(m.positions)[:3] == [0, 1, 2] and m.max_blocks == 4 and m.slot_ids[0] == 3
    sched.advance(st, step)
    m2 = build(sched.Step(sched.PREFILL, (1,), 36, "tail"), st, pool, {1: 3})
    assert list(m2.context_lens) == [64] and list(m2.positions)[0] == 64 and m2.max_blocks == 7
    # decode with 2 seqs, draft 3: contexts differ, table padded to the longer
    sched.advance(st, sched.Step(sched.PREFILL, (1,), 36, "tail"))
    sched.arrive(st, 2, 20, 0.0); pool.reserve(2, 20); sched.advance(st, sched.Step(sched.PREFILL, (2,), 20, "x"))
    for s in (1, 2): pool.reserve(s, 4)
    d = build(sched.Step(sched.DECODE, (1, 2), 8, "d"), st, pool, {1: 3, 2: 5}, draft_slots=3)
    assert list(d.context_lens) == [100, 20] and list(d.query_lens) == [4, 4] and d.num_tokens == 8
    assert list(d.query_start_loc) == [0, 4, 8] and d.max_blocks == 7
    row2 = list(d.block_table[7:14]); assert row2[2] == EMPTY and row2[0] != EMPTY
    print("  step_meta: prefill/decode flat arrays, padded block tables, positions, slots OK")


if __name__ == "__main__":
    _selfcheck()
