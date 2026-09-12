"""CPU regressions for the ST engine's scheduling and memory ownership.

Run with: python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
"""
from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

from engine.base import scheduler as sched
from engine.base.kv import BlockPool, SlotPool
from engine.base.record import Ring
from engine.base.runner import Runner, STEP_RECORD
from engine.base.step_meta import build


CONTRACT = sched.Contract(16, 64, 0, 20.0, 2)


class Model:
    def __init__(self, drafts=0):
        self.live = set()
        self.calls = []
        self.fail_open = False
        self.done = set()
        self.ctx = {}
        self.drafts = drafts

    def open(self, seq, slot):
        self.live.add(seq)
        if self.fail_open:
            raise RuntimeError("open failed")

    def close(self, seq):
        self.live.discard(seq)

    def horizon(self, seq):
        return self.ctx[seq] + 1 + self.drafts

    def prefill(self, seq, start, tokens, blocks, slot):
        self.calls.append(("prefill", seq, start, tokens))
        self.ctx[seq] = start + tokens

    def decode(self, seqs, blocks, slots):
        self.calls.append(("decode", tuple(seqs)))
        for seq in seqs:
            self.ctx[seq] += 1
        return [s in self.done for s in seqs]


def runner(*, blocks=16, slots=5, contract=CONTRACT):
    return Runner(Model(contract.draft_slots), contract, BlockPool(blocks, 16, 4, 16),
                  SlotPool(slots), Ring(16, STEP_RECORD.size))


def pool_state(pool):
    return (list(pool.tokens), list(pool.table), sorted(pool.free_order()), pool.rows_in_use)


class SchedulingTests(unittest.TestCase):
    def test_invalid_contracts_fail_before_planning(self):
        for field, value in [("chunk_align", 0), ("chunk_align", -1),
                             ("token_budget", 0), ("token_budget", 15),
                             ("draft_slots", -1), ("draft_slots", 64),
                             ("max_running", 0), ("max_wait_s", -1),
                             ("max_wait_s", float("nan")),
                             ("max_wait_s", float("inf"))]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(CONTRACT, **{field: value})

    def test_arrival_rejects_invalid_or_duplicate_requests_without_mutation(self):
        state = sched.State()
        sched.arrive(state, 1, 24, 0.0)
        before = copy.deepcopy(state)
        for seq, length in [(1, 24), (2, 0), (2, -1), (-1, 24)]:
            with self.subTest(seq=seq, length=length), self.assertRaises(ValueError):
                sched.arrive(state, seq, length, 1.0)
            self.assertEqual(state, before)

    def test_expired_waiter_does_not_overfill_decode_width(self):
        r = runner(contract=replace(CONTRACT, max_running=1))
        r.submit(0, 16, now=0)
        r.submit(1, 16, now=0)
        r.step(now=0)
        step = r.step(now=21)
        self.assertEqual((step.kind, step.seqs), (sched.DECODE, (0,)))
        self.assertEqual(r.state.running, [0])
        r.model.done.add(0)
        r.step(now=22)
        step = r.step(now=23)
        self.assertEqual((step.kind, step.seqs), (sched.PREFILL, (1,)))

    def test_starvation_valve_still_works_when_decode_has_room(self):
        r = runner()
        r.submit(0, 16, now=0)
        r.submit(1, 16, now=0)
        self.assertEqual(r.step(now=0).kind, sched.PREFILL)
        self.assertEqual(r.step(now=20).kind, sched.DECODE)
        self.assertEqual(r.step(now=21).kind, sched.PREFILL)
        self.assertEqual(r.step(now=22).seqs, (0, 1))

    def test_invalid_running_width_is_not_silently_truncated(self):
        state = sched.State(running=[0, 1, 2])
        with self.assertRaises(ValueError):
            sched.plan(state, CONTRACT, 0)

    def test_long_waiter_yields_between_chunks_without_losing_progress(self):
        r = runner(blocks=32)
        r.submit(0, 16, now=0)
        r.submit(1, 178, now=0)
        r.step(now=0)
        steps = [r.step(now=20 + i) for i in range(7)]
        self.assertEqual([s.kind for s in steps],
                         [sched.DECODE, sched.PREFILL, sched.DECODE,
                          sched.PREFILL, sched.DECODE, sched.PREFILL, sched.DECODE])
        self.assertEqual([c for c in r.model.calls if c[:2] == ("prefill", 1)],
                         [("prefill", 1, 0, 64), ("prefill", 1, 64, 64), ("prefill", 1, 128, 50)])
        self.assertEqual(steps[-1].seqs, (0, 1))

    def test_prefill_continues_without_yields_after_last_decoder_finishes(self):
        r = runner(blocks=32)
        r.submit(0, 16, now=0)
        r.submit(1, 178, now=0)
        r.step(now=0)
        r.step(now=20)
        r.step(now=21)
        r.model.done.add(0)
        self.assertEqual(r.step(now=23).kind, sched.DECODE)
        self.assertEqual(r.step(now=24).kind, sched.PREFILL)
        self.assertEqual(r.step(now=25).kind, sched.PREFILL)
        self.assertEqual(r.state.running, [1])

    def test_cancelling_partial_prefill_preserves_decoder_and_fifo_waiter(self):
        r = runner(blocks=32)
        for seq, length in [(0, 16), (1, 178), (2, 32)]:
            r.submit(seq, length, now=0)
        r.step(now=0); r.step(now=20); r.step(now=21)
        r.cancel(1)
        self.assertEqual(r.step(now=23).seqs, (0,))
        self.assertEqual(r.step(now=24).seqs, (2,))
        self.assertEqual(r.state.running, [0, 2])

    def test_prefill_tail_stays_inside_budget(self):
        c = replace(CONTRACT, draft_slots=3)
        s = sched.State()
        sched.arrive(s, 0, 130, 0)
        sizes = []
        while (step := sched.plan(s, c, 0)).kind == sched.PREFILL:
            sizes.append(step.tokens)
            sched.advance(s, step)
        self.assertEqual(sizes, [48, 48, 34])

    def test_wake_cannot_take_the_paused_prefills_reserved_decode_place(self):
        r = runner(blocks=32)
        r.keep_idle = True
        r.submit(2, 16, now=0)
        r.model.done.add(2)
        r.step(now=0); r.step(now=1)
        r.submit(0, 16, now=2); r.submit(1, 178, now=2)
        r.step(now=2); r.step(now=3); r.step(now=23)
        before = copy.deepcopy(r.state)
        with self.assertRaisesRegex(ValueError, "decode width"):
            r.wake(2)
        self.assertEqual(before, r.state)
        self.assertIn(2, r.idle)


class PoolTests(unittest.TestCase):
    def test_mapping_appends_within_epoch_and_release_invalidates_reuse(self):
        pool = BlockPool(4, 16, 2, 4)
        pool.reserve(0, 1)
        first = pool.row(0)[0]
        epoch = pool.epochs[0]
        pool.reserve_to([0], [16])
        self.assertEqual(pool.epochs[0], epoch)
        pool.reserve_to([0], [17])
        self.assertEqual(pool.epochs[0], epoch)
        self.assertEqual(pool.row(0)[0], first)
        before = (pool_state(pool), list(pool.epochs))
        with self.assertRaises(MemoryError):
            pool.reserve_to([0, 1], [64, 64])
        self.assertEqual((pool_state(pool), list(pool.epochs)), before)
        pool.release(0)
        self.assertGreater(pool.epochs[0], epoch)
        epoch = pool.epochs[0]
        pool.release(0)                        # releasing an empty row changes no mapping
        self.assertEqual(pool.epochs[0], epoch)
        pool.reserve(1, 16)
        pool.reserve(0, 17)
        self.assertEqual(pool.epochs[0], epoch)
        self.assertNotEqual(pool.row(0)[1], first)

    def test_mapping_and_epoch_views_reject_untracked_writes(self):
        pool = BlockPool(4, 16, 2, 4)
        pool.reserve(0, 17)
        before = pool_state(pool)
        for view in (pool.table, pool.row(0), pool.epochs):
            with self.subTest(view=view.format), self.assertRaises(TypeError):
                view[0] = 99
        self.assertEqual(pool_state(pool), before)

    def test_absolute_horizons_reuse_draft_space_and_reserve_atomically(self):
        pool = BlockPool(3, 16, 2, 3)
        pool.reserve_many([0, 1], 16)
        before = pool_state(pool)
        with self.assertRaises(MemoryError):
            pool.reserve_to([0, 1], [20, 40])
        self.assertEqual(pool_state(pool), before)
        pool.reserve_to([0, 1], [20, 16])
        before = pool_state(pool)
        self.assertEqual(pool.reserve_to([0, 1], [18, 12]), 0)
        self.assertEqual(pool_state(pool), before)

    def test_batch_validates_later_rows_before_reserving_earlier_rows(self):
        for seqs, tokens, error in [((0, 0), 1, ValueError),
                                   ((0, 4), 1, IndexError),
                                   ((0, 1), 2**31, ValueError)]:
            p = BlockPool(8, 16, 4, 4)
            before = pool_state(p)
            with self.subTest(seqs=seqs, tokens=tokens), self.assertRaises(error):
                p.reserve_many(seqs, tokens)
            self.assertEqual(pool_state(p), before)
        p.reserve(1, 64)
        before = pool_state(p)
        with self.assertRaises(MemoryError):
            p.reserve_many((0, 1), 1)
        self.assertEqual(pool_state(p), before)

    def test_sequence_cannot_lose_its_slot_through_duplicate_allocation(self):
        p = SlotPool(3)
        slot = p.take(7)
        with self.assertRaises(ValueError):
            p.take(7)
        self.assertEqual(p.available, 1)
        self.assertEqual(p.owner[slot], 7)

    def test_invalid_rows_never_alias_another_sequence(self):
        p = BlockPool(8, 16, 4, 4)
        p.reserve(3, 17)
        before = pool_state(p)
        for method in (p.row, p.release, p.blocks_of):
            for seq in (-1, 4):
                with self.subTest(method=method.__name__, seq=seq), self.assertRaises(IndexError):
                    method(seq)
                self.assertEqual(pool_state(p), before)

    def test_negative_growth_does_not_corrupt_the_next_reservation(self):
        p = BlockPool(8, 16, 4, 4)
        p.reserve(0, 17)
        before = pool_state(p)
        with self.assertRaises(ValueError):
            p.reserve(0, -2)
        self.assertEqual(pool_state(p), before)
        p.reserve(0, 16)
        self.assertEqual(p.release(0), 3)
        self.assertEqual(p.available, 8)

    def test_pool_dimensions_must_be_positive(self):
        for seqs, blocks in [(0, 1), (1, 0), (-1, 1), (1, -1)]:
            with self.subTest(seqs=seqs, blocks=blocks), self.assertRaises(ValueError):
                BlockPool(8, 16, seqs, blocks)

    def test_slot_return_rejects_negative_alias(self):
        p = SlotPool(2)
        slot = p.take(7)
        with self.assertRaises((IndexError, ValueError)):
            p.give(-1)
        self.assertEqual(p.owner[slot], 7)
        self.assertEqual(p.available, 0)

    def test_slot_rejects_the_empty_owner_sentinel(self):
        p = SlotPool(2)
        with self.assertRaises(ValueError):
            p.take(-1)
        self.assertEqual(p.available, 1)


class RunnerTests(unittest.TestCase):
    def assert_empty(self, r):
        self.assertEqual(r.state, sched.State())
        self.assertEqual(r.kv.available, r.kv.num_blocks)
        self.assertEqual(r.kv.rows_in_use, 0)
        self.assertEqual(r.slot_of, {})
        self.assertEqual(r.slots.available, r.slots.num_slots - 1)
        self.assertEqual(r.model.live, set())

    def test_block_exhaustion_does_not_enqueue_a_request(self):
        r = runner(blocks=1)
        with self.assertRaises(MemoryError):
            r.submit(0, 17, now=0)
        self.assert_empty(r)
        r.submit(0, 16, now=0)
        self.assertEqual(r.step(now=0).kind, sched.PREFILL)

    def test_first_prefill_sample_can_finish_without_a_decode_step(self):
        r = runner()
        r.model.prefill = lambda *args: True
        r.submit(0, 16, now=0)
        self.assertEqual(r.step(now=0).kind, sched.PREFILL)
        self.assertIsNone(r.step(now=1))
        self.assert_empty(r)

    def test_model_cannot_finish_before_its_prompt_tail(self):
        r = runner()
        r.model.prefill = lambda *args: True
        r.submit(0, 100, now=0)
        with self.assertRaises(ValueError):
            r.step(now=0)
        self.assertEqual(r.state.computed[0], 0)

    def test_slot_exhaustion_returns_newly_reserved_blocks(self):
        r = runner(slots=2)
        r.submit(0, 16, now=0)
        state, pool = copy.deepcopy(r.state), pool_state(r.kv)
        with self.assertRaises(MemoryError):
            r.submit(1, 16, now=0)
        self.assertEqual(r.state, state)
        self.assertEqual(pool_state(r.kv), pool)
        self.assertEqual(r.slot_of, {0: 1})
        self.assertEqual(r.model.live, {0})

    def test_failed_model_open_is_closed_and_admission_can_retry(self):
        r = runner()
        r.model.fail_open = True
        with self.assertRaisesRegex(RuntimeError, "open failed"):
            r.submit(0, 16, now=0)
        self.assert_empty(r)
        r.model.fail_open = False
        r.submit(0, 16, now=0)
        self.assertEqual(r.step(now=0).seqs, (0,))

    def test_duplicate_and_invalid_submissions_leave_live_resources_untouched(self):
        r = runner()
        r.submit(0, 16, now=0)
        state, pool, owners = copy.deepcopy(r.state), pool_state(r.kv), list(r.slots.owner)
        for seq, length, error in [(0, 16, ValueError), (1, 0, ValueError),
                                   (1, -1, ValueError), (4, 16, IndexError)]:
            with self.subTest(seq=seq, length=length), self.assertRaises(error):
                r.submit(seq, length, now=0)
            self.assertEqual(r.state, state)
            self.assertEqual(pool_state(r.kv), pool)
            self.assertEqual(list(r.slots.owner), owners)

    def test_batch_exhaustion_reserves_none_of_the_decode_rows(self):
        r = runner(blocks=3)
        r.submit(0, 16, now=0)
        r.submit(1, 16, now=0)
        r.step(now=0)
        r.step(now=21)
        before = pool_state(r.kv)
        with self.assertRaises(MemoryError):
            r.step(now=22)
        self.assertEqual(pool_state(r.kv), before)
        self.assertEqual([c[0] for c in r.model.calls], ["prefill", "prefill"])
        self.assertEqual(r.ring.count, 2)

    def test_decode_result_must_cover_every_sequence(self):
        r = runner()
        r.submit(0, 16, now=0)
        r.step(now=0)
        r.model.decode = lambda *args: []
        with self.assertRaises(ValueError):
            r.step(now=1)
        self.assertEqual(r.state.running, [0])
        self.assertEqual(r.ring.count, 1)

    def test_decode_failure_retries_reuse_the_same_reserved_horizon(self):
        for bad in (lambda *a: [], lambda *a: (_ for _ in ()).throw(RuntimeError('decode failed'))):
            r = runner(blocks=2)
            r.submit(0, 16, now=0)
            r.step(now=0)
            r.model.decode = bad
            for tick in range(10):
                with self.assertRaises((RuntimeError, ValueError)):
                    r.step(now=tick + 1)
                self.assertEqual(r.kv.tokens[0], 17)
                self.assertEqual(r.kv.available, 0)
            r.cancel(0)
            self.assert_empty(r)

    def test_metadata_uses_committed_context_after_batch_reservation(self):
        r = runner(contract=replace(CONTRACT, draft_slots=3))
        r.submit(0, 16, now=0)
        r.submit(1, 20, now=0)
        r.step(now=0)
        r.step(now=21)
        metas = []

        def decode(seqs, blocks, slots):
            step = sched.plan(r.state, r.c, 22)
            metas.append(build(step, r.state, r.kv, r.slot_of, draft_slots=3))
            return [True] * len(seqs)

        r.model.decode = decode
        r.step(now=22)
        self.assertEqual(list(metas[0].context_lens), [16, 20])
        self.assertEqual(list(metas[0].positions), [16, 17, 18, 19, 20, 21, 22, 23])
        self.assert_empty(r)

    def test_repeated_requests_reuse_resources_and_keep_instrumentation_bounded(self):
        r = runner()
        r.model.done.add(0)
        with patch("engine.base.instruments._dev_free_bytes", return_value=None):
            for i in range(500):
                r.submit(0, 16, now=i)
                r.step(now=i)
                r.step(now=i)
        self.assert_empty(r)
        self.assertEqual(r.steps, 1000)
        self.assertEqual(len(r.ring.ordered()), 16)
        self.assertEqual(len(r.rec.root.children), 2)
        self.assertEqual([span.calls for span in r.rec.root.children], [500, 500])
        self.assertEqual(r.rec.root.counters["decode_steps"], 500)
        self.assertTrue(all(span.seconds > 0 for span in r.rec.root.children))

    def test_failed_open_cleanup_still_returns_pool_resources_if_close_raises(self):
        r = runner()
        r.model.fail_open = True

        def close(seq):
            r.model.live.discard(seq)
            raise RuntimeError("close failed")

        r.model.close = close
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            r.submit(0, 16, now=0)
        self.assert_empty(r)


if __name__ == "__main__":
    unittest.main()
