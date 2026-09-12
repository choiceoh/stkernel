"""The decode pipeline's pure parts (45차 §23 B3): the batch distribution and the host readback bookkeeping."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from engine.profiles.glm53.pipeline import AsyncDecode, Pending, distribution_batch  # noqa: E402


class DistributionBatchTests(unittest.TestCase):
    def test_greedy_rows_are_one_hot_and_nucleus_rows_keep_the_smallest_prefix(self):
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.0], [1.0, 3.0, 2.0, 0.0], [1.0, 3.0, 2.0, 0.0]])
        temps = torch.tensor([0.0, 1.0, 1.0])
        top_k = torch.zeros(3, dtype=torch.int32)
        top_p = torch.tensor([1.0, 1.0, 0.5])
        # no `nucleus` argument: each row carries its own truncation, so there is no host predicate
        probs = distribution_batch(logits, temps, top_k, top_p)
        self.assertEqual(probs[0].tolist(), [0.0, 1.0, 0.0, 0.0])
        torch.testing.assert_close(probs[1], torch.softmax(logits[1], -1))
        self.assertEqual(probs[2].tolist(), [0.0, 1.0, 0.0, 0.0])                     # the top token alone reaches 0.5
        torch.testing.assert_close(probs.sum(1), torch.ones(3))
        plain = distribution_batch(logits, temps, top_k, torch.ones(3))
        torch.testing.assert_close(plain[2], torch.softmax(logits[2], -1))

    def test_a_row_that_asks_for_top_k_gets_it_without_leaving_this_path(self):
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]])
        probs = distribution_batch(logits, torch.ones(2), torch.tensor([0, 2], dtype=torch.int32), torch.ones(2))
        torch.testing.assert_close(probs[0], torch.softmax(logits[0], -1))
        self.assertEqual(probs[1][2:].tolist(), [0.0, 0.0])
        torch.testing.assert_close(probs[1][:2], torch.softmax(logits[1][:2], -1))


class ShrinkTests(unittest.TestCase):
    """A row finishing does not move the others: every device tensor is re-indexed, on the device.

    What this used to also check -- the remapping of which positions wanted a nucleus -- went away
    with the sort that list chose the payers for (45차 §34). The re-indexing itself did not."""

    def test_every_row_array_follows_the_surviving_rows(self):
        e = SimpleNamespace(drafter=SimpleNamespace(k=2),
                            caches=SimpleNamespace(pool=SimpleNamespace(max_seqs=4), device=torch.device("cpu")),
                            F=SimpleNamespace(block=2))
        p = AsyncDecode(e)
        t, n = p.t, 3
        b = {name: torch.arange(n) for name in
             ("seqs", "real_slot", "slot", "ctx", "generated", "limit", "temps", "top_k", "top_p", "anchor")}
        b.update(ends=torch.zeros(n, 1, dtype=torch.int64), alive=torch.ones(n, dtype=torch.bool),
                 drafts=torch.zeros(n, 2, dtype=torch.int64), ids=torch.arange(n * t), qcand=None, qprob=None)
        p.buf, p.batch = b, (1, 2, 3)
        p._shrink([3, 1])                                                     # old row 2 -> new 0, old row 0 -> new 1
        self.assertEqual(b["seqs"].tolist(), [2, 0])
        self.assertEqual(b["top_k"].tolist(), [2, 0], "a row's truncation follows the row")
        self.assertEqual(b["top_p"].tolist(), [2, 0])
        self.assertEqual(b["ids"].tolist(), list(range(2 * t, 3 * t)) + list(range(t)))
        self.assertEqual(p.batch, (3, 1))


class ResolveTests(unittest.TestCase):
    def test_resolve_applies_counts_in_launch_order_and_ignores_released_rows(self):
        e = SimpleNamespace(drafter=SimpleNamespace(k=2), caches=SimpleNamespace(pool=SimpleNamespace(max_seqs=4), device=torch.device("cpu")),
                            tokens={1: [5], 2: [6]}, ctx={1: 1, 2: 1}, inflight={1: 1, 2: 1}, accepted_total=0, drafted_total=0, steps=0,
                            F=SimpleNamespace(block=2), staged={})
        p = AsyncDecode(e)
        lane = p.free.pop(0)
        p.host[lane]["tokens"][:2] = torch.tensor([[7, 8, 9], [1, 2, 3]])
        p.host[lane]["count"][:2] = torch.tensor([2, 3])
        p.host[lane]["done"][:2] = torch.tensor([False, True])
        p.host[lane]["accepted"][:2] = torch.tensor([1, 2])
        first = Pending(p, lane, [1, 2], None)
        p.pending.append(first)
        second = Pending(p, 0, [1], None)
        p.pending.append(second)
        with self.assertRaises(RuntimeError):
            p.resolve(second)                                                         # launch order
        del e.tokens[2]                                                               # released meanwhile: nothing applied
        self.assertEqual(first.resolve(), [False, True])
        self.assertEqual(e.tokens[1], [5, 7, 8])
        self.assertEqual((e.ctx[1], e.inflight[1], e.inflight[2]), (3, 0, 0))
        self.assertEqual(e.staged, {1: 2})                                            # 1 -> 3 crossed the block boundary at 2
        self.assertEqual((e.accepted_total, e.drafted_total, e.steps), (1, 2, 1))
        self.assertIn(lane, p.free)


class StageClockTests(unittest.TestCase):
    """The decode step's stage meter. It must never make the step wait, and must not measure every step."""

    def clock(self, every=4):
        from engine.base.stage_clock import StageClock
        c = StageClock(every=every, device=None)
        c._torch = FakeTorch()                       # a device without a device: the sampling logic is the point
        return c

    def test_it_measures_one_step_in_every(self):
        c = self.clock(every=4)
        self.assertEqual([c.step() for _ in range(8)], [False, False, False, True] * 2)

    def test_an_unsampled_step_records_nothing_and_costs_nothing(self):
        c = self.clock(every=4)
        c.step()
        with c.mark("forward"):
            pass
        self.assertEqual(c._pending, [], "a step that is not sampled must not create events")

    def test_a_sampled_step_is_read_back_a_round_later_not_now(self):
        """Reading an event on the step that recorded it would synchronise. The whole design is that it does not."""
        c = self.clock(every=1)
        c.step()
        with c.mark("forward"):
            pass
        with c.mark("verify"):
            pass
        self.assertEqual(c.totals, {}, "nothing is read on the step that recorded it")
        self.assertEqual(len(c._pending), 2)
        c.step()                                      # the next round drains the previous one
        self.assertEqual(sorted(c.totals), ["forward", "verify"])
        self.assertEqual(c.samples, 1)

    def test_an_unfinished_round_is_left_alone_rather_than_waited_on(self):
        c = self.clock(every=1)
        c.step()
        with c.mark("forward"):
            pass
        c._pending[0][2].done = False                 # still running on the device
        c.step()
        self.assertEqual(c.totals, {}, "it must not block; the round waits for the one after")

    def test_both_ends_of_a_span_are_recorded(self):
        """A start that is never recorded, or an end recorded outside the span, times something else entirely."""
        c = self.clock(every=1)
        c.step()
        with c.mark("forward"):
            start, end = c._pending[0][1], c._pending[0][2]
            self.assertEqual((start.records, end.records), (1, 0), "start at entry, end not yet")
        self.assertEqual((start.records, end.records), (1, 1), "and the end at exit")

    def test_a_drained_round_is_not_counted_twice(self):
        c = self.clock(every=1)
        c.step()
        with c.mark("forward"):
            pass
        c.step(); once = dict(c.totals)
        c._drain()
        self.assertEqual(c.totals, once, "draining again must not add the same events a second time")

    def test_marking_before_the_first_step_is_inert_rather_than_an_error(self):
        from engine.base.stage_clock import StageClock
        c = StageClock(device=None)
        with c.mark("forward"):
            pass

    def test_a_cuda_device_arms_it_and_anything_else_does_not(self):
        from types import SimpleNamespace
        from engine.base.stage_clock import StageClock
        self.assertIsNone(StageClock(device=SimpleNamespace(type="cpu"))._torch)
        self.assertIsNone(StageClock(device=None)._torch)
        armed = StageClock(device=SimpleNamespace(type="cuda"))
        self.assertIsNotNone(armed._torch, "a cuda device must bring torch in, or nothing is ever measured")

    def test_shares_are_the_shape_of_a_step(self):
        c = self.clock()
        c.totals = {"forward": 9.0, "verify": 1.0}
        self.assertEqual(c.shares(), {"forward": 0.9, "verify": 0.1})
        self.assertEqual(self.clock().shares(), {}, "no samples yet is not a division by zero")

    def test_without_a_cuda_device_it_is_inert(self):
        from engine.base.stage_clock import StageClock
        c = StageClock(device=None)
        self.assertFalse(any(c.step() for _ in range(200)))
        with c.mark("forward"):
            pass
        self.assertEqual((c.totals, c._pending), ({}, []))


class FakeEvent:
    def __init__(self, **kw):
        self.done = True
        self.records = 0

    def record(self):
        self.records += 1

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return 2.0


class FakeTorch:
    class cuda:
        Event = FakeEvent


class BatchTransitionTests(unittest.TestCase):
    """Run the real launch/commit/readback chain; only model kernels are replaced."""
    def engine(self):
        e = SimpleNamespace(
            drafter=SimpleNamespace(k=1, F=SimpleNamespace(sel_top_k=4), decode_graphs=None,
                                    propose_rows=lambda field, slots, anchors, ctx, alive=None: torch.full((len(slots), 1), 7)),
            caches=SimpleNamespace(pool=SimpleNamespace(max_seqs=4), device=torch.device('cpu'),
                                   draft_field=lambda: torch.zeros(1), stage_boundaries=lambda *args: None),
            F=SimpleNamespace(vocab=32, block=16), tokens={1: [5], 2: [6]}, ctx={1: 1, 2: 1},
            limits={1: (10, 0.0), 2: (10, 0.0)}, options={}, ends={}, eos={31}, top_p=1.0,
            inflight={}, staged={}, accepted_total=0, drafted_total=0, steps=0,
            gen=torch.Generator().manual_seed(1))
        e._generated_count = lambda seq: len(e.tokens[seq]) - 1
        e.decode_graphs = SimpleNamespace(shape_for=lambda n, end: (n, 2, 64),
                                         run_device=lambda *args: (None, None, None))
        e.sampling_graphs = SimpleNamespace(greedy=SimpleNamespace(
            run=lambda shape, fill: torch.tensor([7, 8] * shape[0])))
        return e

    def test_first_sampled_request_uses_the_real_drafter_candidate_width(self):
        from tests.test_engine_glm53 import tiny_facts
        e = self.engine()
        e.F = tiny_facts()  # target Facts has no drafter selector_top_k field
        e.drafter.F = SimpleNamespace(sel_top_k=16)
        e.limits[1] = (10, .7)
        candidates = torch.arange(16).view(1, 1, 16)
        probabilities = torch.full((1, 1, 16), 1/16)
        e.drafter.propose_rows = lambda *args, **kwargs: (torch.tensor([[9]]), candidates, probabilities)
        p = AsyncDecode(e)
        p._build([1], [1])
        torch.testing.assert_close(p.buf['qcand'], candidates)
        torch.testing.assert_close(p.buf['qprob'], probabilities)

    def test_new_request_rebuilds_after_the_previous_batch_has_drained(self):
        for stale in (False, True):
            with self.subTest(stale=stale):
                e = self.engine()
                p = AsyncDecode(e)
                p.launch([1], [1]).resolve()
                self.assertEqual(e.tokens[1], [5, 7, 8])
                p.stale = stale
                e.ctx[2] = 20                             # a new prefill's host position
                p.launch([2], [3]).resolve()
                self.assertEqual((p.batch, p.buf['real_slot'].tolist()), ((2,), [3]))
                self.assertEqual(e.ctx[2], 22)
                self.assertEqual(e.tokens[2], [6, 7, 8])
                self.assertEqual(p.pending, [])

    def test_joining_keeps_survivors_ahead_of_the_host(self):
        e = self.engine()
        p = AsyncDecode(e)
        first = p.launch([1], [1])
        second = p.launch([2, 1], [2, 1])
        self.assertEqual(e.ctx, {1: 1, 2: 1})
        self.assertEqual(p.buf['ctx'].tolist(), [3, 5])
        self.assertEqual(p.batch, (2, 1))
        first.resolve()
        second.resolve()
        self.assertEqual(e.ctx, {1: 5, 2: 3})
        self.assertEqual(e.tokens[1], [5, 7, 8, 7, 8])

    def test_invalidating_an_outstanding_row_requires_a_drain(self):
        e = self.engine()
        p = AsyncDecode(e)
        first = p.launch([1], [1])
        p.stale = True
        with self.assertRaisesRegex(RuntimeError, 'must drain first'):
            p.launch([1], [1])
        first.resolve()
        p.launch([1, 2], [1, 2]).resolve()
        self.assertEqual(e.ctx, {1: 5, 2: 3})

    def test_prefill_invalidates_only_its_row_and_keeps_survivor_proposals(self):
        e = self.engine()
        proposed = []
        def propose(field, slots, anchors, ctx, alive=None):
            proposed.append(slots.tolist())
            return torch.full((len(slots), 1), 7)
        e.drafter.propose_rows = propose
        p = AsyncDecode(e)
        p.launch([1], [1]).resolve()
        proposed.clear()
        e.ctx[2] = 25
        p.invalidate([2])
        p.launch([1, 2], [1, 2]).resolve()
        self.assertEqual(proposed, [[2], [1, 2]])
        self.assertEqual(e.ctx, {1: 5, 2: 27})
        self.assertEqual(p.buf['generated'].tolist(), [4, 2])

    def test_replacing_an_inflight_slot_is_refused_and_rebuilds_after_drain(self):
        e = self.engine()
        p = AsyncDecode(e)
        first = p.launch([1], [1])
        with self.assertRaisesRegex(RuntimeError, 'must drain first'):
            p.launch([1], [3])
        first.resolve()
        p.launch([1], [3]).resolve()
        self.assertEqual(p.buf['real_slot'].tolist(), [3])

    def test_mixed_join_preserves_greedy_drafts_and_pads_end_token_sets(self):
        e = self.engine()
        p = AsyncDecode(e)
        p._build([1], [1])
        e.limits[2] = (10, 0.7)
        e.ends[2] = {29, 30, 31}
        def sampled(field, slots, anchors, ctx, **kwargs):
            ids = torch.full((len(slots), 1), 9)
            cand = ids.unsqueeze(2).expand(len(slots), 1, 4).contiguous()
            q = torch.zeros(len(slots), 1, 4); q[..., 0] = 1.0
            return ids, cand, q
        e.drafter.propose_rows = sampled
        p._merge([2, 1], [2, 1])
        self.assertEqual(p.buf['drafts'].tolist(), [[9], [7]])
        self.assertEqual(p.buf['qcand'][..., 0].tolist(), [[9], [7]])     # the candidate carrying the mass
        # the greedy survivor becomes a point mass: ALL of it on that one candidate, or the accept test divides
        # the target by a draft that does not sum to one and the verification stops being unbiased
        self.assertEqual(p.buf['qprob'][1].tolist(), [[1.0, 0.0, 0.0, 0.0]])
        self.assertEqual(p.buf['qprob'][0].tolist(), [[1.0, 0.0, 0.0, 0.0]])
        self.assertEqual(float(p.buf['qprob'].sum()), 2.0)
        self.assertEqual(p.buf['ends'].tolist(), [[29, 30, 31], [31, -1, -1]])
        self.assertEqual(p.buf['temps'].tolist(), [torch.tensor(.7).item(), 0.])
        self.assertEqual(p.buf['ids'].tolist(), [6, 9, 5, 7])
        p._merge([1], [1])
        self.assertFalse(p.buf['stochastic'])
        self.assertIsNone(p.buf['qprob'])
        self.assertEqual(p.buf['ids'].tolist(), [5, 7])

    def test_a_sampled_survivor_keeps_its_own_draft_when_a_row_joins(self):
        """`_merge` only invents a point mass for the side that has none. A survivor that was already sampling
        carries a real distribution, and overwriting it would throw away the draws its drafts came from."""
        e = self.engine()
        p = AsyncDecode(e)
        e.limits[1] = (10, 0.9)
        def sampled(field, slots, anchors, ctx, **kwargs):
            ids = torch.full((len(slots), 1), 9)
            cand = torch.arange(4).view(1, 1, 4).expand(len(slots), 1, 4).contiguous()
            q = torch.full((len(slots), 1, 4), 0.25)
            return ids, cand, q
        greedy = e.drafter.propose_rows
        e.drafter.propose_rows = sampled
        p._build([1], [1])                                   # a sampled batch of one
        self.assertEqual(p.buf['qprob'][0].tolist(), [[0.25] * 4])
        e.limits[2] = (10, 0.0)
        e.ends[2] = {31}
        e.drafter.propose_rows = greedy                      # the joining row walks greedily
        p._merge([1, 2], [1, 2])
        held = p.batch.index(1)
        self.assertEqual(p.buf['qprob'][held].tolist(), [[0.25] * 4], "the survivor's own draft, untouched")
        self.assertEqual(float(p.buf['qprob'][1 - held].sum()), 1.0, "the joining greedy row is a point mass")

    def test_shrinking_keeps_device_progress_ahead_of_the_host(self):
        e = self.engine()
        p = AsyncDecode(e)
        first = p.launch([1, 2], [1, 2])
        second = p.launch([2], [2])
        self.assertEqual(p.buf['ctx'].tolist(), [5])
        self.assertEqual(e.ctx, {1: 1, 2: 1})
        first.resolve()
        second.resolve()
        self.assertEqual(e.ctx, {1: 3, 2: 5})
        self.assertEqual(e.tokens[2], [6, 7, 8, 7, 8])


if __name__ == "__main__":
    unittest.main()
