"""Four independent host ranks reject divergent readback before publishing it.

The CPU graph oracle executes the serving pipeline; no GPU or throughput claim.
"""
from copy import deepcopy
from types import MethodType, SimpleNamespace as NS
import unittest

from engine.base.tripwire import CollectiveDivergence
from engine.profiles.glm53.adapter import Glm53Engine
from engine.profiles.glm53.pipeline import AsyncDecode
from tests.test_engine_burst_decode import CpuBurst, CpuQueue, engine
from tests.test_engine_runner_async import Model, runner
from tests.test_engine_tripwire import run


class DecodeConsensusTests(unittest.TestCase):
    def make_bursts(self, rows, queue=False):
        pipelines, pendings, snapshots, published = [], [], [], [[] for _ in range(4)]
        for rank in range(4):
            p = CpuBurst(engine(rows), 4)
            if queue:
                p.queue = CpuQueue()
            p.e.on_decode_progress = lambda r=rank: published[r].append(True)
            pending = p.launch(list(range(1, rows+1)), list(range(1, rows+1)))
            pipelines.append(p); pendings.append(pending)
            snapshots.append((deepcopy(p.e.tokens), dict(p.e.ctx)))
        return pipelines, pendings, snapshots, published

    def assert_divergence(self, results, pipelines, snapshots):
        for (status, exc), p, before in zip(results, pipelines, snapshots):
            self.assertEqual(status, 'raised', exc)
            self.assertIsInstance(exc, CollectiveDivergence)
            self.assertEqual(exc.details['site'], 'decode:outcome')
            self.assertEqual((p.e.tokens, p.e.ctx), before, 'no rank applies the unagreed result')

    def test_c1_and_c4_reject_count_token_done_context_and_acceptance_skew(self):
        for rows in (1, 4):
            for field in ('count', 'tokens', 'done', 'before', 'accepted'):
                with self.subTest(rows=rows, field=field):
                    pipelines, pendings, before, published = self.make_bursts(rows, queue=True)
                    result = pipelines[1].queue.results[0]
                    row = rows - 1
                    if field == 'tokens':
                        result[field][row][0] += 1
                    elif field == 'done':
                        result[field][row] = not result[field][row]
                    elif field == 'before':
                        result[field][row] += 1
                    else:
                        result[field][row] -= 1
                    def body(wire, rank):
                        pipelines[rank].e.net.comm = wire.comm
                        return pendings[rank].resolve()
                    results = run(4, body)
                    self.assert_divergence(results, pipelines, before)
                    self.assertEqual(published, [[], [], [], []])
                    for pending, (_, exc) in zip(pendings, results):
                        with self.assertRaises(CollectiveDivergence) as caught:
                            pending.resolve()
                        self.assertIs(caught.exception, exc, 'cleanup never re-enters the collective')

    def test_unanimous_bursts_publish_each_iteration_and_ignore_uncommitted_padding(self):
        for rows in (1, 4):
            pipelines, pendings, before, published = self.make_bursts(rows, queue=True)
            # Padding after the committed prefix is not an output token.
            for rank, p in enumerate(pipelines):
                for result in p.queue.results:
                    for tokens in result['tokens']:
                        tokens.append(100 + rank)
            def body(wire, rank):
                pipelines[rank].e.net.comm = wire.comm
                return pendings[rank].resolve()
            self.assertTrue(all(status == 'ok' for status, _ in run(4, body)))
            self.assertEqual(published, [[True] * 4 for _ in range(4)])
            self.assertTrue(all(p.e.tokens == pipelines[0].e.tokens for p in pipelines))
            self.assertTrue(all(p.e.ctx != old[1] for p, old in zip(pipelines, before)))

    def test_nonmapped_burst_and_ordinary_async_readback_have_the_same_guard(self):
        for bounded in (False, True):
            pipelines, pendings, before = [], [], []
            for rank in range(4):
                p = CpuBurst(engine(1), 4) if bounded else AsyncDecode(engine(1))
                pending = p.launch([1], [1])
                if rank == 2:
                    if bounded:
                        p.readback['tokens'][0, 0, 0] += 1
                    else:
                        p.host[pending.slot]['tokens'][0, 0] += 1
                pipelines.append(p); pendings.append(pending)
                before.append((deepcopy(p.e.tokens), dict(p.e.ctx)))
            def body(wire, rank):
                pipelines[rank].e.net.comm = wire.comm
                return pendings[rank].resolve()
            self.assert_divergence(run(4, body), pipelines, before)


class PlanConsensusTests(unittest.TestCase):
    def model(self):
        m = Model()
        m.tokens, m.inflight = {}, {}
        m.agree_step = MethodType(Glm53Engine.agree_step, m)
        return m

    def test_one_rank_needing_synchronous_decode_switches_all_four(self):
        runners = [runner(self.model()) for _ in range(4)]
        for r in runners:
            r.submit(1, 20, now=0)
        def body(wire, rank):
            r = runners[rank]
            r.model.net = NS(comm=wire.comm)
            r.step(now=0)  # prefill
            r.model.ready = rank != 1
            r.step(now=0)
            return r.model.log[-1]
        self.assertEqual(run(4, body), [('ok', ('decode', (1,)))] * 4)

    def test_different_planned_rows_fail_before_any_model_work(self):
        runners = [runner(self.model()) for _ in range(4)]
        for rank, r in enumerate(runners):
            r.submit(2 if rank == 3 else 1, 20, now=0)
        def body(wire, rank):
            r = runners[rank]
            r.model.net = NS(comm=wire.comm)
            return r.step(now=0)
        for status, exc in run(4, body):
            self.assertEqual(status, 'raised')
            self.assertIsInstance(exc, CollectiveDivergence)
            self.assertEqual(exc.details['site'], 'runner:plan')
        self.assertEqual([r.model.log for r in runners], [[], [], [], []])

    def test_readiness_change_drains_outstanding_steps_on_every_rank(self):
        runners = [runner(self.model()) for _ in range(4)]
        for r in runners:
            r.submit(1, 20, now=0)
        def body(wire, rank):
            r = runners[rank]
            r.model.net = NS(comm=wire.comm)
            r.step(now=0)
            r.step(now=0); r.step(now=0)
            r.model.ready = rank != 1
            r.step(now=0)
            return r.model.log[-3:], len(r.inflight)
        expected = ([('resolve', (1,)), ('resolve', (1,)), ('decode', (1,))], 0)
        self.assertEqual(run(4, body), [('ok', expected)] * 4)


if __name__ == '__main__':
    unittest.main()
