"""Served burst reservation, result retirement and row churn through a CPU graph oracle."""
from types import SimpleNamespace as NS, MethodType
import unittest
from unittest.mock import patch
import torch

from engine.profiles.glm53.adapter import Glm53Engine
from engine.profiles.glm53.burst_decode import BurstDecode, BurstPending
from engine.profiles.glm53.pipeline import AsyncDecode
from tests import test_engine_pipeline as pipeline_tests


class Event:
    def record(self):
        pass
    def synchronize(self):
        pass


class CpuLoop:
    """Execute the actual burst body sequentially; no GPU timing claim."""
    def __init__(self, pipeline, shape):
        self.pipeline, self.shape = pipeline, shape
        self.timings = torch.zeros(4, 2, dtype=torch.int64)
        self.body = NS(reset=lambda: None)

    def replay(self):
        p, c = self.pipeline, self.pipeline.controls[self.shape[0]]
        c['count'].zero_()
        for i in range(p.iterations):
            self.timings[i] = torch.tensor([i*10000, i*10000+9000])
            p._body(self.shape)
            c['count'].add_(1)
            if c['stop'].item():
                break

    def close(self):
        pass


class CpuBurst(BurstDecode):
    def _capture(self):
        for n in range(1, 5):
            self.states[n], self.logs[n], self.controls[n] = self._state(n)
            self.loops[n, self.t, 64] = CpuLoop(self, (n, self.t, 64))

    def iterate(self, shape, b, host_step=None, *, measured=False, stage_clock=None):
        held = {k: v for k, v in b.items() if isinstance(v, torch.Tensor)}
        result = super().iterate(shape, b, host_step, measured=measured, stage_clock=stage_clock)
        # The CUDA commit mutates stable buffers. The eager oracle replaces
        # three entries; retain those stable addresses for the same contract.
        for key, value in held.items():
            if b[key] is not value:
                value.copy_(b[key])
                b[key] = value
        return result

    def launch(self, seqs, slots):
        with patch('torch.cuda.Event', Event):
            return super().launch(seqs, slots)


def engine(rows=4):
    e = pipeline_tests.BatchTransitionTests().engine()
    e.max_context, e.memory = 64, None
    e.net = NS(comm=NS(world_size=4, transport=NS(eligible_max=lambda t: True), all_reduce_max=lambda t: t))
    e.caches.pool.tokens = [64] * 5
    e.caches.prepare = lambda step: None
    e.decode_graphs.run_inputs = lambda *args: (None, None, None)
    for seq in range(1, rows+1):
        e.tokens[seq], e.ctx[seq], e.limits[seq] = [5+seq], 1, (100, 0.)
    return e


class ServedBurstTests(unittest.TestCase):
    def test_every_iteration_matches_the_existing_pipeline_at_c1_and_c4(self):
        for n in (1, 4):
            for limit in (2, 4):
                e, ref = engine(n), engine(n)
                p, base = CpuBurst(e, limit), AsyncDecode(ref)
                rows = list(range(1, n+1))
                for _ in range(3):
                    old = dict(e.ctx)
                    pending = p.launch(rows, rows)
                    self.assertEqual(e.ctx, old, 'host tokens stay pending until readback')
                    with self.assertRaisesRegex(RuntimeError, 'resolve the current burst'):
                        p.launch(rows, rows)
                    for _ in range(limit):
                        base.launch(rows, rows).resolve()
                    self.assertEqual(pending.resolve(), [False]*n)
                    self.assertEqual(e.tokens, ref.tokens)
                    self.assertEqual(e.ctx, ref.ctx)
                    self.assertEqual((e.accepted_total, e.drafted_total, e.steps),
                                     (ref.accepted_total, ref.drafted_total, ref.steps))
                    self.assertEqual(len(pending.iteration_seconds), limit)
                    self.assertTrue(all(value == 0 for value in e.inflight.values()))
                p.close()

    def test_prefix_and_eos_stop_before_another_iteration(self):
        e = engine()
        p = CpuBurst(e, 4)
        e.ctx[1] = 13
        pending = p.launch([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertEqual(pending.resolve(), [False]*4)
        self.assertEqual(len(pending.iteration_seconds), 2)
        self.assertEqual(e.staged, {1: 16})
        p.invalidate([2])
        e.ends[2] = {8}
        pending = p.launch([1, 2], [1, 2])
        self.assertEqual(pending.resolve(), [False, True])
        self.assertEqual(len(pending.iteration_seconds), 1)
        p.close()

    def test_reserved_end_and_bucket_limit_even_when_loop_allows_four(self):
        for end, start in ((5, 1), (64, 60)):
            e = engine(1)
            e.ctx[1], e.caches.pool.tokens[1] = start, end
            p = CpuBurst(e, 4)
            pending = p.launch([1], [1])
            pending.resolve()
            self.assertEqual(e.ctx[1], end)
            self.assertEqual(len(pending.iteration_seconds), 2)
            p.close()

    def test_new_row_and_slot_reuse_do_not_keep_old_device_identity(self):
        e = engine()
        p = CpuBurst(e, 2)
        p.launch([1, 2], [1, 2]).resolve()
        e.ctx[3] = 20
        p.launch([3, 2], [4, 2]).resolve()
        self.assertEqual(e.ctx, {1: 5, 2: 9, 3: 24, 4: 1})
        self.assertEqual(p.buf['real_slot'].tolist(), [4, 2])
        p.invalidate([2])
        e.ctx[2] = 40
        p.launch([2], [3]).resolve()
        self.assertEqual(e.ctx[2], 44)
        self.assertEqual(p.buf['real_slot'].tolist(), [3])
        p.close()

    def test_large_end_set_keeps_the_existing_single_step_path(self):
        e = engine(1)
        e.ends[1] = set(range(20, 31))
        p = CpuBurst(e, 4)
        self.assertEqual(p.reserve_steps(1), 1)
        pending = p.launch([1], [1])
        self.assertNotIsInstance(pending, BurstPending)
        pending.resolve()
        self.assertEqual(e.ctx[1], 3)
        e.limits[1] = (100, .7)
        self.assertEqual(p.reserve_steps(1), 1)
        p.close()

    def test_text_stops_keep_the_ordinary_host_verification_cadence(self):
        from engine.base.sampler import validate_options
        e = engine(1)
        e.options[1] = dict(_host_stop=True)
        validate_options(e.options[1])
        with self.assertRaises(ValueError):
            validate_options(dict(_host_stop=1))
        p = CpuBurst(e, 4)
        self.assertEqual(p.reserve_steps(1), 1)
        pending = p.launch([1], [1])
        self.assertNotIsInstance(pending, BurstPending)
        pending.resolve()
        self.assertEqual(e.ctx[1], 3)
        p.close()

    def test_whole_burst_horizon_and_reasoning_cap_cover_every_iteration(self):
        e = engine(1)
        e.pipeline = CpuBurst(e, 4)
        e.horizon = MethodType(Glm53Engine.horizon, e)
        self.assertEqual(e.horizon(1), 9)
        e.ctx[1] = 60
        self.assertEqual(e.horizon(1), 64)
        e.ctx[1] = 1
        e.thinking, e.prompt_len = {1: True}, {1: 1}
        e.options[1] = dict(reasoning_budget=6, reasoning_end=31)
        self.assertTrue(Glm53Engine._reasoning_boundary(e, 1))
        e.pipeline.close()

    def test_per_stage_device_nanoseconds_are_reported_as_microseconds_and_seconds(self):
        p = CpuBurst(engine(1), 2)
        pending = p.launch([1], [1])
        p.readback['stages'][:2, :2] = torch.tensor([[1000, 1001000], [2000, 2002000]])
        pending.resolve()
        self.assertEqual([r['stages_us']['forward'] for r in pending.iteration_records], [1000., 2000.])
        self.assertAlmostEqual(p.clock.totals['forward'], .003)
        p.close()

    def test_zero_progress_cannot_silently_schedule_bursts_forever(self):
        p = CpuBurst(engine(1), 2)
        pending = p.launch([1], [1])
        p.readback['count'].zero_()
        p.readback['done'].zero_()
        with self.assertRaisesRegex(RuntimeError, 'made no progress'):
            pending.resolve()

    def test_runner_records_real_iterations_without_inventing_host_timings(self):
        from tests import test_engine_runner_async as runner_tests
        from engine.base.runner import STEP_RECORD
        class Model(runner_tests.Model):
            def decode_async(self, seqs, blocks, slots):
                pending = super().decode_async(seqs, blocks, slots)
                pending.iteration_seconds = [.01, .02, .03]
                pending.iteration_records = [dict(positions=[16+i], committed=[1], accepted=[0]) for i in range(3)]
                return pending
        r = runner_tests.runner(Model())
        r.submit(1, 16, now=0)
        r.step(now=25)
        r.step(now=25)
        records = []
        r.latency = NS(active=True, row=lambda **v: records.append(v))
        r.resolve_oldest()
        self.assertEqual((r.steps, r.async_steps, r.decode_batches[1]), (4, 3, 3))
        self.assertEqual(r.rec.root.counters['decode_steps'], 3)
        iterations = [v for v in records if v['kind'] == 'gpu_iteration']
        self.assertEqual([v['duration_us'] for v in iterations], [10000., 20000., 30000.])
        wall_record = STEP_RECORD.unpack(r.ring.ordered()[-1])
        self.assertEqual((wall_record[0], wall_record[4]), (4, 3))


if __name__ == '__main__':
    unittest.main()
