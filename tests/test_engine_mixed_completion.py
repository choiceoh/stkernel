"""Cold queue coverage and actual eager completion ordering without a GPU."""
import ast
from pathlib import Path
import random
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.modules.mixed_experts import ExpertInvocation, plan_experts
from engine.modules.mixed_completion import plan_cold

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ExpertInvocation(3, 1, 2, 3)


class ColdLayoutTests(unittest.TestCase):
    def test_complete_exclusive_coverage_and_encoded_task_windows(self):
        rng = random.Random(89515)
        for d, p, quota in ((1, 3, 1), (8, 9240, 48), (32, 32768, 128)):
            decode = [rng.sample(range(288), 8) for _ in range(d)]
            prefill = [[(row * 8 + k) % 288 for k in range(8)] for row in range(p)]
            mixed = plan_experts(decode, prefill, identity=IDENTITY)
            cold = plan_cold(mixed, task_quota=quota)
            physical, covered = {}, []
            for expert, dest, token, slot in cold.sources:
                self.assertNotIn(dest, physical)
                self.assertEqual(expert, prefill[token][slot])
                physical[dest] = expert
                covered.append((token, slot))
            covered.extend((s[3], s[4]) for s in mixed.sources[d*8:])
            self.assertEqual(sorted(covered), [(r, s) for r in range(p) for s in range(8)])
            dispatched = []
            for start, stop in cold.windows:
                self.assertTrue(0 < stop-start <= quota)
                for index in range(start, stop):
                    task, valid = cold.task_expert[index], cold.task_valid_rows[index]
                    expert, tile = task & 65535, (task >> 16) & 65535
                    rows, begin, count = valid & 255, (valid >> 8) & 4095, (valid >> 20) & 4095
                    self.assertEqual((begin, count), (0, 4))
                    self.assertTrue(1 <= rows <= 128)
                    for dest in range(tile*128, tile*128+rows):
                        self.assertEqual(physical[dest], expert)
                        dispatched.append(dest)
            self.assertEqual(sorted(dispatched), sorted(physical))
            self.assertEqual(len(cold.task_expert), mixed.work()['cold_tiles'])
            self.assertEqual(cold.physical_rows, len(cold.task_expert)*128)

    def test_no_hot_no_cold_and_partial_tiles(self):
        for p in (1, 14, 127, 128, 140, 143):
            for quota in (0, 128):
                mixed = plan_experts([list(range(8))]*2, [list(range(8))]*p,
                                     identity=IDENTITY, hot_route_quota=quota)
                cold = plan_cold(mixed, task_quota=1)
                self.assertEqual(sum(cold.counts)+mixed.hot_routes, p*8)
                self.assertEqual(len(cold.windows), mixed.work()['cold_tiles'])
                for e in range(288):
                    self.assertEqual(cold.counts[e], mixed.prefill_counts[e]-mixed.hot_counts[e])
        self.assertEqual(plan_cold(plan_experts([list(range(8))], [list(range(8))],
            identity=IDENTITY)).windows, ())

    def test_invalid_quota_and_unowned_plan(self):
        mixed = plan_experts([list(range(8))], [list(range(8))], identity=IDENTITY)
        for quota in (0, -1, 129, True, 1.5):
            with self.assertRaises(ValueError):
                plan_cold(mixed, task_quota=quota)
        with self.assertRaises(ValueError):
            plan_cold(object())

    def test_private_consumer_reuses_body_and_cannot_reset_accumulator(self):
        tree = ast.parse((ROOT/'engine/kernels/b12x/moe_prepared_prefill.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertEqual([ast.unparse(n) for n in cls.bases], ['MoEGatedDynamicKernelSF6Prefill'])
        self.assertEqual([n.name for n in cls.body], ['initialize_route_q0_and_publish'])
        calls = [ast.unparse(n.func) for n in ast.walk(cls) if isinstance(n, ast.Call)]
        self.assertEqual(calls, ['cute.arch.sync_threads'])

    def test_actual_compiler_guard_refuses_incompatible_modes(self):
        tree = ast.parse((ROOT/'engine/kernels/b12x/moe_dispatch.py').read_text())
        compiler = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_get_dynamic_kernel')
        guards = [n for n in compiler.body if isinstance(n, ast.If) and
                  ast.unparse(n.test) in ('type(_prepared_prefill) is not bool', '_prepared_prefill')]
        self.assertEqual(len(guards), 2)
        guard = compile(ast.Module(body=guards, type_ignores=[]), '<actual compiler guards>', 'exec')
        good = dict(_prepared_prefill=True, prefill_word_unpack=True, prefill_reuse=False,
            _prefill_packets=False, _prefill_scale_expansion=False, _prefill_tile64=False,
            _prefill_n128=False, _prefill_q0_batch8=False, input_scales_are_reciprocal=False,
            fast_math=True, topk_ids_dtype='i32', torch=SimpleNamespace(int32='i32'), cache_key=())
        output = dict(good); exec(guard, output)
        self.assertEqual(output['cache_key'], ('prepared_cold_window_v1',))
        for name in ('prefill_word_unpack', 'prefill_reuse', '_prefill_packets', '_prefill_scale_expansion',
                     '_prefill_tile64', '_prefill_n128', '_prefill_q0_batch8', 'input_scales_are_reciprocal', 'fast_math'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                exec(guard, dict(good, **{name: not good[name]}))
        with self.assertRaises(TypeError):
            exec(guard, dict(good, _prepared_prefill=1))
        with self.assertRaises(ValueError):
            exec(guard, dict(good, topk_ids_dtype='i64'))
        ordinary = dict(good, _prepared_prefill=False); exec(guard, ordinary)
        self.assertEqual(ordinary['cache_key'], ())


class CompletionOrderingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        namespace = runpy.run_path(str(ROOT/'engine/kernels/b12x/moe_mixed_completion.py'))
        self.cls = namespace['PreparedMixedCompletion']
        self.calls = []
        # Actual owner methods and tensor reduction; only device launches and
        # shared GEMMs are replaced, so this is an ordering gate, not numerics.
        self.shared_patch = patch.dict(self.cls.begin.__globals__, shared_ffn=self.shared)
        self.shared_patch.start()
        self.addCleanup(self.shared_patch.stop)

    def shared(self, x, weights):
        self.calls.append(('shared', len(x)))
        return self.torch.full_like(x, 5.)

    def owner(self):
        t = self.torch
        o = self.cls.__new__(self.cls)
        o.plan = SimpleNamespace(identity=IDENTITY, decode_routes=8, decode=(0,),
                                 sources=tuple(range(10)), hot_routes=2)
        o._decode_source = t.zeros(1, 4096, dtype=t.bfloat16)
        o.hot = SimpleNamespace(partials=t.ones(40, 4096),
            validate=lambda identity: self.assertEqual(identity, IDENTITY),
            run=lambda identity: t.ones(10, 4, 4096))
        o.cold = SimpleNamespace(sources=(1,), windows=((0, 2), (2, 3)))
        o._accumulator = t.full((3, 4096), -999., dtype=t.bfloat16)
        o._window_args = (t.zeros(1, dtype=t.int32), t.zeros(1, dtype=t.int32))
        o._producer_args = (t.zeros_like(o._accumulator),)
        o._compute_args = (object(),)
        o.producer = lambda *args: self.calls.append(('pack', args))
        def compute(*args):
            self.assertEqual(args, o._compute_args)
            self.calls.append(('body', int(o._window_args[0]), int(o._window_args[1])))
            o._accumulator.add_(10.)
        o.compiled = compute
        o._hot_rows, o._hot_dest = t.tensor([0]), t.tensor([0, 0])
        o._hot_sum = t.empty(1, 4096)
        o._shared_weights = (t.ones(1), t.ones(1))
        o._shared_execution = None
        o._owned, o._versions = o._shared_weights, (0, 0)
        o.stream = 'owned'
        o.decode_ready = SimpleNamespace(record=lambda stream: self.calls.append(('decode_ready', stream)))
        o.prefill_ready = SimpleNamespace(record=lambda stream: self.calls.append(('prefill_ready', stream)))
        o.state, o.next_window = 'new', 0
        o._prefill_output = None
        return o

    def test_bound_shared_decode_callback_and_prefill_execute_in_separate_phases(self):
        o = self.owner()
        def shared_decode(x, routed):
            self.calls.append(('bound_decode', len(x)))
            return routed()+7.
        o._shared_execution = SimpleNamespace(validate=lambda: None, decode=shared_decode,
            prefill=lambda x: (self.calls.append(('bound_prefill', len(x))) or self.torch.full_like(x, 6.)))
        self.assertTrue(bool((o.begin(IDENTITY) == 39.).all()))
        self.assertEqual([c[0] for c in self.calls], ['bound_decode', 'decode_ready'])
        o.advance(IDENTITY)
        self.assertNotIn('bound_prefill', [c[0] for c in self.calls])
        o.finish(IDENTITY)
        self.assertEqual([c[0] for c in self.calls][-2:], ['bound_prefill', 'prefill_ready'])

    def test_completion_after_all_windows_hot_routes_and_shared(self):
        o = self.owner()
        for method in (o.advance, o.finish, o.prefill_result):
            with self.assertRaises(RuntimeError):
                method(IDENTITY)
        self.assertTrue(bool((o.begin(IDENTITY) == 37.).all()))
        self.assertEqual(self.calls[-1], ('decode_ready', 'owned'))
        with self.assertRaises(RuntimeError):
            o.begin(IDENTITY)
        self.assertFalse(o.advance(IDENTITY))
        self.assertTrue(bool((o._accumulator == 10.).all()))
        with self.assertRaisesRegex(RuntimeError, 'eight'):
            o.prefill_result(IDENTITY)
        result = o.finish(IDENTITY)
        self.assertTrue(bool((result[0] == 33.).all()))
        self.assertTrue(bool((result[1:] == 25.).all()))
        self.assertEqual([c for c in self.calls if c[0]=='body'], [('body', 0, 2), ('body', 2, 3)])
        self.assertEqual(sum(c[0]=='pack' for c in self.calls), 1)
        self.assertEqual(self.calls[-2:], [('shared', 3), ('prefill_ready', 'owned')])
        self.assertIs(o.prefill_result(IDENTITY)[0], result)
        with self.assertRaises(RuntimeError):
            o.finish(IDENTITY)
        o.begin(IDENTITY); again = o.finish(IDENTITY)
        self.assertTrue(self.torch.equal(result, again))

    def test_no_hot_no_cold_still_requires_shared_completion(self):
        o = self.owner()
        o.plan.hot_routes = 0
        o.cold = SimpleNamespace(sources=(), windows=())
        o.begin(IDENTITY)
        self.assertTrue(bool((o.finish(IDENTITY) == 5.).all()))
        self.assertFalse(any(c[0] in ('body', 'pack') for c in self.calls))

    def test_failed_partial_dispatch_cannot_be_retried_or_published(self):
        o = self.owner(); o.begin(IDENTITY)
        def broken(*args):
            o._accumulator.add_(3.)
            raise RuntimeError('device launch failed')
        o.compiled = broken
        with self.assertRaisesRegex(RuntimeError, 'device launch'):
            o.advance(IDENTITY)
        for method in (o.begin, o.advance, o.finish, o.prefill_result):
            with self.assertRaises(RuntimeError):
                method(IDENTITY)
        self.assertFalse(any(c[0]=='prefill_ready' for c in self.calls))

    def test_changed_shared_owner_blocks_every_phase(self):
        o = self.owner(); o.begin(IDENTITY)
        o._shared_weights[0].add_(0.)
        for method in (o.advance, o.finish, o.prefill_result):
            with self.assertRaisesRegex(RuntimeError, 'ownership changed'):
                method(IDENTITY)
        self.assertFalse(any(c[0]=='body' for c in self.calls))

    def test_bounded_probe_error_matches_whole_tensor_metrics_and_refuses_bad_output(self):
        from probes.engine_mixed_completion_check import output_error
        t = self.torch
        # Cross the 2048-row comparison boundary without allocating H4096.
        reference = t.linspace(.1, 2., 2100*16).reshape(2100, 16)
        actual = reference * 1.0005
        result = output_error(actual, reference)
        delta = actual-reference
        self.assertAlmostEqual(result['relative_max'], float(delta.abs().max()/reference.abs().max()), places=8)
        self.assertAlmostEqual(result['relative_rms'], float(delta.square().mean().sqrt()/reference.square().mean().sqrt()), places=8)
        for bad in (reference*1.1, t.full_like(reference, float('nan'))):
            with self.assertRaises(RuntimeError):
                output_error(bad, reference)


if __name__ == '__main__':
    unittest.main()
