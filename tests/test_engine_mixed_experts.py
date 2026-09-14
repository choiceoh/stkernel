"""Admission, complete route accounting, and isolation of prepared expert work."""
import ast
import hashlib
import json
from pathlib import Path
import random
import unittest

from engine.modules.mixed_experts import ExpertInvocation, plan_experts


IDENTITY = ExpertInvocation(3, 1, 2, 3)


def canonical_ast(node):
    # ast.dump's empty-field formatting and FunctionDef.type_params differ
    # across Python 3.9/3.12/3.14. Only ignore the absent/empty generic field.
    if isinstance(node, ast.AST):
        return [type(node).__name__, [[name, canonical_ast(value)] for name, value in ast.iter_fields(node)
                if not (name == 'type_params' and not value)]]
    if isinstance(node, list):
        return [canonical_ast(value) for value in node]
    return node


class MixedExpertTests(unittest.TestCase):
    def test_actual_c1_tile_and_only_useful_prefill_tails(self):
        decode = [list(range(8)) for _ in range(2)]
        for rows, hot, saved in ((100, 0, 0), (128, 0, 0), (140, 96, 8), (143, 0, 0)):
            plan = plan_experts(decode, [list(range(8))]*rows, identity=IDENTITY)
            work = plan.work()
            self.assertEqual((work['decode_tile_m'], plan.hot_routes, work['removed_prefill_tiles']), (16, hot, saved))
            self.assertEqual(work['mixed_tiles'], work['decode_tiles'])
        plan = plan_experts([list(range(8))]*32, [list(range(8))]*140, identity=IDENTITY)
        self.assertEqual((plan.tile_m, plan.hot_routes), (32, 0))

    def test_quota_prefers_smallest_tail_with_deterministic_ties(self):
        plan = plan_experts([list(range(8))]*2, [list(range(8))]*140,
                            identity=IDENTITY, hot_route_quota=25)
        self.assertEqual(plan.hot_counts, (12, 12) + (0,)*286)
        self.assertEqual(plan.work()['removed_prefill_tiles'], 2)

    def test_every_route_exactly_once_and_decode_order_unchanged(self):
        rng = random.Random(895)
        for count in (1, 7, 8, 9, 16, 24, 32):
            decode = [rng.sample(range(288), 8) for _ in range(count)]
            prefill = [rng.sample(range(288), 8) for _ in range(300)]
            plan = plan_experts(decode, prefill, identity=IDENTITY)
            self.assertEqual([(s[2], s[3], s[4]) for s in plan.sources[:count*8]],
                             [(0, r, s) for r in range(count) for s in range(8)])
            covered = [(s[3], s[4]) for s in plan.sources[count*8:]] + list(plan.cold_routes)
            self.assertEqual(sorted(covered), [(r, s) for r in range(300) for s in range(8)])
            positions = set()
            for local, row, kind, token, slot in plan.sources:
                self.assertNotIn((local, row), positions)
                positions.add((local, row))
                self.assertLess(row, 32)
                expert = (decode, prefill)[kind][token][slot]
                self.assertEqual(plan.experts[local], expert)
            work = plan.work()
            self.assertEqual(work['decode_tiles'], work['mixed_tiles'])
            self.assertLessEqual(plan.hot_routes, 128)
            self.assertEqual(work['removed_prefill_tiles'], sum(bool(h) for h in plan.hot_counts))

    def test_immutable_plan_and_no_hot_work_outside_decode_experts(self):
        decode, prefill = [list(range(8))], [list(range(8, 16))]
        plan = plan_experts(decode, prefill, identity=IDENTITY)
        decode[0][0] = 100
        self.assertEqual(plan.decode[0][0], 0)
        self.assertEqual(plan.hot_routes, 0)
        with self.assertRaises(AttributeError):
            plan.quota = 999

    def test_invalid_routes_and_generation_fail_before_work(self):
        valid = [list(range(8))]
        for value in ([], [[0]*8], [[True]+list(range(1, 8))], [list(range(7))],
                      [[288]+list(range(1, 8))], valid*33):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan_experts(value, valid, identity=IDENTITY)
        for value in (-1, 129, True, 1.5):
            with self.assertRaises(ValueError):
                plan_experts(valid, valid, identity=IDENTITY, hot_route_quota=value)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                ExpertInvocation(3, value, 1, 1)

    def test_normal_kernel_body_retains_every_statement(self):
        # Remove only the private frontend conditional and compare the whole
        # ordinary @cute.kernel AST from main 2ac7de6f (#923, compact input/FC2 staging).
        # Future body changes still require review.
        path = Path(__file__).resolve().parents[1]/'engine/kernels/b12x/moe_static_kernel_v4.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
        class Ordinary(ast.NodeTransformer):
            def visit_If(self, n):
                if ast.unparse(n.test) == 'cutlass.const_expr(not self.prepared_routes)':
                    return n.body
                return self.generic_visit(n)
        dump = json.dumps(canonical_ast(Ordinary().visit(node)), separators=(',', ':'))
        self.assertEqual(hashlib.sha256(dump.encode()).hexdigest(), 'bb79038065fc1628b7a992f33fe4350d14599488f2455bb387a4c424ca940312')

    def test_prepared_config_is_bounded_and_keeps_decode_geometry(self):
        path = Path(__file__).resolve().parents[1]/'engine/kernels/b12x/moe_dispatch.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_static_v2_decode_config')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        normalize = namespace[node.name]
        config = dict(tiled=True, reform_sf_pack=True, decode_reform=True, probe_prepared_routes=384)
        for rows in (1, 8, 9, 16, 24, 32):
            once = normalize(config, rows)
            self.assertEqual(once['decode_reform'], rows <= 8)
            self.assertEqual(normalize(once, rows), once)
        for field, value in (('split', True), ('even', True), ('stamps', True), ('skip_a', True),
                             ('a_ring', True), ('sf_pack', True), ('probe_route_scatter', True),
                             ('probe_prepared_routes', 385), ('probe_prepared_routes', True),
                             ('probe_prepared_routes', 10), ('tiled', False), ('decode_reform', False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize(dict(config, **{field: value}), 8)
        for rows in (0, 33):
            with self.assertRaises(ValueError):
                normalize(config, rows)

    def test_stale_identity_mutation_and_foreign_stream_never_launch(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from unittest.mock import patch
        import runpy
        import torch
        # The owner's guards have no GPU dependency. Load that actual file
        # without b12x/__init__ eagerly importing FlashInfer/CuTe on CPU CI.
        path = Path(__file__).resolve().parents[1]/'engine/kernels/b12x/moe_mixed.py'
        PreparedMixedExperts = runpy.run_path(str(path))['PreparedMixedExperts']
        owner = PreparedMixedExperts.__new__(PreparedMixedExperts)
        owner.plan = SimpleNamespace(identity=IDENTITY)
        for field in ('layer', 'epoch', 'slot_generation', 'source_generation'):
            with self.assertRaisesRegex(ValueError, 'stale'):
                owner.run(replace(IDENTITY, **{field: getattr(IDENTITY, field)+1}))
        owner.decode = torch.zeros(1)
        owner._owned = (owner.decode,)
        owner._versions = tuple(t._version for t in owner._owned)
        owner.stream = 'original'
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                patch.object(torch.cuda, 'current_stream', return_value='foreign'):
            with self.assertRaisesRegex(RuntimeError, 'original eager stream'):
                owner.run(IDENTITY)
        owner.decode.add_(0.)
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                patch.object(torch.cuda, 'current_stream', return_value='original'):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                owner.run(IDENTITY)

    def test_run_uses_frozen_arguments_and_publishes_only_after_body(self):
        import runpy
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        path = Path(__file__).resolve().parents[1]/'engine/kernels/b12x/moe_mixed.py'
        cls = runpy.run_path(str(path))['PreparedMixedExperts']
        owner = cls.__new__(cls)
        owner.plan = SimpleNamespace(identity=IDENTITY, sources=(1, 2))
        owner.decode = torch.ones(1)
        owner._owned, owner._versions = (owner.decode,), (owner.decode._version,)
        owner.stream, calls = 'original', []
        owner.partials = torch.full((12, 4096), float('nan'))
        owner._producer_args, owner._compute_args = (object(),), (object(),)
        owner.producer = lambda *args: calls.append(('producer', args))
        def compute(*args):
            calls.append(('body', args))
            owner.partials[:8].fill_(3.)
        owner.compiled = compute
        owner.last_reader = SimpleNamespace(record=lambda stream: calls.append(('event', stream)))
        # Replacing this high-level object cannot replace the frozen arguments.
        owner.weights = object()
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                patch.object(torch.cuda, 'current_stream', return_value='original'):
            result = owner.run(IDENTITY)
        self.assertEqual(calls, [('producer', owner._producer_args), ('body', owner._compute_args), ('event', 'original')])
        self.assertEqual(result.shape, (2, 4, 4096))
        self.assertEqual(result.data_ptr(), owner.partials.data_ptr())
        self.assertTrue(bool((result == 3.).all()))
        self.assertTrue(bool(owner.partials[8:].isnan().all()))

    def test_value_readback_overlaps_planning_and_is_drained_on_plan_failure(self):
        import runpy
        from types import SimpleNamespace
        from unittest.mock import patch
        path = Path(__file__).resolve().parents[1]/'engine/kernels/b12x/moe_mixed.py'
        fn = runpy.run_path(str(path))['checked_routes']
        for broken in (False, True):
            calls = []
            ids = SimpleNamespace(cpu=lambda: (calls.append('ids') or SimpleNamespace(numpy=lambda: 'snapshot')))
            def plan(*args, **kwargs):
                self.assertEqual(args, ('snapshot', 'snapshot'))
                calls.append('plan')
                if broken:
                    raise ValueError('bad routes')
                return 'planned'
            pending = SimpleNamespace(wait=lambda: calls.append('wait'))
            with patch.dict(fn.__globals__, prepare_routes=plan,
                            check_values=lambda *args: (calls.append('check') or pending)):
                if broken:
                    with self.assertRaisesRegex(ValueError, 'bad routes'):
                        fn(ids, ids, (), ())
                else:
                    self.assertEqual(fn(ids, ids, (), ()), 'planned')
            self.assertEqual(calls, ['ids', 'ids', 'check', 'plan', 'wait'])


if __name__ == '__main__':
    unittest.main()
