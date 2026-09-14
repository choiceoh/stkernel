"""Head-gate ownership, bound routing and reserved-device numerical gates."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from engine.profiles.glm53.net import Glm53Net

ROWS = (8, 16, 24, 32)


class CudaView:
    """Only the device-admission surface is mocked; storage and strides are real."""
    is_cuda = True

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


class HeadGateTests(unittest.TestCase):
    def test_owner_keeps_fp32_storage_and_covers_each_declared_row_without_aliasing(self):
        from engine.kernels.indexer_gate import IndexerHeadGate
        w = CudaView(torch.randn(32, 4096))
        declaration = list(ROWS)
        kernel = MagicMock()
        with patch('torch.cuda.is_current_stream_capturing', return_value=False), patch(
                'engine.kernels.indexer_gate._gate_partials', kernel):
            owner = IndexerHeadGate(w, rows=declaration)
            declaration.clear()
            self.assertIs(owner.weight, w)
            for m in reversed(ROWS):
                x = torch.empty(m, 4104, dtype=torch.bfloat16)[:, 4:4100]
                a, b = owner(x), owner(x)
                self.assertEqual(a.shape, (m, 16, 32))
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
                self.assertTrue(a.is_contiguous())
                args = kernel.__getitem__.return_value.call_args.args
                self.assertIs(args[0], x)
                self.assertIs(args[1], w)
                self.assertEqual(args[3], 4104)
            self.assertEqual(owner.executed, set(ROWS))
            for x in (torch.empty(7, 4096).bfloat16(), torch.empty(8, 4096),
                      torch.empty(4096, 8).bfloat16().T):
                with self.assertRaises(ValueError):
                    owner(x)
            for rows in ((), (8, 24), (16,), (8.,), ROWS + (40,)):
                with self.assertRaises(ValueError):
                    IndexerHeadGate(w, rows=rows)
            for weight in (torch.empty(32, 4096), CudaView(torch.empty(32, 4096).bfloat16()),
                           CudaView(torch.empty(4096, 32).T)):
                with self.assertRaises(ValueError):
                    IndexerHeadGate(weight, rows=ROWS)
        with patch('torch.cuda.is_current_stream_capturing', return_value=True):
            with self.assertRaises(RuntimeError):
                IndexerHeadGate(w, rows=ROWS)
        with patch('engine.base.kernel_shape.bound', return_value=NS(hidden=5120,
                   indexer=NS(heads=32, head_dim=128, compress='kpool'))):
            with self.assertRaisesRegex(ValueError, 'compiled for'):
                IndexerHeadGate(w, rows=ROWS)

    def test_model_routes_only_declared_captured_target_rows(self):
        net = Glm53Net.__new__(Glm53Net)
        net.p = {'L3.idx.w_heads': torch.randn(32, 4096)}
        for captured, probe, rows, enabled in ((True, False, 8, True), (True, False, 32, True),
                (False, False, 8, True), (True, True, 8, True), (True, False, 7, True),
                (True, False, 8, False), (False, False, 128, True)):
            net.probe, net.decode_indexer_gate_rows = probe, ROWS if enabled else ()
            partials = torch.empty(rows, 16, 32)
            owner = Mock(return_value=partials)
            net._indexer_head_gates = {3: owner}
            x = torch.randn(rows, 4096).bfloat16()
            got, splits = net._indexer_head_gate(3, x, NS(captured=captured))
            used = captured and not probe and enabled and rows in ROWS
            self.assertEqual(owner.call_count, int(used))
            self.assertEqual(splits, 16 if used else 1)
            if used:
                self.assertIs(got, partials)
            else:
                torch.testing.assert_close(got, x.float() @ net.p['L3.idx.w_heads'].T, rtol=0, atol=0)

    def test_preparation_requires_bound_paired_boundary_and_original_weight(self):
        net = Glm53Net.__new__(Glm53Net)
        net._indexer_head_gates = {}
        net.decode_fastpath_rows = ROWS
        net.F, net.layers = NS(spec_k=7, is_dsa=lambda L: L == 3), [0, 3]
        net._decode_pairs = {}
        net.p = {'L3.idx.w_heads': object()}
        with self.assertRaisesRegex(ValueError, 'every paired'):
            net.prepare_decode_indexer_gate(ROWS)
        net._decode_pairs = {3: object()}
        with patch('engine.kernels.indexer_gate.IndexerHeadGate') as owner:
            net.prepare_decode_indexer_gate(ROWS)
            owner.assert_called_once_with(net.p['L3.idx.w_heads'], rows=ROWS)
            self.assertEqual(net.decode_indexer_gate_rows, ROWS)
            with self.assertRaises(ValueError):
                net.prepare_decode_indexer_gate(ROWS)

    def test_captured_indexer_passes_partials_to_boundary_then_pool_selection(self):
        net = Glm53Net.__new__(Glm53Net)
        net.F = NS(kpool=4, idx_heads=32, idx_dim=128, topk=2048, spec_k=7, idx_scale=.015625)
        net.probe, net.prefill_indexer_shards = False, False
        net._mla_prefix = lambda *args: 0
        net.decode_indexer_gate_rows = ROWS
        parts = torch.empty(8, 16, 32)
        net._indexer_head_gates = {3: Mock(return_value=parts)}
        net.p = {'L3.idx.k_norm_w': torch.ones(128), 'L3.idx.k_norm_b': torch.zeros(128)}
        pair = Mock(return_value=(torch.empty(8, 128).bfloat16(), torch.empty(8, 128).bfloat16()))
        pair.rows = ROWS
        net._decode_pair = Mock(return_value=pair)
        net._indexer_rows = lambda *args: 1
        net._select_rows = Mock()
        step = NS(captured=True, tokens=8, segments=(object(),), contexts=torch.zeros(1, dtype=torch.int64))
        cache = NS(pool_keys=lambda L: None, pool_scales=lambda L: None, tails=lambda L: torch.empty(1, 10))
        q, key, effective = torch.empty(8, 32, 128), torch.empty(8, 128), torch.empty(8, 32)
        with patch('engine.kernels.decode_projection.indexer_boundary', return_value=(q, key, effective)) as boundary, patch(
                'engine.profiles.glm53.decode_graphs.complete_pools', return_value=object()):
            net._indexer(3, torch.zeros(8, 4096).bfloat16(), torch.zeros(8, 1536).bfloat16(),
                         step, cache, query=torch.zeros(8, 4096).bfloat16())
            self.assertIs(boundary.call_args.args[2], parts)
            self.assertEqual(boundary.call_args.kwargs['head_splits'], 16)
            self.assertIs(net._select_rows.call_args.args[1], q)
            self.assertIs(net._select_rows.call_args.args[2], effective)

    def test_operator_enabled_default_dependency_and_complete_boot_proof(self):
        from engine.base.config import ConfigError
        from engine.profiles.glm53.boot import decode_indexer_gate_report
        from engine.profiles.glm53.execution import ExecutionPlan
        from tests.test_engine_knobs import KnobDeclarationTests
        declared = KnobDeclarationTests()._declared
        self.assertFalse(ExecutionPlan().decode_indexer_gate)
        plan = ExecutionPlan(decode_fastpaths=True, decode_indexer_gate=True)
        self.assertTrue(plan.active)
        self.assertIn('decode_indexer_gate=1', plan.label())
        for values in (dict(decode_indexer_gate=True), dict(decode_indexer_gate=1, decode_fastpaths=True)):
            with self.assertRaises(ValueError):
                ExecutionPlan(**values)
        for production in (False, True):
            self.assertEqual(declared({}, production=production)['decode_indexer_gate'], 1)
        self.assertEqual(declared({'STK_decode_indexer_gate': '1'})['decode_indexer_gate'], 1)
        self.assertEqual(declared({'STK_decode_indexer_gate': '0'})['decode_indexer_gate'], 0)
        with self.assertRaises(ConfigError):
            declared({'STK_decode_indexer_gate': '1'}, production=True)
        net = NS(layers=[3, 7], F=NS(is_dsa=lambda L: True), decode_indexer_gate_rows=ROWS)
        for L in net.layers:
            for m in ROWS:
                net._indexer_head_gates = {n: NS(executed=set(ROWS)) for n in net.layers}
                net._indexer_head_gates[L].executed.remove(m)
                with self.assertRaisesRegex(RuntimeError, 'every layer/width'):
                    decode_indexer_gate_report(net)
        net._indexer_head_gates = {L: NS(executed=set(ROWS)) for L in net.layers}
        report = decode_indexer_gate_report(net)
        self.assertEqual(report['resident_bytes'], 0)
        self.assertEqual(report['partial_bytes_per_call'][32], 65536)

    def test_boundary_validates_partials_and_keeps_token_major_outputs(self):
        from engine.kernels.decode_projection import indexer_boundary
        q = CudaView(torch.zeros(8, 32, 128).bfloat16())
        k = torch.zeros(8, 256).bfloat16()[:, :128]
        p, nw, nb = torch.empty(8, 16, 32), torch.ones(128), torch.zeros(128)
        with patch('engine.kernels.decode_projection._indexer_boundary', MagicMock()) as kernel:
            out = indexer_boundary(q, k, p, nw, nb, .01, rows=ROWS, head_splits=16)
            self.assertEqual([t.shape for t in out], [(8, 32, 128), (8, 128), (8, 32)])
            self.assertTrue(all(t.is_contiguous() for t in out))
            self.assertIs(kernel.__getitem__.return_value.call_args.args[2], p)
            for parts, splits in ((p, 1), (p, 8), (p[:, :, ::2], 16), (p.bfloat16(), 16)):
                with self.assertRaises(ValueError):
                    indexer_boundary(q, k, parts, nw, nb, .01, rows=ROWS, head_splits=splits)


@unittest.skipUnless(torch.cuda.is_available(), 'requires the reserved GB10 GPU')
class CudaTests(unittest.TestCase):
    def test_fp32_gate_boundary_numerics_and_changed_graph_replay(self):
        from probes.engine_decode_indexer_gate import check
        check(lambda *a, **kw: None, timing=False, layers=1, ranking=False)


if __name__ == '__main__':
    unittest.main()
