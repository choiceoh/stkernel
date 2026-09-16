"""Model-bound K=7 routes; CPU wiring is not GPU numerical or speed proof."""
from types import SimpleNamespace as NS
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from engine.kernels import decode_projection as projections
from engine.kernels import dense
from engine.profiles.glm53.net import Glm53Net

ROWS = (8, 16, 24, 32)


class BoundDecodeTests(unittest.TestCase):
    def test_serving_default_on_with_experimental_rollback_and_neutral_bare_plan(self):
        from engine.base.config import ConfigError
        from engine.profiles.glm53.execution import ExecutionPlan
        from tests.test_engine_knobs import KnobDeclarationTests
        declared = KnobDeclarationTests()._declared
        self.assertFalse(ExecutionPlan().decode_fastpaths)
        self.assertTrue(ExecutionPlan(decode_fastpaths=True).active)
        self.assertIn('decode_fastpaths=1', ExecutionPlan(decode_fastpaths=True).label())
        with self.assertRaises(ValueError):
            ExecutionPlan(decode_fastpaths=1)
        for production in (False, True):
            cfg = declared({}, production=production)
            self.assertEqual(cfg['decode_fastpaths'], 1)
            self.assertEqual(cfg['kda_state_dtype'], 'fp32')
            self.assertEqual(cfg['prefill_dense_prefix'], 1)
            self.assertEqual(cfg['prefill_absorb_tiles'], 1)
        self.assertEqual(declared({'STK_decode_fastpaths': '1'})['decode_fastpaths'], 1)
        self.assertEqual(declared({'STK_decode_fastpaths': '0'})['decode_fastpaths'], 0)
        for value in ('0', '1'):
            with self.assertRaises(ConfigError):
                declared({'STK_decode_fastpaths': value}, production=True)

    @staticmethod
    def model(k=7):
        net = Glm53Net.__new__(Glm53Net)
        net.F = NS(spec_k=k, is_dsa=lambda L: False)
        net.layers = [0, 1]
        net.dense = {'L0.kda.in_proj': NS(packs=[NS(rows=6416, cols=4096)], bound_input_executed=set())}
        net.p = {f'L{L}.kda.{name}': object() for L in net.layers for name in ('f_b', 'g_b')}
        net._decode_pairs = {}
        net.decode_fastpath_rows = ()
        net.decode_pairs_executed = set()
        return net

    def test_binding_preserves_old_rows_and_cannot_affect_another_model(self):
        old = projections.DECODE_ROWS
        factory = lambda *weights, **kw: NS(rows=kw['rows'])
        candidate, control = self.model(), self.model()
        with patch.object(projections, 'KdaPair', side_effect=factory):
            candidate.prepare_decode_projections(None, capture_rows=ROWS)
            control.prepare_decode_projections(None)
        self.assertIs(projections.DECODE_ROWS, old)
        self.assertEqual(candidate._decode_pairs[0].rows, tuple(sorted(set(old + ROWS))))
        self.assertIsNone(control._decode_pairs[0].rows)
        self.assertEqual(candidate.dense['L0.kda.in_proj'].decode_input_rows, ROWS)
        for rows in old + ROWS:
            self.assertIsNotNone(candidate._decode_pair(0, NS(captured=True), rows))
            self.assertEqual(control._decode_pair(0, NS(captured=True), rows) is not None, rows in old)
        self.assertIsNone(candidate._decode_pair(0, NS(captured=False), 8))
        self.assertIsNone(candidate._decode_pair(0, NS(captured=True), 9))
        with self.assertRaises(RuntimeError):
            candidate.prepare_decode_projections(None, capture_rows=ROWS)
        for rows in ((), (16,), (8, 24), (8, 16, 24, 32, 40), (8.,), (True,), (8, 8)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.model().prepare_decode_projections(None, capture_rows=rows)
        with self.assertRaises(ValueError):
            self.model(6).prepare_decode_projections(None, capture_rows=ROWS)

    def test_same_bound_pack_routes_ordinary_private_and_direct_outputs(self):
        # Real DenseLinear call/write_slot methods; only the CUDA entry is mocked.
        ext = Mock()
        layer = dense.DenseLinear.__new__(dense.DenseLinear)
        layer.rows, layer.cols = 4096, 4096
        pack = NS(rows=4096, cols=4096, data=torch.empty(1, dtype=torch.uint8),
                  scale=torch.empty(1, dtype=torch.int8), rowscale=torch.ones(4096))
        layer.packs, layer.observer, layer.executed = [pack], None, 0
        layer.decode_input_rows, layer.bound_input_executed = ROWS, set()
        descriptor = torch.tensor([1234], dtype=torch.int64)
        with patch.object(dense, 'extension', return_value=ext):
            for workspace in (None, torch.zeros(32)):
                layer.workspace = workspace
                for rows in (8, 16, 24, 32, 7):
                    x = torch.zeros(rows, 4104, dtype=torch.bfloat16)[:, 4:4100]
                    ext.reset_mock()
                    layer(x)
                    layer._write_slot(x, descriptor)
                    if rows in ROWS:
                        self.assertEqual(ext.run_gemm_bound_input.call_count, 2)
                        ordinary, direct = ext.run_gemm_bound_input.call_args_list
                        self.assertEqual(ordinary.args[0].stride(), x.stride())
                        self.assertIs(ordinary.args[6], workspace)
                        self.assertIsNone(ordinary.args[7])
                        self.assertIs(direct.args[3], descriptor)
                        self.assertIs(direct.args[6], workspace)
                        self.assertIs(direct.args[7], descriptor)
                    else:
                        ext.run_gemm_bound_input.assert_not_called()
                        ext.run_gemm_to_slot.assert_called_once()
        self.assertEqual(layer.bound_input_executed, set(ROWS))
        self.assertTrue(dense.bound_input_cell(16, 6416, 4096))  # sixteen-row CTAs (st_c2_dense_cells_20260915)
        self.assertFalse(dense.bound_input_cell(16, 4096, 1536))  # QueryPair shares the pack; M14 kept it out
        self.assertFalse(dense.bound_input_cell(14, 6416, 4096))
        self.assertTrue(dense.bound_input_cell(24, 6416, 4096))
        self.assertTrue(dense.bound_input_cell(8, 4096, 2048))
        self.assertTrue(dense.bound_input_cell(8, 4096, 3072))
        self.assertFalse(dense.bound_input_cell(8, 4096, 1536))  # the query-pair owner selects its layout
        self.assertTrue(dense.bound_input_cell(16, 4096, 2048))
        self.assertFalse(dense.bound_input_cell(32, 4096, 20480))

    def test_proof_requires_every_candidate_layer_and_capture_width(self):
        from engine.profiles.glm53.boot import decode_fastpath_report
        net = self.model()
        self.assertEqual(decode_fastpath_report(net), {})
        net.decode_fastpath_rows = ROWS
        pairs = {(L, m) for L in net.layers for m in ROWS}
        layer = net.dense['L0.kda.in_proj']
        layer.bound_input_executed = {8, 16, 24, 32}
        for missing in pairs:
            net.decode_pairs_executed = pairs - {missing}
            with self.assertRaisesRegex(RuntimeError, 'not executed'):
                decode_fastpath_report(net)
        net.decode_pairs_executed = pairs
        self.assertEqual(decode_fastpath_report(net)['dense'], {'L0.kda.in_proj': [8, 16, 24, 32]})
        for missing in (8, 16, 24, 32):
            layer.bound_input_executed = {8, 16, 24, 32} - {missing}
            with self.assertRaisesRegex(RuntimeError, 'not executed'):
                decode_fastpath_report(net)

    def test_indexer_owner_freezes_rows_and_preserves_both_projections(self):
        # Only CUDA admission is bypassed: execute the actual owner, copied
        # weights, strided input validation, joined GEMM and output slicing.
        torch.manual_seed(91420)
        weights = [torch.randn(128, 4096).bfloat16() for _ in range(2)]
        declaration = list(ROWS)
        with patch.object(projections, '_weights'), patch('torch.cuda.is_current_stream_capturing', return_value=False):
            owner = projections.IndexerPair(*weights, rows=declaration)
        declaration.clear()
        self.assertEqual(owner.rows, ROWS)
        for rows in (32, 8, 24, 16):
            x = torch.randn(rows, 4104).bfloat16()[:, 4:4100]
            for actual, weight in zip(owner(x), weights):
                torch.testing.assert_close(actual, torch.nn.functional.linear(x, weight), rtol=0, atol=0)
        for rows in (7, 9, 33):
            with self.assertRaises(ValueError):
                owner(torch.zeros(rows, 4096, dtype=torch.bfloat16))


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA; not exercised by the CPU gate')
class BoundDecodeGpuTests(unittest.TestCase):
    def test_bound_ordinary_and_direct_graphs_follow_changed_inputs_and_addresses(self):
        from probes.engine_decode_fusions import _capture
        torch.manual_seed(91419)
        # Same packed weights and reduction plan; alternate rows to expose stale
        # shared scratch, and change descriptor after capture on every replay.
        for n, k in ((4096, 4096), (6416, 4096), (6144, 4096), (4096, 2048),
                     (2048, 4096), (4096, 3072), (4096, 1536)):
            layer = dense.DenseLinear(torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .02, prefill=False)
            for private in (False, True):
                if private:
                    layer.isolate_workspace()
                for rows in (32, 8, 24, 16, 8):
                    if not dense.bound_input_cell(rows, n, k):
                        continue
                    x = torch.randn(rows, k+8, device='cuda', dtype=torch.bfloat16)[:, 4:k+4]
                    guard = torch.full((2, rows+2, n), -123., device='cuda', dtype=torch.bfloat16)
                    address = torch.tensor([guard[0, 1].data_ptr()], device='cuda', dtype=torch.int64)
                    layer.decode_input_rows = ROWS
                    graph, actual = _capture(lambda: layer(x))
                    direct = _capture(lambda: layer._write_slot(x, address))[0] if n == 4096 else None
                    try:
                        layer.decode_input_rows = ()
                        for i, magnitude in enumerate((0., .001, 1., 50., 0.)):
                            x.normal_().mul_(magnitude)
                            expected = layer(x)
                            graph.replay()
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            if direct is not None:
                                guard.fill_(-123.)
                                address.fill_(guard[i % 2, 1].data_ptr())
                                direct.replay()
                                torch.testing.assert_close(guard[i % 2, 1:-1], expected, rtol=0, atol=0)
                                self.assertTrue(bool(guard[i % 2, 0].eq(-123.).all()))
                                self.assertTrue(bool(guard[i % 2, -1].eq(-123.).all()))
                                self.assertTrue(bool(guard[1-i % 2].eq(-123.).all()))
                    finally:
                        graph.reset()
                        if direct is not None:
                            direct.reset()


if __name__ == '__main__':
    unittest.main()


class CompactM8AgreesWithBoundC1Tests(unittest.TestCase):
    def test_a_shape_is_never_both_compact_m8_and_a_bound_cell_at_eight_rows(self):
        """The two kernels pick ksr from the same occupancy, and they must not disagree about it.

        `mk_use_compact_m8` sends a shape to the compact instantiation, whose smaller shared memory
        raises its occupancy, so `mk_choose_ksr2` reads g_gemm2_m8_bps instead of g_gemm2_bps and can
        return a different ksr. `mk_run_gemm_bound_input`'s own contract admits only ksr 3 at
        (n 4096, k 2048) and ksr 2 or 3 at (n 6144, k 4096); off those it raises "bound C1 input plan
        is outside the declared reduction geometry" and the boot dies before its door.

        On 2026-09-16 eight rows were admitted to the compact path unmeasured, which made those two
        shapes both compact AND bound cells, and every boot on main died there -- caught by a
        measurement arm, not by CI, because nothing here can boot. Six rows never met it: K=5 does not
        serve eight. Re-admitting eight rows means making the two agree first.
        """
        from engine.kernels.dense import bound_input_cell
        source = (Path(__file__).resolve().parents[1] / "engine/kernels/dense/kernels.cu").read_text()
        body = source.split("bool mk_use_compact_m8(")[1].split("}")[0]
        code = " ".join(l.split("//")[0] for l in body.splitlines())   # the comments name m == 8 to explain it
        compact_at_eight = "m == 8" in code
        compact_shapes = ((4096, 2048), (6144, 4096))          # the two the function names
        for n, k in compact_shapes:
            self.assertIn(f"n == {n} && k == {k}", code, "the compact shapes moved; re-read this test")
            if compact_at_eight:
                self.assertFalse(bound_input_cell(8, n, k),
                                 f"({n}, {k}) is compact at eight rows AND a bound cell: ksr comes from "
                                 "g_gemm2_m8_bps and the bound C1 contract will refuse the launch")
