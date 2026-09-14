"""Decode absorb ownership, model routing, boot proof and reserved GPU gate."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from engine.profiles.glm53.net import Glm53Net

ROWS = (8, 16, 24, 32)


class CudaView:
    is_cuda = True

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, key):
        return getattr(self.tensor, key)


class DecodeAbsorbTests(unittest.TestCase):
    def test_original_weight_views_fresh_outputs_and_all_capture_widths(self):
        from engine.kernels.mla.decode_absorb import DecodeAbsorb
        storage = torch.empty(16, 512, 512).bfloat16()
        weights = tuple(CudaView(w) for w in (storage[:, :256], storage[:, 256:]))
        declaration = list(ROWS)
        with patch('torch.cuda.is_current_stream_capturing', return_value=False), patch(
                'engine.kernels.mla.decode_absorb._absorb', MagicMock()) as kernel:
            owner = DecodeAbsorb(*weights, rows=declaration)
            declaration.clear()
            self.assertIs(owner.weights[0], weights[0])
            self.assertIs(owner.weights[1], weights[1])
            for m in ROWS:
                for transpose in (False, True):
                    inner, outer = (512, 256) if transpose else (256, 512)
                    x = torch.empty(m, 16, inner).bfloat16()
                    a, b = owner(x, transpose=transpose), owner(x, transpose=transpose)
                    self.assertEqual(a.shape, (m, 16, outer))
                    self.assertNotEqual(a.data_ptr(), b.data_ptr())
                    self.assertEqual(a.reshape(m, -1).data_ptr(), a.data_ptr())
                    self.assertEqual(a.contiguous().data_ptr(), a.data_ptr())
                    args = kernel.__getitem__.return_value.call_args.args
                    self.assertIs(args[1], weights[int(transpose)])
                    self.assertEqual(args[-4:], (transpose, 16 if m <= 16 else 32, 64, 64))
            self.assertEqual(owner.executed, {(m, s) for m in ROWS for s in ('query', 'output')})
            for x in (torch.empty(7, 16, 256).bfloat16(), torch.empty(8, 16, 256),
                      torch.empty(8, 16, 512).bfloat16()[:, :, ::2]):
                with self.assertRaises(ValueError):
                    owner(x)
            for rows in ((), (16,), (8, 24), (8.,), ROWS+(40,)):
                with self.assertRaises(ValueError):
                    DecodeAbsorb(*weights, rows=rows)
            with self.assertRaises(ValueError):
                DecodeAbsorb(storage[:, :256], storage[:, 256:], rows=ROWS)
        with patch('torch.cuda.is_current_stream_capturing', return_value=True):
            with self.assertRaises(RuntimeError):
                DecodeAbsorb(*weights, rows=ROWS)

    def test_model_routes_captured_widths_and_preserves_reference_execution(self):
        net = Glm53Net.__new__(Glm53Net)
        net.prefill_absorb_tiles = False
        net.lanes = NS(mla_absorb=None)
        for m, captured, probe, enabled in ((8, True, False, True), (32, True, False, True),
                (8, False, False, True), (8, True, True, True), (7, True, False, True),
                (8, True, False, False), (128, False, False, True)):
            net.probe, net.decode_absorb_rows = probe, ROWS if enabled else ()
            owner = Mock(return_value=object())
            net._decode_absorb = {3: owner}
            w = torch.randn(2, 3, 5).bfloat16()
            for transpose, inner in ((False, 3), (True, 5)):
                x = torch.randn(m, 2, inner).bfloat16()
                got = net._mla_absorb(3, x, w, NS(captured=captured, segments=(None,)), transpose=transpose)
                if enabled and captured and not probe and m in ROWS:
                    self.assertIs(got, owner.return_value)
                    owner.assert_called_with(x, transpose=transpose)
                else:
                    expected = torch.einsum('thc,hvc->thv' if transpose else 'thd,hdc->thc', x, w)
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)
            self.assertEqual(owner.call_count, 2 if enabled and captured and not probe and m in ROWS else 0)

    def test_preparation_retains_both_offsets_and_boot_demands_both_sides(self):
        from engine.profiles.glm53.boot import decode_absorb_report
        net = Glm53Net.__new__(Glm53Net)
        net._decode_absorb, net.Hl, net.layers = {}, 16, (0, 3)
        net.F = NS(spec_k=7, qk_nope=256, v_dim=256, kv_lora=512, is_dsa=lambda L: L == 3)
        w = torch.empty(8192, 512).bfloat16()
        net.p = {'L3.mla.kv_b': w}
        with patch('engine.kernels.mla.decode_absorb.DecodeAbsorb') as cls:
            net.prepare_decode_absorb(ROWS)
            q, out = cls.call_args.args
            self.assertEqual(q.data_ptr(), w.data_ptr())
            self.assertEqual(out.data_ptr(), w.data_ptr()+256*512*2)
            self.assertEqual(q.stride(), out.stride())
            with self.assertRaises(ValueError):
                net.prepare_decode_absorb(ROWS)
        expected = {(m, s) for m in ROWS for s in ('query', 'output')}
        for omitted in expected:
            net._decode_absorb = {3: NS(executed=expected-{omitted})}
            with self.assertRaisesRegex(RuntimeError, 'both contractions'):
                decode_absorb_report(net)
        net._decode_absorb = {3: NS(executed=expected)}
        self.assertEqual(decode_absorb_report(net)['resident_bytes'], 0)

    def test_all_three_improvements_default_on_with_declared_rollbacks(self):
        from engine.base.config import ConfigError
        from engine.profiles.glm53.execution import ExecutionPlan
        from tests.test_engine_knobs import KnobDeclarationTests
        declared = KnobDeclarationTests()._declared
        self.assertFalse(ExecutionPlan().decode_absorb_tiles)
        self.assertTrue(ExecutionPlan(decode_absorb_tiles=True).active)
        self.assertIn('decode_absorb_tiles=1', ExecutionPlan(decode_absorb_tiles=True).label())
        with self.assertRaises(ValueError):
            ExecutionPlan(decode_absorb_tiles=1)
        for name in ('decode_dsa_inputs', 'decode_indexer_gate', 'decode_absorb_tiles'):
            for production in (False, True):
                self.assertEqual(declared({}, production=production)[name], 1)
            self.assertEqual(declared({'STK_'+name: '0'})[name], 0)
            with self.assertRaises(ConfigError):
                declared({'STK_'+name: '0'}, production=True)


@unittest.skipUnless(torch.cuda.is_available(), 'requires the reserved GB10 GPU')
class CudaTests(unittest.TestCase):
    def test_both_contractions_changed_replay_and_consumer_layout(self):
        from probes.engine_decode_absorb import check
        check(lambda *a, **kw: None, layers=1, timing=False)


if __name__ == '__main__':
    unittest.main()
