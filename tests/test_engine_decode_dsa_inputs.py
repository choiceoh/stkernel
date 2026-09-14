"""DSA ownership/routing and same-build GPU bytes/replay gates (no serving-speed verdict)."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from engine.kernels.dense import DenseLinear, W4Pack
from engine.kernels.dense.query_pair import QueryPair
from engine.profiles.glm53.net import Glm53Net

ROWS = (8, 16, 24, 32)


def cpu_layer(n=4096):
    layer = DenseLinear.__new__(DenseLinear)
    layer.cols, layer.rows, layer.decode_precision = 1536, n, 'w4'
    layer.packs = (W4Pack(torch.empty(1, dtype=torch.uint8), torch.empty(1, dtype=torch.int8),
                         torch.ones(n), n, 1536),)
    layer.workspace, layer.observer, layer.executed = None, Mock(), 0
    layer.decode_input_rows, layer.bound_input_executed = ROWS, set()
    return layer


class OwnershipTests(unittest.TestCase):
    def test_pair_retains_packs_observations_strides_and_disjoint_outputs(self):
        layers = cpu_layer(), cpu_layer(8192)
        original = tuple(layer.packs[0] for layer in layers)
        declaration = list(ROWS)
        pair = QueryPair(*layers, rows=declaration)
        declaration.clear()
        ext = Mock()
        with patch('engine.kernels.dense.query_pair.extension', return_value=ext):
            for m in reversed(ROWS):
                x = torch.zeros(m, 1544, dtype=torch.bfloat16)[:, 4:1540]
                a, b = pair(x)
                self.assertEqual((a.shape, b.shape), ((m, 4096), (m, 8192)))
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
                args = ext.run_query_pair.call_args.args
                self.assertIs(args[0], x)
                for i, layer in enumerate(layers):
                    self.assertIs(layer.packs[0], original[i])
                    self.assertIs(args[1][i], original[i].data)
                    layer.observer.assert_called_with(x, None)
                    self.assertEqual(layer.executed, 1)
        self.assertEqual(pair.executed, set(ROWS))
        self.assertEqual(layers[0].bound_input_executed, {24, 32})
        self.assertEqual(layers[1].bound_input_executed, set())
        with self.assertRaises(ValueError):
            pair(torch.zeros(7, 1536, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            pair(torch.zeros(8, 1537, dtype=torch.bfloat16)[:, :1536])

    def test_invalid_owners_and_mutable_capture_declarations_are_rejected(self):
        for rows in ((), (8, 24), (16,), (8, 16, 24, 32, 40), (8.,)):
            with self.assertRaises(ValueError):
                QueryPair(cpu_layer(), cpu_layer(), rows=rows)
        for attr, value in (('workspace', object()), ('decode_precision', 'fp8'), ('cols', 4096), ('packs', ())):
            bad = cpu_layer()
            setattr(bad, attr, value)
            with self.assertRaises(ValueError):
                QueryPair(cpu_layer(), bad, rows=ROWS)
        bad = cpu_layer()
        with self.assertRaises(ValueError):
            QueryPair(bad, bad, rows=ROWS)

    def test_operator_enabled_default_and_experimental_rollback(self):
        from engine.profiles.glm53.execution import ExecutionPlan
        from engine.base.config import ConfigError
        from tests.test_engine_knobs import KnobDeclarationTests
        declared = KnobDeclarationTests()._declared
        self.assertFalse(ExecutionPlan().decode_dsa_inputs)
        self.assertTrue(ExecutionPlan(decode_dsa_inputs=True).active)
        self.assertIn('decode_dsa_inputs=1', ExecutionPlan(decode_dsa_inputs=True).label())
        with self.assertRaises(ValueError):
            ExecutionPlan(decode_dsa_inputs=1)
        for production in (False, True):
            self.assertEqual(declared({}, production=production)['decode_dsa_inputs'], 1)
        self.assertEqual(declared({'STK_decode_dsa_inputs': '1'})['decode_dsa_inputs'], 1)
        self.assertEqual(declared({'STK_decode_dsa_inputs': '0'})['decode_dsa_inputs'], 0)
        with self.assertRaises(ConfigError):
            declared({'STK_decode_dsa_inputs': '1'}, production=True)

    def test_boot_proof_requires_each_query_and_latent_width(self):
        from engine.profiles.glm53.boot import decode_dsa_report
        net = NS(decode_dsa_rows=ROWS, layers=(0, 1, 2), F=NS(is_dsa=lambda L: L != 0),
                 _query_pairs={L: NS(executed=set(ROWS)) for L in (1, 2)})
        expected = {(L, m) for L in (1, 2) for m in ROWS}
        net.decode_pools_executed = expected
        for omitted in expected:
            net.decode_latents_executed = expected - {omitted}
            with self.assertRaisesRegex(RuntimeError, 'not executed'):
                decode_dsa_report(net)
        net.decode_latents_executed = expected
        self.assertEqual(decode_dsa_report(net)['input_pack_bytes'], 50688)
        for omitted in expected:
            net.decode_pools_executed = expected - {omitted}
            with self.assertRaisesRegex(RuntimeError, 'pools='):
                decode_dsa_report(net)
        net.decode_pools_executed = expected
        net._query_pairs[1].executed.remove(8)
        with self.assertRaisesRegex(RuntimeError, 'queries='):
            decode_dsa_report(net)

    def test_model_only_fuses_full_bound_captured_decode(self):
        for captured, probe, enabled, m in ((True, False, True, 8), (True, False, True, 32),
                                            (False, False, True, 8), (True, True, True, 8),
                                            (True, False, False, 8), (True, False, True, 7)):
            net = Glm53Net.__new__(Glm53Net)
            net.F = NS(q_lora=1536, kv_lora=512, qk_nope=4, v_dim=4, rms_eps=1e-6)
            net.Hl, net.probe = 1, probe
            net.p = {'L1.mla.q_a_norm': torch.ones(1536), 'L1.mla.kv_a_norm': torch.ones(512),
                     'L1.mla.kv_b': torch.zeros(8, 512)}
            net.decode_dsa_rows, net.decode_latents_executed = ROWS if enabled else (), set()
            query = torch.zeros(m, 4096, dtype=torch.bfloat16)
            pair = Mock(return_value=(torch.zeros(m, 4, dtype=torch.bfloat16), query))
            net._query_pairs = {1: pair}
            net._norm = Mock(side_effect=lambda x, *args: x)
            net.linear = Mock(side_effect=lambda x, name: torch.zeros(m, 2048 if name.endswith('qkv_a') else 4, dtype=torch.bfloat16))
            net.lanes = NS(latent_norm_write=Mock(), decode_rows=NS(latent_write=Mock()))
            net._indexer_rows = lambda *args: int(captured)
            net._indexer = Mock(return_value=(None, None))
            net._mla_absorb = lambda L, x, *args, **kw: x
            net._mla_context = lambda L, x, *args: x
            net.comm = NS(all_reduce=lambda x: x)
            step = NS(captured=captured, tokens=8 if m in ROWS else 7, contexts=torch.zeros(1, dtype=torch.int64),
                      segments=(NS(start=0, length=m, ctx=0, seq=0),))
            cache = NS(latent=lambda L: torch.zeros(m, 512).to(torch.float8_e4m3fn), token_maps=lambda L: (),
                       token_slots=lambda L, seq, pos: pos)
            net._dsa(1, torch.zeros(m, 4, dtype=torch.bfloat16), step, cache)
            used = captured and not probe and enabled and m in ROWS
            self.assertEqual(pair.call_count, int(used))
            self.assertEqual(net.lanes.latent_norm_write.call_count, int(used))
            self.assertEqual(net._norm.call_count, 1 if used else 2)
            self.assertEqual(net.lanes.decode_rows.latent_write.call_count, int(captured and not used))
            self.assertEqual(net.decode_latents_executed, {(1, m)} if used else set())
            if used:
                self.assertIs(net._indexer.call_args.kwargs['query'], query)


@unittest.skipUnless(torch.cuda.is_available(), 'requires the reserved GB10 GPU')
class CudaTests(unittest.TestCase):
    def test_shared_query_input_bytes_and_changed_graph_replay(self):
        from probes.engine_decode_dsa_inputs import query_check
        query_check(lambda *a, **k: None, layers=1, timing=False)

    def test_fused_latents_bytes_and_rebound_graph_slots(self):
        from probes.engine_decode_dsa_inputs import latent_check
        latent_check(lambda *a, **k: None, timing=False)


if __name__ == '__main__':
    unittest.main()
