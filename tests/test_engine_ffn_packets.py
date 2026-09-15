"""Packet ownership, collective agreement and unchanged FFN arithmetic boundaries."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from engine.base.comm import LocalTP
from engine.modules.prefill_packets import PacketBatch, PacketGeometry, agreed_layers, ffn_packet_rows

ROOT = Path(__file__).resolve().parents[1]


class PacketContractTests(unittest.TestCase):
    def test_reference_expert_selection_disables_its_packet_reader(self):
        from engine.profiles.glm53.lanes import Lanes, _apply_reference_lanes
        fields = {name: None for name in Lanes.__dataclass_fields__}
        fields.update(moe=object(), moe_packets=object(), moe_packets_supported=object())
        table, ref = Lanes(**fields), NS(moe=object())
        self.assertIs(_apply_reference_lanes(table, ref, ()), table)
        selected = _apply_reference_lanes(table, ref, ('moe',))
        self.assertIs(selected.moe, ref.moe)
        self.assertIsNone(selected.moe_packets)
        self.assertIsNone(selected.moe_packets_supported)
        self.assertIsNotNone(table.moe_packets)

    def test_packet_plan_activity_is_boolean_and_keeps_chunk_order(self):
        from engine.profiles.glm53.execution import ExecutionPlan
        self.assertIs(ExecutionPlan().active, False)
        self.assertIs(ExecutionPlan(prefill_ffn_packets=True).active, True)
        self.assertIs(ExecutionPlan(decode_absorb_tiles=True).active, True)
        for value in (1, None, 'on'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExecutionPlan(prefill_ffn_packets=value)
        with self.assertRaises(ValueError):
            ExecutionPlan(prefill_ffn_packets=True, prefill_tiles=2)

    def test_only_the_existing_long_prefill_band_is_eligible(self):
        for rows, expected in ((8192, False), (8193, True), (32768, True),
                               (32769, False), (True, False), (9216., False)):
            self.assertEqual(ffn_packet_rows(rows), expected)

    def test_ragged_geometry_preserves_rank_order_and_scale_alignment(self):
        for rows in (8193, 8194, 8195, 8196, 32256, 32768):
            g = PacketGeometry(rows, (rows+3)//4)
            self.assertLess(g.padded_rows-rows, 4)
            self.assertEqual(g.stride % 128, 0)
            for rank in range(4):
                for local_row in (0, g.local_rows-1):
                    for col in (0, 2047, 2048, 4095):
                        element = local_row*4096 + col
                        value = rank*g.stride + element
                        scale = rank*g.stride + g.local_elements + 4*(element//2048)
                        self.assertLess(value, rank*g.stride+g.local_elements)
                        self.assertLess(scale+3, (rank+1)*g.stride)
                        self.assertEqual(scale % 4, 0)
            self.assertGreater(g.workspace()['replaced_bf16_bytes'], g.nbytes)
        self.assertEqual(PacketGeometry(32256, 8064).workspace(), dict(
            received_bytes=132378624, source_payload_bytes=33094656,
            replaced_bf16_bytes=264241152, shared_q_scale_bytes=136249344,
            expert_shared_stage_bytes=32768))

    def test_invalid_geometry_and_cpu_storage_fail_before_launch(self):
        for args in ((0, 32), (128, 33), (True, 32), (128, 32, 1024),
                     (128, 32, 4096, 2), (128, 32, 4096, 4, 1024)):
            with self.assertRaises(ValueError):
                PacketGeometry(*args)
        with self.assertRaises(ValueError):
            PacketBatch(torch.empty(0, dtype=torch.uint8), PacketGeometry(8193, 2049))

    def test_routed_geometry_keeps_activation_offsets_and_lossless_top8(self):
        for rows in (8193, 8194, 8195, 9216, 32768):
            old = PacketGeometry(rows, (rows+3)//4)
            g = PacketGeometry(rows, old.local_rows, routed=True)
            self.assertEqual(g.activation_bytes, old.stride)
            self.assertEqual(g.route_weights_offset, old.stride+g.local_rows*16)
            self.assertEqual(g.route_weights_offset % 4, 0)
            end = g.route_weights_offset+g.local_rows*32
            self.assertTrue(end <= g.stride < end+128)
            self.assertEqual(g.stride % 128, 0)
        g = PacketGeometry(32768, 8192, routed=True)
        self.assertEqual(g.workspace()['sender_roundtrip_bytes'], 64 << 20)
        self.assertEqual(g.workspace()['sender_router_fp32_bytes'], 128 << 20)
        self.assertEqual(g.workspace()['route_metadata_bytes'], 32768*48)
        with self.assertRaises(ValueError):
            PacketGeometry(8193, 2049, routed=1)

    def test_one_rank_missing_any_reader_falls_back_everywhere(self):
        def run(comm):
            supported = {3, 4, 5} if comm.rank != 2 else {3, 5}
            result = agreed_layers(comm, (0, 3, 4, 5), supported)
            self.assertEqual(comm.tripwire.calls, 1)
            return result
        self.assertEqual(LocalTP(4).run(run), [frozenset((3, 5))]*4)

    def test_duplicate_or_oversized_layer_votes_are_rejected(self):
        for layers in ((3, 3), range(65)):
            with self.assertRaises(ValueError):
                agreed_layers(None, layers, set())

    def test_short_prefill_and_decode_do_not_vote(self):
        from engine.profiles.glm53.net import Glm53Net
        net = Glm53Net.__new__(Glm53Net)
        net.prefill_ffn_packets, net.comm = True, Mock()
        for rows in (1, 7, 128, 8192, 32769):
            self.assertEqual(net._packet_ffn_layers(rows), frozenset())
        self.assertEqual(net.comm.mock_calls, [])

    def test_shared_projector_crops_before_the_gemm(self):
        from engine.kernels.dense import DenseLinear
        layer = DenseLinear.__new__(DenseLinear)
        layer.fp8 = NS(project_quantized=Mock(return_value='gemm'))
        layer.executed = 0
        layer.cols, layer.observer = 4096, None
        layer.fp8.observer = None
        q, s = object(), object()
        quant = Mock(return_value=(q, s))
        with patch.dict('sys.modules', {'engine.kernels.prefill_collectives.consumer': NS(quantize_gather=quant)}):
            self.assertEqual(layer.packet_projector()('packet', 2049, real_rows=8193), 'gemm')
        quant.assert_called_once_with('packet', 2049, real_rows=8193)
        layer.fp8.project_quantized.assert_called_once_with(q, s)
        for owner in (layer, layer.fp8):
            owner.observer = object()
            self.assertIsNone(layer.packet_projector())
            owner.observer = None

    def test_plan_refuses_layer_major_packet_execution(self):
        from engine.profiles.glm53.execution import ExecutionPlan
        with self.assertRaises(ValueError):
            ExecutionPlan(prefill_ffn_packets=True, prefill_tiles=2)
        self.assertIn('prefill_ffn_packets', ExecutionPlan(prefill_ffn_packets=True).label())


class PacketForwardTests(unittest.TestCase):
    def test_real_forward_routes_each_local_shard_once_and_keeps_one_exchange(self):
        from tests.test_engine_prefill_outputs import network, TokenShards as Transport
        from engine.profiles.glm53.net import Step

        for rows, missing in ((8193, None), (8194, 1), (8195, None), (8196, None)):
            with self.subTest(rows=rows, missing_rank=missing):
                def run(comm):
                    counts = dict(gather=0, packets=0, router=[], expert=[], shared=[])
                    net = network(comm, sharded=True)
                    net.F.is_moe, net.F.swiglu_limit = lambda L: True, 10.
                    net._moe = lambda L, x, reduce: reduce(x/4)
                    step = Step.prefill(torch.arange(rows), 0, 0, 1)
                    expected = net.forward(step, None, aux_layers=(1, 0, 1), last_hidden_only=True)

                    class PacketTransport(Transport):
                        def all_gather(self, x):
                            counts['gather'] += 1
                            return super().all_gather(x)

                        def all_gather_packets(self, x, *, rows, route):
                            counts['packets'] += 1
                            ids, _ = route(x)
                            counts['router'].append(len(x))
                            # The mocked data and routes share the same single
                            # collective, as they do in the CUDA byte packet.
                            joined = self.comm.all_gather(torch.cat((x, ids), dim=-1), dim=0)[:rows]
                            values, ids = joined.chunk(2, dim=-1)
                            return NS(received=values, ids=ids,
                                      geometry=PacketGeometry(rows, len(x), routed=True))

                    net.prefill_transport = PacketTransport(comm)
                    net.prefill_ffn_packets = True
                    net.prefill_packet_planned, net.prefill_packet_executed = set(), set()
                    net.prefill_packet_peak_bytes = 0
                    net._packet_capabilities = {L: lambda rows: True for L in net.layers}
                    def expert(batch, ids, weights):
                        counts['expert'].append(batch.received)
                        self.assertEqual(ids.shape[0], rows)
                        return ids/8
                    net._packet_experts = {L: expert for L in net.layers}
                    net.dense = {}
                    net._router_layers = set(net.layers)
                    net._router_weights = {L: object() for L in net.layers}
                    net._router_fp32 = set()
                    net._select_routes = lambda L, logits: (logits, None)
                    net._activation = lambda g, u, limit: g
                    net.linear = lambda x, name: x
                    for L in net.layers:
                        net.p[f'L{L}.moe.gate'] = NS(is_cuda=True, is_contiguous=lambda: True,
                                                    dtype=torch.bfloat16, shape=(288, 4096))
                        def project(received, local_rows, *, real_rows, routed):
                            self.assertIs(routed, True)
                            counts['shared'].append(received)
                            self.assertEqual((len(received), real_rows), (rows, rows))
                            return torch.cat((received/8, torch.zeros_like(received)), dim=-1)
                        enabled = not (comm.rank == missing and L == 0)
                        net.dense[f'L{L}.moe.sh_gate_up'] = NS(
                            packet_projector=(lambda: project) if enabled else (lambda: None))
                    actual = net.forward(step, None, aux_layers=(1, 0, 1), last_hidden_only=True)
                    for got, want in zip(actual, expected):
                        self.assertTrue(torch.equal(got, want))
                    layers = {0, 1} if missing is None else {1}
                    self.assertEqual(net.prefill_packet_executed, layers)
                    self.assertEqual(net.prefill_packet_planned, layers)
                    self.assertEqual(counts['packets'], len(layers))
                    self.assertEqual(counts['router'], [(rows+3)//4]*len(layers))
                    self.assertEqual(counts['gather'], 4-len(layers))  # two attention + ordinary FFNs
                    self.assertEqual(len(counts['expert']), len(layers))
                    for expert_input, shared_input in zip(counts['expert'], counts['shared']):
                        self.assertIs(expert_input, shared_input)
                    return actual

                with patch.dict('sys.modules', {
                        'engine.kernels.prefill_router': NS(router_shard_logits=lambda x, weight: x),
                        'engine.kernels.prefill_collectives.routes': NS(packet_routes=lambda batch: (batch.ids, None))}):
                    results = LocalTP(4).run(run)
                for got in results[1:]:
                    for value, expected in zip(got, results[0]):
                        self.assertTrue(torch.equal(value, expected))


class PacketProducerTests(unittest.TestCase):
    def test_frontend_evidence_uses_expert_token_identity_across_atomic_permutations(self):
        from probes.engine_ffn_packets_check import frontend
        def workspace(reverse):
            ws = NS(row_counts=torch.tensor([8]*8+[0]*280),
                    expert_tile_base=torch.tensor(list(range(9))+[8]*280),
                    token_map=torch.zeros(8*128, dtype=torch.int32),
                    token_weights=torch.zeros(8*128),
                    packed_input=torch.zeros(8*128, 2048, dtype=torch.uint8),
                    packed_input_scale=torch.zeros(8*128*256, dtype=torch.uint8))
            group = torch.arange(256)
            for expert in range(8):
                tokens = torch.arange(8).flip(0) if reverse else torch.arange(8)
                for row, token in enumerate(tokens):
                    r = expert*128+row
                    ws.token_map[r] = token
                    ws.token_weights[r] = (token+1)/8
                    ws.packed_input[r].fill_(expert*8+int(token))
                    offset = (r >> 7)*(64*512)+(r & 31)*16+((r >> 5) & 3)*4+(group >> 2)*512+(group & 3)
                    ws.packed_input_scale[offset] = (group+expert+token).to(torch.uint8)
            return ws
        a, b = workspace(False), workspace(True)
        self.assertEqual(frontend(a, 8), frontend(b, 8))
        b.packed_input_scale[0] ^= 1
        self.assertNotEqual(frontend(a, 8), frontend(b, 8))

    def test_packet_producer_preserves_the_entire_route_quantize_publish_body(self):
        folder = ROOT/'engine/kernels/b12x'
        source = folder/'moe_dynamic_gated_sf6_prefill.py'
        candidate = ast.parse((folder/'moe_dynamic_prefill_packets.py').read_text())
        expected = next(ast.literal_eval(n.value) for n in candidate.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'PREFILL_SOURCE_SHA256' for t in n.targets))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected)
        def method(tree):
            return copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                                      and n.name == 'initialize_route_q0_and_publish'))
        original, changed = method(ast.parse(source.read_text())), method(candidate)

        class InputCopyOnly(ast.NodeTransformer):
            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
                    'first_copy_tokens', 'second_copy_tokens', 'first_copy_bytes',
                    'second_copy_bytes', 'q0_bulk_phase', 'q0_ready'}:
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                code = ast.unparse(node)
                if (code.startswith('if first_copy_tokens >')
                        or ('q0_bulk_barrier_init(' in code and code.startswith('if tidx'))
                        or ('q0_cp_async_bulk(' in code and code.startswith('if warp_idx'))):
                    return None
                return self.generic_visit(node)

            def visit_While(self, node):
                if ast.unparse(node.test) == 'q0_ready == Int32(0)':
                    return None
                return self.generic_visit(node)

            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'stage_packet_input':
                    return None
                return self.generic_visit(node)

        self.assertEqual(ast.dump(InputCopyOnly().visit(original), include_attributes=False),
                         ast.dump(InputCopyOnly().visit(changed), include_attributes=False))
        cls = next(n for n in candidate.body if isinstance(n, ast.ClassDef))
        self.assertEqual([ast.unparse(b) for b in cls.bases], ['MoEGatedDynamicKernelSF6Prefill'])
        self.assertEqual({n.name for n in cls.body if isinstance(n, ast.FunctionDef)},
                         {'_setup_attributes', 'initialize_route_q0_and_publish'})


class PairedTimingTests(unittest.TestCase):
    def test_median_improvement_does_not_hide_slower_mean_or_paired_cycle(self):
        from probes.engine_ffn_packets_check import paired_summary
        result = paired_summary([[100., 10., 100., 100.], [95., 95., 95., 95.]])
        self.assertEqual(result['mean_ms'], [77.5, 95.])
        self.assertGreater(result['mean_change_pct'], 0)
        self.assertEqual(result['cycle_mean_ms'], [[55., 95.], [100., 95.]])
        self.assertGreater(result['cycle_change_pct'][0], 0)
        self.assertLess(result['cycle_change_pct'][1], 0)
        for invalid in ([[1.]*4, [1.]*2], [[1.]*3, [1.]*3]):
            with self.assertRaises(ValueError):
                paired_summary(invalid)


if __name__ == '__main__':
    unittest.main()
