"""The one-launch decode router (2026-09-17 cell): its wrapper's admission, the source contract the probe
relies on, the probe's wiring, and the selection rule the kernel documents -- all on the CPU.

The kernel itself runs on the single-GPU lane (probes/engine_router_cells.py), which checks exact order
and weights against the served CUDA path. CPU tests cover the selection set, routing integration and
counter ownership; CPU torch.topk does not define the CUDA tie permutation.
"""
import ast
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def _kernel_rule(logits, bias, topk=8, scale=2.5):
    """Selection-set oracle, shown in lower-id tie order (not the CUDA output permutation)."""
    s = torch.sigmoid(logits)
    scores = s + bias
    rows, experts = scores.shape
    ids = torch.empty((rows, topk), dtype=torch.int32)
    for r in range(rows):
        order = sorted(range(experts), key=lambda e: (-float(scores[r, e]), e))
        ids[r] = torch.tensor(order[:topk], dtype=torch.int32)
    w = s.gather(1, ids.long())
    return ids, w / (w.sum(1, keepdim=True) + 1e-20) * scale


class RouterFusedTests(unittest.TestCase):
    def test_arrival_counters_are_isolated_between_streams_and_reused_within_one(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from engine.kernels import router_fused as rf
        device = torch.device('cuda:0')
        with patch.dict(rf._TICKETS, {}, clear=True), \
             patch.object(torch.cuda, 'current_stream', return_value=NS(cuda_stream=101)) as current, \
             patch.object(torch, 'zeros', side_effect=lambda *a, **k: object()) as allocate:
            first = rf._ticket(device)
            self.assertIs(rf._ticket(device), first)
            current.return_value = NS(cuda_stream=202)
            second = rf._ticket(device)
            self.assertIsNot(first, second)
            current.return_value = NS(cuda_stream=101)
            self.assertIs(rf._ticket(device), first)
            self.assertEqual(allocate.call_count, 2)

    def test_wrapper_admits_only_the_served_shapes_before_any_build(self):
        from engine.kernels import router_fused as rf
        self.assertIsNone(rf._EXT)
        gate = torch.zeros(288, 4096)
        bias = torch.zeros(288)
        with self.assertRaisesRegex(ValueError, r'\[1\.\.16, 4096\]'):
            rf.route(torch.zeros(17, 4096, dtype=torch.bfloat16), gate, bias, 8, 2.5)
        with self.assertRaisesRegex(ValueError, r'\[1\.\.16, 4096\]'):
            rf.route(torch.zeros(8, 4096, dtype=torch.bfloat16), gate, bias, 8, 2.5)   # CPU tensors never pass
        self.assertIsNone(rf._EXT)

    def test_kernel_source_keeps_the_contract_the_cell_measures(self):
        src = (ROOT / 'engine/kernels/router_fused.cu').read_text()
        for needle in ('#define ST_RT_CTAS 96', '#define ST_RT_PER_CTA 3', '#define ST_RT_TOPK 8',
                       '__launch_bounds__(ST_RT_THREADS, 2)', 'float4 g[ST_RT_LANE_CHUNKS][ST_RT_PER_CTA];',
                       'atomicAdd(ticket, 1u) == gridDim.x - 1', '*ticket = 0u;', '__threadfence();',
                       '__fdiv_rn(1.0f, 1.0f + expf(-v))', '__fdiv_rn(s, sum + 1e-20f) * scale', '__ldcs(',
                       'ob > best || (ob == best && oe < be)', 'fmaf(g[j][e].x, a, s)',
                       'if constexpr (ROWS <= 8) {', 'fmaf(g[j][e].x, xa[t], s)'):
            self.assertIn(needle, src)
        self.assertNotIn('__expf', src)          # the accurate libdevice exp, as Triton's libdevice.exp
        self.assertNotIn('__fdividef', src)
        py = (ROOT / 'engine/kernels/router_fused.py').read_text()
        self.assertIn("'-O3', '-gencode', 'arch=compute_121a,code=sm_121a'", py)
        self.assertNotIn('use_fast_math', py)

    def test_probe_is_wired_and_reads_the_served_route(self):
        check = (ROOT / 'probes/engine_kernel_check.py').read_text()
        self.assertIn("args.lanes == 'router_cells'", check)
        tree = ast.parse((ROOT / 'probes/engine_router_cells.py').read_text())
        constants = {n.targets[0].id: n.value for n in tree.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.literal_eval(constants['FIXTURES']), (('c2_two_requests', 16, 2), ('c1_one_request', 8, 1)))
        self.assertEqual(ast.literal_eval(constants['ARMS']), ('served', 'served_b', 'fused_legacy', 'fused'))
        self.assertEqual(ast.literal_eval(constants['MAX_ULPS']), 64)
        self.assertIn('range(3, 45)', ast.unparse(constants['LAYERS']))

    def test_documented_selection_rule_matches_the_engine_reference_off_ties(self):
        torch.manual_seed(7)
        logits = torch.randn(16, 288) * 3
        bias = torch.randn(288) * 0.1
        ids, w = _kernel_rule(logits, bias)
        s = torch.sigmoid(logits)
        ref = (s + bias).topk(8, dim=-1).indices.to(torch.int32)
        self.assertTrue(torch.equal(ids, ref))
        wr = s.gather(1, ref.long())
        wr = wr / (wr.sum(1, keepdim=True) + 1e-20) * 2.5
        self.assertTrue(torch.allclose(w, wr, rtol=0, atol=0))
        # an exact tie at the boundary: the rule keeps the lower id (torch's order there is implementation-defined)
        tied = logits.clone()
        tied[0, 100] = tied[0, 5] = 50.
        tb = torch.zeros(288)
        ids_t, _ = _kernel_rule(tied, tb)
        self.assertEqual(ids_t[0, 0].item(), 5)
        self.assertEqual(ids_t[0, 1].item(), 100)

class RouterConsumerTests(unittest.TestCase):
    def test_default_admits_only_the_qualified_glm53_k7_geometry(self):
        from dataclasses import replace
        from types import SimpleNamespace as NS
        from engine.profiles.glm53 import facts
        from engine.profiles.glm53.net import Glm53Net
        from tests.test_engine_kernel_shape import GLM53_TEXT_CONFIG
        profile = facts.architecture(GLM53_TEXT_CONFIG)
        for change in ({}, {'hidden': 128}, {'experts': 32}, {'topk_experts': 4}, {'spec_k': 3}):
            with self.subTest(change=change):
                net = Glm53Net(replace(profile, **change), NS(rank=0, world_size=4),
                               NS(rmsnorm=None, swiglu=None, route_weights=None), [3])
                self.assertFalse(net.fused_decode_router)

    def test_fusion_preserves_the_capture_route_skip_by_using_the_common_path(self):
        from types import MethodType, SimpleNamespace as NS
        from unittest.mock import patch
        from engine.profiles.glm53.net import Glm53Net
        from engine.kernels import glm_pointwise, router_fused
        for rows in (8, 16):
            for skip in (.05, {3: .05}):
                ids = torch.arange(8, dtype=torch.int32).repeat(rows, 1)
                weights = torch.tensor([[.9, .5, .4, .3, .2, .1, .06, .04]]).repeat(rows, 1)
                net = NS(F=NS(topk_experts=8, routed_scale=2.5), p={'L3.moe.bias': None},
                         lanes=NS(route_weights=lambda *a: (ids, weights)), route_skip=skip,
                         fused_decode_router=True, _router_layers={3}, decode_fastpath_rows=(8, 16),
                         _router_weights={3: None}, _router_fused_bias={3: None},
                         _router_fp32=set(), _router_fused_executed=set())
                net._select_routes = MethodType(Glm53Net._select_routes, net)
                expected = net._select_routes(3, None)
                with patch.object(router_fused, 'route', side_effect=AssertionError('bypassed skip')), \
                     patch.object(glm_pointwise, 'router_logits', return_value=None):
                    actual = Glm53Net.route(net, 3, torch.empty(rows, 4096, dtype=torch.bfloat16))
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                self.assertEqual((actual[1] != 0).sum(-1).tolist(), [5] * rows)

    def test_bias_is_budgeted_resident_fp32_and_only_bound_rows_use_fusion(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from engine.base.arena import Arena
        from engine.profiles.glm53 import facts
        from engine.profiles.glm53.net import Glm53Net
        from engine.kernels import glm_pointwise, router_fused
        from tests.test_engine_kernel_shape import GLM53_TEXT_CONFIG
        net = Glm53Net(facts.architecture(GLM53_TEXT_CONFIG), NS(rank=0, world_size=4),
                       NS(rmsnorm=None, swiglu=None, route_weights=None), [3])
        gate = torch.zeros(288, 4096, dtype=torch.bfloat16)
        bias = torch.randn(288).bfloat16()
        net.p = {'L3.moe.gate': gate, 'L3.moe.bias': bias}
        self.assertTrue(net.fused_decode_router)
        net.decode_fastpath_rows = (8, 16)
        arena = Arena(net.router_nbytes(), device='cpu')
        net.prepare_routers(arena)
        self.assertEqual(arena.remaining, 0)
        resident = net._router_fused_bias[3]
        self.assertEqual(resident.untyped_storage().data_ptr(), arena.buf.data_ptr())
        torch.testing.assert_close(resident, bias.float(), rtol=0, atol=0)
        fused, plain = [], []
        def route(x, actual_gate, actual_bias, topk, scale):
            self.assertIs(actual_gate, net._router_weights[3])
            self.assertIs(actual_bias, resident)
            fused.append(x.shape[0])
            return torch.zeros(x.shape[0], 8, dtype=torch.int32), torch.zeros(x.shape[0], 8)
        def logits(x, actual_gate):
            self.assertIs(actual_gate, net._router_weights[3])
            plain.append(x.shape[0])
            return torch.zeros(x.shape[0], 288)
        with patch.object(router_fused, 'route', route), patch.object(glm_pointwise, 'router_logits', logits):
            for rows in (1, 7, 8, 16, 24, 32):
                net.route(3, torch.zeros(rows, 4096, dtype=torch.bfloat16))
            net.decode_fastpath_rows = (8,)
            net.route(3, torch.zeros(16, 4096, dtype=torch.bfloat16))
            net.fused_decode_router = False
            net.route(3, torch.zeros(8, 4096, dtype=torch.bfloat16))
        self.assertEqual(fused, [8, 16])
        self.assertEqual(plain, [1, 7, 24, 32, 16, 8])
        self.assertEqual(net._router_fused_executed, {(3, 8), (3, 16)})
        self.assertEqual(net._router_fp32, {3})


if __name__ == '__main__':
    unittest.main()
