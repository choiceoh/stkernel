"""The one-launch decode router (2026-09-17 cell): its wrapper's admission, the source contract the probe
relies on, the probe's wiring, and the selection rule the kernel documents -- all on the CPU.

The kernel itself runs on the single-GPU lane (probes/engine_router_cells.py); here the rule it claims to
implement -- torch.topk's set over sigmoid(logits) + bias, an exact tie to the lower expert id, weights
from the raw sigmoid renormalised and scaled -- is pinned against the engine's own unbound reference.
"""
import ast
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def _kernel_rule(logits, bias, topk=8, scale=2.5):
    """The documented selection: descending score, ties to the lower id; weights from the raw sigmoid."""
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
        self.assertEqual(ast.literal_eval(constants['ARMS']), ('served', 'served_b', 'fused'))
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
        net.fused_decode_router = True
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
