"""Native routing projects every width on the tensor cores, not the seven-row decode step alone.

PR #789 opened `router_logits` (BF16 operands, FP32 accumulation) for M <= K+1 and left wider steps and
every prefill chunk on `x.float() @ gate.float().T`. The 2026-09-13 chunk profile priced that remainder:
`magma_sgemmEx` 121 ms plus the `x.float()` copy per 9,216-token chunk (13 us a token), and a cuBLAS SIMT
SGEMM of 1.7 ms in a four-row decode step. The GPU test (tests/test_engine_decode_seven.RouterTensorCoreTests)
pins the selection equal at 1..2,304 rows; this pins which path each width takes.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import torch

from engine.profiles.glm53.net import Glm53Net


def fake_net(prepared: bool):
    torch.manual_seed(7)
    p = {"L3.moe.gate": (torch.randn(32, 128) * .05).bfloat16(), "L3.moe.bias": torch.randn(32) * .1}
    return NS(F=NS(spec_k=6, topk_experts=8, routed_scale=2.5), p=p, lanes=NS(route_weights=None),
              _router_weights={3: None} if prepared else {}, _router_tensorcore=set())


class RouterWidthTests(unittest.TestCase):
    def test_every_width_takes_the_tensor_core_projection_when_routers_are_prepared(self):
        from engine.kernels import glm_pointwise
        widths = []

        def counted(x, gate):
            widths.append(x.shape[0])
            return x.float() @ gate.float().T                 # the same numbers on CPU; the GPU test pins the real kernel

        net = fake_net(prepared=True)
        with mock.patch.object(glm_pointwise, "router_logits", counted):
            for rows in (1, 7, 28, 2304):
                sel, w = Glm53Net.route(net, 3, torch.randn(rows, 128).bfloat16())
                self.assertEqual((sel.shape, sel.dtype, w.shape), ((rows, 8), torch.int32, (rows, 8)))
                torch.testing.assert_close(w.sum(-1), torch.full((rows,), 2.5), rtol=1e-5, atol=1e-5)
        self.assertEqual(widths, [1, 7, 28, 2304])
        self.assertEqual(net._router_tensorcore, {3})

    def test_stock_execution_keeps_the_fp32_projection_of_the_checkpoint_weight(self):
        from engine.kernels import glm_pointwise
        net = fake_net(prepared=False)
        with mock.patch.object(glm_pointwise, "router_logits", side_effect=AssertionError("stock execution must not")):
            sel, w = Glm53Net.route(net, 3, torch.randn(28, 128).bfloat16())
        self.assertEqual(sel.shape, (28, 8))
        self.assertEqual(net._router_tensorcore, set())

    def test_the_two_projections_select_the_same_experts_on_this_input(self):
        """Exact products either way (BF16 operands in FP32); only the summation order can differ."""
        net = fake_net(prepared=False)
        x = torch.randn(64, 128).bfloat16()
        ref, _ = Glm53Net.route(net, 3, x)
        net._router_weights = {3: None}
        with mock.patch.object(__import__("engine.kernels.glm_pointwise", fromlist=["router_logits"]),
                               "router_logits", lambda x, g: torch.mm(x.float(), g.float().T)):
            got, _ = Glm53Net.route(net, 3, x)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
