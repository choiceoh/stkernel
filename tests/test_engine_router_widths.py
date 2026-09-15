"""Every decode/prefill width must project the resident FP32 gate."""
import unittest
from types import MethodType, SimpleNamespace as NS
from unittest import mock

import torch

from engine.profiles.glm53.net import Glm53Net


def fake_net(prepared: bool):
    torch.manual_seed(7)
    p = {"L3.moe.gate": (torch.randn(32, 128) * .05).bfloat16(), "L3.moe.bias": torch.randn(32) * .1}
    net = NS(F=NS(spec_k=7, topk_experts=8, routed_scale=2.5), p=p, lanes=NS(route_weights=None),
             _router_layers={3} if prepared else None,
             _router_weights={3: p["L3.moe.gate"].float()} if prepared else {}, _router_fp32=set())
    net._select_routes = MethodType(Glm53Net._select_routes, net)
    return net


class RouterWidthTests(unittest.TestCase):
    def test_every_width_reads_the_resident_fp32_gate(self):
        from engine.kernels import glm_pointwise
        widths = []
        net = fake_net(prepared=True)
        net.p['L3.moe.gate'].zero_()

        def counted(x, gate):
            widths.append(x.shape[0])
            self.assertIs(gate, net._router_weights[3])
            self.assertEqual(gate.dtype, torch.float32)
            return x.float() @ gate.T

        # Include both production decode widths and the old long-prefill
        # dispatch boundaries. A native path must never select the BF16 copy.
        rows_to_check = (1, 7, 8, 16, 28, 2304, 8192, 8193, 9216, 32768, 32769)
        with mock.patch.object(glm_pointwise, "router_logits", counted):
            for rows in rows_to_check:
                sel, w = Glm53Net.route(net, 3, torch.randn(rows, 128).bfloat16())
                self.assertEqual((sel.shape, sel.dtype, w.shape), ((rows, 8), torch.int32, (rows, 8)))
                torch.testing.assert_close(w.sum(-1), torch.full((rows,), 2.5), rtol=1e-5, atol=1e-5)
        self.assertEqual(widths, list(rows_to_check))
        self.assertEqual(net._router_fp32, {3})

    def test_stock_execution_projects_the_checkpoint_in_fp32(self):
        from engine.kernels import glm_pointwise
        net = fake_net(prepared=False)
        with mock.patch.object(glm_pointwise, "router_logits", side_effect=AssertionError("unprepared resident")):
            sel, w = Glm53Net.route(net, 3, torch.randn(28, 128).bfloat16())
        self.assertEqual(sel.shape, (28, 8))
        self.assertEqual(net._router_fp32, set())

    def test_native_and_reference_routes_and_weights_agree(self):
        from engine.kernels import glm_pointwise
        net = fake_net(prepared=False)
        x = torch.randn(64, 128).bfloat16()
        reference = Glm53Net.route(net, 3, x)
        net._router_layers, net._router_weights = {3}, {3: net.p["L3.moe.gate"].float()}
        with mock.patch.object(glm_pointwise, "router_logits", lambda x, g: torch.mm(x.float(), g.T)):
            actual = Glm53Net.route(net, 3, x)
        for got, expected in zip(actual, reference):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_sender_uses_the_same_resident_and_records_execution(self):
        from engine.kernels import prefill_router
        net = fake_net(prepared=True)
        x = torch.randn(2304, 128).bfloat16()
        def project(x, gate):
            self.assertIs(gate, net._router_weights[3])
            return x.float() @ gate.T
        with mock.patch.object(prefill_router, "router_shard_logits", project):
            actual = Glm53Net._sender_routes(net, 3, x)
        expected = net._select_routes(3, x.float() @ net._router_weights[3].T)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(net._router_fp32, {3})


if __name__ == "__main__":
    unittest.main()
