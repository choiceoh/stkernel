"""Opt-in device numerical/worker checks. CPU CI skips; no fleet reservation here."""
import unittest

import torch

from engine.modules.speculative_tree import Tree
from engine.modules.tree_kda import verify


@unittest.skipUnless(torch.cuda.is_available(), "requires an explicitly available CUDA device")
class NativeTreeDataflowTests(unittest.TestCase):
    def test_w4a8_workers_preserve_existing_dense_lane(self):
        if torch.cuda.get_device_capability() != (12, 1):
            self.skipTest("persistent dataflow is an SM121-only experiment")
        from dataclasses import replace
        from engine.kernels.dense import w4_gemm
        from engine.kernels.tile_dataflow import execute_w4a8
        from engine.modules.w4a8_dataflow import W4A8Plan, W4A8Weights
        from engine.kernels.glm_pointwise import swiglu_clamped
        from tests.test_engine_w4a8_dataflow import packed_weights
        w = packed_weights()
        def move(p):
            return replace(p, data=p.data.cuda(), scale=p.scale.cuda(), rowscale=p.rowscale.cuda())
        device = W4A8Weights(move(w.gate_up), move(w.down))
        for rows in (1, 4, 16, 32):
            torch.manual_seed(814)
            x = torch.randn(rows, w.hidden, device="cuda").bfloat16()
            gate, up = w4_gemm(x, device.gate_up).chunk(2, -1)
            expected = w4_gemm(swiglu_clamped(gate, up, 10.), device.down)
            for workers in (1, 4, 48):
                plan = W4A8Plan(rows, w.hidden, w.intermediate, workers=workers)
                for _ in range(2):
                    actual, error = execute_w4a8(plan, x, device, 10.)
                    self.assertEqual(error, 0)
                    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=.01)

    def test_tree_outputs_and_accepted_fp32_state(self):
        torch.manual_seed(532)
        tree = Tree((1, 2, 3, 4, 5, 6), (-1, 0, 0, 1, 2, 4))
        q, k, v, g = [torch.randn(6, 2, 128).bfloat16() for _ in range(4)]
        beta, a, bias, initial = torch.randn(6, 2), torch.randn(2), torch.randn(256), torch.randn(2, 128, 128)*.1
        args = (q, k, v, g, beta, a, bias, initial)
        expected, factors = verify(tree, *args, -5.)
        actual, device = verify(tree, *(t.cuda() for t in args), -5.)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=.01)
        for node in range(6):
            torch.testing.assert_close(device.state(node).cpu(), factors.state(node), atol=2e-5, rtol=1e-3)
        self.assertEqual(device.initial.dtype, torch.float32)

    def test_nvfp4_workers_raw_tiled_sf6_and_repeated_invocations(self):
        if torch.cuda.get_device_capability() != (12, 1):
            self.skipTest("persistent dataflow is an SM121-only experiment")
        from dataclasses import replace
        from engine.kernels.tile_dataflow import execute_nvfp4
        from engine.modules.nvfp4_dataflow import NVFP4Plan, reference
        from engine.profiles.glm53.modelopt_scales import ModelOptScales
        from tests.test_engine_nvfp4_dataflow import packed_weights
        torch.manual_seed(901)
        for tiled, sf6 in ((False, False), (True, False), (True, True)):
            w = packed_weights(tiled=tiled, sf6=sf6)
            x = torch.randn(4, w.hidden).bfloat16()
            expected, _ = reference(NVFP4Plan(4, w.hidden, w.intermediate), x, w, 10.)
            s = w.scales
            ds = ModelOptScales.bind(*(t.cuda() for t in (s.weight13, s.input13, s.weight2, s.input2)),
                                     experts=1, device=torch.device("cuda:0"))
            device = replace(w, w13=w.w13.cuda(), w2=w.w2.cuda(), sf13=w.sf13.cuda(), sf2=w.sf2.cuda(), scales=ds)
            for workers in (1, 4, 48):
                plan = NVFP4Plan(4, w.hidden, w.intermediate, workers=workers)
                for _ in range(2):
                    actual, error = execute_nvfp4(plan, x.cuda(), device, 10.)
                    self.assertEqual(error, 0)
                    torch.testing.assert_close(actual.cpu(), expected, atol=3e-4, rtol=.02)


if __name__ == "__main__":
    unittest.main()
