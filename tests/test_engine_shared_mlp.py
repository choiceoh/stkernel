"""Shared MLP arithmetic, observer inputs and reuse of native graph scratch."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires GB10 CUDA")
class SharedMLPTests(unittest.TestCase):
    @staticmethod
    def layers(width=512, seed=131):
        from engine.kernels.dense import DenseLinear
        from engine.kernels.dense.shared_mlp import SharedMLP
        torch.manual_seed(seed)
        gu = DenseLinear((torch.randn(2 * width, 4096, device="cuda") * .02).bfloat16(), prefill=False)
        down = DenseLinear((torch.randn(4096, width, device="cuda") * .02).bfloat16(), prefill=False)
        return gu, down, SharedMLP(gu, down, 10.)

    @staticmethod
    def reference(x, gu, down):
        from engine.kernels.glm_pointwise import swiglu_clamped
        gate, up = gu(x).chunk(2, -1)
        return down(swiglu_clamped(gate, up, 10.))

    def close(self, actual, expected):
        self.assertTrue(torch.isfinite(actual).all().item())
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-10)
        # The existing CUDA expf and Triton libdevice sigmoid can differ at
        # a BF16 rounding tie before FP8 quantization. The same-pack MLP
        # error budget is deliberately much tighter than W4 vs BF16's .16.
        self.assertLessEqual(error.item(), .002)

    def test_same_packs_clamping_and_declared_decode_widths(self):
        for width in (512, 1280):
            gu, down, fused = self.layers(width)
            for rows in (1, 7, 14, 28, 32):
                for scale in (0., 1., 15.):
                    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * scale
                    self.close(fused(x), self.reference(x, gu, down))
            self.assertTrue(fused.executed and gu.executed & 1 and down.executed & 1)
            with self.assertRaisesRegex(ValueError, "1..32"):
                fused(torch.empty(33, 4096, device="cuda", dtype=torch.bfloat16))

    def test_observers_receive_inputs_and_actual_bf16_activation(self):
        from engine.kernels.glm_pointwise import swiglu_clamped
        gu, down, fused = self.layers()
        x = torch.randn(7, 4096, device="cuda", dtype=torch.bfloat16)
        g, u = gu(x).chunk(2, -1)
        expected = swiglu_clamped(g, u, 10.)
        seen = {}
        mask = torch.tensor([1, 1, 0, 1, 0, 0, 1], device="cuda", dtype=torch.bool)
        gu.observer = lambda value, rows: seen.update(gu=value.clone(), gu_mask=rows)
        down.observer = lambda value, rows: seen.update(down=value.clone(), down_mask=rows)
        fused(x, mask)
        self.assertTrue(torch.equal(seen['gu'], x))
        self.assertIs(seen['gu_mask'], mask)
        self.assertIs(seen['down_mask'], mask)
        self.assertEqual(seen['down'].dtype, torch.bfloat16)
        torch.testing.assert_close(seen['down'], expected, rtol=.008, atol=1e-7)

    def test_replayed_changed_inputs_and_other_layer_between_calls(self):
        gu, down, fused = self.layers()
        gu2, down2, second = self.layers(seed=132)
        for rows in (7, 28):
            x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
            other = torch.randn_like(x)
            fused(x); second(other)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                a, b, c = fused(x), second(other), fused(x)
            try:
                for _ in range(5):
                    x.normal_(); other.normal_()
                    expected = self.reference(x, gu, down)
                    expected2 = self.reference(other, gu2, down2)
                    graph.replay()
                    self.close(a, expected); self.close(b, expected2)
                    self.assertTrue(torch.equal(a, c))
            finally:
                graph.reset()

    def test_projection_parent_stride_and_graph_replay_are_exact(self):
        from engine.kernels.dense import DenseLinear
        weight = (torch.randn(2048, 128, device="cuda") * .02).bfloat16()
        layer = DenseLinear(weight, prefill=False)
        for rows in (1, 7, 28):
            parent = torch.randn(rows, 6416, device="cuda", dtype=torch.bfloat16)
            x = parent[:, 6160:6288]
            expected = layer(x.contiguous())
            self.assertTrue(torch.equal(layer(x), expected))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = layer(x)
            try:
                for _ in range(3):
                    parent.normal_()
                    expected = layer(x.contiguous())
                    graph.replay()
                    self.assertTrue(torch.equal(out, expected))
            finally:
                graph.reset()


if __name__ == "__main__":
    unittest.main()
