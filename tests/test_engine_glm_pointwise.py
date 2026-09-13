"""Actual GLM shapes, rounding, routing ties and changed-input graph replay."""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "native kernels require CUDA")
class PointwiseTests(unittest.TestCase):
    def graph_check(self, fn, inputs, expected, *, rtol, atol):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn(*inputs)
        for iteration in range(3):
            for x in inputs:
                x.copy_(torch.randn_like(x) * (iteration + 1))
            graph.replay()
            torch.testing.assert_close(output, expected(*inputs), rtol=rtol, atol=atol)

    def test_clamped_activation_on_strided_projection_halves(self):
        from engine.kernels.glm_pointwise import swiglu_clamped
        from engine.profiles.glm53.lanes import swiglu_clamped as reference
        for rows, width in ((1, 1280), (7, 1280), (28, 4096), (257, 95)):
            fused = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16) * 20
            g, u = fused.chunk(2, -1)
            torch.testing.assert_close(swiglu_clamped(g, u, 10.), reference(g, u, 10.), rtol=.008, atol=1e-6)
            self.graph_check(lambda g, u: swiglu_clamped(g, u, 10.), (g, u),
                             lambda g, u: reference(g, u, 10.), rtol=.008, atol=1e-6)

    def test_router_preserves_topk_ties_and_unbiased_weights(self):
        from engine.kernels.glm_pointwise import route_weights
        def reference(x, bias):
            s = x.sigmoid()
            selected = (s + bias).topk(8, -1).indices
            w = s.gather(-1, selected)
            return selected.int(), w / w.sum(-1, keepdim=True) * 2.5
        for rows in (1, 7, 28, 256):
            x = torch.randn(rows, 288, device="cuda") * 3
            bias = torch.randn(288, device="cuda") * .02
            for tied in (False, True):
                if tied:
                    x.zero_(); bias.zero_()
                actual = route_weights(x, bias, 8, 2.5)
                wanted = reference(x, bias)
                torch.testing.assert_close(actual[0], wanted[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], wanted[1], rtol=4e-7, atol=1e-7)
            self.graph_check(lambda x, bias: route_weights(x, bias, 8, 2.5), (x, bias),
                             reference, rtol=4e-7, atol=1e-7)

    def test_indexer_layernorm_and_target_rmsnorm(self):
        from engine.kernels.glm_pointwise import layernorm
        from engine.kernels.common.norm_rope import norm
        from engine.profiles.glm53.net import rmsnorm
        for rows, width in ((1, 128), (7, 512), (28, 1536), (257, 4096)):
            x = torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)[:, :width]
            w = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            self.graph_check(lambda x, w: norm(x, w, 1e-6), (x, w),
                             lambda x, w: rmsnorm(x, w, 1e-6), rtol=.016, atol=1e-6)
        for rows in (1, 7, 28, 257):
            x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)[:, :128]
            w, bias = (torch.randn(128, device="cuda") for _ in range(2))
            self.graph_check(lambda x, w, b: layernorm(x, w, b, 1e-6), (x, w, bias),
                             lambda x, w, b: torch.nn.functional.layer_norm(x.float(), (128,), w, b, 1e-6).bfloat16(),
                             rtol=.008, atol=2e-6)

    def test_bf16_expert_join_retains_fp32_sum_rounding(self):
        for rows in (1, 7, 28, 257):
            a, b = (torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(2))
            torch.testing.assert_close(a + b, (a.float() + b.float()).bfloat16(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
