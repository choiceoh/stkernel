"""Current C=1 verification width must retain every output of the W4 lane."""
import unittest

import torch

from tests.image_kernels import PRESENT, REASON


@unittest.skipUnless(torch.cuda.is_available() and PRESENT, 'requires CUDA; ' + REASON)
class SevenRowDenseTests(unittest.TestCase):
    def test_input_reuse_reads_strided_tiles_and_preserves_wide_pack_folding(self):
        from engine.kernels.dense import DenseLinear, extension
        ext = extension()
        before, mode, state = ext.gemm_input_mode(), ext.gemm_input_cta_mode(), ext.probe_state()
        torch.manual_seed(91408)
        try:
            ext.set_input_cta(4)
            ext.set_gemm2(0)
            for n in (6416, 4096, 6144):
                layer = DenseLinear((torch.randn(n, 4096, device='cuda') * .02).bfloat16(), prefill=False)
                for rows in (6, 7):
                    for stride, offset in ((20480, 0), (20480, 4096), (20480, 16384), (4104, 4)):
                        parent = torch.full((rows, stride), float('nan'), device='cuda', dtype=torch.bfloat16)
                        x = parent[:, offset:offset+4096]
                        x.normal_()
                        self.assertFalse(x.is_contiguous())
                        ext.set_gemm_input(1)
                        layer(x)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = layer(x)
                        try:
                            for magnitude in (0., .001, 1., 50.):
                                x.normal_().mul_(magnitude)
                                ext.set_gemm_input(0)
                                expected = layer(x.contiguous())
                                graph.replay()
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        finally:
                            graph.reset()
            # The drafter's uncalibrated 20K context projection keeps five
            # tiles with distinct row scales. Each addend rounds to BF16.
            weight = torch.randn(4096, 20480, device='cuda', dtype=torch.bfloat16)
            for tile in range(5):
                weight[:, tile*4096:(tile+1)*4096].mul_(2.**tile)
            wide = DenseLinear(weight, prefill=False)
            self.assertEqual(len(wide.packs), 5)
            for rows in (6, 7):
                x = torch.randn(rows, 20480, device='cuda', dtype=torch.bfloat16)
                ext.set_gemm_input(1)
                wide(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = wide(x)
                try:
                    for magnitude in (0., .01, 1.):
                        x.normal_().mul_(magnitude)
                        ext.set_gemm_input(0)
                        expected = wide(x)
                        graph.replay()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                finally:
                    graph.reset()
        finally:
            ext.set_gemm_input(before)
            ext.set_input_cta(mode)
            ext.restore_probe_state(state)

    def test_input_reuse_matches_same_packs_every_row_and_changed_graph_inputs(self):
        from engine.kernels.dense import DenseLinear, extension
        ext = extension()
        before, mode = ext.gemm_input_mode(), ext.gemm_input_cta_mode()
        state = ext.probe_state()
        torch.manual_seed(91407)
        try:
            ext.set_input_cta(4)
            ext.set_gemm2(0)
            for n in (6416, 4096, 6144):
                layer = DenseLinear((torch.randn(n, 4096, device='cuda') * .02).bfloat16(), prefill=False)
                for rows in (6, 7):
                    x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
                    ext.set_gemm_input(1)
                    plan = ext.gemm_input_plan(rows, n, 4096, False, False)
                    self.assertTrue(plan[0], (rows, n, plan))
                    self.assertEqual(plan[1], ext.gemm2_plan(rows, n, 4096)[0])
                    self.assertFalse(ext.gemm_input_plan(rows, n, 4096, True, False)[0])
                    self.assertFalse(ext.gemm_input_plan(rows, n, 4096, False, True)[0])
                    layer(x)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = layer(x)
                    try:
                        for magnitude in (0., .001, 1., 50.):
                            x.normal_().mul_(magnitude)
                            ext.set_gemm_input(0)
                            expected = layer(x)
                            graph.replay()
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    finally:
                        graph.reset()
                ext.set_gemm_input(1)
                for rows in (1, 5, 8, 14, 28, 32):
                    self.assertFalse(ext.gemm_input_plan(rows, n, 4096, False, False)[0])
        finally:
            ext.set_gemm_input(before)
            ext.set_input_cta(mode)
            ext.restore_probe_state(state)


@unittest.skipUnless(torch.cuda.is_available(), 'tensor-core router requires CUDA')
class RouterTensorCoreTests(unittest.TestCase):
    def test_fp32_logits_and_selection_on_random_repeated_and_tied_experts(self):
        from engine.kernels.glm_pointwise import router_logits, route_weights
        torch.manual_seed(91507)
        for rows in (1, 7, 28, 2304):                    # a decode row, the seven-row step, four rows, a prefill chunk
            x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
            gate = (torch.randn(288, 4096, device='cuda') * .02).bfloat16()
            bias = torch.randn(288, device='cuda') * .1
            for tied in (False, True):
                if tied:
                    gate[1::2].copy_(gate[::2])
                    bias[1::2].copy_(bias[::2])
                expected = x.float() @ gate.float().T
                actual = router_logits(x, gate)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=5e-6)
                ids, weights = route_weights(actual, bias, 8, 2.5)
                ref_ids, ref_weights = route_weights(expected, bias, 8, 2.5)
                torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
                torch.testing.assert_close(weights, ref_weights, rtol=3e-5, atol=3e-6)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = router_logits(x, gate)
            try:
                for _ in range(3):
                    x.normal_()
                    expected = router_logits(x, gate)
                    graph.replay()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
            finally:
                graph.reset()


if __name__ == '__main__':
    unittest.main()
