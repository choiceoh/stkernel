"""Projection boundary, graph replay and scratch reduction numerical contracts."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec('torch'):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires GB10 CUDA')
class ProjectionTests(unittest.TestCase):
    def test_kda_mixed_input_strides_and_declared_rows(self):
        from engine.kernels.decode_projection import KdaPair
        from probes.engine_decode_fusions import _capture
        torch.manual_seed(91327)
        weights = [torch.randn(2048, 128, device='cuda', dtype=torch.bfloat16) * .02 for _ in range(2)]
        owner = KdaPair(*weights)
        for rows in (1, 6, 7, 14, 21, 28):
            parent = torch.randn(rows, 6416, device='cuda', dtype=torch.bfloat16)
            a, b = parent[:, 6160:6288].contiguous(), parent[:, 6288:6416]
            graph, outputs = _capture(lambda: owner(a, b))
            try:
                for scale in (0., .01, 1., 16.):
                    a.normal_().mul_(scale); parent.normal_().mul_(scale)
                    graph.replay()
                    for actual, x, w in zip(outputs, (a, b), weights):
                        expected = torch.nn.functional.linear(x, w)
                        self.assertTrue(actual.isfinite().all().item())
                        error = (actual.float()-expected.float()).norm() / expected.float().norm().clamp_min(1e-10)
                        self.assertLessEqual(error.item(), .0005)
            finally:
                graph.reset()
        for rows in (0, 8, 32):
            with self.assertRaises(ValueError):
                owner(torch.empty(rows, 128, device='cuda', dtype=torch.bfloat16),
                      torch.empty(rows, 128, device='cuda', dtype=torch.bfloat16))

    def test_route_reduction_layout_and_replayed_zero_overwrite(self):
        from probes.engine_moe_scatter import _reduce_routes
        from probes.engine_decode_fusions import _capture
        for rows in (7, 28):
            # Exact integer fixtures distinguish all 32 route/part positions;
            # unchanged sentinels around the destination detect out-of-bounds stores.
            partial = torch.empty(rows, 8, 4, 4096, device='cuda')
            storage = torch.full((rows + 2, 4096), -123., device='cuda')
            out = storage[1:-1]
            graph, _ = _capture(lambda: _reduce_routes[(rows, 32)](
                partial, out, 4096, 32, 128, num_warps=4, enable_fp_fusion=False))
            try:
                for cycle in range(4):
                    partial.copy_((torch.arange(partial.numel(), device='cuda') % (13 + cycle)).reshape_as(partial))
                    if cycle == 3:
                        partial.zero_()
                    out.fill_(float('nan')); graph.replay()
                    torch.testing.assert_close(out, partial.sum((1, 2)), rtol=0, atol=0)
                    self.assertTrue(storage[0].eq(-123.).all().item())
                    self.assertTrue(storage[-1].eq(-123.).all().item())
                partial.zero_()
                tiny = torch.finfo(torch.float32).tiny
                partial[:, 0, 0, 0] = tiny / 2
                partial[:, 0, 1, 0] = tiny / 2
                partial[:, 0, 0, 1] = tiny
                partial[:, 0, 1, 1] = -tiny / 2
                graph.replay()
                self.assertTrue(out[:, 0].eq(0.).all().item())
                self.assertTrue(out[:, 1].eq(tiny).all().item())
            finally:
                graph.reset()


if __name__ == '__main__':
    unittest.main()
