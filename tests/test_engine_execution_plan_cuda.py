"""Private W4 scratch must survive overlapping, changing graph replays."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires the admitted GB10 GPU lane")
class PrivateWorkspaceTests(unittest.TestCase):
    def test_concurrent_projection_preserves_outputs_and_rearms_counters(self):
        from engine.kernels.dense import DenseLinear, extension
        if torch.cuda.get_device_capability() != (12, 1):
            self.skipTest("GB10 SM121 only")
        torch.manual_seed(9313)
        # Different inputs and row counts expose shared arrival/partial races.
        weights = (torch.randn(4096, 4096, device="cuda") * .02).bfloat16()
        layer = DenseLinear(weights, prefill=False)
        other = DenseLinear(weights, prefill=False)
        size = other.isolate_workspace()
        self.assertGreater(size, 0)
        side = torch.cuda.Stream()
        for rows in (7, 28):
            x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(7, 4096, device="cuda", dtype=torch.bfloat16)
            layer(x); other(y)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    parent = torch.cuda.current_stream()
                    side.wait_stream(parent)
                    with torch.cuda.stream(side):
                        b = other(y)
                    a = layer(x)
                    parent.wait_stream(side)
                for repeat in range(20):
                    x.normal_().mul_(1 + repeat / 10)
                    y.normal_().mul_(.1 + repeat / 10)
                    want_a = layer(x)
                    # Same packs and default workspace: only scratch ownership differs.
                    private = other.workspace
                    other.workspace = None
                    try:
                        want_b = other(y)
                    finally:
                        other.workspace = private
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(a, want_a, rtol=0, atol=0)
                    torch.testing.assert_close(b, want_b, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(private[-320:]).item(), 0)
            finally:
                graph.reset()


if __name__ == "__main__":
    unittest.main()
