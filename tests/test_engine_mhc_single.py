"""Compare the one-token-per-block probe with the actual packed mHC consumer."""
import importlib.util
import unittest

from tests.image_kernels import PRESENT, REASON

torch = None
if importlib.util.find_spec('torch'):
    import torch


class SingleGrid:
    def __init__(self, extension):
        self.extension = extension

    def run_mhc(self, *args):
        return self.extension.run_mhc(*args, single_token_grid=True)


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and PRESENT,
                     'requires CUDA; ' + REASON)
class SingleTokenMhcTests(unittest.TestCase):
    @staticmethod
    def inputs(rows):
        from engine.kernels.dense.mhc import MHC
        torch.manual_seed(91613)
        fn = (torch.randn(24, 16384, device='cuda') * .006).bfloat16().float()
        owner = MHC({'fn': fn})
        values = [torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16),
                  torch.randn(rows, 4, 4096, device='cuda', dtype=torch.bfloat16),
                  torch.rand(rows, 4, 1, device='cuda'), torch.rand(rows, 4, 4, device='cuda'),
                  torch.tensor([.2, .3, .4], device='cuda'), torch.randn(24, device='cuda') * .1,
                  torch.randn(4096, device='cuda', dtype=torch.bfloat16)]
        return owner, values

    @staticmethod
    def call(owner, values, single):
        previous = owner.ext
        if single:
            owner.ext = SingleGrid(previous)
        try:
            return owner('fn', *values, 1e-5, 1e-6, 2., 20)
        finally:
            owner.ext = previous

    def test_exact_changed_input_replay_and_rearmed_tickets(self):
        for rows in (1, 6, 7, 8):
            with self.subTest(rows=rows):
                owner, values = self.inputs(rows)
                outputs, graphs = [], []
                try:
                    for single in (False, True):
                        self.call(owner, values, single)
                        graph = torch.cuda.CUDAGraph()
                        graphs.append(graph)
                        with torch.cuda.graph(graph):
                            outputs.append(self.call(owner, values, single))
                    for scale in (0., .001, 1., 64., 1.):
                        for value in values[:4]:
                            value.normal_().mul_(scale)
                        # Reverse replay order too: both share native ticket
                        # counters, which must be reset at every launch exit.
                        for order in ((0, 1), (1, 0)):
                            for index in order:
                                graphs[index].replay()
                            for actual, expected in zip(outputs[1], outputs[0]):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                finally:
                    for graph in graphs:
                        graph.reset()

    def test_unpacked_coefficients_cannot_reach_the_probe_kernel(self):
        owner, values = self.inputs(7)
        ext = owner.ext
        with self.assertRaisesRegex(RuntimeError, 'single-token MHC requires packed'):
            ext.run_mhc([0] * 18, [1e-5, 1e-6, 1e-6, 2., 1e-5], [7, 20, 4096],
                        False, True, True)
        with self.assertRaisesRegex(RuntimeError, 'single-token MHC requires packed'):
            ext.run_mhc([0] * 18, [1e-5, 1e-6, 1e-6, 2., 1e-5], [7, 20, 5120],
                        True, True, True)


if __name__ == '__main__':
    unittest.main()
