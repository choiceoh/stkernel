"""Blackwell max3 must preserve NVFP4 scales/bytes, including exceptional inputs."""
import unittest
from pathlib import Path
import torch

from tests.image_kernels import PRESENT, REASON


@unittest.skipUnless(torch.cuda.is_available() and PRESENT, 'CUDA required; ' + REASON)
class FP4InstructionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 12:
            raise unittest.SkipTest('SM12x NVFP4 device required')
        from probes.engine_fp4_instructions import adversarial_inputs, compile_probe
        cls.x, cls.gs = adversarial_inputs()
        cls.calls = [compile_probe(v, cls.x, cls.gs) for v in (0, 1)]

    def compare(self):
        for call, _ in self.calls:
            call()
        for old, new in zip(self.calls[0][1], self.calls[1][1]):
            self.assertTrue(torch.equal(old.view(torch.uint8), new.view(torch.uint8)))

    def test_every_bf16_encoding_fp32_range_and_fp4_ties(self):
        self.compare()
        # Independent max oracle ignores NaNs, retains infinities and returns
        # +0 for an all-NaN block, matching the zero-initialized old reduction.
        expected = torch.where(torch.isnan(self.x), 0., self.x).abs().amax(-1)
        self.assertTrue(torch.equal(self.calls[1][1][2].view(torch.int32), expected.view(torch.int32)))

    def test_zero_global_scale_produces_zero_packed_values_and_scales(self):
        saved = self.gs.clone()
        try:
            for zero in (0., -0.):
                self.gs.fill_(zero)
                self.compare()
                packed, scale, _ = self.calls[1][1]
                self.assertEqual(torch.count_nonzero(packed).item(), 0)
                self.assertEqual(torch.count_nonzero(scale).item(), 0)
        finally:
            self.gs.copy_(saved)

    def test_current_stream_graph_replay_overwrites_poisoned_outputs(self):
        call, outputs = self.calls[1]
        call()
        expected = [o.clone() for o in outputs]
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                call()
            for output in outputs:
                output.fill_(42)
            graph.replay()
            for output, old in zip(outputs, expected):
                self.assertTrue(torch.equal(output.view(torch.uint8), old.view(torch.uint8)))
        finally:
            graph.reset()


class FP4InstructionCacheTests(unittest.TestCase):
    def test_device_helper_participates_in_persistent_cache_identity(self):
        try:
            from engine.kernels.b12x import moe_dispatch as md
        except ImportError as exc:
            self.skipTest(str(exc))
        helper = Path(md.__file__).with_name('fp4_quant.py').resolve()
        self.assertIn(helper, [Path(p).resolve() for p in md._kernel_source_files()])


if __name__ == '__main__':
    unittest.main()
