"""CPU fixture checks; run with the probe image's PyTorch/Triton installed."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
try:
    import torch
    import triton
except ModuleNotFoundError as exc:
    if exc.name not in ("torch", "triton"):
        raise
    torch = None
else:
    from kda_prefill_direct_out import fixture, paired_summary, sequence_cases


@unittest.skipIf(torch is None, "probe fixture checks require CPU torch and triton")
class PrefillProbeTests(unittest.TestCase):
    def test_fixture_shapes_and_layouts(self):
        for lengths in ((1,), (65,), (1, 63, 64, 65, 127)):
            reference = fixture(lengths, "contiguous", "cpu")
            for layout in ("contiguous", "conv", "qkv"):
                with self.subTest(lengths=lengths, layout=layout):
                    inp = fixture(lengths, layout, "cpu")
                    for name in ("q", "k", "v", "raw_g"):
                        self.assertEqual(inp[name].shape, (1, sum(lengths), 16, 128))
                        self.assertEqual(inp[name].dtype, torch.bfloat16)
                        self.assertTrue(torch.equal(inp[name], reference[name]))
                    self.assertEqual(inp["initial_state"].shape, (len(lengths), 16, 128, 128))
                    self.assertEqual(inp["beta"].dtype, torch.float32)
                    self.assertEqual(inp["cu_seqlens"].diff().tolist(), list(lengths))
                    if sum(lengths) > 1:
                        self.assertEqual(inp["v"].is_contiguous(), layout == "contiguous")
                    else:
                        # A singleton conv view aliases v.contiguous(), so
                        # stock overwrites it despite the layout's name.
                        self.assertTrue(inp["v"].is_contiguous())

    def test_abba_summary_uses_paired_rounds(self):
        summary = paired_summary([[100, 80, 80, 100], [200, 160, 160, 200]])
        self.assertEqual(summary["stock_us"], 150)
        self.assertEqual(summary["direct_us"], 120)
        self.assertEqual(summary["speedup_pct"], 25)

    def test_invalid_lengths(self):
        for lengths in ([], [0], [-1]):
            with self.assertRaises(ValueError):
                sequence_cases(lengths)


if __name__ == "__main__":
    unittest.main()
