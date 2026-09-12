"""The gated activation, fused against the two ops it replaces (45차 §86).

`silu(gate) * up` is not an approximation here: silu rounds to the input's dtype exactly where
`torch.nn.functional.silu` does, so the fused launch is bit identical to the pair.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from engine.kernels.swiglu import swiglu  # noqa: E402

CUDA = torch.cuda.is_available()


class ContractTests(unittest.TestCase):
    def test_it_takes_one_fused_projection(self):
        with self.assertRaises(ValueError):
            swiglu(torch.zeros(4, 6, 8))
        with self.assertRaises(ValueError):
            swiglu(torch.zeros(4, 7))


@unittest.skipUnless(CUDA, "the fused activation is the CUDA path")
class KernelTests(unittest.TestCase):
    def test_it_is_bit_identical_to_the_pair(self):
        for rows, inter in ((6, 3072), (1, 3072), (24, 3072), (6, 12288), (6, 128), (32, 3072)):
            with self.subTest(rows=rows, inter=inter):
                gen = torch.Generator(device="cuda").manual_seed(rows + inter)
                x = torch.randn(rows, 2 * inter, device="cuda", generator=gen).bfloat16()
                gate, up = x.chunk(2, -1)
                self.assertTrue(torch.equal(Fn.silu(gate) * up, swiglu(x)))

    def test_it_reads_a_projection_where_it_lies(self):
        """What the block hands over is a W4 pack's output; a view of one is not contiguous."""
        x = torch.randn(6, 2, 6144, device="cuda").bfloat16()[:, 1]
        gate, up = x.chunk(2, -1)
        self.assertFalse(x.is_contiguous())
        self.assertTrue(torch.equal(Fn.silu(gate) * up, swiglu(x)))


if __name__ == "__main__":
    unittest.main()
