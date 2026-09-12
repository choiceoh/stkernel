"""The fused tap mix against the torch form it replaces (45차 §83).

`kernels/draft_conv._by_torch` is the definition the drafter used to run inline; `tap_mix` is the same mix in
one launch. These pin the two things that can go wrong in a causal convolution written as a kernel -- the
boundary the taps must not cross, and the rounding of a sum that used to be rounded at every step.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from engine.kernels.draft_conv import _by_torch, tap_mix  # noqa: E402

CUDA = torch.cuda.is_available()


def rows(rows, width, taps, group, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    kind = dict(device=device, generator=gen, dtype=torch.float32)
    return (torch.randn(rows, width, **kind).bfloat16(),
            torch.randn(rows, taps, width // group, **kind).bfloat16(),
            torch.randn(taps, width, **kind).bfloat16())


def term_scale(x, delta, base, group, block):
    """The summed magnitude of the terms: what a rounding of the sum is measured against."""
    n, wide = x.shape[0] // block, x.shape[1]
    taps, groups = delta.shape[1], wide // group
    coeff = (base.reshape(1, 1, taps, groups, group) + delta.view(n, block, taps, groups, 1)).flatten(-2).float()
    blocks = x.view(n, block, wide).float()
    out = torch.zeros_like(blocks)
    for tap in range(taps):
        shifted = torch.zeros_like(blocks)
        shifted[:, tap:] = blocks[:, :block - tap] if tap else blocks
        out += (coeff[:, :, tap] * shifted).abs()
    return out.reshape_as(x)


class ContractTests(unittest.TestCase):
    def test_what_the_mix_refuses(self):
        x = torch.zeros(6, 64)
        delta = torch.zeros(6, 2, 4)
        base = torch.zeros(2, 64)
        with self.assertRaises(ValueError):
            tap_mix(x.reshape(2, 3, 64), delta, base, 16)             # x is [rows, width]
        with self.assertRaises(ValueError):
            tap_mix(x, torch.zeros(6, 2, 3), base, 16)                # a delta per group of every tap
        with self.assertRaises(ValueError):
            tap_mix(x, delta, torch.zeros(2, 32), 16)                 # the base covers every tap's width
        with self.assertRaises(ValueError):
            tap_mix(x, delta.bfloat16(), base, 16)                    # one dtype: the coefficient is rounded once
        with self.assertRaises(ValueError):
            tap_mix(x, delta, base, 16, block=4)                      # 6 rows are not whole blocks of 4

    def test_the_reference_keeps_the_taps_inside_a_block(self):
        """Two blocks of three, the second all zeros: nothing from the first may leak across the boundary."""
        x = torch.zeros(6, 32)
        x[2] = 1.0
        delta = torch.zeros(6, 2, 2)
        base = torch.ones(2, 32)
        out = _by_torch(x, delta, base, 16, block=3)
        self.assertTrue(torch.equal(out[3:], torch.zeros(3, 32)))
        self.assertTrue(torch.equal(out[2], torch.ones(32)))          # its own tap 0
        self.assertTrue(torch.equal(out[0], torch.zeros(32)))


@unittest.skipUnless(CUDA, "the fused mix is the CUDA path")
class KernelTests(unittest.TestCase):
    cases = [(6, 4096, 2, 16, 6), (24, 4096, 2, 16, 6), (4, 4096, 2, 16, 4),
             (12, 4096, 2, 16, 6), (6, 4096, 3, 16, 3), (1, 4096, 2, 16, 1), (6, 256, 2, 16, 6)]

    def test_a_row_whose_only_live_tap_is_the_first_is_the_old_answer_exactly(self):
        for count, width, taps, group, block in self.cases:
            with self.subTest(rows=count, block=block, taps=taps):
                x, delta, base = rows(count, width, taps, group, count + taps + block, "cuda")
                ref = _by_torch(x, delta, base, group, block).view(count // block, block, width)
                got = tap_mix(x, delta, base, group, block).view(count // block, block, width)
                self.assertTrue(torch.equal(ref[:, 0], got[:, 0]))

    def test_the_sum_is_within_two_half_steps_of_the_form_that_rounded_every_tap(self):
        for count, width, taps, group, block in self.cases:
            with self.subTest(rows=count, block=block, taps=taps):
                x, delta, base = rows(count, width, taps, group, count + taps + block, "cuda")
                gap = (_by_torch(x, delta, base, group, block).float()
                       - tap_mix(x, delta, base, group, block).float()).abs()
                scale = term_scale(x, delta, base, group, block).clamp_min(1e-30)
                self.assertLessEqual((gap / scale / 2 ** -8).max().item(), 2.0 + 1e-3)

    def test_a_delta_that_is_a_slice_of_a_projection_is_read_where_it_lies(self):
        """What the block hands over is `coeff[:, 0]` of a [rows, 2, taps, groups] projection -- a view whose
        row stride is twice the packed one. Binding the caller's stride to a contiguous copy reads the wrong
        coefficients, and only in production, where the two are never the same tensor."""
        x, _, base = rows(6, 4096, 2, 16, 11, "cuda")
        gen = torch.Generator(device="cuda").manual_seed(12)
        projection = torch.randn(6, 2, 2, 256, device="cuda", generator=gen).bfloat16()
        for half in (0, 1):
            delta = projection[:, half]
            self.assertFalse(delta.is_contiguous())
            self.assertTrue(torch.equal(tap_mix(x, delta, base, 16, 6),
                                        tap_mix(x, delta.contiguous(), base, 16, 6)))

    def test_the_taps_do_not_cross_a_block_boundary(self):
        x = torch.zeros(6, 4096, device="cuda", dtype=torch.bfloat16)
        x[2] = 1.0
        delta = torch.zeros(6, 2, 256, device="cuda", dtype=torch.bfloat16)
        base = torch.ones(2, 4096, device="cuda", dtype=torch.bfloat16)
        out = tap_mix(x, delta, base, 16, block=3)
        self.assertEqual(out[3:].abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(out[2], torch.ones(4096, device="cuda", dtype=torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
