"""A failed hold says where its error sits and whose it is, and a NaN does not pass one.

    drift                 a NaN element is an infinite error (it used to pass: `nan > band` is false)
    skinny_gemv.qualify   the same hole in its own running max -- `max(0.0, nan)` is 0.0
    blame                 the elements past the band (count, the span of each index, the worst one's two values), whether
                          each side repeats itself, and which side leaves the CPU's reference
    qsa.qualify           its error carries `blame` for every (cell, rows) that failed -- the kernel stood in for by
                          torch here, so this runs where Triton does not

The GB10 lane's boot qualify died once with `norm_rope_4x128 (0.938, 0.0171)` and never again
(measurements/qwen38_lane_20260919): two numbers, and nothing to tell a kernel's fault from the reference's.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from engine.kernels import qsa
    RUNS, RUNS_REASON = True, ""
except Exception as exc:                                   # noqa: BLE001 -- torch or triton absent: nothing to hold
    RUNS, RUNS_REASON = False, f"engine.kernels.qsa does not import here: {exc!r}"

FACTS = dict(heads=((6, 256), (4, 128)), rotary_dim=64, theta=1e7, eps=1e-6)


def reference(x, w, eps, positions, theta, rotary_dim):
    """What `norm_rope_partial` is held to, under its signature."""
    from engine.modules.norm import rmsnorm_unit_offset
    from engine.modules.rotary import apply_rope, rope_tables
    cos, sin = rope_tables(positions, rotary_dim, theta, dtype=x.dtype)
    return apply_rope(rmsnorm_unit_offset(x, w, eps), cos, sin)


def one_head_lost(times: int):
    """A stand-in kernel that is the reference, except that its first `times` launches at the indexer's 300-row cell
    lose one head's rotated half: row 5, head 2, channels 32..63."""
    left = [times]

    def kernel(x, w, eps, positions, theta, rotary_dim):
        out = reference(x, w, eps, positions, theta, rotary_dim).clone()
        if tuple(x.shape) == (300, 4, 128) and left[0] > 0:
            left[0] -= 1
            out[5, 2, 32:64] += 100.0
        return out
    return kernel


@unittest.skipUnless(RUNS, RUNS_REASON)
class DriftTests(unittest.TestCase):
    def test_a_nan_is_an_infinite_error(self):
        from engine.kernels.gated_residual import drift
        ref = torch.randn(4, 8).to(torch.bfloat16)
        ours = ref.clone()
        self.assertEqual(drift(ours, ref), (0.0, 0.0))
        ours[1, 3] = float("nan")
        self.assertEqual(drift(ours, ref), (float("inf"), float("inf")))

    def test_a_kernel_that_gives_a_nan_does_not_qualify(self):
        def kernel(x, w, eps, positions, theta, rotary_dim):
            out = reference(x, w, eps, positions, theta, rotary_dim).clone()
            out[0, 0, 0] = float("nan")
            return out
        with mock.patch.object(qsa, "norm_rope_partial", kernel):
            with self.assertRaisesRegex(RuntimeError, "inf"):
                qsa.qualify(torch.device("cpu"), **FACTS)

    def test_a_skinny_gemv_that_gives_a_nan_does_not_qualify(self):
        from engine.kernels.common import skinny_gemv

        def gemv(x, w, cfg):                               # the product itself, but for one element
            out = (x.float() @ w.float().t()).to(torch.bfloat16)
            out[0, 0] = float("nan")
            return out
        with mock.patch.object(skinny_gemv, "gemv", gemv):
            with self.assertRaisesRegex(RuntimeError, "inf"):
                skinny_gemv.qualify(torch.device("cpu"))
        with mock.patch.object(skinny_gemv, "gemv", lambda x, w, cfg: (x.float() @ w.float().t()).to(torch.bfloat16)):
            skinny_gemv.qualify(torch.device("cpu"))       # and the product qualifies


@unittest.skipUnless(RUNS, RUNS_REASON)
class BlameTests(unittest.TestCase):
    def test_the_reference_itself_qualifies(self):
        with mock.patch.object(qsa, "norm_rope_partial", reference):
            worst = qsa.qualify(torch.device("cpu"), **FACTS)
        self.assertEqual(worst, {"norm_rope_6x256": (0.0, 0.0), "norm_rope_4x128": (0.0, 0.0)})

    def test_a_failure_that_does_not_come_back_is_named_as_one(self):
        with mock.patch.object(qsa, "norm_rope_partial", one_head_lost(times=1)):
            with self.assertRaises(RuntimeError) as raised:
                qsa.qualify(torch.device("cpu"), **FACTS)
        said = str(raised.exception)
        self.assertIn("norm_rope_4x128 at 300 rows: 32 of 153600 elements past the max band", said)
        self.assertIn("dim 0 5..5 (1 distinct), dim 1 2..2 (1 distinct), dim 2 32..63 (32 distinct)", said)
        self.assertIn("ours computed again differs in 32 elements -- it does not repeat itself", said)
        self.assertIn("the reference computed again gives the same bytes", said)
        self.assertNotIn("norm_rope_6x256 at", said)       # the cell that held is not blamed

    def test_a_failure_that_stays_is_laid_at_the_side_that_leaves_the_cpu(self):
        with mock.patch.object(qsa, "norm_rope_partial", one_head_lost(times=1 << 30)):
            with self.assertRaises(RuntimeError) as raised:
                qsa.qualify(torch.device("cpu"), **FACTS)
        said = str(raised.exception)
        self.assertIn("ours computed again gives the same bytes", said)
        self.assertRegex(said, r"against the CPU's reference ours drifts \([0-9.e+]+, [0-9.e+-]+\) and the device's "
                               r"reference \(0\.0, 0\.0\)")

    def test_an_rms_failure_with_no_element_past_the_band_says_so(self):
        from engine.kernels.gated_residual import blame
        ref = torch.ones(64, 64)
        ours = ref + 0.04                                  # every element 4% off: under max 0.05, over rms 0.02
        said = blame(ours, ref, lambda: ours, lambda: ref)
        self.assertIn("no element is past the max band (the rms band is what failed)", said)
        self.assertNotIn("CPU", said)                      # no host reference was offered

    def test_the_worst_element_is_the_one_quoted(self):
        from engine.kernels.gated_residual import blame
        ref = torch.ones(3, 4, 8)
        ours = ref.clone()
        ours[1, 2, 5], ours[2, 0, 1] = 3.0, float("nan")
        said = blame(ours, ref, lambda: ours, lambda: ref)
        self.assertIn("2 of 96 elements past the max band", said)
        self.assertIn("the worst at (2, 0, 1) is nan against the reference's 1", said)
        self.assertIn("ours computed again gives the same bytes", said)   # a NaN that stays a NaN has not moved


if __name__ == "__main__":
    unittest.main()
