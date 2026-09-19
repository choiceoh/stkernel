"""Qwen3.8's decode projections take the W4 GEMM's input reuse (engine/QWEN38_CARRY.md S2): the input quantized once
by mk_input_pack_kernel and read by every tile of mk_gemm_input_kernel, instead of quantized again by every tile. The
tiles, k-slices, MMA and fixed-order split fold are the ordinary launch's, so the output is its bytes -- held here the
way tests/test_engine_decode_seven.py holds GLM-5.3's K=4096 cells: a captured launch with reuse, replayed over changed
and strided inputs, against the eager launch without it, rtol 0 and atol 0.

    the GDN in_proj   4120 x 2560 (33 tiles, the last one partial)
    the QSA in_proj   4224 x 2560
    both o_proj       2560 x 1536
    rows              2..8: a C x (K+1) verify step at K=1 up to four requests (the input pack holds eight rows)

GPU only, in the ST image (the native dense module); the single-GPU lane runs it through
probes/engine_qwen38_cells.GLUE_CASES.
"""
import unittest

import torch

from tests.image_kernels import PRESENT, REASON

SHAPES = ((4120, 2560), (4224, 2560), (2560, 1536))     # (n, k): facts' GDN / QSA in_proj, o_proj at TP=4
ROWS = tuple(range(2, 9))


@unittest.skipUnless(torch.cuda.is_available() and PRESENT, "requires CUDA; " + REASON)
class InputReuseTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.dense import extension
        self.ext = extension()
        self.saved = (self.ext.gemm_input_mode(), self.ext.gemm_input_cta_mode(), self.ext.probe_state())
        self.ext.set_input_cta(4)
        self.ext.set_gemm2(0)

    def tearDown(self):
        before, cta, state = self.saved
        self.ext.set_gemm_input(before)
        self.ext.set_input_cta(cta)
        self.ext.restore_probe_state(state)

    def test_the_plan_admits_the_qwen38_shapes_at_their_ordinary_split(self):
        ext = self.ext
        ext.set_gemm_input(2)
        for n, k in SHAPES:
            ordinary = ext.gemm2_plan(2, n, k)[0]
            self.assertIn(ordinary, (2, 3), (n, k))                      # what the input path requires
            for rows in ROWS:
                plan = ext.gemm_input_plan(rows, n, k, False, False)
                with self.subTest(n=n, k=k, rows=rows):
                    self.assertTrue(plan[0], plan)
                    self.assertEqual(plan[1], ext.gemm2_plan(rows, n, k)[0])      # the ordinary reduction order
                    self.assertFalse(ext.gemm_input_plan(rows, n, k, True, False)[0])   # no bias gate
                    self.assertFalse(ext.gemm_input_plan(rows, n, k, False, True)[0])   # no low-rank correction
                    self.assertEqual(ext.gemm_input_cta_plan(rows, n, k, False, False)[0], 0)   # not a CTA kernel
            for rows in (1, 9, 16, 32):                                   # outside the eight-row pack, or one row
                self.assertFalse(ext.gemm_input_plan(rows, n, k, False, False)[0], (n, k, rows))
        for n, k in ((4120, 4096), (4224, 1536), (2560, 2560), (6144, 2560)):        # only the served pairs
            self.assertFalse(ext.gemm_input_plan(4, n, k, False, False)[0], (n, k))
        ext.set_gemm_input(1)                                             # mode 1 is GLM-5.3's old gate alone
        for n, k in SHAPES:
            self.assertFalse(ext.gemm_input_plan(4, n, k, False, False)[0], (n, k))
        # GLM-5.3's cells are what they were: six to eight rows at K=4096 only
        ext.set_gemm_input(2)
        for rows in (1, 2, 5):
            self.assertFalse(ext.gemm_input_plan(rows, 6144, 4096, False, False)[0], rows)
        self.assertTrue(ext.gemm_input_plan(8, 6144, 4096, False, False)[0])

    def test_reuse_is_the_ordinary_launch_s_bytes_over_changed_and_strided_inputs(self):
        from engine.kernels.dense import DenseLinear
        ext = self.ext
        torch.manual_seed(20260919)
        for n, k in SHAPES:
            layer = DenseLinear((torch.randn(n, k, device="cuda") * .02).bfloat16(), prefill=False)
            for rows in ROWS:
                for stride, offset in ((k, 0), (k + 64, 8), (3 * k, k)):
                    parent = torch.full((rows, stride), float("nan"), device="cuda", dtype=torch.bfloat16)
                    x = parent[:, offset:offset + k]
                    x.normal_()
                    ext.set_gemm_input(2)
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
                            torch.cuda.synchronize()
                            with self.subTest(n=n, k=k, rows=rows, stride=stride, magnitude=magnitude):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                                self.assertEqual(bool(actual.abs().sum() > 0), magnitude > 0)
                    finally:
                        graph.reset()

    def test_eager_reuse_is_the_ordinary_launch_s_bytes(self):
        from engine.kernels.dense import DenseLinear
        ext = self.ext
        torch.manual_seed(20260920)
        for n, k in SHAPES:
            layer = DenseLinear((torch.randn(n, k, device="cuda") * .02).bfloat16(), prefill=False)
            for rows in ROWS:
                x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                ext.set_gemm_input(2)
                reused = layer(x)
                ext.set_gemm_input(0)
                ordinary = layer(x)
                with self.subTest(n=n, k=k, rows=rows):
                    torch.testing.assert_close(reused, ordinary, rtol=0, atol=0)
                    self.assertEqual(tuple(reused.shape), (rows, n))


if __name__ == "__main__":
    unittest.main()
