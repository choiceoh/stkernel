"""engine/kernels/common/skinny_gemv: the product, the split's last-arrival sum and its reset, and where torch.mm serves.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_skinny_gemv

The interpreter runs the programs one after another, so the split's last program is the grid's last; on a GPU the
order is the scheduler's, and the sum in split order is what makes the result the same either way (qualify, at boot).
"""
import importlib.util
import os
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
READY = torch is not None and importlib.util.find_spec("triton") is not None
DEVICE = "cpu" if INTERPRET else "cuda"


@unittest.skipUnless(READY, "torch and triton required")
class TableTests(unittest.TestCase):
    def test_every_config_is_a_tile_the_kernel_takes(self):
        from engine.kernels.common import skinny_gemv as sg
        for (n, k), (block_n, block_k, split, warps, stages) in sg.CONFIGS.items():
            for block in (block_n, block_k):
                self.assertEqual(block & (block - 1), 0, (n, k))
                self.assertGreaterEqual(block, 16, (n, k))
            self.assertLessEqual(-(-n // block_n), sg.MAX_BLOCKS)
            if split > 1:                               # one program's K tile may pass K: masked (the MTP shared down's 160)
                self.assertLessEqual(split * block_k, k, "a split past K launches programs with nothing to read")
            self.assertIn(warps, (1, 2, 4, 8))
            self.assertGreaterEqual(stages, 1)

    def test_torch_mm_serves_what_the_kernel_does_not_take(self):
        from unittest.mock import patch
        from engine.kernels.common import skinny_gemv as sg
        n, k = next(iter(sg.CONFIGS))
        w = torch.randn(n, k).bfloat16()
        cases = {"cpu": torch.randn(4, k).bfloat16(), "rows": torch.randn(sg.MAX_ROWS + 1, k).bfloat16(),
                 "fp32": torch.randn(4, k)}
        if torch.cuda.is_available():
            cases["one row"] = torch.randn(1, k).bfloat16().cuda()
            w_cuda = w.cuda()
        with patch.object(sg, "gemv", side_effect=AssertionError("the kernel was called")):
            for label, x in cases.items():
                ww = w.float() if x.dtype == torch.float32 else (w_cuda if x.is_cuda else w)
                self.assertTrue(torch.equal(sg.linear_rows(x, ww), torch.mm(x, ww.t())), label)
            other = torch.randn(n + 1, k).bfloat16()
            x = cases["cpu"]
            self.assertTrue(torch.equal(sg.linear_rows(x, other), torch.mm(x, other.t())))

    def test_one_row_goes_to_the_kernel_only_for_a_one_row_shape(self):
        from unittest.mock import Mock, patch
        from engine.kernels.common import skinny_gemv as sg
        self.assertTrue(sg.ONE_ROW <= set(sg.CONFIGS), "a one-row shape with no tile")
        self.assertNotIn((513, 2560), sg.ONE_ROW, "the router's one row ties cuBLAS's gemv (q38gemv-0919c)")
        self.assertNotIn((320, 2560), sg.ONE_ROW, "the shared gate_up's one row is cuBLAS's (q38gemv-0919e)")

        def operand(shape):                             # CUDA-shaped operands, so the dispatch runs on the CPU
            return Mock(is_cuda=True, shape=shape, dtype=torch.bfloat16, stride=lambda d: 1, t=lambda: "w.T")

        with patch.object(sg, "gemv", return_value="kernel") as kernel, patch.object(torch, "mm", return_value="mm"):
            for n, k in sorted(sg.CONFIGS):
                for rows in (1, 2, sg.MAX_ROWS, sg.MAX_ROWS + 1):
                    want = "kernel" if (rows >= 2 or (n, k) in sg.ONE_ROW) and rows <= sg.MAX_ROWS else "mm"
                    self.assertEqual(sg.linear_rows(operand((rows, k)), operand((n, k))), want, (n, k, rows))
            self.assertTrue(all(call.args[2] == sg.CONFIGS[tuple(call.args[1].shape)] for call in kernel.call_args_list))


@unittest.skipUnless(READY and (INTERPRET or (torch is not None and torch.cuda.is_available())),
                     "CUDA or Triton interpreter required")
class ProductTests(unittest.TestCase):
    def check(self, n, k, cfg, rows=(1, 3, 16)):
        from engine.kernels.common import skinny_gemv as sg
        gen = torch.Generator().manual_seed(n * 131 + k)
        w = (torch.randn(n, k, generator=gen) * 0.02).bfloat16().to(DEVICE)
        for m in rows:
            x = torch.randn(m, k, generator=gen).bfloat16().to(DEVICE)
            ref = x.float() @ w.float().t()
            first = sg.gemv(x, w, cfg)
            again = sg.gemv(x, w, cfg)
            self.assertEqual(tuple(first.shape), (m, n))
            self.assertEqual(first.dtype, torch.bfloat16)
            err = float(((first.float() - ref).abs().max() / ref.abs().max()).item())
            self.assertLess(err, 2.0 ** -7, (n, k, cfg, m))        # one output rounding, not a wrong tile or split
            self.assertTrue(torch.equal(first, again), (n, k, cfg, m))
            if cfg[2] > 1:
                self.assertEqual(int(sg.prepare(DEVICE).abs().sum()), 0, "an arrival word left set")

    def test_one_program_a_column_block(self):
        self.check(40, 320, (16, 64, 1, 4, 2))          # a ragged last column block, K in five tiles

    def test_one_k_tile_past_k(self):
        self.check(40, 160, (64, 256, 1, 4, 1))         # the MTP shared down's K: one tile, masked past 160

    def test_a_split_sums_its_partials_in_split_order(self):
        self.check(40, 1280, (16, 128, 4, 4, 2))        # 40 columns: three blocks, the last ragged

    def test_a_split_with_programs_past_k(self):
        self.check(24, 512, (16, 256, 4, 4, 1))         # two tiles of K over four splits: two programs read nothing

    def test_a_split_that_does_not_divide_k(self):
        self.check(24, 640, (16, 128, 3, 4, 1))         # five tiles over three splits: 2, 2, 1

    def test_linear_rows_is_the_product_where_it_serves(self):
        from engine.kernels.common import skinny_gemv as sg
        if DEVICE != "cuda":
            self.skipTest("linear_rows serves CUDA tensors only")
        (n, k), cfg = next(iter(sg.CONFIGS.items()))
        w = (torch.randn(n, k) * 0.02).bfloat16().cuda()
        x = torch.randn(4, k).bfloat16().cuda()
        self.assertTrue(torch.equal(sg.linear_rows(x, w), sg.gemv(x, w, cfg)))


if __name__ == "__main__":
    unittest.main()
