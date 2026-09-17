"""Resident W4 layout is a lossless permutation, including the padded tail."""
import unittest

import torch

from engine.kernels.dense import DenseLinear, W4Pack, repack_w4


class CtaLayoutTests(unittest.TestCase):
    def test_logical_bytes_padding_and_metadata_survive_round_trip(self):
        for n, k in ((129, 256), (6416, 4096)):
            padded = (n + 127) // 128 * 128
            q = torch.randint(256, (padded, k // 2), dtype=torch.uint8)
            s = torch.randint(-32, 40, (padded, k // 16), dtype=torch.int8)
            r = torch.rand(padded)
            old = W4Pack(q.view(padded // 128, 128, k // 128, 64).permute(0, 2, 1, 3).contiguous(),
                         s.view(padded // 128, 128, k // 128, 8).permute(0, 2, 1, 3).contiguous(), r, n, k, True)
            new = repack_w4(old)
            self.assertEqual(new.data.shape, (padded // 16, k // 128, 16, 64))
            self.assertTrue(torch.equal(new.data.permute(0, 2, 1, 3).reshape_as(q), q))
            self.assertTrue(torch.equal(new.scale.permute(0, 2, 1, 3).reshape_as(s), s))
            self.assertIs(new.rowscale, r)
            self.assertTrue(new.calibrated)
            self.assertIs(repack_w4(new), new)
            back = repack_w4(new, 128)
            self.assertTrue(torch.equal(back.data, old.data))
            self.assertTrue(torch.equal(back.scale, old.scale))
            self.assertEqual(new.data.nbytes + new.scale.nbytes, old.data.nbytes + old.scale.nbytes)

    def test_dequantized_fp8_values_are_identical(self):
        from engine.kernels.dense.packing import mk_w4_dequant
        old = W4Pack(torch.randint(256, (2, 3, 128, 64), dtype=torch.uint8),
                     torch.randint(-24, 24, (2, 3, 128, 8), dtype=torch.int8),
                     torch.rand(256), 241, 384)
        new = repack_w4(old)
        a, b = [mk_w4_dequant(p.data, p.scale, p.rows, rgs=p.rowscale) for p in (old, new)]
        self.assertTrue(torch.equal(a, b))

    def test_mixed_or_invalid_layouts_are_refused(self):
        p = W4Pack(torch.zeros(1, 2, 128, 64, dtype=torch.uint8),
                   torch.zeros(8, 2, 16, 8, dtype=torch.int8), torch.ones(128), 128, 256)
        with self.assertRaises(ValueError):
            repack_w4(p)
        with self.assertRaises(ValueError):
            repack_w4(p, 32)

    def test_repacking_is_restricted_to_target_shape_before_execution(self):
        layer = DenseLinear.__new__(DenseLinear)
        layer.rows, layer.cols, layer.packs = 4096, 4096, [None]
        with self.assertRaises(ValueError):
            layer.prepare_cta_layout()
        layer.rows, layer.executed = 6416, 1
        with self.assertRaises(RuntimeError):
            layer.prepare_cta_layout()


if __name__ == '__main__':
    unittest.main()
