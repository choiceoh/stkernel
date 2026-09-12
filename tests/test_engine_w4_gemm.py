"""kernels/w4_gemm: the packing's arithmetic and the reference the kernel is judged against (the kernel itself needs CUDA)."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None and importlib.util.find_spec("triton") is not None:
    import torch


@unittest.skipUnless(torch is not None, "requires PyTorch and Triton")
class PackTests(unittest.TestCase):
    def weight(self, n=48, k=256, seed=1):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16)

    def test_codes_are_symmetric_levels_and_the_error_is_bounded_by_half_a_step(self):
        from engine.kernels.w4_gemm import GROUP, KBLOCK, LEVELS, pack_w4, unpack_w4
        w = self.weight()
        packed, scales = pack_w4(w)
        self.assertEqual((packed.dtype, tuple(packed.shape)), (torch.uint8, (48, 128)))
        self.assertEqual((scales.dtype, tuple(scales.shape)), (torch.bfloat16, (48, 256 // GROUP)))
        codes = torch.cat([packed & 0xF, packed >> 4], -1)
        self.assertTrue(bool((codes >= 8 - LEVELS).all() and (codes <= 8 + LEVELS).all()), "levels -7..7 stored as q+8")
        packed, scales = pack_w4(w, clip_grid=(1.0,))                   # round-to-nearest: every weight inside the range
        deq = unpack_w4(packed, scales).float()
        step = scales.float().repeat_interleave(GROUP, dim=1)
        self.assertTrue(bool(((deq - w.float()).abs() <= step * 0.5 + 2 ** -7 * deq.abs()).all()),
                        "within half a level, plus the bf16 rounding of the product")
        with self.assertRaises(ValueError):
            pack_w4(torch.zeros(4, KBLOCK + 64, dtype=torch.bfloat16))

    def test_block_planar_nibbles_map_back_to_their_columns(self):
        from engine.kernels.w4_gemm import pack_w4, unpack_w4
        w = torch.zeros(2, 256, dtype=torch.bfloat16)
        w[0, 5] = 1.0                                    # first half of block 0 -> low nibble of byte 5
        w[1, 200] = -1.0                                 # block 1, column 72 -> high nibble of byte 64 + 8
        packed, scales = pack_w4(w)
        self.assertEqual(int(packed[0, 5] & 0xF), 8 + 7)
        self.assertEqual(int(packed[1, 64 + 8] >> 4), 8 - 7)
        deq = unpack_w4(packed, scales)
        self.assertEqual((deq != 0).nonzero().tolist(), [[0, 5], [1, 200]])

    def test_the_clip_search_never_loses_to_round_to_nearest(self):
        from engine.kernels.w4_gemm import pack_w4, unpack_w4
        w = self.weight(seed=7)
        searched = unpack_w4(*pack_w4(w)).float()
        plain = unpack_w4(*pack_w4(w, clip_grid=(1.0,))).float()
        self.assertLessEqual(float((searched - w.float()).pow(2).sum()), float((plain - w.float()).pow(2).sum()))

    def test_the_reference_linear_is_the_unpacked_weight(self):
        from engine.kernels.w4_gemm import pack_w4, unpack_w4, w4_linear
        w = self.weight()
        x = torch.randn(6, 256, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
        packed, scales = pack_w4(w)
        got = w4_linear(x, packed, scales)
        self.assertTrue(torch.equal(got, torch.nn.functional.linear(x, unpack_w4(packed, scales))))
        rel = (got.float() - torch.nn.functional.linear(x, w).float()).norm() / torch.nn.functional.linear(x, w).float().norm()
        self.assertLess(float(rel), 0.15, "int4 with a scale per 32 columns: a few percent of the bf16 product")
        with self.assertRaises(TypeError):
            w4_linear(x.float(), packed, scales)

    def test_the_plan_covers_k_with_whole_blocks_and_enough_programs(self):
        from engine.kernels.w4_gemm import KBLOCK, plan
        for M, N, K in ((6, 1024, 4096), (24, 4096, 4096), (12, 12288, 4096), (6, 4096, 20480), (24, 4096, 12288)):
            block_m, block_n, split, kb, warps = plan(M, N, K)
            self.assertEqual((K // KBLOCK) % (split * kb), 0)
            self.assertGreaterEqual(block_m, M)
            tiles = -(-N // block_n)
            self.assertTrue(tiles >= 128 or split > 1 or tiles * split >= 64, (M, N, K, tiles, split))


if __name__ == "__main__":
    unittest.main()
