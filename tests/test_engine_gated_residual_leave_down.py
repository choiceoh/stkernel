"""engine/kernels/gated_residual.leave_down_block -- a prefill site's leave and down fold in one launch -- held to the
two launches it replaces: the streams left into byte for byte, the scales within a few FP32 ulps, the mixer's outputs
(through leave_mix_block's up fold) within the oracle's band of the two-launch site's and of the torch form; the column
blocks unrolled (one, several, a narrow tail), the leave without an output, the host's refusals, the table, and where
`site` takes it. Small widths, so the interpreter runs it in seconds.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_gated_residual_leave_down
"""
import importlib.util
import os
import unittest
from unittest.mock import patch

torch = None
if importlib.util.find_spec("torch"):
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
READY = torch is not None and TRITON
RUNS = READY and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET or torch is None or not torch.cuda.is_available() else "cuda"
HC, HIDDEN, EPS = 4, 80, 1e-6                     # 80 = 5 x 16: whole K tiles of 16
BAND = 2.0 ** -6


def operands(rows, rank, device, seed=1, hidden=HIDDEN, hc=HC):
    from engine.kernels.gated_residual import pack_down_inject
    gen = torch.Generator().manual_seed(seed)
    width = hc * hidden
    down = (torch.randn(rank, width, generator=gen) * 0.02).bfloat16().to(device)
    inj_w = (torch.randn(hc, width, generator=gen) * 0.02).bfloat16().to(device)
    return dict(h=torch.randn(rows, width, generator=gen).bfloat16().to(device),
                out=torch.randn(rows, hidden, generator=gen).bfloat16().to(device),
                injection=(torch.rand(rows, hc, generator=gen) * 2).bfloat16().to(device),
                w=(torch.randn(width, generator=gen) * 0.1).bfloat16().to(device),
                di=pack_down_inject(down, inj_w), up=(torch.randn(width, rank, generator=gen) * 0.02).bfloat16().to(device),
                down=down, inj_w=inj_w)


def rel_err(got, want):
    return float((got.float() - want.float()).abs().max() / want.float().abs().max())


SMALL = {"down": (16, 16, 16, 4, 1, 1), "up": (16, 16, 16, 4, 1)}


@unittest.skipUnless(RUNS, "requires CUDA and Triton, or TRITON_INTERPRET=1 with Triton")
class LeaveDownTests(unittest.TestCase):
    def two_and_one(self, rows, rank, tile, *, leave=True, inject=True, seed=1):
        """(the two-launch site's (h, scale, mixed, injection), the fused site's, the torch form's) from one seed."""
        from engine.kernels import gated_residual as hcr
        from engine.modules.hyper_connection import gated_residual
        o = operands(rows, rank, DEVICE, seed)
        out, injection = (o["out"], o["injection"]) if leave else (None, None)
        di = o["di"] if inject else hcr.pack_down_inject(o["down"], None)
        tiles = {**SMALL, "leave_down": tile}
        h_two = o["h"].clone()
        scale_two = hcr.stream_scales(h_two, out, injection, EPS, HC)
        two = hcr.mix_block(h_two, di, o["up"], HC, inject=inject, tiles=tiles, norm=(scale_two, o["w"]))
        h_one = o["h"].clone()
        gates = torch.empty(rows, rank, dtype=torch.bfloat16, device=DEVICE)
        ij = torch.empty(rows, HC, dtype=torch.bfloat16, device=DEVICE) if inject else gates
        scale_one = hcr.leave_down_block(h_one, out, injection, o["w"], EPS, HC, di, gates, ij, inject=inject, tile=tile)
        one = hcr.leave_mix_block(o["h"].clone(), out, injection, o["w"], EPS, HC, di, o["up"], inject=inject, tiles=tiles)
        ref = gated_residual(h_two, o["w"], o["down"], o["up"], o["inj_w"] if inject else None, HC, EPS)
        return (h_two, scale_two, *two), (h_one, scale_one, *one), (ref if inject else (ref, None))

    def test_the_fused_launch_is_the_two_launches_within_the_band_and_the_leave_byte_for_byte(self):
        """One column block (rank 16 + 4 in 32s), two (in 16s), one and a 16-wide tail (rank 32 + 4 in 32s), and a
        block wider than the columns (in 64s): the same leave, the scale to FP32 ulps, the outputs in band."""
        for rank, tile in ((16, (16, 32, 16, 4, 1)), (16, (16, 16, 16, 4, 1)), (32, (16, 32, 16, 4, 1)),
                           (32, (16, 64, 16, 4, 1))):
            two, one, ref = self.two_and_one(37, rank, tile)
            with self.subTest(rank=rank, tile=tile):
                self.assertTrue(torch.equal(one[0], two[0]), "the leave")
                self.assertTrue(torch.allclose(one[1], two[1], rtol=1e-5, atol=0.0), "the scales")
                self.assertLess(rel_err(one[2], two[2]), BAND)
                self.assertLess(rel_err(one[3], two[3]), BAND)
                self.assertLess(rel_err(one[2], ref[0]), BAND)
                self.assertLess(rel_err(one[3], ref[1]), BAND)
                self.assertGreater(float(one[3].float().abs().max()), 0.0)

    def test_without_an_output_and_without_an_injection(self):
        """The first site (nothing to leave) and the closing mixer (no injection columns)."""
        for leave, inject in ((False, True), (True, False), (False, False)):
            two, one, ref = self.two_and_one(20, 16, (16, 16, 16, 4, 1), leave=leave, inject=inject, seed=3)
            with self.subTest(leave=leave, inject=inject):
                self.assertTrue(torch.equal(one[0], two[0]))
                self.assertTrue(torch.allclose(one[1], two[1], rtol=1e-5, atol=0.0))
                self.assertLess(rel_err(one[2], two[2]), BAND)
                self.assertLess(rel_err(one[2], ref[0]), BAND)
                if inject:
                    self.assertLess(rel_err(one[3], two[3]), BAND)
                else:
                    self.assertIsNone(one[3])
                    self.assertIsNone(two[3])

    def test_the_host_refuses_what_the_kernel_cannot_take(self):
        from engine.kernels import gated_residual as hcr
        o = operands(20, 16, DEVICE)
        gates = torch.empty(20, 16, dtype=torch.bfloat16, device=DEVICE)
        ij = torch.empty(20, HC, dtype=torch.bfloat16, device=DEVICE)
        call = lambda **kw: hcr.leave_down_block(o["h"].clone(), kw.pop("out", o["out"]), kw.pop("injection", o["injection"]),
                                                 kw.pop("w", o["w"]), EPS, HC, kw.pop("di", o["di"]), gates, ij,
                                                 inject=True, tile=kw.pop("tile", (16, 16, 16, 4, 1)), **kw)
        with self.assertRaisesRegex(ValueError, "inside one stream"):
            call(tile=(16, 16, 32, 4, 1))                                    # 32 does not divide 80
        with self.assertRaisesRegex(ValueError, "column blocks"):
            wide = torch.zeros(16 * 16 + 4, HC * HIDDEN, dtype=torch.bfloat16, device=DEVICE)
            hcr.leave_down_block(o["h"].clone(), o["out"], o["injection"], o["w"], EPS, HC, wide,
                                 torch.empty(20, 16 * 16, dtype=torch.bfloat16, device=DEVICE), ij, inject=True,
                                 tile=(16, 16, 16, 4, 1))                    # 260 columns in 16s: 17 blocks
        with self.assertRaisesRegex(ValueError, "the output"):
            call(out=o["out"][:, :HIDDEN // 2])
        with self.assertRaisesRegex(ValueError, "norm's weight"):
            call(w=o["w"][:HIDDEN])
        with self.assertRaisesRegex(ValueError, "pdl"):
            call(pdl=1)
        with self.assertRaisesRegex(ValueError, "fused leave tile"):
            hcr.leave_mix_block(o["h"].clone(), o["out"], o["injection"], o["w"], EPS, HC, o["di"], o["up"],
                                tiles={**SMALL, "leave_down": None})

    def test_no_rows_is_no_launch(self):
        from engine.kernels import gated_residual as hcr
        o = operands(0, 16, DEVICE)
        gates = torch.empty(0, 16, dtype=torch.bfloat16, device=DEVICE)
        scale = hcr.leave_down_block(o["h"], o["out"], o["injection"], o["w"], EPS, HC, o["di"], gates, gates[:, :HC],
                                     inject=True, tile=(16, 16, 16, 4, 1))
        self.assertEqual(tuple(scale.shape), (0, HC))


@unittest.skipUnless(READY, "torch and triton required")
class TableTests(unittest.TestCase):
    def test_the_table_serves_the_prefill_rows_and_the_mixer_s_columns(self):
        from engine.kernels import gated_residual as hcr
        least = min(r for r, _ in hcr.LEAVE_DOWN_TILES)
        self.assertEqual(least, hcr.PREFILL_ROWS)
        self.assertIsNone(hcr.block_tiles(least - 1)["leave_down"])
        for rows, tile in hcr.LEAVE_DOWN_TILES:
            bm, bn, bk, warps, stages = tile
            self.assertEqual(hcr.block_tiles(rows)["leave_down"], tile)
            self.assertEqual(2560 % bk, 0)                                   # a K tile inside one stream
            full = -(-324 // bn) - (1 if hcr.narrow_tail(324, bn) else 0)
            self.assertLessEqual(full, hcr.LEAVE_DOWN_BLOCKS)                # the mixer's 320 + 4 columns unroll
            self.assertLessEqual(-(-32768 // bm), 4096)                      # the longest prefill's arrival words

    def test_site_takes_the_fused_leave_at_prefill_rows(self):
        """On a device `site` hands a prefill step's rows to leave_mix_block (its leave-and-down launch and the up
        fold) and leaves other rows to the two calls."""
        from engine.kernels import gated_residual as hcr
        calls = []

        def fused(h, out, injection, w, eps, hc, di, up, *, inject, tiles, pdl):
            calls.append((h.shape[0], tiles["leave_down"], pdl))
            return torch.empty(h.shape[0], h.shape[1] // hc, dtype=h.dtype), torch.empty(h.shape[0], hc, dtype=h.dtype)

        o = operands(hcr.PREFILL_ROWS, 16, "cpu", hidden=2560)              # the model's width: the table's K tile divides it
        with patch.object(hcr, "leave_mix_block", fused), \
                patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)):
            hcr.site(o["h"].clone(), o["out"], o["injection"], o["w"], EPS, HC, o["di"], o["up"], pdl=True)
        self.assertEqual(calls, [(hcr.PREFILL_ROWS, hcr.block_tiles(hcr.PREFILL_ROWS)["leave_down"], True)])


if __name__ == "__main__":
    unittest.main()
