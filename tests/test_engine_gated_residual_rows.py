"""engine/kernels/gated_residual.mix_rows -- a decode step's mixer in two launches (carry H2) -- held byte for byte to
the unfolded mixer's four launches on the same products (skinny_gemv's, then `_gates` and `_mix_mean`), and
where `mix` takes it. The oracle band is test_engine_qwen38_kernels' (GatedResidualTests: on a GPU its 1- and 7-row
sites fold) and the boot's qualify.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_gated_residual_rows
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
HC, HIDDEN, RANK = 4, 2560, 320                     # Qwen3.8's site: the shapes skinny_gemv tiles


def site(inject: bool, rows: int, device="cpu", seed=0):
    gen = torch.Generator().manual_seed(seed)
    width = HC * HIDDEN
    down = (torch.randn(RANK + (HC if inject else 0), width, generator=gen) * 0.02).bfloat16().to(device)
    up = (torch.randn(width, RANK, generator=gen) * 0.02).bfloat16().to(device)
    normed = torch.randn(rows, width, generator=gen).bfloat16().to(device)
    return normed, down, up


def unfolded(normed, down, up, inject):
    """The unfolded mixer's four launches on skinny_gemv's products at mix_rows' tiles (the site probe reads it too)."""
    import triton
    from engine.kernels import gated_residual as hcr
    from engine.kernels.common import skinny_gemv as sg
    rows = normed.shape[0]
    di = sg.gemv(normed, down, sg.CONFIGS[tuple(down.shape)])
    gates = torch.empty(rows, RANK, dtype=normed.dtype, device=normed.device)
    inj = torch.empty(rows, HC, dtype=normed.dtype, device=normed.device) if inject else gates
    hcr._gates[(rows,)](di, gates, inj, di.stride(0), gates.stride(0), inj.stride(0), float(HC), R=RANK,
                        BR=triton.next_power_of_2(RANK), HC=HC, BH=triton.next_power_of_2(HC), WITH_INJECT=inject,
                        num_warps=4)
    block_d, block_k, warps, stages = hcr.UP_TILE
    weights = sg.gemv(gates, up, (block_d, block_k, 1, warps, stages))
    mixed = torch.empty(rows, HIDDEN, dtype=normed.dtype, device=normed.device)
    hcr._mix_mean[(rows,)](weights, normed, mixed, weights.stride(0), normed.stride(0), mixed.stride(0), float(HC),
                           HID=HIDDEN, BD=triton.next_power_of_2(HIDDEN), HC=HC, num_warps=8)
    return mixed, (inj if inject else None)


@unittest.skipUnless(READY, "torch and triton required")
class FoldsTests(unittest.TestCase):
    def test_only_decode_rows_of_a_tiled_projection_fold(self):
        from engine.kernels import gated_residual as hcr
        normed, down, up = site(True, 4)
        self.assertTrue(hcr.folds(normed, down, up))
        self.assertTrue(hcr.folds(*site(False, hcr.DECODE_ROWS)))
        self.assertFalse(hcr.folds(*site(True, hcr.DECODE_ROWS + 1)))
        self.assertFalse(hcr.folds(normed.float(), down.float(), up.float()))
        self.assertFalse(hcr.folds(normed, down[:300], up), "a down projection skinny_gemv has no tile for")
        self.assertFalse(hcr.folds(normed, down, up.t().contiguous().t()), "an up weight not packed along its rank")

    def test_a_projection_lane_keeps_the_unfolded_mixer(self):
        """hc_fp8's projections replace the two BF16 products; the fold is of those products, so it steps aside."""
        import inspect
        from engine.kernels import gated_residual as hcr
        body = inspect.getsource(hcr.mix)
        taken = body.index("return mix_rows(")
        self.assertIn("if project_down is None and project_up is None and folds(normed, down_inject, up):",
                      body[:taken])
        self.assertGreater(taken, body.index("if not normed.is_cuda:"), "the CPU form stays torch's")

    def test_it_refuses_what_it_does_not_fold(self):
        from engine.kernels import gated_residual as hcr
        with self.assertRaises(ValueError):
            hcr.mix_rows(*site(True, hcr.DECODE_ROWS + 1), HC)


@unittest.skipUnless(READY and (INTERPRET or (torch is not None and torch.cuda.is_available())),
                     "CUDA or Triton interpreter required")
class MixRowsTests(unittest.TestCase):
    def test_the_fold_is_the_unfolded_mixer_byte_for_byte(self):
        from engine.kernels import gated_residual as hcr
        from engine.kernels.common import skinny_gemv as sg
        for inject in (True, False):
            for rows in (1, 3, 16):
                normed, down, up = site(inject, rows, DEVICE, seed=rows + 7 * inject)
                mixed, injection = hcr.mix_rows(normed, down, up, HC, inject=inject)
                again = hcr.mix_rows(normed, down, up, HC, inject=inject)
                want_mixed, want_injection = unfolded(normed, down, up, inject)
                with self.subTest(inject=inject, rows=rows):
                    self.assertEqual((tuple(mixed.shape), mixed.dtype), ((rows, HIDDEN), torch.bfloat16))
                    self.assertTrue(torch.equal(mixed, want_mixed))
                    self.assertTrue(torch.equal(mixed, again[0]), "the split's sum is in split order")
                    if inject:
                        self.assertEqual(tuple(injection.shape), (rows, HC))
                        self.assertTrue(torch.equal(injection, want_injection))
                    else:
                        self.assertIsNone(injection)
                    self.assertEqual(int(sg.prepare(DEVICE).abs().sum()), 0, "an arrival word left set")

    def test_mix_takes_it_on_cuda(self):
        from engine.kernels import gated_residual as hcr
        if DEVICE != "cuda":
            self.skipTest("mix folds CUDA rows only")
        normed, down, up = site(True, 4, "cuda")
        got = hcr.mix(normed, down, up, HC)
        want = hcr.mix_rows(normed, down, up, HC)
        self.assertTrue(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))


if __name__ == "__main__":
    unittest.main()
