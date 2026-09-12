"""kernels/dense/calibration: the sums a boot keeps for the GPTQ packs the store lacks, and the store's part of it."""
import tempfile
import unittest
from pathlib import Path

import torch

from engine.kernels.dense.calibration import BUDGET_BYTES, ROWS_TARGET, Calibration
from engine.kernels.dense.store import Need, PackStore


class FakeLayer:
    def __init__(self, cols, name="x"):
        self.cols, self.name, self.observer = cols, name, None

    def __call__(self, x, rows_ok=None):
        if self.observer is not None:
            self.observer(x.reshape(-1, self.cols), rows_ok)
        return x


class CalibrationTests(unittest.TestCase):
    def test_wider_decode_does_not_calibrate_on_unmasked_ghost_rows(self):
        c = Calibration('cpu', budget_bytes=1 << 20, max_decode_rows=48)
        layer = FakeLayer(32)
        c.attach(layer.name, layer, PackStore.tiles(layer.name, 32), small_rows=False)
        c.arm()
        for rows in (6, 24, 36, 48):
            layer(torch.randn(rows, 32).bfloat16())
        self.assertEqual(c.progress(), 0)
        x = torch.randn(80, 32).bfloat16()
        layer(x)
        self.assertEqual(c.progress(), 80)
        torch.testing.assert_close(c.H[layer.name], x.float().T @ x.float(), rtol=0, atol=0)

    def test_sums_the_real_rows_only_once_armed_and_files_the_stores_blobs(self):
        c = Calibration("cpu", budget_bytes=1 << 20)
        layer = FakeLayer(64, "Some/model.layers.0.self_attn.o_proj")
        self.assertTrue(c.attach(layer.name, layer, PackStore.tiles(layer.name, 64), small_rows=True))
        x = torch.randn(6, 64).bfloat16()
        layer(x, torch.tensor([1, 1, 0, 1, 0, 0], dtype=torch.bool))
        self.assertEqual(c.progress(), 0, "nothing before arm(): warm-ups feed junk")
        self.assertEqual(float(c.H[layer.name].abs().sum()), 0.0)
        c.arm()
        layer(x, torch.tensor([1, 1, 0, 1, 0, 0], dtype=torch.bool))
        kept = x[[0, 1, 3]].float()
        torch.testing.assert_close(c.H[layer.name], kept.T @ kept, atol=1e-3, rtol=1e-3)
        self.assertEqual(c.progress(), 3)
        layer(x)                                                              # no mask: every row real
        self.assertEqual(c.progress(), 9)
        self.assertFalse(c.complete())
        c.rows[layer.name].fill_(ROWS_TARGET)
        self.assertTrue(c.complete())
        with tempfile.TemporaryDirectory() as tmp:
            written = c.save(tmp, rank=3)
            self.assertEqual(written, [Path(tmp) / "mkcalib" / "rank3" / (layer.name + ".pt")])
            blob = torch.load(written[0])
            self.assertEqual((blob["name"], blob["ntok"], tuple(blob["H"].shape), blob["H"].dtype, str(blob["H"].device)),
                             (layer.name, ROWS_TARGET, (64, 64), torch.float32, "cpu"))
            store = PackStore(tmp, 3)
            self.assertEqual(store.missing_calibration(layer.name, 64), [])   # the store now has it
            self.assertIn("filed", c.status())

    def test_a_layer_whose_small_calls_may_be_ghosts_sums_only_its_large_calls(self):
        c = Calibration("cpu", budget_bytes=1 << 20)
        layer = FakeLayer(32, "Target/model.layers.1.mlp.down_proj")
        c.attach(layer.name, layer, PackStore.tiles(layer.name, 32), small_rows=False)
        c.arm()
        layer(torch.randn(8, 32).bfloat16())                                  # a decode step: rows may be ghosts, no mask -> skipped
        self.assertEqual(c.progress(), 0)
        layer(torch.randn(40, 32).bfloat16())                                 # a prefill chunk: every row real
        self.assertEqual(c.progress(), 40)

    def test_a_wide_weight_is_summed_whole_and_the_budget_defers_whole_layers(self):
        """A weight wider than the decode tile gets ONE Hessian over its whole K (pack_wide's error feedback crosses the
        tiles), so its blob is the full width; what does not fit the budget waits for a later boot."""
        wide_cols = 2 * PackStore.TILE
        c = Calibration("cpu", budget_bytes=Calibration.nbytes(PackStore.tiles("Target/model.fc", wide_cols)))
        wide = FakeLayer(wide_cols, "Target/model.fc")
        self.assertEqual(PackStore.tiles(wide.name, wide_cols), [Need(wide.name, 0, wide_cols, hessian=True)])
        self.assertTrue(c.attach(wide.name, wide, PackStore.tiles(wide.name, wide.cols), small_rows=True))
        self.assertEqual(list(c.H), [wide.name])
        self.assertEqual(tuple(c.H[wide.name].shape), (wide_cols, wide_cols))
        c.arm()
        x = torch.randn(3, wide_cols).bfloat16()
        wide(x, torch.tensor([True, False, True]))
        kept = x[[0, 2]].float()
        torch.testing.assert_close(c.H[wide.name], kept.T @ kept, atol=1e-2, rtol=1e-3)
        self.assertEqual(c.progress(), 2)
        late = FakeLayer(64, "Target/model.late")
        self.assertFalse(c.attach(late.name, late, PackStore.tiles(late.name, 64), small_rows=True), "over budget: waits for a later boot")
        self.assertEqual(c.deferred, [(late.name, late.name)])
        self.assertIsNone(late.observer)
        self.assertIn("1 tiles deferred", c.status())
        self.assertLessEqual(Calibration.nbytes(PackStore.tiles("y", 20480)), BUDGET_BYTES)    # the drafter's fc fits a boot's budget

    def test_the_store_reports_missing_tiles_and_refuses_foreign_blobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PackStore(tmp, 0)
            self.assertEqual(store.missing_calibration("A/model.x", 5 * PackStore.TILE), [Need("A/model.x", 0, 5 * PackStore.TILE)])
            self.assertFalse(store.calibrated("A/model.x"))
            path = store.calibration_path("A/model.y")
            path.parent.mkdir(parents=True)
            torch.save({"H": torch.eye(128), "amax": torch.ones(128), "ntok": 10, "name": "A/model.y"}, path)
            self.assertEqual(store.missing_calibration("A/model.y", 128), [])
            self.assertTrue(store.calibrated("A/model.y"))
            with self.assertRaisesRegex(ValueError, "do not fit"):
                store.missing_calibration("A/model.y", 256)                  # a TP-sharded width the old dump does not match


class PeaksOnlyTests(unittest.TestCase):
    """A blob whose Hessian already fits the served weight but predates the channel peaks: only the peaks are summed."""

    def blob(self, store, key, width, rows=4096):
        g = torch.Generator().manual_seed(11)
        x = torch.randn(rows, width, generator=g)
        path = store.calibration_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"H": x.T @ x, "ntok": rows, "name": key}, path)              # an older stack's dump: no "amax"
        return x

    def test_a_blob_that_only_lacks_its_peaks_costs_its_columns_not_its_square(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PackStore(tmp, 0)
            self.blob(store, "A/model.w", 512)
            need = store.missing_calibration("A/model.w", 512)
            self.assertEqual(need, [Need("A/model.w", 0, 512, hessian=False)])
            self.assertLess(Calibration.nbytes(need), 512 * 512 * 4 // 100, "a [K, K] Hessian is not allocated for [K] peaks")
            self.assertEqual(Calibration.nbytes(need), 512 * 4 + 4096)
            whole = store.missing_calibration("A/model.absent", 512)
            self.assertEqual(Calibration.nbytes(whole), 512 * 512 * 4 + 512 * 4 + 4096)

    def test_the_peaks_are_summed_and_filed_into_the_blob_the_store_already_had(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PackStore(tmp, 2)
            x = self.blob(store, "A/model.w", 64)
            kept = torch.load(store.calibration_path("A/model.w"), weights_only=True)
            c = Calibration("cpu", budget_bytes=1 << 20)
            layer = FakeLayer(64, "A/model.w")
            self.assertTrue(c.attach(layer.name, layer, store.missing_calibration(layer.name, 64), small_rows=True))
            self.assertEqual(list(c.H), [], "no Gram buffer for a blob whose Hessian is on disk")
            self.assertEqual(list(c.rows), [layer.name])
            c.arm()
            seen = torch.randn(20, 64, generator=torch.Generator().manual_seed(3))
            layer(seen.bfloat16())
            self.assertEqual(c.progress(), 20)
            self.assertIn("1 for their channel peaks alone", c.status())
            written = c.save(tmp, rank=2)
            self.assertEqual(written, [store.calibration_path("A/model.w")])
            after = torch.load(written[0], weights_only=True)
            self.assertTrue(torch.equal(after["H"], kept["H"]), "the store's Hessian is kept, byte for byte")
            self.assertEqual((after["ntok"], after["name"]), (kept["ntok"], "A/model.w"), "and the rows that built it")
            torch.testing.assert_close(after["amax"], seen.bfloat16().float().abs().amax(0))
            self.assertEqual(PackStore(tmp, 2).missing_calibration("A/model.w", 64), [], "nothing left to sum")
            self.assertIsNotNone(PackStore(tmp, 2).amax("A/model.w"))
            del x


class FactorSharingTests(unittest.TestCase):
    def test_both_lanes_of_a_weight_share_one_factorisation_and_get_the_same_answer(self):
        from engine.kernels.dense.packing import fp8_gptq, gptq_factor
        g = torch.Generator().manual_seed(7)
        K = 128
        w = (torch.randn(64, K, generator=g) * 0.05).bfloat16()
        x = torch.randn(512, K, generator=g) @ (torch.randn(K, K, generator=g) * 0.3 + torch.eye(K))
        H = x.T @ x
        with tempfile.TemporaryDirectory() as tmp:
            store = PackStore(tmp, 0)
            first = store._factor("A/model.w", H, "none", "cpu")
            second = store._factor("A/model.w", H, "none", "cpu")
            self.assertIs(first, second)
            self.assertEqual((store.stats["factor_built"], store.stats["factor_reused"]), (1, 1))
            store._factor("A/model.other", H, "none", "cpu")
            self.assertEqual((store.stats["factor_built"], store.stats["factor_reused"]), (2, 1), "one entry: the next weight replaces it")
            store.FACTOR_BYTES = 0
            store._factor("A/model.big", H, "none", "cpu")
            store._factor("A/model.big", H, "none", "cpu")
            self.assertEqual(store.stats["factor_reused"], 1, "a factor too large to hold is used and dropped")
        perm, U, dead = gptq_factor(H, act_order=True, factor_device="cpu")
        torch.testing.assert_close(U, first[1])
        self.assertTrue(torch.equal(perm, first[0]))
        shared, alone = fp8_gptq(w, H, factor=first), fp8_gptq(w, H)
        self.assertTrue(torch.equal(shared[0].view(torch.uint8), alone[0].view(torch.uint8)), "the shared factor changes no code")
        self.assertTrue(torch.equal(shared[1], alone[1]))
        with self.assertRaisesRegex(ValueError, "column order"):
            fp8_gptq(w, H, act_order=False, factor=first)                 # a factor walked in a different order is not this pack's
        plain = gptq_factor(H, act_order=False, factor_device="cpu")
        self.assertIsNone(plain[0])
        self.assertTrue(torch.equal(fp8_gptq(w, H, act_order=False, factor=plain)[0].view(torch.uint8),
                                    fp8_gptq(w, H, act_order=False)[0].view(torch.uint8)))


class Fp8GptqTests(unittest.TestCase):
    def test_fp8_gptq_keeps_the_lanes_block_scales_and_lowers_the_output_error(self):
        from engine.kernels.dense.packing import FP8_BLOCK, fp8_block_scales, fp8_gptq, fp8_rtn
        g = torch.Generator().manual_seed(5)
        N, K, M = 200, 256, 2048
        w = (torch.randn(N, K, generator=g) * 0.05).bfloat16()
        mix = torch.randn(K, K, generator=g) * 0.4 + torch.eye(K)
        x = torch.randn(M, K, generator=g) @ mix
        H = x.T @ x
        q_rtn, s_rtn = fp8_rtn(w)
        q_gptq, s_gptq = fp8_gptq(w, H)
        self.assertEqual((q_rtn.dtype, tuple(q_rtn.shape), tuple(s_rtn.shape)), (torch.float8_e4m3fn, (256, K), (2, K // FP8_BLOCK)))
        self.assertTrue(torch.equal(s_rtn, s_gptq), "the served UE8M0 block scales are static")
        self.assertTrue(torch.equal(s_rtn, torch.exp2(torch.log2(s_rtn))) and bool((torch.log2(s_rtn) == torch.log2(s_rtn).round()).all()), "powers of two")
        per = s_rtn.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
        deq = lambda q: (q.float() * per)[:N]
        err = lambda d: float(((w.float() - d).double() @ H.double() * (w.float() - d).double()).sum())
        self.assertLess(err(deq(q_gptq)), 0.5 * err(deq(q_rtn)))
        self.assertTrue(bool((q_gptq[N:] == 0).all()), "padded rows stay zero")
        self.assertTrue(torch.isfinite(deq(q_gptq)).all())

    def test_the_store_packs_fp8_only_from_calibration_and_caches_it(self):
        from engine.kernels.dense.packing import fp8_rtn
        with tempfile.TemporaryDirectory() as tmp:
            store = PackStore(tmp, 0)
            w = (torch.randn(128, 256) * 0.05).bfloat16()
            self.assertIsNone(store.pack_fp8(w, "A/model.h"))
            x = torch.randn(1024, 256) @ (torch.randn(256, 256) * 0.3 + torch.eye(256))
            path = store.calibration_path("A/model.h"); path.parent.mkdir(parents=True)
            torch.save({"H": x.T @ x, "ntok": 1024, "name": "A/model.h"}, path)
            q, s = store.pack_fp8(w, "A/model.h")
            self.assertEqual(dict(store.stats), {"factor_built": 1, "fp8_built": 1, "fp8_gptq": 1})
            self.assertTrue(torch.equal(s, fp8_rtn(w)[1]))
            q2, _ = PackStore(tmp, 0).pack_fp8(w, "A/model.h")
            self.assertTrue(torch.equal(q.view(torch.uint8), q2.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
