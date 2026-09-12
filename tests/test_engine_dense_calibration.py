"""kernels/dense/calibration: the sums a boot keeps for the GPTQ packs the store lacks, and the store's part of it."""
import tempfile
import unittest
from pathlib import Path

import torch

from engine.kernels.dense.calibration import BUDGET_BYTES, ROWS_TARGET, Calibration
from engine.kernels.dense.store import PackStore


class FakeLayer:
    def __init__(self, cols, name="x"):
        self.cols, self.name, self.observer = cols, name, None

    def __call__(self, x, rows_ok=None):
        if self.observer is not None:
            self.observer(x.reshape(-1, self.cols), rows_ok)
        return x


class CalibrationTests(unittest.TestCase):
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
        self.assertEqual(PackStore.tiles(wide.name, wide_cols), [(wide.name, 0, wide_cols)])
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
            self.assertEqual(store.missing_calibration("A/model.x", 5 * PackStore.TILE), [("A/model.x", 0, 5 * PackStore.TILE)])
            self.assertFalse(store.calibrated("A/model.x"))
            path = store.calibration_path("A/model.y")
            path.parent.mkdir(parents=True)
            torch.save({"H": torch.eye(128), "amax": torch.ones(128), "ntok": 10, "name": "A/model.y"}, path)
            self.assertEqual(store.missing_calibration("A/model.y", 128), [])
            self.assertTrue(store.calibrated("A/model.y"))
            with self.assertRaisesRegex(ValueError, "do not fit"):
                store.missing_calibration("A/model.y", 256)                  # a TP-sharded width the old dump does not match


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
            self.assertEqual(dict(store.stats), {"fp8_built": 1, "fp8_gptq": 1})
            self.assertTrue(torch.equal(s, fp8_rtn(w)[1]))
            q2, _ = PackStore(tmp, 0).pack_fp8(w, "A/model.h")
            self.assertTrue(torch.equal(q.view(torch.uint8), q2.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
