"""KDA storage precision: capacity, typed copies and durable format identity."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.base.kv_tier import NvmeTier, SECTOR
from engine.profiles.glm53.caches import cache_capacity, layout, snapshot_layout, stage_bytes, state_dtype
from tests.test_engine_glm53 import tiny_facts

torch = None
if importlib.util.find_spec("torch"):
    import torch


class StateCapacityTests(unittest.TestCase):
    def test_full_geometry_returns_memory_without_increasing_capacity(self):
        base = replace(tiny_facts(), kda_state_dtype="fp32", layers=45, kinds=("kda",)*34 + ("dsa",)*11,
                       kda_heads=64, kda_dim=128, block=768, kv_lora=512, spec_k=6)
        half = replace(base, kda_state_dtype="fp16")
        draft = (5, 2048, 2, 128)  # 10 MiB of sharded BF16 drafter KV
        for concurrency in (1, 4):
            for snapshot_gib in (2.125, 4.25):
                a = cache_capacity(base, range(45), draft, 7., concurrency, snapshot_gib)
                b = cache_capacity(half, range(45), draft, 7., concurrency, snapshot_gib)
                self.assertEqual(a, b)
                self.assertEqual(a[1], 48 if snapshot_gib == 2.125 else 96)
                self.assertEqual(layout(base, range(45), draft).block_bytes,
                                 layout(half, range(45), draft).block_bytes)
                self.assertEqual(layout(base, range(45), draft).slot_bytes
                                 - layout(half, range(45), draft).slot_bytes, 119 << 20)
                self.assertEqual(snapshot_layout(base, range(45), draft)[0]
                                 - snapshot_layout(half, range(45), draft)[0], 17 << 20)
                saved = (concurrency + 1) * (119 << 20) + a[1] * (17 << 20)
                saved += stage_bytes(base, range(45), concurrency) - stage_bytes(half, range(45), concurrency)
                self.assertGreater(saved, 1 << 30)
                if concurrency == 4 and snapshot_gib == 2.125:
                    self.assertEqual(saved, 1496 << 20)

    def test_invalid_precision_is_rejected_before_allocation(self):
        for value in ("bf16", "int8", "FP16", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                state_dtype(value)
        with self.assertRaises(ValueError):
            layout(replace(tiny_facts(), kda_state_dtype="bf16"), [0])

    def test_durable_formats_are_not_interchangeable_even_with_equal_byte_counts(self):
        tier = NvmeTier.__new__(NvmeTier)
        tier.block_bytes = SECTOR
        tier.index = {
            "1": dict(block_bytes=SECTOR, extra=SECTOR, bytes=2*SECTOR, at=1),
            "2": dict(block_bytes=SECTOR, extra=SECTOR, bytes=2*SECTOR, at=2,
                      state_format="glm53-kda-fp16-v1"),
        }
        self.assertEqual(tier.keys(), [1])
        self.assertEqual(tier.stale(), ["2"])
        tier.state_format = "glm53-kda-fp16-v1"
        self.assertEqual(tier.keys(), [2])
        self.assertEqual(tier.stale(), ["1"])
        self.assertEqual(tier.oldest(), 1)
        self.assertEqual(tier.stale_bytes(), 2*SECTOR)

    def test_format_is_committed_in_manifest_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            tier = NvmeTier.__new__(NvmeTier)
            tier.dir = Path(directory)
            tier.manifest = tier.dir / "manifest.json"
            tier.index = {}
            tier.block_bytes = SECTOR
            tier.state_format = "glm53-kda-fp16-v1"
            with patch.object(tier, "_sync_directory"):
                tier._publish(2, tier.dir / "seq-2.kv", 1, 16, 2*SECTOR, extra=SECTOR)
            tier.index = json.loads(tier.manifest.read_text())
            self.assertTrue(tier.has(2))
            tier.state_format = ""
            self.assertFalse(tier.has(2))


@unittest.skipUnless(torch is not None, "requires torch CPU")
class StateCopyTests(unittest.TestCase):
    def test_snapshot_boundary_and_restore_keep_fp16_bits(self):
        from engine.base.arena import Arena
        from engine.profiles.glm53.caches import Glm53Caches
        F = replace(tiny_facts(), kda_state_dtype="fp16")
        layers = range(F.layers)
        size = layout(F, layers).nbytes(2, 2) + 2*snapshot_layout(F, layers)[0] + stage_bytes(F, layers, 2)
        arena = Arena(size + 4096, device="cpu")  # alignment after the small block table
        caches = Glm53Caches(arena, F, layers, 2, 2, snapshots=2, stage=True)
        conv, rec = caches.kda(0, 1)
        self.assertEqual(rec.dtype, torch.float16)
        self.assertEqual(conv.dtype, torch.bfloat16)
        self.assertEqual(caches._stage["rec", 0].dtype, torch.float16)
        rec.copy_(torch.linspace(-2., 2., rec.numel()).reshape(rec.shape))
        conv.fill_(.25)
        caches.checkpoint(1, 16, 0)
        caches.stage_boundaries(torch.tensor([1]), torch.tensor([14]), torch.tensor([3]))
        caches.checkpoint_from_stage(1, 1)
        self.assertTrue(torch.equal(caches._snap["rec", 0][0].view(torch.uint8),
                                    caches._snap["rec", 0][1].view(torch.uint8)))
        caches.restore(2, 16, 0)
        self.assertTrue(torch.equal(caches.kda(0, 2)[1][15 % (F.spec_k + 1)].view(torch.uint8),
                                    rec[15 % (F.spec_k + 1)].view(torch.uint8)))
        # A prefill marker arrives in FP32, rounds once, then copies exactly.
        source = torch.linspace(-1., 1., rec[0].numel()).reshape(rec[0].shape)
        taps = torch.full((F.conv - 1, conv.shape[0]), .125, dtype=torch.bfloat16)
        caches.mark_kda(0, 0, source, taps)
        caches.restore(2, 16, 0)
        self.assertTrue(torch.equal(caches.kda(0, 2)[1][15 % (F.spec_k + 1)], source.half()))
        self.assertLessEqual(arena.used, size + 256)

    def test_experiment_selection_and_production_fact_are_explicit(self):
        import os
        from types import SimpleNamespace
        from engine.base.config import ConfigError
        from engine.profiles.glm53 import boot, facts
        args = SimpleNamespace(ckpt_meta="/metadata", ranks="/ranks", kv_gib=7., port=8000, production=False)
        with patch.dict(os.environ, {"STK_kda_state_dtype": "fp16"}, clear=True):
            self.assertEqual(boot.declared(args, 4)["kda_state_dtype"], "fp16")
            args.production = True
            with self.assertRaises(ConfigError):
                boot.declared(args, 4)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(boot.declared(args, 4)["kda_state_dtype"], facts.KDA_STATE_DTYPE)

    def test_declared_budget_keeps_paged_capacity_and_returns_state_savings(self):
        from engine.profiles.glm53 import budget
        F = replace(tiny_facts(), kda_state_dtype="fp32", kda_heads=64, kda_dim=128, spec_k=6)
        with tempfile.TemporaryDirectory() as directory, patch.object(budget.facts, "load", return_value=F):
            args = dict(kv_gib=1., max_seqs=4, ckpt=directory, box_gib=128., drafter_dir=None, snapshots=48)
            a = budget.budget(**args, kda_state_dtype="fp32")
            b = budget.budget(**args, kda_state_dtype="fp16")
        self.assertEqual(a.paged_gib, b.paged_gib)
        self.assertEqual(a.kv_declared_gib, b.kv_declared_gib)
        # One KDA layer: 5*7 active states + 48 snapshots + 5 staged states.
        self.assertAlmostEqual(b.kv_gib - a.kv_gib, 88 * .5 / 1024)


if __name__ == "__main__":
    unittest.main()
