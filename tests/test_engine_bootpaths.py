"""Alternate model/drafter paths must reach every boot mode before allocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class StopAtBuild(Exception):
    pass


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class BootPathTests(unittest.TestCase):
    def test_local_http_and_fleet_forward_both_model_directories(self):
        from engine.profiles.glm53 import boot
        args = SimpleNamespace(ckpt_meta="/alternate/config", drafter_dir="/alternate/draft",
                               ranks="/alternate/ranks", layers="0-0", seed=0, prompt=1,
                               seqs=1, kv_gib=.25, park=False, drafter=True, max_new=1,
                               temperature=0., tier_dir="/unused", port=8000)
        comm = SimpleNamespace(rank=0, world_size=4, close=Mock())
        tp = SimpleNamespace(run=lambda fn: fn(comm))
        for mode in ("local", "http", "fleet"):
            args.serve = mode == "http"
            with self.subTest(mode=mode), \
                 patch.object(boot.facts, "check_box", return_value="test"), \
                 patch.object(boot, "declared"), \
                 patch.object(boot, "LocalTP", return_value=tp), \
                 patch.object(boot.Comm, "init", return_value=comm), \
                 patch.object(boot.lane_tables, "reference"), \
                 patch.object(boot.lane_tables, "served"), \
                 patch.object(boot, "build", side_effect=StopAtBuild) as build:
                with self.assertRaises(StopAtBuild):
                    (boot.fleet if mode == "fleet" else boot.local)(args)
                self.assertEqual(build.call_args.kwargs["ckpt_meta"], args.ckpt_meta)
                self.assertEqual(build.call_args.kwargs["drafter_dir"], args.drafter_dir)
        comm.close.assert_called_once_with()

    def test_build_reads_drafter_facts_from_the_selected_directory(self):
        from engine.profiles.glm53 import boot
        draft = SimpleNamespace(layers=1, window=8, kv_heads=1, head_dim=8)
        cache = SimpleNamespace(block_bytes=4096, slot_bytes=4096)
        with patch.object(boot.facts, "load") as model_load, \
             patch.object(boot, "Glm53Net"), \
             patch.object(boot.drafter_mod, "load", return_value=draft) as draft_load, \
             patch.object(boot.drafter_mod, "specs", return_value=[]), \
             patch.object(boot, "layout", return_value=cache), \
             patch.object(boot, "rank_loader", side_effect=StopAtBuild):
            with self.assertRaises(StopAtBuild):
                boot.build(SimpleNamespace(rank=0), [0], None, "/ranks", 1., 1, True, None,
                           ckpt_meta="/alternate/config", drafter_dir="/alternate/draft")
            model_load.assert_called_once_with("/alternate/config")
            draft_load.assert_called_once_with(Path("/alternate/draft"))


if __name__ == "__main__":
    unittest.main()
