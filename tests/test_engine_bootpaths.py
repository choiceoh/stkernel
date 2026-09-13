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
    def test_the_repo_default_width_is_the_one_production_serves(self):
        """They disagreed, and that is how two onepass runs 27 minutes apart on one commit came
        out incomparable: one captured widths 1-4 with a 1,035,264 ceiling, the other 1-8 with
        364,032, and nothing in either record said so (45차 §72).

        4 is not a smaller engine, it is the measured one. From two ledgers, same commit:
        graph pool 0.60 vs 2.50 GiB, captured ceiling 1,035,264 vs 364,032, state slots 1.21 vs
        2.17 GiB, boot 161.9 vs 208.7 s. 8 existed for kernel coverage (48 target tokens at
        K=5) and 24 is inside the same kernels.
        """
        from engine.profiles.glm53 import boot
        self.assertEqual(boot.MAX_SEQS, 4)
        for path in sorted(Path("/home/choiceoh/st-releases").glob("*/engine/profiles/glm53/boot.py")):
            if "prod-" not in path.parts[-5]:
                continue                                        # only what has actually served
            declared = [l for l in path.read_text().splitlines() if l.startswith("MAX_SEQS")]
            self.assertTrue(declared and declared[0].split("=")[1].split("#")[0].strip() == "4",
                            f"{path.parts[-5]} serves a width this repo no longer defaults to")

    def test_local_http_and_fleet_forward_both_model_directories(self):
        from engine.profiles.glm53 import boot
        args = SimpleNamespace(ckpt_meta="/alternate/config", drafter_dir="/alternate/draft",
                               ranks="/alternate/ranks", layers="0-0", seed=0, prompt=1,
                               seqs=1, kv_gib=.25, park=False, drafter=True, max_new=1,
                               temperature=0., tier_dir="/unused", port=8000, lanes="reference")
        comm = SimpleNamespace(rank=0, world_size=4, close=Mock(), prepare_oneshot=Mock())
        tp = SimpleNamespace(run=lambda fn: fn(comm))
        for mode in ("local", "http", "fleet"):
            args.serve = mode == "http"
            # the fleet path refuses to boot without a reservation (fleet_lease_of), so this one holds it
            with self.subTest(mode=mode), \
                 patch.object(boot, "fleet_lease_of", return_value={"owner": "test", "path": "/unused"}), \
                 patch.object(boot.facts, "check_box", return_value="test"), \
                 patch.object(boot, "declared") as declared, \
                 patch.object(boot, "LocalTP", return_value=tp), \
                 patch.object(boot.Comm, "init", return_value=comm), \
                 patch.object(boot.lane_tables, "reference"), \
                 patch.object(boot.lane_tables, "served"), \
                 patch.object(boot, "build", side_effect=StopAtBuild) as build:
                declared.return_value.__getitem__.side_effect = lambda k: {
                    "execution_overlap": 0, "early_observe": 0, "prefill_tiles": 1, "direct_mhc": 0, "prefill_project_tiles": 0,
                    "nvme_mapped_staging": 0,
                    "moe_static": "t,r,sf6,q0", "mla_prefill": "tile32", "context_ceiling": 0,
                    "execution": "native", "kda_state_dtype": "fp32"}[k]
                with self.assertRaises(StopAtBuild):
                    (boot.fleet if mode == "fleet" else boot.local)(args)
                self.assertEqual(build.call_args.kwargs["ckpt_meta"], args.ckpt_meta)
                self.assertEqual(build.call_args.kwargs["drafter_dir"], args.drafter_dir)
        comm.close.assert_called_once_with()

    def test_build_reads_drafter_facts_from_the_selected_directory(self):
        from engine.profiles.glm53 import boot
        draft = SimpleNamespace(layers=1, window=8, block=8, kv_heads=1, head_dim=8)
        cache = SimpleNamespace(block_bytes=4096, slot_bytes=4096)
        with patch.object(boot.facts, "load") as model_load, \
             patch.object(boot, "Glm53Net"), \
             patch.object(boot.drafter_mod, "load", return_value=draft) as draft_load, \
             patch.object(boot.drafter_mod, "specs", return_value=[]), \
             patch.object(boot, "layout", return_value=cache), \
             patch.object(boot, "cache_capacity", return_value=(100, 9)), \
             patch.object(boot, "rank_loader", side_effect=StopAtBuild):
            with self.assertRaises(StopAtBuild):
                boot.build(SimpleNamespace(rank=0), [0], None, "/ranks", 1., 1, True, None,
                           ckpt_meta="/alternate/config", drafter_dir="/alternate/draft")
            model_load.assert_called_once_with("/alternate/config")
            draft_load.assert_called_once_with(Path("/alternate/draft"))


class LauncherTests(unittest.TestCase):
    """The fleet launcher's shape: the nodes prepare in parallel and fail together."""

    def setUp(self):
        self.text = (Path(__file__).resolve().parents[1] / "launchers/start-st-glm53.sh").read_text()

    def test_nodes_start_in_parallel_and_report_in_rank_order(self):
        self.assertIn('start_rank "$r" >"$stage/rank$r.log" 2>&1 &', self.text)
        self.assertIn('pids[$r]=$!', self.text)
        self.assertIn('wait "${pids[$r]}"', self.text)
        # the buffered output is printed in the loop that waits, so rank order survives
        self.assertLess(self.text.index('pids[$r]=$!'), self.text.index('cat "$stage/rank$r.log"'))

    def test_a_node_that_fails_stops_the_rest(self):
        self.assertIn('failed="$failed $r"', self.text)
        self.assertIn('bash "$0" stop', self.text)
        # a failing step inside a node returns, it does not exit the whole script mid-fleet
        self.assertNotIn("exit 1; }", self.text.split("start_rank() {")[1].split("\npids=()")[0])
        self.assertIn("return 1; }", self.text)


if __name__ == "__main__":
    unittest.main()
