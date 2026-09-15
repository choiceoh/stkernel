"""Alternate model/drafter paths must reach every boot mode before allocation."""
import datetime
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class StopAtBuild(Exception):
    pass


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class BootPathTests(unittest.TestCase):
    def test_the_repo_default_width_is_the_one_production_serves(self):
        """Current declarations agree; immutable older releases keep their own width."""
        from engine.profiles.glm53 import boot
        from engine.base.config import Config
        self.assertEqual(boot.MAX_SEQS, 2)
        for production in (False, True):
            args = SimpleNamespace(production=production, ckpt_meta="/meta", ranks="/ranks", kv_gib=24., port=8000)
            with patch.object(boot, "Config", side_effect=lambda facts, knobs: Config(
                    facts, knobs, env={}, today=datetime.date(2026, 9, 14))):
                cfg = boot.declared(args, 4)
            self.assertEqual(cfg["max_seqs"], boot.MAX_SEQS)

    def test_an_empty_tier_directory_reaches_the_fleet_boot_as_no_tier(self):
        """The launcher's `--tier-dir=` (tier off) parses to "", which every tier guard reads as none."""
        from engine.profiles.glm53 import boot
        with patch("engine.runtime.verify.verify"), patch.object(boot, "fleet", return_value=0) as fleet:
            boot.main(["--tier-dir="])
            boot.main(["--tier-dir", "/x/tier"])
        self.assertEqual([c.args[0].tier_dir for c in fleet.call_args_list], ["", "/x/tier"])

    def test_local_http_and_fleet_forward_both_model_directories(self):
        from engine.base import kernel_shape
        from engine.base.config import Config
        from engine.profiles.glm53 import boot
        from tests.test_engine_glm53 import tiny_facts
        kernel_shape.reset()                         # every mode binds the checkpoint's shape once per process
        self.addCleanup(kernel_shape.reset)
        args = SimpleNamespace(ckpt_meta="/alternate/config", drafter_dir="/alternate/draft",
                               ranks="/alternate/ranks", layers="0-0", seed=0, prompt=1,
                               seqs=1, kv_gib=.25, park=False, drafter=True, max_new=1,
                               temperature=0., tier_dir="/unused", port=8000, lanes="reference")
        comm = SimpleNamespace(rank=0, world_size=4, close=Mock(), prepare_oneshot=Mock(),
                               transport=SimpleNamespace(rails=2, latency={}))
        tp = SimpleNamespace(run=lambda fn: fn(comm))
        for mode, production in (("local", False), ("http", False), ("fleet", False), ("fleet", True)):
            args.serve = mode == "http"
            args.production = production
            # the fleet path refuses to boot without a reservation (fleet_lease_of), so this one holds it
            with self.subTest(mode=mode, production=production), \
                 patch.object(boot, "fleet_lease_of", return_value={"owner": "test", "path": "/unused"}), \
                 patch.object(boot.facts, "check_box", return_value="test"), \
                 patch.object(boot.facts, "load", return_value=tiny_facts()) as facts_load, \
                 patch.object(boot, "Config", side_effect=lambda facts, knobs: Config(
                     facts, knobs, env={}, today=datetime.date(2026, 9, 13))), \
                 patch.object(boot, "LocalTP", return_value=tp), \
                 patch.object(boot.Comm, "init", return_value=comm), \
                 patch.object(boot.lane_tables, "reference"), \
                 patch.object(boot.lane_tables, "served"), \
                 patch.object(boot, "build", side_effect=StopAtBuild) as build:
                with self.assertRaises(StopAtBuild):
                    (boot.fleet if mode == "fleet" else boot.local)(args)
                self.assertEqual(build.call_args.kwargs["ckpt_meta"], args.ckpt_meta)
                self.assertEqual(build.call_args.kwargs["drafter_dir"], args.drafter_dir)
                self.assertEqual(build.call_args.args[5], 2)
                if mode == "fleet":
                    comm.prepare_oneshot.assert_called_with(rails=2, inline_flags=True)
                    plan = build.call_args.kwargs["execution_plan"]
                    self.assertEqual((plan.direct_mhc, plan.prefill_project_tiles, plan.decode_iterations),
                                     (True, True, 4))
                    self.assertEqual((plan.overlap, plan.early_observe, plan.prefill_tiles), (False, False, 1))
                    self.assertTrue(build.call_args.kwargs["nvme_mapped_staging"])
                    self.assertEqual(build.call_args.kwargs["kda_state_dtype"], "fp32")
                else:
                    # LocalTP/reference builds do not own the real TP4 transport.
                    self.assertNotIn("execution_plan", build.call_args.kwargs)
                facts_load.assert_called_with(args.ckpt_meta)        # the kernel shape comes from the selected config
                self.assertTrue(kernel_shape.is_bound())
                # no --workspace-gib: build takes the profile's ceiling (budget.WORKSPACE_GIB)
                self.assertIsNone(build.call_args.kwargs["workspace_gib"])
        self.assertEqual(comm.close.call_args_list, [unittest.mock.call(), unittest.mock.call()])
        args.workspace_gib = 10.5                                   # a shape that spends more says so, and every mode passes it on
        for mode in ("local", "fleet"):
            args.serve, args.production = False, False
            with self.subTest(mode=mode, workspace_gib=10.5), \
                 patch.object(boot, "fleet_lease_of", return_value={"owner": "test", "path": "/unused"}), \
                 patch.object(boot.facts, "check_box", return_value="test"), \
                 patch.object(boot.facts, "load", return_value=tiny_facts()), \
                 patch.object(boot, "Config", side_effect=lambda facts, knobs: Config(
                     facts, knobs, env={}, today=datetime.date(2026, 9, 13))), \
                 patch.object(boot, "LocalTP", return_value=tp), \
                 patch.object(boot.Comm, "init", return_value=comm), \
                 patch.object(boot.lane_tables, "reference"), \
                 patch.object(boot.lane_tables, "served"), \
                 patch.object(boot, "build", side_effect=StopAtBuild) as build:
                with self.assertRaises(StopAtBuild):
                    (boot.fleet if mode == "fleet" else boot.local)(args)
                self.assertEqual(build.call_args.kwargs["workspace_gib"], 10.5)

    def test_build_reads_drafter_facts_from_the_selected_directory(self):
        from engine.base import kernel_shape
        from engine.profiles.glm53 import boot
        kernel_shape.reset()                         # build binds the drafter's geometry from what it loaded
        self.addCleanup(kernel_shape.reset)
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
            self.assertEqual(kernel_shape.drafter().head_dim, 8)     # the draft kernels admit what was loaded


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

    def test_the_nvme_tier_is_off_unless_a_directory_is_named(self):
        start = self.text.index("TIER_DIR=${ST_TIER_DIR:-off}")
        block = self.text[start:self.text.index("esac", start) + len("esac")]
        self.assertIn("--drafter-dir $DRAFTER $TIER_ARG --dump-dir $DUMP_DIR", self.text)
        for env, want in (({}, "--tier-dir="), ({"ST_TIER_DIR": "off"}, "--tier-dir="),
                          ({"ST_TIER_DIR": "/x/tier"}, "--tier-dir /x/tier")):
            out = subprocess.run(["bash", "-c", block + '\nprintf %s "$TIER_ARG"'], env={"PATH": os.environ["PATH"], **env},
                                 capture_output=True, text=True, check=True).stdout
            self.assertEqual(out, want)

    def test_a_node_that_fails_stops_the_rest(self):
        self.assertIn('failed="$failed $r"', self.text)
        self.assertIn('bash "$0" stop', self.text)
        # a failing step inside a node returns, it does not exit the whole script mid-fleet
        self.assertNotIn("exit 1; }", self.text.split("start_rank() {")[1].split("\npids=()")[0])
        self.assertIn("return 1; }", self.text)


if __name__ == "__main__":
    unittest.main()
