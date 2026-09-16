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
        order = []
        comm = SimpleNamespace(rank=0, world_size=4, close=Mock(),
                               prepare_oneshot=Mock(side_effect=lambda **kw: order.append("one-shot")),
                               wait_prepared=Mock(side_effect=lambda phase, **kw: order.append(phase)),
                               transport=SimpleNamespace(rails=2, latency={}))
        built = []
        native_builds = [("dense", lambda: built.append("dense")), ("one-shot", lambda: built.append("one-shot"))]
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
                 patch.object(boot.lane_tables, "import_kernels"), \
                 patch.object(boot.natives, "builds", return_value=native_builds) as natives, \
                 patch.object(boot, "build", side_effect=StopAtBuild) as build:
                order.clear(), built.clear()
                with self.assertRaises(StopAtBuild):
                    (boot.fleet if mode == "fleet" else boot.local)(args)
                if mode == "fleet":
                    # every native is built, and the ranks meet, before the one-shot transport's first sum
                    natives.assert_called_once_with(2, True)
                    self.assertEqual(sorted(built), ["dense", "one-shot"])
                    self.assertEqual(order[:2], ["native-builds", "one-shot"])
                else:
                    natives.assert_not_called()
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
                 patch.object(boot.lane_tables, "import_kernels"), \
                 patch.object(boot.natives, "builds", return_value=native_builds), \
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

    def test_the_nvme_tier_is_on_by_default_and_only_the_word_off_takes_it_away(self):
        """Since 2026-09-16 the default is a tier, not none: `off` is a deliberate word, not a fallback.

        With no tier a finished turn is never registered as a conversation, so its prefix boundaries
        are dropped when the row is reclaimed -- production measured 17 hits in 100 queries against
        1,144 evictions with 97% of the blocks free (MEASUREMENTS.md, PR #1042).
        """
        for env, want in (({}, "--tier-dir /home/choiceoh/glm53-logs/st-tier"),
                          ({"ST_TIER_DIR": "off"}, "--tier-dir="),
                          ({"ST_TIER_DIR": "/home/choiceoh/glm53-logs/st-bracket-tier/x-base"},
                           "--tier-dir /home/choiceoh/glm53-logs/st-bracket-tier/x-base")):
            self.assertEqual(self._tier_arg(env), want, env)
        self.assertIn("--drafter-dir $DRAFTER $TIER_ARG --dump-dir $DUMP_DIR", self.text)

    def test_the_kv_budget_production_serves_on_is_in_the_tree_not_on_the_box(self):
        """Production's KV came from a hand-edited env file. The deploy relaunches from the tree.

        14.0 is what 2026-09-16 measured: 2,987 blocks, declared paged KV 13.16 GiB, unassigned
        +19.59 GiB, door up in 135 s on the first attempt (PR #1042). boot.py's own 24.0 is the
        vLLM-parity default for a single box, not this fleet's shape.
        """
        self.assertIn("KV_GIB=${ST_KV_GIB:-14.0}", self.text)
        kv = '$KV_GIB"'
        for env, want in (({}, "--kv-gib 14.0"), ({"ST_KV_GIB": "7.0"}, "--kv-gib 7.0")):
            self.assertEqual(self._launcher_var("KV_ARG", "KV_GIB=${ST_KV_GIB", kv, env), want, env)
        # a budget that is not a number is a refusal, not a `--kv-gib` the boot has to parse
        self.assertEqual(self._launcher_var("KV_ARG", "KV_GIB=${ST_KV_GIB", kv,
                                            {"ST_KV_GIB": "lots"}, check=False), ("", 2))

    def test_a_tier_off_the_one_mounted_directory_is_refused_not_quietly_made_ephemeral(self):
        """A tier the containers cannot see boots fine and throws everything away with the container.

        `docker run` binds exactly one host directory for state. A tier outside it lands in the
        container's writable layer: live counters, working reuse within the boot, nothing after it.
        `~/st-tier` did that to a production boot on 2026-09-16, and `~/st-fleet.lock` did it to the
        fleet lease before that -- so the launcher refuses instead of booting.
        """
        root = "/home/choiceoh/glm53-logs"
        self.assertIn("MOUNTED_ROOT=" + root, self.text)
        self.assertIn("-v %s:%s " % (root, root), self.text,
                      "MOUNTED_ROOT must name the directory `docker run` actually binds")
        for bad in ("/home/choiceoh/st-tier",      # the one that cost the 09-16 boot
                    "/home/choiceoh/glm53-logs",   # the root itself is not a tier
                    "/tmp/tier", "off2", "relative/tier"):
            out, code = self._tier_arg({"ST_TIER_DIR": bad}, check=False)
            self.assertEqual(code, 2, bad)
            self.assertEqual(out, "", "a refused tier must not leave a half-built argument")

    @staticmethod
    def _bash(script, env):
        """Run a snippet of the launcher. Bytes, and through stdin.

        The snippets carry `;;`, quotes and parens, so as one `bash -c` argument they are at the
        mercy of the host's argument quoting; a temp file hands bash a path its host may not
        resolve; and text=True writes stdin through the platform's newline translation, where a
        shell reading `case ... in\\r` says only `syntax error`. Bytes over stdin are the same
        everywhere.
        """
        run = subprocess.run(["bash", "-s"], input=script.encode(), env=env, capture_output=True)
        return SimpleNamespace(returncode=run.returncode,
                               stdout=run.stdout.decode(errors="replace"),
                               stderr=run.stderr.decode(errors="replace"))

    def _launcher_var(self, var, first, last, env, check=True):
        """Run the launcher block from `first` through `last`, and print what it set `var` to."""
        # A `bash` that does not inherit the environment cannot answer these (Windows resolves the
        # name to WSL's, which starts a fresh Linux environment). Skip rather than read its default
        # as an answer -- every case would come back as the default and some would "pass".
        if self._bash('printf %s "$ST_PROBE"\n', {**os.environ, "ST_PROBE": "reached"}).stdout \
                != "reached":
            self.skipTest("this box's `bash` does not inherit the environment")
        start = self.text.index(first)
        block = self.text[start:self.text.index(last, start) + len(last)]
        blank = {name: "" for name in ("ST_TIER_DIR", "ST_KV_GIB")}  # unset, whatever ran the suite
        run = self._bash(block + '\nprintf %%s "$%s"\n' % var, {**os.environ, **blank, **env})
        if not check:
            return run.stdout, run.returncode
        self.assertEqual(run.returncode, 0, run.stderr)
        return run.stdout

    def _tier_arg(self, env, check=True):
        return self._launcher_var("TIER_ARG", "MOUNTED_ROOT=", "esac", env, check)

    def test_a_node_that_fails_stops_the_rest(self):
        self.assertIn('failed="$failed $r"', self.text)
        self.assertIn('bash "$0" stop', self.text)
        # a failing step inside a node returns, it does not exit the whole script mid-fleet
        self.assertNotIn("exit 1; }", self.text.split("start_rank() {")[1].split("\npids=()")[0])
        self.assertIn("return 1; }", self.text)


if __name__ == "__main__":
    unittest.main()
