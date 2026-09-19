"""The deploy watcher's three refusals (45차 §34 뒤).

Everything here is the part that DECIDES. The part that acts -- cutting a release, stopping the
supervisor, launching four nodes -- is not exercised: it can only be judged by doing it, and doing
it takes production down. What is judged is every way this says no, because a deploy watcher is
worth exactly what its refusals are worth.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launchers"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "st_deploy_watch", Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class WantedTests(unittest.TestCase):
    def test_auto_deploy_and_recovery_inherit_the_same_production_environment(self):
        root = Path(__file__).resolve().parents[1] / 'launchers'
        def files(unit):
            return [line.split('=', 1)[1] for line in (root / unit).read_text().splitlines()
                    if line.startswith('EnvironmentFile=')]
        recovery = files('st-glm53.service')
        self.assertTrue(recovery, 'production configuration must be explicit')
        self.assertEqual(files('st-deploy-watch.service'), recovery)

    def test_the_deployed_sha_is_not_a_candidate(self):
        self.assertIsNone(watch.wanted("abc123", {"deployed": "abc123"}))

    def test_a_moved_main_is(self):
        self.assertIn("abc123", watch.wanted("abc123", {"deployed": "def456"}))

    def test_a_first_deploy_is_a_candidate(self):
        self.assertIsNotNone(watch.wanted("abc123", {}))

    def test_a_rejected_sha_is_not_retried_until_main_moves_past_it(self):
        """A merge the gate refused must not become a restart loop."""
        held = {"deployed": "old", "rejected": "bad"}
        self.assertIsNone(watch.wanted("bad", held))
        self.assertIsNotNone(watch.wanted("newer", held), "and the next merge is looked at again")

    def test_nothing_to_do_without_a_head(self):
        self.assertIsNone(watch.wanted("", {"deployed": "abc"}))


class BusyTests(unittest.TestCase):
    def metrics(self, running=0, waiting=0, handing=0, quiet=1, drop=()):
        rows = {"vllm:num_requests_running": running, "vllm:num_requests_waiting": waiting,
                "st:handing_over": handing, "st:quiet": quiet}
        return "\n".join(f"{k} {v}" for k, v in rows.items() if k not in drop) + "\n"

    def test_an_idle_engine_reads_zero(self):
        self.assertEqual(watch.busy(self.metrics()), 0)

    def test_a_request_in_the_step_or_in_the_queue_counts(self):
        self.assertEqual(watch.busy(self.metrics(running=2)), 2)
        self.assertEqual(watch.busy(self.metrics(waiting=3)), 3)
        self.assertEqual(watch.busy(self.metrics(running=1, waiting=1)), 2)

    def test_a_handover_in_progress_is_never_quiet(self):
        """The fleet is being given to another session: whatever the counters say, this is not a gap."""
        self.assertGreaterEqual(watch.busy(self.metrics(handing=1)), 1)

    def test_the_engine_s_own_quiet_outranks_the_request_counts(self):
        """A conversation moving to or from the NVMe tier runs on its own thread and is not a request.
        Both gauges read zero through the whole transfer; `st:quiet` is the one that knows."""
        self.assertGreaterEqual(watch.busy(self.metrics(running=0, waiting=0, quiet=0)), 1)
        self.assertEqual(watch.busy(self.metrics(running=0, waiting=0, quiet=1)), 0)

    def test_an_engine_too_old_to_publish_st_quiet_is_not_assumed_quiet(self):
        self.assertIsNone(watch.busy(self.metrics(drop=("st:quiet",))))

    def test_an_engine_that_cannot_be_asked_is_not_known_to_be_quiet(self):
        """None is not zero. A door that answers something unrecognisable must not read as a gap."""
        self.assertIsNone(watch.busy(""))
        self.assertIsNone(watch.busy(self.metrics(drop=("vllm:num_requests_waiting",))))
        self.assertIsNone(watch.busy("vllm:num_requests_running not-a-number\n"))

    def test_labels_and_floats_are_read(self):
        body = ('vllm:num_requests_running{model="glm-5.3"} 0.0\n'
                'vllm:num_requests_waiting{model="glm-5.3"} 0.0\n'
                "st:handing_over 0.0\nst:quiet 1.0\n")
        self.assertEqual(watch.busy(body), 0)

    def test_a_prefix_of_the_name_is_not_the_name(self):
        body = ("vllm:num_requests_running_total 9\nvllm:num_requests_running 0\n"
                "vllm:num_requests_waiting 0\nst:handing_over 0\nst:quiet 1\n")
        self.assertEqual(watch.busy(body), 0)

    def test_the_metric_it_depends_on_is_one_the_door_always_publishes(self):
        """`st:quiet` has to be unconditional in serve.py, or this refuses forever."""
        serve = (Path(__file__).resolve().parents[1] / "engine/base/serve.py").read_text()
        self.assertIn('"st:quiet"', serve)
        self.assertIn("int(self._quiet())", serve, "and it must be _quiet's answer, not a restatement")


class GateTests(unittest.TestCase):
    """A dozen engine test files fail on any tree today, so the gate is a difference, not a bar."""

    def test_the_same_failures_are_not_a_regression(self):
        both = {"test_engine_kda_ring": "FAILED (errors=7)", "test_engine_serve": "FAILED (errors=12)"}
        self.assertEqual(watch.regressed(both, dict(both)), [])

    def test_a_newly_failing_file_is(self):
        before = {"test_engine_serve": "FAILED (errors=12)"}
        after = {**before, "test_engine_sampling": "FAILED (failures=1)"}
        self.assertEqual(watch.regressed(before, after), ["test_engine_sampling"])

    def test_the_same_file_failing_worse_is(self):
        self.assertEqual(watch.regressed({"test_engine_serve": "FAILED (errors=12)"},
                                         {"test_engine_serve": "FAILED (errors=13)"}),
                         ["test_engine_serve"])

    def test_a_file_that_stopped_failing_is_not_held_against_the_candidate(self):
        self.assertEqual(watch.regressed({"test_engine_serve": "FAILED (errors=12)"}, {}), [])

    def test_a_file_with_no_verdict_at_all_is_a_regression(self):
        """A run that crashed before unittest could say anything is not a pass."""
        self.assertEqual(watch.regressed({}, {"test_engine_glm53": "NO VERDICT"}), ["test_engine_glm53"])


class GateContainerTests(unittest.TestCase):
    """The suite runs in the seed image the tree pins, not under the head's python (no torch there: 2026-09-15)."""

    def tree(self, root, names=("test_engine_a", "test_engine_b"), seed="sha256:" + "5" * 64):
        import json
        tree = Path(root)
        (tree / "engine/runtime").mkdir(parents=True)
        if seed is not None:
            (tree / "engine/runtime/dependencies.json").write_text(json.dumps({"seed_image_id": seed}))
        (tree / "tests").mkdir()
        for name in names:
            (tree / "tests" / f"{name}.py").write_text("")
        return tree

    def test_the_files_run_in_the_tree_s_seed_image_with_cuda_hidden_and_no_network(self):
        import json
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            tree = self.tree(root)
            said = "noise\n" + watch.GATE_MARK + json.dumps({"test_engine_a": "OK", "test_engine_b": "FAILED (errors=1, id='42')"})
            with patch.object(watch, "run", return_value=(0, said + "\n", "")) as run:
                self.assertEqual(watch.failures(tree, 900), {"test_engine_b": "FAILED (errors=1, id=..)"})
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[:2], ["docker", "run"])
            for flag in (["--pull", "never"], ["--network", "none"], ["-e", "CUDA_VISIBLE_DEVICES="],
                         ["-e", "NVIDIA_VISIBLE_DEVICES=void"], ["-v", f"{tree}:/repo:ro"]):
                self.assertTrue(any(cmd[i:i + 2] == flag for i in range(len(cmd) - 1)), flag)
            self.assertIn("sha256:" + "5" * 64, cmd)
            self.assertNotIn(sys.executable, cmd[:cmd.index("--entrypoint")])

    def test_a_container_that_says_nothing_leaves_every_file_without_a_verdict(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            tree = self.tree(root)
            with patch.object(watch, "run", return_value=(125, "", "Unable to find image 'sha256:555' locally")):
                after = watch.failures(tree, 900)
        self.assertEqual(sorted(after), ["test_engine_a", "test_engine_b"])
        self.assertTrue(all(v.startswith("NO VERDICT (gate container rc=125") for v in after.values()))
        self.assertEqual(watch.regressed({}, after), ["test_engine_a", "test_engine_b"])   # refused, not waved through

    def test_a_tree_that_pins_no_seed_is_not_judged_on_some_other_image(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            tree = self.tree(root, seed=None)
            with patch.object(watch, "run") as run:
                after = watch.failures(tree, 900)
            run.assert_not_called()
        self.assertTrue(all(v.startswith("NO VERDICT (no seed image pinned") for v in after.values()))

    def test_the_driver_reads_unittest_s_own_verdict_per_file(self):
        """The in-container driver, run here on stdlib unittest: pass, fail, error, and a file that prints OK lines."""
        import json
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            tests = Path(root) / "tests"
            tests.mkdir()
            (tests / "__init__.py").write_text("")
            body = "import unittest\nclass T(unittest.TestCase):\n    def test(self):\n        {}\n"
            (tests / "test_engine_pass.py").write_text(body.format("print('  a check OK')"))
            (tests / "test_engine_fail.py").write_text(body.format("self.fail('no')"))
            (tests / "test_engine_error.py").write_text(body.format("raise RuntimeError('boom')"))
            done = subprocess.run([sys.executable, "-c", watch.GATE_DRIVER, "60", "2"], cwd=root,
                                  capture_output=True, text=True, timeout=120)
        line = next(l for l in done.stdout.splitlines() if l.startswith(watch.GATE_MARK))
        self.assertEqual(json.loads(line[len(watch.GATE_MARK):]),
                         {"test_engine_error": "FAILED (errors=1)", "test_engine_fail": "FAILED (failures=1)",
                          "test_engine_pass": "OK"})


class GateTreeTests(unittest.TestCase):
    """The gate judges commits whole: the tests read bench/, tools/ and measurements/, which a release does not carry."""

    def test_the_gate_block_judges_extracted_commits_not_the_release(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("    if a.gate and deployed_tree is not None:"):source.index("    elif a.gate:")]
        self.assertIn("failures(judged", body)
        self.assertIn("failures(baseline", body)
        self.assertNotIn("failures(release", body)
        self.assertNotIn("failures(deployed_tree", body)
        self.assertIn("return 1", body[:body.index("after = failures")])   # an extraction that failed refuses

    @unittest.skipUnless(__import__("shutil").which("git"), "requires git")
    def test_a_commit_is_extracted_whole_and_once(self):
        import subprocess
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            repo, releases = Path(root) / "repo", Path(root) / "releases"
            (repo / "tests").mkdir(parents=True)
            (repo / "bench").mkdir()
            (repo / "tests/test_engine_x.py").write_text("")
            (repo / "bench/fleet.sh").write_text("#!/bin/sh\n")
            git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(git + ["add", "-A"], check=True)
            subprocess.run(git + ["commit", "-qm", "c"], check=True)
            sha = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            logs = []
            with patch.object(watch, "SOURCE", repo), patch.object(watch, "GATE_TREES", releases / "gate"):
                tree = watch.gate_tree(sha, logs.append)
                self.assertTrue((tree / "bench/fleet.sh").is_file() and (tree / "tests/test_engine_x.py").is_file())
                self.assertEqual(watch.gate_tree(sha, logs.append), tree)
                self.assertIsNone(watch.gate_tree("0" * 40, logs.append))
                self.assertFalse((releases / "gate" / ("0" * 12)).exists())
                watch.prune_gate_trees((sha,))
                self.assertTrue(tree.is_dir())
                watch.prune_gate_trees(("f" * 40,))
                self.assertFalse(tree.exists())


class GateIsNotOptionalTests(unittest.TestCase):
    def test_a_first_deploy_with_nothing_to_compare_against_is_refused_not_waved_through(self):
        """Skipping here would make the only ungated deploy the first one, which nobody is watching."""
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("    elif a.gate:"):]
        body = body[:body.index("\n    log(", 10)]
        self.assertIn("REFUSED", body)
        self.assertIn("return 1", body)
        self.assertIn("--seed", body, "and it says how to give the gate something to compare with")


class OlderEngineTests(unittest.TestCase):
    """`st:quiet` ships in the tree this deploys, so the engine it first meets does not have it."""

    def body(self, quiet=True):
        rows = ["vllm:num_requests_running 0", "vllm:num_requests_waiting 0", "st:handing_over 0"]
        return "\n".join(rows + (["st:quiet 1"] if quiet else [])) + "\n"

    def test_an_engine_without_the_metric_is_named_rather_than_waited_on(self):
        self.assertTrue(watch.unsupported(self.body(quiet=False)))
        self.assertFalse(watch.unsupported(self.body(quiet=True)))

    def test_silence_is_not_the_same_case(self):
        """Waiting fixes a door that was busy answering. It does not fix a tree that predates the metric."""
        self.assertFalse(watch.unsupported(""), "nothing that looks like the door: not this case")
        self.assertFalse(watch.unsupported("some other exporter 1\n"))

    def test_the_cycle_stops_on_it_instead_of_polling_for_an_hour(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        loop = source[source.index('log(f"  waiting for'):]
        loop = loop[:loop.index("    release = cut(")]
        self.assertIn("unsupported(body)", loop)
        self.assertIn("return 0", loop[loop.index("unsupported(body)"):],
                      "it has to leave the cycle, not fall through to the deadline")


class LaunchFailureTests(unittest.TestCase):
    def test_a_launch_that_failed_is_not_recorded_as_deployed(self):
        """What is serving after a failed launch is whatever the supervisor recovered -- not this
        release. Recording it as deployed would both stop the retry and make the next gate compare
        against a tree that is not running."""
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("    ok = deploy(release, log, profile)"):]
        body = body[:body.index("\n\ndef ", 10)]
        self.assertIn("if not ok:", body)
        self.assertIn('"rejected": head', body[body.index("if not ok:"):], "a failed launch is a rejection")
        self.assertNotIn('"deployed": head', body[body.index("if not ok:"): body.index("return 1")])
        self.assertIn('"release": None', body[body.index("if not ok:"):],
                      "and the baseline goes too: the rsync lands before the step that failed, so "
                      "which tree the supervisor recovered onto is not known")


class ReleaseCuttingTests(unittest.TestCase):
    def test_a_half_written_archive_cannot_be_published_as_a_release(self):
        """Without pipefail the exit code is tar's, and tar extracts the prefix of a dead stream
        happily -- a release that looks whole, with a tests/ the gate then silently under-runs."""
        source = (Path(__file__).resolve().parents[1] / "launchers/st_release.py").read_text()
        body = source[source.index("def cut("):]
        body = body[:body.index("\n\ndef ", 10)]
        self.assertIn("set -o pipefail", body)
        self.assertIn("| tar -x", body)
        watch = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        self.assertIn("st_release.cut(sha, source=SOURCE, releases=RELEASES", watch)   # one cut, shared with the bracket

    def test_pipefail_is_what_it_claims_to_be(self):
        """The property the line rests on, checked against the shell rather than assumed."""
        import subprocess
        without = subprocess.run(["bash", "-c", "false | true"]).returncode
        with_it = subprocess.run(["bash", "-c", "set -o pipefail; false | true"]).returncode
        self.assertEqual(without, 0)
        self.assertNotEqual(with_it, 0)


class StateTests(unittest.TestCase):
    def test_an_unreadable_state_file_reads_as_nothing_deployed(self):
        self.assertEqual(watch.state_of(Path("/nonexistent/deploy-state.json")), {})

    def test_the_units_point_at_the_rsynced_tree_not_a_worktree(self):
        """systemd runs what is on the head node, which is what the launcher rsynced there."""
        here = Path(__file__).resolve().parents[1] / "launchers"
        unit = (here / "st-deploy-watch.service").read_text()
        self.assertIn("/home/choiceoh/st-engine/launchers/st-deploy-watch.py", unit)
        self.assertIn("--once", unit, "the loop belongs to the timer, not to a service that never exits")
        self.assertIn("After=st-glm53.service", unit, "it restarts that service: it must not race its start")
        self.assertIn("KillMode=process", unit, "the probe waiter a cycle queues must outlive the oneshot's cgroup "
                                               "(the first armed cycle's ticket died 0.25 s after enqueue, 2026-09-13)")
        timer = (here / "st-deploy-watch.timer").read_text()
        self.assertNotIn("Persistent=true", timer, "a missed cycle sees the same main; there is nothing to catch up")


class AfterDeployTests(unittest.TestCase):
    """After a deploy the queue follows: the controller checkout moves to the deployed commit and one
    D17 probe ticket is queued for it. Neither is allowed to fail the deploy, and a dry run does neither."""

    def setUp(self):
        import tempfile
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp = Path(self.temporary.name)
        self.lines = []
        # These shas are not in any tree, and the probe now asks the commit whether it claims speed
        # (`claims_speed`). What is judged here is the ticket, so the claim is stated rather than read.
        claims = patch.object(watch, "commit_message", return_value="perf: a speed claim")
        claims.start()
        self.addCleanup(claims.stop)

    def log(self, line):
        self.lines.append(line)

    def git(self, repo, *args):
        import subprocess
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              env={**dict(__import__("os").environ), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                                   "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.strip()

    def commit(self, repo, name):
        (repo / name).write_text(name)
        self.git(repo, "add", name)
        self.git(repo, "commit", "--quiet", "-m", name)
        return self.git(repo, "rev-parse", "HEAD")

    def stub_controller(self, rc=0):
        controller = self.tmp / "controller"
        (controller / "bench").mkdir(parents=True)
        fleet = controller / "bench" / "fleet.sh"
        fleet.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{self.tmp}/argv"\necho queued; exit {rc}\n')
        return controller

    def test_the_controller_checkout_moves_to_the_deployed_commit(self):
        """A controller 639 commits behind main once called a serving fleet FREE (45차 §91)."""
        origin = self.tmp / "origin"
        origin.mkdir()
        self.git(origin, "init", "--quiet", "-b", "main")
        self.commit(origin, "one")
        self.git(self.tmp, "clone", "--quiet", str(origin), "controller")
        controller = self.tmp / "controller"
        two = self.commit(origin, "two")
        self.assertNotEqual(self.git(controller, "rev-parse", "HEAD"), two)
        self.assertTrue(watch.follow_controller(two, self.log, controller))
        self.assertEqual(self.git(controller, "rev-parse", "HEAD"), two)
        self.assertIn(f"now at {two[:12]}", "\n".join(self.lines))

    def test_a_directory_that_is_not_a_checkout_is_left_alone(self):
        plain = self.tmp / "plain"
        plain.mkdir()
        self.assertFalse(watch.follow_controller("0" * 40, self.log, plain))
        self.assertIn("not a checkout", "\n".join(self.lines))
        self.assertFalse(watch.follow_controller("0" * 40, self.log, self.tmp / "absent"))

    def test_the_probe_is_one_ticket_in_the_live_lane_for_the_deployed_commit(self):
        controller = self.stub_controller()
        head = "0123abcdef" * 4
        self.assertTrue(watch.queue_probe(head, self.log, controller))
        argv = (self.tmp / "argv").read_text().splitlines()
        self.assertEqual(argv, ["st-probe", "--detach", "d17-0123abcdef01", head, "10", "D17 after deploy 0123abcdef01"])
        self.assertIn("D17 probe d17-0123abcdef01 queued", "\n".join(self.lines))

    def test_a_refused_ticket_is_reported_and_is_not_a_failed_deploy(self):
        controller = self.stub_controller(rc=3)
        self.assertFalse(watch.queue_probe("0" * 40, self.log, controller))
        self.assertIn("was not queued (rc=3)", "\n".join(self.lines))
        self.assertFalse(watch.queue_probe("0" * 40, self.log, self.tmp / "absent"))
        self.assertIn("no D17 probe queued", "\n".join(self.lines))

    def test_a_dry_run_and_the_two_flags_hold_it_back(self):
        from types import SimpleNamespace
        from unittest import mock
        controller = self.stub_controller()
        state = mock.patch.object(watch, "STATE", self.tmp / "deploy-state.json")
        state.start()
        self.addCleanup(state.stop)
        watch.after_deploy("0" * 40, SimpleNamespace(dry_run=True, controller=str(controller)), self.log)
        self.assertFalse((self.tmp / "argv").exists(), "a dry run queues nothing")
        watch.after_deploy("0" * 40, SimpleNamespace(dry_run=False, controller=str(controller), follow=False, probe=False), self.log)
        self.assertFalse((self.tmp / "argv").exists())
        watch.after_deploy("0" * 40, SimpleNamespace(dry_run=False, controller=str(controller), follow=False, probe=True), self.log)
        self.assertTrue((self.tmp / "argv").exists())
        self.assertNotIn("controller", "\n".join(self.lines), "--no-follow: the checkout was not even looked at")
        tally = watch.state_of(self.tmp / "deploy-state.json")["probe"]
        self.assertEqual((tally["sha"], tally["attempts"], tally["queued"]), ("0" * 40, 1, True), "the ticket is tallied")

    def test_a_deploy_that_claims_no_speed_queues_no_probe(self):
        """The behaviour this gate exists for: most deploys are a `fix:` or a `docs:`, and a probe
        reserves the door for a whole bracket -- every other request answers 409 while it runs."""
        from types import SimpleNamespace
        from unittest import mock
        controller = self.stub_controller()
        state = mock.patch.object(watch, "STATE", self.tmp / "deploy-state.json")
        state.start()
        self.addCleanup(state.stop)
        with patch.object(watch, "commit_message", return_value="fix: a thing that is not speed"):
            watch.after_deploy("0" * 40, SimpleNamespace(dry_run=False, controller=str(controller),
                                                         follow=False, probe=True), self.log)
        self.assertFalse((self.tmp / "argv").exists(), "no ticket for a commit that claims nothing")
        self.assertIn("no D17 probe", chr(10).join(self.lines))

    def test_it_runs_only_after_a_deploy_that_is_recorded(self):
        """A failed launch is a rejection; nothing follows it, and no probe samples a fleet in recovery."""
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("    ok = deploy(release, log, profile)"):]
        body = body[:body.index("\n\ndef ", 10)]
        failed = body[body.index("if not ok:"):body.index("return 1")]
        self.assertNotIn("after_deploy", failed)
        recorded = body[body.index('"launched_ok": True'):]
        self.assertIn("after_deploy(head, a, log, profile)", recorded)
        for flag in ("--controller", "--no-follow", "--no-probe"):
            self.assertIn(flag, source)


class ProbeSelfHealTests(unittest.TestCase):
    """The deployed commit keeps its warm sample. The ticket after the deploy is the first; when
    it ran into traffic or was cancelled, later cycles queue another -- bounded, tallied, never
    while one is already queued or holding, never when the judge cannot be asked."""

    SHA = "0123abcdef" * 4

    def setUp(self):
        import shutil
        import tempfile
        from types import SimpleNamespace
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp = Path(self.temporary.name)
        self.controller = self.tmp / "controller"
        (self.controller / "bench").mkdir(parents=True)
        (self.controller / "bench" / "fleet.sh").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{self.tmp}/argv"\necho queued; exit 0\n')
        shutil.copy(Path(__file__).resolve().parents[1] / "bench/st_judge.py", self.controller / "bench" / "st_judge.py")
        self.fleet = self.tmp / "fleet"
        self.fleet.mkdir()
        (self.fleet / "queue").write_text("")
        self.jsonl = self.tmp / "onepass.jsonl"
        self.state = self.tmp / "deploy-state.json"
        self.lines = []
        self.a = SimpleNamespace(dry_run=False, probe=True, controller=str(self.controller), probe_attempts=3, probe_gap=0)

    def log(self, line):
        self.lines.append(line)

    def ensure(self, held=None, sha=None, a=None, boot="live|now", tree="", claims="perf: a speed claim"):
        # The commit's own words are the gate now (`claims_speed`): these tests are about the ticket,
        # so they say the claim outright instead of reaching for a git tree that is not here.
        with patch.object(watch, "commit_message", return_value=claims):
            return watch.ensure_probe(self.SHA if sha is None else sha, held or {}, a or self.a, self.log,
                                      controller=self.controller, fleet_dir=self.fleet, jsonl=self.jsonl,
                                      state=self.state, boot=boot, tree=tree)

    def sample(self, sha=None, run_index=2, boot="b|1", tree=None):
        import json as _json
        row = {"engine": "st", "arm_sha": sha or self.SHA, "run_index": run_index, "boot_id": boot, "quality": {"ok": 9, "total": 9},
               "korean": {"dirty": 0, "n": 5}, "decode": {"windows_med": 12.0}, "traffic": {"issues": []}}
        if tree:
            row["arm_tree"] = tree
        with self.jsonl.open("a") as fh:
            fh.write(_json.dumps(row) + "\n")

    def queued(self):
        return (self.tmp / "argv").exists()

    def test_no_sample_and_no_ticket_means_one_more_ticket(self):
        self.assertTrue(self.ensure())
        argv = (self.tmp / "argv").read_text().splitlines()
        self.assertEqual(argv[:3], ["st-probe", "--detach", "d17-0123abcdef01"])
        tally = watch.state_of(self.state)["probe"]
        self.assertEqual((tally["sha"], tally["attempts"], tally["queued"]), (self.SHA, 1, True))
        self.assertIn("has 0 of 1 warm samples and no probe ticket on its way (attempt 1/3 on this boot)", "\n".join(self.lines))

    def test_a_warm_sample_is_enough(self):
        self.sample()
        self.assertFalse(self.ensure())
        self.assertFalse(self.queued())
        self.jsonl.unlink()
        self.sample(run_index=1)                                     # a boot's cold run is not the sample
        self.assertTrue(self.ensure())

    def test_the_adopted_candidate_s_sample_spares_the_deployed_commit_a_probe(self):
        """The operator's rule: a candidate the bracket measured and main adopted needs no probe --
        its measurement is the baseline. The squash has another sha; the engine tree is the same."""
        self.sample(sha="dddd" * 10, tree="0d3c61aa802d")
        self.assertFalse(self.ensure(tree="0d3c61aa802d"))
        self.assertFalse(self.queued())
        self.assertTrue(self.ensure(tree="ffffffffffff"), "another engine tree: no sample of its own")

    def test_a_boot_that_gave_its_sample_is_not_probed_again_and_nothing_serving_is_not_probed(self):
        from types import SimpleNamespace
        two = SimpleNamespace(dry_run=False, probe=True, controller=str(self.controller), probe_attempts=3, probe_gap=0, probe_samples=2)
        self.sample(boot="live|now")
        self.assertFalse(self.ensure(a=two, boot="live|now"), "this boot already gave its sample")
        self.assertTrue(self.ensure(a=two, boot="next|later"), "the next production boot gives the next")
        (self.tmp / "argv").unlink()
        self.assertFalse(self.ensure(a=two, boot=None), "nothing serves: nothing to sample")
        held = {"probe": {"sha": self.SHA, "boot": "old|1", "attempts": 3, "last_at": 0}}
        self.assertTrue(self.ensure(held, a=two, boot="next|later"), "a new boot starts its own count")

    def test_a_sample_of_another_commit_is_not_this_one_s(self):
        self.sample(sha="deadbeef00" * 4)
        self.assertTrue(self.ensure())

    def test_a_ticket_already_queued_or_holding_is_not_doubled(self):
        (self.fleet / "queue").write_text("7|d17-0123abcdef01|100|10|D17|probe|4242\n")
        self.assertFalse(self.ensure())
        (self.fleet / "queue").write_text("7|d17-0123abcdef01-2|100|10|D17|probe|4242\n")     # a later attempt counts too
        self.assertFalse(self.ensure())
        (self.fleet / "queue").write_text("7|d17-0123abcdef0199|100|10|D17|probe|4242\n")    # another sha does not
        self.assertTrue(self.ensure())
        (self.tmp / "argv").unlink()
        (self.fleet / "queue").write_text("")
        (self.fleet / "holder").write_text("d17-0123abcdef01|4242|srv2|100|10|D17|probe\n")
        self.assertFalse(self.ensure())
        (self.fleet / "holder").unlink()
        self.assertTrue(self.ensure())

    def test_every_attempt_gets_a_name_of_its_own(self):
        """The launch layer answers a same-name, same-arguments detached launch with the old
        launch's record -- disposition "existing", its exit code replayed -- so a ticket that died
        cannot be queued again under its own name (srv2, 2026-09-13 02:51: rc=143 replayed)."""
        self.assertEqual(watch.probe_session(self.SHA), "d17-0123abcdef01")
        self.assertEqual(watch.probe_session(self.SHA, 2), "d17-0123abcdef01-2")
        held = {"probe": {"sha": self.SHA, "boot": "live|now", "attempts": 1, "last_at": 0}}
        self.assertTrue(self.ensure(held))
        self.assertEqual((self.tmp / "argv").read_text().splitlines()[2], "d17-0123abcdef01-2")
        self.assertEqual(watch.state_of(self.state)["probe"]["attempts"], 2)

    def test_the_tally_bounds_it(self):
        held = {"probe": {"sha": self.SHA, "boot": "live|now", "attempts": 3, "last_at": 0}}
        self.assertFalse(self.ensure(held))
        self.assertFalse(self.queued())
        self.assertIn("queuing no more", "\n".join(self.lines))
        self.assertTrue(watch.state_of(self.state)["probe"]["gave_up"])
        n = len(self.lines)
        self.assertFalse(self.ensure(watch.state_of(self.state)))
        self.assertEqual(len(self.lines), n, "said once")
        self.assertTrue(self.ensure({"probe": {"sha": "f" * 40, "boot": "live|now", "attempts": 3, "last_at": 0}}), "another sha starts afresh")

    def test_the_gap_between_two_tickets_is_kept(self):
        import time
        from types import SimpleNamespace
        held = {"probe": {"sha": self.SHA, "boot": "live|now", "attempts": 1, "last_at": time.time()}}
        slow = SimpleNamespace(dry_run=False, probe=True, controller=str(self.controller), probe_attempts=3, probe_gap=1800)
        self.assertFalse(self.ensure(held, a=slow))
        held["probe"]["last_at"] = time.time() - 3600
        self.assertTrue(self.ensure(held, a=slow))
        self.assertEqual(watch.state_of(self.state)["probe"]["attempts"], 2)

    def test_off_switches_and_an_unknown_sha_queue_nothing(self):
        from types import SimpleNamespace
        for a in (SimpleNamespace(dry_run=True, probe=True, controller=str(self.controller)),
                  SimpleNamespace(dry_run=False, probe=False, controller=str(self.controller))):
            self.assertFalse(self.ensure(a=a))
        self.assertFalse(self.ensure(sha=""))
        self.assertFalse(self.queued())

    def test_a_judge_that_cannot_be_asked_queues_nothing(self):
        (self.controller / "bench" / "st_judge.py").unlink()
        self.assertFalse(self.ensure())
        self.assertFalse(self.queued())
        self.assertIn("cannot tell whether", "\n".join(self.lines))

    def test_the_cycle_asks_at_its_start_candidate_or_not(self):
        """Tied to the nothing-to-deploy branch, the self-heal never ran while main kept moving: 05:12
        to 06:12 on 2026-09-13 the cycle chased a candidate through an hour of quiet-polling instead."""
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("def cycle("):source.index("    log(f\"candidate: {why}\")")]
        self.assertIn("ensure_probe(held.get(\"deployed\")", body)
        self.assertLess(body.index("ensure_probe("), body.index("wanted(head, held)"))
        self.assertEqual(body.count("ensure_probe("), 1)
        wait = source[source.index("while time.time() < deadline:"):source.index("never went quiet")]
        self.assertIn("fleet_busy_with_tickets()", wait, "the quiet wait yields to the queue instead of polling a door it does not own")
        for flag in ("--probe-attempts", "--probe-gap"):
            self.assertIn(flag, source)


class SameEngineTests(unittest.TestCase):
    """A main that moved without touching engine/ is deployed by bookkeeping, not by a boot."""

    def test_the_same_engine_tree_under_another_commit_is_no_deploy(self):
        from unittest import mock
        with mock.patch.object(watch, "engine_tree", side_effect=lambda sha: {"a" * 40: "t1", "b" * 40: "t1", "c" * 40: "t2"}.get(sha, "")):
            self.assertTrue(watch.same_engine("b" * 40, "a" * 40))
            self.assertFalse(watch.same_engine("c" * 40, "a" * 40))
            self.assertFalse(watch.same_engine("a" * 40, "a" * 40), "the same commit is 'nothing to deploy', not this")
            self.assertFalse(watch.same_engine("d" * 40, "a" * 40), "no tree, no claim")

    def test_the_cycle_records_it_before_waiting_for_quiet(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("def cycle("):source.index("    log(f\"  waiting for {a.quiet}s of quiet\")")]
        self.assertIn("if same_engine(head, held.get(\"deployed\") or \"\"):", body)
        self.assertIn("recorded as deployed without a boot", body)
        self.assertIn('"same_engine_as": held.get("deployed")', body)
        self.assertLess(body.index("same_engine("), body.index("since < a.min_gap") if "since < a.min_gap" in body else len(body))


class QueueGraceTests(unittest.TestCase):
    """A deploy right after a ticket ended takes the fleet from the next one (srv2, 2026-09-13)."""

    def setUp(self):
        import tempfile
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fleet = Path(self.temporary.name)

    def clock(self, ago):
        import json as _json
        import time
        (self.fleet / "idle-recovery.json").write_text(_json.dumps({"updated_at": time.time() - ago, "reason": "release"}))

    def test_a_queue_active_a_moment_ago_defers(self):
        self.clock(ago=20)
        self.assertIsNotNone(watch.queue_active_within(300, self.fleet))
        self.assertLess(watch.queue_active_within(300, self.fleet), 60)

    def test_a_quiet_queue_and_a_missing_clock_do_not(self):
        self.clock(ago=1000)
        self.assertIsNone(watch.queue_active_within(300, self.fleet))
        (self.fleet / "idle-recovery.json").unlink()
        self.assertIsNone(watch.queue_active_within(300, self.fleet))
        (self.fleet / "idle-recovery.json").write_text("not json")
        self.assertIsNone(watch.queue_active_within(300, self.fleet))

    def test_a_waiting_boot_ticket_goes_first(self):
        (self.fleet / "queue").write_text("3|kda-probe|100|5|note|probe|11\n4|kda-fp16-pair|100|60|note|boot|12\n")
        self.assertEqual(watch.boot_ticket_waiting(self.fleet), "kda-fp16-pair")
        (self.fleet / "queue").write_text("3|kda-probe|100|5|note|probe|11\n5|one-gpu|100|5|note|single|13\n")
        self.assertIsNone(watch.boot_ticket_waiting(self.fleet), "probes and single-GPU checks do not take the fleet")
        (self.fleet / "queue").write_text("")
        self.assertIsNone(watch.boot_ticket_waiting(self.fleet))
        (self.fleet / "queue").unlink()
        self.assertIsNone(watch.boot_ticket_waiting(self.fleet))

    def test_the_deploy_waits_as_long_as_production_would(self):
        import json as _json
        import time
        self.assertEqual(watch.pace_grace(self.fleet), 300, "no record: the floor")
        (self.fleet / "restore-grace.json").write_text(_json.dumps({"seconds": 900}))
        self.assertEqual(watch.pace_grace(self.fleet), 900)
        (self.fleet / "window.json").write_text(_json.dumps({"session": "s", "until": time.time() + 1200}))
        self.assertIn(watch.pace_grace(self.fleet), (1199, 1200))
        (self.fleet / "window.json").write_text(_json.dumps({"session": "s", "until": time.time() - 5}))
        self.assertEqual(watch.pace_grace(self.fleet), 900, "an expired window is no window")
        self.clock(ago=600)
        self.assertIsNotNone(watch.queue_active_within(None, self.fleet), "None means: the queue's pace (900 s here)")

    def test_the_cycle_asks_after_the_lease_and_before_the_deploy(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("def cycle("):source.index("def cycle(") + source[source.index("def cycle("):].index("\n\n\n")]
        self.assertLess(body.index("fleet_taken_by_another(log)"), body.index("queue_active_within(a.queue_grace)"))
        self.assertLess(body.index("queue_active_within(a.queue_grace)"), body.index("ok = deploy(release, log, profile)"))
        self.assertLess(body.index("boot_ticket_waiting()"), body.index("ok = deploy(release, log, profile)"))
        self.assertIn("--queue-grace", source)


if __name__ == "__main__":
    unittest.main()


class CachedBaselineTests(unittest.TestCase):
    """The gate ran the suite twice, and the second run was a value it already had.

    10m55s, 11m00s and 11m03s on 2026-09-16, and the deploy that followed each took 16 seconds. Half
    of that gate is the deployed commit's verdicts -- which this watcher measured itself, on the same
    commit, the same seed image and the same box, on the cycle that deployed it.
    """

    def tree(self, image="sha256:seed"):
        import json
        import tempfile
        root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, root, True)
        (root / "engine/runtime").mkdir(parents=True)
        (root / "engine/runtime/dependencies.json").write_text(json.dumps({"seed_image_id": image}))
        return root

    def held(self, **over):
        verdicts = {"test_engine_a": "OK", "test_engine_b": "FAILED (errors=1)"}
        base = {"deployed": "a" * 40,
                "gate": {"sha": "a" * 40, "image": "sha256:seed", "verdicts": verdicts, "at": 0}}
        return {**base, **over}

    def test_the_verdicts_are_reused_when_the_sha_and_the_image_still_hold(self):
        logs = []
        before, why = watch.cached_baseline(self.held(), self.tree(), logs.append)
        self.assertEqual(why, "cached")
        self.assertEqual(before, {"test_engine_a": "OK", "test_engine_b": "FAILED (errors=1)"})
        self.assertIn("not re-run", logs[0])

    def test_a_baseline_for_another_commit_is_not_this_one_s(self):
        before, why = watch.cached_baseline(self.held(deployed="b" * 40), self.tree(), print)
        self.assertIsNone(before)
        self.assertIn("aaaaaaaaaaaa", why)

    def test_a_new_seed_image_runs_it_again(self):
        """The tests run inside the image the tree pins, so a moved image is a different answer."""
        before, why = watch.cached_baseline(self.held(), self.tree(image="sha256:other"), print)
        self.assertIsNone(before)
        self.assertEqual(why, "the seed image moved")

    def test_nothing_recorded_and_a_half_written_record_both_run_it(self):
        for cache in (None, {}, {"sha": "a" * 40, "image": "sha256:seed"}, {"verdicts": "not a mapping"}):
            with self.subTest(cache=cache):
                before, why = watch.cached_baseline(self.held(gate=cache), self.tree(), print)
                self.assertIsNone(before)
                self.assertTrue(why)

    def test_an_unreadable_pin_runs_it_rather_than_trusting_the_cache(self):
        root = self.tree()
        (root / "engine/runtime/dependencies.json").unlink()
        before, why = watch.cached_baseline(self.held(), root, print)
        self.assertIsNone(before)
        self.assertIn("unreadable", why)

    def test_the_gate_asks_for_the_cache_before_it_extracts_the_other_commit(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text(encoding="utf-8")
        body = source[source.index("    if a.gate and deployed_tree is not None:"):source.index("    elif a.gate:")]
        self.assertLess(body.index("cached_baseline(held, deployed_tree, log)"),
                        body.index('gate_tree(held["deployed"], log)'))
        self.assertIn("failures(judged", body)          # the candidate is always measured, never cached

    def test_a_deploy_records_what_it_measured_and_a_failed_launch_drops_it(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text(encoding="utf-8")
        kept = source[source.index('state = {"deployed": head'):source.index("STATE.write_text(json.dumps(state")]
        self.assertIn('state["gate"] = {"sha": head, "image": gate_image(judged), "verdicts": after', kept)
        dropped = source[source.index('"rejected_by": "launch"') - 400:source.index('"rejected_by": "launch"')]
        self.assertIn('if k != "gate"', dropped)


class SpeedClaimTests(unittest.TestCase):
    """A D17 probe is queued for a commit that CLAIMS speed, not for every deploy.

    The probe reserves the door for a whole C1/C2/32K bracket and answers 409 to every other request
    while it runs -- on 2026-09-16 that reached a user as `API error 409` from the assistant. Paying
    that after every deploy buys a baseline for changes that never claimed to move one.
    """

    NL = chr(10)

    def test_a_perf_type_claims_speed(self):
        for subject in ("perf: 무언가", "perf(loader): 로드가", "PERF(boot): x", "perf(engine): y"):
            with self.subTest(subject=subject):
                claims, why = watch.claims_speed(subject + self.NL + self.NL + "body")
                self.assertTrue(claims, why)
                self.assertIn("type", why)

    def test_every_other_type_does_not(self):
        for subject in ("fix: x", "feat: x", "docs: x", "ci: x", "test: x", "revert: x",
                        "measure(boot): x", "chore: x", "no conventional type at all"):
            with self.subTest(subject=subject):
                claims, why = watch.claims_speed(subject)
                self.assertFalse(claims, why)
                self.assertIn("claims no speed", why)

    def test_a_change_that_knows_better_than_its_type_says_so(self):
        """A `fix:` that moves the step asks for the probe; a `perf:` already bracketed declines it."""
        claims, why = watch.claims_speed("fix: the router did it twice" + self.NL * 2 + "D17-probe: yes" + self.NL)
        self.assertTrue(claims)
        self.assertIn("D17-probe", why)
        claims, why = watch.claims_speed("perf(engine): x" + self.NL * 2 + "D17-probe: no" + self.NL)
        self.assertFalse(claims)
        self.assertIn("D17-probe", why)

    def test_an_empty_message_claims_nothing(self):
        self.assertFalse(watch.claims_speed("")[0])
        self.assertFalse(watch.claims_speed("   " + self.NL + self.NL + "  ")[0])

    def test_the_gate_runs_before_anything_that_costs(self):
        """The first thing `ensure_probe` asks after the flags: no controller, no judge, no ticket."""
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text(encoding="utf-8")
        head = source.index("def ensure_probe(")
        body = source[head:head + 3000]
        self.assertLess(body.index("probe_wanted(sha, log)"), body.index("sample_boots("))
        after = source[source.index("def after_deploy("):source.index("def main(")]
        # both callers or neither: after_deploy queues its own ticket without going through ensure_probe
        self.assertIn("probe_wanted(head, log)", after)
        self.assertIn('log(f"  no D17 probe for', source)


class PrebuildTests(unittest.TestCase):
    """The release's b12x kernels are compiled before the deploy stops production, and nothing about it can stop a
    deploy: a prebuild that fails leaves the boot to compile, as before."""

    def release(self, root, script=True):
        release = Path(root) / "st-releases" / "abc123"
        (release / "launchers").mkdir(parents=True)
        if script:
            (release / "launchers" / "b12x-prebuild.sh").write_text("exit 0\n")
        return release

    def test_it_runs_the_releases_own_script_for_production(self):
        import tempfile
        lines = []
        with tempfile.TemporaryDirectory() as root:
            release = self.release(root)
            with patch.object(watch, "run", return_value=(0, "10.10.10.2: {\"summary\": {}}\n", "")) as run:
                watch.prebuild(release, lines.append)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd, ["bash", str(release / "launchers/b12x-prebuild.sh"), "--tree", str(release),
                               "--profile", "glm53"])
        self.assertTrue(any("summary" in line for line in lines))

    def test_a_prebuild_that_times_out_or_is_missing_is_said_and_passed(self):
        import subprocess
        import tempfile
        lines = []
        with tempfile.TemporaryDirectory() as root:
            with patch.object(watch, "run", side_effect=subprocess.TimeoutExpired("bash", 1800)):
                self.assertIsNone(watch.prebuild(self.release(root), lines.append))
            with tempfile.TemporaryDirectory() as other, patch.object(watch, "run") as run:
                watch.prebuild(self.release(other, script=False), lines.append)
                run.assert_not_called()
        self.assertTrue(any("TimeoutExpired" in line for line in lines))
        self.assertTrue(any("no launchers/b12x-prebuild.sh" in line for line in lines))

    def test_the_cycle_prebuilds_after_the_deferrals_and_before_production_stops(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("def cycle("):source.index("def follow_controller(")]
        pre = body.index("prebuild(release, log, profile)")
        self.assertGreater(pre, body.index("boot_ticket_waiting()"))
        self.assertLess(pre, body.index("ok = deploy(release, log, profile)"))
        self.assertIn('getattr(a, "prebuild", True)', body)


class ProductionModelDeployTests(unittest.TestCase):
    """A deploy boots the model production serves (launchers/st_production.py), and waits while it is between two:
    booting the selected model over the other one's containers would be refused and recorded as a failed launch,
    which drops the gate's baseline until a person --seeds it."""

    def setUp(self):
        import os
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        patcher = patch.dict(os.environ, {"ST_PRODUCTION_FILE": str(self.dir / "st-production.json"),
                                          "ST_PRODUCTION_STATE": str(self.dir / "st-production-state.json"),
                                          "ST_PROFILE_CONFIG_DIR": str(self.dir / "config")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_box_that_never_chose_deploys_as_it_always_did(self):
        self.assertIsNone(watch.production_switching(), "no selection and no state: glm53, as before the selection")

    def test_a_deploy_waits_while_production_is_between_models(self):
        import json
        st = watch.st_production
        st.select("qwen38", by="deneb")
        self.assertIn("no supervisor says it serves it", watch.production_switching())
        st.publish_state("glm53", "qwen38", "switching", "glm53 -> qwen38")
        self.assertIn("qwen38 is selected, glm53 serves (switching)", watch.production_switching())
        st.publish_state("qwen38", "qwen38", "serving")
        self.assertIsNone(watch.production_switching())
        self.assertEqual(json.loads((self.dir / "st-production-state.json").read_text())["serving"], "qwen38")

    def test_the_prebuild_replays_the_served_model_s_kernels(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            release = Path(root) / "st-releases" / "abc123"
            (release / "launchers").mkdir(parents=True)
            (release / "launchers" / "b12x-prebuild.sh").write_text("exit 0\n")
            with patch.object(watch, "run", return_value=(0, "", "")) as run:
                watch.prebuild(release, [].append, "qwen38")
        self.assertEqual(run.call_args.args[0][-2:], ["--profile", "qwen38"])

    def test_the_cycle_reads_the_model_once_and_waits_before_it_prebuilds(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("def cycle("):source.index("def follow_controller(")]
        self.assertEqual(body.count("st_production.selected()"), 1, "one reading per cycle: prebuild and boot agree")
        switching = body.index("production_switching()")
        self.assertGreater(switching, body.index("boot_ticket_waiting()"))
        self.assertLess(switching, body.index("prebuild(release, log, profile)"))
        self.assertIn("# deferred, not rejected", body[switching:body.index("prebuild(release, log, profile)")])
        self.assertIn("if profile == st_production.DEFAULT:\n        ensure_probe(", body,
                      "a D17 sample is GLM-5.3's series: none is taken while production serves another model")
