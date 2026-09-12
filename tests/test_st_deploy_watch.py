"""The deploy watcher's three refusals (45차 §34 뒤).

Everything here is the part that DECIDES. The part that acts -- cutting a release, stopping the
supervisor, launching four nodes -- is not exercised: it can only be judged by doing it, and doing
it takes production down. What is judged is every way this says no, because a deploy watcher is
worth exactly what its refusals are worth.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launchers"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "st_deploy_watch", Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class WantedTests(unittest.TestCase):
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
        body = source[source.index("    ok = deploy(release, log)"):]
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

    def test_the_idle_controller_runs_from_the_checkout_the_hook_moves(self):
        """fleet-idle-recovery restores production by the queue's rules: those must be the deployed
        commit's, which is the checkout follow_controller keeps, not whatever ~/stkernel is on."""
        unit = (Path(__file__).resolve().parents[1] / "launchers/fleet-idle-recovery.service").read_text()
        self.assertIn("WorkingDirectory=/home/choiceoh/fleet-controller", unit)
        self.assertIn("ExecStart=/usr/bin/python3 /home/choiceoh/fleet-controller/bench/fleet_idle.py tick", unit)
        self.assertNotIn("stkernel/bench/fleet_idle.py", unit)
        self.assertTrue(str(watch.CONTROLLER).endswith("fleet-controller") or "FLEET_CONTROLLER_REPO" in __import__("os").environ)

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

    def test_it_runs_only_after_a_deploy_that_is_recorded(self):
        """A failed launch is a rejection; nothing follows it, and no probe samples a fleet in recovery."""
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("    ok = deploy(release, log)"):]
        body = body[:body.index("\n\ndef ", 10)]
        failed = body[body.index("if not ok:"):body.index("return 1")]
        self.assertNotIn("after_deploy", failed)
        recorded = body[body.index('"launched_ok": True'):]
        self.assertIn("after_deploy(head, a, log)", recorded)
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

    def ensure(self, held=None, sha=None, a=None):
        return watch.ensure_probe(self.SHA if sha is None else sha, held or {}, a or self.a, self.log,
                                  controller=self.controller, fleet_dir=self.fleet, jsonl=self.jsonl, state=self.state)

    def sample(self, sha=None, run_index=2):
        import json as _json
        row = {"engine": "st", "arm_sha": sha or self.SHA, "run_index": run_index, "boot_id": "b|1", "quality": {"ok": 9, "total": 9},
               "korean": {"dirty": 0, "n": 5}, "decode": {"windows_med": 12.0}, "traffic": {"issues": []}}
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
        self.assertIn("no warm sample and no probe ticket on its way (attempt 1/3)", "\n".join(self.lines))

    def test_a_warm_sample_is_enough(self):
        self.sample()
        self.assertFalse(self.ensure())
        self.assertFalse(self.queued())
        self.jsonl.unlink()
        self.sample(run_index=1)                                     # a cold run is not the sample
        self.assertTrue(self.ensure())

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
        held = {"probe": {"sha": self.SHA, "attempts": 1, "last_at": 0}}
        self.assertTrue(self.ensure(held))
        self.assertEqual((self.tmp / "argv").read_text().splitlines()[2], "d17-0123abcdef01-2")
        self.assertEqual(watch.state_of(self.state)["probe"]["attempts"], 2)

    def test_the_tally_bounds_it(self):
        held = {"probe": {"sha": self.SHA, "attempts": 3, "last_at": 0}}
        self.assertFalse(self.ensure(held))
        self.assertFalse(self.queued())
        self.assertIn("queuing no more", "\n".join(self.lines))
        self.assertTrue(watch.state_of(self.state)["probe"]["gave_up"])
        n = len(self.lines)
        self.assertFalse(self.ensure(watch.state_of(self.state)))
        self.assertEqual(len(self.lines), n, "said once")
        self.assertTrue(self.ensure({"probe": {"sha": "f" * 40, "attempts": 3, "last_at": 0}}), "another sha starts afresh")

    def test_the_gap_between_two_tickets_is_kept(self):
        import time
        from types import SimpleNamespace
        held = {"probe": {"sha": self.SHA, "attempts": 1, "last_at": time.time()}}
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

    def test_the_cycle_asks_only_when_there_is_nothing_to_deploy(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("def cycle("):source.index("    log(f\"candidate: {why}\")")]
        self.assertIn("ensure_probe(held.get(\"deployed\")", body)
        for flag in ("--probe-attempts", "--probe-gap"):
            self.assertIn(flag, source)


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

    def test_the_cycle_asks_after_the_lease_and_before_the_deploy(self):
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("def cycle("):source.index("def cycle(") + source[source.index("def cycle("):].index("\n\n\n")]
        self.assertLess(body.index("fleet_taken_by_another(log)"), body.index("queue_active_within(a.queue_grace)"))
        self.assertLess(body.index("queue_active_within(a.queue_grace)"), body.index("ok = deploy(release, log)"))
        self.assertLess(body.index("boot_ticket_waiting()"), body.index("ok = deploy(release, log)"))
        self.assertIn("--queue-grace", source)


if __name__ == "__main__":
    unittest.main()
