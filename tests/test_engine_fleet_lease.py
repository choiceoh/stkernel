"""The engine's own fleet reservation: one owner, evidence before heartbeat.

The reservation primitive is the engine's because the engine is what takes the
fleet. The campaign queue (aging, priority, pause/resume, handoff, evidence,
runner pinning) stays in bench/fleet.sh: it is a solved problem, and a second
copy here would repeat the mistake this module exists to end.
"""
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.base.fleet_lease import (                                    # noqa: E402
    GRACE_S, LeaseHeld, LeaseLost, acquire, alive, describe, read, release, renew)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "lease"

    def test_one_owner_at_a_time(self):
        self.assertIsNone(read(self.path))
        held = acquire("boot-a", path=self.path, container="st-glm53", note="45 layers", est_minutes=30)
        self.assertEqual(read(self.path)["owner"], "boot-a")
        self.assertIn("boot-a", describe(held))
        with self.assertRaisesRegex(LeaseHeld, "boot-a"):
            acquire("boot-b", path=self.path, container_up=lambda name: True)

    def test_evidence_comes_before_the_heartbeat(self):
        acquire("boot-a", path=self.path, container="st-glm53")
        record = dict(read(self.path), pid=0, beat=time.time() - 10 * GRACE_S)
        self.assertTrue(alive(record, container_up=lambda name: True), "its container is up: held")
        self.assertTrue(alive(record, container_up=lambda name: None), "unreachable is not free")
        self.assertFalse(alive(record, container_up=lambda name: False), "no container, no beat: stale")

    def test_a_fresh_heartbeat_holds_a_boot_that_has_no_container_yet(self):
        acquire("boot-a", path=self.path, container="st-glm53")
        record = dict(read(self.path), pid=0)
        self.assertTrue(alive(record, container_up=lambda name: False))   # between acquire and the containers
        renewed = renew("boot-a", path=self.path)
        self.assertGreaterEqual(renewed["beat"], renewed["since"])

    def test_a_stale_lease_is_reclaimed_once_and_recorded_to_the_new_owner(self):
        acquire("boot-a", path=self.path, container="st-glm53")
        stale = dict(read(self.path), pid=0, beat=time.time() - 2 * GRACE_S, since=time.time() - 2 * GRACE_S)
        self.path.write_text(json.dumps(stale) + "\n")
        taken = acquire("boot-b", path=self.path, container_up=lambda name: False)
        self.assertEqual((taken["owner"], read(self.path)["owner"]), ("boot-b", "boot-b"))

    def test_only_the_owner_releases_or_renews(self):
        acquire("boot-a", path=self.path)
        for call in (lambda: release("boot-b", path=self.path), lambda: renew("boot-b", path=self.path)):
            with self.assertRaises(LeaseLost):
                call()
        self.assertEqual(release("boot-a", path=self.path)["owner"], "boot-a")
        self.assertIsNone(read(self.path))
        self.assertIsNone(release("boot-a", path=self.path))              # releasing nothing is not an error
        acquire("boot-a", path=self.path)
        self.assertEqual(release("anyone", path=self.path, force=True)["owner"], "boot-a")

    def test_an_older_plain_text_lock_is_a_lease_we_honour(self):
        """The launcher wrote `echo user@host ... > lock` until 2026-09-12. That record
        carries no evidence and no heartbeat, so nothing may ever call it stale."""
        self.path.write_text("choiceoh@srv2 st-glm53 2026-09-12 10:00:00\n")
        opaque = read(self.path)
        self.assertTrue(opaque["opaque"] and opaque["owner"].startswith("choiceoh@srv2"))
        self.assertTrue(alive(opaque, container_up=lambda name: False))
        with self.assertRaises(LeaseHeld):
            acquire("boot-b", path=self.path, container_up=lambda name: False)

    def test_a_lease_owner_is_one_line(self):
        for bad in ("", "two\nlines"):
            with self.assertRaises(ValueError):
                acquire(bad, path=self.path)


class LauncherAndQueueTests(unittest.TestCase):
    """Both sides of the reservation, pinned as text: the launcher takes the engine's lease,
    and the queue refuses to answer about a queue that is not the controller's."""

    def setUp(self):
        self.launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()
        self.fleet = (ROOT / "bench/fleet.sh").read_text()

    def test_the_launcher_takes_the_engine_lease(self):
        self.assertIn("engine/base/fleet_lease.py", self.launcher)
        self.assertIn("lease acquire --owner", self.launcher)
        self.assertNotIn("echo '$(whoami)@$(hostname) st-glm53", self.launcher)
        # through the one shared helper, which pipes the module to the head node: taking
        # the lease never rsyncs over a live session's engine tree
        self.assertIn("launchers/lib/fleet-lease.sh", self.launcher)
        helper = (ROOT / "launchers/lib/fleet-lease.sh").read_text()
        self.assertIn('ssh $FLEET_LEASE_SSH "choiceoh@$FLEET_HEAD" "python3 - $* --path $FLEET_LEASE_PATH" < "$module"', helper)

    def test_the_queue_refuses_to_answer_off_the_controller(self):
        """Homes are not shared: elsewhere it would create a second, empty queue."""
        self.assertIn('FLEET_CONTROLLER=${FLEET_CONTROLLER:-srv2}', self.fleet)
        self.assertIn("require_controller", self.fleet)
        self.assertIn("classify|preflight|version|nodes|busy) ;;", self.fleet)

    def test_the_runner_snapshot_pins_what_it_admits(self):
        """An entry admitted at queue time but absent from the pinned runner passes
        admission and then stalls before it runs (2026-09-12: three ST reservations)."""
        sys.path.insert(0, str(ROOT / "bench"))
        import fleet_onepass
        import fleet_pin
        pinned = set(fleet_pin.source_files(ROOT))
        for relative in (*fleet_onepass.ST_ENTRIES, *fleet_onepass.ST_PROBES):
            if (ROOT / relative).is_file():
                self.assertIn(relative, pinned, relative)


if __name__ == "__main__":
    unittest.main()


class YieldProtocolTests(unittest.TestCase):
    """Asking a running engine to hand the fleet over, rather than waiting for it or
    killing it. The queue can order the line; only the engine can finish what it is
    holding and park it where the next holder finds it (D16)."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "lease"
        acquire("holder", path=self.path, container="st-glm53")

    def test_the_holder_publishes_what_it_is_doing(self):
        from engine.base.fleet_lease import publish, yield_requested
        publish("holder", path=self.path, phase="serving", running=3, free_in_minutes=12)
        line = describe(read(self.path))
        for fragment in ("phase=serving", "running=3", "free_in_minutes=12"):
            self.assertIn(fragment, line)
        self.assertIsNone(yield_requested(read(self.path)))

    def test_asking_and_publishing_do_not_drop_each_other(self):
        """Both sides write the same record: the holder must not erase the request, and
        the request must not erase what the holder said."""
        from engine.base.fleet_lease import publish, request_yield, yield_requested
        publish("holder", path=self.path, phase="serving", running=3)
        request_yield("waiter", path=self.path, reason="graph replay check")
        publish("holder", path=self.path, running=0)
        record = read(self.path)
        self.assertEqual(yield_requested(record)["requester"], "waiter")
        self.assertEqual(record["state"]["phase"], "serving")
        self.assertEqual(record["state"]["running"], 0)
        self.assertIn("asked to yield to waiter", describe(record))

    def test_asking_is_not_taking(self):
        from engine.base.fleet_lease import request_yield
        request_yield("waiter", path=self.path)
        with self.assertRaisesRegex(LeaseHeld, "holder"):
            acquire("waiter", path=self.path, container_up=lambda name: True)
        self.assertEqual(read(self.path)["owner"], "holder")

    def test_a_holder_that_predates_the_protocol_cannot_be_asked(self):
        from engine.base.fleet_lease import request_yield
        self.path.write_text("choiceoh@srv2 st-glm53 2026-09-12 10:00:00\n")
        with self.assertRaisesRegex(LeaseHeld, "predates"):
            request_yield("waiter", path=self.path)

    def test_a_free_fleet_has_nobody_to_ask(self):
        from engine.base.fleet_lease import request_yield, release as rel
        rel("holder", path=self.path)
        self.assertIsNone(request_yield("waiter", path=self.path))

    def test_the_holder_clears_the_request_when_it_has_let_go(self):
        from engine.base.fleet_lease import clear_yield, request_yield, yield_requested
        request_yield("waiter", path=self.path)
        clear_yield("someone-else", path=self.path)
        self.assertIsNotNone(yield_requested(read(self.path)))      # not theirs to clear
        clear_yield("holder", path=self.path)
        self.assertIsNone(yield_requested(read(self.path)))


class EngineHandoverTests(unittest.TestCase):
    """The engine side of the handover, pinned as text and as behaviour."""

    def setUp(self):
        self.serve = (ROOT / "engine/base/serve.py").read_text()

    def test_the_step_loop_asks_parks_and_lets_go(self):
        self.assertIn("def _yield_asked(self)", self.serve)
        self.assertIn("def _quiet(self)", self.serve)
        self.assertIn("def _hand_over(self)", self.serve)
        # the decision travels with the step's other decisions, so every rank drains together
        self.assertIn("self._yield_asked()) if self.comm.rank == 0 else None)", self.serve)
        self.assertIn("self.draining = draining", self.serve)
        # and it is visible to anyone scraping, not only to the lease
        self.assertIn('"st:handing_over"', self.serve)

    def test_a_draining_door_refuses_new_work_with_503(self):
        self.assertIn("the engine is handing the fleet over; retry shortly\", 503", self.serve)

    def test_the_handover_parks_under_the_conversation_key(self):
        """That key is what the next holder resumes by, so a turn survives the change."""
        self.assertIn("self.runner.park(row, key=conversation)", self.serve)

    def test_a_draining_engine_refuses_admission_and_ends_its_loop(self):
        import test_engine_serve as T
        server = T.server()
        server.submit([1, 2, 3], max_new=2, temperature=0.0)
        server.draining = "waiter"
        from engine.base.serve import RequestError
        with self.assertRaises(RequestError):
            server.submit([4, 5], max_new=1, temperature=0.0)
        for _ in range(80):                                    # what is already in finishes
            server.once()
            if not server.alive:
                break
        self.assertTrue(server.drained and not server.alive, "a quiet draining engine lets go")

    def test_an_engine_without_a_lease_serves_exactly_as_before(self):
        import test_engine_serve as T
        server = T.server()
        self.assertIsNone(server.lease)
        self.assertIsNone(server._yield_asked())
        server.submit([1, 2, 3], max_new=2, temperature=0.0)
        for _ in range(80):
            if not server.once() and not server._waiting:
                break
        self.assertIsNone(server.draining)
        self.assertTrue(server.alive)


class SmoothnessTests(unittest.TestCase):
    """The gaps an audit of §28 found: a feature nobody can reach easily is not done."""

    def setUp(self):
        self.launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()
        self.probe = (ROOT / "probes/run_engine_probe.sh").read_text()
        self.fleet = (ROOT / "bench/fleet.sh").read_text()
        self.helper = (ROOT / "launchers/lib/fleet-lease.sh").read_text()

    def test_every_caller_uses_one_lease_file_on_the_head_node(self):
        """The probe took $HOME/st-fleet.lock on whatever node it ran on while a boot took
        the head node's: homes are not shared, so the two never met."""
        self.assertIn("FLEET_HEAD=${FLEET_HEAD:-10.10.10.2}", self.helper)
        for source in (self.launcher, self.probe):
            self.assertIn("launchers/lib/fleet-lease.sh", source)
        self.assertNotIn("$HOME/st-fleet.lock", self.probe)

    def test_a_long_hold_keeps_its_lease_fresh(self):
        """The head node's docker cannot see a probe container on another node, so without
        a heartbeat the lease would go stale under a probe that is still running."""
        self.assertIn("fleet_lease_beat", self.helper)
        self.assertIn("BEAT=$(fleet_lease_beat", self.probe)
        self.assertIn("kill $BEAT", self.probe)

    def test_a_cpu_only_probe_reserves_nothing(self):
        """Taking four Sparks for an import check is the opposite of smooth."""
        self.assertIn('[ "${ST_PROBE_NO_LEASE:-0}" != 1 ] && [ "${ST_PROBE_NO_GPU:-0}" != 1 ]', self.probe)

    def test_a_killed_probe_leaks_a_lease_that_goes_stale_on_its_own(self):
        """SIGKILL runs no trap. The lease must not need a human: its evidence is on another
        node, so it ages out on the grace and `read` says so meanwhile."""
        self.assertIn("goes stale after the grace", self.probe)
        module = (ROOT / "engine/base/fleet_lease.py").read_text()
        self.assertIn('print("free (stale: " + describe(held) + ")")', module)

    def test_an_opaque_lock_naming_a_dead_pid_is_not_a_dead_hand(self):
        """Treating every unparseable record as permanently held made a departed session
        block the fleet until a human ran `stop` -- three queued reservations died on it."""
        import os
        here = os.uname().nodename.split(".")[0]
        path = Path(self.probe)                                  # any path; we write our own below
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lease"
            path.write_text(f"choiceoh@{here} st-glm53 2026-09-12 02:55:58 UTC pid=999999999\n")
            record = read(path)
            self.assertTrue(record["opaque"] and record["pid"] == 999999999)
            self.assertEqual(record["host"], here)
            # its pid is gone and nothing is running: reclaimable
            self.assertFalse(alive(record, container_up=lambda name: False))
            # but not while an ST container is up, and not if we cannot ask
            self.assertTrue(alive(record, container_up=lambda name: True))
            self.assertTrue(alive(record, container_up=None))
            # a live pid, or another host, stays held
            path.write_text(f"choiceoh@{here} st-glm53 pid={os.getpid()}\n")
            self.assertTrue(alive(read(path), container_up=lambda name: False))
            path.write_text("choiceoh@some-other-node st-glm53 pid=999999999\n")
            self.assertTrue(alive(read(path), container_up=lambda name: False))
            path.write_text("no pid here at all\n")
            self.assertTrue(alive(read(path), container_up=lambda name: False))

    def test_the_probe_waits_for_the_fleet_instead_of_losing_its_turn(self):
        self.assertIn("ST_PROBE_WAIT_MINUTES", self.probe)
        self.assertIn("waiting for the fleet", self.probe)
        self.assertIn("the fleet stayed held for", self.probe)

    def test_the_queue_reads_the_lease_not_only_containers(self):
        """They disagreed once and the queue granted while the lease was still held."""
        self.assertIn("fleet_lease read", self.fleet)
        self.assertIn("st_engine_up() {", self.fleet)

    def test_the_lease_lives_where_a_container_can_read_it(self):
        """The engine must READ its lease to notice a yield request. ~/st-fleet.lock is not
        mounted into any ST container, so the whole handover was inert on the fleet."""
        from engine.base.fleet_lease import DEFAULT_PATH
        self.assertEqual(str(DEFAULT_PATH), "/home/choiceoh/glm53-logs/st-fleet.lock")
        mounted = "-v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs"
        self.assertIn(mounted, self.launcher)                       # same path inside and out
        self.assertIn("FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock", self.helper)
        self.assertIn("LOCK=${FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock}", self.launcher)

    def test_a_session_on_the_older_lock_path_still_blocks_us(self):
        """Moving the path must not split the lock: an older launcher writes the old one."""
        self.assertIn("LEGACY_LOCK=/home/choiceoh/st-fleet.lock", self.launcher)
        self.assertIn("a session on the older lock path holds the fleet", self.launcher)
        self.assertIn('node_sh "${NODES[0]}" "rm -f $LEGACY_LOCK"', self.launcher)   # stop clears both

    def test_the_head_node_runs_the_lease_locally(self):
        """A node cannot ssh to itself here, and the head is where the queue's controller
        runs -- so the most important caller was the one that could not ask (exit 255)."""
        self.assertIn("_fleet_lease_is_head()", self.helper)
        self.assertIn('if _fleet_lease_is_head; then\n    python3 "$module"', self.helper)

    def test_release_takes_the_same_lock_as_publish(self):
        """Without it a holder's next heartbeat, landing between the read and the unlink,
        rewrites the file and the released lease comes back."""
        module = (ROOT / "engine/base/fleet_lease.py").read_text()
        body = module[module.index("def release("):module.index("def _selfcheck(")]
        self.assertIn("with _Mutation(path):", body)

    def test_a_heartbeat_cannot_resurrect_a_released_lease(self):
        import tempfile
        from engine.base.fleet_lease import publish, release as rel
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lease"
            acquire("holder", path=path, container="st-glm53")
            rel("holder", path=path)
            with self.assertRaises(LeaseLost):
                publish("holder", path=path, running=1)
            self.assertIsNone(read(path))

    def test_yield_waits_for_the_handover(self):
        """Asking and leaving the caller to poll is not a handover."""
        self.assertIn("YIELD_WAIT_MINUTES", self.launcher)
        self.assertIn("the fleet is free: start when ready", self.launcher)
        self.assertIn("did not let go within", self.launcher)

    def test_the_queue_asks_instead_of_only_refusing(self):
        """Otherwise a queued session waits for a human to go and ask."""
        self.assertIn("st_engine_yield()", self.fleet)
        self.assertIn('st_engine_yield "$s"', self.fleet)
        self.assertIn("asking it to yield to", self.fleet)

    def test_a_sourced_helper_keeps_its_defaults(self):
        """Assignment prefixes on `.` are temporary in bash: the helper's own defaults are
        discarded when the builtin returns, and the next call sees an unbound variable."""
        self.assertIn("use_lease() {", self.launcher)
        self.assertNotIn("FLEET_REPO=$REPO FLEET_HEAD=", self.launcher)

    def test_the_launcher_defines_what_its_subcommands_use(self):
        lock = self.launcher.index("LOCK=/home/choiceoh/st-fleet.lock")
        for token in ("use_lease() {", 'case "${1:-start}" in'):
            self.assertLess(lock, self.launcher.index(token), token)
        self.assertEqual(self.launcher.count("LOCK=/home/choiceoh/st-fleet.lock"), 1)

    def test_a_handover_counts_what_it_saved_and_what_it_lost(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        self.assertIn("self.handed_over = {\"to\": self.draining", serve)
        self.assertIn("conversations parked, {lost} LOST", serve)
        self.assertIn('"st:handover_conversations_lost"', serve)

    def test_the_handover_reports_through_the_lease_before_letting_go(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        self.assertIn('phase="handed over", parked=parked, lost=lost', serve)


class ProbeLeaseTests(unittest.TestCase):
    def test_the_probe_runner_takes_and_releases_the_lease(self):
        """A probe takes the same GPUs as a boot. Until now it was a bare `docker run`
        that no launcher and no queue could see."""
        runner = (ROOT / "probes/run_engine_probe.sh").read_text()
        self.assertIn("fleet_lease acquire --owner", runner)
        self.assertIn("trap ", runner)
        self.assertIn("release --owner", runner)
        self.assertIn('docker run --rm --name "$NAME"', runner)      # named: the lease's evidence
        self.assertIn("ST_PROBE_NO_LEASE", runner)                   # an explicit way out, for a nested run

    def test_the_launcher_can_ask_and_can_report(self):
        launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()
        self.assertIn("  yield)", launcher)
        self.assertIn("  held)", launcher)
        self.assertIn("ST_LEASE_OWNER", launcher)
        self.assertIn("usage: $0 [start|stop|yield [reason]|held|logs r]", launcher)

    def test_the_boot_hands_the_lease_to_the_engine(self):
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn("def fleet_lease_of()", boot)
        self.assertIn("lease=fleet_lease_of()).loop()", boot)


class QueueMaintenanceTests(unittest.TestCase):
    """The pre-existing queue's rough edges (§31), fixed."""

    def setUp(self):
        self.fleet = (ROOT / "bench/fleet.sh").read_text()

    def test_the_audit_pins_are_current(self):
        """One stale hash turns off CPU reuse and contract narrowing for everyone, and the
        only signal used to be four unit tests everybody called 'pre-existing'."""
        import hashlib
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "bench"))
        import cpu_contracts, cpu_evidence
        sha = lambda rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        self.assertEqual(sha("tests/test_logic.py"), cpu_contracts.LOGIC_AUDIT)
        for name in ("LOGIC_SOURCE_AUDIT", "FLEET_AUDIT", "STARTUP_AUDIT"):
            for rel, want in (getattr(cpu_evidence, name, {}) or {}).items():
                with self.subTest(audit=name, file=rel):
                    self.assertEqual(sha(rel), want)

    def test_a_stale_pin_is_reported_where_people_look(self):
        self.assertIn("audit_line()", self.fleet)
        self.assertIn("audit: STALE", self.fleet)
        # defined before the dispatch that calls it
        self.assertLess(self.fleet.index("audit_line() {"), self.fleet.index('case "$cmd" in'))

    def test_a_remote_holder_is_judged_by_evidence(self):
        """Blind trust for 3x the estimate meant a crashed holder blocked the fleet for two
        hours at est 40, and the recovery the header promises is not installed."""
        self.assertIn("holder_probe()", self.fleet)
        self.assertIn("case \"$answer\" in alive) return 0 ;; gone) return 1 ;; esac", self.fleet)
        # /proc, because kill -0 answers "gone" for a process you do not own
        self.assertIn("[ -d /proc/$pid ] && echo alive || echo gone", self.fleet)
        self.assertNotIn("kill -0 $pid 2>/dev/null && echo alive", self.fleet)
        self.assertIn("HOLDER_PROBE_TTL_S", self.fleet)          # not an ssh per poll

    def test_the_queue_can_prune_its_own_debris(self):
        self.assertIn("  prune)", self.fleet)
        import importlib.util
        spec = importlib.util.spec_from_file_location("fleet_prune", ROOT / "bench/fleet_prune.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._selfcheck()
