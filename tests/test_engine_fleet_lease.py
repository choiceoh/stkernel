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
        # piped, so taking the lease never rsyncs over a live session's engine tree
        self.assertIn('python3 - $* --path $LOCK" < "$REPO/engine/base/fleet_lease.py"', self.launcher)

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


class ProbeLeaseTests(unittest.TestCase):
    def test_the_probe_runner_takes_and_releases_the_lease(self):
        """A probe takes the same GPUs as a boot. Until now it was a bare `docker run`
        that no launcher and no queue could see."""
        runner = (ROOT / "probes/run_engine_probe.sh").read_text()
        self.assertIn("fleet_lease.py\" acquire --owner", runner)
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
