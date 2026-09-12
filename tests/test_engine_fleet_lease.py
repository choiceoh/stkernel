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
