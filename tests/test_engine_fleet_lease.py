"""The engine's own fleet reservation: one owner, evidence before heartbeat.

The reservation primitive is the engine's because the engine is what takes the
fleet. The campaign queue (aging, priority, pause/resume, handoff, evidence,
runner pinning) stays in bench/fleet.sh: it is a solved problem, and a second
copy here would repeat the mistake this module exists to end.
"""
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

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

    def test_every_lease_has_a_kind_and_a_dead_queue_holder_is_free(self):
        """The queue's supervisor on the head IS the holder of a `queue` lease: gone means free,
        without the grace a launcher's exit needs (its containers outlive it)."""
        from engine.base.fleet_lease import KINDS
        acquire("boot-a", path=self.path)
        self.assertEqual(read(self.path)["kind"], "session")
        release("boot-a", path=self.path)
        with self.assertRaises(ValueError):
            acquire("x", path=self.path, kind="nonsense")
        acquire("queue/t1", path=self.path, kind="queue", pid=os.getpid())
        self.assertTrue(alive(read(self.path), container_up=lambda name: False))
        dead = dict(read(self.path), pid=999999999)
        self.assertFalse(alive(dead, container_up=lambda name: False), "no grace for a dead supervisor")
        self.assertTrue(alive(dict(dead, kind="session"), container_up=lambda name: False),
                        "a session's boot keeps its grace: its launcher exits while its containers serve")
        self.assertEqual(KINDS, ("session", "production", "queue", "probe"))

    def test_a_handover_is_a_transfer(self):
        """Released, the fleet reads as free for a moment and the production supervisor relaunches
        into the window somebody was granted; transferred, it is the requester's in one step."""
        from engine.base.fleet_lease import request_yield, transfer, verify, yield_requested
        acquire("holder", path=self.path, container="st-glm53")
        request_yield("queue/t1", path=self.path, reason="a ticket", kind="queue", pid=os.getpid(), est_minutes=20)
        asked = yield_requested(read(self.path))
        with self.assertRaises(LeaseLost):
            transfer("somebody-else", asked["requester"], path=self.path)
        transfer("holder", asked["requester"], path=self.path, kind=asked["kind"], pid=asked["pid"],
                 host=asked["host"], est_minutes=asked["est_minutes"])
        record = read(self.path)
        self.assertEqual((record["owner"], record["kind"], record["pid"], record["handed_from"]),
                         ("queue/t1", "queue", os.getpid(), "holder"))
        self.assertIsNone(yield_requested(record))
        self.assertEqual(verify("queue/t1", path=self.path)["owner"], "queue/t1")
        with self.assertRaises(LeaseLost):
            verify("holder", path=self.path)

    def test_taken_is_judged_by_kind_and_liveness(self):
        from engine.base.fleet_lease import taken
        self.assertEqual(taken(None), "")
        acquire("production/srv2/1", path=self.path, kind="production", container="st-glm53")
        self.assertEqual(taken(read(self.path), mine_kind="production", container_up=lambda n: True), "")
        self.assertIn("production/srv2/1", taken(read(self.path), mine_kind="queue", container_up=lambda n: True))
        stale = dict(read(self.path), pid=0, beat=time.time() - 2 * GRACE_S, since=time.time() - 2 * GRACE_S)
        self.assertEqual(taken(stale, mine_kind="queue", container_up=lambda n: False), "")

    def test_the_asker_can_take_its_request_back(self):
        """A cancelled ticket must not leave production draining for nobody."""
        from engine.base.fleet_lease import request_yield, withdraw_yield, yield_requested
        acquire("holder", path=self.path)
        request_yield("queue/t1", path=self.path)
        self.assertFalse(withdraw_yield("queue/t2", path=self.path))
        self.assertTrue(withdraw_yield("queue/t1", path=self.path))
        self.assertIsNone(yield_requested(read(self.path)))

    def test_a_container_is_attached_as_evidence_later(self):
        from engine.base.fleet_lease import attach, owner_for
        acquire("queue/t1", path=self.path, kind="queue")
        self.assertEqual(owner_for("st-glm53", path=self.path), "queue/t1")   # no container yet: the owner all the same
        attach("queue/t1", "st-glm53", path=self.path)
        self.assertEqual(read(self.path)["container"], "st-glm53")
        with self.assertRaises(LeaseHeld):
            owner_for("st-probe-7", path=self.path)

    def test_the_quiet_gate_reads_the_door_like_deploy_watch(self):
        """One definition of quiet: the queue asks production to hand over by the rule
        deploy-watch applies to its own restarts."""
        from engine.base.fleet_lease import door_load, door_unsupported
        quiet = "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nst:handing_over 0\nst:quiet 1\n"
        self.assertEqual(door_load(quiet), 0)
        self.assertEqual(door_load(quiet.replace("waiting 0", "waiting 2")), 2)
        self.assertEqual(door_load(quiet.replace("st:quiet 1", "st:quiet 0")), 1)     # a tier transfer counts
        self.assertIsNone(door_load(""))                                                # no answer is not quiet
        self.assertTrue(door_unsupported("vllm:num_requests_running 0\n"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("st_deploy_watch", ROOT / "launchers/st-deploy-watch.py")
        watch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(watch)
        self.assertIs(watch.busy, door_load)
        self.assertIs(watch.unsupported, door_unsupported)


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
        self.assertIn('ssh $FLEET_LEASE_SSH "choiceoh@$FLEET_HEAD" "python3 - $quoted" < "$module"', helper)

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
        for relative in ('launchers/lib/fleet-lease.sh', 'engine/base/fleet_lease.py',
                         *fleet_onepass.ST_ENTRIES, *fleet_onepass.ST_PROBES):
            if (ROOT / relative).is_file():
                self.assertIn(relative, pinned, relative)
        # and the lease itself: a runner that cannot read the lease counts it as occupied,
        # which meant no runner-driven ticket could ever be granted (2026-09-12)
        self.assertIn("engine/base/fleet_lease.py", pinned)
        self.assertIn("launchers/lib/fleet-lease.sh", pinned)


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

    def test_a_ticket_s_probe_needs_no_heartbeat(self):
        """The head node's docker cannot see a probe container on another node. A probe used to beat
        for itself; now its lease is the ticket's, and the ticket's supervisor on the head node is the
        record's pid -- conclusive evidence, no heartbeat, and gone means free."""
        self.assertIn("fleet_lease_beat", self.helper)                # still there for shell callers
        self.assertNotIn("BEAT=$(fleet_lease_beat", self.probe)
        self.assertIn("the ticket's supervisor on the head node is the lease's", self.probe)

    def test_heartbeat_pid_capture_returns_while_renewal_is_running(self):
        import signal
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            mark = Path(tmp) / "renewed"
            script = '''
source "$1"
sleep() { command sleep 0.02; }
fleet_lease() { printf '%s\\n' "$*" >> "$RENEW_MARK"; }
beat=$(fleet_lease_beat test-owner)
trap 'kill "$beat" 2>/dev/null || true' EXIT
kill -0 "$beat" || exit 2
for i in {1..100}; do
  [ ! -s "$RENEW_MARK" ] || { printf 'ready\\n'; exit 0; }
  command sleep 0.02
done
exit 3
'''
            proc = subprocess.Popen(["bash", "-c", script, "test", str(ROOT / "launchers/lib/fleet-lease.sh")],
                                    env={**os.environ, "RENEW_MARK": str(mark)}, start_new_session=True,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = proc.communicate(timeout=5)
                self.assertEqual((proc.returncode, out), (0, "ready\n"), err)
                self.assertIn("renew --owner test-owner", mark.read_text())
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate()

    def test_a_cpu_only_probe_reserves_nothing(self):
        """Taking four Sparks for an import check is the opposite of smooth."""
        self.assertIn('if [ "${ST_PROBE_NO_GPU:-0}" != 1 ]; then', self.probe)
        self.assertIn("ST_PROBE_NO_GPU=1 for an import/source check that needs no GPU", self.probe)

    def test_a_killed_probe_leaks_nothing(self):
        """SIGKILL runs no trap. The lease is the queue's, so there is nothing here to leak: the
        ticket's supervisor sees the payload end and passes the lease on or lets it go. And a lease
        that does go stale is still reported as such."""
        self.assertIn("A probe killed outright leaks", self.probe)
        self.assertNotIn("fleet_lease release", self.probe)          # nothing of its own to release
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
        """The previous holder's handover lands as a transfer to this ticket; a probe the queue
        just sent in must not lose its turn to that last second."""
        self.assertIn("ST_PROBE_WAIT_MINUTES", self.probe)
        self.assertIn("waiting for the ticket's lease", self.probe)
        self.assertIn("the lease is not this ticket's after", self.probe)

    def test_the_queue_reads_the_lease_not_only_containers(self):
        """They disagreed once and the queue granted while the lease was still held."""
        self.assertIn("lease_state() { lease read", self.fleet)
        self.assertIn("st_engine_up() {", self.fleet)
        # and a lease handed to the very ticket that is asking is not occupation
        self.assertIn('[ -n "${ST_MINE:-}" ] && lease_mine "$ST_MINE" && return 0', self.fleet)

    def test_occupancy_is_a_fleet_fact_not_this_node_s_fact(self):
        """`docker ps` here answers for one of four nodes. A fleet whose rank 0 had gone
        while the other three still held their GPUs read exactly like an empty one."""
        self.assertIn("st_engine_elsewhere()", self.fleet)
        self.assertIn("FLEET_NODES_IPS:-10.10.10.1 10.10.10.2 10.10.10.3 10.10.10.4", self.fleet)
        body = self.fleet[self.fleet.index("st_engine_elsewhere()"):self.fleet.index("st_engine_lease()")]
        self.assertIn("ssh -o BatchMode=yes", body)                 # it really asks the other nodes
        self.assertIn("ST_PROBE_TTL", body)                         # _try_hold asks once a second

    def test_free_is_never_an_answer_from_ignorance(self):
        """A missing lease helper and an unreadable lease both used to fall through to
        `return 1` -- and free is the one answer an occupancy check may not guess (D3)."""
        lease = self.fleet[self.fleet.index("st_engine_lease()"):self.fleet.index("st_engine_evidence()")]
        self.assertIn("cannot say the fleet is free", lease)         # unreadable => evidence
        self.assertNotIn("|| return 1", lease)                       # ... never silence
        scan = self.fleet[self.fleet.index("st_engine_elsewhere()"):self.fleet.index("st_engine_lease()")]
        self.assertIn("unreachable -- this node cannot say the fleet is free", scan)

    def test_the_taken_line_always_names_its_evidence(self):
        """With the containers already gone and the lease still held, the status line read
        `TAKEN by the ST engine, outside this queue ()` -- true, and unreadable."""
        self.assertIn("TAKEN by the ST engine, outside this queue -- $(st_engine_line)", self.fleet)
        self.assertIn("st_engine_line() { st_engine_evidence |", self.fleet)

    def test_a_copy_answering_by_older_rules_says_so(self):
        """2026-09-12: `cd ~/stkernel && bash bench/fleet.sh status` on the controller said
        FREE with four nodes serving, because that checkout predated the ST-engine check.
        Hashes and mtimes cannot judge that -- a fresh checkout of an old branch is new by
        both -- so the answers carry a number."""
        self.assertRegex(self.fleet, r"(?m)^FLEET_RULES=[0-9]+$")
        self.assertIn("entry_line() {", self.fleet)
        self.assertIn("OLDER RULES", self.fleet)
        status = self.fleet[self.fleet.index('echo "fleet: $('):]
        self.assertIn("entry_line", status[:400])                    # and it prints under the verdict

    def test_preflight_cannot_move_the_shared_entry_backwards(self):
        """That sync copies whatever $REPO the caller ran from over $LOGD/fleet.sh, so the
        one copy every probe runs could be regressed by any stale checkout."""
        body = self.fleet[self.fleet.index('for pair in "ab-lever2.sh'):]
        body = body[:body.index("done")]
        self.assertIn("refusing to move the shared entry back", body)
        self.assertIn("entry_rules", body)

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
        """Otherwise a queued session waits for a human to go and ask -- and whom it may ask is
        the holder's KIND: production through the quiet gate, a session never (45차 §91)."""
        self.assertIn("st_engine_ask() {", self.fleet)
        self.assertIn('st_engine_ask "$s" "$pid" "$est" "$note"', self.fleet)
        self.assertIn("asked it to hand over to", self.fleet)

    def test_an_ask_that_could_not_be_made_is_said(self):
        """A runner runs out of a snapshot of bench/, engine/ and probes/. The ask used to go through
        launchers/lib/fleet-lease.sh, absent from every snapshot, and a logged ask nobody made left the
        waiter and the holder each believing the other had been told (45차 §91). Now the module is pinned
        into the snapshot and the ask goes through it; an ask that still fails is said to have failed."""
        self.assertIn("could not be asked", self.fleet)
        body = self.fleet.split("st_engine_ask() {", 1)[1].split("\n}", 1)[0]
        self.assertNotIn("|| true", body.split("lease yield", 1)[1].split("\n", 1)[0])   # the status decides the log line
        self.assertNotIn("launchers/lib/fleet-lease.sh", self.fleet)

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

    def test_the_handover_transfers_the_lease_to_the_requester(self):
        """Released, the fleet read as free for a moment and the production supervisor could relaunch
        into the window this drain was for (its own comment names that race). The record becomes the
        requester's in one step, as the request named it: kind, pid, host."""
        serve = (ROOT / "engine/base/serve.py").read_text()
        self.assertIn('fleet_lease.transfer(self.lease["owner"], to,', serve)
        import test_engine_serve as T
        from engine.base.fleet_lease import request_yield
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lease"
            acquire("holder", path=path, container="st-glm53")
            request_yield("queue/t1", path=path, reason="a ticket", kind="queue", pid=os.getpid())
            server = T.server()
            server.lease = {"owner": "holder", "path": str(path)}
            server.draining = "queue/t1"
            server._hand_over()
            record = read(path)
            self.assertEqual((record["owner"], record["kind"], record["handed_from"]), ("queue/t1", "queue", "holder"))
            self.assertFalse(server.alive)


class ProbeLeaseTests(unittest.TestCase):
    def test_the_probe_runner_verifies_the_ticket_s_lease(self):
        """A probe takes the same GPUs as a boot, so it runs under the queue's lease: the ticket hands
        ST_LEASE_OWNER here and the runner only verifies it. A bare run has no ticket and is refused --
        that was the last way onto the GPUs that no launcher and no queue could see."""
        runner = (ROOT / "probes/run_engine_probe.sh").read_text()
        self.assertIn('fleet_lease verify --owner "$ST_LEASE_OWNER"', runner)
        self.assertIn("this probe holds no fleet reservation", runner)
        self.assertIn("bash bench/fleet.sh run --gpu <session>", runner)
        self.assertNotIn("fleet_lease acquire", runner)              # one record: the queue's
        self.assertNotIn("release --owner", runner)
        self.assertIn('docker run --rm --name "$NAME"', runner)
        self.assertIn("ST_PROBE_NO_LEASE", runner)                   # a nested run under the same ticket

    def test_the_launcher_can_ask_and_can_report(self):
        launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()
        self.assertIn("  yield)", launcher)
        self.assertIn("  held)", launcher)
        self.assertIn("ST_LEASE_OWNER", launcher)
        self.assertIn("usage: $0 [start|stop|yield [reason]|held|logs r]", launcher)

    def test_the_boot_hands_the_lease_to_the_engine(self):
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn("def fleet_lease_of()", boot)
        # the reservation is now taken at the top of fleet() -- before the 67 GiB, not after -- and carried
        self.assertIn("lease = fleet_lease_of()", boot)
        self.assertIn("lease=lease).loop()", boot)


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


class ReservationIsRequiredTests(unittest.TestCase):
    """A reservation nobody has to hold is not one.

    On 2026-09-12 a yield was asked for through the protocol and granted -- the holder parked its
    conversations and let go -- and the containers came straight back up on all four nodes, because whatever
    started them never needed the lock. Nobody was at fault; there was nothing to be at fault against. So the
    check moved into the engine, which is the thing that actually occupies the fleet: a launcher can be
    bypassed with one `docker run`, and the engine cannot.
    """

    def boot(self):
        import importlib
        return importlib.import_module("engine.profiles.glm53.boot")

    def test_no_environment_means_no_boot(self):
        boot = self.boot()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                boot.fleet_lease_of()
        self.assertIn("holds no fleet reservation", str(caught.exception))
        self.assertIn("start-st-glm53.sh", str(caught.exception), "it has to say how to get one")

    def test_the_ranks_that_have_no_copy_of_the_lock_still_boot(self):
        """The lock is ONE file on the head node -- homes are not shared between the Sparks, which is
        why the helper pipes the module there. Requiring a record of every rank refused the fleet
        outright: rank 0 came up and the other three exited in under a second (2026-09-12, the first
        boot of this gate on real nodes). They are still held to the environment only the launcher
        sets, which is what a bare `docker run` would not have."""
        boot = self.boot()
        with tempfile.TemporaryDirectory() as tmp:
            absent = os.path.join(tmp, "st-fleet.lock")           # rank 1's node: no copy, by design
            env = {"ST_LEASE_OWNER": "me", "ST_LEASE_PATH": absent}
            with mock.patch.dict(os.environ, dict(env, RANK="1"), clear=True):
                self.assertEqual(boot.fleet_lease_of(), {"owner": "me", "path": absent})
            with mock.patch.dict(os.environ, dict(env, RANK="0"), clear=True):
                with self.assertRaises(RuntimeError):             # the head still has to hold it
                    boot.fleet_lease_of()
            with mock.patch.dict(os.environ, {"RANK": "3"}, clear=True):
                with self.assertRaises(RuntimeError) as caught:   # and no environment is still no boot
                    boot.fleet_lease_of()
            self.assertIn("holds no fleet reservation", str(caught.exception))

    def test_a_rank_that_can_read_the_lock_must_still_agree_with_it(self):
        boot = self.boot()
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "st-fleet.lock"
            acquire("someone-else", path=lock, container="st-glm53")
            with mock.patch.dict(os.environ, {"ST_LEASE_OWNER": "me", "ST_LEASE_PATH": str(lock),
                                              "RANK": "2"}, clear=True):
                with self.assertRaises(RuntimeError) as caught:
                    boot.fleet_lease_of()
            self.assertIn("someone-else", str(caught.exception))

    def test_an_empty_lock_is_not_a_reservation(self):
        boot = self.boot()
        with tempfile.TemporaryDirectory() as tmp:
            lock = os.path.join(tmp, "st-fleet.lock")
            with mock.patch.dict(os.environ, {"ST_LEASE_OWNER": "me", "ST_LEASE_PATH": lock}, clear=True):
                with self.assertRaises(RuntimeError) as caught:
                    boot.fleet_lease_of()
        self.assertIn("empty", str(caught.exception))

    def test_another_owner_s_reservation_is_not_this_boot_s(self):
        boot = self.boot()
        with tempfile.TemporaryDirectory() as tmp:
            lock = pathlib.Path(tmp) / "st-fleet.lock"
            acquire("someone@elsewhere", path=lock, container="st-glm53")
            with mock.patch.dict(os.environ, {"ST_LEASE_OWNER": "me", "ST_LEASE_PATH": str(lock)}, clear=True):
                with self.assertRaises(RuntimeError) as caught:
                    boot.fleet_lease_of()
        self.assertIn("someone@elsewhere", str(caught.exception))
        self.assertIn("yield", str(caught.exception), "it has to say how to ask for it")

    def test_the_boot_that_holds_it_proceeds(self):
        boot = self.boot()
        with tempfile.TemporaryDirectory() as tmp:
            lock = pathlib.Path(tmp) / "st-fleet.lock"
            acquire("me@srv4/1", path=lock, container="st-glm53")
            with mock.patch.dict(os.environ, {"ST_LEASE_OWNER": "me@srv4/1", "ST_LEASE_PATH": str(lock)}, clear=True):
                got = boot.fleet_lease_of()
        self.assertEqual(got, {"owner": "me@srv4/1", "path": str(lock)})


class OneRecordTests(unittest.TestCase):
    """The lease is ONE record and the queue is its authority (2026-09-12, after the operator asked
    why the queue and the lease were two things). Every boot holds one; the holder's KIND decides
    what the queue may ask of it; a handover is a transfer; production returns when nobody waits.

    Two live defects closed with it: a runner snapshot could not read the lease (the helper under
    launchers/ was never pinned) and "cannot read" rightly counted as occupied, so no runner-driven
    ticket could be granted; and the production supervisor read the legacy lock path, so a ticket's
    boot was invisible to it and its crash recovery would have evicted that boot 90 s in."""

    def setUp(self):
        self.fleet = (ROOT / "bench/fleet.sh").read_text()
        self.launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()
        self.supervisor = (ROOT / "launchers/st-glm53-supervisor.sh").read_text()
        self.watch = (ROOT / "launchers/st-deploy-watch.py").read_text()
        self.boot = (ROOT / "bench/fleet_boot.py").read_text()
        self.handoff = (ROOT / "bench/fleet_handoff.py").read_text()

    def test_the_queue_takes_the_lease_at_go_from_its_pinned_module(self):
        self.assertIn('lease acquire --owner "queue/$s" --kind queue --pid "$pid"', self.fleet)
        self.assertIn('lease_mine() { lease verify --owner "queue/$1"', self.fleet)
        self.assertIn('lease() { python3 "${FLEET_RUNNER_REPO:-$REPO}/engine/base/fleet_lease.py"', self.fleet)
        self.assertNotIn("launchers/lib/fleet-lease.sh", self.fleet)     # the helper no snapshot carried
        hold = self.fleet[self.fleet.index("_try_hold() {"):self.fleet.index("_ledger_row() {")]
        self.assertIn('if [ "$kind" = boot ] && ! lease_mine "$s"; then', hold)    # a probe runs beside production; the single lane is not the fleet
        self.assertIn('ST_MINE=$s st_engine_up', hold)
        self.assertRegex(self.fleet, r"(?m)^FLEET_RULES=[3-9][0-9]*$")   # occupancy answers changed here (3); later admission changes bump it further

    def test_a_boot_holder_lives_by_the_lease(self):
        body = self.fleet[self.fleet.index("holder_alive() {"):self.fleet.index("holder_probe() {")]
        self.assertIn('lease_mine "$s" && return 0', body)
        self.assertIn("the lease is somebody else's", body)
        self.assertIn("kill -0", body)                                    # a probe holder, and a pre-lease holder, as before

    def test_the_lease_goes_ticket_to_ticket_and_back_to_production_when_nobody_waits(self):
        self.assertIn("_lease_pass_on() {", self.fleet)
        self.assertIn('fleet_handoff.py" next "$FLEET_DIR" "$1"', self.fleet)
        self.assertIn('lease transfer --owner "queue/$1" --to "queue/$next" --kind queue --pid "$npid"', self.fleet)
        self.assertIn("no boot ticket waits: production restores itself", self.fleet)
        release = self.fleet[self.fleet.index("_release() {"):self.fleet.index("_adopt() {")]
        self.assertIn('[ "${hkind:-boot}" != boot ] || [ "${FLEET_KEEP_LEASE:-0}" = 1 ] || _lease_pass_on "$1"', release)
        self.assertIn("FLEET_KEEP_LEASE=1 FLEET_NO_RESTORE_CHECK=1 with_lock _release", self.fleet)   # a yield to a probe keeps it
        self.assertIn("elif args.action == 'next':", self.handoff)
        dequeue = self.fleet[self.fleet.index("_dequeue() {"):self.fleet.index("_withdraw_owned() {")]
        self.assertIn('lease withdraw-yield --requester "queue/$1"', dequeue)   # a cancelled ticket takes its ask back
        self.assertIn('|| ! lease_mine "$1" || _lease_pass_on "$1"', dequeue)

    def test_production_is_asked_only_through_the_quiet_gate_and_a_session_never(self):
        self.assertIn("production_quiet() {", self.fleet)
        self.assertIn("FLEET_QUIET_S=${FLEET_QUIET_S:-120}", self.fleet)       # deploy-watch's --quiet default
        ask = self.fleet[self.fleet.index("st_engine_ask() {"):self.fleet.index("serving_idle() {")]
        self.assertIn("production) ;;", ask)
        self.assertIn("is not asked", ask)
        self.assertIn('lease yield --requester "queue/$s" --kind queue --pid "$pid" --host "$(me)"', ask)
        self.assertIn('[ -f "$marker" ] && return 0', ask)                       # once per ticket
        self.assertIn("production_quiet ||", ask.replace("if ! production_quiet; then", "production_quiet ||"))
        self.assertNotIn("st_engine_yield", self.fleet)

    def test_the_launcher_verifies_a_ticket_s_lease_and_refuses_a_bare_boot(self):
        self.assertIn('held=$(lease verify --owner "$LEASE_OWNER" 2>&1)', self.launcher)
        self.assertIn('elif [ "${ST_LEASE_KIND:-}" = production ]; then', self.launcher)
        self.assertIn("--kind production", self.launcher)
        self.assertIn("this boot holds no reservation", self.launcher)
        self.assertIn('elif [ "${ST_LEASE_KIND:-}" = session ]; then', self.launcher)   # by hand, and said so
        self.assertIn("--kind session", self.launcher)
        self.assertNotIn("FLEET_HOLDER", self.launcher)                         # the holder file is not a second record
        self.assertIn('lease attach --owner "$LEASE_OWNER" --container "$NAME"', self.launcher)

    def test_only_the_holder_stops_its_boot(self):
        """Every boot is named st-glm53: a stop resolved by container name alone let the production
        supervisor's crash recovery evict a ticket's boot."""
        stop = self.launcher[self.launcher.index("  stop)"):self.launcher.index("  yield)")]
        self.assertIn("not by this ticket", stop)
        self.assertIn("STOP_FORCE", stop)
        self.assertIn('case "$held_kind" in "$ST_LEASE_KIND"|free) ;;', stop)
        self.assertIn('[ "$held_origin" = explicit ]', stop)                    # records from before kinds: the old rule
        self.assertIn("--state phase=stopped", stop)                            # a ticket's lease is the queue's to let go

    def test_a_lease_from_before_kinds_is_judged_by_its_container(self):
        """The production lease running on 2026-09-12 names no kind. The supervisor of that era
        judged "mine" by the container name, so such records keep being judged by it -- or the
        first deploy of this code would wait forever behind its own production."""
        from engine.base.fleet_lease import _kind, origin, taken
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lease"
            path.write_text(json.dumps({"owner": "prod-release", "container": "st-glm53", "since": time.time(), "beat": time.time()}) + "\n")
            self.assertEqual((origin(read(path)), _kind(read(path))), ("legacy", "production"))
            self.assertEqual(taken(read(path), mine_kind="production", container_up=lambda n: True), "")
            path.write_text(json.dumps({"owner": "other-task", "container": "st-probe", "since": time.time(), "beat": time.time()}) + "\n")
            self.assertEqual(_kind(read(path)), "session")
            self.assertIn("other-task", taken(read(path), mine_kind="production", container_up=lambda n: True))
            path.write_text("choiceoh@srv2 st-glm53 2026-09-12 10:00:00\n")
            self.assertEqual((origin(read(path)), _kind(read(path))), ("opaque", "production"))
            acquire("queue/t1", path=Path(tmp) / "fresh", kind="queue")
            self.assertEqual(origin(read(Path(tmp) / "fresh")), "explicit")

    def test_the_supervisor_reads_the_one_lease_file_by_kind(self):
        self.assertIn("LOCK=${FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock}", self.supervisor)
        self.assertIn("lease_head taken --kind production", self.supervisor)
        self.assertIn("ST_LEASE_KIND=production LEASE_OWNER_PRODUCTION=$PROD_OWNER", self.supervisor)
        taken = self.supervisor[self.supervisor.index("fleet_taken(){"):self.supervisor.index("forensics(){")]
        # the older path keeps the older rule (a plain-text lock naming the container); the one
        # lease file is judged by kind, never by the container name every boot shares
        self.assertIn('lease_at "$LEGACY_LOCK" owner --container "$NAME"', taken)
        self.assertEqual(taken.count("owner --container"), 1)
        self.assertNotIn("/home/choiceoh/st-fleet.lock", taken)
        self.assertIn('*) echo "lease unreadable (rc=$rc)"; return 0 ;;', taken)   # unreadable is not free

    def test_deploy_watch_defers_to_a_granted_window_without_recording_a_rejection(self):
        self.assertIn("def fleet_taken_by_another(log)", self.watch)
        self.assertIn('"ST_LEASE_KIND": "production"', self.watch)
        cycle = self.watch[self.watch.index("def cycle("):self.watch.index("def main(")]
        self.assertLess(cycle.index("fleet_taken_by_another(log)"), cycle.index("ok = deploy(release, log)"))
        self.assertIn("# deferred, not rejected", cycle)

    def test_the_supervisor_hands_the_ticket_s_owner_to_the_payload(self):
        self.assertIn("self.env['ST_LEASE_OWNER'] = 'queue/' + session", self.boot)
        self.assertIn("'ST_LEASE_OWNER', 'ST_LEASE_PATH', 'FLEET_LEASE_PATH'", self.boot)
