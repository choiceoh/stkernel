"""The queue holds the fleet lease -- bench/fleet.sh's own functions, in a sandbox.

Taken at GO, asked for through the quiet gate, handed ticket to ticket, released when nobody
waits (2026-09-12, the operator's "대기 예약이 없을 때만 되돌리면 되지"). A private FLEET_DIR, a
private lease file, a door whose /metrics this test writes, a docker that sees no container, no
ssh. What is exercised is the real `_try_hold`, `_release`, `_dequeue` and `holder_alive`, with
the real fleet_handoff / fleet_priority / fleet_pause / fleet_idle helpers under them.

Linux only: fleet.sh needs flock and GNU date/stat/find. On a Mac, run it in a container.
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = sys.platform.startswith("linux") and bool(shutil.which("flock"))
QUIET = "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nst:handing_over 0\nst:quiet 1\n"
BUSY = QUIET.replace("running 0", "running 2")
STUBS = """
st_engine_elsewhere() { :; }                 # no other nodes to ask in a sandbox
legacy_busy() { return 1; }
busy_procs() { echo 0; }
busy_reqs() { echo 0; }
"""                                          # serving_idle and booting are the real ones: the stub docker and curl feed them


@unittest.skipUnless(LINUX, "bench/fleet.sh needs flock and GNU coreutils: run this inside a Linux container")
class LeaseQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.fleet_dir = root / "fleet"
        self.fleet_dir.mkdir()
        self.logd = root / "logs"
        self.logd.mkdir()
        self.lease = root / "st-fleet.lock"
        self.metrics = root / "metrics.txt"
        self.metrics.write_text(QUIET)
        stubs = root / "bin"
        stubs.mkdir()
        docker = ('#!/bin/sh\n'                       # no container unless the test says so (FAKE_DOCKER_PS)
                  'case "$*" in\n'
                  '  *"{{.Status}}"*) [ -z "${FAKE_DOCKER_PS:-}" ] || printf "%s Up 5 hours\\n" "$FAKE_DOCKER_PS";;\n'
                  '  *ps*) [ -z "${FAKE_DOCKER_PS:-}" ] || printf "%s\\n" "$FAKE_DOCKER_PS";;\n'
                  'esac\nexit 0\n')
        for name, body in (("docker", docker),
                           ("curl", f'#!/bin/sh\ncat "{self.metrics}" 2>/dev/null\n'),
                           ("ssh", "#!/bin/sh\nexit 255\n")):
            (stubs / name).write_text(body)
            (stubs / name).chmod(0o755)
        text = (ROOT / "bench/fleet.sh").read_text()
        prelude = text[:text.index("\ncmd=${1:-status}")] + "\n" + STUBS
        self.prelude = root / "prelude.sh"
        self.prelude.write_text(prelude)
        self.env = dict(os.environ, PATH=f"{stubs}:{os.environ['PATH']}", FLEET_DIR=str(self.fleet_dir),
                        LOGD=str(self.logd), FLEET_LEASE_PATH=str(self.lease), HEAD_URL="http://door",
                        FLEET_QUIET_S="0", REPO=str(ROOT), FLEET_RUNNER_REPO=str(ROOT))
        self.host = socket.gethostname().split(".")[0]

    # -- the sandbox's verbs --------------------------------------------------------------------
    def sh(self, script, check=True):
        done = subprocess.run(["bash", "-c", f"source '{self.prelude}'\n{script}"], env=self.env,
                              capture_output=True, text=True, timeout=120)
        if check and done.returncode:
            raise AssertionError(f"rc={done.returncode}\n{script}\n{done.stdout}\n{done.stderr}\n{self.log()}")
        return done

    def sleeper(self):
        process = subprocess.Popen(["sleep", "300"])

        def reap():
            process.kill()
            process.wait()
        self.addCleanup(reap)
        return process.pid

    def enqueue(self, session, pid, kind="boot", est=10):
        self.sh(f'with_lock _enqueue {session} {est} note {kind} {pid}')
        self.sh(f'python3 "$REPO/bench/fleet_handoff.py" ready "$FLEET_DIR" {session} {pid}')

    def try_hold(self, session, pid, kind="boot"):
        return self.sh(f'with_lock _try_hold {session} {pid} 10 note {kind}', check=False).returncode

    def lease_cmd(self, *args):
        done = subprocess.run([sys.executable, str(ROOT / "engine/base/fleet_lease.py"), *args, "--path", str(self.lease)],
                              env=self.env, capture_output=True, text=True, timeout=60)
        return done.returncode, done.stdout.strip(), done.stderr.strip()

    def record(self):
        sys.path.insert(0, str(ROOT))
        from engine.base.fleet_lease import read
        return read(self.lease)

    def log(self):
        try:
            return (self.fleet_dir / "log").read_text()
        except FileNotFoundError:
            return ""

    def holder(self):
        try:
            return (self.fleet_dir / "holder").read_text()
        except FileNotFoundError:
            return ""

    # -- the scenarios -----------------------------------------------------------------------------
    def test_production_is_asked_through_the_quiet_gate_and_the_handover_lands_as_go(self):
        self.lease_cmd("acquire", "--owner", "production/srv2/1", "--kind", "production", "--container", "st-glm53")
        pid = self.sleeper()
        self.enqueue("t1", pid)
        self.metrics.write_text(BUSY)
        self.assertEqual(self.try_hold("t1", pid), 1)
        self.assertIn("production holds the fleet and is not quiet", self.log())
        self.assertEqual(self.lease_cmd("asked")[1], "no")                     # a busy engine is not asked
        self.metrics.write_text(QUIET)
        self.assertEqual(self.try_hold("t1", pid), 1)                          # asked, not granted
        self.assertIn("asked it to hand over to t1", self.log())
        asked = self.record()["yield_to"]
        self.assertEqual((asked["requester"], asked["kind"], asked["pid"], asked["host"]), ("queue/t1", "queue", pid, self.host))
        self.assertEqual(self.try_hold("t1", pid), 1)
        self.assertEqual(self.log().count("asked it to hand over"), 1)         # once per ticket
        # the engine drains and hands the lease to exactly that record (engine/base/serve._hand_over)
        self.lease_cmd("transfer", "--owner", "production/srv2/1", "--to", "queue/t1", "--kind", "queue",
                       "--pid", str(pid), "--host", self.host)
        self.assertEqual(self.try_hold("t1", pid), 0)                          # GO
        self.assertTrue(self.holder().startswith(f"t1|{pid}|"), self.holder())
        self.assertEqual((self.record()["owner"], self.record()["kind"]), ("queue/t1", "queue"))
        self.assertIn(f"GO t1 (pid {pid}) holding the lease as queue/t1", self.log())
        self.assertEqual(self.sh("holder_alive", check=False).returncode, 0)
        self.assertEqual(self.sh("ST_MINE=t1 st_engine_up", check=False).returncode, 1)   # its own lease is not occupation
        self.assertEqual(self.sh("st_engine_up", check=False).returncode, 0)              # to anyone else it is

    def test_the_lease_goes_ticket_to_ticket_and_back_when_nobody_waits(self):
        p1, p2 = self.sleeper(), self.sleeper()
        self.enqueue("t1", p1)
        self.assertEqual(self.try_hold("t1", p1), 0)                           # a free fleet: taken at GO
        self.assertEqual((self.record()["owner"], self.record()["kind"], self.record()["pid"]), ("queue/t1", "queue", p1))
        self.enqueue("t2", p2)
        self.sh("with_lock _release t1")
        self.assertEqual(self.holder(), "")
        self.assertIn("lease handed from t1 to t2", self.log())
        handed = self.record()
        self.assertEqual((handed["owner"], handed["pid"], handed["handed_from"]), ("queue/t2", p2, "queue/t1"))
        self.assertEqual(self.try_hold("t2", p2), 0)                           # already its: GO without acquiring
        self.sh("with_lock _release t2")
        self.assertIsNone(self.record())
        self.assertIn("no boot ticket waits: production restores itself", self.log())

    def test_a_waiting_probe_is_a_reason_to_let_go(self):
        p1, p2 = self.sleeper(), self.sleeper()
        self.enqueue("t1", p1)
        self.assertEqual(self.try_hold("t1", p1), 0)
        self.enqueue("probe1", p2, kind="probe")
        self.sh("with_lock _release t1")
        self.assertIsNone(self.record(), "a probe runs beside production, so production comes back")
        self.assertIn("no boot ticket waits", self.log())

    def test_a_session_holder_is_never_asked_and_a_cancelled_ticket_takes_its_ask_back(self):
        self.lease_cmd("acquire", "--owner", "someone@srv4/7", "--kind", "session", "--container", "st-glm53")
        pid = self.sleeper()
        self.enqueue("t1", pid)
        self.assertEqual(self.try_hold("t1", pid), 1)
        self.assertIn("t1 waits (a session holder is not asked)", self.log())
        self.assertEqual(self.lease_cmd("asked")[1], "no")
        self.lease_cmd("release", "--owner", "someone@srv4/7")
        self.lease_cmd("acquire", "--owner", "production/srv2/1", "--kind", "production", "--container", "st-glm53")
        self.assertEqual(self.try_hold("t1", pid), 1)
        self.assertTrue(self.lease_cmd("asked")[1].startswith("queue/t1"))
        self.sh("with_lock _dequeue t1")
        self.assertEqual(self.lease_cmd("asked")[1], "no")                     # production is not left draining for nobody
        self.assertEqual(self.record()["owner"], "production/srv2/1")

    def test_a_dead_holder_is_kicked_and_its_lease_reclaimed(self):
        dead = 999999999
        self.lease_cmd("acquire", "--owner", "queue/t9", "--kind", "queue", "--pid", str(dead))
        (self.fleet_dir / "holder").write_text(f"t9|{dead}|{self.host}|{int(time.time())}|10|note|boot\n")
        self.assertEqual(self.sh("holder_alive", check=False).returncode, 1)  # its supervisor is gone: no grace
        pid = self.sleeper()
        self.enqueue("t1", pid)
        self.assertEqual(self.try_hold("t1", pid), 0)
        self.assertIn("auto-kick dead holder", self.log())
        self.assertEqual(self.record()["owner"], "queue/t1")

    def test_a_probe_ticket_runs_beside_an_idle_production_lease_and_takes_no_lease(self):
        """The live onepass (D17's sample of the deployed commit) needs production UP and idle: the
        production lease is not occupation for it, the door's own quiet reading is its condition,
        and it takes no lease. Behind a session's boot it waits like everything else."""
        self.lease_cmd("acquire", "--owner", "production/srv2/1", "--kind", "production", "--container", "st-glm53")
        self.env["FAKE_DOCKER_PS"] = "st-glm53"                 # production's containers are up
        pid = self.sleeper()
        self.enqueue("p1", pid, kind="probe")
        self.metrics.write_text(BUSY)
        self.assertEqual(self.try_hold("p1", pid, kind="probe"), 1)          # a busy door is not idle
        self.metrics.write_text(QUIET)
        self.assertEqual(self.try_hold("p1", pid, kind="probe"), 0)          # GO, beside production
        self.assertTrue(self.holder().startswith(f"p1|{pid}|"))
        self.assertEqual(self.record()["owner"], "production/srv2/1", "the probe took no lease")
        self.assertNotIn("asked it to hand over", self.log())
        self.sh("with_lock _release p1")
        self.assertEqual(self.record()["owner"], "production/srv2/1", "and released none")
        # behind a session's boot: refused, and the session is not asked
        self.lease_cmd("release", "--owner", "production/srv2/1")
        self.lease_cmd("acquire", "--owner", "someone@srv4/7", "--kind", "session", "--container", "st-glm53")
        pid2 = self.sleeper()
        self.enqueue("p2", pid2, kind="probe")
        self.assertEqual(self.try_hold("p2", pid2, kind="probe"), 1)
        self.assertIn("p2 waits (a session holder is not asked)", self.log())

    def test_status_names_the_lease(self):
        self.lease_cmd("acquire", "--owner", "production/srv2/1", "--kind", "production", "--container", "st-glm53")
        done = subprocess.run(["bash", str(ROOT / "bench/fleet.sh"), "status"], env=self.env,
                              capture_output=True, text=True, timeout=180)
        self.assertIn("lease: production production/srv2/1 on", done.stdout, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
