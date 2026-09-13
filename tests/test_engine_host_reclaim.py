"""Admission asks the host for the page cache instead of mapping the shortfall (2026-09-13).

srv2 runs overcommit_memory=2 at the default ratio 50: CommitLimit 75.8 GiB, and the engine's reclaim was one
anonymous mapping of the allocation plus headroom, 73-76 GiB, refused however clean the cache under it. The
kernel setting is the operator's to change. The engine now asks a broker on its node's host
(launchers/st-reclaim-broker.sh), which drops the cache without allocating anything, and falls back to its own
fault only when nobody serves or the host's drop was not enough.

The client is exercised against a thread playing the broker and against the real broker script with sudo,
sync and docker stubbed on PATH.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.base.arena import GIB, host_reclaim, prepare_allocation              # noqa: E402
from engine.base.runtime_memory import reclaim_preparation_pages                # noqa: E402


def meminfo(free, available, cached=0):
    return (f"MemFree: {free * GIB // 1024} kB\nMemAvailable: {available * GIB // 1024} kB\n"
            f"Cached: {cached * GIB // 1024} kB\n")


class FakeBroker(threading.Thread):
    """The broker's side of the files, answering with `rc` and doing `effect` first (the drop)."""

    def __init__(self, root, rc=0, line="file cache returned: MemFree 34.0 GiB -> MemFree 96.0 GiB", effect=None, answer_id=None):
        super().__init__(daemon=True)
        self.root, self.rc, self.line, self.effect, self.answer_id = Path(root), rc, line, effect, answer_id
        self.stop, self.served = threading.Event(), []

    def run(self):
        while not self.stop.is_set():
            (self.root / "heartbeat").touch()
            request = self.root / "request"
            if request.exists():
                ident = request.read_text().strip()
                request.unlink()
                self.served.append(ident)
                if self.effect:
                    self.effect()
                (self.root / "done.tmp").write_text(f"{self.answer_id or ident}\n{self.rc}\n{self.line}\n")
                os.replace(self.root / "done.tmp", self.root / "done")
            time.sleep(0.01)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def broker(self, **kw):
        b = FakeBroker(self.root, **kw)
        b.start()
        self.addCleanup(b.stop.set)
        for _ in range(200):
            if (self.root / "heartbeat").exists():
                break
            time.sleep(0.005)
        return b

    def test_nobody_to_ask_means_no_question_and_no_wait(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ST_RECLAIM_DIR", None)
            self.assertIsNone(host_reclaim())
        self.assertIsNone(host_reclaim(self.root), "no heartbeat: no broker")
        (self.root / "heartbeat").touch()
        stale = time.time() - 60
        os.utime(self.root / "heartbeat", (stale, stale))
        started = time.monotonic()
        self.assertIsNone(host_reclaim(self.root), "a heartbeat from a broker that ended")
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse((self.root / "request").exists(), "and nothing is left for a later broker to act on")

    def test_a_serving_broker_answers_this_question_with_its_report(self):
        b = self.broker()
        with patch.dict(os.environ, {"ST_RECLAIM_DIR": str(self.root)}):
            answer = host_reclaim(timeout_s=10)
        self.assertEqual(answer, dict(returned=True, rc=0, line="file cache returned: MemFree 34.0 GiB -> MemFree 96.0 GiB"))
        self.assertEqual(len(b.served), 1)
        self.assertFalse((self.root / "done").exists(), "the answer is taken")

    def test_a_host_that_could_not_drop_it_says_so(self):
        self.broker(rc=3, line="file cache NOT returned (sudo -n refused): MemFree 34.0 GiB")
        answer = host_reclaim(self.root, timeout_s=10)
        self.assertEqual((answer["returned"], answer["rc"]), (False, 3))
        self.assertIn("sudo -n refused", answer["line"])

    def test_a_rank_asks_at_most_once_in_the_gap(self):
        b = self.broker()
        first = host_reclaim(self.root, timeout_s=10)
        self.assertTrue(first["returned"])
        self.assertIsNone(host_reclaim(self.root, timeout_s=10), "within the gap: the caller falls back, nothing is written")
        self.assertFalse((self.root / "request").exists())
        self.assertEqual(len(b.served), 1)
        self.assertTrue(host_reclaim(self.root, timeout_s=10, min_gap_s=0)["returned"], "past the gap it asks again")
        self.assertEqual(len(b.served), 2)

    def test_an_answer_to_another_question_is_not_this_ones(self):
        (self.root / "done").write_text("an-earlier-boot\n0\nstale\n")
        self.broker(answer_id="someone-else")
        clock = iter(range(0, 1000, 1))
        answer = host_reclaim(self.root, timeout_s=3, clock=lambda: next(clock), sleep=lambda s: time.sleep(0.02))
        self.assertEqual((answer["returned"], answer["rc"]), (False, None))
        self.assertIn("did not answer in 3 s", answer["line"])


class AdmissionTests(unittest.TestCase):
    def test_when_the_host_drop_is_enough_nothing_is_mapped(self):
        states = [meminfo(34, 104, 75)]
        asked = []

        def host():
            asked.append(1)
            states.append(meminfo(96, 104, 3))
            return dict(returned=True, rc=0, line="file cache returned")

        with patch.object(Path, "read_text", side_effect=lambda *a, **k: states[-1]):
            report = prepare_allocation(int(55.47 * GIB), [], 18 * GIB, lambda: 120 * GIB,
                                        reclaim=lambda n: self.fail("no anonymous mapping once the host dropped it"),
                                        host_reclaim=host)
        self.assertEqual(asked, [1])
        self.assertEqual((report["reclaimed"], report["immediately_free"]), (0, 96 * GIB))
        self.assertEqual(report["host_reclaim"]["line"], "file cache returned")
        self.assertEqual(report["file_cache"], 75 * GIB, "what admission found before it asked")

    def test_when_the_host_is_not_enough_the_old_fault_still_runs_and_the_refusal_says_what_the_host_said(self):
        states = [meminfo(60, 99, 20)]
        faulted = []

        def host():
            states.append(meminfo(66, 99, 14))
            return dict(returned=False, rc=3, line="file cache NOT returned (sudo -n refused)")

        def pump(n):
            faulted.append(n)
            states.append(meminfo(100, 99, 0))
            return n

        with patch("engine.base.runtime_memory.oom_floor", return_value=(6 * GIB, 9 * GIB // 2)), \
             patch.object(Path, "read_text", side_effect=lambda *a, **k: states[-1]):
            report = prepare_allocation(int(55.47 * GIB), [], 18 * GIB, lambda: 120 * GIB, reclaim=pump, host_reclaim=host)
        self.assertEqual(len(faulted), 1)
        self.assertEqual(report["host_reclaim"]["rc"], 3)
        # 75 GiB available covers the 73.47 needed, so the host is asked; faulting it would cross the SIGTERM line
        with patch("engine.base.runtime_memory.oom_floor", return_value=(6 * GIB, 9 * GIB // 2)), \
             patch.object(Path, "read_text", return_value=meminfo(40, 75, 35)):
            with self.assertRaisesRegex(MemoryError, r"cannot be reclaimed.*the host did not return file cache: refused"):
                prepare_allocation(int(55.47 * GIB), [], 18 * GIB, lambda: 120 * GIB, reclaim=lambda n: n,
                                   host_reclaim=lambda: dict(returned=False, rc=3, line="refused"))

    def test_no_broker_is_the_admission_it_always_was(self):
        with patch.object(Path, "read_text", return_value=meminfo(96, 104, 3)):
            report = prepare_allocation(int(55.47 * GIB), [], 18 * GIB, lambda: 120 * GIB, reclaim=None,
                                        host_reclaim=lambda: None)
        self.assertIsNone(report["host_reclaim"])

    def test_the_host_is_asked_only_when_admission_is_short(self):
        with patch.object(Path, "read_text", return_value=meminfo(96, 104, 3)):
            prepare_allocation(int(55.47 * GIB), [], 18 * GIB, lambda: 120 * GIB, reclaim=None,
                               host_reclaim=lambda: self.fail("enough was free"))

    def test_a_node_short_of_memory_rather_than_of_cache_is_not_asked(self):
        """MemAvailable under the need: dropping every clean page still would not cover it."""
        with patch("engine.base.runtime_memory.oom_floor", return_value=(6 * GIB, 9 * GIB // 2)), \
             patch.object(Path, "read_text", return_value=meminfo(40, 70, 30)):
            with self.assertRaises(MemoryError) as caught:
                prepare_allocation(int(55.47 * GIB), [], 18 * GIB, lambda: 120 * GIB, reclaim=lambda n: n,
                                   host_reclaim=lambda: self.fail("nothing the host drops can cover it"))
        self.assertNotIn("the host", str(caught.exception))

    def test_warmup_asks_only_when_the_cache_covers_the_shortfall(self):
        """Checkpoints run by the hundred: a node short of memory, not of cache, must not drop the box's cache at each."""
        states = [meminfo(10, 15, 5)]
        with patch("engine.base.arena._meminfo", side_effect=lambda: {k: int(v.split()[0]) * 1024 for k, v in
                                                                     (l.split(":") for l in states[-1].splitlines())}), \
             patch("engine.base.arena.touch_pages", side_effect=lambda n: self.fail("refused by the MemAvailable gate")):
            self.assertEqual(reclaim_preparation_pages(20 * GIB, 8 * GIB, host_reclaim=lambda: self.fail("15 < 20")), 0)

    def test_warmup_reclaim_asks_the_host_before_it_faults(self):
        states = [meminfo(5, 60, 40)]

        def host():
            states.append(meminfo(30, 60, 15))
            return dict(returned=True, rc=0, line="returned")

        with patch("engine.base.arena._meminfo", side_effect=lambda: {k: int(v.split()[0]) * 1024 for k, v in
                                                                     (l.split(":") for l in states[-1].splitlines())}), \
             patch("engine.base.arena.touch_pages", side_effect=lambda n: self.fail("the host's drop was enough")):
            self.assertEqual(reclaim_preparation_pages(20 * GIB, 8 * GIB, host_reclaim=host), 0)


class BrokerScriptTests(unittest.TestCase):
    """launchers/st-reclaim-broker.sh itself: start, answer through the real return script, end with its container."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        home = Path(self.tmp.name)
        self.bin, self.dir = home / "bin", home / "st-reclaim" / "rank2"
        self.bin.mkdir()
        self.running = home / "running"
        self.running.write_text("true\n")
        for name, body in (("sudo", f'#!/bin/sh\necho drop >> "{home}/drops"\nexit 0\n'),
                           ("sync", "#!/bin/sh\nexit 0\n"), ("timeout", '#!/bin/sh\nshift\nexec "$@"\n'),
                           ("docker", f'#!/bin/sh\ncat "{self.running}"\n')):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        (home / "meminfo").write_text("MemFree: 35651584 kB\nMemAvailable: 109051904 kB\nCached: 78643200 kB\n")
        self.drops = home / "drops"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", ST_MEMINFO=str(home / "meminfo"),
                        ST_OVERCOMMIT=str(home / "none"), ST_RECLAIM_POLL_S="0.05", ST_RECLAIM_CHECK_S="1")
        self.addCleanup(self.broker, "stop", str(self.dir))

    def broker(self, *args):
        return subprocess.run(["bash", str(ROOT / "launchers/st-reclaim-broker.sh"), *args], env=self.env, text=True,
                              capture_output=True, timeout=30)

    def pid(self):
        text = (self.dir / "broker.pid").read_text().strip() if (self.dir / "broker.pid").exists() else ""
        return int(text) if text else None

    def wait_gone(self, pid, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return True
            time.sleep(0.05)
        return False

    def test_it_answers_the_engine_and_ends_when_its_container_is_gone(self):
        started = self.broker("start", str(self.dir), "st-glm53")
        self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
        self.assertIn(f"reclaim broker serving {self.dir} (pid ", started.stdout)
        pid = self.pid()
        answer = host_reclaim(self.dir, timeout_s=20)
        self.assertEqual((answer["returned"], answer["rc"]), (True, 0), answer)
        self.assertTrue(answer["line"].startswith("file cache returned: MemFree 34.0 GiB, MemAvailable 104.0 GiB, Cached 75.0 GiB"),
                        answer["line"])
        self.assertEqual(self.drops.read_text().split(), ["drop"])
        time.sleep(1.2)                                        # it has seen the container run
        self.running.write_text("false\n")
        self.assertTrue(self.wait_gone(pid), "a broker outlived its container")
        self.assertFalse((self.dir / "heartbeat").exists())
        self.assertIsNone(host_reclaim(self.dir, timeout_s=1, min_gap_s=0), "and nobody asks a broker that ended")

    def test_a_new_boot_replaces_the_old_broker_and_stop_all_ends_it(self):
        self.assertEqual(self.broker("start", str(self.dir), "st-glm53").returncode, 0)
        first = self.pid()
        second_start = self.broker("start", str(self.dir), "st-glm53")
        self.assertEqual(second_start.returncode, 0, second_start.stdout + second_start.stderr)
        second = self.pid()
        self.assertNotEqual(first, second)
        self.assertTrue(self.wait_gone(first), "the old broker was ended")
        self.assertEqual(self.broker("stop-all", str(self.dir.parent)).returncode, 0)
        self.assertTrue(self.wait_gone(second))
        self.assertFalse((self.dir / "broker.pid").exists())

    def test_a_request_left_from_an_earlier_boot_is_not_served(self):
        self.dir.mkdir(parents=True)
        (self.dir / "request").write_text("left-over\n")
        (self.dir / "done").write_text("left-over\n0\nstale\n")
        self.assertEqual(self.broker("start", str(self.dir), "st-glm53").returncode, 0)
        time.sleep(0.5)                                        # ten polls
        self.assertFalse(self.drops.exists())
        self.assertFalse((self.dir / "done").exists())

    def test_a_container_that_never_runs_does_not_keep_a_broker(self):
        self.running.write_text("\n")
        self.env["ST_RECLAIM_UNSEEN_S"] = "1"
        self.dir.mkdir(parents=True)
        # Exercise termination directly: a one-second wall-clock deadline
        # can expire before the detached start's heartbeat poll sees it.
        ended = self.broker("serve", str(self.dir), "st-glm53")
        self.assertEqual(ended.returncode, 0, ended.stdout + ended.stderr)
        self.assertIn("st-glm53 never ran: done", ended.stdout)
        self.assertFalse((self.dir / "broker.pid").exists())
        self.assertFalse((self.dir / "heartbeat").exists())


class WiringTests(unittest.TestCase):
    def test_the_boot_asks_the_host_at_admission_and_at_warmup(self):
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn("from engine.base.arena import Arena, host_reclaim, prepare_allocation", boot)
        self.assertEqual(boot.count("host_reclaim=host_reclaim"), 2, "prepare_allocation and the warmup reclaim")
        self.assertIn('recorder.gauge("boot_host_reclaim", 0 if report["host_reclaim"] is None else 1 + int(report["host_reclaim"]["returned"]))', boot)

    def test_the_launcher_starts_the_broker_before_the_container_and_names_its_directory(self):
        text = (ROOT / "launchers/start-st-glm53.sh").read_text()
        rank = text[text.index("start_rank() {"):text.index("pids=()")]
        self.assertLess(rank.index("st-return-file-cache.sh"), rank.index("st-reclaim-broker.sh start"))
        self.assertLess(rank.index("st-reclaim-broker.sh start"), rank.index("docker run -d --name $NAME"))
        self.assertIn('reclaim_env="-e ST_RECLAIM_DIR=$RECLAIM_ROOT/rank$r"', rank)
        self.assertIn('ST_RELEASE="$(basename "$ENGINE_DIR")" $reclaim_env', rank)
        stop = text[text.index("  stop)"):text.index("exit 0 ;;", text.index("  stop)"))]
        self.assertLess(stop.index("docker rm -f $NAME"), stop.index("st-reclaim-broker.sh stop-all $RECLAIM_ROOT"))


if __name__ == "__main__":
    unittest.main()
