"""A step that never ends is bounded from beside the loop (45차, 2026-09-12 22:31).

The watch runs here with an injected clock, ring dump and kill; the serving loop's use of
it, the boot's wiring, the launcher's heartbeat bound and the bracket's four-rank
forensics are pinned in the sources, the way the fleet's other lockstep contracts are.
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.base.stall import StepWatch                                # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class StepWatchTests(unittest.TestCase):
    def watch(self, tmp, **kw):
        clock, said, killed, dumped = Clock(), [], [], []
        w = StepWatch(2, note_s=60, trap_s=300, notes_dir=tmp, dump=lambda: dumped.append(1), clock=clock,
                      kill=lambda: killed.append(1), say=lambda *a, **k: said.append(a[0]), **kw)
        return w, clock, said, killed, dumped

    def test_a_note_after_a_minute_and_a_trap_after_five_with_the_ring_written_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, clock, said, killed, dumped = self.watch(tmp)
            self.assertIsNone(w.check(), "nothing entered, nothing to time")
            w.enter("step 2299")
            clock.t += 59
            self.assertIsNone(w.check())
            clock.t += 1
            self.assertEqual(w.check(), "note")
            self.assertIn("STALL rank=2 step 2299 has not returned after 60s", said[-1])
            self.assertIn("a trap follows at 300s", said[-1])
            clock.t += 100
            self.assertIsNone(w.check(), "one note per stall")
            self.assertFalse(killed)
            clock.t += 140
            self.assertEqual(w.check(), "trap")
            self.assertEqual((dumped, killed), ([1], [1]), "the ring is written, then the rank is killed")
            note = json.loads(Path(w.trapped["path"]).read_text())
            self.assertEqual((note["rank"], note["what"], note["waited_s"]), (2, "step 2299", 300.0))
            self.assertEqual(Path(w.trapped["path"]).parent, Path(tmp))
            self.assertIn("STALL TRAP rank=2 step 2299", said[-1])

    def test_leaving_and_re_entering_restart_the_clock_and_the_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, clock, said, killed, dumped = self.watch(tmp)
            w.enter("step 1")
            clock.t += 70
            self.assertEqual(w.check(), "note")
            w.leave()
            clock.t += 1000
            self.assertIsNone(w.check(), "nothing in flight, nothing to trap")
            w.enter("step 2")
            clock.t += 61
            self.assertEqual(w.check(), "note")
            w.enter("after the step")
            clock.t += 30
            self.assertIsNone(w.check(), "a new phase restarts the clock")
            self.assertEqual(w.notes, 2)
            self.assertFalse(killed)

    def test_a_dump_that_fails_does_not_stop_the_kill_and_zero_disarms(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock, killed = Clock(), []
            w = StepWatch(0, note_s=0, trap_s=10, notes_dir=tmp, clock=clock, say=lambda *a, **k: None,
                          dump=lambda: (_ for _ in ()).throw(OSError("no ring")), kill=lambda: killed.append(1))
            w.enter("step 1")
            clock.t += 10
            self.assertEqual(w.check(), "trap")
            self.assertEqual(killed, [1])
            off = StepWatch(0, note_s=0, trap_s=0, clock=clock, kill=lambda: killed.append(2), say=lambda *a, **k: None)
            self.assertFalse(off.armed)
            off.enter("step 1")
            clock.t += 10_000
            self.assertIsNone(off.check())
            self.assertEqual(killed, [1])
            off.start()
            self.assertIsNone(off._thread, "a disarmed watch starts no thread")
        with self.assertRaises(ValueError):
            StepWatch(0, note_s=-1)

    def test_the_thread_checks_on_its_own_and_stops(self):
        clock, killed = Clock(), []
        w = StepWatch(1, note_s=0, trap_s=5, clock=clock, kill=lambda: killed.append(1),
                      say=lambda *a, **k: None, period_s=0.01)
        w.enter("step 7")
        clock.t += 5
        w.start()
        for _ in range(300):
            if killed:
                break
            time.sleep(0.01)
        self.assertEqual(killed, [1])
        w.stop()


class ContractTests(unittest.TestCase):
    def test_the_loop_says_what_it_is_doing_and_the_boot_hands_the_watch_the_ring(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        once = serve[serve.index("    def once(self) -> bool:"):serve.index("    def _serve_http(self):")]
        self.assertLess(once.index('self.watch.enter("the step\'s broadcast")'), once.index("self.comm.broadcast_object("))
        self.assertLess(once.index('self.watch.enter("settling transfers")'), once.index("self._settle()"))
        self.assertLess(once.index('self.watch.enter(f"step {self.runner.steps + 1}")'), once.index("step = self.runner.step()"))
        self.assertIn("finally:\n            self.watch.leave()", once)
        loop = serve[serve.index("    def loop(self"):]
        self.assertIn("self.watch.start()", loop)
        self.assertIn("self.watch.stop()", loop)
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn("step_watch=StepWatch(comm.rank, notes_dir=a.dump_dir, dump=dump.write_now)", boot)

    def test_the_launcher_bounds_a_hung_watchdog_to_five_minutes(self):
        launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()
        self.assertIn("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${ST_NCCL_HEARTBEAT_S:-300}", launcher)
        self.assertNotIn("HEARTBEAT_TIMEOUT_SEC=7200", launcher)

    def test_the_bracket_judges_all_four_ranks_and_keeps_their_logs_before_the_stop(self):
        bracket = (ROOT / "bench/st_bracket.sh").read_text()
        wait = bracket[bracket.index("wait_door() {"):bracket.index("boot_arm() {")]
        self.assertIn("dead=$(dead_ranks)", wait)
        self.assertLess(wait.index('forensics "$DUMPS"'), wait.index("return 1"), "the logs are pulled before the boot is declared dead")
        self.assertNotIn("docker inspect --format '{{.State.Running}}' st-glm53", wait, "rank 0 alone no longer decides")
        self.assertIn('node_sh "$ip" "docker logs --tail=400 st-glm53"', bracket)
        self.assertIn("wait_door || { stop_arm; return 1; }", bracket, "and the stop, which erases them, comes after")
        self.assertIn("NODES=(${ST_NODES:-10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4})", bracket)


if __name__ == "__main__":
    unittest.main()
