"""The test boot mode: production's path on one box (45차 §92, §94).

`--local` ran four ranks as four threads but only on the reference lanes -- torch arithmetic that proves the
plumbing and never the kernels -- and it captured nothing, so its decode was eager. Both make it a different
engine, and a different engine's numbers answer about nothing.

`--test` keeps the path and drops the rest:

  same   the served lanes, and decode replays captured graphs as production's does.
  off    what a fleet boot must qualify before its door opens: the vision tower, the grammars, the
         parked-conversation tier, the calibration sums.
  fast   one captured decode width instead of max_seqs of them. Capture is three quarters of a fleet boot,
         and it is paid per width.
  open   every step timed instead of one in sixty-four.

What it is NOT is a speed measurement: four ranks are four threads on ONE GPU, so the device does four ranks'
arithmetic. D17 -- a change that claims speed is not finished until the fleet has measured it -- is unchanged
by this mode, and the printed table says so.

The memory floor is a side condition: GB10 has one pool for host and device, earlyoom's floor is absolute and
the engine is a preferred kill target -- on 2026-09-11 a smoke test beside production killed the fleet's
worker rather than itself.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def boot():
    from engine.profiles.glm53 import boot as module
    return module


class MemoryGuardTests(unittest.TestCase):
    def test_it_reports_what_is_left_after_the_kv_this_boot_declares(self):
        module = boot()
        whole = module.memory_left(0.0)
        self.assertGreater(whole, 0.0)
        self.assertAlmostEqual(module.memory_left(2.0), whole - 2.0, places=3)

    def test_it_refuses_rather_than_guesses_and_says_by_how_much(self):
        module = boot()
        with self.assertRaises(SystemExit) as raised:
            module.guard_test_memory(1.0, floor=10_000.0)
        message = str(raised.exception)
        self.assertIn("--test refuses", message)
        self.assertIn("10000.0", message)                  # the floor it was measured against
        self.assertIn("--layers", message)                 # and the lever that makes it fit

    def test_a_boot_with_room_is_allowed(self):
        boot().guard_test_memory(0.0, floor=0.0)

    def test_the_floor_is_not_zero(self):
        """A floor of zero is not a floor: earlyoom kills before MemAvailable reaches it."""
        self.assertGreaterEqual(boot().TEST_FLOOR_GIB, 8.0)


class MeasurementTests(unittest.TestCase):
    def test_it_captures_what_production_captures(self):
        """An eager decode is a different engine. The graphs are the path being tested."""
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        armed = source.split("def arm_test_measurement", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("engine.capture_decode(TEST_WIDTHS)", armed)
        self.assertEqual(boot().TEST_WIDTHS, 1)            # one width, not max_seqs of them

    def test_it_times_every_step(self):
        self.assertEqual(boot().TEST_CLOCK_EVERY, 1)
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn("StageClock(every=TEST_CLOCK_EVERY", source)

    def test_the_table_says_whose_time_it_is(self):
        """Four ranks on one GPU: the path is production's and the milliseconds are not."""
        from engine.base.stage_clock import StageClock
        module = boot()
        self.assertIn("nothing timed", module.stage_table(None, 0))
        clock = StageClock()
        clock.totals, clock.samples = {"forward": 0.476, "observe": 0.160, "propose": 0.113}, 10
        table = module.stage_table(clock, 10)
        self.assertIn("forward", table.splitlines()[1])     # biggest stage first
        self.assertIn("not the fleet", table)
        self.assertIn("63.6%", table)

    def test_a_boot_without_a_drafter_times_nothing_rather_than_crashing(self):
        """The stages belong to the async decode pipeline, and no drafter means no pipeline."""
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        armed = source.split("def arm_test_measurement", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("if pipeline is None", armed)


class FlagTests(unittest.TestCase):
    def parse(self, *argv):
        import argparse
        import contextlib
        import io
        module = boot()
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn('ap.add_argument("--test"', source)
        self.assertIn('choices=("reference", "served")', source)
        return module, source

    def test_test_implies_local_and_the_served_lanes(self):
        _, source = self.parse()
        self.assertIn('a.local, a.lanes = True, "served"', source)

    def test_the_served_lane_is_guarded_and_the_reference_one_is_not(self):
        """The reference lanes hold no weights worth guarding; the served ones bring the model."""
        _, source = self.parse()
        served = source.split("def local(a) -> int:", 1)[1].split("def ", 1)[0]
        self.assertIn('if a.lanes == "served":', served)
        self.assertIn("guard_test_memory(a.kv_gib)", served)
        self.assertIn("lane_tables.served() if a.lanes == \"served\" else lane_tables.reference()", served)

    def test_it_says_what_is_production_s_and_what_is_not(self):
        """It holds no lease, and four ranks share one GPU -- somebody reading the log has to be told both,
        or they will read a step/s off a box that does four ranks' arithmetic on one device."""
        _, source = self.parse()
        self.assertIn("it holds no lease", source)
        self.assertIn("the PATH is ", source)
        self.assertIn("TIMES are this box's", source)
        self.assertIn("D17 still wants the fleet for a speed claim", source)


if __name__ == "__main__":
    unittest.main()
