"""The test boot mode: a real engine on one box, without the fleet (45차 §92).

`--local` has always run four ranks as four threads, but only on the reference lanes -- torch arithmetic that
proves the plumbing and never the kernels. So every judgement about what production actually runs needed the
whole fleet, which means a window, which means downtime. `--test` is `--local` on the lanes production runs.

What it must not do is be the thing that takes production down. GB10 has one pool for host and device,
earlyoom's floor is absolute and the engine is a preferred kill target: on 2026-09-11 a smoke test beside
production killed the fleet's worker rather than itself.
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

    def test_it_says_it_is_not_the_fleet(self):
        """A local boot holds no lease, and somebody reading the log has to be able to tell."""
        _, source = self.parse()
        self.assertIn("This is not the fleet and", source)
        self.assertIn("holds no lease", source)


if __name__ == "__main__":
    unittest.main()
