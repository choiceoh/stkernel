"""The qualify soak's bookkeeping, its lane in the kernel check, and that it imports only what the single-GPU lane ships.

The run itself needs a GB10 (probes/engine_qwen38_qualify_soak.py); what is held here is what the record means:
failures are counted and the first few kept with the repeat they came at, and the calls that held are grouped by their
result, so ONE distinct result is what a steady card gives.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes import engine_qwen38_qualify_soak as probe  # noqa: E402

# lanes.qualify's two forms: (max, rms) a key, and the skinny GEMV's one number a shape
HELD = {"qsa_norm_rope": {"norm_rope_4x128": (0.0069, 0.0023)}, "skinny_gemv": {"513x2560": 0.002814}}
MOVED = {"qsa_norm_rope": {"norm_rope_4x128": (0.0071, 0.0023)}, "skinny_gemv": {"513x2560": 0.002814}}


def scripted(*outcomes):
    """A qualify that gives `outcomes` in turn: a dict is returned, a string is raised as the boot's RuntimeError."""
    left = list(outcomes)

    def qualify():
        outcome = left.pop(0)
        if isinstance(outcome, str):
            raise RuntimeError(outcome)
        return outcome
    return qualify


class SoakTests(unittest.TestCase):
    def test_a_steady_card_gives_one_result_and_no_failure(self):
        record = probe.soak(scripted(*[HELD] * 6), 6)
        self.assertEqual((record["repeats"], record["failures"], record["errors"], record["distinct_results"]), (6, 0, [], 1))
        self.assertEqual(record["results"], [{"calls": 6, "worst": {"qsa_norm_rope": {"norm_rope_4x128": [0.0069, 0.0023]},
                                                                    "skinny_gemv": {"513x2560": 0.002814}}}])

    def test_failures_are_counted_and_the_first_few_kept_with_their_repeat(self):
        record = probe.soak(scripted(HELD, "drift A", HELD, "drift B", "drift C", HELD), 6, kept=2)
        self.assertEqual(record["failures"], 3)
        self.assertEqual(record["errors"], [{"repeat": 1, "error": "drift A"}, {"repeat": 3, "error": "drift B"}])
        self.assertEqual(record["distinct_results"], 1)

    def test_a_launch_that_moved_within_the_band_is_a_second_result(self):
        record = probe.soak(scripted(HELD, HELD, MOVED, HELD), 4)
        self.assertEqual((record["failures"], record["distinct_results"]), (0, 2))
        self.assertEqual([r["calls"] for r in record["results"]], [3, 1])      # most frequent first

    def test_only_the_boots_own_error_is_a_failure(self):
        def qualify():
            raise ValueError("not the hold's error: a bug in the probe or the lane must stop the run")
        with self.assertRaises(ValueError):
            probe.soak(qualify, 3)


class LaneTests(unittest.TestCase):
    def test_the_kernel_check_runs_it_with_the_repeats_after_the_colon(self):
        source = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("args.lanes == 'qwen38_qualify_soak' or args.lanes.startswith('qwen38_qualify_soak:')", source)
        self.assertIn("repeats=int((args.lanes.split(':')[1:] or [REPEATS])[0])", source)
        self.assertIn("--lanes qwen38_qualify_soak:2000", probe.__doc__)       # the runner line the docstring promises

    def test_it_imports_only_what_the_lane_ships(self):
        tree = ast.parse((ROOT / "probes/engine_qwen38_qualify_soak.py").read_text(encoding="utf-8"))
        shipped = ("engine", "probes", "tests")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertTrue(node.module.split(".")[0] in shipped + ("pathlib",), node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], ("json", "sys", "time", "torch"), alias.name)


if __name__ == "__main__":
    unittest.main()
