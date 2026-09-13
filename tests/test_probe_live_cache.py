"""A D17 probe measures on the live door without emptying production's prefix cache (45차, 2026-09-13).

The probe used to POST /v1/prefix/reset before its run: every boundary in memory and every slot of
the prefix tier went, after every deploy. It needs no reset -- its requests carry unique cache salts,
so they cannot hit what production cached, and they say retain false, so the boundaries they make
leave with their rows and never reach the tier (engine: tests/test_engine_transient_prefix.py).
Its run is labelled cold=live, and the judge keeps it in the warm column as it kept cold=reset.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import st_judge                                                         # noqa: E402

SHA = "a" * 40


def record(cold, run_index=1):
    return dict(engine="st", arm_sha=SHA, name="d17", run_index=run_index, cold=cold, boot_id="live|started",
                harness=45, recording=dict(status="complete"), quality=dict(ok=9, total=9),
                korean=dict(dirty=0, n=5), traffic=dict(issues=[]), decode=dict(windows_med=12.0))


class JudgeTests(unittest.TestCase):
    def test_a_live_run_is_warm_and_never_cold_like_the_reset_runs_before_it(self):
        for cold in ("live", "reset"):
            with self.subTest(cold=cold):
                self.assertTrue(st_judge.warm(record(cold)))
                self.assertEqual(st_judge.colds([record(cold)], SHA), [])
        self.assertFalse(st_judge.warm(record("boot")), "a boot's run 1 is still the cold column")
        self.assertEqual(len(st_judge.colds([record("boot")], SHA)), 1)


class SourceTests(unittest.TestCase):
    def test_the_probe_labels_its_run_live_and_resets_nothing(self):
        bracket = (ROOT / "bench/st_bracket.sh").read_text()
        probe = bracket[bracket.index("probe() {"):]
        probe = probe[:probe.index("\n}\n")]
        self.assertIn('ST_BRACKET_COLD=live measure "$run"', probe)
        self.assertNotIn("reset_prefix", probe)
        self.assertNotIn("ST_BRACKET_COLD=reset", probe)
        leg = bracket[bracket.index("leg() {"):]
        self.assertIn("reset_prefix", leg[:leg.index("\n}\n")], "a bracket's own boot still resets between its runs")


if __name__ == "__main__":
    unittest.main()
