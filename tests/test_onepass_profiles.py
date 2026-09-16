"""Two named workloads, and a judge that will not read one against the other.

The D17 probe reserves the live door for its whole run and answers 409 to everything else, so the
routine measurement wants to be cheap. It could not simply BE made cheap: `bench/st_bracket.sh` reuses
the probe's warm sample as a bracket's base arm, so cutting the probe alone would have left base and
candidate measuring different things -- and nothing would have said so, because `fixed_concurrency_tokens`
was not part of the recorded workload at all. So the workload is named, the name is in the record, and
records of different names are not each other's baseline.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import measurement_contract as contract     # noqa: E402
import st_judge                             # noqa: E402


class ProfileTests(unittest.TestCase):
    def test_the_default_is_the_cheap_one(self):
        default = contract.profile("default")
        self.assertEqual(default["ctx"], [2000, 32000])       # no 128K
        self.assertEqual(default["fixed_concurrency_tokens"], 0)   # no C=N arm
        self.assertEqual(contract.DEFAULT_PROFILE, "default")

    def test_extended_is_the_full_set(self):
        extended = contract.profile("extended")
        self.assertEqual(extended["ctx"], [2000, 32000, 128000])
        self.assertEqual(extended["fixed_concurrency_tokens"], 1024)

    def test_an_unknown_name_is_a_refusal_not_a_default(self):
        with self.assertRaises(ValueError) as caught:
            contract.profile("cheap")
        self.assertIn("default", str(caught.exception))
        self.assertIn("extended", str(caught.exception))

    def test_a_name_round_trips_and_anything_else_is_custom(self):
        for name in contract.PROFILES:
            self.assertEqual(contract.profile_of(contract.profile(name)), name)
        tweaked = dict(contract.profile("default"), seed=8)
        self.assertEqual(contract.profile_of(tweaked), "custom")

    def test_the_concurrency_arm_is_part_of_the_identity(self):
        """It decides whether the run measures C=N at all; two records that differ in it differ."""
        self.assertIn("fixed_concurrency_tokens", contract.DEFAULTS)
        one = contract.workload(dict(contract.profile("default"), fixed_concurrency_tokens=1024))
        self.assertNotEqual(one, contract.profile("default"))
        self.assertEqual(contract.profile_of(one), "custom")

    def test_the_record_carries_the_name(self):
        meta = contract.metadata(contract.profile("extended"))
        self.assertEqual(meta["workload_profile"], "extended")
        self.assertEqual(meta["workload"], contract.profile("extended"))
        self.assertEqual(meta["harness"], contract.HARNESS)

    def test_the_harness_moved_with_the_meaning(self):
        """A measurement that changed is a new generation; the number is how the ledger says so."""
        self.assertGreaterEqual(contract.HARNESS, 46)


class JudgeTests(unittest.TestCase):
    def record(self, **over):
        row = {"engine": "st", "arm_sha": "a" * 40, "run_index": 2, "boot_id": "b1",
               "workload_profile": "default", "quality": {"ok": 9, "total": 9},
               "korean": {"dirty": 0, "n": 5}, "decode": {"windows_med": 12.0}, "traffic": {"issues": []}}
        row.update(over)
        return row

    def test_two_records_of_one_profile_are_each_other_s_baseline(self):
        self.assertTrue(st_judge.comparable(self.record(), self.record(boot_id="b2")))

    def test_two_profiles_are_not(self):
        self.assertFalse(st_judge.comparable(self.record(), self.record(workload_profile="extended")))

    def test_a_record_from_before_the_name_pairs_with_nothing(self):
        old = self.record()
        old.pop("workload_profile")
        self.assertFalse(st_judge.comparable(old, old))
        self.assertFalse(st_judge.comparable(self.record(), old))

    def test_counting_samples_can_be_asked_for_one_profile(self):
        rows = [self.record(boot_id="b1"), self.record(boot_id="b2", workload_profile="extended")]
        self.assertEqual(len(st_judge.samples(rows, "a" * 40, profile="default")), 1)
        self.assertEqual(len(st_judge.samples(rows, "a" * 40, profile="extended")), 1)
        self.assertEqual(len(st_judge.samples(rows, "a" * 40)), 2)      # unasked: as before harness 46

    def test_a_verdict_drops_a_base_that_measured_another_workload(self):
        rows = [self.record(arm_sha="c" * 40, boot_id="c1"),
                self.record(arm_sha="b" * 40, boot_id="b1", workload_profile="extended")]
        out = st_judge.judge(rows, "c" * 40, "b" * 40)
        self.assertEqual(out["base_summary"]["n"], 0, "the base measured another workload")
        self.assertEqual(out["cand_summary"]["n"], 1)
        self.assertIn("NO BASE", out["verdict"])


class WiringTests(unittest.TestCase):
    def test_onepass_takes_the_profile_and_lets_an_explicit_flag_win(self):
        source = (ROOT / "bench/onepass.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--profile"', source)
        self.assertIn('choices=sorted(contract.PROFILES)', source)
        # the profile supplies the defaults; QUALITY_CTX and the flag still override
        self.assertIn('chosen = contract.profile(os.environ.get("ONEPASS_PROFILE"', source)
        self.assertIn('os.environ.get("QUALITY_CTX", ",".join(map(str, chosen["ctx"])))', source)
        self.assertIn('str(chosen["fixed_concurrency_tokens"])', source)

    def test_the_bracket_only_reuses_a_base_of_its_own_profile(self):
        source = (ROOT / "bench/st_bracket.sh").read_text(encoding="utf-8")
        line = source[source.index('have=$(python3 "$REPO/bench/st_judge.py" samples'):]
        self.assertIn('--profile "${ONEPASS_PROFILE:-default}"', line[:400])


if __name__ == "__main__":
    unittest.main()
