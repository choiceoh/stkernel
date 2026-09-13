"""Measured draft-prefix probabilities and incomplete-observation rejection; no GPU."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import step_acceptance as acceptance
import step_peek as peek
import step_sim as sim


def scrape(hist, k=6, engine="st"):
    # Match the server's sparse zero-bucket exposition, including the empty baseline.
    out = {f'{acceptance.HIST}{{engine="{engine}",accepted="{i}"}}': count
           for i, count in enumerate(hist) if count or i <= 1}
    out[f'{acceptance.ACCEPTED}{{engine="{engine}"}}'] = sum(i*n for i, n in enumerate(hist))
    out[f'{acceptance.DRAFTED}{{engine="{engine}"}}'] = k*sum(hist)
    out[f'{acceptance.LANE}{{engine="{engine}",spec_k="{k}"}}'] = 1
    return out


class AcceptanceTest(unittest.TestCase):
    def test_sixth_cumulative_and_conditional_have_different_denominators(self):
        a, b = scrape([0]*7), scrape([20, 10, 10, 10, 10, 20, 20])
        data = acceptance.profile(acceptance.histogram(a, b))
        tail = data["positions"][-1]
        self.assertEqual((data["rows"], tail["accepted_rows"], tail["reached_rows"]), (100, 20, 40))
        self.assertEqual(tail["cumulative"], .2)
        self.assertEqual(tail["conditional"], .5)
        self.assertAlmostEqual(data["raw_acceptance"], 320/600)
        self.assertIn("20.0%", acceptance.format_profile(data))

    def test_sparse_new_buckets_count_from_zero_and_keep_unobserved_tail(self):
        a, b = scrape([0]*7), scrape([10, 20, 0, 0, 0, 0, 0])
        data = acceptance.profile(acceptance.histogram(a, b))
        self.assertEqual(data["histogram"], [10, 20, 0, 0, 0, 0, 0])
        self.assertEqual(data["positions"][-1]["cumulative"], 0)
        self.assertIsNone(data["positions"][-1]["conditional"])
        c = scrape([10, 20, 0, 0, 0, 0, 5])
        self.assertEqual(peek.acc_hist_from_scrapes(b, c), [0, 0, 0, 0, 0, 0, 5])

    def test_missing_empty_and_missing_k_are_unavailable(self):
        a = scrape([0]*7)
        self.assertIsNone(peek.acc_hist_from_scrapes({}, a))
        self.assertIsNone(peek.acc_hist_from_scrapes(a, a))
        b = scrape([0, 0, 0, 0, 0, 0, 5])
        for sample in (a, b):
            del sample[f'{acceptance.LANE}{{engine="st",spec_k="6"}}']
        with self.assertRaisesRegex(ValueError, "spec_k is missing"):
            acceptance.profile(acceptance.histogram(a, b))
        self.assertEqual(acceptance.profile(acceptance.histogram(a, b, 6))["positions"][-1]["cumulative"], 1)

    def test_histogram_must_cover_aggregate_accepted_and_rejected_rows(self):
        a = scrape([0]*7)
        for name, value in ((acceptance.ACCEPTED, 50), (acceptance.DRAFTED, 60)):
            with self.subTest(counter=name):
                b = dict(a)
                b[f'{name}{{engine="st"}}'] = value
                with self.assertRaisesRegex(ValueError, "does not cover"):
                    acceptance.histogram(a, b)
                self.assertIsNone(peek.acc_hist_from_scrapes(a, b))

    def test_reset_or_disappearance_rejects_the_whole_interval(self):
        a = scrape([10, 0, 0, 0, 0, 0, 3])
        for b in (scrape([9, 0, 0, 0, 0, 0, 4]), scrape([30, 0, 0, 0, 0, 0, 0])):
            with self.subTest(after=b):
                self.assertIsNone(peek.acc_hist_from_scrapes(a, b))
        samples = [scrape([i]*7) for i in (1, 0, 2)]
        self.assertIsNotNone(peek.acc_hist_from_scrapes(samples[0], samples[-1]))
        self.assertIsNone(peek.acc_hist_from_samples(samples))

    def test_lane_k_or_group_changes_are_rejected(self):
        a = scrape([0]*7)
        for b in (scrape([1]*7, k=5), scrape([1]*7, engine="other")):
            self.assertIsNone(peek.acc_hist_from_scrapes(a, b))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            acceptance.histogram(a, scrape([1]*7), 5)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            acceptance.histogram(scrape([0]*8), scrape([0]*7 + [1]))

    def test_labels_are_canonical_and_groups_pool_without_overwrite(self):
        a = {**scrape([0]*7), **scrape([0]*7, engine="other")}
        b = {**scrape([10, 0, 0, 0, 0, 0, 5]), **scrape([20, 0, 0, 0, 0, 0, 6], engine="other")}
        key = f'{acceptance.HIST}{{engine="st",accepted="6"}}'
        b[f'{acceptance.HIST}{{accepted="6",engine="st"}}'] = b.pop(key)
        self.assertEqual(peek.acc_hist_from_scrapes(a, b), [30, 0, 0, 0, 0, 0, 11])
        # A new group cannot be mistaken for a newly materialized zero bucket.
        self.assertIsNone(peek.acc_hist_from_scrapes(scrape([0]*7), b))

    def test_invalid_counts_or_positions_are_not_measurements(self):
        a = scrape([0]*7)
        key = f'{acceptance.HIST}{{engine="st",accepted="0"}}'
        for value in (-1, 1.5, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(peek.acc_hist_from_scrapes(a, {**a, key: value}))
        b = {**a, f'{acceptance.HIST}{{engine="st",accepted="bad"}}': 1}
        self.assertIsNone(peek.acc_hist_from_scrapes(a, b))

    def test_saved_peek_replay_and_simulator_keep_the_same_distribution(self):
        a, b = scrape([0]*7), scrape([20, 10, 10, 10, 10, 20, 20])
        samples = [peek.track(s) for s in (a, b)]
        summary = peek.summarize(list(enumerate(samples)))
        self.assertEqual(summary["acceptance"]["positions"][-1]["cumulative"], .2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "peek.jsonl"
            path.write_text("\n".join(json.dumps({"monotonic": i, "series": s}) for i, s in enumerate(samples)))
            self.assertEqual(sim.acc_hist_from_peek(path, 6), summary["acceptance"]["histogram"])
            self.assertIsNone(sim.acc_hist_from_peek(path, 5))
            result = subprocess.run([sys.executable, str(ROOT / "bench/storacle.py"), "acceptance", str(path), "--json"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), summary["acceptance"])

    def test_metrics_text_cli_reports_unavailable_without_a_fake_zero_rate(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / name for name in ("before.txt", "after.txt")]
            a, b = scrape([0]*7), scrape([1]*7)
            for path, sample in zip(paths, (a, b)):
                path.write_text("\n".join(f"{key} {value}" for key, value in sample.items()))
            command = [sys.executable, str(ROOT / "bench/storacle.py"), "acceptance",
                       "--before", str(paths[0]), "--after", str(paths[1]), "--json"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["rows"], 7)
            paths[1].write_text("")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(set(json.loads(result.stdout)), {"error"})


if __name__ == "__main__":
    unittest.main()
