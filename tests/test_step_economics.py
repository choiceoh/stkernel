"""Observed prefix yield and marginal draft value, without invented tail rates."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import step_acceptance as acceptance


def example():
    # 100 synthetic rows: E[X]=3.2; sixth survival=20%, fifth=40%.
    return dict(k=6, histogram=[20, 10, 10, 10, 10, 20, 20])


def write_peek(path):
    records = []
    for bins in ([0]*7, example()['histogram']):
        metrics = {f'{acceptance.HIST}{{accepted="{i}"}}': count for i, count in enumerate(bins)}
        metrics[acceptance.ACCEPTED] = sum(i*n for i, n in enumerate(bins))
        metrics[acceptance.DRAFTED] = 6 * sum(bins)
        metrics[f'{acceptance.LANE}{{spec_k="6"}}'] = 1
        records.append(dict(series=metrics))
    path.write_text('\n'.join(json.dumps(record) for record in records))


class EconomicsTests(unittest.TestCase):
    def test_sixth_twenty_percent_has_five_percent_step_overhead_budget(self):
        report = acceptance.economics(example())
        fifth, sixth = report['rows'][4:6]
        self.assertAlmostEqual(fifth['tokens_per_row'], 4)
        self.assertAlmostEqual(sixth['tokens_per_row'], 4.2)
        self.assertAlmostEqual(sixth['max_relative_step_increase'], .05)
        self.assertAlmostEqual(4 / 50, 4.2 / 52.5)
        self.assertGreater(4.2 / 52, 4 / 50)
        self.assertLess(4.2 / 53, 4 / 50)
        self.assertEqual(report['next_position']['cumulative_range'], [0, .2])
        self.assertAlmostEqual(report['next_position']['max_relative_step_increase_range'][1], .2/4.2)

    def test_observed_prefixes_are_truncated_by_depth_not_refitted_to_average(self):
        for k, expected in ((0, 1), (3, 3.1), (5, 4), (6, 4.2)):
            with self.subTest(k=k):
                lo, hi = acceptance.yield_bounds(k, observed=example())
                self.assertAlmostEqual(lo, expected)
                self.assertEqual(lo, hi)

    def test_unobserved_tail_remains_a_range_without_geometric_extrapolation(self):
        for k, upper in ((7, 4.4), (9, 4.8)):
            lo, hi = acceptance.yield_bounds(k, observed=example())
            self.assertAlmostEqual(lo, 4.2)
            self.assertAlmostEqual(hi, upper)
        self.assertEqual(acceptance.yield_bounds(10, observed=dict(k=6, histogram=[10, 0, 0, 0, 0, 0, 0])), (1, 1))

    def test_a_raw_average_does_not_identify_another_depth(self):
        self.assertEqual(acceptance.yield_bounds(6, reference_k=6, raw_acceptance=.5), (4, 4))
        self.assertEqual(acceptance.yield_bounds(3, reference_k=6, raw_acceptance=.5), (2.5, 4))
        self.assertEqual(acceptance.yield_bounds(7, reference_k=6, raw_acceptance=.5), (4, 4.5))
        self.assertEqual(acceptance.yield_bounds(3, reference_k=0), (1, 4))

    def test_invalid_histograms_and_depths_are_not_yield_estimates(self):
        for data in (dict(k=6, histogram=[1]), dict(k=0, histogram=[1]),
                     dict(k=1, histogram=[1.5, 2]), dict(k=1, histogram=[1, -1]),
                     dict(k=1, histogram=[0, 0])):
            with self.subTest(data=data), self.assertRaises(ValueError):
                acceptance.yield_bounds(6, observed=data)
        for k in (-1, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                acceptance.yield_bounds(k)

    def test_cli_reports_economics_from_validated_scrapes_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'peek.jsonl'
            write_peek(path)
            command = [sys.executable, str(ROOT / 'bench/storacle.py'), 'acceptance', str(path), '--economics', '--json']
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(run.stdout)
            self.assertEqual(data['rows'], 100)
            self.assertAlmostEqual(data['economics']['rows'][-1]['max_relative_step_increase'], .05)
            path.write_text(json.dumps(dict(summary=data)))
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertIn('error', json.loads(run.stdout))


if __name__ == '__main__':
    unittest.main()
