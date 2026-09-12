"""Verbosity evidence keeps censored runs and unknown reasoning counts visible."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from onepass_recording import Run, generation_summary
from onepass import kda_state_rounding


class GenerationSummaryTests(unittest.TestCase):
    def row(self, **changes):
        return dict(dict(phase='measure-c1', ctx=32768, question='proof', concurrency=1,
                         max_tokens=1000, reasoning_budget=800, completion_tokens=900,
                         reasoning_tokens=800, finish_reason='stop'), **changes)

    def test_budget_hit_remains_visible_even_when_overall_reason_is_stop(self):
        result, = generation_summary([self.row(), self.row(completion_tokens=1000, reasoning_tokens=None, finish_reason='length')])
        self.assertEqual(result['completion_tokens'], dict(n=2, mean=950., p50=900, p95=1000))
        self.assertEqual(result['reasoning_tokens']['n'], 1)
        self.assertEqual(result['reasoning_budget_known'], 1)
        self.assertEqual(result['reasoning_budget_hits'], 1)
        self.assertEqual(result['reasoning_budget_hit_rate'], 1.)
        self.assertEqual(result['stop_rate'], .5)
        self.assertEqual(result['length_limit_rate'], .5)

    def test_measurement_groups_exclude_fixed_and_preparation_and_retain_missing(self):
        rows = [self.row(phase=p) for p in ('prepare-c1', 'diagnostic-c1')]
        rows += [self.row(fixed_decode=True), self.row(min_tokens=1000)]
        rows += [self.row(), self.row(phase='measure-c4', concurrency=4),
                 self.row(ctx=131072, completion_tokens=None, reasoning_tokens=None, finish_reason=None)]
        result = generation_summary(rows)
        self.assertEqual(len(result), 3)
        missing = next(r for r in result if r['ctx'] == 131072)
        self.assertEqual(missing['completion_tokens']['n'], 0)
        self.assertIsNone(missing['reasoning_budget_hit_rate'])
        self.assertEqual(missing['finish_reasons'], {'unknown': 1})

    def test_all_windows_survive_begin_resets_and_partial_finish(self):
        with TemporaryDirectory() as root, patch('urllib.request.urlopen', side_effect=OSError('offline')):
            run = Run({}, Path(root)/'runs.jsonl', 'http://localhost/v1/chat/completions')
            for phase in ('measure-c1', 'measure-c4', 'diagnostic-c4'):
                run.begin(phase)
                run.request(self.row(), 'answer', [])
            run.finish('interrupted')
            record = json.loads((run.path/'record.json').read_text())
            self.assertEqual([r['phase'] for r in record['generation_summary']], ['measure-c1', 'measure-c4'])
            self.assertEqual(record['recording']['status'], 'incomplete')

    def test_rounding_is_attested_not_inferred_from_fp16(self):
        self.assertEqual(kda_state_rounding('st:lane_info{kda_state_dtype="fp16",kda_state_rounding="sr-philox10-v1"} 1'), 'sr-philox10-v1')
        self.assertIsNone(kda_state_rounding('st:lane_info{kda_state_dtype="fp16"} 1'))


if __name__ == '__main__':
    unittest.main()
