"""Short fleet screening with the real recorder and a CPU-only stream fixture."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import st_screen as screen
from onepass_recording import CURRENT, Run


def rank_report(concurrency, *, changed=False):
    return dict(ranks=[dict(rank=i, complete=True, diagnostic=False,
        preparation_changed=changed,
        preparation_after={'observers': ['triton.runtime.jit._do_compile']},
        rows=[dict(kind='host_step', phase='decode', rows=list(range(concurrency)),
                   duration_us=10000.0)]) for i in range(4)])


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(CURRENT.set, None)
        self.directory = Path(self.temporary.name)
        self.finished = 0
        self.calls = []
        self.changed = False
        self.failure = None
        self.lock = threading.Lock()
        self.bd = SimpleNamespace(URL='http://fixture/v1/chat/completions',
                                  METRICS='http://fixture/metrics', _parse_spec_metrics=lambda _: {})
        self.bracket = SimpleNamespace(_spec_delta=lambda *_: (0.5, 0.5))
        self.item = dict(ctx=2000, question='screen-ledger', content='fixture', seed=42,
                         min_tokens=128, max_tokens=512, reasoning_budget=256,
                         quality_cases=screen.op.quality.cases(2042)[:1])

    def door(self, request, **kwargs):
        body = json.loads(request.data) if hasattr(request, 'data') and request.data else {}
        if body.get('op') == 'begin':
            self.concurrency = body['concurrency']
        reply = (rank_report(self.concurrency, changed=self.changed)
                 if body.get('op') == 'end' else {'ranks': [{}, {}, {}, {}]})
        if not body:
            reply = dict(schema=1, ranks=4)
        reply['boot_id'] = 'fixture-boot'
        return io.BytesIO(json.dumps(reply).encode())

    def metrics(self, _):
        return (f'vllm:request_success_total {self.finished}\n'
                'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n')

    def ask(self, url, model, content, max_tokens, timing, *, min_tokens, seed,
            reasoning_budget, channel_trace):
        run = CURRENT.get()
        if self.failure and run.phase == 'measure-c4':
            raise RuntimeError(self.failure)
        with self.lock:
            self.finished += 1
            self.calls.append((run.phase, min_tokens, max_tokens))
        timing.update(started_monotonic=10.0, ended_monotonic=12.0, elapsed_s=2.0,
                      ttft_s=0.25, completion_tokens=max_tokens, decode_tok_s=10.0,
                      cached_tokens=0, finish_reason='length')
        events = [{'reasoning_content': 'unfinished reasoning'}]
        channel_trace.append(events)
        run.request(timing, 'unfinished reasoning', events)
        return 'unfinished reasoning', 0.25, 2000, max_tokens, 'length'

    def execute(self):
        record = dict(engine='st', boot_id='fixture-boot', evidence_scope='screen')
        with patch('urllib.request.urlopen', self.door), \
             patch.object(screen.op, '_metrics_text', self.metrics), \
             patch.object(screen.op, 'ask_stream', self.ask):
            run = Run(record, self.directory / 'records.jsonl', self.bd.URL)
            try:
                screen.collect(run, self.item, self.bd, 'fixture', None, self.bracket)
            except RuntimeError as exc:
                run.finish(error=exc)
                raise
            run.finish()
        return record, run.path

    def test_short_screen_records_both_arms_and_quality_misses_without_full_gate(self):
        record, path = self.execute()
        self.assertEqual([r['concurrency'] for r in record['screen']], [1, 4])
        self.assertEqual(record['screen_status'], 'observed')
        self.assertFalse(record['adoption_eligible'])
        self.assertEqual(record['recording']['status'], 'complete')
        self.assertEqual(self.calls.count(('prepare-c1', 64, 64)), 1)
        self.assertEqual(self.calls.count(('prepare-c4', 64, 64)), 4)
        self.assertEqual(self.calls.count(('measure-c1', 128, 512)), 1)
        self.assertEqual(self.calls.count(('measure-c4', 128, 512)), 4)
        self.assertEqual(len(self.calls), 10, 'no second pass, long contexts or profiler replay')
        requests = [r for arm in record['screen'] for r in arm['requests']]
        self.assertTrue(all(not all(q['passed'] for q in r['quality']) for r in requests))
        self.assertEqual(len((path / 'requests.jsonl').read_text().splitlines()), 10)
        self.assertEqual(len((path / 'quality.jsonl').read_text().splitlines()), 5)
        for concurrency in (1, 4):
            self.assertTrue((path / f'measure-c{concurrency}' / 'latency.jsonl').is_file())
            self.assertTrue((path / f'measure-c{concurrency}' / 'latency-summary.json').is_file())
            self.assertTrue(json.loads((path / f'measure-c{concurrency}' / 'latency-summary.json').read_text())['operations'])

    def test_compilation_observation_invalidates_timing_but_does_not_block_screening(self):
        self.changed = True
        record, _ = self.execute()
        self.assertEqual(record['screen_status'], 'timing_unverified')
        self.assertTrue(all(not arm['valid'] and arm['timing_issues'] for arm in record['screen']))
        self.assertTrue(all(not arm['health_errors'] for arm in record['screen']))

    def test_runtime_failure_keeps_partial_record_and_fails(self):
        for failure in ('CUDA out of memory', 'nonfinite state', 'token order mismatch', 'collective failed'):
            with self.subTest(failure=failure):
                self.failure = failure
                with self.assertRaisesRegex(RuntimeError, failure):
                    self.execute()
                record = json.loads((self.directory / 'records.jsonl').read_text().splitlines()[-1])
                self.assertEqual(record['recording']['status'], 'incomplete')
                self.assertEqual(record['recording']['error'], failure)
                self.assertEqual(len(record['screen']), 1, 'C=1 evidence survives a C=4 failure')

    def test_rank_loss_bad_width_incomplete_output_and_nonfinite_timing_remain_failures(self):
        request = dict(completion_tokens=128, finish_reason='stop',
                       ttft_s=0.25, elapsed_s=2.0, decode_tok_s=10.0)
        self.assertEqual(screen.health_errors(rank_report(4), [request] * 4, 4), [])
        self.assertEqual(screen.health_errors(rank_report(1), [request] * 4, 4,
                                              require_width=False), [])
        for report, requests in (
            ({'ranks': rank_report(4)['ranks'][:3]}, [request] * 4),
            (rank_report(1), [request] * 4),
            (rank_report(4), [dict(request, finish_reason=None)] * 4),
            (rank_report(4), [dict(request, decode_tok_s=float('nan'))] * 4),
            (rank_report(4), [dict(request, ttft_s=float('inf'))] * 4),
        ):
            with self.subTest(report=report, requests=requests):
                self.assertTrue(screen.health_errors(report, requests, 4))
        report = copy.deepcopy(rank_report(4))
        report['ranks'][0]['error'] = 'collective failed'
        self.assertTrue(screen.health_errors(report, [request] * 4, 4))


if __name__ == '__main__':
    unittest.main()
