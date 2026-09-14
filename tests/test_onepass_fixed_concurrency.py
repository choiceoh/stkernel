"""bench-dec's C=4/C=1 multiplier inside onepass: fixed lengths, different prompts, its own validity."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import onepass
from onepass_recording import group


class FixedConcurrencyTests(unittest.TestCase):
    def test_four_different_prompts_forced_to_the_same_length(self):
        cq = SimpleNamespace(filler=lambda n, rng: '배경 문단. ' * 3)
        items = onepass.fixed_concurrency_items(7, cq, 1024)
        self.assertEqual(len(items), onepass.FIXED_CONCURRENCY_CLIENTS)
        self.assertEqual(len({item['content'] for item in items}), 4)
        self.assertEqual({(item['ctx'], item['max_tokens'], item['min_tokens'], item['reasoning_budget'])
                          for item in items}, {(2000, 1024, 1024, 512)})
        self.assertEqual(len({item['question'] for item in items}), 4)
        self.assertEqual([item['content'] for item in onepass.fixed_concurrency_items(7, cq, 1024)],
                         [item['content'] for item in items])

    def test_group_sends_each_client_its_own_request_together(self):
        lock, active, maximum, seen = threading.Lock(), 0, 0, {}
        def ask(url, model, content, limit, timing, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                seen[timing['client']] = (content, limit, kwargs['min_tokens'])
            time.sleep(.02)
            timing.update(started_monotonic=10 + timing['client'], ended_monotonic=20, completion_tokens=limit)
            with lock:
                active -= 1
            return 'answer', 1, 2, limit, 'length'
        items = [dict(ctx=2000, question=f'q{i}', content=f'prompt {i}', max_tokens=64, min_tokens=64) for i in range(4)]
        got = group(None, ask, 'unused', 'test', items, 4)
        self.assertEqual(maximum, 4)
        self.assertEqual(seen, {i: (f'prompt {i}', 64, 64) for i in range(4)})
        self.assertEqual([r['question'] for r in got['requests']], ['q0', 'q1', 'q2', 'q3'])
        self.assertEqual(got['aggregate_output_tok_s'], 4 * 64 / 10)
        with self.assertRaises(ValueError):
            group(None, ask, 'unused', 'test', items[:3], 4)

    def test_summary_is_bench_dec_rate_and_decode_multiplier(self):
        c1 = [dict(completion_tokens=1024, elapsed_s=12.8, decode_tok_s=d) for d in (80., 82., 84., 86.)]
        c4 = dict(aggregate_output_tok_s=4 * 1024 / 25.6,
                  requests=[dict(completion_tokens=1024, decode_tok_s=41.) for _ in range(4)])
        got = onepass.fixed_concurrency_summary(1024, c1, c4)
        self.assertEqual(got['issues'], [])
        self.assertAlmostEqual(got['c1_tok_s'], 80.)            # pooled: 4,096 tokens over 51.2 s
        self.assertAlmostEqual(got['c4_tok_s'], 160.)
        self.assertAlmostEqual(got['multiplier'], 2.)
        self.assertAlmostEqual(got['c1_decode_tok_s'], 83.)
        self.assertAlmostEqual(got['decode_multiplier'], 164. / 83.)

    def test_a_short_or_missing_answer_invalidates_only_the_block(self):
        c1 = [dict(completion_tokens=1024, elapsed_s=12.8, decode_tok_s=80.) for _ in range(4)]
        c4 = dict(aggregate_output_tok_s=150., requests=[dict(completion_tokens=n, decode_tok_s=40.)
                                                         for n in (1024, 1024, 1024, 900)])
        got = onepass.fixed_concurrency_summary(1024, c1, c4)
        self.assertEqual(got['issues'], ['fixed concurrency output lengths [1024, 900] != [1024]'])
        missing = onepass.fixed_concurrency_summary(1024, c1[:3], dict(aggregate_output_tok_s=None, requests=[]))
        self.assertIn('fixed concurrency needs 4 requests at each concurrency', missing['issues'])
        self.assertIsNone(missing['multiplier'])


if __name__ == '__main__':
    unittest.main()
