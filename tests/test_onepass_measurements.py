"""CPU fixtures for measurement contamination and streamed token timing."""
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import onepass
from window_metrics import decode_windows, exclusive_errors, traffic_state


def state(n, running=0, waiting=0):
    return dict(finished=n, running=running, waiting=waiting)


class TrafficTests(unittest.TestCase):
    def test_label_series_are_summed_and_missing_is_unknown(self):
        parsed = traffic_state('''# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",engine="0"} 12
vllm:request_success_total{finished_reason="length",engine="0"} 3.0
vllm:request_success_total{finished_reason="abort",engine="0"} 1e0
vllm:num_requests_running{engine="0"} 0.0
''')
        self.assertEqual(parsed, dict(finished=16, running=0, waiting=None))
        self.assertIn("missing", exclusive_errors(parsed, parsed, [], 0)[0])

    def test_clean_serial_workload(self):
        self.assertEqual(exclusive_errors(state(20), state(28),
                         [state(20, 1), state(23), state(27, 1)], 8), [])

    def test_short_external_request_between_polls(self):
        errors = exclusive_errors(state(20), state(29),
                                  [state(20, 1), state(27, 1)], 8)
        self.assertTrue(any("completed requests 9 != own requests 8" in e for e in errors))

    def test_queued_external_request_and_boundary_activity(self):
        errors = exclusive_errors(state(20, 1), state(28, waiting=1),
                                  [state(21, 1, 1)], 8)
        self.assertEqual(len(errors), 3)
        self.assertTrue(any("concurrent or queued" in e for e in errors))

    def test_counter_reset_even_if_final_delta_happens_to_match(self):
        errors = exclusive_errors(state(20), state(28), [state(1, 1)], 8)
        self.assertTrue(any("reset" in e for e in errors))


class WindowTests(unittest.TestCase):
    def test_edges_idle_gaps_and_contexts_do_not_mix(self):
        phases = [(2000, 0, 8), (32000, 10, 18), (2000, 20, 28)]
        samples = [(t, 20 * t) for t in range(0, 31, 2)]
        by_ctx, fixed = decode_windows(samples, phases, [phases[-1]])
        self.assertEqual(by_ctx, {2000: [20, 20, 20, 20], 32000: [20, 20]})
        self.assertEqual([(w["start"], w["end"]) for w in fixed], [(22, 24), (24, 26)])
        self.assertEqual(sum(w["steps"] for w in fixed) / sum(w["seconds"] for w in fixed), 20)

    def test_backward_or_empty_intervals_are_not_rates(self):
        by_ctx, fixed = decode_windows([(2, 100), (2, 120), (4, 0), (6, 0)], [(2000, 0, 10)])
        self.assertEqual((by_ctx, fixed), ({}, []))


class StreamTests(unittest.TestCase):
    def test_fixed_length_payload_and_tpot_use_tokens_not_chunks(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "근거"}}]},
            {"choices": [{"delta": {"content": "답변"}, "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 2000, "completion_tokens": 1536}},
        ]
        stream = io.BytesIO(b"".join(("data: " + json.dumps(e) + "\n\n").encode()
                                     for e in events) + b"data: [DONE]\n")
        timing = {}
        with patch.object(onepass.urllib.request, "urlopen", return_value=stream) as urlopen, \
             patch.object(onepass.time, "monotonic", side_effect=[100, 102, 122, 122.1]):
            result = onepass.ask_stream("http://fixture/v1/chat/completions", "glm", "질문", 1536,
                                        timing, min_tokens=1536, seed=7)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual((payload["min_tokens"], payload["max_tokens"], payload["seed"]), (1536, 1536, 7))
        self.assertEqual(result, ("근거답변", 2, 2000, 1536, "length"))
        self.assertAlmostEqual(timing["tpot_ms"], 1000 * 20.1 / 1535)
        self.assertEqual(timing["chunk_gaps_ms"], [20000])


if __name__ == "__main__":
    unittest.main()
