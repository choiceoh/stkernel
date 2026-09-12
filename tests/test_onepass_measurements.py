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

    def test_fixed_active_stall_still_costs_elapsed_time(self):
        phase = (2000, 0, 10)
        _, fixed = decode_windows([(2, 0), (4, 40), (6, 40), (8, 80)], [phase], [phase])
        self.assertEqual([w["steps"] for w in fixed], [40, 0, 40])
        self.assertAlmostEqual(sum(w["steps"] for w in fixed) / sum(w["seconds"] for w in fixed), 80 / 6)


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
        self.assertEqual((timing["min_tokens"], timing["max_tokens"], timing["seed"], timing["prompt_tokens"]),
                         (1536, 1536, 7, 2000))
        self.assertEqual(timing["request_sha256"], onepass.hashlib.sha256(urlopen.call_args.args[0].data).hexdigest())
        self.assertAlmostEqual(timing["tpot_ms"], 1000 * 20.1 / 1535)
        self.assertEqual(timing["chunk_gaps_ms"], [20000])

    def test_combined_reasoning_budget_is_forwarded_and_recorded(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "근거"}}]},
            {"choices": [{"delta": {"content": "답변"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 32545, "completion_tokens": 900}},
        ]
        stream = io.BytesIO(b"".join(("data: " + json.dumps(e) + "\n\n").encode()
                                     for e in events) + b"data: [DONE]\n")
        timing = {}
        with patch.object(onepass.urllib.request, "urlopen", return_value=stream) as urlopen, \
                patch.object(onepass.time, "monotonic", side_effect=[100, 102, 122, 122.1]):
            result = onepass.ask_stream("http://fixture/v1/chat/completions", "glm", "질문", 2400,
                                        timing, reasoning_budget=900)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["max_tokens"], 2400)
        self.assertEqual(payload["reasoning_budget"], 900)
        self.assertEqual(result, ("근거답변", 2, 32545, 900, "stop"))
        self.assertEqual(timing["reasoning_budget"], 900)




class PrefixReuseColumnTests(unittest.TestCase):
    """A cold TTFT is prefill throughput only if the prompt was actually computed.

    Three rows on 2026-09-12 were not, and nothing in the record said so: ST-GRAPH-45c's 128K reads
    46,554 tok/s at 2.76 s, and an evening run read 32K in 5.13 s and 128K in 5.22 s -- the same
    number for two prompts four times apart. They then sat in the ledger's cold column beside rows
    that had prefilled (45차 §82). CHARTER D17 says to drop them; this makes the record say which.
    """

    def series(self, reused):
        return ('# TYPE st:prefix_reused_tokens_total counter\n'
                f'st:prefix_reused_tokens_total{{engine="st"}} {reused}\n')

    def test_the_engines_own_series_is_read_and_vllms_reader_does_not_see_it(self):
        text = self.series(201984)
        self.assertEqual(onepass._st_counter(text, "prefix_reused_tokens_total"), 201984.0)
        self.assertEqual(onepass._counter(text, "prefix_reused_tokens_total"), 0.0, "vllm: dialect only")
        self.assertEqual(onepass._st_counter("", "prefix_reused_tokens_total"), 0.0)

    def test_the_gap_between_sharing_a_preamble_and_resuming_a_boundary(self):
        """Half is not a knob, it is a gap: a bracket that only shares the system preamble reuses
        about 1% of a 32,545-token prompt, and one that resumed a boundary reuses nearly all of it."""
        tok = 32545
        self.assertLess(320 / tok, onepass.CACHE_HIT_FRACTION, "a preamble is not a hit")
        self.assertGreater(32000 / tok, onepass.CACHE_HIT_FRACTION, "a resumed boundary is")

    def test_the_row_carries_the_number_whether_or_not_it_is_called_a_hit(self):
        source = Path(onepass.__file__).read_text()
        for field in ('"reused_tok"', '"reused_frac"', '"cache_hit"'):
            self.assertIn(field, source, field)
        self.assertIn("CACHE_HIT_FRACTION", source)
        self.assertIn("not a prefill, CHARTER D17", source, "and the printed table says it too")


if __name__ == "__main__":
    unittest.main()
