import importlib.util
import json
from pathlib import Path
import unittest
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fresh_module = load("fresh", "bench/onepass_fresh.py")
onepass = load("onepass_for_fresh_test", "bench/onepass.py")


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":2}}\n'
        yield b'data: [DONE]\n'


class FreshTests(unittest.TestCase):
    def run_request(self, hits=(0, 0)):
        events, requests = [], []
        counts = iter(hits)
        def metrics():
            events.append("metrics")
            return dict(traffic=dict(running=0, waiting=0), prefix_hits=next(counts))
        def opener(request, *args, **kwargs):
            events.append("send")
            requests.append(json.loads(request.data))
            return Response()
        fresh = fresh_module.FreshRequests("http://test/v1/chat/completions", onepass.ask_stream, opener, metrics)
        return fresh, events, requests

    def test_real_onepass_preserves_inputs_and_uses_unique_salt(self):
        fresh, events, requests = self.run_request((0, 0, 0, 0))
        timings = []
        with patch.object(urllib.request, "urlopen", fresh.open):
            for _ in range(2):
                timing = {}
                result = fresh.call(fresh.url, "model", "document", 20, timing)
                self.assertEqual(result[0], "answer")
                timings.append(timing)
        self.assertEqual(events, ["metrics", "send", "metrics"] * 2)
        self.assertNotEqual(requests[0].pop("cache_salt"), requests[1].pop("cache_salt"))
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[0]["messages"], [{"role": "user", "content": "document"}])
        for entry, timing in zip(fresh.records, timings):
            self.assertEqual(entry["unsalted_sha256"], timing["request_sha256"])
            self.assertNotEqual(entry["unsalted_sha256"], entry["wire_sha256"])
            self.assertFalse(entry["issues"])
            self.assertEqual(entry["ttft_s"], timing["ttft_s"])

    def test_hits_reset_and_unavailable_counters_fail(self):
        for hits in ((0, 1), (2, 0), (None, 0), (0, float("nan"))):
            fresh, _, _ = self.run_request(hits)
            with self.subTest(hits=hits), patch.object(urllib.request, "urlopen", fresh.open), self.assertRaises(RuntimeError):
                fresh.call(fresh.url, "model", "document", 20)
            self.assertTrue(fresh.records[0]["issues"])

    def test_busy_server_refuses_before_model_request(self):
        fresh, events, _ = self.run_request()
        fresh.metrics = lambda: dict(traffic=dict(running=1, waiting=0), prefix_hits=0)
        with self.assertRaises(RuntimeError):
            fresh.call(fresh.url, "model", "document", 20)
        self.assertEqual(events, [])

    def test_non_model_requests_are_untouched(self):
        seen = []
        fresh = fresh_module.FreshRequests("http://test/v1/chat/completions", None,
                                          lambda r, **kw: seen.append(r), None)
        request = urllib.request.Request("http://test/metrics")
        fresh.open(request)
        self.assertIs(seen[0], request)
        with self.assertRaises(RuntimeError):
            fresh.open(urllib.request.Request(fresh.url, data=b"{}"))


if __name__ == "__main__":
    unittest.main()
