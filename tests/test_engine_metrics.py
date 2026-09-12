"""The door's /metrics: the bench's contract, and what an operator needs on top of it.

The first block of names is read by bench/window_metrics.py and bench/bracket.py, so it
is pinned here by name and meaning. The rest is this session's addition: latency measured
from admission, cache saturation, prefix reuse, and the step split D9 promises.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from bench.window_metrics import metric_sum, traffic_state          # noqa: E402


def _series(text, name):
    return metric_sum(text, name)


def _exposition_is_wellformed(case, text):
    """Every series carries HELP and TYPE, and every bucket set is monotone."""
    typed = dict(re.findall(r"^# TYPE (\S+) (\S+)$", text, re.M))
    helped = {m.group(1) for m in re.finditer(r"^# HELP (\S+) .+$", text, re.M)}
    case.assertEqual(set(typed), helped, "every TYPE needs its HELP")
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        base = re.sub(r"_(bucket|sum|count)$", "", name)
        case.assertTrue(name in typed or base in typed, f"{name} has no TYPE")
    for histogram in set(n for n, t in typed.items() if t == "histogram"):
        pairs = re.findall(re.escape(histogram) + r'_bucket\{engine="st",le="([^"]+)"\} (\d+)', text)
        case.assertTrue(pairs, histogram)
        counts = [int(v) for _, v in pairs]
        case.assertEqual(counts, sorted(counts), f"{histogram} buckets must be cumulative")
        case.assertEqual(counts[-1], int(_series(text, histogram + "_count")), histogram)
        case.assertEqual(pairs[-1][0], "+Inf")


class MetricsTests(unittest.TestCase):
    def _served(self, *, steps=60, tick=0.05, **kwargs):
        import test_engine_serve as T
        server = T.server(**kwargs)
        clock = [0.0]
        server.clock = lambda: clock[0]
        server.submit([1, 2, 3], max_new=4, temperature=0.0)
        for _ in range(steps):
            clock[0] += tick
            if not server.once() and not server._waiting:
                break
        return server

    def test_the_bench_dialect_still_reads_its_counters(self):
        server = self._served()
        text = server.metrics()
        self.assertEqual(traffic_state(text), {"finished": 1.0, "running": 0.0, "waiting": 0.0})
        for name in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                     "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                     "vllm:iteration_tokens_total_count", "st:requests_cancelled_total"):
            self.assertIsNotNone(_series(text, name), name)
        self.assertEqual(_series(text, "vllm:prompt_tokens_total"), 3.0)
        self.assertEqual(_series(text, "vllm:generation_tokens_total"), 4.0)

    def test_latency_is_measured_from_admission_and_split_by_token(self):
        server = self._served(tick=0.05)
        text = server.metrics()
        # four tokens: the first is the time to first token, the other three are intervals
        self.assertEqual(_series(text, "vllm:time_to_first_token_seconds_count"), 1.0)
        self.assertEqual(_series(text, "vllm:time_to_first_token_seconds_sum"), 0.05)
        self.assertEqual(_series(text, "vllm:time_per_output_token_seconds_count"), 3.0)
        self.assertEqual(_series(text, "vllm:e2e_request_latency_seconds_count"), 1.0)
        self.assertEqual(_series(text, "vllm:e2e_request_latency_seconds_sum"), 0.2)
        self.assertAlmostEqual(_series(text, "vllm:time_per_output_token_seconds_sum"), 0.15, places=6)

    def test_the_step_split_adds_up_to_the_step_counter(self):
        server = self._served()
        text = server.metrics()
        prefill = _series(text, "st:steps_prefill_total")
        decode = _series(text, "st:steps_decode_total")
        self.assertEqual(prefill, 1.0)                                  # one chunk covers this prompt
        self.assertEqual(decode, 3.0)
        self.assertEqual(prefill + decode, _series(text, "vllm:iteration_tokens_total_count"))

    def test_capacity_gauges_report_raw_occupancy(self):
        import test_engine_serve as T
        server = T.server(blocks=16)
        text = server.metrics()
        self.assertEqual(_series(text, "st:kv_blocks_total"), 16.0)
        self.assertEqual(_series(text, "st:kv_blocks_used"), 0.0)
        self.assertEqual(_series(text, "st:kv_blocks_free"), 16.0)
        self.assertEqual(_series(text, "vllm:gpu_cache_usage_perc"), 0.0)
        free = _series(text, "st:state_slots_free")
        self.assertEqual(free, _series(text, "st:state_slots_total"))    # slot 0 is the null slot, never counted
        server.runner.kv.reserve(0, 12)                                  # three blocks of four tokens
        text = server.metrics()
        self.assertEqual(_series(text, "st:kv_blocks_used"), 3.0)
        self.assertEqual(_series(text, "st:kv_blocks_free"), 13.0)
        self.assertAlmostEqual(_series(text, "vllm:gpu_cache_usage_perc"), 3 / 16, places=6)

    def test_a_timeout_is_counted_apart_from_other_cancellations(self):
        import test_engine_serve as T
        server = T.server()
        clock = [0.0]
        server.clock = lambda: clock[0]
        request, _ = server.submit([1, 2, 3], max_new=4, temperature=0.0)
        server.cancel(request, "client")
        server.once()
        text = server.metrics()
        self.assertEqual(_series(text, "st:requests_cancelled_total"), 1.0)
        self.assertEqual(_series(text, "st:requests_timed_out_total"), 0.0)
        other, _ = server.submit([1, 2, 3], max_new=4, temperature=0.0)
        clock[0] += server.request_timeout_s + 1
        server.once()
        text = server.metrics()
        self.assertEqual(_series(text, "st:requests_cancelled_total"), 2.0)
        self.assertEqual(_series(text, "st:requests_timed_out_total"), 1.0)
        # a cancelled request leaves no timing behind for a later row to inherit
        self.assertEqual(server._arrived, {})
        self.assertEqual(server._token_at, {})

    def test_the_tier_series_appear_only_with_a_tier(self):
        import test_engine_serve as T
        self.assertIsNone(_series(T.server().metrics(), "st:conversations_parked"))
        text = T.server(tiered=True).metrics()
        self.assertEqual(_series(text, "st:conversations_parked"), 0.0)
        # the byte counters belong to the NVMe tier; a tier double without them is not an error
        self.assertIsNone(_series(text, "st:tier_bytes_written_total"))
        from engine.base.kv_tier import NvmeTier
        self.assertTrue(hasattr(NvmeTier, "__init__"))
        import test_engine_serve as T2
        s2 = T2.server(tiered=True)
        s2.runner.tiered.tier.bytes_written = 4096
        s2.runner.tiered.tier.bytes_read = 512
        text = s2.metrics()
        self.assertEqual(_series(text, "st:tier_bytes_written_total"), 4096.0)
        self.assertEqual(_series(text, "st:tier_bytes_read_total"), 512.0)

    def test_the_exposition_is_wellformed(self):
        for server in (self._served(), self._served(tiered=True)):
            _exposition_is_wellformed(self, server.metrics())


class HistogramTests(unittest.TestCase):
    def _histogram(self):
        from engine.base.serve import _Histogram
        return _Histogram((0.1, 0.5, 1.0))

    def test_buckets_are_cumulative_and_the_sum_is_exact(self):
        h = self._histogram()
        for value in (0.05, 0.2, 0.7, 3.0):
            h.observe(value)
        rows = dict(h.rows("x"))
        self.assertEqual(rows['x_bucket{engine="st",le="0.1"}'], 1)
        self.assertEqual(rows['x_bucket{engine="st",le="0.5"}'], 2)
        self.assertEqual(rows['x_bucket{engine="st",le="1.0"}'], 3)
        self.assertEqual(rows['x_bucket{engine="st",le="+Inf"}'], 4)
        self.assertEqual(rows['x_count{engine="st"}'], 4)
        self.assertAlmostEqual(rows['x_sum{engine="st"}'], 3.95, places=6)

    def test_a_bound_is_inclusive_and_bad_samples_are_dropped(self):
        h = self._histogram()
        h.observe(0.1)                                     # le is inclusive
        self.assertEqual(dict(h.rows("x"))['x_bucket{engine="st",le="0.1"}'], 1)
        for bad in (-1.0, float("nan")):
            h.observe(bad)
        self.assertEqual(h.total, 1)

    def test_buckets_stay_monotone_when_a_scrape_lands_mid_observation(self):
        """The exposition may not emit a bucket smaller than a narrower one, whatever the
        step loop was doing when the HTTP thread read the counters."""
        h = self._histogram()
        h.observe(0.05)
        h.counts[0] = 5                                    # a torn read: the narrow bucket ran ahead
        counts = [v for k, v in h.rows("x") if "_bucket" in k]
        self.assertEqual(counts, sorted(counts))


if __name__ == "__main__":
    unittest.main()
