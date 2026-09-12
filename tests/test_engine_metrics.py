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
    """Every series carries HELP and TYPE, and every bucket set is cumulative and closed."""
    typed = dict(re.findall(r"^# TYPE (\S+) (\S+)$", text, re.M))
    helped = {m.group(1) for m in re.finditer(r"^# HELP (\S+) .+$", text, re.M)}
    case.assertEqual(set(typed), helped, "every TYPE needs its HELP")
    buckets, counts = {}, {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        series, _, value = line.rpartition(" ")
        name, _, labels = series.partition("{")
        labels = dict(re.findall(r'(\w+)="([^"]*)"', labels))
        base = re.sub(r"_(bucket|sum|count)$", "", name)
        case.assertTrue(name in typed or base in typed, f"{name} has no TYPE")
        case.assertEqual(labels.get("engine"), "st", line)
        if typed.get(base) != "histogram":
            continue
        key = (base, tuple(sorted((k, v) for k, v in labels.items() if k != "le")))
        if name.endswith("_bucket"):
            buckets.setdefault(key, []).append((labels["le"], int(value)))
        elif name.endswith("_count"):
            counts[key] = int(value)
    case.assertTrue(buckets, "no histogram buckets were emitted")
    for key, pairs in buckets.items():
        values = [v for _, v in pairs]
        case.assertEqual(values, sorted(values), f"{key} buckets must be cumulative")
        case.assertEqual(pairs[-1][0], "+Inf", f"{key} must close at +Inf")
        case.assertIn(key, counts, f"{key} has no _count")
        case.assertEqual(values[-1], counts[key], f"{key}: +Inf must equal the count")


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

    def test_the_prefix_tiers_bytes_are_its_own_and_not_the_conversation_tiers(self):
        """`st:tier_bytes_*` reads runner.tiered -- the CONVERSATION tier. Everything the boundary
        tier moved was invisible: a live fleet showed 42 prefix spills and 2 restores against
        `st:tier_bytes_read_total` 0, so the one thing that decides the snapshot pool's size --
        what a faded boundary actually costs to read back -- could not be read at all (2026-09-12)."""
        import types
        import test_engine_serve as T
        s = T.server(tiered=True)
        self.assertIsNone(_series(s.metrics(), "st:prefix_tier_bytes_written_total"))   # no prefix tier, no series
        s.runner.prefix_tier = types.SimpleNamespace(
            tier=types.SimpleNamespace(bytes_written=1 << 20, bytes_read=3 << 20))
        s.runner.tiered.tier.bytes_written, s.runner.tiered.tier.bytes_read = 4096, 512
        text = s.metrics()
        self.assertEqual(_series(text, "st:prefix_tier_bytes_written_total"), float(1 << 20))
        self.assertEqual(_series(text, "st:prefix_tier_bytes_read_total"), float(3 << 20))
        self.assertEqual(_series(text, "st:tier_bytes_written_total"), 4096.0, "the two tiers stay apart")
        self.assertEqual(_series(text, "st:tier_bytes_read_total"), 512.0)
        _exposition_is_wellformed(self, text)

    def test_the_exposition_is_wellformed(self):
        for server in (self._served(), self._served(tiered=True)):
            _exposition_is_wellformed(self, server.metrics())


class KernelAndServingTests(unittest.TestCase):
    """The series no vLLM counter answers, because no vLLM has these parts. Each one exists
    to settle a decision this engine otherwise has to re-measure with a probe."""

    def test_decode_shape_counters_name_the_graph_that_ran(self):
        import test_engine_serve as T
        server = T.server()
        server.engine.decode_shape_counts = {(1, 4096): 7, (2, 4096): 3, (2, 8192): 2}
        text = server.metrics()
        self.assertEqual(_series(text, 'st:decode_steps_by_sequences_total'), 12.0)
        for labels, expected in (('sequences="1"', 7), ('sequences="2"', 5)):
            self.assertIn(f'st:decode_steps_by_sequences_total{{engine="st",{labels}}} {expected}', text)
        # the capacity bucket is the only production evidence for how far the ladder must reach
        self.assertIn('st:decode_capacity_bucket_total{engine="st",capacity="4096"} 10', text)
        self.assertIn('st:decode_capacity_bucket_total{engine="st",capacity="8192"} 2', text)
        self.assertEqual(_series(text, "st:decode_capacity_bucket_total"), 12.0)

    def test_the_engine_counts_the_shape_it_replayed(self):
        """The counter comes from the shape the adapter already computed: no extra work."""
        source = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        self.assertIn("key = (shape[0], shape[2])", source)
        self.assertIn("self.decode_shape_counts[key] = self.decode_shape_counts.get(key, 0) + 1", source)
        self.assertIn("self.accepted_per_step[committed] += 1", source)

    def test_acceptance_is_exposed_as_a_distribution(self):
        import test_engine_serve as T
        server = T.server()
        server.engine.accepted_per_step = [4, 2, 1, 0, 0, 5]
        text = server.metrics()
        self.assertIn('st:spec_accepted_per_step_total{engine="st",accepted="0"} 4', text)
        self.assertIn('st:spec_accepted_per_step_total{engine="st",accepted="5"} 5', text)
        self.assertNotIn('accepted="3"', text)                  # empty middles are not emitted
        self.assertEqual(_series(text, "st:spec_accepted_per_step_total"), 12.0)

    def test_step_seconds_are_split_by_kind_and_counted_once_per_step(self):
        """One observation per step, under the kind that step was. The value is host wall
        time, so the test pins the count and the split, not a duration the fixture invents."""
        import test_engine_serve as T
        server = T.server()
        ticks = [0.0]
        server.clock = lambda: ticks.__setitem__(0, ticks[0] + 0.01) or ticks[0]
        server.submit([1, 2, 3], max_new=4, temperature=0.0)
        for _ in range(60):
            if not server.once() and not server._waiting:
                break
        text = server.metrics()
        prefill = _series(text, 'st:step_seconds_count{engine="st",kind="prefill"}')
        decode = _series(text, 'st:step_seconds_count{engine="st",kind="decode"}')
        self.assertEqual(prefill, _series(text, "st:steps_prefill_total"))
        self.assertEqual(decode, _series(text, "st:steps_decode_total"))
        self.assertEqual(prefill + decode, _series(text, "vllm:iteration_tokens_total_count"))
        self.assertGreater(prefill, 0)
        self.assertGreater(decode, 0)
        for kind in ("prefill", "decode"):
            self.assertGreater(_series(text, f'st:step_seconds_sum{{engine="st",kind="{kind}"}}'), 0)
            self.assertEqual(_series(text, f'st:step_seconds_bucket{{engine="st",kind="{kind}",le="+Inf"}}'),
                             _series(text, f'st:step_seconds_count{{engine="st",kind="{kind}"}}'))

    def test_lane_info_says_what_is_actually_bound(self):
        import test_engine_serve as T
        server = T.server()
        self.assertIsNone(_series(server.metrics(), "st:lane_info"))
        server.engine.lane_info = {"lanes": "served", "moe_static": "t,r,sf6", "spec_k": "5"}
        text = server.metrics()
        self.assertIn('st:lane_info{engine="st",lanes="served",moe_static="t,r,sf6",spec_k="5"} 1', text)

    def test_the_fleet_boot_publishes_lane_info(self):
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn('engine.lane_info = {"lanes": lanes.name, "moe_static": cfg["moe_static"]', source)


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
