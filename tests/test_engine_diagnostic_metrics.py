import types
import unittest
from unittest.mock import patch

from engine.base.diagnostic_metrics import DiagnosticMetrics, context_band


class DiagnosticMetricsTests(unittest.TestCase):
    def fixture(self):
        counts = {1: 0, 2: 0}
        model = types.SimpleNamespace(context=lambda s: 32000 if s == 1 else 64000,
                                      generated_count=lambda s: counts[s], drafted_total=0, accepted_total=0,
                                      drafter=types.SimpleNamespace(k=7, decode_precision='w4'))
        server = types.SimpleNamespace(engine=model, _active={1: (101, None), 2: (102, None)},
                                       _cached={101: 0, 102: 500}, latency_boot_id='boot')
        with patch('engine.base.diagnostic_metrics.build_identity', return_value='build'):
            metrics = DiagnosticMetrics(server)
        return metrics, model, counts

    def test_actual_joint_cohorts_and_completed_burst_steps(self):
        m, model, counts = self.fixture()
        before = m.begin((1, 2))
        counts.update({1: 6, 2: 8})
        model.drafted_total, model.accepted_total = 28, 10
        m.end(before, .1, iterations=2, timing='device_burst')
        out = m.render()
        self.assertIn('cache="mixed",context="131072",sequences="2",timing="device_burst"', out)
        self.assertIn('st:condition_steps_total{cache="mixed",context="131072",sequences="2",timing="device_burst"} 2', out)
        self.assertIn('st:condition_tokens_total{cache="mixed",context="131072",sequences="2",timing="device_burst"} 14', out)
        self.assertIn('precision="w4"', out)

    def test_ghost_rows_do_not_mislabel_a_wider_graph_as_c1(self):
        m, _, _ = self.fixture()
        del m.server._active[2]
        self.assertIsNone(m.begin((1, 2)))
        self.assertIsNotNone(m.begin((1,)))
        m.server.runner = types.SimpleNamespace(state=types.SimpleNamespace(running=set()))
        self.assertIsNone(m.begin((1,)))

    def test_lengths_include_zero_and_histogram_overflow(self):
        m, _, _ = self.fixture()
        choice = types.SimpleNamespace(finish_reason=lambda:'length', streams={
            'reasoning_content':types.SimpleNamespace(ids=range(140000)),
            'content':types.SimpleNamespace(ids=[])})
        m.response([choice])
        out=m.render()
        self.assertIn('st:response_tokens_bucket{le="+Inf",part="reasoning"} 1',out)
        self.assertIn('st:response_tokens_bucket{le="131072",part="reasoning"} 0',out)
        self.assertIn('st:response_tokens_sum{part="answer"} 0',out)
        self.assertIn('st:response_finished_total{reason="length"} 1',out)

    def test_prefill_counts_only_computed_chunk(self):
        m, _, _=self.fixture()
        m.prefill(128,.25)
        self.assertIn('st:prefill_computed_tokens_total{} 128',m.render())
        self.assertEqual([context_band(n) for n in (8192,8193,32769,131073)],
                         ['8192','32768','131072','over128k'])


if __name__=='__main__': unittest.main()
