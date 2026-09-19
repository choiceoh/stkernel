"""The benchmark must record the model it actually targets, even beside GLM."""
import os
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
import onepass


class ContainerIdentityTests(unittest.TestCase):
    def test_real_onepass_dependencies_load_without_retired_files(self):
        with patch.dict(os.environ, {"BENCH_MODEL": "fixture", "GLM53_API_PORT": "8001"}):
            for filename in ("korean-corruption.py", "check-quality.py", "onepass_metrics.py"):
                module = onepass._load(filename, "actual_onepass_dependency")
            self.assertEqual(module.URL, "http://127.0.0.1:8001/v1/chat/completions")
            self.assertEqual(module.METRICS, "http://127.0.0.1:8001/metrics")

    def test_actual_st_counters_preserve_raw_acceptance_and_effective_k(self):
        metrics = onepass._load("onepass_metrics.py", "counter_fixture")
        def payload(accepted, drafted, rounds):
            return (f'vllm:spec_decode_num_accepted_tokens_total{{model="qwen"}} {accepted}\n'
                    f'vllm:spec_decode_num_draft_tokens_total{{model="qwen"}} {drafted}\n'
                    f'vllm:spec_decode_num_drafts_total{{model="qwen"}} {rounds}\n')
        before = metrics._parse_spec_metrics(payload(30, 120, 40))
        after = metrics._parse_spec_metrics(payload(180, 420, 140))
        legacy, raw = metrics._spec_delta(before, after)
        self.assertEqual((legacy, raw, metrics.spec_k_eff(before, after)), (0.375, 0.5, 3.0))
        self.assertEqual(metrics._spec_delta({}, {}), (None, None))
        self.assertIsNone(metrics.spec_k_eff({}, {}))

    def test_step_sampler_reads_total_and_keeps_missing_traffic_unknown(self):
        metrics = onepass._load("onepass_metrics.py", "counter_fixture")
        sampler = metrics._StepWindows(metrics)
        data = (b'vllm:iteration_tokens_total_count{model="qwen"} 321\n'
                b'st:steps_prefill_total{} 21\nst:steps_decode_total{} 300\n')
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
            self.assertEqual(sampler._steps(), 321)
        self.assertEqual(sampler.traffic_samples, [dict(finished=None, running=None, waiting=None)])

    def test_explicit_qwen_identity_does_not_report_resident_glm(self):
        with patch.dict(os.environ, {"ONEPASS_ST_CONTAINER": "st-qwen38"}), \
             patch("subprocess.run", return_value=SimpleNamespace(stdout="st-glm53\nst-qwen38\n")), \
             patch.object(onepass, "_st_build", return_value={"boot_id": "qwen"}) as inspect:
            self.assertEqual(onepass._served_build("unused"), {"boot_id": "qwen"})
            inspect.assert_called_once_with(["st-glm53", "st-qwen38"], name="st-qwen38")

    def test_missing_explicit_container_never_borrows_glm_identity(self):
        with patch.dict(os.environ, {"ONEPASS_ST_CONTAINER": "st-qwen38"}), \
             patch("subprocess.run", return_value=SimpleNamespace(stdout="st-glm53\n")), \
             patch.object(onepass, "_st_build") as inspect:
            self.assertEqual(onepass._served_build("unused"), {})
            inspect.assert_not_called()

    def test_default_remains_glm(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("subprocess.run", return_value=SimpleNamespace(stdout="st-glm53\n")), \
             patch.object(onepass, "_st_build", return_value={"boot_id": "glm"}) as inspect:
            self.assertEqual(onepass._served_build("unused"), {"boot_id": "glm"})
            inspect.assert_called_once_with(["st-glm53"], name="st-glm53")


if __name__ == "__main__":
    unittest.main()
