"""Run the pinned worker's actual memory-sizing method with fake GPU counters."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class GraphProfileTests(unittest.TestCase):
    def run_worker(self, *, opt="1", estimate=False, v2=True, model="glm5_next",
                   graphs=True, cuda=True, explicit=None, profile_error=False):
        tree = ast.parse((ROOT / "overlay/modules/glm53_runtime/gpu_worker.py").read_text())
        worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker")
        method = next(n for n in worker.body if isinstance(n, ast.FunctionDef)
                      and n.name == "determine_available_memory")
        method.decorator_list = []
        events = []
        @contextmanager
        def profiling(*args, **kwargs):
            events.append("measure-start")
            yield NS(total_consumed=400, transient_peak_headroom=50,
                     non_kv_cache_memory=450, after_profile=NS(free_memory=600))
            events.append("measure-end")
        def model_profile():
            events.append("model-and-mm")
            if profile_error:
                raise RuntimeError("model profile failed")
        def graph_profile():
            events.append("dry-capture")
            return 25
        logger = Mock()
        namespace = dict(os=os, envs=NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=estimate),
            current_platform=NS(is_cuda_alike=lambda: cuda), CUDAGraphMode=NS(NONE=0),
            memory_profiling=profiling, maybe_apply_startup_plan=lambda w: None,
            reserve_mm_ipc_gpu_memory=lambda size, *a: size, format_gib=lambda x: x,
            logger=logger)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "gpu_worker.py", "exec"), namespace)
        obj = NS(rank=2, use_v2_model_runner=v2,
            model_config=NS(hf_config=NS(model_type=model), multimodal_config=None),
            cache_config=NS(kv_cache_memory_bytes=explicit, gpu_memory_utilization=0.8),
            parallel_config=NS(_api_process_count=1), requested_memory=800,
            init_snapshot=NS(free_memory=1000, total_memory=1000),
            vllm_config=NS(compilation_config=NS(cudagraph_mode=int(graphs))),
            model_runner=NS(model_memory_usage=400, profile_run=model_profile,
                            profile_cudagraph_memory=graph_profile))
        with patch.dict(os.environ, {"VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE": opt}):
            result = namespace["determine_available_memory"](obj)
        return result, events, obj, logger

    def test_skip_preserves_measured_budget_and_real_model_profile(self):
        base, before, old, _ = self.run_worker(opt="0")
        fast, after, new, log = self.run_worker()
        self.assertEqual(before, ["measure-start", "model-and-mm", "measure-end", "dry-capture"])
        self.assertEqual(after, before[:-1])
        self.assertEqual((base, fast), (350, 350))
        self.assertEqual(old.peak_activation_memory, new.peak_activation_memory)
        self.assertEqual(old.total_consumed, new.total_consumed)
        self.assertEqual(new.cudagraph_memory_estimate, 0)
        self.assertTrue(any("skipped unused estimate" in str(c) for c in log.info.call_args_list))

    def test_enabled_estimator_always_runs_and_subtracts_headroom(self):
        for opt in ("0", "1"):
            with self.subTest(opt=opt):
                result, events, worker, _ = self.run_worker(opt=opt, estimate=True)
                self.assertIn("dry-capture", events)
                self.assertEqual(result, 325)
                self.assertEqual(worker.peak_activation_memory, 75)

    def test_unsupported_paths_and_invalid_opt_in_preserve_stock(self):
        for kwargs in ({"v2": False}, {"model": "other"}, {"model": None},
                       {"opt": "0"}, {"opt": "true"}, {"opt": ""}):
            with self.subTest(kwargs=kwargs):
                result, events, _, _ = self.run_worker(**kwargs)
                self.assertEqual(result, 350)
                self.assertIn("dry-capture", events)

    def test_no_graph_or_non_cuda_does_not_dry_capture(self):
        for kwargs in ({"graphs": False}, {"cuda": False}):
            for opt in ("0", "1"):
                result, events, _, _ = self.run_worker(opt=opt, **kwargs)
                self.assertEqual(result, 350)
                self.assertNotIn("dry-capture", events)

    def test_explicit_kv_retains_required_model_warmup(self):
        for opt in ("0", "1"):
            result, events, _, _ = self.run_worker(opt=opt, explicit=300)
            self.assertEqual((result, events), (300, ["model-and-mm"]))

    def test_model_profile_failure_is_not_suppressed(self):
        with self.assertRaisesRegex(RuntimeError, "model profile failed"):
            self.run_worker(profile_error=True)


if __name__ == "__main__":
    unittest.main()
