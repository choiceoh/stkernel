"""Early MM warmup joins before readiness and preserves retry/cache semantics."""
from concurrent.futures import ThreadPoolExecutor
import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RendererWarmupTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('early_mm', ROOT / 'overlay/modules/glm53_runtime/glm53_renderer_warmup.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        env = patch.dict(os.environ, {'VLLM_GLM53_EARLY_MM_WARMUP': '1',
                                      'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'})
        env.start()
        self.addCleanup(env.stop)
        self.active_guards = 0
        self.events = []
        @contextmanager
        def guard(threads):
            self.assertEqual(threads, 1)
            self.assertEqual(self.active_guards, 0, 'overlapping global Torch thread guards')
            self.active_guards += 1
            try:
                yield
            finally:
                self.active_guards -= 1
        modules = patch.dict(sys.modules, {'vllm.utils.torch_utils': types.SimpleNamespace(set_default_torch_num_threads=guard)})
        modules.start()
        self.addCleanup(modules.stop)
        self.guard = guard
        self.renderer = self.make_renderer()

    def make_renderer(self):
        owner = self
        class Processor:
            def __init__(self, name):
                self.name, self.calls, self.clears, self.cache = name, 0, 0, []
        class Renderer:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type='glm5_next'),
                        multimodal_config=types.SimpleNamespace(mm_ipc_gpu_memory_gb=0,
                            mm_device_do_normalize=False, use_gpu_video_backend=lambda: False)),
                    parallel_config=types.SimpleNamespace(_api_process_count=1))
                self.mm_processor = Processor('main')
                self._readonly_mm_processor = Processor('readonly')
                self._mm_executor = ThreadPoolExecutor(max_workers=1)
                self.entered, self.release = threading.Event(), threading.Event()
                self.release.set()
                self.fail_once = False
            def _warmup_mm_processor(self, processor, *, log_prefix):
                self.entered.set()
                if not self.release.wait(5):
                    raise TimeoutError('test did not release warmup')
                processor.calls += 1
                processor.cache.append('dummy')
                owner.events.append(('mm', processor.name))
                if self.fail_once and processor is self.mm_processor and processor.calls == 1:
                    raise ValueError('injected warmup error')
            @staticmethod
            def _clear_processor_cache(processor):
                processor.clears += 1
                processor.cache.clear()
            def clear_mm_cache(self):
                self._clear_processor_cache(self.mm_processor)
            def shutdown(self):
                owner.events.append(('close', None))
                self._mm_executor.shutdown(wait=False)
            def warmup(self, params):
                with owner.guard(1):
                    owner.events.append(('chat', params))
                    for proc, label, clear in ((self.mm_processor, 'Multi-modal', self.clear_mm_cache),
                        (self._readonly_mm_processor, 'Readonly multi-modal', lambda: self._clear_processor_cache(self._readonly_mm_processor))):
                        if proc is not None:
                            try:
                                self._warmup_mm_processor(proc, log_prefix=label)
                            except Exception:
                                pass
                            finally:
                                clear()
                return params
        renderer = Renderer()
        self.addCleanup(renderer._mm_executor.shutdown)
        return renderer

    def test_success_is_consumed_once_and_cache_is_cleared(self):
        r = self.renderer
        self.assertTrue(self.module.start_renderer_warmup(r))
        self.assertTrue(self.module.start_renderer_warmup(r))
        r._glm53_early_mm_future.result(5)
        params = object()
        self.assertIs(r.warmup(params), params)
        self.assertEqual(self.events, [('mm', 'main'), ('mm', 'readonly'), ('chat', params)])
        for p in (r.mm_processor, r._readonly_mm_processor):
            self.assertEqual((p.calls, p.clears, p.cache), (1, 2, []))
        r.warmup(params)
        self.assertEqual(r.mm_processor.calls, 2)
        self.assertEqual(r._readonly_mm_processor.calls, 2)

    def test_pending_task_joins_before_chat_or_thread_guard(self):
        r = self.renderer
        r.release.clear()
        self.assertTrue(self.module.start_renderer_warmup(r))
        self.assertTrue(r.entered.wait(2))
        with ThreadPoolExecutor(max_workers=1) as caller:
            result = caller.submit(r.warmup, 'actual parameters')
            self.assertFalse(result.done())
            self.assertFalse(any(kind == 'chat' for kind, _ in self.events))
            r.release.set()
            self.assertEqual(result.result(5), 'actual parameters')
        self.assertEqual(self.active_guards, 0)

    def test_shutdown_joins_before_closing_cache(self):
        r = self.renderer
        r.release.clear()
        self.module.start_renderer_warmup(r)
        self.assertTrue(r.entered.wait(2))
        with ThreadPoolExecutor(max_workers=1) as caller:
            result = caller.submit(r.shutdown)
            self.assertFalse(result.done())
            self.assertNotIn(('close', None), self.events)
            r.release.set()
            result.result(5)
        self.assertEqual(self.events[-1], ('close', None))
        self.assertEqual(self.active_guards, 0)

    def test_failed_processor_retries_while_successful_one_is_reused(self):
        r = self.renderer
        r.fail_once = True
        with self.assertLogs('early_mm', level='WARNING'):
            self.assertTrue(self.module.start_renderer_warmup(r))
            r._glm53_early_mm_future.result(5)
        self.assertEqual(r.mm_processor.cache, [])
        r.warmup(None)
        self.assertEqual(r.mm_processor.calls, 2)
        self.assertEqual(r._readonly_mm_processor.calls, 1)
        self.assertEqual(r.mm_processor.cache, [])
        self.assertEqual(r._readonly_mm_processor.cache, [])

    def test_changed_processor_does_not_consume_another_instances_warmup(self):
        r = self.renderer
        self.module.start_renderer_warmup(r)
        r._glm53_early_mm_future.result(5)
        replacement = self.make_renderer().mm_processor
        r.mm_processor = replacement
        r.warmup(None)
        self.assertEqual(replacement.calls, 1)
        self.assertEqual(r._readonly_mm_processor.calls, 1)

    def test_disabled_and_unsupported_paths_do_not_schedule_work(self):
        r = self.renderer
        with patch.dict(os.environ, {'VLLM_GLM53_EARLY_MM_WARMUP': '0'}):
            self.assertFalse(self.module.start_renderer_warmup(r))
        with patch.dict(os.environ, {'VLLM_WORKER_MULTIPROC_METHOD': 'fork'}):
            self.assertFalse(self.module.start_renderer_warmup(r))
        for owner, name, value in ((r.config.model_config.hf_config, 'model_type', 'other'),
            (r.config.parallel_config, '_api_process_count', 2),
            (r.config.model_config.multimodal_config, 'mm_ipc_gpu_memory_gb', 1),
            (r.config.model_config.multimodal_config, 'mm_device_do_normalize', True),
            (r.config.model_config.multimodal_config, 'use_gpu_video_backend', lambda: True)):
            with patch.object(owner, name, value):
                self.assertFalse(self.module.start_renderer_warmup(r))
        self.assertFalse(r.entered.is_set())
        self.assertFalse(hasattr(r, '_glm53_early_mm_future'))

    def test_executor_failure_retains_original_methods(self):
        r = self.renderer
        before = r.warmup, r._warmup_mm_processor
        r._mm_executor.shutdown()
        with self.assertLogs('early_mm', level='WARNING'):
            self.assertFalse(self.module.start_renderer_warmup(r))
        self.assertEqual(before, (r.warmup, r._warmup_mm_processor))
        r.warmup(None)
        self.assertEqual(r.mm_processor.calls, 1)

    def test_backend_identity_ignores_frontend_scheduling(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch required')
        spec = importlib.util.spec_from_file_location('cache', ROOT / 'overlay/modules/glm53_model/glm53_startup_cache.py')
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        before = common.environment_identity()
        with patch.dict(os.environ, {'VLLM_GLM53_EARLY_MM_WARMUP': '0'}):
            self.assertEqual(before, common.environment_identity())

    def test_async_engine_schedules_after_input_budget_thread_guard(self):
        # Run the actual initialization statements with a guarded budget stub:
        # scheduling before InputProcessor would overlap two process-wide guards.
        tree = ast.parse((ROOT / 'overlay/modules/glm53_runtime/async_llm.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AsyncLLM')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        selected = []
        for node in init.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr in
                    ('renderer', 'input_processor', 'output_processor', 'engine_core') for t in node.targets):
                selected.append(node)
            if isinstance(node, ast.If) and 'VLLM_GLM53_EARLY_MM_WARMUP' in ast.unparse(node.test):
                selected.append(node)
        events = []
        renderer = types.SimpleNamespace(tokenizer=None)
        def budget(*args):
            self.assertNotIn('warmup', events, 'early warmup overlaps budget thread guard')
            events.append('budget')
        def start(r):
            self.assertIs(r, renderer)
            self.assertEqual(events, ['budget'])
            events.append('warmup')
        def engine(**kwargs):
            self.assertEqual(events, ['budget', 'warmup'])
            events.append('engine')
        owner = types.SimpleNamespace(vllm_config=types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(stream_interval=1)), log_stats=False)
        namespace = dict(self=owner, os=os, renderer_from_config=lambda config: renderer,
            InputProcessor=budget, OutputProcessor=lambda *a, **kw: None,
            EngineCoreClient=types.SimpleNamespace(make_async_mp_client=engine),
            vllm_config=owner.vllm_config, executor_class=None, tracing_endpoint=None,
            client_addresses=None, client_count=1, client_index=0)
        with patch.dict(sys.modules, {'vllm.renderers.glm53_renderer_warmup':
                                      types.SimpleNamespace(start_renderer_warmup=start)}):
            exec(compile(ast.Module(body=selected, type_ignores=[]), '<async-init>', 'exec'), namespace)
        self.assertEqual(events, ['budget', 'warmup', 'engine'])


if __name__ == '__main__':
    unittest.main()
