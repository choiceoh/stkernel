"""Exercise the minimal reproducer's order without initializing CUDA."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import subprocess
import tempfile
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ep_binding_probe', ROOT/'probes/glm53_ep_binding_check.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class BindingOrderTests(unittest.TestCase):
    def modules(self, count_code=0):
        calls = []
        cuda = SimpleNamespace(initialized=False)
        cuda.is_initialized = lambda: cuda.initialized
        cuda.synchronize = lambda: calls.append('synchronize')
        def empty(shape, device):
            self.assertEqual((shape, device), ((1,), 'cuda'))
            cuda.initialized = True
            calls.append('context')
            return SimpleNamespace(numel=lambda: 1)
        def count():
            self.assertTrue(cuda.initialized)
            calls.append('count')
            return count_code, None if count_code else 1
        def version():
            calls.append('version')
            return 0, 13000
        parent = ModuleType('cuda')
        bindings = ModuleType('cuda.bindings')
        bindings.driver = SimpleNamespace(cuDeviceGetCount=count, cuDriverGetVersion=version)
        parent.bindings = bindings
        return calls, cuda, {'torch': SimpleNamespace(cuda=cuda, empty=empty),
                            'cuda': parent, 'cuda.bindings': bindings}

    def test_context_precedes_first_binding_count(self):
        calls, cuda, modules = self.modules()
        result = {}
        with patch.dict(sys.modules, modules), patch.object(probe.importlib.metadata, 'version', return_value='13.3.1'), redirect_stdout(io.StringIO()):
            probe.observe(result)
        self.assertEqual(calls, ['context', 'synchronize', 'count', 'version'])
        self.assertFalse(result['cuda_initialized_before'])
        self.assertTrue(result['cuda_initialized_after'])
        self.assertEqual(result['count_result'], [0, 1])
        self.assertEqual(result['verdict'], 'OBSERVED')

    def test_existing_context_is_rejected_before_api_or_allocation(self):
        calls, cuda, modules = self.modules()
        cuda.initialized = True
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, 'fresh process'):
            probe.observe({})
        self.assertEqual(calls, [])

    def test_null_failed_api_result_is_preserved_and_fatal(self):
        _, _, modules = self.modules(count_code=3)
        result = {}
        with patch.dict(sys.modules, modules), patch.object(probe.importlib.metadata, 'version', return_value='13.3.1'), redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'returned an error'):
            probe.observe(result)
        self.assertEqual(result['count_result'], [3, None])


class DiagnosticLifecycleTests(unittest.TestCase):
    def test_nonzero_sanitizer_keeps_failure_and_restores_original(self):
        sys.path.insert(0, str(ROOT/'probes'))
        import run_glm53_ep_binding_offline as runner
        before = {node: dict(id=node, running=True, auto_remove=False,
                            image=runner.lifecycle.IMAGE, overlays={'source':'sha'},
                            manifest='sha', port=8000) for node in runner.lifecycle.NODES}
        stopped = {node: dict(state, running=False) for node, state in before.items()}
        transitions = []
        def transition(states, action):
            transitions.append(action)
            return stopped if action == 'stop' else before
        def process(command, **kwargs):
            if command[:2] == ['docker', 'run']:
                kwargs['stdout'].write('========= ERROR SUMMARY: 34 errors\n')
                return subprocess.CompletedProcess(command, 86)
            self.assertEqual(command[:2], ['docker', 'inspect'])
            return subprocess.CompletedProcess(command, 1, '', '')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'capture'
            args = ['runner', '--revision', 'a'*40, '--out', str(output)]
            with (patch.object(sys, 'argv', args),
                  patch.dict(runner.os.environ, {'FLEET_SESSION':'unit-binding'}),
                  patch.object(runner.signal, 'signal'),
                  patch.object(runner.lifecycle, 'check_holder'),
                  patch.object(runner.lifecycle, 'pinned'),
                  patch.object(runner.lifecycle, 'idle'),
                  patch.object(runner.lifecycle, 'snapshot', side_effect=[before, stopped]),
                  patch.object(runner.lifecycle, 'transition_all', side_effect=transition),
                  patch.object(runner.lifecycle, 'wait_restore', return_value=before),
                  patch.object(runner, 'resources', return_value={}),
                  patch.object(runner.sanitizer, 'preflight', return_value={'verdict':'PASS'}),
                  patch.object(runner.sanitizer, 'mount_args', return_value=[]),
                  patch.object(runner.subprocess, 'run', side_effect=process),
                  patch.object(runner.subprocess, 'check_output', return_value=''),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 1)
            result = json.loads((output/'completion.json').read_text())
            self.assertEqual(transitions, ['stop', 'start'])
            self.assertEqual(result['sanitizer_exit_code'], 86)
            self.assertTrue(result['restored_original'])
            self.assertFalse(result['performance_acceptance'])
            self.assertNotIn('sanitizer_summary', result)


if __name__ == '__main__':
    unittest.main()
