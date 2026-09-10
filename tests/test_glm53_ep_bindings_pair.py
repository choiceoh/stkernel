"""Pure CPU contracts for the isolated bindings pair; never launch Docker/CUDA."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
import glm53_ep_bindings_pair_check as probe
import run_glm53_ep_bindings_pair_offline as runner

MANIFEST_SHA = 'b'*64


class ImportIdentityTests(unittest.TestCase):
    def test_actual_import_paths_hashes_and_both_distribution_versions_are_bound(self):
        self.identity_fixture()

    def test_wrong_import_origin_fails(self):
        self.identity_fixture('origin')

    def test_wrong_binary_hash_fails(self):
        self.identity_fixture('hash')

    def test_global_metapackage_cannot_masquerade_as_candidate(self):
        self.identity_fixture('metadata')

    def identity_fixture(self, bad=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            files, modules, distributions = {}, {}, {}
            for name, relative in probe.MODULE_FILES.items():
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                files[relative] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                modules[name] = SimpleNamespace(__file__=str(path))
            for name in ('cuda-bindings', 'cuda-python'):
                relative = name.replace('-', '_')+'-13.0.3.dist-info/METADATA'
                path = root/relative
                path.parent.mkdir()
                text = 'Name: '+name+'\nVersion: 13.0.3\n'
                path.write_text(text)
                files[relative] = {'sha256': hashlib.sha256(text.encode()).hexdigest()}
                distributions[name] = SimpleNamespace(
                    version='13.0.3', locate_file=lambda relative: root/relative,
                    read_text=lambda filename, text=text: text)
            if bad == 'origin':
                modules['cuda.bindings.driver'].__file__ = str(root/'wrong.so')
            elif bad == 'hash':
                files[probe.MODULE_FILES['cuda.bindings.driver']]['sha256'] = '0'*64
            elif bad == 'metadata':
                distributions['cuda-python'].version = '13.3.1'
            with (patch.object(probe, 'validate_capsule', return_value={'files': files}) as validate,
                  patch.object(probe.importlib, 'import_module', side_effect=modules.__getitem__),
                  patch.object(probe.importlib.metadata, 'distribution', side_effect=distributions.__getitem__)):
                if bad:
                    with self.assertRaises(RuntimeError):
                        probe.verify_import_identity('candidate', root, MANIFEST_SHA)
                else:
                    result = probe.verify_import_identity('candidate', root, MANIFEST_SHA)
                    self.assertEqual(set(result['modules']), set(probe.MODULE_FILES))
                    self.assertEqual(set(result['distributions']), {'cuda-bindings', 'cuda-python'})
                    validate.assert_called_once_with(root, MANIFEST_SHA)


class ProbeOrderTests(unittest.TestCase):
    def test_both_arms_keep_torch_then_identity_then_first_api(self):
        for arm in runner.VERSIONS:
            with self.subTest(arm=arm):
                self.observe(arm)

    def test_identity_failure_prevents_both_binding_calls(self):
        self.observe('candidate', bad_identity=True)

    def test_unexpected_count_or_driver_version_is_not_observed(self):
        self.observe('candidate', count=2)
        self.observe('candidate', version=13030)

    def observe(self, arm, bad_identity=False, count=1, version=13000):
        calls = []
        cuda_state = SimpleNamespace(initialized=False)
        cuda_state.is_initialized = lambda: cuda_state.initialized
        cuda_state.synchronize = lambda: calls.append('synchronize')
        def empty(shape, device):
            self.assertEqual((shape, device), ((1,), 'cuda'))
            cuda_state.initialized = True
            calls.append('context')
            return SimpleNamespace(numel=lambda: 1)
        def identity(*args):
            self.assertTrue(cuda_state.initialized)
            calls.append('identity')
            if bad_identity:
                raise RuntimeError('wrong binding identity')
            return {'version': runner.VERSIONS[arm]}
        bindings = ModuleType('cuda.bindings')
        bindings.driver = SimpleNamespace(
            cuDeviceGetCount=lambda: (calls.append('count') or (0, count)),
            cuDriverGetVersion=lambda: (calls.append('version') or (0, version)))
        parent = ModuleType('cuda')
        parent.bindings = bindings
        modules = {'torch': SimpleNamespace(cuda=cuda_state, empty=empty),
                   'cuda': parent, 'cuda.bindings': bindings}
        result = {}
        with (patch.dict(sys.modules, modules), patch.object(probe, 'verify_import_identity', side_effect=identity),
              patch.object(probe.importlib.metadata, 'version', return_value=runner.VERSIONS[arm]),
              redirect_stdout(io.StringIO())):
            if bad_identity or count != 1 or version != 13000:
                with self.assertRaises(RuntimeError):
                    probe.observe(result, arm, Path('/capsule'), MANIFEST_SHA)
                self.assertNotEqual(result.get('verdict'), 'OBSERVED')
            else:
                probe.observe(result, arm, Path('/capsule'), MANIFEST_SHA)
                self.assertEqual(result['verdict'], 'OBSERVED')
        expected = ['context', 'synchronize', 'identity']
        self.assertEqual(calls, expected if bad_identity else expected+['count', 'version'])


class PairLifecycleTests(unittest.TestCase):
    def test_v5_reports_must_be_between_unique_count_markers_after_identity(self):
        prefix = ('EP_BINDING_CONTEXT_BEGIN\nEP_BINDING_CONTEXT_READY\n'
                  'EP_BINDING_IDENTITY_VERIFIED\n')
        begin, end = 'EP_BINDING_DEVICE_COUNT_BEGIN\n', 'EP_BINDING_DEVICE_COUNT_END\n'
        error = '========= Program hit CUDA_ERROR_INVALID_VALUE in cuGetProcAddress_v2.\n'
        good = prefix+begin+error*34+end
        self.assertTrue(runner.v5_error_window(good))
        for bad in (prefix+error+begin+error*33+end,
                    prefix+begin+error*33+end+error,
                    prefix+begin+begin+error*34+end,
                    good.replace('EP_BINDING_IDENTITY_VERIFIED\n', ''),
                    begin+prefix+error*34+end):
            with self.subTest(log=bad):
                self.assertFalse(runner.v5_error_window(bad))

    def test_baseline_failure_candidate_runs_one_pause_exact_restore_and_failure_exit(self):
        result, commands, transitions = self.run_pair()
        self.assertEqual(result['verdict'], 'COMPATIBILITY_OBSERVED')
        self.assertTrue(result['pair_completed'])
        self.assertTrue(result['capsule_unchanged_after_pair'])
        self.assertEqual([c['verdict'] for c in result['cells']], ['FAIL', 'CLEAN_DIAGNOSTIC'])
        self.assertEqual(result['cells'][0]['sanitizer_exit_code'], 86)
        self.assertTrue(result['cells'][0]['matches_v5_reproducer'])
        self.assertEqual(transitions, ['stop', 'start'])
        self.assertNotEqual(commands[0][commands[0].index('--name')+1], commands[1][commands[1].index('--name')+1])
        for command in commands:
            self.assertIn('--error-exitcode=86', command)
            self.assertEqual(command[command.index('--tool')+1], 'memcheck')
            self.assertIn('PYTHONDONTWRITEBYTECODE=1', command)
            self.assertIn('PYTHONNOUSERSITE=1', command)
            self.assertIn('--network=none', command)
            self.assertIn('--memory=4g', command)
            self.assertIn('--memory-swap=4g', command)
            self.assertIn('--cpus=2', command)
            self.assertIn(runner.lifecycle.IMAGE, command)
            self.assertTrue(any('target='+runner.CAPSULE+',readonly' in arg for arg in command))
            self.assertFalse(any('suppress' in arg for arg in command))
        self.assertIn('PYTHONPATH=', commands[0])
        self.assertIn('PYTHONPATH='+runner.CAPSULE, commands[1])

    def test_candidate_sanitizer_failure_cannot_observe_compatibility(self):
        result, commands, _ = self.run_pair(candidate_errors=1)
        self.assertEqual(len(commands), 2)
        self.assertEqual(result['verdict'], 'FAIL')

    def test_unmatched_baseline_error_cannot_be_waived(self):
        result, commands, _ = self.run_pair(baseline_errors=1)
        self.assertEqual(len(commands), 2)
        self.assertEqual(result['verdict'], 'FAIL')

    def test_capsule_drift_at_candidate_go_aborts_and_restores(self):
        result, commands, transitions = self.run_pair(capsule_failure=3)
        self.assertEqual(len(commands), 1)
        self.assertFalse(result['pair_completed'])
        self.assertEqual(transitions, ['stop', 'start'])

    def test_capsule_drift_after_pair_prevents_compatibility_verdict(self):
        result, commands, _ = self.run_pair(capsule_failure=4)
        self.assertEqual(len(commands), 2)
        self.assertEqual(result['verdict'], 'FAIL')

    def test_stopped_donor_is_restored_stopped_without_idle_or_boot(self):
        result, _, transitions = self.run_pair(running=False)
        self.assertEqual(transitions, ['stop'])
        self.assertEqual(result['incoming_mode'], 'stopped')

    def test_interruption_stops_pair_and_restores_original(self):
        result, commands, transitions = self.run_pair(interrupt=True)
        self.assertEqual(len(commands), 1)
        self.assertFalse(result['pair_completed'])
        self.assertEqual(transitions, ['stop', 'start'])

    def run_pair(self, *, running=True, baseline_errors=34, candidate_errors=0,
                 capsule_failure=None, interrupt=False):
        before = {node: dict(id=node, running=running, auto_remove=False,
                            image=runner.lifecycle.IMAGE, overlays={'source': 'sha'},
                            manifest='sha', port=8000) for node in runner.lifecycle.NODES}
        stopped = {node: dict(state, running=False) for node, state in before.items()}
        transitions, commands, capsule_checks = [], [], []
        def transition(states, action):
            transitions.append(action)
            return stopped if action == 'stop' else before
        def validate(*args):
            capsule_checks.append(args)
            if len(capsule_checks) == capsule_failure:
                raise ValueError('capsule changed')
            return {'schema': 1, 'files': {}}
        with tempfile.TemporaryDirectory() as directory:
            output, capsule = Path(directory)/'capture', Path(directory)/'capsule'
            capsule.mkdir()
            def process(command, **kwargs):
                if command[:2] != ['docker', 'run']:
                    self.assertEqual(command[:2], ['docker', 'inspect'])
                    return subprocess.CompletedProcess(command, 1, '', '')
                commands.append(command)
                if interrupt:
                    raise InterruptedError('termination requested')
                arm = command[command.index('--arm')+1]
                errors = baseline_errors if arm == 'baseline' else candidate_errors
                kwargs['stdout'].write('EP_BINDING_CONTEXT_BEGIN\nEP_BINDING_CONTEXT_READY\n'
                                       'EP_BINDING_IDENTITY_VERIFIED\nEP_BINDING_DEVICE_COUNT_BEGIN\n')
                kwargs['stdout'].write('========= Program hit CUDA_ERROR_INVALID_VALUE in cuGetProcAddress_v2.\n'*errors)
                kwargs['stdout'].write('EP_BINDING_DEVICE_COUNT_END\n')
                kwargs['stdout'].write(f'========= ERROR SUMMARY: {errors} errors\n')
                inner = dict(verdict='OBSERVED', arm=arm, performance_acceptance=False,
                             full_gpu_acceptance=False, capsule_manifest_sha256=MANIFEST_SHA,
                             cuda_bindings=runner.VERSIONS[arm], cuda_initialized_before=False,
                             cuda_initialized_after=True, count_result=[0, 1], version_result=[0, 13000],
                             binding_identity={'version': runner.VERSIONS[arm]})
                (output/(arm+'.json')).write_text(json.dumps(inner))
                return subprocess.CompletedProcess(command, 86 if errors else 0)
            args = ['pair', '--revision', 'a'*40, '--out', str(output),
                    '--capsule-root', str(capsule), '--manifest-sha256', MANIFEST_SHA]
            with (patch.object(sys, 'argv', args), patch.dict(runner.os.environ, {'FLEET_SESSION': 'unit-pair'}),
                  patch.object(runner.signal, 'signal'), patch.object(runner.lifecycle, 'check_holder'),
                  patch.object(runner.lifecycle, 'pinned') as pinned,
                  patch.object(runner.lifecycle, 'idle') as idle,
                  patch.object(runner.lifecycle, 'restore_public') as public_restore,
                  patch.object(runner.lifecycle, 'snapshot', side_effect=[before, stopped, stopped]),
                  patch.object(runner.lifecycle, 'transition_all', side_effect=transition),
                  patch.object(runner.lifecycle, 'wait_restore', return_value=before),
                  patch.object(runner, 'resources', return_value={}),
                  patch.object(runner, 'validate_capsule', side_effect=validate),
                  patch.object(runner.sanitizer, 'preflight', return_value={'verdict': 'PASS'}),
                  patch.object(runner.sanitizer, 'mount_args', return_value=[]),
                  patch.object(runner.subprocess, 'run', side_effect=process),
                  patch.object(runner.subprocess, 'check_output', return_value=''),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 1)
            result = json.loads((output/'completion.json').read_text())
            self.assertEqual(result['exit_code'], 1)
            self.assertTrue(result['restored_original'])
            self.assertFalse(result['performance_acceptance'])
            self.assertFalse(result['full_gpu_acceptance'])
            self.assertEqual(json.loads((output/'restored.json').read_text()), before)
            self.assertGreaterEqual(pinned.call_count, len(commands)+1)
            if not running:
                idle.assert_not_called()
            public_restore.assert_not_called()
            return result, commands, transitions


if __name__ == '__main__':
    unittest.main()
