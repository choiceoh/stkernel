"""CPU receipt coverage and failure handling; no Docker or GPU is invoked."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'decode_next_cpu_runner', ROOT/'probes/run_decode_transport_sf_cpu.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def report(self, selected):
        return dict(expected_stages=['transport-00', 'transport-11'],
                    selected_stages=selected, stages=[])

    def test_partial_stage_success_does_not_claim_complete_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'report.json'
            result = self.report(['transport-11'])
            result['stages'] = [dict(name='transport-11', passed=True, cleanup=dict(passed=True))]
            runner.write_report(result, path)
            receipt = json.loads(path.read_text())
            self.assertTrue(receipt['selected_passed'])
            self.assertFalse(receipt['coverage_complete'])
            self.assertFalse(receipt['passed'])
            result['selected_stages'] = result['expected_stages']
            result['stages'].append(dict(name='transport-00', passed=True))
            runner.write_report(result, path)
            self.assertFalse(result['passed'], 'cleanup has not completed')
            result['stages'][-1]['cleanup'] = dict(passed=True)
            runner.write_report(result, path)
            self.assertTrue(result['passed'])
            self.assertTrue(result['coverage_complete'])

    def test_missing_or_duplicate_spec_cannot_hide_behind_pass(self):
        payload = ['compile.py', '--specs', 't,r|t,r,sf6']
        baseline = '[t,r] compiled in 1.0 s: kernel mac=48\n'
        candidate = '[t,r,sf6] compiled in 1.0 s: kernel mac=48\n'
        for text in (baseline + '[t,r,sf6] parse -> stock (nothing to compile)\nVERDICT: PASS\n',
                     baseline + candidate + candidate + 'VERDICT: PASS\n',
                     candidate + 'VERDICT: PASS\n'):
            with self.assertRaisesRegex(AssertionError, 'did not all compile'):
                runner.validate_compile_log(payload, text)
        runner.validate_compile_log(payload, baseline + candidate + 'VERDICT: PASS\n')

    def test_requested_dynamic_arm_must_also_compile(self):
        payload = ['compile.py', '--specs', 't,r|t,r,sf6', '--dynamic', 'tiled']
        static = '[t,r] compiled in 1 s\n[t,r,sf6] compiled in 1 s\n'
        with self.assertRaises(AssertionError):
            runner.validate_compile_log(payload, static + 'VERDICT: PASS\n')
        runner.validate_compile_log(payload, static + '[dynamic tiled=True] compiled in 1 s\nVERDICT: PASS\n')

    def test_payload_timeout_receipt_survives_both_cleanup_timeouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = self.report(['transport-11'])
            path = output/'report.json'
            calls = []

            def fail(command, **kwargs):
                calls.append(command)
                if command[0] == 'compile':
                    kwargs['stdout'].write('compiler partial output\n')
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])

            with patch.object(runner.subprocess, 'run', side_effect=fail):
                with self.assertRaises(subprocess.TimeoutExpired) as raised:
                    runner.run_stage('transport-11', [], ['compile'], 'owned-test-container',
                                     output, result, path)
            self.assertEqual(raised.exception.cmd, ['compile'])
            receipt = json.loads(path.read_text())
            record = receipt['stages'][0]
            self.assertFalse(receipt['passed'])
            self.assertIn('TimeoutExpired', record['error'])
            self.assertEqual(record['log_sha256'], hashlib.sha256((output/'stdout.log').read_bytes()).hexdigest())
            self.assertFalse(record['cleanup']['passed'])
            self.assertEqual(len(record['cleanup']['attempts']), 2)
            self.assertEqual(calls[1:], [['docker','stop','-t','1','owned-test-container'],
                                        ['docker','rm','-f','owned-test-container']])

    def test_successful_compile_with_failed_cleanup_is_not_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = dict(expected_stages=['sf-m6'], selected_stages=['sf-m6'], stages=[])
            payload = ['compile.py', '--specs', 't,r|t,r,sf6']

            def invoke(command, **kwargs):
                if command[0] == 'compile':
                    kwargs['stdout'].write('[t,r] compiled in 1 s\n[t,r,sf6] compiled in 1 s\nVERDICT: PASS\n')
                    return subprocess.CompletedProcess(command, 0)
                raise subprocess.TimeoutExpired(command, 10)

            path = output/'report.json'
            with patch.object(runner.subprocess, 'run', side_effect=invoke):
                with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                    runner.run_stage('sf-m6', payload, ['compile'], 'owned-test-container',
                                     output, report, path)
            self.assertFalse(json.loads(path.read_text())['passed'])
            self.assertFalse(report['stages'][0]['passed'])

    def test_absent_auto_removed_container_counts_as_clean(self):
        response = subprocess.CompletedProcess([], 1, '', 'Error: No such container: already-removed')
        with patch.object(runner.subprocess, 'run', return_value=response) as invoke:
            result = runner.cleanup_container('already-removed')
        self.assertTrue(result['passed'])
        self.assertEqual(invoke.call_count, 1)

    def test_transport_receipt_requires_requested_modes_and_no_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output/'result.json'
            payload = ['compile.py', '--compact', '0', '--inline', '0']
            result = dict(status='PASS', evidence='compile-only', modes=[0,0,0,0,0], cuda_initialized=False)
            path.write_text(json.dumps(result))
            runner.validate_stage('transport-00', payload, output)
            for field, value in (('modes', [1,1,0,0,0]), ('modes', [0,0,8,8,8]),
                                 ('cuda_initialized', True), ('evidence', 'gpu')):
                invalid = dict(result, **{field: value})
                path.write_text(json.dumps(invalid))
                with self.assertRaises(AssertionError):
                    runner.validate_stage('transport-00', payload, output)


if __name__ == '__main__':
    unittest.main()
