"""Documentation equivalence cannot discard runnable or declared dependencies."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import cpu_evidence
import fleet_source as source


class SourceFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'fixture')
        for path, content in {'input.py':'print(1)\n', 'docs/report.md':'prose\n',
                'measurements/run/result.json':'{}\n', 'measurements/run/check.py':'pass\n',
                'profiles/README.md':'runtime contract\n', 'README.md':'intro\n',
                'bench/EXPERIMENTS.md':'usage\n'}.items():
            self.write(path, content)
        self.commit()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True).strip()

    def write(self, path, content):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def advance_ref(self, path, content):
        old = self.git('rev-parse', 'HEAD')
        self.write(path, content)
        new = self.commit()
        self.git('update-ref', 'refs/remotes/origin/main', new)
        self.git('checkout', '--detach', '-q', old)
        return old


class SourceTests(SourceFixture):
    def test_doc_and_measurement_commit_preserves_effective_source(self):
        before = source.identity(self.repo)
        self.write('docs/report.md', 'new prose\n')
        self.write('measurements/run/result.json', '{"value": 3}\n')
        self.write('bench/EXPERIMENTS.md', 'new usage\n')
        self.commit()
        self.assertTrue(source.compare(before, source.identity(self.repo))['equal'])
        self.write('input.py', 'print(2)\n')
        self.assertEqual(source.compare(before, source.identity(self.repo))['changed_paths'], ['input.py'])

    def test_script_mode_symlink_and_declared_docs_remain_dependencies(self):
        protected = ['docs/report.md', 'measurements/run/result.json']
        before = source.identity(self.repo, protected_paths=protected)
        self.write('docs/report.md', 'changed fixture input\n')
        self.write('measurements/run/result.json', '{"input": 1}\n')
        self.write('measurements/run/check.py', 'raise SystemExit(1)\n')
        self.write('profiles/README.md', 'broken runtime contract\n')
        self.write('README.md', '#!/bin/sh\nexit 1\n').chmod(0o755)
        (self.repo / 'docs/link.md').symlink_to('../input.py')
        changed = source.compare(before, source.identity(self.repo, protected_paths=protected))['changed_paths']
        self.assertEqual(set(changed), set(protected + ['measurements/run/check.py', 'profiles/README.md',
                                                      'README.md', 'docs/link.md']))

    def test_required_main_allows_only_upstream_documentation(self):
        self.advance_ref('docs/report.md', 'upstream docs\n')
        self.write('input.py', 'candidate code\n')
        self.commit()
        result = source.require_base(self.repo, 'origin/main')
        self.assertTrue(result['ok'])
        self.assertEqual(result['ignored_changes'], ['docs/report.md'])
        with self.assertRaises(source.SourceMismatch) as caught:
            source.require_base(self.repo, 'origin/main', protected_paths=['docs/report.md'])
        self.assertEqual(caught.exception.details['relevant_changes'], ['docs/report.md'])

    def test_required_main_rejects_real_code_even_if_candidate_has_other_changes(self):
        self.advance_ref('input.py', 'upstream fix\n')
        self.write('other.py', 'candidate\n')
        self.commit()
        with self.assertRaises(source.SourceMismatch) as caught:
            source.require_base(self.repo, 'origin/main')
        self.assertEqual(caught.exception.details['relevant_changes'], ['input.py'])

    def test_measurement_runner_and_new_executable_are_required_main(self):
        self.advance_ref('measurements/run/check.py', 'new executable logic\n')
        with self.assertRaisesRegex(source.SourceMismatch, 'measurements/run/check.py'):
            source.require_base(self.repo, 'origin/main')
        self.git('checkout', '--detach', '-q', 'origin/main')
        old = self.git('rev-parse', 'HEAD')
        (self.repo / 'docs/report.md').chmod(0o755)
        self.commit()
        self.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        self.git('checkout', '--detach', '-q', old)
        with self.assertRaisesRegex(source.SourceMismatch, 'docs/report.md'):
            source.require_base(self.repo, 'origin/main')

    def test_literal_wrapper_guard_runs_before_gpu_work(self):
        wrappers = [p for p in (ROOT / 'probes').glob('run_*.sh')
                    if 'python3 bench/fleet_source.py require-base origin/main' in p.read_text()]
        self.assertEqual(len(wrappers), 8)
        self.advance_ref('docs/report.md', 'new prose\n')
        # Run the same helper command, without any wrapper service/GPU actions.
        command = [sys.executable, str(ROOT / 'bench/fleet_source.py'), 'require-base', 'origin/main']
        result = subprocess.run(command, cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['ok'])

    def test_declared_ignored_input_is_included(self):
        self.write('.gitignore', 'measurements/private/\n')
        self.commit()
        self.write('measurements/private/input.json', '{}\n')
        before = source.identity(self.repo, protected_paths=['measurements/private'])
        self.assertIn('measurements/private/input.json', before['files'])
        self.write('measurements/private/input.json', '{"changed": true}\n')
        after = source.identity(self.repo, protected_paths=['measurements/private'])
        self.assertEqual(source.compare(before, after)['changed_paths'], ['measurements/private/input.json'])

    def test_prior_evidence_environment_input_is_never_discarded(self):
        with mock.patch.dict(os.environ, INPUT_REUSE_GPU_EVIDENCE=str(self.repo / 'measurements/run/result.json')):
            before = source.identity(self.repo)
            self.advance_ref('measurements/run/result.json', '{"changed_input": 1}\n')
            with self.assertRaisesRegex(source.SourceMismatch, 'measurements/run/result.json'):
                source.require_base(self.repo, 'origin/main')
            self.write('measurements/run/result.json', '{"changed_input": 1}\n')
            self.assertEqual(source.compare(before, source.identity(self.repo))['changed_paths'],
                             ['measurements/run/result.json'])

    def test_wrapper_audit_closes_when_execution_contract_changes(self):
        for name in source.WRAPPER_AUDIT:
            self.assertTrue(source.audited_wrapper(ROOT / name, ROOT), name)
        name = next(iter(source.WRAPPER_AUDIT))
        self.write(name, (ROOT / name).read_text())
        self.assertTrue(source.audited_wrapper(name, self.repo))
        self.write(name, (ROOT / name).read_text() + '\n# new dependency\n')
        self.assertFalse(source.audited_wrapper(name, self.repo))


class CpuSourceTests(SourceFixture):
    def setUp(self):
        super().setUp()
        for name in cpu_evidence.LOGIC_SOURCE_AUDIT:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.commit()
        self.spec = dict(command=[sys.executable, 'bench/cpu_checks.py', '--suite', 'logic'],
                         env={}, context={}, inputs=[], timeout_s=120, outputs=[])

    def cpu_identity(self):
        # The environment probe is independent of file projection; pin its
        # output to make source fixtures deterministic and avoid package scans.
        runtime = subprocess.CompletedProcess([], 0, '[]', '')
        original = subprocess.run
        def run(argv, *args, **kwargs):
            if len(argv) > 2 and argv[1] == '-c':
                return runtime
            return original(argv, *args, **kwargs)
        with mock.patch.object(cpu_evidence.subprocess, 'run', side_effect=run):
            return cpu_evidence.identity(self.repo, self.spec, {'PATH':os.environ['PATH']})

    def test_logic_docs_reuse_reports_and_real_changes_invalidate(self):
        before = self.cpu_identity()
        self.assertEqual(before['scope'], 'audited-logic')
        self.write('docs/report.md', 'changed docs\n')
        self.write('measurements/run/result.json', '{"report": 2}\n')
        self.write('bench/EXPERIMENTS.md', 'changed instructions\n')
        self.commit()
        self.assertEqual(self.cpu_identity(), before)
        self.write('input.py', 'changed source\n')
        self.commit()
        difference = cpu_evidence.difference(before, self.cpu_identity())
        self.assertFalse(difference['equal'])
        self.assertEqual(difference['changed_paths'], ['input.py'])
        self.assertEqual(difference['changed_components'], [])

    def test_declared_measurement_input_and_environment_remain_bound(self):
        self.spec['inputs'] = [str(self.repo / 'measurements/run/result.json')]
        before = self.cpu_identity()
        self.write('measurements/run/result.json', '{"runtime_input": 5}\n')
        self.commit()
        difference = cpu_evidence.difference(before, self.cpu_identity())
        self.assertEqual(difference['changed_paths'], ['measurements/run/result.json'])
        self.assertIn('inputs', difference['changed_components'])
        before = self.cpu_identity()
        self.spec['env'] = {'TEST_MODE':'changed'}
        self.assertIn('environment', cpu_evidence.difference(before, self.cpu_identity())['changed_components'])

    def test_changed_audited_test_falls_back_and_docs_then_invalidate(self):
        path = self.repo / 'tests/test_logic.py'
        path.write_text(path.read_text() + '\n# new unaudited read path\n')
        self.commit()
        before = self.cpu_identity()
        self.assertEqual(before['scope'], 'full-tree')
        self.write('docs/report.md', 'potential new test input\n')
        self.commit()
        self.assertNotEqual(self.cpu_identity()['key'], before['key'])


if __name__ == '__main__':
    unittest.main()
