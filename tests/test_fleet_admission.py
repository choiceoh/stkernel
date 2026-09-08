"""Small source fixtures for the GPU experiment admission contract."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_admission as admission


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.env = dict(PATH=os.environ.get('PATH', os.defpath), HOME=str(self.root),
                        CUDA_VISIBLE_DEVICES='', LANG='C', LC_ALL='C')
        self.git('init', '-q', '-b', 'main')
        self.git('config', 'user.name', 'admission fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        source = Path(__file__).resolve().parents[1]
        for name in ('launchers/compose-overlays.sh', 'launchers/lib/common-tp4.sh'):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
        files = {
            '.gitignore': 'build/\n__pycache__/\n',
            'bench/fleet_validation.py': '# fixture validator\n',
            'profiles/demo.env': 'MODULES="sample"\nTARGET_PREFIX="/packages/"\n',
            'profiles/unselected.env': 'this unrelated profile is invalid (\n',
            'launchers/deploy-overlays.sh': '#!/bin/bash\nexit 81\n',
            'launchers/check-glm53-chat.sh': '#!/bin/bash\nexit 82\n',
            'bench/cpu_checks.py': 'raise AssertionError("full CPU suite ran")\n',
            'tests/test_logic.py': 'raise AssertionError("logic suite ran")\n',
            'overlay/modules/sample/manifest.tsv': 'sample.py\tvllm/sample.py\tabsent\n',
            # Successful compile must never import torch or execute this code.
            'overlay/modules/sample/sample.py': 'import unavailable_torch\nraise AssertionError("overlay executed")\n',
        }
        for name, content in files.items():
            self.write(name, content)
        self.commit()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True,
                                       stderr=subprocess.PIPE).strip()

    def write(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')

    def check(self):
        return admission.check(self.repo, 'demo', self.env)

    def test_selected_profile_passes_without_executing_heavy_checks_or_overlays(self):
        result = self.check()
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['selected'], list(admission.CHECKS))
        self.assertEqual(result['tests_run'], 3)
        self.assertEqual(result['checks'][-1]['python_files'], 1)
        self.assertFalse((self.repo / 'build/unselected').exists())
        self.assertFalse(list((self.repo / 'build').rglob('*.pyc')))

    def test_invalid_python_and_shell_fail_with_incomplete_evidence(self):
        for filename, code in [('overlay/modules/sample/sample.py', 'def broken(:\n'),
                               ('profiles/demo.env', 'if then\n')]:
            with self.subTest(filename=filename):
                original = (self.repo / filename).read_text()
                self.write(filename, code)
                result = self.check()
                self.assertFalse(result['passed'])
                self.assertFalse(result['coverage_complete'])
                self.assertTrue(result['failed'])
                self.write(filename, original)

    def test_bad_manifest_contracts_and_duplicates_fail(self):
        rows = [
            'sample.py\tvllm/sample.py\tnot-a-hash\n',
            '../sample.py\tvllm/sample.py\tabsent\n',
            'sample.py\t../escape.py\tabsent\n',
            'sample.py\tvllm/sample.py\tabsent\nother.py\tvllm/sample.py\tabsent\n',
            'sample.py\tvllm/sample.py\tabsent\nother.py\t/packages/vllm/sample.py\tabsent\n',
        ]
        self.write('overlay/modules/sample/other.py', 'x = 1\n')
        for text in rows:
            with self.subTest(text=text):
                self.write('overlay/modules/sample/manifest.tsv', text)
                self.assertFalse(self.check()['passed'])

    def test_external_symlink_source_is_rejected_before_composition(self):
        outside = self.root / 'external.py'
        outside.write_text('x = 1\n')
        overlay = self.repo / 'overlay/modules/sample/sample.py'
        overlay.unlink()
        overlay.symlink_to(outside)
        result = self.check()
        self.assertFalse(result['passed'])
        self.assertIn('repository file', result['failed'][0]['error'])
        self.assertFalse((self.repo / 'build').exists())

    def test_identity_is_stable_but_binds_source_profile_validator_and_tools(self):
        first, spec = admission.identity(self.repo, 'demo', self.env)
        self.assertEqual(first, admission.identity(self.repo, 'demo', self.env)[0])
        self.assertEqual(spec['command'][1:3], ['-I', '-S'])
        self.assertEqual(spec['context']['gate'], 'overlay-admission')
        changed, _ = admission.identity(self.repo, 'demo', self.env, validator_sha='changed')
        self.assertNotEqual(first['key'], changed['key'])
        real_sha = admission.sha
        with mock.patch.object(admission, 'sha', side_effect=lambda p:
                               'changed-binary' if str(p) == shutil.which('bash', path=self.env['PATH'])
                               else real_sha(p)):
            self.assertNotEqual(first['key'], admission.identity(self.repo, 'demo', self.env)[0]['key'])
        self.write('overlay/modules/sample/sample.py', 'new_source = 1\n')
        with self.assertRaisesRegex(ValueError, 'clean source'):
            admission.identity(self.repo, 'demo', self.env)
        self.commit()
        self.assertNotEqual(first['key'], admission.identity(self.repo, 'demo', self.env)[0]['key'])

    def test_identical_pinned_checker_copy_preserves_receipt_identity(self):
        first, spec = admission.identity(self.repo, 'demo', self.env)
        pinned = self.root / 'pinned' / 'fleet_admission.py'
        pinned.parent.mkdir()
        shutil.copyfile(admission.__file__, pinned)
        with mock.patch.object(admission, '__file__', str(pinned)):
            second, pinned_spec = admission.identity(self.repo, 'demo', self.env)
        self.assertNotEqual(spec['command'], pinned_spec['command'])
        self.assertEqual(first, second)

    def test_cli_emits_pass_and_failure_report(self):
        report = self.root / 'report.json'
        command = [sys.executable, '-I', '-S', str(Path(admission.__file__).resolve()),
                   '--repo', str(self.repo), '--profile', 'demo', '--out', str(report)]
        result = subprocess.run(command, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(report.read_text())['passed'])
        self.write('overlay/modules/sample/sample.py', 'broken syntax !\n')
        result = subprocess.run(command, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(report.read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
