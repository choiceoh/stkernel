"""Small local Git fixtures for stable release recovery selection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_recovery as recovery
import fleet_validation as validation


OLD_HELPER = '''import hashlib, json, pathlib, subprocess, sys, time
assert sys.argv[1] == 'validate' and '--level' not in sys.argv
reused = '--verify-only' in sys.argv
if reused:
 path = sys.argv[sys.argv.index('--receipt') + 1]
 value = json.loads(pathlib.Path(path).read_text())
else:
 repo = sys.argv[sys.argv.index('--repo') + 1]
 store = pathlib.Path(sys.argv[sys.argv.index('--store') + 1])
 source = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
 context = dict(gate='overlay-deploy', profile='glm53', validator=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 key = hashlib.sha256(json.dumps([source,context],sort_keys=True).encode()).hexdigest()
 path = str(store / 'receipts' / (key + '.json'))
 pathlib.Path(path).parent.mkdir(parents=True,exist_ok=True)
 value = dict(key=key,context=context,profile='glm53',source=source,passed=True,coverage_complete=True,tests_run=1,completed_at=time.time())
 pathlib.Path(path).write_text(json.dumps(value))
 pathlib.Path(path).chmod(0o600)
assert value['context']['validator'] == hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
assert value['context']['gate'] == 'overlay-deploy'
print(json.dumps(dict(value, receipt=path, reused=reused)))
'''


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q', '-b', 'main')
        self.git('config', 'user.name', 'Recovery fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        (self.repo / 'bench').mkdir()
        (self.repo / 'bench/fleet_validation.py').write_text(OLD_HELPER)
        (self.repo / '.gitignore').write_text('__pycache__/\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'initial')
        self.remote = self.root / 'origin.git'
        self.git('clone', '--quiet', '--bare', str(self.repo), str(self.remote))
        self.git('remote', 'add', 'origin', str(self.remote))
        self.store = self.root / 'store'
        self.current = self.root / 'current_validator.py'
        self.current.write_text(OLD_HELPER)
        self.full_runs = 0
        self.api = types.SimpleNamespace(
            VERSION=1, __file__=str(self.current), held=mock.Mock(return_value=False),
            private_directory=validation.private_directory, lock=validation.lock,
            source=validation.source, run=mock.Mock(wraps=validation.run),
            recovery_info=validation.recovery_info, environment=validation.environment,
            validate=mock.Mock(side_effect=self.validate))

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args],
                                       text=True, stderr=subprocess.PIPE).strip()

    def advance(self):
        path = self.repo / 'change'
        path.write_text(path.read_text() + 'next\n' if path.exists() else 'next\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'next')
        self.git('push', '--quiet', 'origin', 'main')

    def validate(self, repo, store, profile, **options):
        self.assertEqual(profile, 'glm53')
        self.assertEqual(options['level'], 'release')
        self.assertTrue(options['use_profile_defaults'])
        if options.get('verify_only'):
            path = Path(options['require_receipt'])
            return dict(json.loads(path.read_text()), receipt=str(path), reused=True)
        self.full_runs += 1
        source = self.api.source(repo)
        context = dict(gate='overlay-deploy', version=1, profile=profile,
                       validator=hashlib.sha256(Path(self.api.__file__).read_bytes()).hexdigest())
        key = hashlib.sha256(json.dumps([source, context], sort_keys=True).encode()).hexdigest()
        path = self.api.private_directory(Path(store) / 'receipts') / (key + '.json')
        value = dict(version=1, key=key, context=context, source=source, profile=profile,
                     passed=True, coverage_complete=True, tests_run=1,
                     completed_at=self.full_runs)
        recovery.atomic_json(path, value)
        return dict(value, receipt=str(path), reused=False)

    def prepare(self, **options):
        return recovery.prepare(self.api, self.repo, self.store, **options)

    def pointer(self):
        key = hashlib.sha256(str(self.repo).encode()).hexdigest()
        return self.store / 'recovery-current' / (key + '.json')

    def test_default_reuses_approved_source_after_main_advances_and_fetches_once(self):
        first = self.prepare()
        self.advance()
        self.api.run.reset_mock()
        second = self.prepare()
        self.assertEqual(second['source'], first['source'])
        self.assertEqual(second['selection'], 'pinned')
        self.assertTrue(second['reused'])
        self.assertEqual(self.full_runs, 1)
        fetches = [c for c in self.api.run.call_args_list if c.args[0][:2] == ['git', 'fetch']]
        self.assertEqual(len(fetches), 1)

    def test_refresh_updates_only_after_success_and_migration_selects_newest_release(self):
        first = self.prepare()
        self.advance()
        second = self.prepare(refresh=True)
        self.assertNotEqual(first['source'], second['source'])
        self.assertEqual(second['selection'], 'latest-main')
        self.assertEqual(self.full_runs, 2)
        self.advance()
        old_pointer = self.pointer().read_bytes()
        with mock.patch.object(self.api, 'validate', side_effect=ValueError('release gate failed')):
            with self.assertRaisesRegex(ValueError, 'release gate failed'):
                self.prepare(refresh=True)
        self.assertEqual(self.pointer().read_bytes(), old_pointer)
        self.pointer().unlink()
        migrated = self.prepare()
        self.assertEqual(migrated['receipt'], second['receipt'])
        self.assertEqual(migrated['selection'], 'existing')
        self.assertEqual(self.full_runs, 2)

    def test_rewritten_main_is_rejected_without_promoting_a_new_recovery(self):
        self.prepare()
        old_pointer = self.pointer().read_bytes()
        rewritten = self.git('commit-tree', 'HEAD^{tree}', '-m', 'unrelated history')
        self.git('push', '--quiet', '--force', 'origin', rewritten + ':main')
        for refresh in (False, True):
            with self.assertRaisesRegex(recovery.ApprovalChanged, 'no longer an ancestor'):
                self.prepare(refresh=refresh)
        self.assertEqual(self.full_runs, 1)
        self.assertEqual(self.pointer().read_bytes(), old_pointer)

    def test_old_approved_helper_verifies_without_running_current_release_gate(self):
        self.api.__file__ = str(self.repo / 'bench/fleet_validation.py')
        first = self.prepare()
        self.current.write_text('# newer controller\n')
        self.api.__file__ = str(self.current)
        self.advance()
        with mock.patch.object(self.api, 'validate', side_effect=AssertionError('must not execute a CPU gate')):
            result = self.prepare()
            self.assertEqual(result['receipt'], first['receipt'])
            self.assertEqual(recovery.verify(self.api, first['receipt'])['source'], first['source'])
        commands = [c.args[0] for c in self.api.run.call_args_list if c.args[0][0] == sys.executable]
        self.assertTrue(commands)
        self.assertTrue(all('--verify-only' in argv and argv[2] == 'validate' for argv in commands))
        self.assertEqual(self.full_runs, 1)

    def test_admission_evidence_and_changed_checkout_cannot_be_recovery(self):
        first = self.prepare()
        path = Path(first['validation_receipt'])
        original = path.read_bytes()
        value = json.loads(original)
        value['context']['gate'] = 'overlay-admission'
        recovery.atomic_json(path, value)
        self.api.validate.reset_mock()
        with self.assertRaisesRegex(ValueError, 'complete release CPU evidence'):
            recovery.verify(self.api, first['receipt'])
        self.api.validate.assert_not_called()
        path.write_bytes(original)
        (Path(first['repo']) / 'bench/fleet_validation.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'clean source'):
            recovery.verify(self.api, first['receipt'])

    def test_delegated_verification_detects_receipt_change(self):
        self.api.__file__ = str(self.repo / 'bench/fleet_validation.py')
        first = self.prepare()
        self.current.write_text('# newer controller\n')
        self.api.__file__ = str(self.current)
        original_run = self.api.run
        def change_after_read(argv, *args, **kwargs):
            output = original_run(argv, *args, **kwargs)
            if argv[0] == sys.executable:
                path = Path(first['validation_receipt'])
                value = json.loads(path.read_text())
                value['completed_at'] += 1
                recovery.atomic_json(path, value)
            return output
        with mock.patch.object(self.api, 'run', side_effect=change_after_read):
            with self.assertRaisesRegex(ValueError, 'changed during verification'):
                recovery.verify(self.api, first['receipt'])

    def test_hold_never_starts_preparation_and_missing_old_helper_fails_closed(self):
        self.git('rm', 'bench/fleet_validation.py')
        self.git('commit', '-qm', 'pre-helper fixture')
        self.git('push', '--quiet', 'origin', 'main')
        first = self.prepare()
        self.api.held.return_value = True
        self.api.run.reset_mock()
        with self.assertRaisesRegex(ValueError, 'before GPU reservation'):
            self.prepare()
        self.api.run.assert_not_called()
        recovery.verify(self.api, first['receipt'])
        self.assertEqual(self.full_runs, 1)
        self.api.__file__ = str(self.root / 'new_validator.py')
        Path(self.api.__file__).write_text('# next current helper\n')
        with self.assertRaisesRegex(ValueError, 'validator is unavailable'):
            recovery.verify(self.api, first['receipt'])

    def test_setup_and_refresh_use_source_helper_after_controller_changes(self):
        self.current.write_text('# newer controller\n')
        with mock.patch.object(self.api, 'validate', side_effect=AssertionError('use approved source helper')):
            first = self.prepare()
            expected = hashlib.sha256(OLD_HELPER.encode()).hexdigest()
            self.assertEqual(json.loads(Path(first['validation_receipt']).read_text())['context']['validator'], expected)
            self.current.write_text('# another controller revision\n')
            self.assertEqual(recovery.verify(self.api, first['receipt'])['source'], first['source'])
            self.advance()
            second = self.prepare(refresh=True)
            self.assertNotEqual(first['source'], second['source'])
            self.current.write_text('# controller advances again\n')
            self.assertEqual(recovery.verify(self.api, second['receipt'])['source'], second['source'])
        commands = [c.args[0] for c in self.api.run.call_args_list if c.args[0][0] == sys.executable]
        self.assertEqual(sum('--verify-only' not in argv for argv in commands), 2)
        self.assertTrue(all('--level' not in argv for argv in commands))
        self.assertEqual(self.full_runs, 0)


if __name__ == '__main__':
    unittest.main()
