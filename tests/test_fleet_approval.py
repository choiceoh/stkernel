"""A queued candidate keeps its authenticated admission base as main advances."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_approval as approval
import fleet_handoff as handoff
import fleet_idle as idle
import fleet_pending as pending
import fleet_prepare as prep
import fleet_prepared as prepared


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.origin = self.root / 'origin.git'
        self.repo = self.root / 'candidate'
        self.upstream = self.root / 'upstream'
        self.directory = self.root / 'fleet'
        self.session = 'fixed-approval'
        self.command = ['bash', 'bench/pair.sh', 'candidate', 'VLLM_FIXTURE=1']
        self.git(self.root, 'init', '--bare', '-q', str(self.origin))
        self.git(self.root, 'init', '-q', '-b', 'main', str(self.repo))
        self.configure(self.repo)
        (self.repo / 'kernel.py').write_text('value = 1\n')
        self.commit(self.repo, 'initial')
        self.git(self.repo, 'remote', 'add', 'origin', str(self.origin))
        self.git(self.repo, 'push', '-q', '-u', 'origin', 'main')
        self.base = self.git(self.repo, 'rev-parse', 'HEAD')
        self.git(self.root, 'clone', '-q', '-b', 'main', str(self.origin), str(self.upstream))
        self.configure(self.upstream)
        self.directory.mkdir()
        self.secret = prepared.key(self.directory, create=True)

    def git(self, repo, *args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True,
                                       stderr=subprocess.PIPE).strip()

    def configure(self, repo):
        self.git(repo, 'config', 'user.email', 'fixture@example.invalid')
        self.git(repo, 'config', 'user.name', 'fixture')

    def commit(self, repo, message):
        self.git(repo, 'add', '.')
        self.git(repo, 'commit', '-qm', message)

    def value(self, **target):
        return dict(version=2, session=self.session, command=list(self.command), cwd=str(self.repo),
                    checks=[], deployment_targets=[dict(repo=str(self.repo), profile='glm53', **target)])

    def freeze(self, **target):
        value = self.value(**target)
        approval.freeze(value)
        return value

    def advance_main(self):
        (self.upstream / 'kernel.py').write_text('value = 2\n')
        self.commit(self.upstream, 'upstream runtime change')
        self.git(self.upstream, 'push', '-q', 'origin', 'main')
        self.git(self.repo, 'fetch', '-q', 'origin', 'main')
        return self.git(self.repo, 'rev-parse', 'origin/main')

    def install(self, value, name='approval.json'):
        value = copy.deepcopy(value)
        prepared.sign(value, self.secret)
        manifest = self.directory / 'preparations' / name
        manifest.write_text(json.dumps(value))
        owner = dict(session=self.session, pid=os.getpid(), start='fixture-start',
                     host=socket.gethostname(), protocol=handoff.PROTOCOL)
        record = dict(owner, state='running', kind='boot', ticket='fixture-ticket',
                      command=list(value['command']), cwd=value['cwd'],
                      prepare_manifest=str(manifest), revision=1)
        path = pending.path(self.directory, self.session)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record))
        handoff.receipt(self.directory, self.session).write_text(json.dumps(owner))
        (self.directory / 'holder').write_text(
            f'{self.session}|{os.getpid()}|{socket.gethostname().split(".")[0]}|1|5|fixture|boot\n')
        return manifest

    def verify(self, manifest, *, repo=None, profile='glm53', **environment):
        env = dict(FLEET_DIR=str(self.directory), FLEET_SESSION=self.session,
                   FLEET_RUN_KIND='boot', FLEET_PID=str(os.getpid()),
                   FLEET_PREPARE_MANIFEST=str(manifest), **environment)
        with mock.patch.object(idle, 'boot_authorize', return_value=dict(intent='experiment')) as authorize, \
             mock.patch.object(handoff, 'live', return_value=True):
            result = approval.verify(self.directory, self.session, repo or self.repo, profile, environment=env)
        authorize.assert_called_once()
        return result

    def test_main_advancement_does_not_invalidate_queued_candidate(self):
        value = self.freeze()
        accepted = value['deployment_approvals'][0]
        self.assertEqual((accepted['base'], accepted['candidate']), (self.base, self.base))
        manifest = self.install(value)
        newer = self.advance_main()
        self.assertNotEqual(newer, self.base)
        approval.validate(value)
        self.assertEqual(self.verify(manifest)['candidate'], self.base)
        self.assertEqual(self.git(self.repo, 'rev-parse', 'HEAD'), self.base)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_validation_and_deployment_do_not_fetch_or_rebase(self):
        value = self.freeze()
        manifest = self.install(value)
        # An unreachable origin makes any accidental deployment-time fetch fail.
        self.git(self.repo, 'remote', 'set-url', 'origin', str(self.root / 'offline.git'))
        approval.validate(value)
        self.verify(manifest)
        self.assertEqual(self.git(self.repo, 'rev-parse', 'HEAD'), self.base)

    def test_candidate_only_commit_is_approved_and_remains_frozen(self):
        (self.repo / 'candidate.py').write_text('enabled = True\n')
        self.commit(self.repo, 'candidate')
        candidate = self.git(self.repo, 'rev-parse', 'HEAD')
        value = self.freeze()
        accepted = value['deployment_approvals'][0]
        self.assertEqual((accepted['base'], accepted['candidate']), (self.base, candidate))
        self.advance_main()
        approval.validate(value)

    def test_changed_candidate_revision_is_rejected_even_with_identical_tree(self):
        value = self.freeze()
        manifest = self.install(value)
        self.git(self.repo, 'commit', '--allow-empty', '-qm', 'changed candidate')
        with self.assertRaises(ValueError):
            approval.validate(value)
        with self.assertRaises(ValueError):
            self.verify(manifest)

    def test_dirty_candidate_is_rejected_at_preparation_and_deployment(self):
        value = self.freeze()
        manifest = self.install(value)
        for name in ('kernel.py', 'untracked.py'):
            with self.subTest(name=name):
                path = self.repo / name
                original = path.read_text() if path.exists() else None
                path.write_text('unapproved = True\n')
                with self.assertRaises(ValueError):
                    self.freeze()
                with self.assertRaises(ValueError):
                    approval.validate(value)
                with self.assertRaises(ValueError):
                    self.verify(manifest)
                if original is None:
                    path.unlink()
                else:
                    path.write_text(original)

    def test_discovered_moving_ref_is_frozen_to_exact_commit(self):
        value = self.value()
        value['checks'] = [dict(kind=kind, repo=str(self.repo), ref='origin/main', fetch='main')
                           for kind in ('ancestor', 'source-base')]
        with mock.patch.object(approval, 'git', wraps=approval.git) as git:
            approval.freeze(value)
        fetches = [call for call in git.call_args_list if call.args[1:2] == ('fetch',)]
        self.assertEqual(len(fetches), 1)
        self.assertEqual([check['accepted_ref'] for check in value['checks']], [self.base, self.base])
        self.advance_main()
        approval.validate(value)

    def test_forged_receipt_cannot_change_the_approved_candidate(self):
        manifest = self.install(self.freeze())
        value = json.loads(manifest.read_text())
        value['deployment_approvals'][0]['base'] = '0' * 40
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'modified|forged|authentication'):
            self.verify(manifest)

    def test_approval_is_bound_to_target_profile_and_runtime_overrides(self):
        model = str(self.root / 'fixture-model')
        manifest = self.install(self.freeze(image='fixture-image', model=model))
        self.verify(manifest, IMAGE='fixture-image', MODEL_HOST_PATH=model)
        cases = [dict(repo=self.upstream, IMAGE='fixture-image', MODEL_HOST_PATH=model),
                 dict(profile='dsv4', IMAGE='fixture-image', MODEL_HOST_PATH=model),
                 dict(IMAGE='other-image', MODEL_HOST_PATH=model),
                 dict(IMAGE='fixture-image', MODEL_HOST_PATH=str(self.root / 'other-model'))]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.verify(manifest, **kwargs)

    def test_new_environment_override_cannot_bypass_default_target(self):
        manifest = self.install(self.freeze())
        for env in (dict(IMAGE='other-image'), dict(MODEL_HOST_PATH='other-model')):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.verify(manifest, **env)

    def test_pending_edit_makes_old_manifest_unusable(self):
        value = self.freeze()
        old_manifest = self.install(value, 'old.json')
        value['command'] = ['bash', 'bench/pair.sh', 'replacement', 'VLLM_FIXTURE=1']
        new_manifest = self.install(value, 'replacement.json')
        with self.assertRaises(ValueError):
            self.verify(old_manifest)
        self.verify(new_manifest)

    def test_signed_receipt_must_match_current_pending_command_and_session(self):
        value = self.freeze()
        manifest = self.install(value)
        path = pending.path(self.directory, self.session)
        record = json.loads(path.read_text())
        record['command'] = ['bash', 'unapproved.sh']
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            self.verify(manifest)
        value['session'] = 'another-session'
        manifest = self.install(value)
        with self.assertRaises(ValueError):
            self.verify(manifest)

    def test_queued_or_stale_supervisor_cannot_deploy(self):
        manifest = self.install(self.freeze())
        with mock.patch.object(idle, 'boot_authorize', return_value=dict(intent='experiment')), \
             mock.patch.object(handoff, 'live', return_value=False), self.assertRaises(ValueError):
            approval.verify(self.directory, self.session, self.repo, 'glm53',
                            environment={'FLEET_PREPARE_MANIFEST': str(manifest)})
        path = pending.path(self.directory, self.session)
        record = json.loads(path.read_text())
        record['state'] = 'queued'
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            self.verify(manifest)

    def cpu_fixture(self):
        """Run a real tiny CPU task, substituting only its audited identity adapter."""
        counter = self.root / 'cpu-count'
        script = self.root / 'cpu.py'
        script.write_text('from pathlib import Path\n'
                          f'p = Path({str(counter)!r})\n'
                          'p.write_text(p.read_text() + "run\\n" if p.exists() else "run\\n")\n')
        classifier = self.root / 'classifier.sh'
        classifier.write_text('echo nogpu\n')
        spec = self.root / 'prepare.json'
        spec.write_text(json.dumps(dict(cpu_command=[sys.executable, str(script)],
                                        deployment_targets=[dict(repo=str(self.repo), profile='glm53')],
                                        git=dict(ancestor='origin/main', clean=True))))
        command = [sys.executable, 'kernel.py']

        def identity(repo, spec, command, cwd, env, required_paths, protected_paths):
            return dict(key='fixture-cpu-runtime', scope='full-tree',
                        source=prepared.cpu_source_identity(repo, 'full-tree', protected_paths))

        return counter, command, dict(spec_path=spec, fleet=classifier), identity

    def test_signed_preparation_reuses_fixed_base_and_cpu_result_after_main_moves(self):
        counter, command, kwargs, identity = self.cpu_fixture()
        with mock.patch.object(prepared, 'cpu_identity', side_effect=identity):
            manifest = prep.prepare(self.directory, self.session, command, self.repo,
                                    approve_deploy=True, **kwargs)
            value = prepared.read(self.directory, manifest)
            self.assertEqual(value['checks'][0]['accepted_ref'], self.base)
            self.assertTrue(value['cpu_result']['successful'])
            self.advance_main()
            self.git(self.repo, 'remote', 'set-url', 'origin', str(self.root / 'offline.git'))
            prep.validate(value, refresh=True)
            reused = prep.prepare(self.directory, self.session, command, self.repo,
                                  prepared=manifest, approve_deploy=True, **kwargs)
        self.assertEqual(reused, manifest)
        self.assertEqual(counter.read_text(), 'run\n')
        self.assertEqual(prepared.read(self.directory, reused)['deployment_approvals'][0]['base'], self.base)

    def test_successful_cpu_receipt_gains_signed_approval_without_rerunning_cpu(self):
        counter, command, kwargs, identity = self.cpu_fixture()
        with mock.patch.object(prepared, 'cpu_identity', side_effect=identity):
            original = prep.prepare(self.directory, self.session, command, self.repo, **kwargs)
            old = prepared.read(self.directory, original)
            self.assertTrue(old['cpu_result']['successful'])
            self.assertNotIn('deployment_approvals', old)
            upgraded = prep.prepare(self.directory, self.session, command, self.repo,
                                    prepared=original, approve_deploy=True, **kwargs)
        new = prepared.read(self.directory, upgraded)
        self.assertNotEqual(upgraded, original)
        self.assertEqual(new['cpu_result'], old['cpu_result'])
        self.assertEqual(new['deployment_approvals'][0]['base'], self.base)
        self.assertEqual(counter.read_text(), 'run\n')
        self.assertNotIn('deployment_approvals', prepared.read(self.directory, original))


if __name__ == '__main__':
    unittest.main()
