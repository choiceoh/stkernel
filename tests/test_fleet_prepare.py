"""Preparation failures must happen before reservation and preserve source identity."""
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
import fleet_prepare as prep


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.directory = self.root / 'fleet'
        # These preparation fixtures predate canonical onepass controllers;
        # identify that legacy controller explicitly while testing receipts.
        self.legacy_fleet = str(self.root / 'legacy-controller/bench/fleet.sh')
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'fixture')
        (self.repo/'input.py').write_text('print(1)\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'initial')

    def git(self, *args):
        return subprocess.check_output(['git','-C',str(self.repo),*args], text=True).strip()

    def prepare(self, command=None, **kwargs):
        path = prep.prepare(self.directory, 'fixture', command or [sys.executable,'input.py'], self.repo, **kwargs)
        return json.loads(path.read_text())

    def test_unchanged_inputs_validate_without_reexecuting_cpu(self):
        script = self.repo/'check.sh'
        script.write_text('echo run >> count\n')
        classifier = self.repo/'fleet.sh'
        classifier.write_text('echo nogpu\n')
        spec = self.root/'spec.json'
        spec.write_text(json.dumps(dict(cpu_command=['bash','check.sh'])))
        value = self.prepare(spec_path=spec, fleet=classifier)
        for _ in range(3): prep.validate(value)
        self.assertEqual((self.repo/'count').read_text(), 'run\n')

    def test_changed_script_and_head_rejected(self):
        value = self.prepare()
        (self.repo/'input.py').write_text('print(2)\n')
        with self.assertRaisesRegex(ValueError, 'queued input changed'): prep.validate(value)
        (self.repo/'input.py').write_text('print(1)\n')
        self.git('commit','--allow-empty','-qm','new revision')
        with self.assertRaisesRegex(ValueError, 'checkout revision changed'): prep.validate(value)

    def queued_record(self, receipt, fleet):
        """A queued reservation the way the waiter registers it: its row in the queue, its record, this process as owner."""
        import socket
        import fleet_handoff as handoff
        import fleet_pending as pending
        pid = os.getpid()
        (self.directory / 'queue').write_text(f'1|fixture|100|10|note|boot|{pid}\n')
        value = dict(session='fixture', ticket='1', enqueued_at='100', pid=pid, start=handoff.identity(pid),
                     host=socket.gethostname(), protocol=handoff.PROTOCOL, state='queued', revision=1, pause_protocol=1,
                     command=[sys.executable, 'input.py'], cwd=str(self.repo.resolve()), estimate_min=10, note='note',
                     kind='boot', fleet=fleet, repo=str(self.repo.resolve()), validation_env={}, experiment=None,
                     launch_id=None, prepare_manifest=str(receipt), prepare_receipt_required=True, history=[])
        pending.save_record(self.directory, value)
        return value

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'a queued reservation proves its owner alive through /proc: Linux only')
    def test_a_moved_checkout_is_prepared_again_and_the_ticket_keeps_its_place(self):
        """45차 §95 stopped the ticket ("queued checkout revision changed") for a person to edit and
        resume; the operator asked for that edit to be the queue's own (2026-09-13). The same
        command is prepared again at the revision that is there now, the record's revision moves
        and its history names both commits; nothing else about the ticket changes."""
        import contextlib
        import io
        import fleet_pending as pending
        controller = self.root / 'controller' / 'bench'
        controller.mkdir(parents=True)
        (controller / 'fleet.sh').write_text('#!/bin/sh\nexit 0\n')          # preflight passes; no fleet_onepass.py beside it
        env = dict(REPO=str(self.repo.resolve()), FLEET_DIR=str(self.directory), FLEET_SESSION='fixture')
        with mock.patch.dict(os.environ, env):
            self.prepare()
            receipt = next((self.directory / 'preparations').glob('*.json'))
            before = json.loads(receipt.read_text())
            record = self.queued_record(receipt, str(controller / 'fleet.sh'))
            prep.check_pending(self.directory, 'fixture')                        # unchanged checkout: nothing to re-pin
            self.assertEqual(pending.read_record(self.directory, 'fixture')['revision'], 1)
            old = self.git('rev-parse', 'HEAD')
            self.git('commit', '--allow-empty', '-qm', 'main moved on')
            new = self.git('rev-parse', 'HEAD')
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                prep.check_pending(self.directory, 'fixture', withdraw_failed=True)   # no pause, no refusal
            self.assertIn(f'RE-PIN fixture: checkout moved {old[:12]} -> {new[:12]}', err.getvalue())
            after = pending.read_record(self.directory, 'fixture')
            self.assertEqual(after['revision'], 2)
            self.assertNotEqual(after['prepare_manifest'], str(receipt), 'a fresh receipt, of the new tree')
            self.assertEqual(json.loads(Path(after['prepare_manifest']).read_text())['head'][1], new)
            self.assertEqual(before['head'][1], old)
            self.assertIn(f'{old[:12]} -> {new[:12]}', after['history'][-1]['repin'])
            self.assertEqual((after['command'], after['cwd'], after['estimate_min'], after['note'], after['ticket']),
                             (record['command'], record['cwd'], 10, 'note', '1'))
            self.assertEqual((self.directory / 'queue').read_text().split('|')[1], 'fixture', 'and it is still in the queue')
            prep.check_pending(self.directory, 'fixture')                        # settled: the next check re-pins nothing
            self.assertEqual(pending.read_record(self.directory, 'fixture')['revision'], 2)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'a queued reservation proves its owner alive through /proc: Linux only')
    def test_the_stop_of_95_is_one_switch_away_and_a_broken_tree_still_pauses(self):
        import fleet_pending as pending
        controller = self.root / 'controller' / 'bench'
        controller.mkdir(parents=True)
        (controller / 'fleet.sh').write_text('#!/bin/sh\nexit 0\n')
        env = dict(REPO=str(self.repo.resolve()), FLEET_DIR=str(self.directory), FLEET_SESSION='fixture')
        with mock.patch.dict(os.environ, env):
            self.prepare()
            receipt = next((self.directory / 'preparations').glob('*.json'))
            self.queued_record(receipt, str(controller / 'fleet.sh'))
            self.git('commit', '--allow-empty', '-qm', 'main moved on')
            with mock.patch.dict(os.environ, {'FLEET_AUTO_REPIN': '0'}):
                with self.assertRaisesRegex(ValueError, 'queued checkout revision changed'):
                    prep.check_pending(self.directory, 'fixture')
            self.assertEqual(pending.read_record(self.directory, 'fixture')['revision'], 1)
            (self.repo / 'input.py').write_text('def :\n')                       # the tree that is there now does not compile
            with self.assertRaisesRegex(ValueError, 're-pin at .* failed'):
                prep.check_pending(self.directory, 'fixture')
            after = pending.read_record(self.directory, 'fixture')
            self.assertEqual((after['revision'], after['prepare_manifest']), (1, str(receipt)), 'the ticket is as it was')

    def test_missing_input_fresh_output_and_image_fail_early(self):
        spec = self.root/'spec.json'
        spec.write_text(json.dumps(dict(required_paths=['missing'])))
        with self.assertRaisesRegex(ValueError,'required path'): self.prepare(spec_path=spec)
        spec.write_text(json.dumps(dict(absent_paths=['input.py'])))
        with self.assertRaisesRegex(ValueError,'fresh output'): self.prepare(spec_path=spec)
        spec.write_text(json.dumps(dict(images=['fixture-image'])))
        original = prep.run
        def run(argv, *args):
            if argv[0] == 'docker': raise ValueError('image missing')
            return original(argv,*args)
        with mock.patch.object(prep,'run',side_effect=run):
            with self.assertRaisesRegex(ValueError,'image missing'): self.prepare(spec_path=spec)

    def test_actual_campaign_ancestry_guard_before_queue_and_after_wait(self):
        script = self.repo/'campaign.sh'
        script.write_text('''#!/bin/bash
set -eu
git merge-base --is-ancestor origin/main HEAD || { echo 'ABORT: candidate needs current main'; exit 2; }
''')
        self.git('add','.'); self.git('commit','-qm','campaign')
        self.git('update-ref','refs/remotes/origin/main','HEAD')
        value = self.prepare(['bash','campaign.sh'])
        prep.validate(value)
        old = self.git('rev-parse','HEAD')
        self.git('commit','--allow-empty','-qm','new main')
        future = self.git('rev-parse','HEAD')
        self.git('reset','--hard',old)
        self.git('update-ref','refs/remotes/origin/main',future)
        with self.assertRaisesRegex(ValueError,'campaign.sh:3: candidate must include origin/main'):
            prep.validate(value)
        with self.assertRaisesRegex(ValueError,'candidate must include origin/main'):
            self.prepare(['bash','campaign.sh'])
        self.assertFalse((self.directory/'holder').exists())

    def test_external_campaign_guard_uses_execution_checkout(self):
        script=self.root/'external.sh'
        script.write_text('git merge-base --is-ancestor origin/main HEAD || exit 2\n')
        self.git('update-ref','refs/remotes/origin/main','HEAD')
        value=self.prepare(['bash',str(script)])
        self.assertEqual(value['checks'][0]['repo'],str(self.repo.resolve()))

    def test_clean_guard_is_literal_and_comments_not_executed(self):
        script=self.repo/'campaign.sh'
        script.write_text('# git merge-base --is-ancestor invalid HEAD\n[[ -z $(git status --porcelain) ]] || exit 2\n')
        self.git('add','.');self.git('commit','-qm','clean')
        value=self.prepare(['bash','campaign.sh'])
        self.assertEqual([c['kind'] for c in value['checks']],['clean'])
        (self.repo/'dirty').touch()
        with self.assertRaisesRegex(ValueError,'clean checkout'): prep.validate(value)

    def test_canonical_ar_target_uses_candidate_repo_and_has_no_moving_main_guard(self):
        canonical = Path(__file__).resolve().parents[1]
        command = ['bash', str(canonical / 'probes/run_ar_consumer_campaign.sh')]
        # The campaign's explicit self-checkout assignment also overrides a
        # caller's different REPO, before its nested pair executes.
        for selected in (canonical, self.repo):
            with self.subTest(selected=selected), mock.patch.dict(os.environ, {'REPO': str(selected)}, clear=True):
                targets = prep.deployment_targets(command, self.repo, {})
                checks = prep.discover(command, self.repo)
            self.assertEqual(targets, [dict(repo=str(canonical), profile='glm53')])
            self.assertFalse(any(check['kind'] in ('ancestor', 'source-base') for check in checks))

    def test_missing_entrypoint_is_refused_but_inline_code_is_not_a_filename(self):
        with self.assertRaisesRegex(ValueError,'entrypoint script is missing'):
            self.prepare(['bash','missing.sh'])
        with self.assertRaisesRegex(ValueError,'entrypoint script is missing'):
            self.prepare([sys.executable,'missing.py'])
        self.prepare([sys.executable,'-c','print("example")'])
        self.prepare(['bash','-c','echo missing.sh'])

    def test_empty_argument_and_large_binary_are_valid(self):
        self.prepare([sys.executable,'-c','pass',''])
        binary=self.root/'compiler'
        with binary.open('wb') as stream: stream.truncate(9*1024*1024)
        binary.chmod(0o755)
        value=self.prepare([str(binary),'--version'])
        self.assertNotIn(str(binary),value['files'])
        prep.validate(value)
        binary.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'queued executable changed'): prep.validate(value)

    def test_quoted_clean_guard_is_recognized(self):
        script=self.repo/'campaign.sh'
        script.write_text('[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || exit 2\n')
        self.git('add','.');self.git('commit','-qm','clean')
        value=self.prepare(['bash','campaign.sh'])
        (self.repo/'dirty').touch()
        with self.assertRaisesRegex(ValueError,'clean checkout'): prep.validate(value)

    def test_bad_syntax_and_gpu_cpu_command_refused(self):
        (self.repo/'broken.py').write_text('def :\n')
        with self.assertRaises(SyntaxError): self.prepare([sys.executable,'broken.py'])
        classifier = self.repo/'fleet.sh';classifier.write_text('echo gpu\n')
        spec=self.root/'spec.json';spec.write_text(json.dumps(dict(cpu_command=['nvidia-smi'])))
        with self.assertRaisesRegex(ValueError,'shows GPU use'): self.prepare(spec_path=spec,fleet=classifier)

    def test_failed_check_passes_exact_record_to_pause_callback(self):
        import fleet_pending
        self.prepare()
        path=next((self.directory/'preparations').glob('*.json'))
        record=dict(fleet=self.legacy_fleet,state='queued',prepare_manifest=str(path),command=[sys.executable,'input.py'],cwd=str(self.repo.resolve()),
                    ticket='1',pid=123,start='same',revision=1)
        pause = mock.Mock(return_value=False)
        with mock.patch.dict(sys.modules, fleet_pause=types.SimpleNamespace(pause_failed=pause)), \
             mock.patch.object(fleet_pending,'read_record',side_effect=[record,dict(record,revision=2)]) as read, \
             mock.patch.object(prep,'validate',side_effect=ValueError('old input changed')):
            # False needs an independent proof that a concurrent edit won.
            prep.check_pending(self.directory,'fixture',withdraw_failed=True)
            pause.assert_called_once_with(self.directory,'fixture',record,'old input changed')
            read.side_effect = None
            read.return_value = record
            pause.return_value = True
            with self.assertRaises(prep.PreparedPaused):
                prep.check_pending(self.directory,'fixture',withdraw_failed=True)
            pause.reset_mock()
            with self.assertRaisesRegex(ValueError,'old input changed'):
                prep.check_pending(self.directory,'fixture',external=False,withdraw_failed=True)
            pause.assert_not_called()

    def test_unsupported_legacy_pause_still_withdraws_failed_matching_ticket(self):
        import fleet_pending
        self.prepare()
        path=next((self.directory/'preparations').glob('*.json'))
        record=dict(fleet=self.legacy_fleet,state='queued',prepare_manifest=str(path),command=[sys.executable,'input.py'],cwd=str(self.repo.resolve()),
                    ticket='1',pid=123,start='same',revision=1)
        original='1|fixture|1|2|test|boot|123\n2|other|2|2|next|boot|456\n'
        (self.directory/'queue').write_text(original)
        (self.directory/'priority-front').write_text('fixture')
        (self.directory/'priority-yield').write_text('other')
        with mock.patch.dict(sys.modules, fleet_pause=types.SimpleNamespace(pause_failed=mock.Mock(return_value=False))), \
             mock.patch.object(fleet_pending,'read_record',return_value=record), \
             mock.patch.object(prep,'validate',side_effect=ValueError('legacy input changed')):
            with self.assertRaisesRegex(ValueError,'legacy input changed'):
                prep.check_pending(self.directory,'fixture',withdraw_failed=True)
        self.assertEqual((self.directory/'queue').read_text(),'2|other|2|2|next|boot|456\n')
        self.assertFalse((self.directory/'priority-front').exists())
        self.assertEqual((self.directory/'priority-yield').read_text(),'other')

    def test_same_modern_record_or_missing_record_cannot_swallow_failed_pause(self):
        import fleet_pending
        self.prepare()
        path=next((self.directory/'preparations').glob('*.json'))
        record=dict(fleet=self.legacy_fleet,state='queued',prepare_manifest=str(path),command=[sys.executable,'input.py'],cwd=str(self.repo.resolve()),
                    ticket='1',pid=123,start='same',revision=1,pause_protocol=1)
        row='1|fixture|1|2|test|boot|123\n'
        (self.directory/'queue').write_text(row)
        for current in (record,None):
            with self.subTest(current=current), \
                 mock.patch.dict(sys.modules, fleet_pause=types.SimpleNamespace(pause_failed=mock.Mock(return_value=False))), \
                 mock.patch.object(fleet_pending,'read_record',side_effect=[record,current]), \
                 mock.patch.object(prep,'validate',side_effect=ValueError('current input changed')):
                with self.assertRaisesRegex(ValueError,'current input changed'):
                    prep.check_pending(self.directory,'fixture',withdraw_failed=True)
        self.assertEqual((self.directory/'queue').read_text(),row)

    def test_recovery_reclaim_does_not_reapply_payload_preparation(self):
        import fleet_pending
        record=dict(state='finishing',prepare_manifest='/missing/old-input.json')
        with mock.patch.object(fleet_pending,'read_record',return_value=record):
            prep.check_pending(self.directory,'fixture',withdraw_failed=True)

    def test_manifest_matches_edited_command_and_is_private(self):
        import fleet_pending
        value=self.prepare()
        path=next((self.directory/'preparations').glob('*.json'))
        self.assertEqual(path.stat().st_mode & 0o777,0o600)
        record=dict(fleet=self.legacy_fleet,state='queued',prepare_manifest=str(path),command=[sys.executable,'-c','pass'],cwd=str(self.repo))
        with mock.patch.object(fleet_pending,'read_record',return_value=record):
            with self.assertRaisesRegex(ValueError,'does not match'): prep.check_pending(self.directory,'fixture')


    def cpu_receipt_fixture(self):
        (self.repo/'bench').mkdir(exist_ok=True)
        counter = self.root/'cpu-counter'
        (self.repo/'bench'/'cpu_checks.py').write_text(
            'from pathlib import Path\np=Path(' + repr(str(counter)) + ')\n'
            'p.write_text((p.read_text() if p.exists() else "") + "run\\n")\n')
        classifier = self.root/'classifier.sh'
        classifier.write_text('echo nogpu\n')
        spec = self.root/'spec.json'
        spec.write_text(json.dumps(dict(cpu_command=[sys.executable,'bench/cpu_checks.py','--suite','logic'])))
        self.git('add','.'); self.git('commit','-qm','CPU fixture')
        return spec, classifier, counter

    def test_prepared_receipt_reuses_successful_cpu_once_across_run_and_edit(self):
        spec, classifier, counter = self.cpu_receipt_fixture()
        command = [sys.executable,'input.py']
        manifest = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec,fleet=classifier)
        original = manifest.read_bytes()
        self.assertTrue(json.loads(original)['cpu_result']['reusable'])
        # Standalone preparation, run ingestion and same-command edit all use
        # one immutable receipt despite controller bookkeeping differences.
        for metadata in ({'FLEET_SESSION':'other','FLEET_LAUNCH_ID':'launch'},
                         {'FLEET_PREPARE_MANIFEST':str(manifest),'FLEET_RUN_KIND':'boot','SHLVL':'99'},
                         {'SSH_CONNECTION':'source 50123 target 22','SSH_CLIENT':'source 50123 22','SSH_TTY':'/dev/pts/8'}):
            with mock.patch.dict(os.environ, metadata):
                reused = prep.prepare(self.directory,'fixture',command,self.repo,prepared=manifest)
                self.assertEqual(reused,manifest)
        self.assertEqual(counter.read_text(),'run\n')
        self.assertEqual(manifest.read_bytes(),original)
        transport = prep.fleet_prepared.payload_environment({'SSH_CONNECTION':'ignored','SSH_AUTH_SOCK':'needed','PATH':'real'})
        self.assertEqual(transport, {'SSH_AUTH_SOCK':'needed','PATH':'real'})

    def test_receipt_environment_is_private_and_changes_refuse_reuse(self):
        spec, classifier, counter = self.cpu_receipt_fixture()
        command = [sys.executable,'input.py']
        with mock.patch.dict(os.environ, {'FIXTURE_PRIVATE_TOKEN':'sensitive-preparation-value'}):
            path = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec,fleet=classifier)
            saved = path.read_text()
            self.assertNotIn('sensitive-preparation-value',saved)
            self.assertNotIn('FIXTURE_PRIVATE_TOKEN',saved)
            with mock.patch.dict(os.environ, {'FIXTURE_PRIVATE_TOKEN':'changed-secret'}):
                with self.assertRaisesRegex(ValueError,'environment changed'):
                    prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)
        self.assertEqual(counter.read_text(),'run\n')

    def test_dirty_transitive_cpu_source_rejects_receipt(self):
        spec, classifier, counter = self.cpu_receipt_fixture()
        dependency = self.repo/'dependency.py'
        dependency.write_text('value=1\n')
        self.git('add','.');self.git('commit','-qm','transitive source')
        command = [sys.executable,'input.py']
        path = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec,fleet=classifier)
        dependency.write_text('value=2\n')
        with self.assertRaisesRegex(ValueError,'CPU source changed.*dependency.py'):
            prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)
        self.assertEqual(counter.read_text(),'run\n')

    def test_changed_command_spec_and_required_file_refuse_reuse(self):
        spec = self.root/'spec.json'
        required = self.root/'input.bin'; required.write_bytes(b'old-input')
        spec.write_text(json.dumps(dict(required_paths=[str(required)])))
        command = [sys.executable,'input.py']
        path = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec)
        with self.assertRaisesRegex(ValueError,'command argv changed'):
            prep.prepare(self.directory,'fixture',[sys.executable,'-c','pass'],self.repo,prepared=path)
        required.write_bytes(b'new-input')
        with self.assertRaisesRegex(ValueError,'required input changed'):
            prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)
        required.write_bytes(b'old-input')
        spec.write_text(json.dumps(dict(required_paths=[str(required)],timeout_seconds=30)))
        with self.assertRaisesRegex(ValueError,'specification changed'):
            prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)

    def test_forged_and_tampered_receipts_are_refused_before_cpu(self):
        command = [sys.executable,'input.py']
        path = prep.prepare(self.directory,'fixture',command,self.repo)
        value = json.loads(path.read_text())
        value['command'] = [sys.executable,'-c','pass']
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'modified or forged'):
            prep.prepare(self.directory,'fixture',value['command'],self.repo,prepared=path)
        value.pop('receipt'); path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'no authenticated receipt'):
            prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)

    def test_unaudited_and_unexecuted_cpu_commands_cannot_claim_reuse(self):
        classifier = self.root/'classifier.sh'; classifier.write_text('echo nogpu\n')
        spec = self.root/'spec.json'
        spec.write_text(json.dumps(dict(cpu_command=[sys.executable,'-c','pass'])))
        command = [sys.executable,'input.py']
        path = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec,fleet=classifier)
        with self.assertRaisesRegex(ValueError,'no audited dependency contract'):
            prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)
        path = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec,fleet=classifier,execute_cpu=False)
        with self.assertRaisesRegex(ValueError,'no successful CPU result'):
            prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)

    def test_cpu_runtime_change_refuses_reuse(self):
        spec, classifier, counter = self.cpu_receipt_fixture()
        command = [sys.executable,'input.py']
        path = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec,fleet=classifier)
        current = json.loads(path.read_text())['cpu_result']['identity']
        current['key'] = 'different-runtime'
        with mock.patch.object(prep.fleet_prepared,'cpu_identity',return_value=current):
            with self.assertRaisesRegex(ValueError,'CPU source or runtime changed'):
                prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)
        self.assertEqual(counter.read_text(),'run\n')


    def test_go_checks_transitive_source_without_reexecuting_cpu(self):
        import fleet_pending
        spec, classifier, counter = self.cpu_receipt_fixture()
        dependency = self.repo/'dependency.py'; dependency.write_text('value=1\n')
        self.git('add','.');self.git('commit','-qm','dependency')
        command = [sys.executable,'input.py']
        path = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec,fleet=classifier)
        record = dict(fleet=self.legacy_fleet,state='queued',prepare_manifest=str(path),command=command,cwd=str(self.repo.resolve()),prepare_receipt_required=True)
        dependency.write_text('value=2\n')
        with mock.patch.object(fleet_pending,'read_record',return_value=record):
            with self.assertRaisesRegex(ValueError,'CPU source changed'):
                prep.check_pending(self.directory,'fixture',external=False)
        self.assertEqual(counter.read_text(),'run\n')

    def test_go_cannot_downgrade_authenticated_receipt_to_legacy(self):
        import fleet_pending
        command = [sys.executable,'input.py']
        path = prep.prepare(self.directory,'fixture',command,self.repo)
        value = json.loads(path.read_text()); value['version']=1; value.pop('receipt')
        path.write_text(json.dumps(value))
        record = dict(fleet=self.legacy_fleet,state='queued',prepare_manifest=str(path),command=command,cwd=str(self.repo.resolve()),prepare_receipt_required=True)
        with mock.patch.object(fleet_pending,'read_record',return_value=record):
            with self.assertRaisesRegex(ValueError,'no authenticated receipt'):
                prep.check_pending(self.directory,'fixture',external=False)

    def test_paused_check_cli_returns_editable_status_four(self):
        with mock.patch.dict(os.environ, FLEET_DIR=str(self.directory)), \
             mock.patch.object(prep,'check_pending',side_effect=prep.PreparedPaused('input changed')):
            self.assertEqual(prep.main(['check','fixture','--withdraw-failed']),4)

    def test_audited_source_guard_accepts_docs_commit_but_rejects_code(self):
        import fleet_source
        script = self.repo/'campaign.sh'
        script.write_text('python3 bench/fleet_source.py require-base origin/main\n')
        (self.repo/'README.md').write_text('old docs\n')
        self.git('add','.');self.git('commit','-qm','guard')
        self.git('update-ref','refs/remotes/origin/main','HEAD')
        with mock.patch.object(fleet_source,'audited_wrapper',return_value=True):
            value = self.prepare(['bash','campaign.sh'])
        self.assertIn('source_identity',value)
        (self.repo/'README.md').write_text('new docs\n')
        self.git('add','.');self.git('commit','-qm','docs')
        prep.validate(value)
        (self.repo/'new_code.py').write_text('value=2\n')
        with self.assertRaisesRegex(ValueError,'queued source changed.*new_code.py'):
            prep.validate(value)

    def test_unaudited_source_guard_retains_exact_head(self):
        script = self.repo/'campaign.sh'
        script.write_text('python3 bench/fleet_source.py require-base origin/main\n')
        self.git('add','.');self.git('commit','-qm','guard')
        self.git('update-ref','refs/remotes/origin/main','HEAD')
        value = self.prepare(['bash','campaign.sh'])
        self.assertIn('head',value)
        self.git('commit','--allow-empty','-qm','new revision')
        with self.assertRaisesRegex(ValueError,'checkout revision changed'):
            prep.validate(value)


    def test_reuse_binds_requested_gpu_environment_and_image_identity(self):
        command = [sys.executable,'input.py']
        spec = self.root/'spec.json'; spec.write_text(json.dumps(dict(images=['fixture:latest'])))
        original = prep.run
        image_id = ['sha256:original']
        def run(argv, *args):
            return image_id[0] if argv[0] == 'docker' else original(argv,*args)
        with mock.patch.object(prep,'run',side_effect=run), \
             mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            path = prep.prepare(self.directory,'fixture',command,self.repo,spec_path=spec)
            with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='1'):
                with self.assertRaisesRegex(ValueError,'environment changed'):
                    prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)
            image_id[0] = 'sha256:replacement'
            with self.assertRaisesRegex(ValueError,'image identity changed'):
                prep.prepare(self.directory,'fixture',command,self.repo,prepared=path)


    def test_deployment_target_resolves_wrapper_checkout_and_exported_image(self):
        import fleet_source
        (self.repo/'probes').mkdir()
        wrapper=self.repo/'probes'/'wrapper.sh'
        wrapper.write_text('cd "$(dirname "$0")/.."\nexport REPO=$PWD\n'
                           'export IMAGE=sha256:fixed MODEL_HOST_PATH=/models/candidate\n'
                           'bash launchers/deploy-overlays.sh glm53\n')
        with mock.patch.object(fleet_source,'audited_wrapper',return_value=True):
            targets=prep.deployment_targets(['bash',str(wrapper)],self.root,{})
        self.assertEqual(targets,[dict(repo=str(self.repo.resolve()),profile='glm53',image='sha256:fixed',model='/models/candidate')])
        self.assertEqual(prep.execution_cwd(wrapper,self.root,wrapper.read_text()),self.repo)
        wrapper.write_text('cd '+str(self.repo)+'\nbash launchers/deploy-overlays.sh glm53\n')
        with mock.patch.object(fleet_source,'audited_wrapper',return_value=True):
            targets=prep.deployment_targets(['bash',str(wrapper)],self.root,{})
        self.assertEqual(targets[0]['repo'],str(self.repo.resolve()))

    def test_deployment_target_respects_repo_default_and_export_semantics(self):
        import fleet_source
        (self.repo/'probes').mkdir()
        wrapper=self.repo/'probes'/'wrapper.sh'
        wrapper.write_text('cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"\n'
                           'IMAGE=sha256:probe-only\nbash launchers/deploy-overlays.sh glm53\n')
        with mock.patch.object(fleet_source,'audited_wrapper',return_value=True), \
             mock.patch.dict(os.environ, {'REPO':str(self.repo)}, clear=True):
            target=prep.deployment_targets(['bash',str(wrapper)],self.root,{})[0]
            self.assertNotIn('image',target)
            with mock.patch.dict(os.environ, IMAGE='inherited-export'):
                target=prep.deployment_targets(['bash',str(wrapper)],self.root,{})[0]
                self.assertEqual(target['image'],'sha256:probe-only')

    def test_dynamic_wrapper_requires_explicit_signed_deployment_target(self):
        import fleet_source
        wrapper=self.repo/'wrapper.sh'
        wrapper.write_text('cd "$(choose_candidate)"\nbash launchers/deploy-overlays.sh "$PROFILE"\n')
        with mock.patch.object(fleet_source,'audited_wrapper',return_value=True):
            with self.assertRaisesRegex(ValueError,'dynamic wrapper working directory'):
                prep.deployment_targets(['bash',str(wrapper)],self.repo,{})
        spec=self.root/'spec.json'
        spec.write_text(json.dumps(dict(deployment_targets=[dict(repo=str(self.repo),profile='glm53',image='candidate:image')])) )
        path=prep.prepare(self.directory,'fixture',['bash',str(wrapper)],self.repo,spec_path=spec)
        value=prep.fleet_prepared.read(self.directory,path)
        self.assertEqual(value['deployment_targets'],[dict(repo=str(self.repo.resolve()),profile='glm53',image='candidate:image')])

    def test_validate_targets_uses_cli_target_and_rejects_changed_candidate_source(self):
        import fleet_source
        (self.repo/'probes').mkdir()
        wrapper=self.repo/'probes'/'wrapper.sh'
        wrapper.write_text('cd "$(dirname "$0")/.."\nexport IMAGE=sha256:candidate\n'
                           'bash launchers/deploy-overlays.sh glm53\n')
        # Caller/controller checkout differs from the actual candidate source.
        controller=self.root/'controller'; controller.mkdir()
        subprocess.run(['git','init','-q',str(controller)],check=True)
        subprocess.run(['git','-C',str(controller),'commit','--allow-empty','-qm','controller',
                        '--author','fixture <fixture@example.invalid>'],check=True,
                       env=dict(os.environ,GIT_COMMITTER_NAME='fixture',GIT_COMMITTER_EMAIL='fixture@example.invalid'))
        with mock.patch.object(fleet_source,'audited_wrapper',return_value=True):
            path=prep.prepare(self.directory,'fixture',['bash',str(wrapper)],controller)
        calls=[]; original=prep.run
        def run(argv,*args,**kwargs):
            if len(argv)>1 and argv[1].endswith('fleet_validation.py'):
                calls.append(argv); return '{}'
            return original(argv,*args,**kwargs)
        with mock.patch.object(prep,'run',side_effect=run):
            prep.validate_targets(self.directory,path,verify_only=True)
            self.assertEqual(calls[0][calls[0].index('--repo')+1],str(self.repo.resolve()))
            self.assertEqual(calls[0][calls[0].index('--image')+1],'sha256:candidate')
            self.assertIn('--verify-only',calls[0])
            (self.repo/'dependency.py').write_text('changed=True\n')
            with self.assertRaisesRegex(ValueError,'deployment source changed'):
                prep.validate_targets(self.directory,path)
        self.assertEqual(len(calls),1)

    def test_direct_deploy_and_literal_private_wrapper_resolve_target(self):
        (self.repo/'launchers').mkdir()
        script=self.repo/'launchers'/'deploy-overlays.sh';script.write_text('# fixture deploy\n')
        with mock.patch.dict(os.environ, {}, clear=True):
            target=prep.deployment_targets(['bash',str(script),'glm53'],self.root,{})[0]
        self.assertEqual(target,dict(repo=str(self.repo.resolve()),profile='glm53'))
        wrapper=self.repo/'wrapper.sh';wrapper.write_text('bash launchers/deploy-overlays.sh glm53\n')
        target=prep.deployment_targets(['bash',str(wrapper)],self.repo,{})[0]
        self.assertEqual(target['repo'],str(self.repo.resolve()))
        private=self.root/'private.sh'
        private.write_text('REPO='+str(self.repo)+'\nexport IMAGE=sha256:literal\nbash "$REPO/launchers/deploy-overlays.sh" glm53\n')
        target=prep.deployment_targets(['bash',str(private)],self.root,{})[0]
        self.assertEqual(target,dict(repo=str(self.repo.resolve()),profile='glm53',image='sha256:literal'))

    def test_generic_command_can_prepare_without_any_git_checkout(self):
        with mock.patch.dict(os.environ,REPO=str(self.root/'not-a-repository')):
            path=prep.prepare(self.directory,'fixture',[sys.executable,'-c','pass'],self.root)
        value=prep.fleet_prepared.read(self.directory,path)
        self.assertEqual(value['deployment_sources'],{})
        self.assertNotIn('head',value)


    def test_literal_env_prefix_resolves_sources_targets_and_unsets(self):
        wrapper=self.root/'private.sh'
        wrapper.write_text('cd "$REPO"\nbash launchers/deploy-overlays.sh "$PROFILE"\n')
        command=['env','-u','MODEL_HOST_PATH','REPO='+str(self.repo),'PROFILE=glm53','IMAGE=sha256:prefix','bash',str(wrapper)]
        with mock.patch.dict(os.environ,MODEL_HOST_PATH='/unwanted/model',IMAGE='inherited'):
            self.assertEqual(prep.sources(command,self.root),[wrapper.resolve()])
            target=prep.deployment_targets(command,self.root,{})[0]
            self.assertEqual(target,dict(repo=str(self.repo.resolve()),profile='glm53',image='sha256:prefix'))
            isolated=['env','-i','REPO='+str(self.repo),'PROFILE=glm53','bash',str(wrapper)]
            target=prep.deployment_targets(isolated,self.root,{})[0]
            self.assertEqual(target,dict(repo=str(self.repo.resolve()),profile='glm53'))
            command=['env','-u','IMAGE','-u','MODEL_HOST_PATH','REPO='+str(self.repo),'PROFILE=glm53','bash',str(wrapper)]
            path=prep.prepare(self.directory,'fixture',command,self.repo)
            calls=[]; original=prep.run
            def run(argv,*args,**kwargs):
                if len(argv)>1 and argv[1].endswith('fleet_validation.py'):
                    calls.append((argv,kwargs['env'])); return '{}'
                return original(argv,*args,**kwargs)
            with mock.patch.object(prep,'run',side_effect=run):
                prep.validate_targets(self.directory,path)
            self.assertNotIn('--image',calls[0][0]);self.assertNotIn('--model',calls[0][0])
            self.assertIn('--level',calls[0][0])
            self.assertEqual(calls[0][0][calls[0][0].index('--level')+1],'admission')
            self.assertNotIn('IMAGE',calls[0][1]);self.assertNotIn('MODEL_HOST_PATH',calls[0][1])
        with self.assertRaisesRegex(ValueError,'unsupported env prefix'):
            prep.deployment_targets(['env','-S','bash private.sh'],self.repo,{})
        with self.assertRaisesRegex(ValueError,'entrypoint script is missing'):
            prep.sources(['env','IMAGE=one','bash','missing.sh'],self.repo)


    def test_release_evidence_drift_pauses_before_go_and_local_check_stays_pure(self):
        import fleet_pending
        import fleet_handoff
        import socket
        command=[sys.executable,'input.py']
        manifest=prep.prepare(self.directory,'fixture',command,self.repo)
        pid=os.getpid()
        record=dict(fleet=self.legacy_fleet,session='fixture',state='queued',prepare_manifest=str(manifest),
                    prepare_receipt_required=True,command=command,cwd=str(self.repo.resolve()),
                    ticket='1',pid=pid,start='fixture-start',host=socket.gethostname(),
                    enqueued_at='1',estimate_min=2,note='test',
                    protocol=fleet_handoff.PROTOCOL,pause_protocol=1,revision=1,kind='boot',
                    validation_env={'FLEET_VALIDATION_REQUIRED':'1'})
        row=f'1|fixture|1|2|test|boot|{pid}\n'
        (self.directory/'queue').write_text(row)
        fleet_pending.save_record(self.directory,record)
        with mock.patch.object(fleet_handoff,'identity',return_value='fixture-start'), \
             mock.patch.object(prep,'validate_targets',side_effect=ValueError('release tokenizer changed')) as gate:
            prep.check_pending(self.directory,'fixture',external=False,withdraw_failed=True)
            gate.assert_not_called()
            self.assertEqual(fleet_pending.read_record(self.directory,'fixture')['state'],'queued')
            with self.assertRaisesRegex(prep.PreparedPaused,'release tokenizer changed'):
                prep.check_pending(self.directory,'fixture',external=True,withdraw_failed=True)
            self.assertEqual(gate.call_count,1)
            self.assertTrue(gate.call_args.kwargs['verify_only'])
            self.assertEqual(gate.call_args.kwargs['_validated_value']['receipt'],json.loads(manifest.read_text())['receipt'])
        paused=fleet_pending.read_record(self.directory,'fixture')
        self.assertEqual(paused['state'],'paused')
        self.assertEqual(paused['pause_reason'],'release tokenizer changed')
        self.assertEqual(paused['ticket'],'1')
        self.assertEqual('|'.join(paused['parked_row'])+'\n',row)
        self.assertEqual((self.directory/'queue').read_text(),'')
        self.assertFalse((self.directory/'holder').exists())


if __name__=='__main__': unittest.main()
