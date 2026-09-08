"""Queued edits preserve ownership, admission and immutable evidence boundaries."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_handoff as handoff
import fleet_pending as pending


class PendingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Preparation now resolves the real executable before an edit can
        # commit. Keep the payloads local and inert while exercising that
        # check, including edits racing admission or another editor.
        bindir = self.root / 'bin'
        bindir.mkdir()
        for name in ('old', 'new', 'too-late', 'stale'):
            command = bindir / name
            command.write_text('#!/bin/sh\nexit 0\n')
            command.chmod(0o700)
        self.pid = os.getpid()
        self.queue = self.root / 'queue'
        self.queue.write_text(f'10|before|100|5|before|boot|{self.pid}\n'
                             f'20|mine|101|10|original|boot|{self.pid}\n'
                             f'30|after|102|5|after|boot|{self.pid}\n')
        for patcher in (patch.object(handoff, 'identity', return_value='owner-start'),
                        patch.dict(os.environ, REPO=str(self.root), FLEET_EXPERIMENT_ID='',
                                   FLEET_PREPARE_MANIFEST='', PATH=str(bindir) + os.pathsep + os.environ['PATH'])):
            patcher.start(); self.addCleanup(patcher.stop)
        pending.register(self.root, 'mine', ['old', 'spaced argument'], '/pinned/fleet.sh', 'boot')

    def saved(self):
        return handoff.read(pending.path(self.root, 'mine'))

    def test_edit_keeps_ticket_age_owner_neighbors_and_priority_markers(self):
        (self.root / 'priority-front').write_text('mine')
        original = self.queue.read_text().splitlines()
        with patch.object(pending, 'validate') as check:
            result = pending.edit(self.root, 'mine', command=['new', '$literal', 'a b'],
                                  note='수정된 실험', estimate=7, expected=1)
        check.assert_called_once()
        rows = self.queue.read_text().splitlines()
        self.assertEqual(rows[0], original[0]); self.assertEqual(rows[2], original[2])
        self.assertEqual(rows[1], f'20|mine|101|7|수정된 실험|boot|{self.pid}')
        self.assertEqual(result['position'], 2)
        self.assertEqual(result['revision'], 2)
        self.assertEqual(self.saved()['command'], ['new', '$literal', 'a b'])
        self.assertEqual(self.saved()['history'][0]['command'], ['old', 'spaced argument'])
        self.assertEqual((self.root / 'priority-front').read_text(), 'mine')

    def test_failed_preflight_invalid_metadata_and_empty_command_keep_original(self):
        before, queue = self.saved(), self.queue.read_text()
        with patch.object(pending, 'validate', side_effect=ValueError('bad preflight')):
            with self.assertRaisesRegex(ValueError, 'bad preflight'):
                pending.edit(self.root, 'mine', command=['invalid'])
        for changes in (dict(note='bad|row'), dict(note='bad\nrow'), dict(estimate=0), dict(command=[])):
            with self.assertRaises(ValueError):
                pending.edit(self.root, 'mine', **changes)
        self.assertEqual(self.saved(), before)
        self.assertEqual(self.queue.read_text(), queue)

    def test_admission_wins_during_preflight_and_edit_cannot_change_payload(self):
        before = self.saved()
        def admitted(*_):
            with pending.lock(self.root):
                self.assertTrue(handoff.admit(self.root, 'mine', self.pid, 'boot'))
        with patch.object(pending, 'validate', side_effect=admitted):
            with self.assertRaisesRegex(ValueError, 'already admitted'):
                pending.edit(self.root, 'mine', command=['too-late'])
        self.assertEqual(self.saved(), before)

    def test_concurrent_editor_wins_without_lost_update(self):
        def another_editor(*_):
            pending.edit(self.root, 'mine', note='other editor')
        with patch.object(pending, 'validate', side_effect=another_editor):
            with self.assertRaisesRegex(ValueError, 'changed during preflight'):
                pending.edit(self.root, 'mine', command=['stale'])
        self.assertEqual(self.saved()['note'], 'other editor')
        self.assertEqual(self.saved()['command'][0], 'old')
        with self.assertRaisesRegex(ValueError, 'revision changed'):
            pending.edit(self.root, 'mine', note='stale', expected=1)

    def test_closed_dead_reused_and_legacy_waiters_are_rejected(self):
        before = self.saved()
        pending.transition(self.root, 'mine', 'finishing')
        with self.assertRaisesRegex(ValueError, 'started'):
            pending.edit(self.root, 'mine', note='too late')
        handoff.write(pending.path(self.root, 'mine'), before)
        with patch.object(handoff, 'identity', return_value='reused-pid'):
            with self.assertRaisesRegex(ValueError, 'no longer alive'):
                pending.edit(self.root, 'mine', note='wrong process')
        self.queue.write_text(self.queue.read_text().replace('20|mine', '21|mine'))
        with self.assertRaisesRegex(ValueError, 'predates'):
            pending.edit(self.root, 'mine', note='new ticket')
        pending.path(self.root, 'mine').unlink()
        with self.assertRaisesRegex(ValueError, 'predates'):
            pending.edit(self.root, 'mine', command=['legacy'])

    def test_experiment_payload_cannot_bypass_snapshot_but_note_can_change(self):
        value = self.saved(); value['experiment'] = 'immutable-id'
        handoff.write(pending.path(self.root, 'mine'), value)
        with self.assertRaisesRegex(ValueError, 'immutable evidence'):
            pending.edit(self.root, 'mine', command=['bypass'])
        with self.assertRaisesRegex(ValueError, 'immutable evidence'):
            pending.edit(self.root, 'mine', cwd=str(self.root))
        self.assertTrue(pending.edit(self.root, 'mine', note='clearer description')['changed'])

    def test_holder_metadata_reads_committed_record_after_interrupted_queue_write(self):
        with patch.object(pending, 'validate'):
            pending.edit(self.root, 'mine', command=['new'], estimate=9, note='new note')
        self.queue.write_text(self.queue.read_text().replace('|9|new note|', '|10|original|'))
        self.assertTrue(handoff.admit(self.root, 'mine', self.pid, 'boot', '10', 'original'))
        self.assertEqual((self.root / 'holder').read_text().split('|')[4:6], ['9', 'new note'])

    def test_target_validation_keeps_pinned_legacy_release_and_new_admission_contracts(self):
        import fleet_prepare
        prepared = dict(command=['old'], cwd=str(self.root), deployment_targets=[
            dict(repo=str(self.root), profile='glm53', image='candidate:image', model='/candidate/model')])
        for level in ('release', 'admission'):
            with self.subTest(level=level):
                pinned = self.root / ('pinned-' + level)
                pinned.mkdir()
                helper = pinned / 'fleet_validation.py'
                helper.write_text('''import argparse, json, os
parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['validate'])
parser.add_argument('--repo', required=True)
parser.add_argument('--profile', required=True)
parser.add_argument('--image')
parser.add_argument('--model')
parser.add_argument('--verify-only', action='store_true')
LEVEL_ARGUMENT
args = parser.parse_args()
assert args.profile == 'glm53' and args.image == 'candidate:image'
assert args.model == '/candidate/model' and args.verify_only
assert 'FLEET_VALIDATION_LEVEL' not in os.environ
print(json.dumps(dict(level=getattr(args, 'level', 'release'), helper=__file__,
                      store=os.environ['FLEET_VALIDATION_STORE'])))
'''.replace('LEVEL_ARGUMENT', "parser.add_argument('--level', choices=['admission'], required=True)"
            if level == 'admission' else '# Legacy CLI has no --level option.'))
                record = self.saved()
                record['fleet'] = str(pinned / 'fleet.sh')
                record['validation_env'] = {'FLEET_VALIDATION_REQUIRED': '1',
                                            'FLEET_VALIDATION_STORE': '/accepted/store'}
                if level == 'admission':
                    record['validation_env']['FLEET_VALIDATION_LEVEL'] = level
                with patch.dict(os.environ, FLEET_VALIDATION_LEVEL='admission',
                                FLEET_VALIDATION_STORE='/editor/store'):
                    result = fleet_prepare.validate_targets(self.root, 'unused', verify_only=True,
                                _validated_value=prepared, controller=record)[0]['validation']
                self.assertEqual(result, dict(level=level, helper=str(helper), store='/accepted/store'))

    def test_edit_preserves_fixed_approval_and_only_upgrades_capable_boot_controllers(self):
        import fleet_prepare
        import fleet_prepared
        original = self.saved()
        pinned = self.root / 'modern'
        pinned.mkdir()
        (pinned / 'fleet_approval.py').write_text('# Fixed approval capable controller.\n')
        cases = (
            ('signed', '/old/fleet.sh', 'boot', {}, True, True),
            ('modern', str(pinned / 'fleet.sh'), 'boot', {'FLEET_VALIDATION_REQUIRED': '1', 'FLEET_VALIDATION_LEVEL': 'admission'}, False, True),
            ('legacy', '/old/fleet.sh', 'boot', {'FLEET_VALIDATION_REQUIRED': '1', 'FLEET_VALIDATION_LEVEL': 'admission'}, False, False),
            ('release', str(pinned / 'fleet.sh'), 'boot', {'FLEET_VALIDATION_REQUIRED': '1'}, False, False),
            ('probe', str(pinned / 'fleet.sh'), 'probe', {'FLEET_VALIDATION_REQUIRED': '1', 'FLEET_VALIDATION_LEVEL': 'admission'}, False, False),
        )
        for name, fleet, kind, validation_env, approved, expected in cases:
            with self.subTest(case=name):
                record = dict(original, fleet=fleet, kind=kind, validation_env=validation_env,
                              prepare_manifest='/old.json', prepare_receipt_required=True)
                handoff.write(pending.path(self.root, 'mine'), record)
                old = dict(spec_path=None, deployment_approvals=[{'candidate': 'approved'}] if approved else [])
                with patch.object(pending, 'validate'), patch.object(fleet_prepared, 'read', return_value=old), \
                        patch.object(fleet_prepare, 'prepare', side_effect=[ValueError('changed argv'), Path('/new.json')]) as prepare, \
                        patch.object(fleet_prepare, 'validate_targets'):
                    result = pending.edit(self.root, 'mine', command=['new'])
                self.assertEqual(result['prepare_manifest'], '/new.json')
                self.assertEqual(result['ticket'], original['ticket'])
                self.assertEqual(len(prepare.call_args_list), 2)
                self.assertTrue(all(bool(call.kwargs.get('approve_deploy')) == expected
                                    for call in prepare.call_args_list))


class AcceptedPayloadTests(unittest.TestCase):
    def test_edited_receipt_wins_over_original_and_payload_environment(self):
        from fleet_boot import Supervisor
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.env = dict(PATH=os.environ['PATH'], FLEET_SESSION='owned', FLEET_DIR='/fleet',
                              FLEET_PREPARE_MANIFEST='/original.json', FLEET_VALIDATION_REQUIRED='1',
                              SSH_CLIENT='transport-only')
        for prefix in ([], ['/usr/bin/env', '-u', 'FLEET_PREPARE_MANIFEST'],
                       ['/usr/bin/env', '-i', 'FLEET_PREPARE_MANIFEST=/forged.json', 'FLEET_SESSION=foreign']):
            with self.subTest(prefix=prefix):
                command, environment = supervisor.accepted_payload(dict(
                    command=[*prefix, sys.executable, '-c', 'pass'], prepare_manifest='/accepted-edit.json'))
                self.assertEqual(command, [sys.executable, '-c', 'pass'])
                self.assertEqual(environment['FLEET_PREPARE_MANIFEST'], '/accepted-edit.json')
                self.assertEqual(environment['FLEET_SESSION'], 'owned')
                self.assertEqual(environment['FLEET_VALIDATION_REQUIRED'], '1')
                self.assertNotIn('SSH_CLIENT', environment)
        _, environment = supervisor.accepted_payload(dict(command=[sys.executable, '-c', 'pass']))
        self.assertNotIn('FLEET_PREPARE_MANIFEST', environment)


if __name__ == '__main__':
    unittest.main()
