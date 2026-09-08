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
        self.pid = os.getpid()
        self.queue = self.root / 'queue'
        self.queue.write_text(f'10|before|100|5|before|boot|{self.pid}\n'
                             f'20|mine|101|10|original|boot|{self.pid}\n'
                             f'30|after|102|5|after|boot|{self.pid}\n')
        for patcher in (patch.object(handoff, 'identity', return_value='owner-start'),
                        patch.dict(os.environ, REPO=str(self.root), FLEET_EXPERIMENT_ID='')):
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


if __name__ == '__main__':
    unittest.main()
