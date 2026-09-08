"""Ticket history remains addressable across session reuse and partial writes."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_handoff as handoff
import fleet_inspect as inspect
import fleet_pending as pending


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pid = os.getpid()
        self.queue = self.root / 'queue'
        for patcher in (patch.object(handoff, 'identity', return_value='owner-start'),
                        patch.dict(os.environ, REPO=str(self.root), FLEET_DIR=str(self.root),
                                   FLEET_EXPERIMENT_ID='', FLEET_LAUNCH_ID='', FLEET_PREPARE_MANIFEST='')):
            patcher.start(); self.addCleanup(patcher.stop)
        self.register('20', ['bash', 'first.sh'])

    def register(self, ticket, command=None):
        self.queue.write_text(f'{ticket}|mine|100|10|fixture|boot|{self.pid}\n')
        return pending.register(self.root, 'mine', command or ['bash', ticket + '.sh'], '/pinned/fleet.sh', 'boot')

    def finish(self, **details):
        self.queue.write_text('')
        pending.transition(self.root, 'mine', 'finished', outcome='succeeded', returncode=0, **details)

    def test_reuse_retains_exact_old_result_and_log_without_current_ownership(self):
        old_log = self.root / 'old.log'; old_log.write_text('old result\n')
        self.finish(log_path=str(old_log))
        self.register('21', ['bash', 'second.sh'])
        current = inspect.show(self.root, 'mine')
        old = inspect.show(self.root, 'mine', '20')
        self.assertEqual(current['ticket'], '21')
        self.assertEqual(current['command'], ['bash', 'second.sh'])
        self.assertTrue(current['editable'])
        self.assertNotIn('outcome', current)
        self.assertEqual(old['state'], 'succeeded')
        self.assertEqual(old['command'], ['bash', 'first.sh'])
        self.assertTrue(old['historical'])
        self.assertFalse(old['editable']); self.assertFalse(old['supervisor_alive'])
        self.assertIsNone(old['position'])
        self.assertEqual(old['actions']['logs'], ['fleet.sh', 'logs', 'mine', '--ticket', '20'])
        self.assertEqual([(v['ticket'], v['state']) for v in inspect.history(self.root, 'mine')],
                         [('21', 'queued'), ('20', 'succeeded')])

    def test_edits_and_terminal_transitions_update_the_same_ticket(self):
        with patch.object(pending, 'validate'):
            pending.edit(self.root, 'mine', command=['bash', 'changed.sh'], expected=1)
        pending.transition(self.root, 'mine', 'running', payload_returncode=0)
        self.finish()
        self.register('21')
        old = pending.read_record(self.root, 'mine', '20')
        self.assertEqual(old['revision'], 2)
        self.assertEqual(old['command'], ['bash', 'changed.sh'])
        self.assertEqual(old['history'][0]['command'], ['bash', 'first.sh'])
        self.assertEqual(old['state'], 'finished')
        self.assertEqual(old['payload_returncode'], 0)

    def test_first_reuse_archives_a_pre_feature_record(self):
        self.finish()
        shutil.rmtree(pending.history_directory(self.root, 'mine').parent)
        self.register('21')
        self.assertEqual(pending.read_record(self.root, 'mine', '20')['outcome'], 'succeeded')
        self.assertEqual([v['ticket'] for v in pending.history(self.root, 'mine')], ['21', '20'])

    def test_discovery_is_bounded_and_old_tickets_remain_directly_addressable(self):
        with patch.object(pending, 'HISTORY_LIMIT', 3):
            for ticket in ('21', '22', '23', '24'):
                self.finish()
                self.register(ticket)
            with patch.object(Path, 'iterdir', side_effect=AssertionError('history must not scan directories')):
                self.assertEqual([v['ticket'] for v in pending.history(self.root, 'mine', 3)], ['24', '23', '22'])
                self.assertEqual(pending.read_record(self.root, 'mine', '20')['ticket'], '20')
                self.assertEqual(len(inspect.history(self.root, 'mine', 2)), 2)
                self.assertEqual(inspect.history(self.root, 'unknown', 2), [])
            with self.assertRaisesRegex(ValueError, '--limit'):
                pending.history(self.root, 'mine', 4)
        for directory in (self.root / 'pending', pending.history_directory(self.root, 'mine').parent,
                          pending.history_directory(self.root, 'mine')):
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)

    def test_partial_projection_write_does_not_commit_an_edit(self):
        before = pending.read_record(self.root, 'mine')
        original_write = handoff.write
        def fail_current(filename, value):
            if filename == pending.path(self.root, 'mine'):
                raise OSError('current record write failed')
            return original_write(filename, value)
        with patch.object(handoff, 'write', side_effect=fail_current):
            with self.assertRaisesRegex(OSError, 'write failed'):
                pending.edit(self.root, 'mine', note='uncommitted edit')
        self.assertEqual(pending.read_record(self.root, 'mine'), before)
        self.assertEqual(pending.read_record(self.root, 'mine', '20'), before)
        self.register('21')
        self.assertEqual(pending.read_record(self.root, 'mine', '20'), before)

    def test_old_active_record_and_reused_pid_are_not_successful_or_editable(self):
        pending.transition(self.root, 'mine', 'running', payload_returncode=0)
        self.register('21')
        old = inspect.show(self.root, 'mine', '20')
        self.assertEqual(old['state'], 'interrupted')
        self.assertFalse(old['editable'])
        self.assertFalse(old['supervisor_alive'])
        self.assertNotIn('outcome', old)
        with patch.object(handoff, 'identity', return_value='different-start'):
            with self.assertRaisesRegex(ValueError, 'different supervisor'):
                self.register('21')

    def test_finished_record_cannot_become_a_new_holder_even_with_the_same_pid(self):
        self.finish()
        (self.root / 'holder').write_text(f'mine|{self.pid}|{inspect.socket.gethostname().split(".")[0]}|200|10|new holder|boot\n')
        with patch.object(inspect, 'process_log', return_value=None):
            current = inspect.show(self.root, 'mine')
            old = inspect.show(self.root, 'mine', '20')
        self.assertEqual(current['source'], 'legacy')
        self.assertNotIn('outcome', current)
        self.assertNotIn('command', current)
        self.assertEqual(old['state'], 'succeeded')
        self.assertTrue(old['historical'])
        self.assertFalse(old['editable'])

    def test_ticket_identity_is_checked_and_cannot_select_another_session(self):
        self.finish()
        self.register('21')
        target = pending.ticket_path(self.root, 'mine', '20')
        value = handoff.read(target); value['session'] = 'someone-else'
        handoff.write(target, value)
        with self.assertRaisesRegex(ValueError, 'identity'):
            pending.read_record(self.root, 'mine', '20')
        with self.assertRaisesRegex(ValueError, 'unknown reservation ticket'):
            inspect.show(self.root, 'mine', '../../unrelated')
        with self.assertRaisesRegex(ValueError, 'requires'):
            inspect.show(self.root, ticket='20')

    def test_cli_history_summary_and_ticket_logs_are_bounded_and_read_only(self):
        old_log = self.root / 'old.log'; old_log.write_text('first\nold result\n')
        self.finish(log_path=str(old_log))
        self.register('21')
        pending.edit(self.root, 'mine', note='n' * 1000)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(inspect.main(['history', 'mine', '--limit', '1', '--json']), 0)
        summary = json.loads(output.getvalue())[0]
        self.assertEqual(len(summary['note']), 240)
        self.assertNotIn('command', summary); self.assertNotIn('cwd', summary)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(inspect.main(['logs', 'mine', '--ticket', '20', '--tail', '1']), 0)
        self.assertEqual(output.getvalue(), 'old result\n')
        with redirect_stderr(io.StringIO()):
            self.assertEqual(inspect.main(['history', 'mine', '--limit', '0']), 2)
        missing = self.root / 'absent'
        self.assertEqual(inspect.history(missing, 'unknown'), [])
        self.assertFalse(missing.exists())

    def test_oversized_index_is_rejected_without_scanning(self):
        index = pending.history_directory(self.root, 'mine') / 'index.json'
        index.write_bytes(b' ' * (pending.INDEX_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'size limit'):
            pending.history(self.root, 'mine')

    def test_launch_and_preparation_receipts_are_preserved_on_registration(self):
        with patch.dict(os.environ, FLEET_PREPARE_MANIFEST='/private/prepared.json'):
            value = self.register('21')
        self.assertEqual(value['prepare_manifest'], '/private/prepared.json')
        self.assertEqual(pending.read_record(self.root, 'mine')['prepare_manifest'], '/private/prepared.json')


if __name__ == '__main__':
    unittest.main()
