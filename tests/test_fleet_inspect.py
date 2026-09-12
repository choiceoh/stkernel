"""Fast, read-only reservation inspection and bounded output retrieval."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_handoff as handoff
import fleet_inspect as inspect
import fleet_pending as pending


class InspectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pid = os.getpid()
        self.queue = self.root / 'queue'
        self.queue.write_text(f'20|mine|100|10|fixture|boot|{self.pid}\n')
        for patcher in (patch.object(handoff, 'identity', return_value='start'),
                        patch.dict(os.environ, REPO=str(self.root), FLEET_EXPERIMENT_ID='')):
            patcher.start(); self.addCleanup(patcher.stop)
        pending.register(self.root, 'mine', ['bash', 'a b.sh'], '/pinned/fleet.sh', 'boot')
        pending.transition(self.root, 'mine', 'queued', log_path=str(self.root / 'output.log'), phase='waiting')

    def test_show_retains_argv_after_go_and_never_mutates_fleet(self):
        before = {p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        result = inspect.show(self.root, 'mine')
        self.assertEqual(result['position'], 1)
        self.assertTrue(result['editable'])
        self.assertEqual(result['actions']['edit'][-2:], ['--expect-revision', '1'])
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()})
        handoff.admit(self.root, 'mine', self.pid, 'boot')
        self.queue.write_text('')
        pending.transition(self.root, 'mine', 'running', phase='payload', started_at=120)
        result = inspect.show(self.root, 'mine')
        self.assertEqual(result['state'], 'running')
        self.assertEqual(result['command'], ['bash', 'a b.sh'])
        self.assertFalse(result['editable'])
        self.assertEqual(result['wait_seconds'], 20)

    def test_a_single_lane_check_shows_beside_the_fleet_holder_and_waits_on_its_own(self):
        """Two lanes, two holders: a queued one-GPU check waits behind holder-single, and the
        fleet's holder is never what it waits for (2026-09-12)."""
        host = inspect.socket.gethostname().split('.')[0]
        (self.root/'holder').write_text(f'boot|{self.pid}|{host}|100|30|a boot|boot\n')
        (self.root/'holder-single').write_text(f'check|{self.pid}|{host}|110|5|a kernel check|single\n')
        self.queue.write_text(f'20|mine|100|10|fixture|boot|{self.pid}\n30|next|120|5|next check|single|{self.pid}\n')
        listing = {row['session']:row for row in inspect.show(self.root)}
        self.assertEqual((listing['check']['state'], listing['check']['kind']), ('running', 'single'))
        self.assertEqual(listing['boot']['state'], 'running')
        self.assertEqual(listing['next']['waiting_for'], 'holder check')
        self.assertEqual(listing['mine']['waiting_for'], 'holder boot')
        (self.root/'holder-single').unlink()
        self.assertEqual(inspect.show(self.root, 'next')['waiting_for'], 'single-GPU admission checks')
        self.assertEqual(inspect.show(self.root, 'mine')['waiting_for'], 'holder boot')

    def test_completed_failure_distinguishes_successful_payload_from_failed_restore(self):
        self.queue.write_text('')
        pending.transition(self.root, 'mine', 'finished', phase='finished', outcome='failed',
                           started_at=120, payload_finished_at=130, finished_at=150,
                           payload_returncode=0, recovery_returncode=1, returncode=1)
        result = inspect.show(self.root, 'mine')
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['returncode'], 1)
        self.assertEqual(result['payload_returncode'], 0)
        self.assertEqual(result['payload_seconds'], 10)
        self.assertFalse(result['editable'])
        self.assertEqual(result['actions']['logs'], ['fleet.sh', 'logs', 'mine'])

    def test_old_finished_or_dead_records_are_never_called_successful(self):
        self.queue.write_text('')
        pending.transition(self.root, 'mine', 'finished')
        self.assertEqual(inspect.show(self.root, 'mine')['state'], 'finished')
        self.assertNotIn('returncode', inspect.show(self.root, 'mine'))
        pending.transition(self.root, 'mine', 'running', payload_returncode=0)
        with patch.object(handoff, 'identity', return_value=None):
            result = inspect.show(self.root, 'mine')
        self.assertEqual(result['state'], 'interrupted')
        self.assertNotIn('outcome', result)

    def test_reused_session_never_exposes_previous_command_or_log(self):
        self.queue.write_text(f'21|mine|200|5|new request|boot|{self.pid}\n')
        with patch.object(inspect, 'process_log', return_value=None):
            result = inspect.show(self.root, 'mine')
        self.assertEqual(result['source'], 'legacy')
        self.assertNotIn('command', result)
        self.assertNotIn('revision', result)
        self.assertFalse(result['editable'])
        self.assertIsNone(result.get('log_path'))

    def test_corrupt_saved_record_leaves_legacy_queue_readable(self):
        pending.path(self.root, 'mine').write_text('{broken')
        with patch.object(inspect, 'process_log', return_value='/existing/output.log'):
            result = inspect.show(self.root, 'mine')
        self.assertEqual(result['state'], 'queued')
        self.assertEqual(result['log_source'], 'existing stdout file')
        self.assertTrue(result['warnings'])
        self.assertFalse(result['editable'])

    def test_empty_snapshot_does_not_create_a_fleet(self):
        missing = self.root / 'not-created'
        self.assertEqual(inspect.show(missing), [])
        self.assertFalse(missing.exists())
        with self.assertRaisesRegex(ValueError, 'unknown reservation'):
            inspect.show(missing, 'missing')

    def test_reused_pid_holder_does_not_inherit_old_success_command_or_log(self):
        self.queue.write_text('')
        pending.transition(self.root, 'mine', 'finished', outcome='succeeded', returncode=0)
        (self.root/'holder').write_text(f'mine|{self.pid}|{inspect.socket.gethostname().split(".")[0]}|200|10|new holder|boot\n')
        with patch.object(handoff, 'identity', return_value='new-process'), patch.object(inspect, 'process_log', return_value=None):
            result = inspect.show(self.root, 'mine')
        self.assertEqual(result['source'], 'legacy')
        self.assertEqual(result['state'], 'running')
        for field in ('outcome', 'command', 'returncode'):
            self.assertNotIn(field, result)
        self.assertFalse(result.get('log_path'))

    def test_dead_finishing_holder_is_interrupted(self):
        handoff.admit(self.root, 'mine', self.pid, 'boot')
        self.queue.write_text('')
        pending.transition(self.root, 'mine', 'finishing', phase='restore')
        with patch.object(handoff, 'identity', return_value=None):
            result = inspect.show(self.root, 'mine')
        self.assertEqual(result['state'], 'interrupted')
        self.assertFalse(result['supervisor_alive'])

    def test_saved_edit_metadata_wins_over_interrupted_queue_projection(self):
        old_queue = self.queue.read_text()
        pending.edit(self.root, 'mine', note='new description', estimate=3)
        self.queue.write_text(old_queue)
        result = inspect.show(self.root, 'mine')
        self.assertEqual(result['note'], 'new description')
        self.assertEqual(result['estimate_min'], 3)
        self.assertEqual(result['position'], 1)

    def test_tail_reads_only_bounded_suffix_and_rejects_fifo(self):
        log = self.root / 'large.log'
        log.write_bytes(b'x' * (inspect.MAX_LOG_BYTES * 5) + b'\nfirst\nsecond\nlast\n')
        self.assertEqual(inspect.tail(log, 2), 'second\nlast')
        self.assertLessEqual(len(inspect.tail(log, 2000).encode()), inspect.MAX_LOG_BYTES)
        fifo = self.root / 'pipe'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, 'not a regular file'):
            inspect.tail(fifo, 2)


if __name__ == '__main__':
    unittest.main()
