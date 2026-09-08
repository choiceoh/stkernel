"""Late supervisor cleanup cannot remove a reused session's reservation."""
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import fleet_boot as boot
import fleet_handoff as handoff


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pid = os.getpid()
        with patch.dict(os.environ, FLEET_DIR=str(self.root), FLEET_RUN_KIND='boot'):
            self.supervisor = boot.Supervisor('/pinned/fleet.sh', 'reused', '5', 'fixture', ['true'])
        patcher = patch.object(handoff, 'identity', return_value='owner-start')
        patcher.start(); self.addCleanup(patcher.stop)
        self.receipt = handoff.receipt(self.root, 'reused')

    def write(self, **changes):
        value = dict(session='reused', pid=self.pid, start='owner-start', host=socket.gethostname(),
                     protocol=handoff.PROTOCOL)
        value.update(changes)
        handoff.write(self.receipt, value)
        return self.receipt.read_bytes()

    def cleanup(self):
        with patch.object(self.supervisor, 'execute', return_value=0) as execute:
            self.assertEqual(self.supervisor.cleanup_reservation(), 0)
        execute.assert_called_once_with(['bash', '/pinned/fleet.sh', 'withdraw', 'reused', '--pid', str(self.pid)],
                                        self.supervisor.env, interruptible=False)

    def test_cancelled_owner_removes_only_its_receipt_and_scopes_withdrawal(self):
        self.write()
        self.supervisor.stopping = 143
        self.cleanup()
        self.assertFalse(self.receipt.exists())
        self.assertEqual(self.supervisor.stopping, 143)

    def test_reused_session_receipt_and_queue_survive_previous_owner_cleanup(self):
        before = self.write(pid=self.pid + 10, start='new-supervisor')
        queue = self.root / 'queue'
        queue.write_text(f'new-ticket|reused|100|5|new request|boot|{self.pid + 10}\n')
        row = queue.read_bytes()
        self.cleanup()
        self.assertEqual(self.receipt.read_bytes(), before)
        self.assertEqual(queue.read_bytes(), row)

    def test_same_pid_with_different_start_host_or_session_is_not_owned(self):
        for changes in (dict(start='another-start'), dict(host='another-host'), dict(session='another-session')):
            with self.subTest(changes=changes):
                before = self.write(**changes)
                self.cleanup()
                self.assertEqual(self.receipt.read_bytes(), before)

    def test_unreadable_receipt_still_permits_owned_queue_cleanup(self):
        self.receipt.write_text('{invalid-json')
        with patch.object(self.supervisor, 'warning') as warning:
            self.cleanup()
        self.assertEqual(self.receipt.read_text(), '{invalid-json')
        warning.assert_called_once()


if __name__ == '__main__':
    unittest.main()
