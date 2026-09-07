"""Restore ownership and real Linux shell admission, using no GPUs or network."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bench'))
import fleet_handoff as handoff
import fleet_entry


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'queue').touch()
        self.patcher = patch.object(handoff, 'identity', side_effect=lambda pid: str(pid) if pid < 1000 else None)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def ready(self, name, pid, kind='boot', minutes=10):
        with (self.root/'queue').open('a') as stream:
            stream.write(f'{pid}|{name}|{time.time()}|{minutes}|fixture|{kind}|{pid}\n')
        return handoff.ready(self.root, name, pid)

    def test_only_actual_next_live_boot_supervisor_accepts_debt(self):
        self.ready('donor', 1)
        self.assertTrue(handoff.admit(self.root, 'donor', 1, 'boot'))
        self.ready('slow', 2, minutes=30)
        self.ready('next', 3, minutes=1)
        self.assertEqual(handoff.offer(self.root, 'donor')['session'], 'next')
        self.assertFalse(handoff.admit(self.root, 'slow', 2, 'boot'))
        self.assertFalse(handoff.admit(self.root, 'next', 3, 'probe'))
        self.assertTrue(handoff.admit(self.root, 'next', 3, 'boot'))
        with self.assertRaises(ValueError):
            handoff.clear(self.root, 'donor')
        handoff.clear(self.root, 'next')
        self.assertFalse((self.root/'restore-debt.json').exists())

    def test_probe_empty_dead_and_unmanaged_successors_require_restore(self):
        self.ready('donor', 1); handoff.admit(self.root, 'donor', 1, 'boot')
        self.assertIsNone(handoff.offer(self.root, 'donor'))
        self.ready('probe', 2, kind='probe', minutes=1)
        self.ready('boot', 3, minutes=30)
        self.assertIsNone(handoff.offer(self.root, 'donor'))
        (self.root/'queue').write_text(f'2|old|{time.time()}|1|fixture|boot|2\n')
        self.assertIsNone(handoff.offer(self.root, 'donor'))
        self.assertFalse(handoff.admit(self.root, 'old', 2, 'boot'))
        (self.root/'queue').write_text(f'9|dead|{time.time()}|1|fixture|boot|1001\n')
        self.assertIsNone(handoff.offer(self.root, 'donor'))

    def test_cancelled_target_and_pid_reuse_do_not_strand_a_new_boot(self):
        self.ready('donor', 1); handoff.admit(self.root, 'donor', 1, 'boot')
        self.ready('cancelled', 2); handoff.offer(self.root, 'donor')
        (self.root/'queue').write_text('')
        self.ready('replacement', 3)
        self.assertTrue(handoff.admit(self.root, 'replacement', 3, 'boot'))
        value = handoff.read(handoff.receipt(self.root, 'replacement'))
        value['start'] = 'different-process'
        self.assertFalse(handoff.live(value))

    def test_entry_requires_metrics_for_live_serving_but_accepts_stopped(self):
        self.assertEqual(fleet_entry.idle(None, 'unused'), 'stopped')
        self.assertEqual(fleet_entry.idle({'State':{'Running':False}}, 'unused'), 'stopped')
        container = dict(State=dict(Running=True), Config=dict(Cmd=['--port','18000']))
        from io import BytesIO
        response = BytesIO(b'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 1\n')
        response.status = 200
        with patch.object(fleet_entry.urllib.request, 'urlopen', return_value=response):
            with self.assertRaises((ValueError, OSError)):
                fleet_entry.idle(container, 'unused')


if __name__ == '__main__':
    unittest.main()
