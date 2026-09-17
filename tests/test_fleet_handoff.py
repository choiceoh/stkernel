"""Fleet admission and deferred recovery, using no GPUs or network."""
import hashlib
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

    def test_managed_admission_never_creates_restore_debt(self):
        self.ready('donor', 1)
        self.assertTrue(handoff.admit(self.root, 'donor', 1, 'boot'))
        self.assertFalse((self.root/'restore-debt.json').exists())
        self.assertTrue((self.root/'holder').read_text().startswith('donor|1|'))

    def test_single_lane_admission_writes_its_own_holder_and_leaves_the_fleet_alone(self):
        """A check on the 5050 is not the fleet being held: its own holder file, no idle-clock
        reset, no restore-debt change, no managed receipt (2026-09-12)."""
        with (self.root/'queue').open('a') as stream:
            stream.write(f'7|check|{time.time()}|5|kernel check|single|7\n')
        handoff.write(self.root/'restore-debt.json', dict(owner=dict(session='old'), target=None))
        self.assertTrue(handoff.admit(self.root, 'check', 7, 'single', '5', 'kernel check'))
        row = (self.root/'holder-single').read_text().strip().split('|')
        self.assertEqual((row[0], row[1], row[4], row[5], row[6]), ('check', '7', '5', 'kernel check', 'single'))
        self.assertFalse((self.root/'holder').exists())               # the fleet is not held
        self.assertTrue((self.root/'restore-debt.json').exists())     # the fleet's debt is not ours to clear
        self.assertFalse((self.root/'idle-recovery.json').exists())   # and no fleet activity was recorded
        self.assertEqual(list(handoff.holders(self.root)), ['single'])
        # the fleet lane admits beside it, into its own file
        self.ready('donor', 1)
        self.assertTrue(handoff.admit(self.root, 'donor', 1, 'boot'))
        self.assertEqual(sorted(handoff.holders(self.root)), ['fleet', 'single'])
        self.assertTrue((self.root/'holder').read_text().startswith('donor|1|'))
        self.assertTrue((self.root/'idle-recovery.json').exists())
        self.assertEqual(handoff.holder_path(self.root, 'single'), self.root/'holder-single')
        self.assertEqual(handoff.holder_path(self.root, 'probe'), self.root/'holder')

    def test_legacy_debt_and_target_do_not_block_queued_probe(self):
        owner = self.ready('old', 1)
        target = self.ready('target', 2)
        handoff.write(self.root/'restore-debt.json', dict(owner=owner, target=target))
        self.ready('probe', 3, kind='probe')
        self.assertTrue(handoff.admit(self.root, 'probe', 3, 'probe'))
        self.assertFalse((self.root/'restore-debt.json').exists())
        self.assertTrue((self.root/'holder').read_text().startswith('probe|3|'))

    def test_legacy_debt_does_not_reorder_priority_or_bypass_probe_readiness(self):
        import fleet_priority
        owner = self.ready('old', 1)
        target = self.ready('target', 2, minutes=30)
        self.ready('probe', 3, kind='probe', minutes=1)
        (self.root/'queue').write_text('\n'.join(line for line in (self.root/'queue').read_text().splitlines()
                                              if line.split('|')[1] != 'old')+'\n')
        handoff.write(self.root/'restore-debt.json', dict(owner=owner, target=target))
        for boot_only, expected in ((False, 'probe'), (True, 'target')):
            with self.subTest(boot_only=boot_only), patch.object(sys, 'argv',
                    ['fleet_priority.py', str(self.root), '--apply'] + (['--boot-only'] if boot_only else [])):
                fleet_priority.main()
                self.assertEqual((self.root/'queue').read_text().splitlines()[0].split('|')[1], expected)

    def test_cancelled_target_and_pid_reuse_do_not_strand_a_new_boot(self):
        owner = self.ready('donor', 1)
        target = self.ready('cancelled', 2)
        handoff.write(self.root/'restore-debt.json', dict(owner=owner, target=target))
        (self.root/'queue').write_text('')
        self.ready('replacement', 3)
        self.assertTrue(handoff.admit(self.root, 'replacement', 3, 'boot'))
        self.assertFalse((self.root/'restore-debt.json').exists())
        value = handoff.read(handoff.receipt(self.root, 'replacement'))
        value['start'] = 'different-process'
        self.assertFalse(handoff.live(value))

    def test_interrupted_admission_leaves_holder_without_restore_debt(self):
        self.ready('next', 2)
        with patch.object(handoff, 'claim_held', side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):
                handoff.admit(self.root, 'next', 2, 'boot')
        self.assertTrue((self.root/'holder').read_text().startswith('next|2|'))
        self.assertFalse((self.root/'restore-debt.json').exists())
        handoff.claim_held(self.root, 'next', 2)
        self.assertFalse((self.root/'restore-debt.json').exists())
        with self.assertRaises(ValueError):
            handoff.claim_held(self.root, 'donor', 1)

    def test_previous_protocol_is_not_a_current_ready_supervisor(self):
        old = self.ready('old', 2); old['protocol'] = 1
        handoff.write(handoff.receipt(self.root, 'old'), old)
        self.assertFalse(handoff.live(old))

    def test_admission_resets_central_idle_clock(self):
        self.ready('next', 2)
        with patch('fleet_idle.activity') as activity:
            self.assertTrue(handoff.admit(self.root, 'next', 2, 'boot'))
        activity.assert_called_once_with(self.root, 'acquire')



if __name__ == '__main__':
    unittest.main()
