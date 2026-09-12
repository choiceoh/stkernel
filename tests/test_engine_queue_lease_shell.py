"""Exercise the pinned shell runner's actual lease dependency and argument path."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import fleet_pin


class QueueLeaseShellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runner = fleet_pin.pin(ROOT, self.root)
        self.lease = self.root / 'lease with spaces'
        self.env = dict(os.environ, FLEET_REPO=str(self.runner), REPO=str(self.runner),
                        FLEET_LEASE_PATH=str(self.lease), FLEET_HEAD='test-head')
        self.helper = '. "$FLEET_REPO/launchers/lib/fleet-lease.sh"\n'
        self.local = self.helper + '_fleet_lease_is_head() { return 0; }\n'

    def shell(self, script):
        return subprocess.run(['bash', '-euc', script], env=self.env, capture_output=True,
                              text=True, timeout=10, check=True).stdout

    def test_pinned_local_and_remote_helpers_preserve_arguments(self):
        owner = "owner's boot $(touch should-not-exist)"
        self.env['TEST_OWNER'] = owner
        for remote in (False, True):
            with self.subTest(remote=remote):
                # Execute the actual SSH command string with Bash, but never open a
                # connection. This also checks shell metacharacters stay literal.
                prelude = self.local if not remote else self.helper + '''
_fleet_lease_is_head() { return 1; }
ssh() { local command="${!#}"; bash -c "$command"; }
'''
                self.shell(prelude + 'fleet_lease acquire --owner "$TEST_OWNER" --note "two word note"')
                record = json.loads(self.lease.read_text())
                self.assertEqual(record['owner'], owner)
                self.assertEqual(record['note'], 'two word note')
                self.shell(prelude + 'fleet_lease renew --owner "$TEST_OWNER"')
                self.shell(prelude + 'fleet_lease release --owner "$TEST_OWNER"')
                self.assertFalse(self.lease.exists())

    def test_actual_queue_functions_read_the_lease_from_the_snapshot(self):
        """The queue reads and takes the lease through its own pinned module (bench/fleet.sh
        `lease`), never through the launchers/ helper: a runner snapshot that could not read
        the lease counted it as occupied, and no runner-driven ticket was ever granted."""
        import tempfile
        source = (self.runner / 'bench/fleet.sh').read_text()
        prelude = source[:source.index('\ncmd=${1:-status}')]
        with tempfile.TemporaryDirectory() as fleet_dir:
            env = dict(self.env, FLEET_DIR=fleet_dir + '/fleet', LOGD=fleet_dir, FLEET_RUNNER_REPO=str(self.runner))
            def queue(script):
                return subprocess.run(['bash', '-c', prelude + '\n' + script], env=env, capture_output=True,
                                      text=True, timeout=30, check=True).stdout.strip()
            self.assertEqual(queue('st_engine_lease'), '')                       # free: silence
            self.assertEqual(queue('lease_kind'), 'free')
            self.shell(self.local + 'fleet_lease acquire --owner "production/srv2/1" --kind production')
            self.assertIn('production production/srv2/1', queue('st_engine_lease'))
            self.assertEqual(queue('lease_kind'), 'production')
            self.shell(self.local + 'fleet_lease release --owner "production/srv2/1"')
            # a lease handed to the very ticket that is asking is not occupation
            self.shell(self.local + 'fleet_lease acquire --owner "queue/t1" --kind queue --pid 1')
            self.assertIn('queue queue/t1', queue('st_engine_lease'))
            self.assertEqual(queue('ST_MINE=t1 st_engine_lease'), '')
            self.assertEqual(queue('lease_mine t1 && echo mine'), 'mine')


if __name__ == '__main__':
    unittest.main()
