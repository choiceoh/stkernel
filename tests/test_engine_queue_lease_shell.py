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

    def test_actual_queue_functions_read_and_request_yield_from_snapshot(self):
        source = (self.runner / 'bench/fleet.sh').read_text()
        functions = '\n'.join(name + '() {' + source.split(name + '() {', 1)[1].split('\n}', 1)[0] + '\n}'
                              for name in ('st_engine_lease', 'st_engine_yield'))
        # A shell function exported into each helper's subshell identifies this host.
        prelude = 'hostname() { printf "test-head\\n"; }\n' + functions + '\n'
        self.assertEqual(self.shell(prelude + 'st_engine_lease'), '')
        self.shell(self.local + 'fleet_lease acquire --owner holder')
        held = self.shell(prelude + 'st_engine_lease; st_engine_yield waiter')
        self.assertIn('holder', held)
        record = json.loads(self.lease.read_text())
        self.assertEqual(record['yield_to']['requester'], 'queue/waiter')
        self.assertEqual(record['yield_to']['reason'], 'a queued reservation needs the fleet')
        # Missing dependency is still a refusal, never a free fleet.
        (self.runner / 'engine/base/fleet_lease.py').unlink()
        self.assertIn('unreadable', self.shell(prelude + 'st_engine_lease'))

    def test_heartbeat_pid_capture_does_not_wait_for_background_loop(self):
        process = subprocess.Popen(['bash', '-euc', self.local +
                                    'BEAT=$(fleet_lease_beat holder); printf "%s\\n" "$BEAT"'],
                                   env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            out, err = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 0, err)
            os.kill(int(out.strip()), 0)  # the heartbeat is alive while its caller has returned
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
