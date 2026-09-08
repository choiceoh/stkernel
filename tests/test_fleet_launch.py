"""Detached launch acknowledgments use real CPU processes, without GPU access."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import fleet_launch as launch


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fleet = self.root / 'fleet'
        self.fleet.mkdir()
        self.helper = ROOT / 'bench/fleet_launch.py'
        self.script = self.root / 'fleet.sh'
        self.driver = self.root / 'driver.py'
        self.script.write_text('exec "' + sys.executable + '" "' + str(self.driver) + '" "$@"\n')
        self.driver.write_text('''import json, os, subprocess, sys, time
from pathlib import Path
import fleet_launch as launch
root = Path(os.environ['FIXTURE_ROOT'])
_, mode, session = sys.argv[1:]
directory = Path(os.environ['FLEET_DIR'])
with (root/'executions').open('a') as stream: stream.write(session+'\\n')
print('STARTUP '+session, flush=True)
if mode == 'fail':
 print('x'*50000+'\\nPREFLIGHT_FIXTURE_FAILED', flush=True)
 raise SystemExit(3)
if mode == 'no-ack':
 print('exited without an acknowledgment', flush=True)
 raise SystemExit(0)
if mode == 'delayed':
 while not (root/'register').exists(): time.sleep(.02)
if mode == 'cpu':
 subprocess.run([sys.executable, os.environ['HELPER'], 'cpu-started', session], check=True)
else:
 pid = os.getpid(); token = launch.identity(pid); ticket = 'ticket-'+str(pid)
 value = dict(session=session, pid=pid, start=token, ticket=ticket, state='queued',
              launch_id=os.environ['FLEET_LAUNCH_ID'])
 (directory/'pending').mkdir(exist_ok=True)
 path = directory/'pending'/(launch.hashlib.sha256(session.encode()).hexdigest()+'.json')
 with launch.locked(directory/'.lock', time.monotonic()+2):
  (directory/'queue').write_text(f'{ticket}|{session}|1|1|fixture|boot|{pid}\\n')
  launch.write(path, value)
  if mode.startswith('forged-'):
   request = Path(os.environ['FLEET_LAUNCH_REQUEST'])
   ack = dict(value)
   key = mode.removeprefix('forged-')
   ack[key] = pid+1 if key == 'pid' else 'wrong'
   launch.write(launch.ack_path(request),ack)
  else:
   launch.acknowledge(directory,session,ticket,pid=pid,start=token)
if mode.startswith('forged-'): raise SystemExit(0)
print('WAITING_OUTPUT '+session, flush=True)
while not (root/'finish').exists(): time.sleep(.02)
if mode == 'cpu':
 subprocess.run([sys.executable, os.environ['HELPER'], 'complete', session, '7'], check=True)
else:
 value.update(state='finished', outcome='succeeded', returncode=0)
 launch.write(path,value)
 (directory/'queue').write_text('')
print('FINISHED_OUTPUT '+session, flush=True)
''')
        self.env = dict(os.environ, FLEET_DIR=str(self.fleet), FIXTURE_ROOT=str(self.root),
                        HELPER=str(self.helper), PYTHONPATH=str(ROOT / 'bench'),
                        FLEET_LAUNCH_TIMEOUT='3')
        self.pids = set()
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for path in (self.fleet / 'launches').glob('*.json'):
            value = launch.read(path)
            if value and value.get('pid') and launch.live(value):
                self.pids.add(value['pid'])
        for pid in self.pids:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def cli(self, mode='queued', session='mine', timeout=None):
        env = dict(self.env)
        if timeout is not None:
            env['FLEET_LAUNCH_TIMEOUT'] = str(timeout)
        result = subprocess.run([sys.executable, str(self.helper), 'start', str(self.script), session,
                                 '--', 'run', mode, session], env=env, capture_output=True, text=True, timeout=8)
        value = json.loads(result.stdout or result.stderr)
        if value.get('pid'):
            self.pids.add(value['pid'])
        return result, value

    def until(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('fixture did not reach its expected state')
            time.sleep(.02)

    def test_registration_returns_before_completion_and_discovers_private_log(self):
        result, value = self.cli()
        self.assertEqual(result.returncode, 0, value)
        self.assertTrue(value['accepted']); self.assertEqual(value['state'], 'queued')
        self.assertTrue(launch.identity(value['pid']))
        self.assertEqual(os.getsid(value['pid']), value['pid'])
        self.assertFalse((self.root / 'finish').exists())
        log = Path(value['startup_log'])
        self.until(lambda:'WAITING_OUTPUT mine' in log.read_text())
        self.assertEqual(log.stat().st_mode & 0o777, 0o600)
        self.assertEqual(log.parent.stat().st_mode & 0o777, 0o700)
        retry, same = self.cli()
        self.assertEqual(retry.returncode, 0, same)
        self.assertEqual(same['pid'], value['pid']); self.assertEqual(same['disposition'], 'existing')
        self.assertEqual((self.root / 'executions').read_text(), 'mine\n')

    def test_preflight_failure_is_bounded_and_retained_on_retry(self):
        result, value = self.cli('fail')
        self.assertEqual(result.returncode, 3, value)
        self.assertFalse(value['accepted'])
        self.assertEqual(value['state'], 'startup-failed')
        self.assertIn('PREFLIGHT_FIXTURE_FAILED', value['log_tail'])
        self.assertLessEqual(len(value['log_tail']), launch.MAX_TAIL)
        self.assertFalse((self.fleet / 'queue').exists())
        again, previous = self.cli('fail')
        self.assertEqual(again.returncode, 3)
        self.assertEqual(previous['launch_id'], value['launch_id'])
        self.assertEqual((self.root / 'executions').read_text(), 'mine\n')

    def test_existing_queue_and_different_command_do_not_launch_duplicates(self):
        (self.fleet / 'queue').write_text(f'old|mine|1|1|old|boot|{os.getpid()}\n')
        result, value = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn('already queued', value['error'])
        self.assertFalse((self.root / 'executions').exists())
        (self.fleet / 'queue').write_text('')
        first, _ = self.cli()
        self.assertEqual(first.returncode, 0)
        result, value = self.cli('cpu')
        self.assertEqual(result.returncode, 2)
        self.assertIn('different arguments', value['error'])
        self.assertEqual((self.root / 'executions').read_text(), 'mine\n')

    def test_timeout_retry_joins_original_startup(self):
        result, value = self.cli('delayed', timeout=.1)
        self.assertEqual(result.returncode, 124, value)
        self.assertFalse(value['accepted']); self.assertEqual(value['state'], 'starting')
        (self.root / 'register').touch()
        result, retried = self.cli('delayed')
        self.assertEqual(result.returncode, 0, retried)
        self.assertEqual(retried['pid'], value['pid'])
        self.assertEqual((self.root / 'executions').read_text(), 'mine\n')

    def test_cpu_starts_without_waiting_and_preserves_terminal_exit(self):
        result, value = self.cli('cpu')
        self.assertEqual(result.returncode, 0, value)
        self.assertEqual(value['state'], 'running-cpu')
        self.assertFalse((self.fleet / 'queue').exists())
        (self.root / 'finish').touch()
        self.until(lambda:'FINISHED_OUTPUT' in Path(value['startup_log']).read_text())
        result, finished = self.cli('cpu')
        self.assertEqual(result.returncode, 7, finished)
        self.assertEqual(finished['state'], 'finished-cpu')
        self.assertEqual(finished['returncode'], 7)
        self.assertEqual((self.root / 'executions').read_text(), 'mine\n')

    def test_successful_process_without_ack_is_not_successful_launch(self):
        result, value = self.cli('no-ack')
        self.assertEqual(result.returncode, 2, value)
        self.assertFalse(value['accepted']); self.assertEqual(value['state'], 'startup-failed')
        self.assertIn('without an acknowledgment', value['log_tail'])

    def test_receipt_nonce_pid_start_and_ticket_must_match(self):
        for key in ('launch_id', 'pid', 'start', 'ticket'):
            with self.subTest(key=key):
                result, value = self.cli('forged-' + key, session='forged-' + key)
                self.assertNotEqual(result.returncode, 0, value)
                self.assertFalse(value['accepted'])
                self.assertEqual(value['state'], 'startup-failed')


if __name__ == '__main__':
    unittest.main()
