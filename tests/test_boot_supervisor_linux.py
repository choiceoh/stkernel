"""Linux-only integration gate; run explicitly on a Linux CPU validation host."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from test_fleet_handoff import ROOT, handoff

class LinuxSupervisorTests(unittest.TestCase):
    """Run this suite on Linux; fixture commands replace all system/GPU I/O."""
    def setUp(self):
        if not Path('/proc/self/stat').exists() or not shutil.which('flock'):
            self.fail('Linux supervisor integration requires Linux and flock; run on the CPU validation host')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.logs, self.bin = [self.root/p for p in ('repo', 'logs', 'bin')]
        for path in (self.repo/'bench', self.repo/'profiles', self.logs/'fleet', self.bin):
            path.mkdir(parents=True)
        for name in ('fleet.sh', 'fleet_boot.py', 'fleet_handoff.py', 'fleet_priority.py', 'experiment_metrics.py'):
            shutil.copy(ROOT/'bench'/name, self.repo/'bench'/name)
        (self.repo/'profiles/glm53.env').write_text('VLLM_TEST=0\n')
        (self.repo/'bench/fleet_restore.sh').write_text('''#!/bin/bash
echo "$FLEET_SESSION" >> "$LOGD/restores"
sleep .05
test ! -e "$LOGD/fail-restore"
''')
        mocks = {'ps':'exit 0', 'ssh':'echo gpu=ok', 'docker':'exit 0',
                 'curl':'case "$*" in *metrics*) echo "vllm:num_requests_running{} 0";; *) echo 200;; esac',
                 'git':'exit 1'}
        for name, code in mocks.items():
            (self.bin/name).write_text('#!/bin/sh\n'+code+'\n'); (self.bin/name).chmod(0o755)
        self.env = {k:v for k,v in os.environ.items() if not k.startswith(('FLEET_', 'ONEPASS_'))}
        self.env.update(REPO=str(self.repo), FLEET_DIR=str(self.logs/'fleet'), LOGD=str(self.logs),
                        PATH=str(self.bin)+':'+os.environ['PATH'], FLEET_NODES_IPS='fixture',
                        FLEET_NO_RESTORE_CHECK='1', PYTHONUNBUFFERED='1')
        self.children = []
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for proc in self.children:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=25)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()

    def launch(self, name, code, kind='--gpu'):
        output = (self.logs/(name+'.log')).open('w')
        self.addCleanup(output.close)
        proc = subprocess.Popen(['bash', str(self.repo/'bench/fleet.sh'), 'run', kind, name, '1', 'fixture', '--',
                                 sys.executable, '-c', code], env=self.env, stdout=output, stderr=subprocess.STDOUT)
        self.children.append(proc)
        return proc

    def until(self, condition):
        deadline = time.monotonic()+12
        while not condition():
            if time.monotonic()>deadline:
                self.fail('\n'.join(p.read_text() for p in self.logs.glob('*.log')))
            time.sleep(.02)

    def held(self, name):
        path = self.logs/'fleet/holder'
        return path.exists() and path.read_text().startswith(name+'|')

    def ready(self, name):
        return handoff.receipt(self.logs/'fleet', name).exists()

    def wait(self, proc):
        return proc.wait(timeout=20)

    def test_two_real_waiters_handoff_with_one_final_restore(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('second', 'pass')
        self.until(lambda:self.ready('second'))
        gate.touch()
        self.assertEqual(self.wait(first), 0)
        self.assertEqual(self.wait(second), 0)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['second'])
        self.assertFalse((self.logs/'fleet/restore-debt.json').exists())
        events = [json.loads(line) for line in (self.logs/'fleet/lifecycle.jsonl').read_text().splitlines()]
        self.assertEqual(sum(r['event']=='handoff-accepted' for r in events), 1)

    def test_receiver_failure_preserves_payload_code_and_restores(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('second', 'raise SystemExit(7)')
        self.until(lambda:self.ready('second')); gate.touch()
        self.assertEqual(self.wait(first), 0); self.assertEqual(self.wait(second), 7)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['second'])

    def test_cancelled_waiter_leaves_donor_to_restore(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('second', 'pass')
        self.until(lambda:self.ready('second')); second.terminate()
        self.assertEqual(self.wait(second), 143)
        gate.touch(); self.assertEqual(self.wait(first), 0)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['first'])

    def test_failed_restore_leaves_visible_debt_and_nonzero_exit(self):
        (self.logs/'fail-restore').touch()
        self.assertEqual(self.wait(self.launch('first', 'pass')), 1)
        self.assertEqual(handoff.read(self.logs/'fleet/restore-debt.json')['owner']['session'], 'first')

    def test_cancel_after_offer_reclaims_and_restores(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('second', 'pass')
        self.until(lambda: 'waiting:' in (self.logs/'second.log').read_text())
        gate.touch()
        events = self.logs/'fleet/lifecycle.jsonl'
        self.until(lambda: 'handoff-offered' in events.read_text())
        second.terminate()
        self.assertEqual(self.wait(second), 143)
        self.assertEqual(self.wait(first), 0)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['first'])
        self.assertIn('handoff-reclaim', events.read_text())

    def test_probe_waits_for_restore_and_active_cancellation_recovers(self):
        first = self.launch('first', 'import time; time.sleep(60)')
        self.until(lambda:self.held('first'))
        # A probe has no boot supervisor; it must observe the restored boundary.
        probe = self.launch('probe', "import os; from pathlib import Path; assert Path(os.environ['LOGD'],'restores').read_text().strip()=='first'", '--probe')
        self.until(lambda:'queued' in (self.logs/'probe.log').read_text() or 'waiting:' in (self.logs/'probe.log').read_text())
        first.terminate()
        self.assertEqual(self.wait(first), 143)
        self.assertEqual(self.wait(probe), 0)

    def test_restore_debt_does_not_allow_probe_to_block_recovery_boot(self):
        (self.logs/'fail-restore').touch()
        self.assertEqual(self.wait(self.launch('failed', 'pass')), 1)
        probe = self.launch('probe', 'pass', '--probe')
        self.until(lambda:'waiting:' in (self.logs/'probe.log').read_text())
        (self.logs/'fail-restore').unlink()
        recovery = self.launch('recovery', 'pass')
        self.assertEqual(self.wait(recovery), 0)
        self.assertEqual(self.wait(probe), 0)

    def test_nested_restore_policy_defers_until_supervisor_finishes(self):
        code = "import os,subprocess; assert subprocess.call(['bash',os.environ['FLEET'],'restore-needed',os.environ['FLEET_SESSION']])==1"
        self.assertEqual(self.wait(self.launch('nested', code)), 0)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['nested'])


if __name__ == '__main__':
    unittest.main()
