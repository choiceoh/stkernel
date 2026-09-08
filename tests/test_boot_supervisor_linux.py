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
            self.skipTest('Linux supervisor integration requires Linux and flock; run on the CPU validation host')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.logs, self.bin = [self.root/p for p in ('repo', 'logs', 'bin')]
        for path in (self.repo/'bench', self.repo/'profiles', self.logs/'fleet', self.bin):
            path.mkdir(parents=True)
        for name in ('fleet.sh', 'fleet_boot.py', 'fleet_handoff.py', 'fleet_priority.py', 'fleet_pin.py', 'fleet_pending.py', 'fleet_inspect.py', 'experiment_metrics.py', 'fleet_launch.py', 'fleet_prepare.py', 'fleet_classify.py'):
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
        second = self.launch('second', "import os,subprocess,hashlib; from pathlib import Path; p=Path(os.environ['FLEET_RUNNER_REPO'],'bench/fleet.sh')\nif hashlib.sha256(p.read_bytes()).hexdigest() not in subprocess.check_output(['bash',os.environ['FLEET'],'version'],text=True): raise RuntimeError('version is not pinned')")
        self.until(lambda:self.ready('second'))
        # A common checkout update must not replace either in-flight controller.
        (self.repo/'bench/fleet_restore.sh').write_text('exit 99\n')
        (self.repo/'bench/fleet.sh').write_text('exit 99\n')
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

    def test_cancel_between_holder_and_debt_transfer_restores(self):
        source = self.repo/'bench/fleet_handoff.py'
        source.write_text(source.read_text().replace(
            "    if managed:\n        claim_held(directory, session, pid)",
            "    if managed:\n        (directory / 'admission-gap').touch()\n        time.sleep(60)\n        claim_held(directory, session, pid)"))
        receiver = self.launch('receiver', 'raise RuntimeError("payload must not start")')
        self.until(lambda:(self.logs/'fleet/admission-gap').exists())
        receiver.terminate()
        self.assertEqual(self.wait(receiver), 143)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['receiver'])
        self.assertFalse((self.logs/'fleet/restore-debt.json').exists())
        self.assertFalse(self.held('receiver'))

    def edit(self, name, *args):
        return subprocess.run(['bash', str(self.repo/'bench/fleet.sh'), 'edit', name, *args],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def test_edit_actual_waiter_executes_only_replacement_in_updated_cwd(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('second', 'raise SystemExit(99)')
        self.until(lambda:self.ready('second'))
        third = self.launch('third', 'pass')
        self.until(lambda:self.ready('third'))
        original = (self.logs/'fleet/queue').read_text().splitlines()
        command = [sys.executable, '-c', "import os,json; from pathlib import Path; Path('accepted.json').write_text(json.dumps(dict(cwd=os.getcwd(),holder=Path(os.environ['FLEET_DIR'],'holder').read_text(),literal=__import__('sys').argv[1:])))", 'a b', '$literal', '한글']
        result = self.edit('second', '--est', '7', '--note', 'updated queued command', '--cwd', str(self.logs), '--', *command)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        current = (self.logs/'fleet/queue').read_text().splitlines()
        self.assertEqual([r.split('|')[:3] for r in current], [r.split('|')[:3] for r in original])
        self.assertEqual(value['revision'], 2)
        # Changing the common controller still cannot change the pinned waiter.
        (self.repo/'bench/fleet.sh').write_text('exit 99\n')
        gate.touch()
        self.assertEqual(self.wait(first), 0); self.assertEqual(self.wait(second), 0)
        self.assertEqual(self.wait(third), 0)
        accepted = json.loads((self.logs/'accepted.json').read_text())
        self.assertEqual(accepted['cwd'], str(self.logs))
        self.assertEqual(accepted['literal'], ['a b', '$literal', '한글'])
        self.assertEqual(accepted['holder'].split('|')[4:6], ['7', 'updated queued command'])
        self.assertEqual(len((self.logs/'restores').read_text().splitlines()), 1)

    def test_failed_edit_keeps_original_then_active_edit_is_refused(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('second', "import os; from pathlib import Path; Path(os.environ['LOGD'],'original-ran').touch()")
        self.until(lambda:self.ready('second'))
        bad = self.logs/'invalid.sh'; bad.write_text('if then\n')
        result = self.edit('second', '--', 'bash', str(bad))
        self.assertEqual(result.returncode, 2)
        self.assertIn('preflight failed', result.stderr)
        self.assertEqual(json.loads(self.edit('second').stdout)['revision'], 1)
        self.assertEqual(self.edit('first', '--note', 'too late').returncode, 2)
        gate.touch(); self.assertEqual(self.wait(first), 0); self.assertEqual(self.wait(second), 0)
        self.assertTrue((self.logs/'original-ran').exists())

    def test_probe_edit_preserves_probe_admission_and_never_restores(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        probe = self.launch('probe', 'raise SystemExit(99)', '--probe')
        from fleet_pending import path
        self.until(lambda:path(self.logs/'fleet', 'probe').exists())
        self.assertEqual(self.edit('probe', '--', sys.executable, '-c', 'pass').returncode, 0)
        gate.touch(); self.assertEqual(self.wait(first), 0); self.assertEqual(self.wait(probe), 0)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['first'])

    def test_go_during_edit_preflight_runs_original_and_rejects_late_commit(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('second', "import os; from pathlib import Path; Path(os.environ['LOGD'],'original-ran').touch()")
        self.until(lambda:self.ready('second'))
        # Pause only the editor's profile fetch; no admission lock is held.
        (self.bin/'git').write_text('#!/bin/sh\ntouch "$LOGD/edit-checking"\nwhile [ ! -e "$LOGD/edit-continue" ]; do sleep .02; done\nexit 1\n')
        with (self.logs/'edit.log').open('w') as output:
            editor = subprocess.Popen(['bash', str(self.repo/'bench/fleet.sh'), 'edit', 'second', '--',
                                       sys.executable, '-c', 'raise SystemExit(99)'], env=self.env,
                                      stdout=output, stderr=subprocess.STDOUT)
            self.children.append(editor)
            self.until(lambda:(self.logs/'edit-checking').exists())
            gate.touch(); self.assertEqual(self.wait(first), 0); self.assertEqual(self.wait(second), 0)
            (self.logs/'edit-continue').touch()
            self.assertEqual(self.wait(editor), 2)
        self.assertTrue((self.logs/'original-ran').exists())
        self.assertIn('no longer queued', (self.logs/'edit.log').read_text())

    def test_damaged_pending_record_cannot_skip_restore_or_leave_hold(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        from fleet_pending import path
        path(self.logs/'fleet', 'first').unlink()
        gate.touch(); self.assertEqual(self.wait(first), 1)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['first'])
        self.assertFalse(self.held('first'))

    def show(self, name):
        result = subprocess.run(['bash', str(self.repo/'bench/fleet.sh'), 'show', name, '--json'],
                                env=self.env, capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_completed_output_and_payload_failure_remain_inspectable(self):
        (self.repo/'bench/fleet_restore.sh').write_text('echo RESTORE_OUTPUT\necho "$FLEET_SESSION" >> "$LOGD/restores"\n')
        worker = self.launch('inspect-failure', 'import sys; print("STDOUT_PAYLOAD"); print("STDERR_PAYLOAD",file=sys.stderr); raise SystemExit(7)')
        self.assertEqual(self.wait(worker), 7)
        result = self.show('inspect-failure')
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['payload_returncode'], 7)
        self.assertEqual(result['returncode'], 7)
        self.assertEqual(result['recovery_returncode'], 0)
        self.assertGreaterEqual(result['finished_at'], result['payload_finished_at'])
        self.assertFalse(result['editable'])
        logs = subprocess.run(['bash', str(self.repo/'bench/fleet.sh'), 'logs', 'inspect-failure', '--tail', '80'],
                              env=self.env, capture_output=True, text=True, timeout=3)
        self.assertEqual(logs.returncode, 0, logs.stderr)
        for marker in ('GO inspect-failure', 'STDOUT_PAYLOAD', 'STDERR_PAYLOAD', 'RESTORE_OUTPUT'):
            self.assertIn(marker, logs.stdout)
            self.assertIn(marker, (self.logs/'inspect-failure.log').read_text())
        self.assertEqual(Path(result['log_path']).stat().st_mode & 0o777, 0o600)

    def test_payload_success_does_not_hide_failed_restore_in_show(self):
        (self.logs/'fail-restore').touch()
        self.assertEqual(self.wait(self.launch('restore-failed', 'pass')), 1)
        result = self.show('restore-failed')
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['payload_returncode'], 0)
        self.assertEqual(result['recovery_returncode'], 1)
        self.assertEqual(result['returncode'], 1)

    def test_log_open_failure_preserves_live_output_and_recovery(self):
        (self.logs/'fleet/run-logs').write_text('fixture collision')
        self.assertEqual(self.wait(self.launch('no-log', 'print("LIVE_FALLBACK")')), 0)
        self.assertIn('LIVE_FALLBACK', (self.logs/'no-log.log').read_text())
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['no-log'])
        self.assertFalse(self.held('no-log'))

    def test_cancelled_waiter_and_reused_session_keep_separate_logs(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('again', 'print("CANCELLED_MUST_NOT_RUN")')
        self.until(lambda:self.ready('again'))
        second.terminate(); self.assertEqual(self.wait(second), 143)
        cancelled = self.show('again')
        self.assertEqual(cancelled['state'], 'cancelled')
        self.assertEqual(cancelled['returncode'], 143)
        self.assertNotIn('payload_returncode', cancelled)
        replacement = self.launch('again', 'print("NEW_TICKET_OUTPUT")')
        self.until(lambda:self.ready('again'))
        gate.touch(); self.assertEqual(self.wait(first), 0); self.assertEqual(self.wait(replacement), 0)
        finished = self.show('again')
        self.assertEqual(finished['state'], 'succeeded')
        self.assertNotEqual(finished['log_path'], cancelled['log_path'])
        self.assertTrue(Path(cancelled['log_path']).is_file())
        self.assertIn('NEW_TICKET_OUTPUT', Path(finished['log_path']).read_text())

    def test_legacy_live_waiter_can_show_existing_stdout_log(self):
        gate = self.logs/'continue'
        first = self.launch('first', f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        second = self.launch('legacy-view', 'pass')
        self.until(lambda:self.ready('legacy-view'))
        from fleet_pending import path
        saved = path(self.logs/'fleet', 'legacy-view'); data = saved.read_bytes(); saved.unlink()
        try:
            result = self.show('legacy-view')
            self.assertEqual(result['state'], 'queued')
            self.assertEqual(result['source'], 'legacy')
            self.assertEqual(result['log_path'], str(self.logs/'legacy-view.log'))
            self.assertFalse(result['editable'])
        finally:
            saved.write_bytes(data)
        gate.touch(); self.assertEqual(self.wait(first), 0); self.assertEqual(self.wait(second), 0)

    def test_unread_client_output_cannot_block_payload_or_recovery(self):
        code='import os; os.write(1,b"x"*(2*1024*1024)); print("RETAINED_END")'
        worker = subprocess.Popen(['bash', str(self.repo/'bench/fleet.sh'), 'run', '--gpu',
                                   'slow-reader', '1', 'fixture', '--', sys.executable, '-c', code],
                                  env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.children.append(worker)
        self.addCleanup(worker.stdout.close)
        # Deliberately never consume the live pipe before process completion.
        self.assertEqual(worker.wait(timeout=12), 0)
        result = self.show('slow-reader')
        self.assertEqual(result['state'], 'succeeded')
        retained = Path(result['log_path']).read_text()
        self.assertGreater(len(retained), 2*1024*1024)
        self.assertIn('RETAINED_END', retained)
        self.assertEqual((self.logs/'restores').read_text().splitlines(), ['slow-reader'])
        self.assertFalse(self.held('slow-reader'))


    def test_detached_cpu_acknowledges_before_completion(self):
        gate=self.logs/'cpu-continue'
        code=f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)\nprint("CPU_DONE")'
        result=subprocess.run(['bash',str(self.repo/'bench/fleet.sh'),'run','--cpu','--detach','cpu-detached','--',
                               sys.executable,'-c',code],env=self.env,cwd=self.repo,capture_output=True,text=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        receipt=json.loads(result.stdout)
        self.assertEqual(receipt['state'],'running-cpu')
        self.assertFalse((self.logs/'fleet/holder').exists())
        gate.touch()
        self.until(lambda:'CPU_DONE' in Path(receipt['startup_log']).read_text())

    def test_detached_gpu_ticket_and_historical_log(self):
        gate=self.logs/'continue'
        first=self.launch('first',f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        command=['bash',str(self.repo/'bench/fleet.sh'),'run','--gpu','--detach','detached','1','fixture','--',
                 sys.executable,'-c','print("DETACHED_DONE")']
        result=subprocess.run(command,env=self.env,cwd=self.repo,capture_output=True,text=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        receipt=json.loads(result.stdout)
        self.assertEqual(receipt['state'],'queued');self.assertTrue(receipt['ticket'])
        again=subprocess.run(command,env=self.env,cwd=self.repo,capture_output=True,text=True,timeout=8)
        self.assertEqual(json.loads(again.stdout)['pid'],receipt['pid'])
        gate.touch();self.assertEqual(self.wait(first),0)
        self.until(lambda:self.show('detached')['state']=='succeeded')
        history=subprocess.run(['bash',str(self.repo/'bench/fleet.sh'),'history','detached','--json'],
                               env=self.env,capture_output=True,text=True,timeout=5)
        self.assertEqual(history.returncode,0,history.stderr)
        self.assertIn(receipt['ticket'],history.stdout)
        logs=subprocess.run(['bash',str(self.repo/'bench/fleet.sh'),'logs','detached','--ticket',receipt['ticket']],
                            env=self.env,capture_output=True,text=True,timeout=5)
        self.assertEqual(logs.returncode,0,logs.stderr);self.assertIn('DETACHED_DONE',logs.stdout)

    def test_preparation_failure_does_not_enqueue_or_restore(self):
        spec=self.repo/'prepare.json';spec.write_text(json.dumps(dict(required_paths=['missing-model'])))
        result=subprocess.run(['bash',str(self.repo/'bench/fleet.sh'),'run','--gpu','--detach','--prepare',str(spec),
                               'not-ready','--',sys.executable,'-c','pass'],env=self.env,cwd=self.repo,
                              capture_output=True,text=True,timeout=8)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('required path is missing',result.stdout+result.stderr)
        self.assertFalse((self.logs/'fleet/holder').exists())
        self.assertEqual((self.logs/'fleet/queue').read_text(),'')
        self.assertFalse((self.logs/'restores').exists())

    def test_same_command_edit_rebinds_changed_source(self):
        gate=self.logs/'continue'
        first=self.launch('first',f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        script=self.repo/'candidate.py';script.write_text('print("OLD")\n')
        output=(self.logs/'rebind.log').open('w');self.addCleanup(output.close)
        second=subprocess.Popen(['bash',str(self.repo/'bench/fleet.sh'),'run','--gpu','rebind','--',
                                 sys.executable,str(script)],env=self.env,cwd=self.repo,stdout=output,stderr=subprocess.STDOUT)
        self.children.append(second);self.until(lambda:'waiting:' in (self.logs/'rebind.log').read_text())
        script.write_text('print("NEW_INPUT")\n')
        result=subprocess.run(['bash',str(self.repo/'bench/fleet.sh'),'edit','rebind','--',sys.executable,str(script)],
                              env=self.env,capture_output=True,text=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(json.loads(result.stdout)['revision'],2)
        gate.touch();self.assertEqual(self.wait(first),0);self.assertEqual(self.wait(second),0)
        self.assertIn('NEW_INPUT',(self.logs/'rebind.log').read_text())

    def test_source_change_while_queued_is_rejected_before_go(self):
        gate=self.logs/'continue'
        first=self.launch('first',f'from pathlib import Path; import time\nwhile not Path({str(gate)!r}).exists(): time.sleep(.02)')
        self.until(lambda:self.held('first'))
        script=self.repo/'candidate.py';script.write_text('print("ORIGINAL")\n')
        output=(self.logs/'source-change.log').open('w');self.addCleanup(output.close)
        second=subprocess.Popen(['bash',str(self.repo/'bench/fleet.sh'),'run','--gpu','source-change','--',
                                 sys.executable,str(script)],env=self.env,cwd=self.repo,stdout=output,stderr=subprocess.STDOUT)
        self.children.append(second)
        self.until(lambda:self.ready('source-change'))
        script.write_text('raise RuntimeError("must never run")\n')
        gate.touch()
        self.assertEqual(self.wait(first),0);self.assertEqual(self.wait(second),3)
        value=self.show('source-change')
        self.assertIsNone(value.get('payload_returncode'))
        self.assertEqual((self.logs/'restores').read_text().splitlines(),['first'])
        self.assertIn('queued input changed',(self.logs/'source-change.log').read_text())


if __name__ == '__main__':
    unittest.main()
