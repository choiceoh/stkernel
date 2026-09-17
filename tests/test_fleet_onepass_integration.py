#!/usr/bin/env python3
"""Real fleet entrypoint refusal, with inert preparation and payload sentinels."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import fleet_handoff as handoff
import fleet_onepass as policy
import fleet_pending as pending
import fleet_pin

BASH = shutil.which('bash')


@unittest.skipUnless(BASH, 'bash is required')
class OnepassIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        self.logs = self.root / 'logs'
        self.directory = self.logs / 'fleet'
        self.directory.mkdir(parents=True)
        (self.repo / 'bench').mkdir(parents=True)
        for name in ('fleet.sh', 'fleet_onepass.py', 'fleet_prepare.py', 'fleet_prepared.py',
                     'fleet_classify.py', 'onepass.py', 'measurement_contract.py',
                     'fleet_handoff.py', 'fleet_pending.py', 'fleet_idle.py', 'st_bracket.sh',
                     'st_screen.py', 'st_judge.py', 'onepass_recording.py', 'onepass_quality.py'):
            shutil.copyfile(ROOT / 'bench' / name, self.repo / 'bench' / name)
        for relative in ('launchers/st_release.py', 'engine/base/latency_trace.py'):
            destination = self.repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        self.prepared = self.root / 'preparation-started'
        self.executed = self.root / 'payload-started'
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        # Refusal regressions remain CPU-only: stop at preparation before any
        # GPU command could be launched, while recording that early rejection
        # failed. Imports of fleet_prepare still use its real source.
        shim = self.bin / 'python3'
        shim.write_text('#!' + sys.executable + '\n'
                        'import json, os, pathlib, sys\n'
                        'if len(sys.argv)>1 and pathlib.Path(sys.argv[1]).name=="fleet_prepare.py":\n'
                        '    pathlib.Path(os.environ["PREPARATION_SENTINEL"]).write_text(json.dumps({k:os.environ[k] for k in ("ST_LEASE_OWNER", "ST_LEASE_PATH") if k in os.environ}))\n'
                        '    raise SystemExit(79)\n'
                        'os.execv(' + repr(sys.executable) + ', [' + repr(sys.executable) + ', *sys.argv[1:]])\n')
        shim.chmod(0o700)
        self.environment = {'PATH': str(self.bin) + os.pathsep + os.environ.get('PATH', ''),
                            'HOME': str(self.root), 'REPO': str(self.repo), 'LOGD': str(self.logs),
                            'FLEET_DIR': str(self.directory), 'PREPARATION_SENTINEL': str(self.prepared)}
        self.preparation_spec = self.root / 'prepare.json'
        self.preparation_spec.write_text(json.dumps({'cpu_command': [sys.executable, '-c',
            'from pathlib import Path; Path(' + repr(str(self.prepared)) + ').write_text("cpu preparation ran")']}))

    def run_fleet(self, *args, **environment):
        return subprocess.run([BASH, str(self.repo / 'bench/fleet.sh'), *args], cwd=self.repo,
                              env=dict(self.environment, **environment), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)

    def assert_no_work(self):
        self.assertFalse(self.prepared.exists(), 'GPU policy must reject before fleet_prepare starts')
        self.assertFalse(self.executed.exists(), 'rejected payload must never execute')
        queue = self.directory / 'queue'
        self.assertTrue(not queue.exists() or not queue.read_text(), 'rejected command reserved the fleet')
        self.assertFalse((self.directory / 'holder').exists())
        self.assertFalse((self.directory / 'pending').exists())

    def test_arbitrary_gpu_command_is_rejected_before_cpu_preparation(self):
        # The marker is harmless even if invoked; the text also identifies
        # this as GPU work to the existing CPU/GPU classifier.
        code = 'from pathlib import Path; Path(' + repr(str(self.executed)) + ').write_text("ran") # torch.cuda'
        result = self.run_fleet('run', '--gpu', '--prepare', str(self.preparation_spec),
                                'custom', '1', 'fixture', '--', 'python3', '-c', code)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('onepass-only', result.stdout)
        self.assert_no_work()

    def test_retired_wrapper_is_not_resurrected_by_a_familiar_name(self):
        result = self.run_fleet('run', '--gpu', '--prepare', str(self.preparation_spec),
                                'chain-custom', '1', 'fixture', '--', 'bash', 'bench/chain.sh',
                                'A=', 'touch ' + str(self.executed))
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('onepass-only', result.stdout)
        self.assert_no_work()

    def test_boot_binds_the_waiters_lease_before_signing_preparation(self):
        for incoming in ({}, {'ST_LEASE_OWNER':'stale-owner', 'ST_LEASE_PATH':'/stale',
                              'FLEET_LEASE_PATH':'/intended/lease'}):
            with self.subTest(incoming=incoming):
                result = self.run_fleet('run', '--gpu', '--fleet', 'lease-fixture', '1',
                                        'fixture', '--', 'python3', 'bench/onepass.py', **incoming)
                self.assertEqual(result.returncode, 3, result.stdout)
                self.assertEqual(json.loads(self.prepared.read_text()), {
                    'ST_LEASE_OWNER':'queue/lease-fixture',
                    'ST_LEASE_PATH':incoming.get('FLEET_LEASE_PATH', '/home/choiceoh/glm53-logs/st-fleet.lock')})
                self.assertFalse(self.executed.exists())
                self.assertFalse((self.directory / 'holder').exists())

    def test_cpu_preparation_does_not_receive_a_queue_lease(self):
        result = self.run_fleet('run', '--cpu', 'cpu-fixture', '1', 'fixture', '--',
                                sys.executable, '-c', 'pass')
        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertEqual(json.loads(self.prepared.read_text()), {})
        self.assertFalse(self.executed.exists())

    def test_bare_reservation_wait_and_adoption_cannot_skip_payload_policy(self):
        for args in (('request', 'bare'), ('wait', 'bare', '1'),
                     ('adopt', 'bare', str(os.getpid()))):
            with self.subTest(args=args):
                result = self.run_fleet(*args)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn('disabled', result.stdout)
                self.assert_no_work()

    def test_rehearsal_does_not_reclassify_arbitrary_gpu_payload_as_cpu(self):
        code = 'from pathlib import Path; Path(' + repr(str(self.executed)) + ').write_text("ran") # torch.cuda'
        result = self.run_fleet('run', '--cpu', '--prepare', str(self.preparation_spec),
                                'false-rehearsal', '1', 'fixture', '--', 'python3', '-c', code,
                                FLEET_REHEARSE='1')
        self.assertEqual(result.returncode, 5, result.stdout)
        self.assertIn('you said --cpu but the job shows GPU use', result.stdout)
        self.assert_no_work()

    def test_pinned_controller_preserves_the_bracket_and_rejects_mutations(self):
        runner = fleet_pin.pin(self.repo, self.directory)
        relative = 'bench/st_bracket.sh'
        self.assertEqual((runner / relative).read_bytes(), (self.repo / relative).read_bytes())
        command = ['bash', relative, 'pair', '0123456789abcdef']
        policy.validate(command, self.repo, runner, self.environment)
        original = (self.repo / relative).read_bytes()
        (self.repo / relative).write_bytes(original + b'\n# an unreviewed additional workload\n')
        with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
            policy.validate(command, self.repo, runner, self.environment)
        (self.repo / relative).write_bytes(original)
        (runner / relative).write_bytes(original + b'\n# corrupted pin\n')
        with self.assertRaisesRegex(ValueError, 'pinned runner integrity'):
            fleet_pin.pin(self.repo, self.directory)

    def test_pinned_controller_can_read_lease_without_the_source_checkout(self):
        # The controller resolves both helpers from its frozen runner. Omitting
        # them makes even an empty fleet report "lease unreadable" forever.
        for relative in ('launchers/lib/fleet-lease.sh', 'engine/base/fleet_lease.py'):
            destination = self.repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        runner = fleet_pin.pin(self.repo, self.directory)
        shutil.rmtree(self.repo / 'launchers')
        shutil.rmtree(self.repo / 'engine')
        result = subprocess.run([BASH, '-c',
            '. "$FLEET_REPO/launchers/lib/fleet-lease.sh"; fleet_lease read'],
            env={**os.environ, 'FLEET_REPO': str(runner),
                 'FLEET_HEAD': subprocess.check_output(['hostname', '-s'], text=True).strip(),
                 'FLEET_LEASE_PATH': str(self.root / 'missing-lease')},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.strip(), 'free')

    def test_lease_helper_preserves_arguments_locally_and_over_ssh(self):
        # Run the real helper with inert Python/SSH transports. Shell metacharacters
        # are data; the local path and the remote shell must deliver identical argv.
        (self.bin / 'python3').write_text('#!' + sys.executable + '\n'
            'import json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        ssh = self.bin / 'ssh'
        ssh.write_text('#!' + sys.executable + '\n'
            'import subprocess, sys\n'
            'raise SystemExit(subprocess.call(["/bin/bash", "-c", sys.argv[-1]]))\n')
        ssh.chmod(0o755)
        owner = "codex/it's a task"
        note = 'spaces and $(printf expanded) and `printf expanded` stay literal'
        lease_path = str(self.root / 'lease path')
        local = subprocess.check_output(['hostname', '-s'], text=True).strip()
        for head in (local, 'remote-test-head'):
            result = subprocess.run([BASH, '-c',
                '. "$FLEET_REPO/launchers/lib/fleet-lease.sh"; '
                'fleet_lease yield --requester "$TEST_OWNER" --note "$TEST_NOTE"'],
                env={**os.environ, 'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
                     'FLEET_REPO': str(ROOT), 'FLEET_HEAD': head, 'FLEET_LEASE_PATH': lease_path,
                     'TEST_OWNER': owner, 'TEST_NOTE': note}, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected_module = str(ROOT / 'engine/base/fleet_lease.py') if head == local else '-'
            self.assertEqual(json.loads(result.stdout),
                [expected_module, 'yield', '--requester', owner, '--note', note, '--path', lease_path])

    def test_lease_heartbeat_returns_its_pid_before_the_first_renewal(self):
        import signal
        process = subprocess.Popen([BASH, '-c',
            '. "$FLEET_REPO/launchers/lib/fleet-lease.sh"; '
            'beat=$(fleet_lease_beat fixture); echo "started:$beat"; kill "$beat"'],
            env={**os.environ, 'FLEET_REPO': str(ROOT)}, start_new_session=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertRegex(stdout.strip(), r'^started:[0-9]+$')
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)

    def test_failed_real_preflight_edit_preserves_ticket_command_and_order(self):
        runner = fleet_pin.pin(self.repo, self.directory)
        pid = os.getpid()
        queue = self.directory / 'queue'
        queue.write_text(f'10|before|100|1|neighbor|boot|{pid}\n'
                         f'20|mine|101|2|original|boot|{pid}\n'
                         f'30|after|102|1|neighbor|boot|{pid}\n')
        original_command = ['bash', str(self.repo / 'bench/st_bracket.sh'), 'pair', '0123456789abcdef']
        with patch.object(handoff, 'identity', return_value='fixture-start'), \
                patch.dict(os.environ, self.environment, clear=True):
            with patch.object(pending.os, 'getcwd', return_value=str(self.repo)):
                original = pending.register(self.directory, 'mine', original_command,
                                            str(runner / 'bench/fleet.sh'), 'boot')
            before = queue.read_bytes()
            code = 'from pathlib import Path; Path(' + repr(str(self.executed)) + ').write_text("ran")'
            with self.assertRaisesRegex(ValueError, 'replacement preflight failed; original reservation retained'):
                pending.edit(self.directory, 'mine', command=['python3', '-c', code], expected=1)
            saved = pending.read_record(self.directory, 'mine')
        self.assertEqual(saved, original)
        self.assertEqual(saved['ticket'], '20')
        self.assertEqual(saved['command'], original_command)
        self.assertEqual(queue.read_bytes(), before)
        self.assertFalse(self.prepared.exists())
        self.assertFalse(self.executed.exists())


if __name__ == '__main__':
    unittest.main()
