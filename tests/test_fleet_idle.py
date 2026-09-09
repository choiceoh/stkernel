"""Bounded idle recovery policy tests; no GPUs, network, or real waiting."""
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import fleet_entry
import fleet_idle as idle
import fleet_pause


class IdleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        (self.directory / 'queue').touch()
        self.now = 1000.0
        self.processes = {os.getpid(): (1, 'controller-start'), 42: (1, 'queued-start')}
        self.patch(idle, 'clock', side_effect=lambda: self.now)
        self.patch(idle, 'boot_id', return_value='boot-a')
        self.patch(idle, 'process', side_effect=lambda pid: self.processes.get(pid))
        self.observe = self.patch(idle, 'observe', return_value=dict(stopped=True, traffic='zero'))
        self.recovery = self.patch(idle, 'recovery', return_value={
            'repo': str(self.directory), 'receipt': str(self.directory / 'approved.json')})
        self.restore = self.patch(idle, 'restore', return_value=0)
        self.patch(fleet_entry, 'inspect', return_value=None)
        self.production = self.patch(fleet_entry, 'production_current', return_value=False)
        self.paused = self.patch(fleet_pause, 'paused', return_value=False)
        idle.activity(self.directory, 'released')

    def patch(self, obj, name, **kwargs):
        patcher = patch.object(obj, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def state(self):
        return idle.read(self.directory / 'idle-recovery.json')

    def queue(self, kind='boot', pid='42', session='next'):
        (self.directory / 'queue').write_text(
            f'ticket|{session}|1000|1|fixture|{kind}|{pid}\n')

    def holder(self, session='held', pid=None, host=None):
        pid = os.getpid() if pid is None else pid
        host = socket.gethostname().split('.')[0] if host is None else host
        (self.directory / 'holder').write_text(f'{session}|{pid}|{host}|1000|1|fixture|boot\n')

    def elapsed(self, seconds):
        self.now = 1000.0 + seconds
        return idle.tick(self.directory)

    def test_exact_five_minute_boundary_and_no_repeat_before_next_quiet_window(self):
        self.assertEqual(self.elapsed(299.99)['phase'], 'waiting')
        self.restore.assert_not_called()
        self.recovery.assert_not_called()
        result = self.elapsed(300)
        self.assertEqual(result['phase'], 'healthy')
        self.restore.assert_called_once()
        self.assertFalse((self.directory / 'holder').exists())
        self.assertFalse((self.directory / 'idle-recovery-owner.json').exists())
        self.elapsed(599.99)
        self.restore.assert_called_once()
        self.production.return_value = True
        self.assertEqual(self.elapsed(600)['phase'], 'healthy')
        self.restore.assert_called_once()
        self.elapsed(615)
        self.assertEqual(self.recovery.call_count, 2)

    def test_enqueue_generation_change_during_observation_defers(self):
        def observe():
            idle.activity(self.directory, 'enqueued then cancelled')
            return dict(stopped=True, traffic='zero')
        self.observe.side_effect = observe
        result = self.elapsed(300)
        self.assertEqual(result['reason'], 'enqueued then cancelled')
        self.assertEqual(result['since'], self.now)
        self.restore.assert_not_called()
        self.recovery.assert_not_called()

    def test_live_and_unknown_remote_holders_reset_idle_age(self):
        for host in (socket.gethostname().split('.')[0], 'unrecognized-remote'):
            with self.subTest(host=host):
                self.holder(host=host)
                result = self.elapsed(300)
                self.assertEqual(result['reason'], 'holder active')
                self.assertEqual(result['since'], self.now)
        self.observe.assert_not_called()
        self.restore.assert_not_called()

    def test_dead_holder_does_not_prevent_central_recovery(self):
        self.holder(pid=43)
        self.assertEqual(self.elapsed(300)['phase'], 'healthy')
        self.restore.assert_called_once()

    def test_live_boot_queue_resets_idle_age(self):
        self.queue()
        result = self.elapsed(300)
        self.assertEqual(result['reason'], 'runnable work')
        self.assertEqual(result['since'], self.now)
        self.restore.assert_not_called()

    def test_paused_and_dead_queue_rows_do_not_block_recovery(self):
        for paused, pid in ((True, '42'), (False, '43')):
            with self.subTest(paused=paused, pid=pid):
                self.now = 1000
                idle.activity(self.directory, 'released')
                self.queue(pid=pid)
                self.paused.return_value = paused
                self.restore.reset_mock()
                self.assertEqual(self.elapsed(300)['phase'], 'healthy')
                self.restore.assert_called_once()

    def test_probe_waiting_for_absent_serving_allows_recovery(self):
        self.queue(kind='probe')
        self.assertEqual(self.elapsed(300)['phase'], 'healthy')
        self.restore.assert_called_once()

    def test_probe_with_existing_serving_keeps_its_turn(self):
        self.queue(kind='probe')
        self.observe.return_value = dict(stopped=False, traffic='zero')
        self.assertEqual(self.elapsed(300)['reason'], 'runnable work')
        self.restore.assert_not_called()

    def test_unknown_runtime_resets_full_idle_window(self):
        self.observe.side_effect = ValueError('Docker unavailable')
        result = self.elapsed(300)
        self.assertTrue(result['reason'].startswith('not idle:'))
        self.assertEqual(result['since'], self.now)
        self.observe.side_effect = None
        self.elapsed(599.99)
        self.restore.assert_not_called()
        self.elapsed(600)
        self.restore.assert_called_once()

    def test_completed_serving_traffic_between_ticks_resets_idle_age(self):
        self.observe.return_value = dict(stopped=False, traffic='one')
        self.elapsed(0)
        self.observe.return_value = dict(stopped=False, traffic='two')
        result = self.elapsed(300)
        self.assertEqual(result['reason'], 'serving traffic')
        self.assertEqual(result['since'], self.now)
        self.restore.assert_not_called()

    def test_new_work_during_recovery_verification_defers(self):
        approved = self.recovery.return_value
        def recovery(directory):
            self.queue()
            idle.activity(directory, 'new boot queued')
            return approved
        self.recovery.side_effect = recovery
        self.assertEqual(self.elapsed(300)['phase'], 'waiting')
        self.restore.assert_not_called()
        self.assertFalse((self.directory / 'idle-recovery-owner.json').exists())

    def test_completed_request_during_verification_defers_recovery(self):
        self.observe.side_effect = [dict(stopped=False, traffic='one'),
                                   dict(stopped=False, traffic='two')]
        result = self.elapsed(300)
        self.assertEqual(result['phase'], 'waiting')
        self.assertEqual(result['since'], self.now)
        self.restore.assert_not_called()

    def test_node_becoming_unobservable_before_recovery_restarts_idle_window(self):
        self.observe.side_effect = [dict(stopped=True, traffic='zero'),
                                   ValueError('GPU observation failed on worker')]
        result = self.elapsed(300)
        self.assertEqual(result['phase'], 'waiting')
        self.assertEqual(result['since'], self.now)
        self.assertIn('GPU observation failed', result['reason'])
        self.restore.assert_not_called()
        self.assertFalse((self.directory / 'holder').exists())
        self.assertFalse((self.directory / 'idle-recovery-owner.json').exists())

    def test_healthy_production_does_not_reboot(self):
        self.production.return_value = True
        self.assertEqual(self.elapsed(300)['phase'], 'healthy')
        self.restore.assert_not_called()

    def test_healthy_check_does_not_overwrite_new_activity(self):
        def healthy(repo, container):
            idle.activity(self.directory, 'new queue activity')
            return True
        self.production.side_effect = healthy
        result = self.elapsed(300)
        self.assertEqual(result['phase'], 'waiting')
        self.assertEqual(result['reason'], 'new queue activity')
        self.restore.assert_not_called()

    def test_failed_recovery_releases_owner_and_waits_five_minutes_before_retry(self):
        self.restore.return_value = 7
        result = self.elapsed(300)
        self.assertEqual((result['phase'], result['returncode']), ('retry', 7))
        self.assertFalse((self.directory / 'holder').exists())
        self.assertFalse((self.directory / 'idle-recovery-owner.json').exists())
        self.elapsed(599.99)
        self.restore.assert_called_once()
        self.restore.return_value = 0
        self.assertEqual(self.elapsed(600)['phase'], 'healthy')
        self.assertEqual(self.restore.call_count, 2)

    def test_restore_exception_cleans_up_and_resets_idle_window(self):
        self.restore.side_effect = subprocess.TimeoutExpired('fixture', 1)
        result = self.elapsed(300)
        self.assertTrue(result['reason'].startswith('idle observation failed:'))
        self.assertFalse((self.directory / 'holder').exists())
        self.assertFalse((self.directory / 'idle-recovery-owner.json').exists())
        self.assertEqual(self.state()['phase'], 'waiting')

    def test_missing_approved_receipt_defers_without_restore(self):
        self.recovery.side_effect = ValueError('no approved recovery receipt')
        self.assertTrue(self.elapsed(300)['reason'].startswith('recovery deferred:'))
        self.restore.assert_not_called()
        self.assertFalse((self.directory / 'holder').exists())

    def test_reboot_and_monotonic_rollback_restart_observation(self):
        for change in ({'boot_id': 'older-boot'}, {'since': 5000}):
            with self.subTest(change=change):
                state = self.state()
                state.update(change)
                idle.write(self.directory / 'idle-recovery.json', state)
                result = self.elapsed(300)
                self.assertEqual(result['since'], self.now)
                self.assertEqual(result['idle_seconds'], 0)
        self.restore.assert_not_called()

    def test_authorization_requires_live_exact_owner_and_descendant(self):
        owner = 88
        self.processes.update({owner: (1, 'owner-start'), os.getpid(): (owner, 'controller-start')})
        self.holder(session='idle-fixture', pid=owner)
        lease = dict(session='idle-fixture', pid=owner, start='owner-start', boot_id='boot-a')
        path = self.directory / 'idle-recovery-owner.json'
        idle.write(path, lease)
        self.assertEqual(idle.authorize(self.directory, 'idle-fixture'), lease)
        for key, value in (('start', 'reused-pid'), ('boot_id', 'previous-boot'),
                           ('session', 'other-session')):
            with self.subTest(key=key):
                idle.write(path, dict(lease, **{key: value}))
                with self.assertRaises(ValueError):
                    idle.authorize(self.directory, 'idle-fixture')
        idle.write(path, lease)
        self.processes[os.getpid()] = (1, 'unrelated')
        with self.assertRaises(ValueError):
            idle.authorize(self.directory, 'idle-fixture')
        self.processes[os.getpid()] = (owner, 'controller-start')
        self.holder(session='other-holder', pid=owner)
        with self.assertRaises(ValueError):
            idle.authorize(self.directory, 'idle-fixture')

    def test_environment_flags_cannot_grant_recovery_authority(self):
        self.holder(session='session')
        with patch.dict(os.environ, FLEET_BOOT_INTENT='recovery', FLEET_RESTORE_MANAGED='1',
                        FLEET_AUTO_RESTORE='1', FLEET_SESSION='session'):
            with self.assertRaises(ValueError):
                idle.authorize(self.directory, 'session')

    def test_experiment_boot_requires_boot_hold_and_descendant(self):
        self.holder(session='experiment')
        with patch.dict(os.environ, clear=True):
            self.assertEqual(idle.boot_authorize(self.directory, 'experiment')['intent'], 'experiment')
            self.holder(session='other')
            with self.assertRaises(ValueError):
                idle.boot_authorize(self.directory, 'experiment')
            self.holder(session='experiment', pid=42)
            with self.assertRaises(ValueError):
                idle.boot_authorize(self.directory, 'experiment')
            self.holder(session='experiment')
            holder = self.directory / 'holder'
            holder.write_text(holder.read_text().replace('|boot\n', '|probe\n'))
            with self.assertRaises(ValueError):
                idle.boot_authorize(self.directory, 'experiment')

    def test_legacy_bridge_only_defers_owned_session_without_restore(self):
        self.holder(session='legacy')
        result = idle.legacy_defer(self.directory, 'legacy')
        self.assertTrue(result['deferred'])
        self.assertTrue((self.directory/'holder').exists())
        self.assertIn('restore-deferred', (self.directory/'lifecycle.jsonl').read_text())
        self.restore.assert_not_called()
        with self.assertRaises(ValueError):
            idle.legacy_defer(self.directory, 'other')

    def test_forged_recovery_intent_cannot_upgrade_experiment_hold(self):
        self.holder(session='experiment')
        with patch.dict(os.environ, FLEET_BOOT_INTENT='recovery'):
            with self.assertRaises(ValueError):
                idle.boot_authorize(self.directory, 'experiment')


class ObservationTests(unittest.TestCase):
    def test_unknown_node_blocks_idle_observation(self):
        with patch.object(fleet_entry, 'inspect', return_value=None), \
                patch.object(fleet_entry, 'idle', return_value='stopped'), \
                patch.object(idle, 'legacy_work', return_value=False), \
                patch.object(idle, 'node_idle', side_effect=ValueError('GPU observation failed')):
            with self.assertRaisesRegex(ValueError, 'GPU observation failed'):
                idle.observe()

    def test_live_glm_requests_block_before_gpu_observation(self):
        with patch.object(fleet_entry, 'inspect', return_value={}), \
                patch.object(fleet_entry, 'idle', side_effect=ValueError('live request counters')), \
                patch.object(idle, 'node_idle') as nodes:
            with self.assertRaisesRegex(ValueError, 'live request'):
                idle.observe()
            nodes.assert_not_called()

    def test_unmanaged_boot_blocks_even_with_stopped_serving(self):
        with patch.object(fleet_entry, 'inspect', return_value=None), \
                patch.object(fleet_entry, 'idle', return_value='stopped'), \
                patch.object(idle, 'legacy_work', return_value=True), \
                patch.object(idle, 'node_idle') as nodes:
            with self.assertRaisesRegex(ValueError, 'unmanaged'):
                idle.observe()
            nodes.assert_not_called()

    def test_cumulative_counter_fingerprint_catches_completed_requests(self):
        one = 'vllm:request_success_total{finished_reason="stop"} 1\nvllm:e2e_request_latency_seconds_count{} 1\n'
        two = one.replace(' 1\n', ' 2\n')
        with patch.object(fleet_entry, 'inspect', return_value={}), \
                patch.object(fleet_entry, 'idle', side_effect=[one, two, '\n'.join(reversed(one.splitlines()))]), \
                patch.object(idle, 'legacy_work', return_value=False), \
                patch.object(idle, 'node_idle'):
            before, after, reordered = idle.observe(), idle.observe(), idle.observe()
        self.assertFalse(before['stopped'])
        self.assertNotEqual(before['traffic'], after['traffic'])
        self.assertEqual(before['traffic'], reordered['traffic'])

    def test_remote_gpu_errors_cannot_be_reported_as_idle(self):
        results = [subprocess.CompletedProcess([], 1, '{"idle": true}', 'denied')]
        results.extend(subprocess.CompletedProcess([], 0, value, '') for value in
                       ('', 'not-json', 'null', '[]', '{}', '{"idle": false}',
                        '{"idle": "true"}', '{"idle": 1}'))
        for result in results:
            with self.subTest(result=result), patch.object(idle.subprocess, 'run', return_value=result):
                with self.assertRaises(ValueError):
                    idle.node_idle('fixture')

    def test_remote_probe_timeout_or_missing_transport_never_becomes_idle(self):
        for error in (subprocess.TimeoutExpired('ssh', 12), FileNotFoundError('ssh')):
            with self.subTest(error=error), \
                    patch.object(idle.subprocess, 'run', side_effect=error) as run:
                with self.assertRaises(type(error)):
                    idle.node_idle('10.10.10.4')
                self.assertEqual(run.call_count, 1)


class LocalHeadObservationTests(unittest.TestCase):
    def address_output(self, address='10.10.10.2'):
        return json.dumps([{'addr_info': [{'family': 'inet', 'local': address}]}])

    def test_canonical_head_uses_same_check_locally_only_with_owned_address(self):
        result = subprocess.CompletedProcess([], 0, '{"idle": true}', '')
        with patch.object(idle.subprocess, 'check_output', return_value=self.address_output()) as addresses, \
                patch.object(idle.subprocess, 'run', return_value=result) as run:
            idle.node_idle('10.10.10.2')
        addresses.assert_called_once_with(
            ['ip', '-j', '-4', 'address', 'show'], text=True, timeout=4)
        self.assertEqual(run.call_args.args[0][:3], [sys.executable, '-B', '-c'])
        self.assertEqual(run.call_args.kwargs['timeout'], 12)

    def test_worker_transport_and_head_on_nonhead_host_stay_remote(self):
        result = subprocess.CompletedProcess([], 0, '{"idle": true}', '')
        for target in ('10.10.10.1', '10.10.10.3', '10.10.10.4', '10.10.10.2'):
            with self.subTest(target=target), \
                    patch.object(idle.subprocess, 'check_output', return_value=self.address_output('10.10.10.1')) as addresses, \
                    patch.object(idle.subprocess, 'run', return_value=result) as run:
                idle.node_idle(target)
                self.assertEqual(run.call_args.args[0][0], 'ssh')
                self.assertIn('choiceoh@' + target, run.call_args.args[0])
                self.assertEqual(addresses.call_count, int(target == '10.10.10.2'))

    def test_unknown_address_inventory_fails_before_ownership_probe(self):
        values = ('not-json', '{}', '[null]', '[{}]', '[{"addr_info": [null]}]')
        for value in values:
            with self.subTest(value=value), \
                    patch.object(idle.subprocess, 'check_output', return_value=value), \
                    patch.object(idle.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    idle.node_idle('10.10.10.2')
                run.assert_not_called()
        for error in (FileNotFoundError('ip absent'), subprocess.TimeoutExpired('ip', 4)):
            with self.subTest(error=error), \
                    patch.object(idle.subprocess, 'check_output', side_effect=error), \
                    patch.object(idle.subprocess, 'run') as run:
                with self.assertRaises(type(error)):
                    idle.node_idle('10.10.10.2')
                run.assert_not_called()

    def test_local_failures_never_retry_ssh_or_become_idle(self):
        results = (subprocess.CompletedProcess([], 1, '', 'error'),
                   subprocess.CompletedProcess([], 0, '{"idle": false}', ''),
                   subprocess.CompletedProcess([], 0, '{"idle": "true"}', ''),
                   subprocess.CompletedProcess([], 0, 'not-json', ''))
        for result in results:
            with self.subTest(result=result), \
                    patch.object(idle, '_local_head', return_value=True), \
                    patch.object(idle.subprocess, 'run', return_value=result) as run:
                with self.assertRaises(ValueError):
                    idle.node_idle('10.10.10.2')
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][0], sys.executable)
        with patch.object(idle, '_local_head', return_value=True), \
                patch.object(idle.subprocess, 'run', side_effect=subprocess.TimeoutExpired('python', 12)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                idle.node_idle('10.10.10.2')
            self.assertEqual(run.call_count, 1)

    def test_local_and_remote_execute_identical_observation_program(self):
        import shlex
        result = subprocess.CompletedProcess([], 0, '{"idle": true}', '')
        with patch.object(idle, '_local_head', side_effect=[True, False]), \
                patch.object(idle.subprocess, 'run', return_value=result) as run:
            idle.node_idle('10.10.10.2')
            idle.node_idle('10.10.10.1')
        local, remote = [call.args[0] for call in run.call_args_list]
        self.assertEqual(local[-1], shlex.split(remote[-1])[-1])

    def observation_program(self):
        result = subprocess.CompletedProcess([], 0, '{"idle": true}', '')
        with patch.object(idle, '_local_head', return_value=True), \
                patch.object(idle.subprocess, 'run', return_value=result) as run:
            idle.node_idle('10.10.10.2')
        return run.call_args.args[0][-1]

    def execute_observation_program(self, program, gpu_pids='', error=None):
        import contextlib
        import io
        import types
        def output(args, **kwargs):
            self.assertEqual(args, ['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'])
            if error is not None:
                raise error
            return gpu_pids
        stream = io.StringIO()
        query = Mock(side_effect=output)
        fake = types.SimpleNamespace(check_output=query, DEVNULL=-3)
        with patch.dict(sys.modules, subprocess=fake), contextlib.redirect_stdout(stream):
            exec(program, {})
        query.assert_called_once()
        return json.loads(stream.getvalue())

    def test_actual_program_allows_resident_foreign_pids_without_ownership_lookup(self):
        program = self.observation_program()
        # These PIDs need not belong to a GLM container. Residency alone does
        # not mean fleet work or GLM traffic, nor does it attest boot capacity.
        for gpu_pids in ('', '11\n12', '11\n99', '99', ' 99 \n\n 11 '):
            with self.subTest(gpu_pids=gpu_pids):
                self.assertIs(self.execute_observation_program(program, gpu_pids)['idle'], True)

    def test_actual_program_rejects_malformed_driver_pid_rows(self):
        program = self.observation_program()
        for gpu_pids in ('N/A', 'No running processes found', '-1', '0', '11\nunknown',
                         '11, 256', '11.0', '1 2', '１２'):
            with self.subTest(gpu_pids=gpu_pids):
                self.assertIs(self.execute_observation_program(program, gpu_pids)['idle'], False)

    def test_actual_program_driver_failure_does_not_emit_success(self):
        program = self.observation_program()
        for error in (subprocess.CalledProcessError(1, 'nvidia-smi'),
                      FileNotFoundError('nvidia-smi')):
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.execute_observation_program(program, error=error)


class RestoreProcessTests(unittest.TestCase):
    def test_timeout_stops_process_group_before_return(self):
        import fleet_validation
        with tempfile.TemporaryDirectory() as temporary:
            child = Mock(pid=900001)
            child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 1),
                                      subprocess.TimeoutExpired('fixture', 1), -9]
            child.poll.return_value = None
            with patch.object(idle.subprocess, 'Popen', return_value=child) as launch, \
                    patch.object(idle.os, 'killpg') as killpg, \
                    patch.object(fleet_validation, 'environment', return_value={}):
                with self.assertRaises(subprocess.TimeoutExpired):
                    idle.restore(Path(temporary), '/approved.json', 'idle-fixture')
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            self.assertEqual(killpg.call_args_list[0].args, (child.pid, signal.SIGTERM))
            self.assertEqual(killpg.call_args_list[-1].args, (child.pid, signal.SIGKILL))
            self.assertEqual(child.wait.call_count, 3)


if __name__ == '__main__':
    unittest.main()
