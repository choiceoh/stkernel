import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('offline', Path(__file__).resolve().parents[1]/'probes/glm53_offline_checks.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class OfflineTests(unittest.TestCase):
    def states(self):
        return {n: dict(id=n, running=True, auto_remove=False, image=m.IMAGE,
                        overlays={'file':'hash'}, manifest='hash', port=8000) for n in m.NODES}

    def test_no_partial_or_automatically_deleted_or_wrong_image_fleet(self):
        self.assertEqual(m.validate_before(self.states()), 'present')
        self.assertEqual(m.validate_before(dict.fromkeys(m.NODES)), 'absent')
        for delta in [None, {'auto_remove': True}, {'running': False}, {'image': 'other'}]:
            state = self.states()
            state['local'] = None if delta is None else dict(state['local'], **delta)
            with self.subTest(delta=delta), self.assertRaises(RuntimeError):
                m.validate_before(state)

    def test_probe_error_restores_before_propagating(self):
        events = []
        def move(before, action): events.append(action); return {}
        def run(): events.append('probe'); raise ValueError('numerics fail')
        with patch.object(m, 'transition_all', move), patch.object(m, 'wait_restore', return_value={}), self.assertRaises(ValueError):
            m.with_paused(self.states(), run, lambda *a: None)
        self.assertEqual(events, ['stop', 'probe', 'start'])

    def test_partial_stop_error_restores_without_running_gpu(self):
        events = []
        def move(before, action):
            events.append(action)
            if action == 'stop': raise RuntimeError('one node failed to stop')
            return {}
        with patch.object(m, 'transition_all', move), patch.object(m, 'wait_restore', return_value={}), self.assertRaises(RuntimeError):
            m.with_paused(self.states(), lambda: self.fail('GPU must not run'), lambda *a: None)
        self.assertEqual(events, ['stop', 'start'])

    def test_restore_failure_is_not_success(self):
        with patch.object(m, 'transition_all', return_value={}), patch.object(m, 'wait_restore', side_effect=RuntimeError('rank exited')), self.assertRaises(RuntimeError):
            m.with_paused(self.states(), lambda: 0, lambda *a: None)

    def test_lost_hold_refuses_before_remote_mutation(self):
        with patch.object(m, 'check_holder', side_effect=RuntimeError('not ours')), patch.object(m, 'remote') as remote, self.assertRaises(RuntimeError):
            m.transition('local', self.states()['local'], 'stop')
        remote.assert_not_called()

    def test_public_restore_uses_actual_fleet_decision_and_refuses_errors(self):
        with patch.object(m, 'check_holder'), patch.dict(m.os.environ, FLEET_SESSION='test'), patch.object(m.subprocess, 'run') as run:
            run.return_value = subprocess.CompletedProcess([], 1, b'no (next boots next and replaces whatever is up)\n')
            result = {}
            m.restore_public(Path('/unused'), lambda *a: None, result)
            self.assertIn('queued boot', result['public_restore'])
            self.assertEqual(run.call_count, 1)
            run.return_value = subprocess.CompletedProcess([], 2, b'')
            with self.assertRaises(RuntimeError):
                m.restore_public(Path('/unused'), lambda *a: None, {})


if __name__ == '__main__':
    unittest.main()
