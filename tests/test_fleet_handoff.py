"""Restore ownership and real Linux shell admission, using no GPUs or network."""
import base64
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

    def test_interrupted_admission_always_has_recoverable_holder(self):
        self.ready('donor', 1); handoff.admit(self.root, 'donor', 1, 'boot')
        self.ready('next', 2); handoff.offer(self.root, 'donor')
        with patch.object(handoff, 'claim_held', side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):
                handoff.admit(self.root, 'next', 2, 'boot')
        self.assertTrue((self.root/'holder').read_text().startswith('next|2|'))
        self.assertEqual(handoff.read(self.root/'restore-debt.json')['owner']['session'], 'donor')
        handoff.claim_held(self.root, 'next', 2)
        self.assertEqual(handoff.read(self.root/'restore-debt.json')['owner']['session'], 'next')
        with self.assertRaises(ValueError):
            handoff.claim_held(self.root, 'donor', 1)

    def test_previous_protocol_is_not_offered_a_handoff(self):
        self.ready('donor', 1); handoff.admit(self.root, 'donor', 1, 'boot')
        old = self.ready('old', 2); old['protocol'] = 1
        handoff.write(handoff.receipt(self.root, 'old'), old)
        self.assertIsNone(handoff.offer(self.root, 'donor'))


class EntryTests(unittest.TestCase):
    def container(self, port='18000', image='sha256:approved'):
        script = f'export NCCL_DEBUG=WARN\nvllm serve model --host 0.0.0.0 --port {port} > /glmlogs/glm53.log 2>&1\n'
        encoded = base64.b64encode(script.encode()).decode()
        return dict(State=dict(Running=True), Image=image,
                    Config=dict(Cmd=['-c', f'echo {encoded} | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh']))

    def test_real_launcher_wrapper_probes_isolated_port(self):
        from io import BytesIO
        health = BytesIO(b''); health.status = 200
        metrics = BytesIO(b'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n')
        with patch.object(fleet_entry.urllib.request, 'urlopen', side_effect=[health, metrics]) as request:
            fleet_entry.idle(self.container(), 'http://public:8000')
        self.assertEqual([c.args[0] for c in request.call_args_list],
                         ['http://127.0.0.1:18000/health', 'http://127.0.0.1:18000/metrics'])
        self.assertEqual(fleet_entry.serve_args(dict(Config=dict(Cmd=['vllm','serve','m','--host=0.0.0.0','--port','8000']))),
                         dict(host='0.0.0.0', port='8000'))

    def test_public_restore_skip_requires_approved_immutable_image(self):
        import types
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'profiles').mkdir()
            (root/'profiles/glm53.env').write_text(f'PROFILE_IMAGE="fixture:approved"\nPROFILE_OVERLAY_DIR="{root}"\n')
            data = b'# source_commit=abc\n'
            (root/'manifest.tsv').write_bytes(data)
            (root/'stamp').write_text(hashlib.sha256(data).hexdigest())
            def output(command, **kwargs):
                return 'sha256:approved\n' if command[0] == 'docker' else 'abc\n'
            with patch.dict(os.environ, MK_OVERLAY_STAMP=str(root/'stamp')), \
                    patch.dict(sys.modules, onepass=types.SimpleNamespace(_served_build=lambda repo:dict(knobs={},boot_id='new'))), \
                    patch.object(fleet_entry.subprocess, 'check_output', side_effect=output), \
                    patch.object(fleet_entry, 'idle', return_value='idle'):
                self.assertTrue(fleet_entry.production_current(root, self.container('8000')))
                self.assertFalse(fleet_entry.production_current(root, self.container('8000', 'sha256:candidate')))
                self.assertFalse(fleet_entry.production_current(root, self.container('18000')))


if __name__ == '__main__':
    unittest.main()
