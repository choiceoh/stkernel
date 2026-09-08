"""Fleet admission and deferred recovery, using no GPUs or network."""
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

    def test_managed_admission_never_creates_restore_debt(self):
        self.ready('donor', 1)
        self.assertTrue(handoff.admit(self.root, 'donor', 1, 'boot'))
        self.assertFalse((self.root/'restore-debt.json').exists())
        self.assertTrue((self.root/'holder').read_text().startswith('donor|1|'))

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
