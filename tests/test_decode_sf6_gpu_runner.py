"""CPU-only fail-closed tests for the isolated SF6 correctness runner."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
import run_decode_sf6_gpu as runner


def source():
    return dict(revision='a' * 40, inputs_sha256={},
                kernels_sha256={name: 'b' * 64 for name in runner.KERNEL_FILES})


def report():
    gates = [dict(block=block, exact_expand_replays=32) for block in (1024, 2048, 4096)]
    for m, unique, fallback in sorted(runner.CASES):
        gates.append(dict(m=m, unique=unique, raw_fallback=fallback, moe_replays=8,
            lanes={arm: dict(kind='stock' if arm == 'stock' else 'static_v2', rows=m,
                        reform=arm != 'stock' and 1 <= m <= 8,
                        sf6=arm == 'sf6' and not fallback)
                   for arm in ('stock', 'baseline', 'sf6')},
            numeric=[dict(replay=replay, arm=arm, max_error=0.001, stock_noise=0., limit=.01)
                     for replay in range(8) for arm in ('baseline', 'sf6')]))
    return dict(mode='gpu', status='PASS', expand_blocks=[1024, 2048, 4096], gates=gates,
                source_sha256=source()['kernels_sha256'])


def container(cid, token, *, oom=False):
    return dict(Id=cid, Image=runner.IMAGE,
                Config=dict(Labels={'decode.sf6.owner': token}),
                State=dict(Running=False, OOMKilled=oom, ExitCode=137 if oom else 0, Error=''),
                HostConfig=dict(Memory=16*runner.GIB, MemorySwap=16*runner.GIB,
                                CpusetCpus='14-17', NetworkMode='none', ShmSize=runner.GIB,
                                DeviceRequests=[dict(DeviceIDs=['0'])]))


class ReportContract(unittest.TestCase):
    def test_complete_graph_activation_and_fallback(self):
        runner.validate_report(report(), source())

    def test_stale_direct_dynamic_kernel_hash_is_rejected(self):
        value = report()
        value['source_sha256']['moe_dynamic_gated_sf6.py'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'executed kernel mismatch: moe_dynamic_gated_sf6.py'):
            runner.validate_report(value, source())

    def test_missing_duplicate_cpu_and_stale_cases_fail(self):
        for mutate in (lambda r: r.update(mode='cpu'), lambda r: r['gates'].pop(),
                       lambda r: r['gates'].__setitem__(3, r['gates'][4]),
                       lambda r: r['source_sha256'].update({'moe_reform_sf_pack.py': 'c'*64}),
                       lambda r: r['gates'][0].update(exact_expand_replays=31),
                       lambda r: r['gates'][3]['numeric'].pop(),
                       lambda r: r['gates'][3]['numeric'].__setitem__(0, r['gates'][3]['numeric'][1]),
                       lambda r: r.update(expand_blocks=[2048, 4096])):
            value = copy.deepcopy(report())
            mutate(value)
            with self.assertRaises(ValueError):
                runner.validate_report(value, source())

    def test_wrong_dispatch_and_fallback_fail(self):
        for select in (lambda g: not g.get('raw_fallback') and g.get('m') == 1,
                       lambda g: g.get('raw_fallback'), lambda g: g.get('m') == 16):
            value = report()
            gate = next(g for g in value['gates'] if select(g))
            gate['lanes']['sf6']['sf6'] = not gate['lanes']['sf6']['sf6']
            with self.assertRaises(ValueError):
                runner.validate_report(value, source())

    def test_nonfinite_or_exceeded_numeric_fails(self):
        for key, bad in [('max_error', float('nan')), ('limit', float('inf')),
                         ('stock_noise', -1), ('max_error', .5), ('max_error', True)]:
            value = report()
            value['gates'][3]['numeric'][0][key] = bad
            with self.assertRaises(ValueError):
                runner.validate_report(value, source())

    def test_container_identity_oom_and_limits(self):
        value = container('c'*64, 'owned')
        runner.validate_container(value, 'c'*64, 'owned')
        for section, key, bad in [('State', 'OOMKilled', True), ('State', 'ExitCode', 1),
                                 ('HostConfig', 'MemorySwap', 32*runner.GIB),
                                 ('HostConfig', 'NetworkMode', 'host'),
                                 ('HostConfig', 'DeviceRequests', [dict(DeviceIDs=['1'])])]:
            value = container('c'*64, 'owned')
            value[section][key] = bad
            with self.assertRaises(ValueError):
                runner.validate_container(value, 'c'*64, 'owned')
        with self.assertRaises(ValueError):
            runner.validate_container(container('c'*64, 'other'), 'c'*64, 'owned')

    def test_network_caps_all_moe_mounts_fresh_cache(self):
        with patch.object(runner, 'mounts', return_value=['--mount', 'all-moe-readonly']) as mounted:
            command = runner.container_command(Path('/fresh'), 'owned', 'token')
        mounted.assert_called_once_with()
        for required in ('all-moe-readonly', '--network=none', '--memory=16g', '--memory-swap=16g',
                         '--cpuset-cpus=14-17', 'device=0', '--gpu', '/evidence/result.json',
                         'type=bind,src=/fresh/cache,dst=/root/.cache'):
            self.assertIn(required, command)
        self.assertNotIn('--rm', command)

    def test_ownership_and_model_guard_fail_closed(self):
        with patch.object(runner, 'HOLDER') as holder, patch.dict(runner.os.environ,
                {'FLEET_SESSION': 'mine', 'FLEET_RESTORE_MANAGED': '1'}, clear=True):
            holder.read_text.return_value = 'mine|123|boot\n'
            self.assertEqual(runner.held_session(), ('mine', 'mine|123|boot'))
            for wrong in ('other|123|boot', 'mine|123|probe'):
                holder.read_text.return_value = wrong
                with self.assertRaises(ValueError):
                    runner.held_session()
        for active in ('glm53\n', 'glm53-worker\n', 'glm53-worker-2\n', 'dsv4\n'):
            with patch.object(runner.subprocess, 'check_output', return_value=active):
                with self.assertRaises(ValueError):
                    runner.model_containers_stopped()
        with patch.object(runner.subprocess, 'check_output', return_value='unrelated-service\n'):
            runner.model_containers_stopped()
        with patch.object(runner, 'held_session', return_value=('mine', 'mine|boot')), \
             patch.object(runner, 'model_containers_stopped'), \
             patch.object(runner, 'memory_available', return_value=12*runner.GIB-1):
            with self.assertRaises(ValueError):
                runner.check_environment(('mine', 'mine|boot'), 12*runner.GIB)

    def test_stop_refuses_other_container_owner(self):
        with patch.object(runner.subprocess, 'check_output',
                return_value=json.dumps([container('c'*64, 'other')])), \
             patch.object(runner.subprocess, 'run') as invoked:
            with self.assertRaises(ValueError):
                runner.stop_owned('c'*64, 'mine')
            invoked.assert_not_called()


class RunnerLifecycle(unittest.TestCase):
    def exercise(self, *, timeout=False, oom=False, create_timeout=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        out = Path(temp.name)/'fresh'
        cid, commands, token = 'c'*64, [], [None]

        def check_output(command, **kwargs):
            commands.append(command)
            if command[:3] == ['docker', 'image', 'inspect']:
                return json.dumps([dict(Id=runner.IMAGE)])
            if command[:2] == ['docker', 'create']:
                token[0] = command[command.index('--label')+1].split('=', 1)[1]
                (out/'container.id').write_text(cid)
                if create_timeout:
                    raise subprocess.TimeoutExpired(command, 30)
                return cid + '\n'
            if command[:2] == ['docker', 'inspect']:
                return json.dumps([container(cid, token[0], oom=oom)])
            self.fail(command)

        def run(command, **kwargs):
            commands.append(command)
            if command[:2] == ['docker', 'start']:
                self.assertEqual(kwargs['timeout'], 900)
                if timeout:
                    raise subprocess.TimeoutExpired(command, 900)
                (out/'result.json').write_text(json.dumps(report()))
                kwargs['stdout'].write('REFORM_SF6_CORRECTNESS_PASS\n')
            if command[:2] == ['docker', 'rm']:
                self.assertTrue((out/'container.json').is_file(), 'inspect must precede cleanup')
                self.assertEqual(command[-1], cid)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(sys, 'argv', ['runner', '--out', str(out)]), \
             patch.object(runner, 'source_identity', side_effect=source), \
             patch.object(runner, 'held_session', return_value=('mine', 'mine|boot')), \
             patch.object(runner, 'check_environment', return_value=24*runner.GIB), \
             patch.object(runner, 'mounts', return_value=[]), \
             patch.object(runner.subprocess, 'check_output', side_effect=check_output), \
             patch.object(runner.subprocess, 'run', side_effect=run):
            if timeout or oom or create_timeout:
                with self.assertRaises(ValueError):
                    runner.main()
            else:
                runner.main()
                runner.verify_admission(out)
                (out/'probe.log').write_text('edited evidence')
                with self.assertRaises(ValueError):
                    runner.verify_admission(out)
        receipt = runner.read_json(out/'admission.json')
        self.assertEqual(receipt['status'], 'FAIL' if timeout or oom or create_timeout else 'PASS')
        self.assertEqual(runner.read_json(out/'container.json')['State']['OOMKilled'], oom)
        self.assertTrue(any(c[:2] == ['docker', 'rm'] for c in commands))

    def test_success_and_artifact_tamper(self):
        self.exercise()

    def test_timeout_preserves_inspect_before_owned_cleanup(self):
        self.exercise(timeout=True)

    def test_oom_is_fail_even_with_pass_report(self):
        self.exercise(oom=True)

    def test_create_timeout_recovers_owned_id_and_diagnostics(self):
        self.exercise(create_timeout=True)


if __name__ == '__main__':
    unittest.main()
