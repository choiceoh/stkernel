"""Exercise automatic GPU gate reuse with committed sources and CPU fake ranks."""
from copy import deepcopy
from contextlib import redirect_stdout
import builtins
import errno
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'probes'))
import ar_consumer_gpu_identity as identity
import reuse_ar_consumer_gpu_evidence as reuse
import run_ar_consumer_gpu as runner


class StageReuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'fixture')
        for name in reuse.SOURCES:
            self.write_source(name, 'fixture source\n')
        for name in reuse.INPUTS:
            self.write_source(name if Path(name).suffix else name + '/fixture.py', 'pass\n')
        self.commit()
        root_patch = patch.object(reuse, 'ROOT', self.repo)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.runtime = {
            'schema': 1,
            'image': reuse.IMAGE,
            'topology': {'nodes': list(reuse.NODES), 'ips': runner.IPS,
                         'init': 'tcp://10.10.10.2:29758'},
            'nodes': {node: {
                'image': reuse.IMAGE,
                'gpu': {'uuid': 'GPU-fixture-' + str(rank), 'name': 'NVIDIA GB10',
                        'driver_version': '580.00', 'pci_bus_id': '0000:01:00.0'},
                'kernel': '6.14.0|fixture',
                'rdma': {'device': 'mlx5_0', 'firmware': 'fixture'},
                'sanitizer_sha256': 'a' * 64,
            } for rank, node in enumerate(reuse.NODES)},
        }

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True).strip()

    def write_source(self, name, value):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')

    def rank_evidence(self, out, name):
        """A successful report has all numerical, lifecycle and ownership cases."""
        rank = int(name.rsplit('rank', 1)[1])
        distributed = not name.startswith('local-')
        stage = name.removeprefix('local-').split('-rank', 1)[0]
        hashes = reuse.source()['source_sha256']
        report = {
            'rank': rank, 'status': 'PASS',
            'mode': 'distributed' if distributed else 'delayed-producer',
            'torch': '2.13.0+cu130', 'cuda': '13.0', 'device': 'NVIDIA GB10',
            'source_sha256': hashes, 'mhc_warmup_capture': 'PASS',
            'mhc_pre_view_cases': [
                {'consumer': early, 'input_value': value, 'passed': True}
                for early in (False, True) for value in (.03125, 0., -.0625)],
            'cases': [{'tokens': tokens, 'fp32_fn': fp32, 'seed': seed,
                       'exact_outputs': 6, 'pass': True}
                      for tokens in (1, 2, 6, 8, 16, 32)
                      for fp32 in (False, True) for seed in (17, 0, 29)],
            'ar_ownership_cases': [
                {'elements': n, 'seed': seed, 'pass': True}
                for n in reuse.AR_OWNERSHIP_SIZES for seed in (17, 0, 29)
            ] if distributed else [],
        }
        limit = (24 if stage == 'racecheck' else 8) * 1024**3
        self.write_json(out / (name + '.json'), report)
        self.write_json(out / (name + '.container.json'), {
            'state': {'ExitCode': 0, 'OOMKilled': False}, 'image': reuse.IMAGE,
            'memory_limit': limit, 'memory_swap_limit': limit, 'cpus': '14-17'})
        (out / (name + '.log')).write_text({
            'probe': 'PASS probe\n',
            'memcheck': 'ERROR SUMMARY: 0 errors\n',
            'racecheck': 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n',
        }[stage])
        return {'node': reuse.NODES[rank], 'stage': name, 'source_sha256': hashes,
                'kernel_filter': runner.RACECHECK_KERNELS if stage == 'racecheck' else None}

    def execute(self, name, candidates=(), runtime=None, fail_group=None):
        out = self.base / name
        calls = []

        def run_group(group):
            calls.append(group)
            stages = reuse.group_stages(group)
            if group == fail_group:
                # Two successful workers cannot certify a four-rank run.
                for stage in stages[:2]:
                    self.rank_evidence(out, stage)
                raise RuntimeError('fixture interrupted distributed stage')
            return [self.rank_evidence(out, stage) for stage in stages]

        current = runtime or self.runtime
        with redirect_stdout(io.StringIO()):
            runner.execute_groups(out, current, list(candidates), run_group,
                                  lambda: deepcopy(current))
        return out, calls

    def test_complete_repeat_executes_zero_gpu_groups(self):
        first, calls = self.execute('first')
        self.assertEqual(calls, list(reuse.GROUPS))
        second, calls = self.execute('second', [first])
        self.assertEqual(calls, [])
        admission = json.loads((second / 'admission.json').read_text())
        self.assertEqual(len(admission['completed']), 15)
        for group in reuse.GROUPS:
            reuse.verify_group(second, group, self.runtime)
            for stage in reuse.group_stages(group):
                for suffix in ('.json', '.container.json', '.log'):
                    self.assertEqual((first / (stage + suffix)).read_bytes(),
                                     (second / (stage + suffix)).read_bytes())

    def test_interrupted_group_preserves_prior_groups_and_reruns_all_its_ranks(self):
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.execute('interrupted', fail_group='memcheck')
        interrupted = self.base / 'interrupted'
        admission = json.loads((interrupted / 'admission.json').read_text())
        self.assertEqual(len(admission['completed']), 7)
        self.assertTrue((interrupted / 'memcheck-rank1.json').exists())
        resumed, calls = self.execute('resumed', [interrupted])
        self.assertEqual(calls, ['memcheck', 'racecheck'])
        _, calls = self.execute('repeat-resumed', [resumed])
        self.assertEqual(calls, [])

    def test_distributed_ranks_from_separate_runs_cannot_complete_a_group(self):
        full, _ = self.execute('full')
        candidates = []
        for label, keep in (('rank01', {0, 1}), ('rank23', {2, 3})):
            out = self.base / label
            shutil.copytree(full, out)
            admission = json.loads((out / 'admission.json').read_text())
            removed = {f'racecheck-rank{rank}' for rank in range(4) if rank not in keep}
            admission['completed'] = [entry for entry in admission['completed']
                                      if entry['stage'] not in removed]
            for stage in removed:
                for suffix in ('.json', '.container.json', '.log'):
                    name = stage + suffix
                    (out / name).unlink()
                    admission['artifacts_sha256'].pop(name, None)
            self.write_json(out / 'admission.json', admission)
            candidates.append(out)
        _, calls = self.execute('cannot-mix-ranks', candidates)
        self.assertEqual(calls, ['racecheck'])

    def test_tampering_and_failed_report_rerun_only_affected_groups(self):
        previous, _ = self.execute('previous')
        with (previous / 'local-memcheck-rank0.log').open('a') as log:
            log.write('changed after admission\n')
        name = 'racecheck-rank2.json'
        report = json.loads((previous / name).read_text())
        report['status'] = 'FAIL'
        self.write_json(previous / name, report)
        # Even a matching artifact checksum cannot bless a failed report.
        admission = json.loads((previous / 'admission.json').read_text())
        admission['artifacts_sha256'][name] = hashlib.sha256((previous / name).read_bytes()).hexdigest()
        self.write_json(previous / 'admission.json', admission)
        _, calls = self.execute('repair', [previous])
        self.assertEqual(calls, ['local-memcheck', 'racecheck'])

    def test_peer_runtime_change_preserves_local_groups_but_image_and_code_do_not(self):
        previous, _ = self.execute('previous')
        runtime = deepcopy(self.runtime)
        runtime['nodes'][reuse.NODES[2]]['gpu']['driver_version'] = '581.00'
        _, calls = self.execute('peer-updated', [previous], runtime)
        self.assertEqual(calls, ['probe', 'memcheck', 'racecheck'])

        image = 'sha256:' + 'b' * 64
        runtime = deepcopy(self.runtime)
        runtime['image'] = image
        for node in runtime['nodes'].values():
            node['image'] = image
        with (patch.object(identity, 'IMAGE', image), patch.object(reuse, 'IMAGE', image),
              patch.object(runner, 'IMAGE', image)):
            _, calls = self.execute('image-updated', [previous], runtime)
        self.assertEqual(calls, list(reuse.GROUPS))

        self.write_source('profiles/fixture.py', 'changed profile\n')
        self.commit()
        _, calls = self.execute('code-updated', [previous])
        self.assertEqual(calls, list(reuse.GROUPS))

    def test_documentation_commit_reuses_evidence_and_legacy_requires_explicit_verify(self):
        previous, _ = self.execute('previous')
        self.write_source('README.md', 'Unrelated documentation.\n')
        self.commit()
        _, calls = self.execute('docs-only', [previous])
        self.assertEqual(calls, [])
        (previous / 'runtime.json').unlink()
        (previous / 'source.json').unlink()
        self.assertEqual(reuse.verify(previous)['stages_passed'], 15)
        _, calls = self.execute('legacy-auto-miss', [previous])
        self.assertEqual(calls, list(reuse.GROUPS))

    def test_legacy_migration_accepts_only_the_reviewed_runner_pair(self):
        previous, _ = self.execute('previous')
        name = 'probes/run_ar_consumer_gpu.py'
        original = (self.repo / name).read_bytes()
        replacement = b'# reviewed orchestration-only replacement\n'
        self.write_source(name, replacement.decode())
        self.commit()
        pair = {hashlib.sha256(original).hexdigest(): hashlib.sha256(replacement).hexdigest()}
        with patch.object(reuse, 'LEGACY_RUNNER_EQUIVALENCE', pair):
            self.assertEqual(reuse.verify(previous)['stages_passed'], 15)
            self.write_source(name, '# another unreviewed change\n')
            self.commit()
            with self.assertRaisesRegex(ValueError, 'outside the reviewed'):
                reuse.verify(previous)

    def test_edited_metadata_cannot_hide_a_changed_profile(self):
        previous, _ = self.execute('previous')
        self.write_source('profiles/fixture.py', 'changed profile\n')
        self.commit()
        recorded = json.loads((previous / 'source.json').read_text())
        recorded['inputs_sha256'] = reuse.source()['inputs_sha256']
        self.write_json(previous / 'source.json', recorded)
        _, calls = self.execute('changed-metadata', [previous])
        self.assertEqual(calls, list(reuse.GROUPS))

    def test_runtime_collection_uses_read_only_fake_remote_commands(self):
        calls = []

        def remote(node, argv, **kwargs):
            calls.append((node, argv))
            self.assertEqual(argv[:2], ['python3', '-c'])
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.runtime['nodes'][node]))

        self.assertEqual(reuse.collect_runtime(remote), self.runtime)
        self.assertEqual({node for node, _ in calls}, set(reuse.NODES))
        self.assertEqual(len(calls), 4)

    def test_runtime_script_accepts_unused_gid_slots_and_preserves_read_failures(self):
        sysfs = self.base / 'infiniband'
        device = sysfs / 'mlx5_0'
        port = device / 'ports/1'
        unused = set()
        for field, value in {'gids': 'fe80::1', 'gid_attrs/ndevs': 'eth0',
                             'gid_attrs/types': 'RoCE v2'}.items():
            directory = port / field
            directory.mkdir(parents=True)
            (directory / '0').write_text(value + '\n')
            (directory / '1').write_text('unused\n')
            unused.add(directory / '1')
        (device / 'node_guid').write_text('fixture-guid\n')
        (port / 'state').write_text('4: ACTIVE\n')
        sanitizer = self.base / 'sanitizer'
        sanitizer.mkdir()
        (sanitizer / 'compute-sanitizer').write_bytes(b'fixture executable, never launched')
        remapped = {'/sys/class/infiniband': sysfs,
                    '/usr/local/cuda/compute-sanitizer': sanitizer}
        network = [{'ifname': name, 'address': 'fixture-mac',
                    'addr_info': [{'family': 'inet', 'local': address,
                                   'prefixlen': 24, 'scope': 'global'}]}
                   for name, address in (('eth0', '10.10.10.2'), ('docker0', '172.17.0.1'))]

        def command(argv, **kwargs):
            if argv == ('docker', 'image', 'inspect', '--format={{.Id}}', reuse.IMAGE):
                return reuse.IMAGE
            if argv[0] == 'nvidia-smi':
                return 'GPU-fixture, NVIDIA GB10, 580.00, 0000:01:00.0\n'
            if argv == ('ip', '-json', 'address', 'show'):
                return json.dumps(network)
            self.fail('unexpected subprocess: ' + repr(argv))

        def import_module(name, *args, **kwargs):
            if name == 'pathlib':
                return SimpleNamespace(Path=lambda path: remapped[path])
            if name == 'subprocess':
                return SimpleNamespace(check_output=command)
            return builtins.__import__(name, *args, **kwargs)

        original_read = Path.read_text

        def run_script(slot_error, permission_error=False, populated=True):
            def read(path, *args, **kwargs):
                if path in unused:
                    raise OSError(slot_error, 'fixture unused GID slot')
                if permission_error and path == device / 'node_guid':
                    raise PermissionError(errno.EACCES, 'fixture permission denied')
                if not populated and path == port / 'gid_attrs/ndevs/0':
                    return '\n'
                return original_read(path, *args, **kwargs)

            namespace = {'IMAGE': reuse.IMAGE, 'TOPOLOGY_IPS': identity.IPS.split(','),
                         'NODE_IP': '10.10.10.2',
                         '__builtins__': dict(vars(builtins), __import__=import_module)}
            output = io.StringIO()
            with patch.object(Path, 'read_text', read), redirect_stdout(output):
                exec(compile(identity.RUNTIME_SCRIPT, '<runtime-script>', 'exec'), namespace)
            return json.loads(output.getvalue())

        for code in (errno.EINVAL, errno.ENODATA):
            with self.subTest(unused_slot=errno.errorcode[code]):
                report = run_script(code)
                fields = report['rdma']['mlx5_0']['ports']['1']
                self.assertEqual(fields['gid_attrs/ndevs']['0'], 'eth0')
                self.assertEqual(fields['gid_attrs/ndevs']['1'],
                                 {'unavailable': errno.errorcode[code]})
                self.assertEqual([row['ifname'] for row in report['rdma']['network']], ['eth0'])
                self.assertEqual(len(report['sanitizer_sha256']), 64)
        with self.assertRaisesRegex(ValueError, 'no populated RDMA'):
            run_script(errno.EINVAL, populated=False)
        with self.assertRaises(PermissionError) as denied:
            run_script(errno.EINVAL, permission_error=True)
        self.assertEqual(denied.exception.errno, errno.EACCES)


if __name__ == '__main__':
    unittest.main()
