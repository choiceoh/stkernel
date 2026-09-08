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
                     'fleet_classify.py', 'pair.sh', 'chain.sh', 'ab-lever.sh', 'onepass.py',
                     'onepass_deploy.py', 'measurement_contract.py', 'fleet_handoff.py',
                     'fleet_pending.py', 'fleet_idle.py'):
            shutil.copyfile(ROOT / 'bench' / name, self.repo / 'bench' / name)
        (self.repo / 'probes').mkdir()
        shutil.copyfile(ROOT / 'probes/run_ar_consumer_campaign.sh',
                        self.repo / 'probes/run_ar_consumer_campaign.sh')
        self.prepared = self.root / 'preparation-started'
        self.executed = self.root / 'payload-started'
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        # Refusal regressions remain CPU-only: stop at preparation before any
        # GPU command could be launched, while recording that early rejection
        # failed. Imports of fleet_prepare still use its real source.
        shim = self.bin / 'python3'
        shim.write_text('#!' + sys.executable + '\n'
                        'import os, pathlib, sys\n'
                        'if len(sys.argv)>1 and pathlib.Path(sys.argv[1]).name=="fleet_prepare.py":\n'
                        '    pathlib.Path(os.environ["PREPARATION_SENTINEL"]).write_text("started")\n'
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

    def prepare_preflight_fixture(self):
        (self.repo / 'profiles').mkdir()
        shutil.copyfile(ROOT / 'profiles/glm53.env', self.repo / 'profiles/glm53.env')
        # Exercise the actual shell preflight and canonical entrypoint policy,
        # but use the copied profile without fetching or executing a payload.
        git = self.bin / 'git'
        git.write_text('#!/bin/sh\nexit 1\n')
        git.chmod(0o700)

    def test_preflight_canonical_chain_ignores_header_example_knobs(self):
        self.prepare_preflight_fixture()
        chain = self.repo / 'bench/chain.sh'
        self.assertIn('NAME="VLLM_X=1 VLLM_Y=1"', chain.read_text())
        result = self.run_fleet('preflight', 'chain-header', '--', 'bash', str(chain),
                                'A=VLLM_GLM53_MEGAKERNEL=1')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('PASS knobs declared', result.stdout)
        self.assertNotIn('VLLM_X', result.stdout)
        self.assertNotIn('VLLM_Y', result.stdout)
        self.assert_no_work()

    def test_preflight_ignores_indented_source_comment_knobs(self):
        self.prepare_preflight_fixture()
        chain = self.repo / 'bench/chain.sh'
        chain.write_text(chain.read_text() + '\n \t# VLLM_COMMENT_EXAMPLE=1\n')
        result = self.run_fleet('preflight', 'chain-comment', '--', 'bash', str(chain), 'A=')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn('VLLM_COMMENT_EXAMPLE', result.stdout)
        self.assert_no_work()

    def test_preflight_still_rejects_example_names_in_actual_caller_knobs(self):
        self.prepare_preflight_fixture()
        result = self.run_fleet('preflight', 'chain-undeclared', '--', 'bash',
                                str(self.repo / 'bench/chain.sh'), 'A=VLLM_X=1 VLLM_Y=1')
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('FAIL undeclared', result.stdout)
        self.assertIn('VLLM_X VLLM_Y', result.stdout)
        self.assert_no_work()

    def test_preflight_still_checks_executable_source_assignments(self):
        self.prepare_preflight_fixture()
        chain = self.repo / 'bench/chain.sh'
        # This fixture's canonical source contains a real assignment, ensuring
        # comment filtering does not disable source knob inspection altogether.
        chain.write_text(chain.read_text() + '\nVLLM_UNDECLARED_SOURCE=1\n')
        result = self.run_fleet('preflight', 'chain-source', '--', 'bash', str(chain), 'A=')
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('FAIL undeclared', result.stdout)
        self.assertIn('VLLM_UNDECLARED_SOURCE', result.stdout)
        self.assert_no_work()

    def test_arbitrary_gpu_command_is_rejected_before_cpu_preparation(self):
        # The marker is harmless even if invoked; the text also identifies
        # this as GPU work to the existing CPU/GPU classifier.
        code = 'from pathlib import Path; Path(' + repr(str(self.executed)) + ').write_text("ran") # torch.cuda'
        result = self.run_fleet('run', '--gpu', '--prepare', str(self.preparation_spec),
                                'custom', '1', 'fixture', '--', 'python3', '-c', code)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('onepass-only', result.stdout)
        self.assert_no_work()

    def test_chain_after_is_rejected_before_cpu_preparation(self):
        result = self.run_fleet('run', '--gpu', '--prepare', str(self.preparation_spec),
                                'chain-custom', '1', 'fixture', '--', 'bash', 'bench/chain.sh',
                                'A=', '--after', 'A', 'touch ' + str(self.executed))
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('without --after or --legs', result.stdout)
        self.assert_no_work()

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

    def test_pinned_controller_preserves_ar_wrapper_and_rejects_mutations(self):
        runner = fleet_pin.pin(self.repo, self.directory)
        relative = 'probes/run_ar_consumer_campaign.sh'
        self.assertEqual((runner / relative).read_bytes(), (self.repo / relative).read_bytes())
        command = ['bash', relative, '--baseline-only']
        policy.validate(command, self.repo, runner, self.environment)
        original = (self.repo / relative).read_bytes()
        (self.repo / relative).write_bytes(original + b'\n# an unreviewed additional workload\n')
        with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
            policy.validate(command, self.repo, runner, self.environment)
        (self.repo / relative).write_bytes(original)
        (runner / relative).write_bytes(original + b'\n# corrupted pin\n')
        with self.assertRaisesRegex(ValueError, 'pinned runner integrity'):
            fleet_pin.pin(self.repo, self.directory)

    def test_failed_real_preflight_edit_preserves_ticket_command_and_order(self):
        runner = fleet_pin.pin(self.repo, self.directory)
        pid = os.getpid()
        queue = self.directory / 'queue'
        queue.write_text(f'10|before|100|1|neighbor|boot|{pid}\n'
                         f'20|mine|101|2|original|boot|{pid}\n'
                         f'30|after|102|1|neighbor|boot|{pid}\n')
        original_command = ['bash', str(self.repo / 'bench/pair.sh'), 'A', '']
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
