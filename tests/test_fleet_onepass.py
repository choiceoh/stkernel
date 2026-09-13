#!/usr/bin/env python3
"""Fast admission fixtures: no GPU, Docker, serving process or fleet wait."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_onepass as policy


class OnepassPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.controller = self.root / 'controller'
        self.repo = self.root / 'candidate'
        for relative in (*policy.SHELL_ENTRIES, *policy.PYTHON_ENTRIES, *policy.ST_PROBES,
                         'bench/fleet.sh', 'bench/serving_group.py', 'bench/experiment_baselines.py',
                         'bench/onepass_deploy.py'):
            for root in (self.controller, self.repo):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# reviewed fixture ' + relative + '\n')
        self.environment = {'REPO': str(self.repo)}

    def validate(self, command, **kwargs):
        return policy.validate(command, self.repo, self.controller,
                               kwargs.pop('environment', self.environment), **kwargs)

    def test_canonical_pair_chain_arm_and_live_onepass(self):
        for command in (['bash', 'bench/pair.sh', 'CAND', 'VLLM_X=1'],
                        ['bash', 'bench/chain.sh', 'A=VLLM_X=1', 'BASE='],
                        ['bash', 'bench/ab-lever.sh', 'ARM', ''],
                        ['python3', 'bench/onepass.py', '--name', 'LIVE', '--ctx', '2000,32000']):
            with self.subTest(command=command):
                self.assertEqual(self.validate(command)['policy'], 'onepass-only')
        self.assertEqual(self.validate(['python3', 'bench/onepass.py'], kind='probe')['kind'], 'probe')

    def test_custom_gpu_program_and_code_strings_are_rejected(self):
        for command in (['bash', '-c', 'python3 bench/onepass.py; docker run --gpus all image'],
                        ['python3', '-c', 'import torch; torch.zeros(1, device="cuda")'],
                        ['bash', 'probes/run_mk_probe.sh'], ['docker', 'run', '--gpus', 'all'],
                        ['bash', 'custom-onepass.sh']):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'onepass-only'):
                self.validate(command)

    def test_after_and_boot_only_legs_fail_before_execution(self):
        for command in (['bash', 'bench/chain.sh', 'A=', '--after', 'A', 'gpu-check'],
                        ['bash', 'bench/chain.sh', 'A=', '--legs', 'A', 'none'],
                        ['env', 'LEGS=none', 'bash', 'bench/ab-lever.sh', 'A'],
                        ['env', 'PREFILL_WARMUP=1', 'bash', 'bench/pair.sh', 'A'],
                        ['bash', 'bench/ab-lever.sh', 'A', 'PREFILL_WARMUP=1'],
                        ['env', 'LEGS=onepass,decode', 'bash', 'bench/pair.sh', 'A']):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'onepass-only'):
                self.validate(command)

    def test_live_lane_never_allows_boots(self):
        for entry in policy.SHELL_ENTRIES:
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, 'live-serving lane'):
                self.validate(['bash', entry, 'A'], kind='probe')

    def test_the_st_engine_checks_queue_like_everything_else(self):
        """They take the same four nodes, so they belong in this queue and not behind a
        second launcher lock no other session can see (2026-09-12)."""
        for command in (['bash', 'probes/run_engine_check.sh', '--layers', '0-4'],
                        ['bash', 'probes/run_engine_check.sh', '--layers', '0-4',
                         '--moe-static', 't,r,sf6', '--mla-prefill', 'stock'],
                        ['bash', 'probes/run_engine_probe.sh', 'probes/engine_decode_graph_check.py'],
                        ['bash', 'probes/run_engine_probe.sh', 'probes/engine_kernel_check.py',
                         '--imports-only'],
                        ['bash', 'probes/run_engine_probe.sh', 'engine/profiles/glm53/check.py',
                         '--layers', '0-4', '--distributed']):
            with self.subTest(command=command):
                self.assertEqual(self.validate(command)['entry'], command[1])

    def test_the_contract_counts_gpus_and_the_single_lane_takes_only_one_gpu_checks(self):
        """A check that needs one GPU goes to the 5050 on ost-97x, not the four Sparks
        (2026-09-12). The lane follows from `gpus`, and naming the lane can never move a
        boot onto one card."""
        one = ['bash', 'probes/run_engine_check.sh', '--layers', '0-4']
        self.assertEqual(self.validate(one)['gpus'], 1)
        self.assertEqual(self.validate(one, kind='single')['kind'], 'single')
        self.assertEqual(self.validate(['bash', 'probes/run_engine_probe.sh',
                                        'probes/engine_kernel_check.py'])['gpus'], 1)
        four = ['bash', 'probes/run_engine_probe.sh', 'engine/profiles/glm53/check.py', '--distributed']
        self.assertEqual(self.validate(four)['gpus'], 4)      # one rank per Spark
        with self.assertRaisesRegex(ValueError, 'needs the four Sparks'):
            self.validate(four, kind='single')
        for command in (['bash', 'bench/pair.sh', 'A'], ['bash', 'bench/chain.sh', 'A='],
                        ['python3', 'bench/onepass.py'], ['bash', 'probes/run_ar_consumer_campaign.sh']):
            with self.subTest(command=command):
                self.assertEqual(self.validate(command)['gpus'], 4)
                with self.assertRaisesRegex(ValueError, 'needs the four Sparks'):
                    self.validate(command, kind='single')
        # where a check runs is the lane's decision, never the command's
        with self.assertRaisesRegex(ValueError, 'ST_PROBE_HOST is set by the single-GPU lane'):
            self.validate(['env', 'ST_PROBE_HOST=srv2', 'bash', 'probes/run_engine_check.sh'])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(policy.main(['--repo', str(self.controller), '--cwd', str(self.repo),
                                          '--kind', 'single', '--', 'bash', 'probes/run_engine_check.sh']), 0)
        self.assertEqual(json.loads(out.getvalue())['gpus'], 1)

    def test_admitting_the_st_runner_never_admits_an_arbitrary_probe(self):
        """The runner is `docker run --gpus all <probe>`; the probe is named, not supplied."""
        for probe in ('probes/invented.py', 'engine/profiles/glm53/boot.py', '../escape.py'):
            with self.subTest(probe=probe), self.assertRaisesRegex(ValueError, 'not a canonical ST check'):
                self.validate(['bash', 'probes/run_engine_probe.sh', probe])
        with self.assertRaisesRegex(ValueError, 'the ST probe runner needs one of'):
            self.validate(['bash', 'probes/run_engine_probe.sh'])

    def test_st_checks_take_literal_flags_only(self):
        for extra in (['--sanitizer', 'on'], ['--layers'], ['; rm -rf /'],
                      ['--moe-static', '$(id)'], ['--output', '/etc/shadow;x']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.validate(['bash', 'probes/run_engine_check.sh', *extra])

    def test_a_modified_st_runner_or_probe_is_refused(self):
        for relative in ('probes/run_engine_probe.sh', 'probes/engine_decode_graph_check.py'):
            with self.subTest(relative=relative):
                path = self.repo / relative
                original = path.read_bytes()
                path.write_bytes(original + b'# docker run --gpus all whatever\n')
                with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
                    self.validate(['bash', 'probes/run_engine_probe.sh',
                                   'probes/engine_decode_graph_check.py'])
                path.write_bytes(original)

    def test_familiar_filename_cannot_hide_modified_source(self):
        (self.repo / 'bench/pair.sh').write_text('docker run --gpus all extra\n')
        with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
            self.validate(['bash', 'bench/pair.sh', 'A'])

    def test_transitive_onepass_or_lever_change_is_rejected(self):
        for relative in ('bench/ab-lever.sh', 'bench/onepass.py'):
            with self.subTest(relative=relative):
                path = self.repo / relative
                original = path.read_bytes()
                path.write_bytes(original + b'# stale workload\n')
                with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
                    self.validate(['bash', 'bench/pair.sh', 'A'])
                path.write_bytes(original)

    def test_env_prefixes_use_effective_values_and_verify_overrides(self):
        lever = self.root / 'ab-lever2.sh'
        shutil.copyfile(self.controller / 'bench/ab-lever.sh', lever)
        env = dict(self.environment, LEVER=str(lever), FLEET=str(self.controller / 'bench/fleet.sh'))
        self.validate(['env', '-u', 'LEGS', 'bash', 'bench/pair.sh', 'A'],
                      environment=dict(env, LEGS='none'))
        self.validate(['env', '-i', 'REPO=' + str(self.repo), 'bash', 'bench/pair.sh', 'A'])
        lever.write_text('custom GPU work\n')
        with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
            self.validate(['env', 'LEVER=' + str(lever), 'bash', 'bench/pair.sh', 'A'])
        with self.assertRaisesRegex(ValueError, 'injection setting'):
            self.validate(['env', 'BASH_ENV=/tmp/custom.sh', 'bash', 'bench/pair.sh', 'A'])

    def test_arm_grammar_rejects_hidden_control_commands_and_duplicates(self):
        for command in (['bash', 'bench/pair.sh', 'A', 'VLLM_X=1;docker run'],
                        ['bash', 'bench/pair.sh', 'A', 'LEVER=/tmp/custom'],
                        ['bash', 'bench/chain.sh', 'A=', 'A=VLLM_X=1'],
                        ['bash', 'bench/ab-lever.sh', 'A', '', 'extra']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.validate(command)

    def test_onepass_accepts_public_arguments_and_rejects_custom_hooks(self):
        self.validate(['python3', 'bench/onepass.py', '--name', 'A', '--max-tokens', '400',
                       '--combined-max-tokens', '2400', '--combined-reasoning-budget', '900',
                       '--num-spec', '7', '--seed', '7', '--combine-min-ctx', '32000',
                       '--fixed-decode-tokens', '20', '--fixed-decode-reps', '2',
                       '--require-exclusive', '--out', str(self.root / 'records.jsonl')])
        for args in (['--after', 'gpu-check'], ['--ctx', '0'], ['--num-spec', '-1'],
                     ['--combined-max-tokens', '1199'],
                     ['--combined-max-tokens', '7200', '--combined-reasoning-budget', '7200'],
                     ['--fixed-decode-tokens', '1', '--fixed-decode-reps', '0']):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.validate(['python3', 'bench/onepass.py', *args])

    def test_only_documented_ar_campaign_arguments_are_allowed(self):
        for args in ([], ['--baseline-only'], ['--gpu-evidence', '/prior/results'],
                     ['--gpu-evidence', '/prior/results', '--baseline-only']):
            self.validate(['bash', 'probes/run_ar_consumer_campaign.sh', *args])
        for args in (['--gpu-evidence'], ['--after', 'gpu-check'], ['--check-gpu']):
            with self.assertRaises(ValueError):
                self.validate(['bash', 'probes/run_ar_consumer_campaign.sh', *args])

    def experiment(self, kind='pair', command=None, repo=None):
        root = self.root / 'experiments'
        root.mkdir(exist_ok=True)
        with contextlib.closing(sqlite3.connect(root / 'experiments.sqlite3')) as connection, connection:
            connection.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,payload TEXT)')
            payload = dict(repo=str(repo or self.repo), bash='/bin/bash',
                           spec=dict(kind=kind, command=command or [], knobs={'VLLM_X': '1'}, env={}))
            connection.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)', ('job1', json.dumps(payload)))
        return ['python3', 'bench/experiments.py', '--root', str(root), 'execute', 'job1']

    def test_recorded_pair_and_baseline_are_admitted_read_only(self):
        for kind in ('pair', 'baseline'):
            self.assertEqual(self.validate(self.experiment(kind))['entry'], 'bench/experiments.py')

    def test_recorded_custom_probe_wrong_repo_and_missing_job_are_rejected(self):
        for kind in ('probe', 'cpu'):
            with self.assertRaisesRegex(ValueError, 'standard pair'):
                self.validate(self.experiment(kind))
        with self.assertRaisesRegex(ValueError, 'standard pair'):
            self.validate(self.experiment(command=['bash', 'custom.sh']))
        with self.assertRaisesRegex(ValueError, 'execution repository'):
            self.validate(self.experiment(repo=self.controller))
        command = self.experiment()
        command[-1] = 'missing'
        with self.assertRaisesRegex(ValueError, 'unknown experiment'):
            self.validate(command)
        with self.assertRaisesRegex(ValueError, 'expected experiments.py'):
            self.validate(['python3', 'bench/experiments.py', 'worker', 'job1'])

    def test_only_fabricating_helpers_receive_cpu_rehearsal_exemption(self):
        prefix = ['--repo', str(self.controller), '--cwd', str(self.repo), '--rehearsal-only', '--']
        with patch.dict(os.environ, dict(self.environment, FLEET_REHEARSE='1'), clear=True), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(policy.main([*prefix, 'bash', 'bench/pair.sh', 'A']), 0)
            self.assertEqual(policy.main([*prefix, 'python3', 'bench/onepass.py']), 2)
            self.assertEqual(policy.main([*prefix, 'bash', 'probes/run_mk_probe.sh']), 2)

    def test_wait_requires_live_registered_supervisor_ancestry(self):
        record = dict(pid=321, state='queued', prepare_manifest='/private/receipt')
        with patch('fleet_pending.read_record', return_value=record), \
                patch('fleet_handoff.live', return_value=True), \
                patch('fleet_idle.descendant', return_value=True):
            self.assertEqual(policy.authorize_wait(self.root, 'mine', 321)['owner'], 321)
        with patch('fleet_pending.read_record', return_value=record), \
                patch('fleet_handoff.live', return_value=True), \
                patch('fleet_idle.descendant', return_value=False), \
                self.assertRaisesRegex(ValueError, 'owning supervisor'):
            policy.authorize_wait(self.root, 'mine', 321)
            self.assertEqual(policy.main([*prefix, 'env', 'FLEET_REHEARSE=0', 'bash', 'bench/pair.sh', 'A']), 2)
            self.assertEqual(policy.main([*prefix, 'env', '-u', 'FLEET_REHEARSE', 'bash', 'bench/pair.sh', 'A']), 2)


if __name__ == '__main__':
    unittest.main()


class FleetOccupancyTests(unittest.TestCase):
    """The queue and the ST launcher reserve the same four nodes, and the fleet LEASE is the one
    record of who holds them: the queue takes it at GO, the launcher verifies it (2026-09-12: four
    nodes ran st-glm53 while `fleet.sh status` answered FREE, because FREE only meant 'the holder
    file is empty' -- and later, a ticket's own boot was refused by the holder file that was its)."""

    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.fleet = (root / 'bench/fleet.sh').read_text()
        self.launcher = (root / 'launchers/start-st-glm53.sh').read_text()

    def test_the_queue_sees_st_containers(self):
        self.assertIn("st_engine_up() {", self.fleet)
        # this node's containers, the head's lease, and the other three nodes -- the local
        # `docker ps` alone answered for one Spark of four (2026-09-12)
        self.assertIn("grep -E '^st-'", self.fleet)
        self.assertIn("st_engine_elsewhere", self.fleet)
        self.assertIn("lease_state() { lease read", self.fleet)      # containers AND the lease: they disagreed once
        # a grant is refused on both paths that hand out the fleet -- unless the lease is the
        # asking ticket's own, handed to it by the holder that drained
        self.assertIn('if ST_MINE=$s st_engine_up; then', self.fleet)
        self.assertIn('if ST_MINE=$s st_engine_up; then echo "ST engine occupies the fleet', self.fleet)
        # refusing alone would leave a queued session waiting for a human to go and ask: the
        # holder's KIND decides -- production through the quiet gate, a session never (45차 §91)
        self.assertIn('st_engine_ask "$s" "$pid" "$est" "$note"', self.fleet)
        self.assertIn('is not asked', self.fleet)
        # and status says so instead of FREE
        self.assertIn('TAKEN by the ST engine, outside this queue', self.fleet)

    def test_the_launcher_verifies_the_queue_s_lease(self):
        """One record. The launcher used to read the queue's holder file as a second one, and
        that file refused a ticket's own boot: no ST boot could run under the queue at all."""
        self.assertNotIn('FLEET_HOLDER', self.launcher)
        self.assertIn('lease verify --owner "$LEASE_OWNER"', self.launcher)
        self.assertIn('this boot holds no reservation', self.launcher)
        # and it still honours the older lock and refuses to share with serving containers
        self.assertIn('st-fleet.lock', self.launcher)
        self.assertIn("grep -E '^(glm53|q38|vllm|st-)'", self.launcher)

    def test_the_queue_resolves_its_own_checkout(self):
        """It hardcoded /home/choiceoh/stkernel, which is whatever branch another session
        left there -- on 2026-09-12 one with no bench/fleet_*.py, so every helper errored."""
        self.assertIn('REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}', self.fleet)
        self.assertNotIn('REPO=${REPO:-/home/choiceoh/stkernel}', self.fleet)

    def test_the_classifier_calls_the_st_runners_gpu(self):
        self.assertIn('run_engine_probe|run_engine_check', self.fleet)
        classify = (Path(__file__).resolve().parents[1] / 'bench/fleet_classify.py').read_text()
        self.assertIn('run_engine_probe|run_engine_check', classify)
