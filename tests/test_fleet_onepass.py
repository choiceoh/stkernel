#!/usr/bin/env python3
"""Fast admission fixtures: no GPU, Docker, serving process or fleet wait."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_onepass as policy
from measurement_contract import COMBINED_MAX_TOKENS


class OnepassPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.controller = self.root / 'controller'
        self.repo = self.root / 'candidate'
        for relative in (*policy.SHELL_ENTRIES, *policy.PYTHON_ENTRIES, *policy.ST_PROBES,
                         *policy.ST_BRACKET_DEPENDENCIES, 'bench/fleet.sh'):
            for root in (self.controller, self.repo):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# reviewed fixture ' + relative + '\n')
        self.environment = {'REPO': str(self.repo)}

    def validate(self, command, **kwargs):
        return policy.validate(command, self.repo, self.controller,
                               kwargs.pop('environment', self.environment), **kwargs)

    def test_canonical_bracket_arm_and_live_onepass(self):
        for command in (['bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'],
                        ['bash', 'bench/st_bracket.sh', 'hold', '0123456789abcdef', '45'],
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

    def test_the_live_lane_accepts_only_onepass_and_the_bracket_probe(self):
        self.assertEqual(self.validate(['bash', 'bench/st_bracket.sh', 'probe'], kind='probe')['entry'],
                         'bench/st_bracket.sh')
        for entry in ('probes/run_engine_probe.sh', 'probes/run_engine_check.sh'):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, 'live-serving lane'):
                self.validate(['bash', entry, 'A'], kind='probe')
        with self.assertRaisesRegex(ValueError, 'live-serving lane'):
            self.validate(['bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'], kind='probe')

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

    def test_commit_only_is_scoped_to_the_pinned_kda_probe(self):
        command = ['bash', 'probes/run_engine_probe.sh', 'probes/engine_kda_deferred_check.py',
                   '--commit-only', '--samples', '20', '--output', '/cache/kda-commit.json']
        self.assertEqual(self.validate(command, kind='single')['gpus'], 1)
        for other in ('probes/engine_kernel_check.py', 'probes/engine_full_check.py'):
            with self.subTest(probe=other), self.assertRaises(ValueError):
                self.validate(command[:2]+[other]+command[3:])
        with self.assertRaises(ValueError):
            self.validate(['bash', 'probes/run_engine_check.sh', '--commit-only'])
        for unsafe in ('/cache/a;echo', '/cache/$(id)', '/cache/a b'):
            with self.subTest(output=unsafe), self.assertRaises(ValueError):
                self.validate(command[:-1]+[unsafe])

    def test_ffn_packets_probe_requires_the_reviewed_bytes(self):
        command = ['bash', 'probes/run_engine_probe.sh', 'probes/engine_ffn_packets_check.py',
                   '--ranks', '/models/st-ranks', '--samples', '8', '--output', '/cache/ffn.json']
        result = self.validate(command, kind='single')
        self.assertEqual(result['gpus'], 1)
        self.assertEqual(policy.probe_budget_gib(command[2]), 8)
        (self.repo/command[2]).write_text('# unreviewed replacement\n')
        with self.assertRaises(ValueError):
            self.validate(command, kind='single')

    def test_router_only_is_scoped_to_the_pinned_packet_probe(self):
        command = ['bash', 'probes/run_engine_probe.sh', 'probes/engine_ffn_packets_check.py',
                   '--router-only', '--samples', '8', '--output', '/cache/router.json']
        self.assertEqual(self.validate(command, kind='single')['gpus'], 1)
        for other in ('probes/engine_kernel_check.py', 'probes/engine_full_check.py'):
            with self.subTest(probe=other), self.assertRaises(ValueError):
                self.validate(command[:2]+[other]+command[3:])
        with self.assertRaises(ValueError):
            self.validate(['bash', 'probes/run_engine_check.sh', '--router-only'])

    def test_retired_mixed_probes_are_not_admitted(self):
        for name in ('experts', 'completion', 'tickets'):
            probe = f'probes/engine_mixed_{name}_check.py'
            # Even an unchanged, executable file cannot re-open a retired probe.
            for tree in (self.repo, self.controller):
                (tree/probe).write_text('# retired experiment\n')
            with self.subTest(probe=probe), self.assertRaises(ValueError):
                self.validate(['bash', 'probes/run_engine_probe.sh', probe], kind='single')

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
        for command in (['bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'],
                        ['python3', 'bench/onepass.py']):
            with self.subTest(command=command):
                self.assertEqual(self.validate(command)['gpus'], 4)
                with self.assertRaisesRegex(ValueError, 'needs the four Sparks'):
                    self.validate(command, kind='single')
        # where a check runs is the lane's decision, never the command's
        with self.assertRaisesRegex(ValueError, 'ST_PROBE_HOST is set by the single-GPU lane'):
            self.validate(['env', 'ST_PROBE_HOST=srv2', 'bash', 'probes/run_engine_check.sh'])
        out = io.StringIO()
        with patch.dict(os.environ, self.environment, clear=True), contextlib.redirect_stdout(out):
            self.assertEqual(policy.main(['--repo', str(self.controller), '--cwd', str(self.repo),
                                          '--kind', 'single', '--', 'bash', 'probes/run_engine_check.sh']), 0)
        self.assertEqual(json.loads(out.getvalue())['gpus'], 1)

    def test_the_contract_carries_the_probe_s_own_memory_budget(self):
        """The queue exports it as ST_PROBE_GIB when the submitter set none (2026-09-13): a full-model probe asks
        for room for the rank file, a kernel check for a kernel check's 8, and nobody has to remember which."""
        def budget(*command):
            return self.validate(['bash', *command]).get('budget_gib')
        self.assertEqual(budget('probes/run_engine_probe.sh', 'probes/engine_prefill_chunk_profile.py', '--lanes', 'timeline'), 64)
        self.assertEqual(budget('probes/run_engine_probe.sh', 'probes/engine_graph_profile.py'), 64)
        self.assertEqual(budget('probes/run_engine_probe.sh', 'probes/engine_kernel_check.py', '--lanes', 'decode_rows'), 8)
        self.assertEqual(budget('probes/run_engine_check.sh', '--layers', '0-4'), 8)
        self.assertIsNone(budget('bench/st_bracket.sh', 'pair', '0123456789abcdef'))   # a fleet boot has no budget beside production
        self.assertTrue(set(policy.ST_PROBE_BUDGET_GIB) <= set(policy.ST_PROBES), "every budgeted probe is an admitted one")

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
        (self.repo / 'bench/st_bracket.sh').write_text('docker run --gpus all extra\n')
        with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
            self.validate(['bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'])

    def test_transitive_onepass_change_is_rejected(self):
        """The bracket's byte-pinned dependencies include the judge and the onepass it runs;
        a stale one in the candidate tree is not the canonical bytes."""
        for relative in ('bench/onepass.py', 'bench/st_judge.py'):
            with self.subTest(relative=relative):
                path = self.repo / relative
                original = path.read_bytes()
                path.write_bytes(original + b'# stale workload\n')
                with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
                    self.validate(['bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'])
                path.write_bytes(original)

    def test_env_prefixes_use_effective_values_and_verify_overrides(self):
        fleet = self.root / 'custom-fleet.sh'
        shutil.copyfile(self.controller / 'bench/fleet.sh', fleet)
        env = dict(self.environment, FLEET=str(self.controller / 'bench/fleet.sh'))
        self.validate(['env', '-u', 'LEGS', 'bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'],
                      environment=dict(env, LEGS='none'))
        self.validate(['env', '-i', 'REPO=' + str(self.repo), 'bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'])
        fleet.write_text('docker run --gpus all custom work\n')
        with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
            self.validate(['env', 'FLEET=' + str(fleet), 'bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'])
        with self.assertRaisesRegex(ValueError, 'injection setting'):
            self.validate(['env', 'BASH_ENV=/tmp/custom.sh', 'bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef'])

    def test_bracket_grammar_rejects_invented_arms_and_bad_shas(self):
        for command in (['bash', 'bench/st_bracket.sh'],
                        ['bash', 'bench/st_bracket.sh', 'pair'],
                        ['bash', 'bench/st_bracket.sh', 'pair', 'nothex'],
                        ['bash', 'bench/st_bracket.sh', 'chain'],
                        ['bash', 'bench/st_bracket.sh', 'deploy', '0123456789abcdef']):
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
                     ['--combined-reasoning-budget', str(COMBINED_MAX_TOKENS)],
                     ['--fixed-decode-tokens', '1', '--fixed-decode-reps', '0']):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.validate(['python3', 'bench/onepass.py', *args])

    def test_only_the_bracket_receives_cpu_rehearsal_exemption(self):
        prefix = ['--repo', str(self.controller), '--cwd', str(self.repo), '--rehearsal-only', '--']
        with patch.dict(os.environ, dict(self.environment, FLEET_REHEARSE='1'), clear=True), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(policy.main([*prefix, 'bash', 'bench/st_bracket.sh', 'pair', '0123456789abcdef']), 0)
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
