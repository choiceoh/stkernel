"""Changing real source changes the forecast; source drift cannot reuse a profile."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import step_source as source


def edit(tree, path, before, after):
    data = tree.files[path].decode()
    if before not in data:
        raise AssertionError(f'test source anchor missing: {path}: {before}')
    return source.Source('edited', tree.commit, {**tree.files, path: data.replace(before, after, 1).encode()}, True)


def constant(tree, path, name, value):
    data, count = re.subn(rf'(?m)^({name}\s*=\s*)[^#\n]+', lambda m: m[1] + repr(value) + ' ', tree.files[path].decode())
    if count != 1:
        raise AssertionError(f'test expected one {name} assignment')
    return source.Source('edited', tree.commit, {**tree.files, path: data.encode()}, True)


class SourcePredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = source.Source.read(ROOT, 'HEAD')

    def compare(self, candidate, base=None, **kwargs):
        return source.compare(base or self.base, candidate, contexts=(32000,), widths=(1,), **kwargs)

    def test_actual_boot_contract_and_layout_not_historical_constants(self):
        p = self.base.inspect()
        k = int(re.search(r'(?m)^SPEC_K\s*=\s*(\d+)', self.base.files[source.FACTS_PATH].decode())[1])
        self.assertEqual(p['facts']['spec_k'], k)
        self.assertEqual(p['prefill_chunk'], 32256)
        self.assertEqual(p['contract']['decode_token_budget'], 2304 + k)
        self.assertEqual(p['execution']['decode_iterations'], 4)
        self.assertEqual(p['memory']['state_fields_bytes']['rec'], 34 * (k + 1) * 16 * 128 * 128 * 4)
        self.assertEqual(p['memory']['resident_slots_bytes'], 5 * p['memory']['slot_bytes'])

    def test_declaration_follows_imported_policy_from_each_source(self):
        policy = 'engine/profiles/glm53/draft_policy.py'
        old = constant(self.base, policy, 'SERVING_POLICY', None)
        old = edit(old, policy, 'SERVING_POLICY = None', "SERVING_POLICY = DraftPolicy('w4', 'shared', False)")
        candidate = edit(old, policy, "DraftPolicy('w4', 'shared', False)", "DraftPolicy('fp8', 'auto', True)")
        result = self.compare(candidate, base=old)
        self.assertEqual(result['base']['settings']['draft_fc_precision'], 'w4')
        self.assertEqual(result['candidate']['settings']['draft_fc_precision'], 'fp8')
        self.assertEqual(result['candidate']['settings']['draft_fc_calibration'], 'auto')
        self.assertEqual(result['candidate']['settings']['draft_diagnostics'], 1)
        self.assertNotEqual(result['base']['fingerprint'], result['candidate']['fingerprint'])
        self.assertIsNone(result['forecasts'][0]['decode']['delta'])

    def test_equal_code_across_refs_has_equal_fingerprint_and_zero_model_delta(self):
        other = source.Source('working-tree', 'another-commit-name', dict(self.base.files), True)
        result = self.compare(other)
        self.assertEqual(result['base']['fingerprint'], result['candidate']['fingerprint'])
        self.assertEqual(result['forecasts'][0]['decode']['delta'], 0)
        self.assertEqual(result['changed_files'], [])

    def test_uncommitted_budget_change_updates_prefill_prediction(self):
        candidate = constant(self.base, source.BOOT_PATH, 'TOKEN_BUDGET', 8192)
        result = self.compare(candidate)
        self.assertEqual(result['candidate']['prefill_chunk'], 6912)
        prefill = result['forecasts'][0]['prefill']
        self.assertGreater(prefill['candidate_ms'], prefill['base_ms'])
        self.assertGreater(prefill['delta'], 0)
        self.assertEqual(result['forecasts'][0]['decode']['modeled_delta'], 0)
        self.assertEqual(result['changed_files'][0]['handling'], 'source_derived')

    def test_changed_chunk_function_is_executed_not_just_its_constants(self):
        candidate = edit(self.base, 'engine/base/shapes.py',
                         'return (usable // align) * align', 'return min(usable // align, 2) * align')
        result = self.compare(candidate)
        self.assertEqual(result['candidate']['prefill_chunk'], 4608)
        self.assertGreater(result['forecasts'][0]['prefill']['modeled_delta'], 0)
        self.assertIsNone(result['forecasts'][0]['prefill']['delta'], 'scheduler code changed beyond the timing model')

    def test_removing_real_layout_scratch_changes_bytes_without_fabricating_speed(self):
        before = edit(self.base, 'engine/profiles/glm53/caches.py',
                      '# A KDA-only slice still has logical blocks',
                      'field("unused_scratch", -1, (512,), "bf16")\n    # A KDA-only slice still has logical blocks')
        result = self.compare(self.base, base=before)
        self.assertEqual(result['memory_delta']['slot_bytes'], -1024)
        self.assertEqual(result['memory_delta']['resident_slots_bytes'], -5120)
        self.assertEqual(result['forecasts'][0]['decode']['modeled_delta'], 0)
        self.assertIsNone(result['forecasts'][0]['decode']['delta'])

    def test_k_changes_verification_cost_and_exact_recurrent_allocation(self):
        candidate = constant(self.base, source.FACTS_PATH, 'SPEC_K', 3)
        result = self.compare(candidate)
        self.assertEqual(result['candidate']['contract']['draft_slots'], 3)
        self.assertEqual(result['candidate']['memory']['state_fields_bytes']['rec'], 34 * 4 * 16 * 128 * 128 * 4)
        self.assertLess(result['forecasts'][0]['decode']['modeled_delta'], 0)
        self.assertIn('drafter', result['forecasts'][0]['decode']['unpriced_components'])

    def test_prefix_histogram_bounds_new_depth_instead_of_assuming_equal_raw_rate(self):
        base = constant(self.base, source.FACTS_PATH, 'SPEC_K', 6)
        candidate = constant(base, source.FACTS_PATH, 'SPEC_K', 7)
        observed = dict(k=6, histogram=[20, 10, 10, 10, 10, 20, 20])
        result = self.compare(candidate, base=base, acceptance_profile=observed)
        rates = result['forecasts'][0]['decode']['output_rate_assumption']
        self.assertAlmostEqual(rates['base']['tokens_per_row'], 4.2)
        self.assertIsNone(rates['candidate']['tokens_per_row'])
        self.assertIsNone(rates['candidate']['per_request_tok_s'])
        self.assertAlmostEqual(rates['candidate']['tokens_per_row_range'][0], 4.2)
        self.assertAlmostEqual(rates['candidate']['tokens_per_row_range'][1], 4.4)
        self.assertEqual(result['acceptance_scenario']['rows'], 100)
        self.assertIn('미계측', source.format_comparison(result))
        mean_only = self.compare(candidate, base=base)['forecasts'][0]['decode']['output_rate_assumption']
        self.assertIsNone(mean_only['candidate']['per_request_tok_s'])
        self.assertAlmostEqual(mean_only['candidate']['tokens_per_row_range'][0], 3.7)
        self.assertAlmostEqual(mean_only['candidate']['tokens_per_row_range'][1], 4.15)

    def test_source_cli_acceptance_input_is_a_validated_explicit_scenario(self):
        from test_step_economics import write_peek
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'peek.jsonl'
            write_peek(path)
            command = [sys.executable, str(ROOT / 'bench/storacle.py'), 'predict', '--base', 'HEAD',
                       '--ctx', '32000', '--width', '1', '--acceptance-from', str(path), '--json']
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(run.stdout)
            self.assertEqual(data['acceptance_scenario']['k'], 6)
            self.assertIn('explicit scenario', data['acceptance_scenario']['receipt']['usage'])
            path.write_text('{}\n')
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertIn('error', json.loads(run.stdout))

    def test_recipe_override_follows_boot_plan_and_changes_fingerprint(self):
        result = self.compare(self.base, settings={'prefill_tiles': 2})
        self.assertEqual(result['candidate']['prefill_chunk'], 64512)
        self.assertEqual(result['candidate']['compute_tile'], 32256)
        self.assertNotEqual(result['base']['fingerprint'], result['candidate']['fingerprint'])
        self.assertIsNone(result['forecasts'][0]['prefill']['delta'])
        for settings in ({'direct_mhc': 2}, {'prefill_tiles': 3}, {'nonexistent': 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.base.inspect(settings)

    def test_comments_do_not_create_a_kernel_cost_change(self):
        candidate = edit(self.base, source.FACTS_PATH, 'SPEC_K =', '# oracle comment\nSPEC_K =')
        result = self.compare(candidate)
        self.assertEqual(result['changed_files'][0]['handling'], 'cosmetic')
        self.assertEqual(result['forecasts'][0]['decode']['delta'], 0)

    def changed_kernel(self):
        return edit(self.base, 'engine/kernels/common/decode_commit.py',
                    'import triton', 'ORACLE_UNPRICED_TEST = 1\nimport triton')

    def profile(self, result, all_components=True):
        costs = result['forecasts'][0]['decode']['components']['base']
        pairs = {name: dict(base_ms=ms, candidate_ms=ms-5 if name == 'nonmoe' else ms, samples=10)
                 for name, ms in costs.items() if all_components or name == 'nonmoe'}
        return dict(schema=1, base=result['base']['fingerprint'], candidate=result['candidate']['fingerprint'],
                    runtime='synthetic test fixture, not a GPU measurement', evidence='unit-test paired costs',
                    rows=[dict(phase='decode', ctx=32000, width=1, components=pairs)])

    def test_source_bound_paired_component_timings_replace_changed_kernel_costs(self):
        candidate = self.changed_kernel()
        result = self.compare(candidate)
        self.assertIsNone(result['forecasts'][0]['decode']['delta'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pair.json'
            path.write_text(json.dumps(self.profile(result)))
            priced = self.compare(candidate, profile=path)['forecasts'][0]['decode']
            self.assertAlmostEqual(priced['candidate_ms'], priced['base_ms']-5)
            self.assertLess(priced['delta'], 0)
            self.assertFalse(priced['unpriced_components'])
            path.write_text(json.dumps(self.profile(result, all_components=False)))
            partial = self.compare(candidate, profile=path)['forecasts'][0]['decode']
            self.assertAlmostEqual(partial['candidate_ms'], partial['base_ms']-5)
            self.assertIsNone(partial['delta'])

    def test_stale_or_unmeasured_profiles_cannot_turn_into_predictions(self):
        candidate = self.changed_kernel()
        result = self.compare(candidate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pair.json'
            data = self.profile(result)
            path.write_text(json.dumps(source.profile_template(result)))
            with self.assertRaises(ValueError):
                self.compare(candidate, profile=path)
            for key, value in (('candidate', 'old-source'), ('runtime', ''), ('evidence', ''), ('rows', [])):
                with self.subTest(key=key):
                    path.write_text(json.dumps({**data, key: value}))
                    with self.assertRaises(ValueError):
                        self.compare(candidate, profile=path)
            for field, value in (('samples', 0), ('base_ms', float('nan')), ('candidate_ms', -1)):
                broken = self.profile(result)
                broken['rows'][0]['components']['nonmoe'][field] = value
                path.write_text(json.dumps(broken))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.compare(candidate, profile=path)
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'fingerprints'):
                self.compare(constant(candidate, source.FACTS_PATH, 'SPEC_K', 3), profile=path)

    def test_source_contract_refuses_unavailable_widths(self):
        candidate = constant(self.base, source.BOOT_PATH, 'MAX_SEQS', 1)
        result = source.compare(self.base, candidate, contexts=(32000,), widths=(1, 4))
        self.assertIn('decode', result['forecasts'][0])
        self.assertIn('unavailable', result['forecasts'][1])

    def test_git_reader_and_cli_include_staged_unstaged_and_untracked_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in self.base.files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            source.git(root, 'init', '-q')
            source.git(root, 'add', 'engine')
            source.git(root, '-c', 'user.name=Oracle Test', '-c', 'user.email=oracle@example.invalid', 'commit', '-qm', 'baseline')
            budget = constant(self.base, source.BOOT_PATH, 'TOKEN_BUDGET', 8192)
            (root / source.BOOT_PATH).write_bytes(budget.files[source.BOOT_PATH])
            source.git(root, 'add', source.BOOT_PATH)  # staged
            (root / source.FACTS_PATH).write_bytes(self.base.files[source.FACTS_PATH] + b'\n# unstaged\n')
            (root / 'engine/oracle_untracked.py').write_text('NEW_COST = 1\n')
            before = source.git(root, 'status', '--porcelain')
            cmd = [sys.executable, str(ROOT / 'bench/storacle.py'), 'predict', '--base=HEAD',
                   '--tree', str(root), '--ctx', '32000', '--width', '1', '--json']
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            data = json.loads(run.stdout)
            self.assertEqual(data['candidate']['prefill_chunk'], 6912)
            self.assertEqual({r['path'] for r in data['changed_files']},
                             {source.BOOT_PATH, source.FACTS_PATH, 'engine/oracle_untracked.py'})
            self.assertEqual(before, source.git(root, 'status', '--porcelain'), 'comparison must leave the source tree alone')


if __name__ == '__main__':
    unittest.main()
