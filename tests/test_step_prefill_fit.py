"""Refit physical token/chunk costs, including partial tails and deficient data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import step_kernels as kernels


class PrefillFitTests(unittest.TestCase):
    def fit(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            path.write_text(json.dumps(data))
            return kernels.fold_prefill_from_profile(path)

    def test_existing_artifact_refit_recovers_its_stored_coefficients(self):
        data = json.loads((ROOT / 'measurements/c4_scaling_20260913/chunk-profile-rank3-sf6.json').read_text())
        stored = data.pop('prefill_fit')
        result = self.fit(data)
        self.assertAlmostEqual(result['ms_per_token'], stored['ms_per_token'], places=9)
        self.assertAlmostEqual(result['fixed_ms_per_chunk'], stored['fixed_ms_per_chunk'], places=8)
        self.assertLess(result['max_relative_residual'], .05)

    def test_partial_tail_steps_and_different_total_lengths_recover_two_costs(self):
        rows = []
        for steps in ([100, 100, 50], [500, 80], [200, 200, 200, 50]):
            rows.append(dict(chunks=len(steps), chunk=max(steps),
                             steps=[dict(tokens=t) for t in steps],
                             total_ms=.25*sum(steps) + 10*len(steps)))
        result = self.fit(dict(prefill=rows))
        self.assertAlmostEqual(result['ms_per_token'], .25)
        self.assertAlmostEqual(result['fixed_ms_per_chunk'], 10)
        # One fewer chunk saves exactly its fixed work, not prompt-token work.
        self.assertAlmostEqual(kernels.prefill_ms(1000, 250, result, fleet=False)
                               - kernels.prefill_ms(1000, 500, result, fleet=False), 20)

    def test_explicit_totals_and_legacy_full_chunks(self):
        for rows in ([dict(tokens=250, chunks=3, total_ms=92.5), dict(tokens=580, chunks=2, total_ms=165)],
                     [dict(chunk=100, chunks=3, total_ms=105), dict(chunk=500, chunks=2, total_ms=270)]):
            result = self.fit(dict(prefill=rows))
            self.assertAlmostEqual(result['ms_per_token'], .25)
            self.assertAlmostEqual(result['fixed_ms_per_chunk'], 10)

    def test_one_chunk_size_cannot_identify_two_coefficients(self):
        data = dict(prefill=[dict(chunk=100, chunks=n, total_ms=35*n) for n in (2, 3, 4)])
        with self.assertRaisesRegex(ValueError, 'cannot separate'):
            self.fit(data)

    def test_invalid_or_inconsistent_measurements_are_rejected(self):
        rows = [dict(tokens=250, chunks=3, total_ms=92.5), dict(tokens=580, chunks=2, total_ms=165)]
        for field, value in (('total_ms', -1), ('total_ms', float('nan')), ('total_ms', 0),
                             ('chunks', 0), ('tokens', 1.5)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.fit(dict(prefill=[{**rows[0], field: value}, rows[1]]))
        for bad in (dict(steps=[dict(tokens=100)]),
                    dict(steps=[dict(tokens=100)]*3)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.fit(dict(prefill=[{**rows[0], **bad}, rows[1]]))
        for fixed in (-1, float('inf')):
            with self.assertRaises(ValueError):
                self.fit(dict(prefill_fit=dict(ms_per_token=.25, fixed_ms_per_chunk=fixed)))


if __name__ == '__main__':
    unittest.main()
