"""Default EP still needs execution evidence when its knob delta is empty."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import baseline
import onepass
import proof

EP = 'VLLM_GLM53_EP_TILED'
PREP = 'VLLM_GLM53_PREP_FUSED'


class DefaultEPProofTests(unittest.TestCase):
    def record(self):
        row = dict(knobs={})
        onepass._require_preparation(row)
        onepass._require_execution_proof(row, EP)
        return row

    def test_default_lanes_are_required_without_mislabeling_baseline(self):
        row = self.record()
        self.assertEqual(row['knobs'], {})
        self.assertEqual(baseline.is_baseline(row), (True, 'knobs'))
        self.assertEqual(onepass._execution_proof_knobs(row), [PREP, EP])
        self.assertEqual(row['proof_ok'], '0/2')
        self.assertFalse(baseline.proof_complete(row))
        onepass._require_observed_decode_opt(row, {'ep_decode_opt':'0'})
        self.assertEqual(row['required_proofs'], [PREP, EP])
        onepass._require_observed_decode_opt(row, {'ep_decode_opt':'1'})
        self.assertIn('VLLM_GLM53_EP_DECODE_OPT',row['required_proofs'])
        self.assertEqual(row['proof_ok'],'0/3')

    def test_requirements_merge_without_erasing_another_lanes_failure(self):
        row = dict(knobs={})
        onepass._require_execution_proof(row, EP)
        onepass._require_preparation(row)
        onepass._require_execution_proof(row, EP)
        self.assertEqual(row['required_proofs'], [EP, PREP])
        self.assertEqual(row['proof'], {EP: False, PREP: False})
        self.assertEqual(row['proof_ok'], '0/2')
        for failed in (EP, PREP):
            candidate = copy.deepcopy(row)
            candidate['proof'] = {EP: True, PREP: True}
            candidate['proof'][failed] = False
            self.assertFalse(baseline.proof_complete(candidate))
        row['proof'] = {EP: True, PREP: True}
        self.assertTrue(baseline.proof_complete(row))

    def test_candidate_and_explicit_disabled_required_lane_cannot_evade_proof(self):
        row = self.record()
        row['knobs'] = {EP: '1', 'VLLM_GLM53_EP_DECODE_OPT': '1'}
        onepass._require_observed_decode_opt(row, {'ep_decode_opt':'1'})
        self.assertEqual(row['required_proofs'],[PREP,EP])
        selected = onepass._execution_proof_knobs(row)
        self.assertEqual(selected.count(EP), 1)
        self.assertEqual(set(selected), {EP, PREP, 'VLLM_GLM53_EP_DECODE_OPT'})
        row['knobs'][EP] = '0'
        self.assertIn(EP, onepass._execution_proof_knobs(row))

    def test_missing_ep_canary_keeps_default_baseline_unproved(self):
        row = self.record()
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'head.log'
            log.write_text('[ep-tiled] armed\n')
            row.update(proof.check(onepass._execution_proof_knobs(row), str(log)))
        self.assertIs(row['proof'][EP], False)
        self.assertEqual(row['proof_ok'], '0/2')
        self.assertFalse(baseline.proof_complete(row))


if __name__ == '__main__':
    unittest.main()
