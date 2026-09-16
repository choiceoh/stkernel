"""The dense cells probe's ksr arms: routes, plans and the 1-ulp gate (the m=16 split-K sweep, 2026-09-17)."""
import unittest

import torch

from probes import engine_dense_cells as probe


class DenseKsrArmTests(unittest.TestCase):
    def test_every_16_row_plan_pairs_its_served_route_with_each_slice_count(self):
        for cell in probe.CELLS:
            plan = cell[5]
            if 16 not in plan:
                continue
            served = plan[16][0][0]
            for n in probe.KSR_ARMS:
                self.assertIn((served, f'{served}_ksr{n}'), plan[16], cell[0])
                self.assertIn(f'{served}_ksr{n}', probe.ROUTES)
            if 8 in plan:
                self.assertFalse(any('_ksr' in b for _, b in plan[8]), cell[0])

    def test_gate_is_exact_for_served_arms_and_one_ulp_for_ksr_arms(self):
        # the ulp is that of the larger of the element and the tensor RMS: an add order moves a sum by a few
        # ulps of its partials' scale, which is the tensor's scale for a sum that cancels
        want = torch.full((8,), 1.0, dtype=torch.bfloat16)
        one_ulp = want.clone()
        one_ulp[3] = 1.0 + 2 ** -7
        tolerant = {}
        probe._gate('cell', 'bound', want.clone(), want, tolerant)
        with self.assertRaises(AssertionError):
            probe._gate('cell', 'bound', one_ulp, want, tolerant)
        probe._gate('cell', 'bound_ksr2', one_ulp, want, tolerant)
        self.assertAlmostEqual(tolerant['bound_ksr2'], 1.0, places=3)
        far = want.clone()
        far[3] = 1.0 + 2 ** -4
        with self.assertRaisesRegex(RuntimeError, 'bf16 ulps'):
            probe._gate('cell', 'bound_ksr2', far, want, tolerant)
        self.assertAlmostEqual(tolerant['bound_ksr2'], 8.0, places=3)
        # a cancelling sum is judged against the tensor RMS, not its own tiny value
        big = torch.full((8,), 100.0, dtype=torch.bfloat16)
        big[7] = 0.0
        tiny = big.clone()
        tiny[7] = 2 ** -20
        probe._gate('cell', 'bound_ksr4', tiny, big, tolerant)
        self.assertLess(tolerant['bound_ksr4'], 0.01)

if __name__ == '__main__':
    unittest.main()
