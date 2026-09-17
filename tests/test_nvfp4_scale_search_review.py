"""Independent numerical boundaries for the offline adoption study."""
import math
import random
import unittest

from probes.nvfp4_scale_search_review import (
    FP4, SCALES, analyse, compare_block, decode_scale, nearest_even,
)


class ScaleSearchReviewTests(unittest.TestCase):
    def test_scale_encoding_boundaries(self):
        self.assertEqual([decode_scale(i) for i in (0, 1, 7, 8, 56, 126)],
                         [0, 2**-9, 7 * 2**-9, 2**-6, 1, 448])
        for bad in (-1, 127):
            with self.assertRaises(ValueError):
                decode_scale(bad)

    def test_fp4_ties_round_to_even_code(self):
        self.assertEqual([nearest_even(x, FP4) for x in (.25, .75, 1.25, 1.75, 2.5, 3.5, 5)],
                         [0, 2, 2, 4, 4, 6, 6])

    def test_e4m3_ties_at_exponent_transition(self):
        self.assertEqual(nearest_even((SCALES[55] + SCALES[56]) / 2, SCALES), 56)
        self.assertEqual(nearest_even((SCALES[56] + SCALES[57]) / 2, SCALES), 56)
        self.assertEqual(nearest_even(1e20, SCALES), 126)

    def test_zero_preserves_base_and_sign(self):
        base, new = compare_block([0., -0.] * 8)
        self.assertEqual(base, new)
        self.assertEqual(new['code'], 0)
        self.assertEqual(new['nibbles'], (0, 8) * 8)

    def test_exact_values_keep_baseline(self):
        base, new = compare_block(list(FP4) + [-v for v in FP4])
        self.assertEqual(base, new)
        self.assertEqual(new['sse'], 0)

    def test_sse_can_improve_while_maximum_error_worsens(self):
        base, new = compare_block([.85] * 15 + [6.])
        self.assertLess(new['sse'], base['sse'])
        self.assertGreater(new['max_abs'], base['max_abs'])
        self.assertEqual(new['code'], base['code'] - 1)

    def test_projection_error_can_worsen_despite_better_activation_sse(self):
        # A projection which reads the outlier exactly reverses the preference.
        values = [.85] * 15 + [6.]
        base, new = compare_block(values)
        self.assertEqual(base['restored'][-1], values[-1])
        self.assertNotEqual(new['restored'][-1], values[-1])

    def test_global_scale_is_multiplier_and_power_of_two_equivariant(self):
        values = [.85] * 15 + [6.]
        for gs in (2**-8, 1., 2**8):
            base, new = compare_block([v * gs for v in values], gs)
            ref_base, ref_new = compare_block(values)
            self.assertEqual((base['code'], base['nibbles']), (ref_base['code'], ref_base['nibbles']))
            self.assertEqual((new['code'], new['nibbles']), (ref_new['code'], ref_new['nibbles']))

    def test_candidates_include_baseline_and_stay_finite(self):
        rng = random.Random(70)
        for amplitude in (2**-20, 1., 2**20):
            for _ in range(20):
                base, new = compare_block([rng.gauss(0, 1) * amplitude for _ in range(16)])
                self.assertLessEqual(new['sse'], base['sse'])
                self.assertLessEqual(abs(new['code'] - base['code']), 2)
                self.assertTrue(all(math.isfinite(v) for v in new['restored']))

    def test_search_radii_form_nested_candidate_sets(self):
        values = [.85] * 15 + [6.]
        base, same = compare_block(values, radius=0)
        _, narrow = compare_block(values, radius=1)
        _, wide = compare_block(values, radius=2)
        self.assertEqual(base, same)
        self.assertLessEqual(wide['sse'], narrow['sse'])
        self.assertLessEqual(narrow['sse'], base['sse'])
        with self.assertRaises(ValueError):
            compare_block(values, radius=3)

    def test_rejects_nonfinite_invalid_shape_and_global_scales(self):
        for values, gs in (([0.] * 15, 1), ([float('nan')] * 16, 1),
                           ([float('inf')] * 16, 1), ([1.] * 16, 0),
                           ([1.] * 16, -1), ([1.] * 16, float('nan'))):
            with self.assertRaises(ValueError):
                compare_block(values, gs)

    def test_report_does_not_hide_worse_maximum_error(self):
        result = analyse([[.85] * 15 + [6.]], [1.])
        self.assertEqual(result['improved_blocks'], 1)
        self.assertEqual(result['worsened_sse_blocks'], 0)
        self.assertEqual(result['worsened_max_abs_blocks'], 1)


if __name__ == '__main__':
    unittest.main()
