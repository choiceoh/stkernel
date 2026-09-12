"""Numerical and data-separation contracts for the calibration-only probe."""
import unittest

import torch

from probes.engine_sparse_nvfp4 import dequant, validate_sparse
from probes.engine_sparse_recovery import (choose_validation, hessian, inverse_factor,
                                          magnitude, pair_mask, sparsegpt_pair, wanda_pair)
from probes.engine_sparse_residual import apply_residual, choose_rank, fit_lowrank
from probes.engine_sparse_block_reconstruct import FP4StraightThrough, SparseExpert


class SparseRecoveryTests(unittest.TestCase):
    def test_inverse_factor_matches_regularized_system(self):
        torch.manual_seed(6)
        x = torch.randn(96, 32)
        x[:, 7] = 0
        h = hessian(x)
        upper, dead = inverse_factor(h, .1)
        expected = h.clone()
        expected.diagonal()[dead] = 1
        expected.diagonal().add_(.1 * expected.diagonal().mean())
        self.assertTrue(dead[7])
        torch.testing.assert_close((upper.T @ upper) @ expected, torch.eye(32), atol=2e-6, rtol=2e-6)

    def test_diagonal_hessian_matches_magnitude_with_quantization(self):
        torch.manual_seed(7)
        w = torch.randn(16, 64) * .25
        result, packed, sf = sparsegpt_pair(w, torch.eye(64), .1)
        reference, _, _ = magnitude(w)
        self.assertTrue(torch.equal(result, reference))
        self.assertTrue(torch.equal(result, dequant(packed, sf)))
        validate_sparse(packed)

    def test_masks_keep_whole_pairs_and_activation_importance(self):
        w = torch.tensor([[4., 4., 3., 3., 2., 2., 1., 1.]]).repeat(4, 4)
        x = torch.ones(100, 32)
        x[:, 4::8] = x[:, 5::8] = 100
        result, packed, _ = wanda_pair(w, x)
        validate_sparse(packed)
        active = result.reshape(4, 4, 4, 2).ne(0).any(-1)
        self.assertTrue(active[..., 0].all())
        self.assertTrue(active[..., 2].all())
        self.assertFalse(active[..., 1].any())
        self.assertFalse(active[..., 3].any())

    def test_correlated_hessian_changes_export_and_keeps_legal_pattern(self):
        torch.manual_seed(17)
        x = torch.randn(128, 32)
        x[:, 8:16] = x[:, :8] + .01 * x[:, 8:16]
        w = torch.randn(16, 32) * .2
        corrected, packed, sf = sparsegpt_pair(w, hessian(x), .01)
        uncorrected, _, _ = magnitude(w)
        validate_sparse(packed)
        self.assertTrue(torch.equal(corrected, dequant(packed, sf)))
        self.assertLess((x @ (corrected-w).T).norm(), (x @ (uncorrected-w).T).norm())

    def test_zero_weights_and_unseen_inputs(self):
        result, packed, sf = sparsegpt_pair(torch.zeros(8, 32), torch.zeros(32, 32), .1)
        self.assertEqual(result.count_nonzero().item(), 0)
        self.assertEqual(packed.count_nonzero().item(), 0)
        self.assertEqual(sf.count_nonzero().item(), 0)

    def test_validation_selection_does_not_use_test(self):
        candidates = {'a': {'validation': {'relative_l2': .2}, 'test': {'relative_l2': .01}},
                      'b': {'validation': {'relative_l2': .1}, 'test': {'relative_l2': .9}}}
        self.assertEqual(choose_validation(candidates), 'b')

    def test_invalid_numerical_inputs_rejected(self):
        with self.assertRaises(ValueError):
            sparsegpt_pair(torch.ones(4, 16), torch.eye(16))
        with self.assertRaises(ValueError):
            sparsegpt_pair(torch.ones(4, 32), torch.eye(16))
        with self.assertRaises(ValueError):
            inverse_factor(torch.eye(32), 0)
        with self.assertRaises(ValueError):
            pair_mask(torch.full((4, 32), float('nan')))
        with self.assertRaises(ValueError):
            hessian(torch.empty(0, 32))

    def test_lowrank_residual_recovers_known_mapping_on_unseen_inputs(self):
        torch.manual_seed(42)
        train, test = torch.randn(256, 32), torch.randn(64, 32)
        a, b = torch.randn(2, 32), torch.randn(16, 2)
        target = train @ a.T @ b.T
        fitted_b, fitted_a = fit_lowrank(train, target, max_rank=2, damping=.0001)
        expected = test @ a.T @ b.T
        prediction = apply_residual(test, fitted_b, fitted_a)
        self.assertLess(((prediction-expected).norm()/expected.norm()).item(), .01)
        self.assertEqual(fitted_a.dtype, torch.bfloat16)

    def test_full_rank_residual_matches_ridge_solution(self):
        torch.manual_seed(19)
        x, y = torch.randn(64, 8), torch.randn(64, 4)
        b, a = fit_lowrank(x, y, max_rank=4, damping=.1)
        h = x.T @ x / len(x)
        h.diagonal().add_(.1 * h.diagonal().mean())
        expected = torch.linalg.solve(h, x.T @ y / len(x)).T
        torch.testing.assert_close(b.float() @ a.float(), expected, atol=.002, rtol=.015)

    def test_residual_rank_selected_without_test_scores(self):
        scores = {16: {'validation': {'relative_l2': .3}, 'test': {'relative_l2': .01}},
                  32: {'validation': {'relative_l2': .2}, 'test': {'relative_l2': .9}}}
        self.assertEqual(choose_rank(scores), 32)

    def test_fp4_ste_forward_is_exact_and_backward_passes_gradient(self):
        torch.manual_seed(31)
        x = torch.randn(4, 32, requires_grad=True)
        y = FP4StraightThrough.apply(x)
        from probes.engine_sparse_nvfp4_prune import quantize32
        self.assertTrue(torch.equal(y, dequant(*quantize32(x.detach()))))
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_joint_block_keeps_masks_and_exported_forward(self):
        torch.manual_seed(38)
        w13 = magnitude(torch.randn(64, 32)*.2)
        w2 = magnitude(torch.randn(32, 32)*.2)
        model = SparseExpert({'w13':w13[1:], 'w2':w2[1:]})
        x = torch.randn(8, 32)
        model(x).square().mean().backward()
        for name in ('w13', 'w2'):
            p, mask = getattr(model, name), getattr(model, name+'_mask')
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertEqual(p.grad[~mask].count_nonzero().item(), 0)
            self.assertGreater(p.grad[mask].count_nonzero().item(), 0)
            with torch.no_grad():
                p.add_(-.01*p.grad)
        exported = model.export()
        restored = SparseExpert(exported)
        self.assertTrue(torch.equal(model(x), restored(x)))
        for packed, _ in exported.values():
            validate_sparse(packed)

    def test_block_preserves_subnormal_scales_without_requantizing_them(self):
        packed = torch.tensor([0x55, 0x55, 0, 0], dtype=torch.uint8).repeat(64, 4)
        scales = torch.ones((64, 1), dtype=torch.uint8)  # minimum positive E4M3
        model = SparseExpert({'w13': (packed, scales), 'w2': (packed[:32], scales[:32])})
        expected = dequant(packed, scales)
        self.assertTrue(torch.equal(model.quantized_weight('w13'), expected))
        self.assertTrue(torch.equal(dequant(*model.export()['w13']), expected))


if __name__ == '__main__':
    unittest.main()
