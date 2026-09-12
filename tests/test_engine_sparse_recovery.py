"""Numerical and data-separation contracts for the calibration-only probe."""
import unittest
import collections
from pathlib import Path
import tempfile

import torch

from probes.engine_sparse_nvfp4 import dequant, validate_sparse
from probes.engine_sparse_recovery import (choose_validation, hessian, inverse_factor,
                                          magnitude, pair_mask, sparsegpt_pair, wanda_pair)
from probes.engine_sparse_residual import apply_residual, choose_rank, fit_lowrank
from probes.engine_sparse_block_reconstruct import FP4StraightThrough, SparseExpert
from probes.engine_sparse_chain_residual import choose_deployed, fit_deployed_residual
from probes.engine_sparse_sequential_reconstruct import anchored_target
from probes.engine_sparse_compact_expert import quantize16_weight, select_groups
from probes.engine_sparse_joint_residual import JointResidual, eligible_score
from probes.engine_sparse_deneb_corpus import sanitize, text_content, window
from probes.engine_sparse_deneb_workloads import Components, mail_keys, phone_payload, read_rows, split_groups


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

    def test_deployed_residual_recovers_shifted_intermediate_mapping(self):
        torch.manual_seed(51)
        train, test = torch.randn(256, 16), torch.randn(64, 16)
        transform = torch.eye(16) + torch.randn(16, 16)*.08
        target_map = torch.randn(8, 16)
        source = train @ transform
        # Simulate an upstream change even though the down projection is exact.
        error = train @ target_map.T - source @ target_map.T
        b, a = fit_deployed_residual(source, error, torch.ones(256), 8, .0001, False)
        prediction = test @ transform @ target_map.T + apply_residual(test @ transform, b, a)
        target = test @ target_map.T
        self.assertLess(((prediction-target).norm()/target.norm()).item(), .01)

    def test_router_conditioning_folds_into_the_same_factor_shapes(self):
        torch.manual_seed(52)
        train, test = torch.randn(256, 16), torch.randn(64, 16)
        scales = torch.logspace(-2, 2, 16)
        weights = torch.randn(6, 16) / scales
        x = train * scales
        coefficient = torch.linspace(.01, 1, len(x))
        b, a = fit_deployed_residual(x, x @ weights.T, coefficient, 6, .0001, True)
        self.assertEqual(b.shape, (6, 6)); self.assertEqual(a.shape, (6, 16))
        expected = test * scales @ weights.T
        actual = apply_residual(test * scales, b, a)
        self.assertLess(((actual-expected).norm()/expected.norm()).item(), .015)

    def test_deployed_selection_uses_routed_validation_without_test(self):
        scores = {'independent': {'validation': {'weighted': {'relative_l2': .3}}, 'test': 0.},
                  'new': {'validation': {'weighted': {'relative_l2': .2}}, 'test': 1.}}
        self.assertEqual(choose_deployed(scores), 'new')

    def test_anchored_target_satisfies_regularized_normal_equations(self):
        torch.manual_seed(53)
        x, y, weight = torch.randn(80, 16), torch.randn(80, 8), torch.randn(8, 16)
        fitted = anchored_target(weight, x, y, .1)
        strength = .1 * (x.T @ x / len(x)).diagonal().mean()
        gradient = (x @ fitted.T-y).T @ x / len(x) + strength*(fitted-weight)
        self.assertLess(gradient.abs().max().item(), 2e-6)

    def test_anchored_target_preserves_an_already_correct_mapping(self):
        torch.manual_seed(54)
        x, weight = torch.randn(80, 16), torch.randn(8, 16)
        fitted = anchored_target(weight, x, x @ weight.T, .1)
        self.assertTrue(torch.equal(fitted, weight))

    def test_chunked_input_quantization_preserves_code_selection(self):
        from probes.engine_sparse_calibrate import inputs32
        from probes.engine_sparse_nvfp4_prune import quantize32
        torch.manual_seed(55)
        x = torch.randn(513, 64)
        self.assertTrue(torch.equal(inputs32(x), dequant(*quantize32(x))))

    def test_compact_scale_search_cannot_worsen_max_over_six_rounding(self):
        from probes.engine_sparse_recovery import _encode_scaled
        torch.manual_seed(57)
        weight = torch.randn(32, 64)*.2
        blocks = weight.reshape(32, 4, 16)
        scale = (blocks.abs().amax(-1)/6).to(torch.float8_e4m3fn)
        baseline = _encode_scaled(blocks, scale.float()[...,None])[0]
        decoded = dequant(*quantize16_weight(weight)).reshape_as(blocks)
        self.assertTrue(((decoded-blocks).square().sum(-1) <=
                         (baseline-blocks).square().sum(-1)+1e-8).all())

    def test_compact_selection_keeps_complete_original_k16_groups(self):
        weight = torch.arange(1, 33).float().repeat_interleave(16)[None].repeat(8,1)
        for mode in ('energy','conditional'):
            chosen = select_groups(weight, torch.eye(512), mode)
            self.assertTrue(torch.equal(chosen, torch.arange(256,512)))

    def test_joint_factor_export_keeps_forward_and_frozen_sparse_weights(self):
        torch.manual_seed(58)
        sparse={'w13':magnitude(torch.randn(64,32)*.2)[0],
                'w2':magnitude(torch.randn(32,32)*.2)[0]}
        factors={'w13':(torch.randn(64,4).bfloat16()*.02,torch.randn(4,32).bfloat16()),
                 'w2':(torch.randn(32,4).bfloat16()*.02,torch.randn(4,32).bfloat16())}
        model=JointResidual(sparse,factors);x=torch.randn(8,32).bfloat16()
        model(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(model.w13.requires_grad);self.assertFalse(model.w2.requires_grad)
        self.assertTrue(torch.equal(model(x),JointResidual(sparse,model.export())(x)))

    def test_language_guard_rejects_a_tradeoff_hidden_by_the_average(self):
        initial={'ko':{'relative_l2':.2},'en':{'relative_l2':.4}}
        worse={'ko':{'relative_l2':.21},'en':{'relative_l2':.1}}
        better={'ko':{'relative_l2':.19},'en':{'relative_l2':.3}}
        self.assertEqual(eligible_score(worse,initial),float('inf'))
        self.assertLess(eligible_score(better,initial),1.)

    def test_private_corpus_excludes_thinking_and_tool_payloads(self):
        content=[{'type':'thinking','thinking':'hidden reasoning'},
                 {'type':'tool_result','content':'private tool body'},
                 {'type':'text','text':'visible request'}]
        self.assertEqual(text_content(content),'visible request')

    def test_private_corpus_redacts_credential_and_contact_patterns(self):
        text='Bearer '+('a'*32)+' email=person@example.test phone=010-5555-6666 password=example-password'
        result,count=sanitize(text)
        self.assertGreaterEqual(count,4)
        for value in ['a'*32,'person@example.test','010-5555-6666','example-password']:
            self.assertNotIn(value,result)

    def test_private_window_preserves_the_current_request_without_mutation(self):
        history=[{'role':'assistant','content':'a'*3000} for _ in range(6)]
        history.append({'role':'user','content':'current request'})
        result=window(history)
        self.assertEqual(result[-1],history[-1])
        self.assertLessEqual(sum(len(m['content']) for m in result),8000)
        self.assertEqual(len(history),7)

    def test_workload_mail_references_and_reply_subjects_share_a_group(self):
        components = Components()
        original = mail_keys(dict(message_id='<a@example.test>', subject='Original'))
        reply = mail_keys(dict(message_id='<b@example.test>', references=['<a@example.test>'], subject='Re: Changed'))
        forwarded = mail_keys(dict(message_id='<c@example.test>', subject='FW: Re: Changed'))
        for keys in (original, reply, forwarded):
            components.link(keys)
        self.assertEqual(len({components.root(keys[0]) for keys in (original,reply,forwarded)}),1)

    def test_phone_payload_does_not_treat_common_guidance_as_an_event(self):
        text='[실시간 스마트폰 이벤트 — 앱 알림]\n출처: business room\n내용:\nstatus changed\n\n위는 사용자 스마트폰에서 방금 발생한 이벤트다.\nshared guidance'
        self.assertEqual(phone_payload(text), ('business room','status changed'))
        self.assertIsNone(phone_payload('some unrelated log message'))

    def test_workload_splits_keep_every_retry_with_its_source(self):
        rows=[dict(group=f'g{i}', attempt=j) for i in range(30) for j in range(i%4+1)]
        split_groups(rows)
        for group in {r['group'] for r in rows}:
            self.assertEqual(len({r['split'] for r in rows if r['group']==group}),1)
        self.assertEqual({r['split'] for r in rows},{'train','validation','test'})

    def test_workload_snapshot_counts_corrupt_records_and_live_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log.jsonl'
            raw=b'{"type":"run.start"}\n\x00\xff\n{"incomplete":'
            path.write_bytes(raw)
            inventory=collections.Counter()
            self.assertEqual(list(read_rows(path,inventory)),[(0,{'type':'run.start'})])
            self.assertEqual(inventory['malformed_records_excluded'],1)
            self.assertEqual(inventory['incomplete_tail'],1)
            self.assertEqual(path.read_bytes(),raw)


if __name__ == '__main__':
    unittest.main()
