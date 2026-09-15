"""Boundary decisions and view addressing against independent mathematical outcomes."""
import unittest

import torch

from engine.base import draws
from engine.base.sampler import (
    _block_verify_by_torch, _rows_by_sorting, block_verify, block_verify_batch, rows,
)


class VerificationBoundaryTests(unittest.TestCase):
    def test_hash_generated_zero_cannot_accept_impossible_drafts(self):
        seed, k = 6754985298932926379, 7
        key = draws.row_key(seed, 0, 0)
        host = [draws.uniform(key, purpose, position) for purpose, position in draws.step_layout(k)]
        self.assertEqual(host[2*k-1], 0.)
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            with self.subTest(device=device):
                zero = torch.zeros(1, dtype=torch.int64, device=device)
                u = draws.step_block(seed, zero, zero, k)[:, k:]
                self.assertEqual(u[0].tolist(), host[k:])
                target = torch.zeros(1, k+1, 17, device=device)
                target[..., 0] = 1
                draft = torch.ones(1, k, dtype=torch.int64, device=device)
                cand = torch.arange(1, 17, device=device).view(1, 1, 16).repeat(1, k, 1)
                q = torch.zeros(1, k, 16, device=device)
                q[..., 0] = 1
                for verify in (block_verify_batch, _block_verify_by_torch):
                    accepted, tokens, count = verify(target, draft, cand, q, u)
                    self.assertEqual(accepted.tolist(), [0])
                    self.assertEqual(tokens[:, 0].tolist(), [0])
                    self.assertEqual(count.tolist(), [1])
                dense_q = torch.zeros(k, 17, device=device)
                dense_q[:, 1] = 1
                self.assertEqual(block_verify(target[0], draft[0].tolist(), dense_q, u[0]), (0, [0]))

    def test_single_draft_acceptance_uses_a_half_open_interval(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            q = torch.tensor([[[.5, .5]]], device=device)
            cand = torch.tensor([[[0, 1]]], device=device)
            draft = torch.zeros(1, 1, dtype=torch.int64, device=device)
            for p, uniform, expected in ((0., 0., 0), (.25, .5, 0), (.25, .49, 1), (1., 1.-2**-24, 1)):
                target = torch.tensor([[[p, 1-p], [1., 0.]]], device=device)
                u = torch.tensor([[uniform, .3]], device=device)
                self.assertEqual(block_verify(target[0], [0], q[0], u[0])[0], expected)
                for verify in (block_verify_batch, _block_verify_by_torch):
                    self.assertEqual(verify(target, draft, cand, q, u)[0].item(), expected)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class KernelViewTests(unittest.TestCase):
    def test_native_router_preparation_allocates_nothing_and_preserves_routes(self):
        from types import SimpleNamespace as NS
        from engine.kernels.glm_pointwise import router_logits, route_weights
        from engine.profiles.glm53.net import Glm53Net
        net = object.__new__(Glm53Net)
        net.F = NS(experts=288, hidden=4096, is_moe=lambda layer: True, topk_experts=8, routed_scale=2.5)
        net.layers = [3]
        net.lanes = NS(route_weights=route_weights)
        net._router_layers, net._router_tensorcore = None, set()
        generator = torch.Generator(device='cuda').manual_seed(91523)
        net.p = {'L3.moe.gate': (torch.randn(288, 4096, device='cuda', generator=generator)*.02).bfloat16(),
                 'L3.moe.bias': torch.zeros(288, device='cuda')}
        x = torch.randn(8, 4096, device='cuda', generator=generator).bfloat16()
        expected = route_weights(router_logits(x, net.p['L3.moe.gate']), net.p['L3.moe.bias'], 8, 2.5)
        before = torch.cuda.memory_allocated()
        net.prepare_routers()
        self.assertEqual(torch.cuda.memory_allocated(), before)
        for got, want in zip(net.route(3, x), expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(net._router_tensorcore, net._router_layers)

    def test_mixed_greedy_matches_vocabulary_argmax_for_zeros_and_nans(self):
        from engine.modules.vocab import argmax

        class OneRank:
            def all_reduce_max(self, value):
                return value

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.tensor([[-0., +0., -1.], [-float('nan'), float('nan'), 0.],
                                   [-0., +0., +0.], [0., -1., -2.]], device='cuda', dtype=dtype)
            t = torch.tensor([0., 0., 0., 1.], device='cuda')
            k, p, u = (torch.zeros(4, device='cuda', dtype=torch.int32),
                       torch.ones(4, device='cuda'), torch.full((4,), .5, device='cuda'))
            got = rows(logits, t, k, p, u)
            self.assertEqual(got[:3].tolist(), [0, 0, 0])
            self.assertTrue(torch.equal(got[:3], argmax(logits[:3], OneRank(), 0)))
            self.assertTrue(torch.equal(got, _rows_by_sorting(logits, t, k, p, u, None, None)))

    def test_each_sampler_policy_view_is_independent_of_padding_on_replay(self):
        logits = torch.tensor([[0., -.3, -1.]]*2, device='cuda')
        for field, poison in enumerate((0., 1, .1, 0.)):
            policy = [torch.ones(2, device='cuda'), torch.zeros(2, device='cuda', dtype=torch.int32),
                      torch.ones(2, device='cuda'), torch.full((2,), .9, device='cuda')]
            backing = torch.zeros(4, dtype=policy[field].dtype, device='cuda')
            backing[::2] = policy[field]
            policy[field] = backing[::2]
            rows(logits, *policy)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = rows(logits, *policy)
            try:
                for _ in range(2):
                    graph.replay()
                    self.assertEqual(got.tolist(), [2, 2])
                    backing[1::2] = poison
                backing[0] = poison
                graph.replay()
                ref = _rows_by_sorting(logits, *policy, None, None)
                self.assertTrue(torch.equal(got, ref))
                self.assertEqual(got.tolist(), [0, 2])
            finally:
                graph.reset()

    def test_sampler_refuses_strided_distribution_output(self):
        logits = torch.zeros(2, 3, device='cuda')
        probs = torch.zeros(2, 6, device='cuda')[:, ::2]
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            rows(logits, torch.ones(2, device='cuda'), torch.zeros(2, device='cuda', dtype=torch.int32),
                 torch.ones(2, device='cuda'), None, probs=probs)

    def test_verifier_uses_each_tensors_own_strides(self):
        from engine.kernels.common.block_verify import verify_rows

        def padded(value, factors, fill):
            storage = torch.full(tuple(d*s for d, s in zip(value.shape, factors)), fill,
                                 dtype=value.dtype, device=value.device)
            view = storage[tuple(slice(None, None, s) for s in factors)]
            view.copy_(value)
            return view

        target = torch.tensor([[[.2, .8, 0., 0., 0.]]*4, [[.8, .2, 0., 0., 0.]]*4], device='cuda')
        draft = torch.zeros(2, 3, dtype=torch.int64, device='cuda')
        cand = torch.tensor([[[0, 1]]*3]*2, device='cuda')
        q = torch.tensor([[[.9, .1]]*3]*2, device='cuda')
        u = torch.full((2, 4), .5, device='cuda')
        expected = _block_verify_by_torch(target, draft, cand, q, u)
        self.assertEqual(expected[0].tolist(), [0, 3])
        args = (padded(target, (2, 2, 2), .3), padded(draft, (2, 2), 1),
                padded(cand, (1, 2, 3), 1), padded(q, (2, 3, 2), .1), padded(u, (2, 2), 0.))
        for got, ref in zip(block_verify_batch(*args), expected):
            self.assertTrue(torch.equal(got, ref))
        storage = torch.full((2, 12), -17., device='cuda')
        rest = storage[:, :10:2]
        baseline = verify_rows(target, draft, cand, q, u[:, :3])
        verify_rows(*args[:4], args[4][:, :3], rest=rest)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = verify_rows(*args[:4], args[4][:, :3], rest=rest)
        try:
            graph.replay()
            for got, ref in zip(result, baseline):
                torch.testing.assert_close(got, ref, rtol=0, atol=0)
            self.assertTrue((storage[:, 1::2] == -17.).all().item())
            self.assertTrue((storage[:, 10] == -17.).all().item())
            args[0].copy_(target.flip(0))
            graph.replay()
            self.assertEqual(result[0].tolist(), [3, 0])
        finally:
            graph.reset()

    def test_norm_and_swiglu_refuse_noncontiguous_columns(self):
        from engine.kernels.common.norm_rope import norm, add_norm, _norm_by_torch
        from engine.kernels.common.swiglu import swiglu
        x = torch.tensor([[1., 90., 2., 80., 3., 70., 4., 60.]], device='cuda', dtype=torch.bfloat16)[:, ::2]
        w = torch.ones(4, device='cuda', dtype=torch.bfloat16)
        strided_w = torch.ones(8, device='cuda', dtype=torch.bfloat16)[::2]
        for bias in (None, torch.zeros(4, device='cuda')):
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                norm(x, w, 1e-6, bias=bias)
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                norm(x.contiguous(), strided_w, 1e-6, bias=bias)
        for a, b, weight in ((x, x.contiguous(), w), (x.contiguous(), x, w),
                              (x.contiguous(), x.contiguous(), strided_w)):
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                add_norm(a, b, weight, 1e-6)
        torch.testing.assert_close(norm(x.contiguous(), w, 1e-6), _norm_by_torch(x, w, 1e-6), rtol=0, atol=0)
        fused = torch.full((1, 16), 10., device='cuda', dtype=torch.bfloat16)
        fused[:, ::2] = torch.cat([x, x], -1)
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            swiglu(fused[:, ::2])
        packed = fused[:, ::2].contiguous()
        gate, up = packed.chunk(2, -1)
        torch.testing.assert_close(swiglu(packed), torch.nn.functional.silu(gate)*up, rtol=0, atol=0)

    def test_rotary_weight_view_tracks_its_values_on_replay(self):
        from engine.kernels.common.norm_rope import norm_rope
        x = torch.tensor([[[1., 2., 3., 4.]]], device='cuda', dtype=torch.bfloat16)
        backing = torch.tensor([1., 90., 2., 80., 3., 70., 4., 60.], device='cuda', dtype=x.dtype)
        weight = backing[::2]
        pos = torch.tensor([9], device='cuda')
        norm_rope(x, weight, 1e-6, pos, 10000.)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = norm_rope(x, weight, 1e-6, pos, 10000.)
        try:
            for _ in range(2):
                graph.replay()
                torch.testing.assert_close(result, norm_rope(x, weight.contiguous(), 1e-6, pos, 10000.), rtol=0, atol=0)
                weight.mul_(2)
        finally:
            graph.reset()

    def test_draft_slots_and_contexts_keep_their_strides_on_replay(self):
        from engine.kernels.draft_attention import attend_rows
        q = torch.zeros(2, 1, 2, 128, device='cuda', dtype=torch.bfloat16)
        kv = torch.zeros(2, 1, 1, 128, device='cuda', dtype=torch.bfloat16)
        field = torch.ones(4, 1, 2, 8, 1, 128, device='cuda', dtype=torch.bfloat16)
        field[:, 0, 0].zero_()
        field[2, 0, 1].fill_(2.)
        slot_storage = torch.tensor([1, 0, 2, 0], device='cuda')
        ctx_storage = torch.tensor([1, 0, 1, 0], device='cuda')
        slots, ctx = slot_storage[::2], ctx_storage[::2]
        attend_rows(q, kv, kv, field, ctx, slot=slots)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = attend_rows(q, kv, kv, field, ctx, slot=slots)
        try:
            for _ in range(2):
                graph.replay()
                self.assertEqual(result[:, 0, 0, 0].tolist(), [.5, 1.])
                slot_storage[1::2] = 3
                ctx_storage[1::2] = 7
            slots.copy_(slots.flip(0))
            ctx[0] = 0
            graph.replay()
            self.assertEqual(result[:, 0, 0, 0].tolist(), [0., .5])
        finally:
            graph.reset()


if __name__ == '__main__':
    unittest.main()
