"""Compare compact merge against dense CUDA top-k, including its tie order."""
import importlib.util
import os
import unittest

torch = None
if importlib.util.find_spec('torch'):
    import torch
INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
READY = torch is not None and importlib.util.find_spec('triton') is not None
DEVICE = 'cpu' if INTERPRET else 'cuda'


@unittest.skipUnless(torch is not None, 'torch required')
class CompactRuntimeTests(unittest.TestCase):
    def test_only_qualified_geometry_and_runtime_enable_the_default(self):
        from unittest.mock import patch
        from engine.modules.vocab import compact_supported
        with patch.object(torch.version, 'git_version', 'cf30153c4c131c8164ee7798e5022d810682e2cb'), \
             patch.object(torch.version, 'cuda', '13.2'), \
             patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)) as capability:
            self.assertTrue(compact_supported(14, 154880, 64, 16, 'cuda'))
            for args in ((0,154880,64,16,'cuda'), (35,154880,64,16,'cuda'),
                         (14,256,64,16,'cuda'), (14,154880,16,16,'cuda'),
                         (14,154880,64,7,'cuda'), (14,154880,64,16,'cpu')):
                self.assertFalse(compact_supported(*args))
            with patch.object(torch.version, 'git_version', 'unqualified'):
                self.assertFalse(compact_supported(14,154880,64,16,'cuda'))
            capability.return_value = (9, 0)
            self.assertFalse(compact_supported(14,154880,64,16,'cuda'))


@unittest.skipUnless(READY and (INTERPRET or torch.cuda.is_available()), 'CUDA or Triton interpreter required')
class VocabMergeTests(unittest.TestCase):
    def packet(self, full, k, decodable=None):
        from engine.kernels.common.vocab_candidates import pack, select
        width = full.shape[1] // 4
        packets = []
        for rank in range(4):
            valid = max(0, min(width, (decodable or full.shape[1])-rank*width))
            if valid:
                values = pack(full[:, rank*width:(rank+1)*width], rank*width, valid)
                selected = select(values, min(k, valid))
                selected = torch.nn.functional.pad(selected, (0, k-selected.shape[1]), value=-(2**63))
            else:
                selected = torch.full((len(full), k), -(2**63), dtype=torch.int64, device=full.device)
            packets.append(selected)
        return torch.cat(packets, -1)

    def test_radix_preorder_including_implicit_background(self):
        from engine.kernels.common.vocab_merge import preorder
        from engine.kernels.common.vocab_candidates import restore
        generator = torch.Generator(device=DEVICE).manual_seed(1709)
        for k in (1, 7, 16, 32):
            full = torch.randn(5, 256, device=DEVICE, generator=generator)
            full[0].fill_(-float('inf'))
            full[1].zero_(); full[1, ::2] = -0.
            full[2, ::3] = float('nan')
            full[3, ::3] = 4.
            full[4, ::3] = float('inf')
            for valid in (256, 137, 3):
                packet = self.packet(full, k, valid)
                dense = restore(packet, 256)
                values, ids = preorder(packet, 256, k)
                # Independent integer sort over the RESTORED dense array,
                # including implicit -inf positions absent from every packet.
                bits = dense.view(torch.int32).long()
                ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
                ordered = torch.where(dense.isnan(), 0x7fffffff, ordered).cpu().tolist()
                for row, scores in enumerate(ordered):
                    winners = sorted(range(256), key=lambda i: (-scores[i], i))[:k]
                    cutoff = min(scores[i] for i in winners)
                    expected = sorted(winners, key=lambda i: (scores[i] == cutoff, i))
                    self.assertEqual(ids[row].tolist(), expected)
                expected_values = dense.gather(-1, ids)
                self.assertTrue(torch.equal(values.view(torch.int32), expected_values.view(torch.int32)))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'actual CUDA ordering required')
    def test_full_vocabulary_exact_ids_ties_masks_and_nonfinite(self):
        from engine.kernels.common.vocab_merge import topk
        from engine.kernels.common.vocab_candidates import restore
        generator = torch.Generator(device='cuda').manual_seed(17103)
        for rows in (1, 7, 14, 28):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                full = torch.randn(rows, 154880, generator=generator, device='cuda').to(dtype)
                for fixture in ('random', 'ties', 'zero', 'negative_inf', 'nonfinite'):
                    if fixture == 'ties': full[:, ::3000] = 5
                    if fixture == 'zero': full.zero_(); full[:, ::2] = -0.
                    if fixture == 'negative_inf': full.fill_(-float('inf'))
                    if fixture == 'nonfinite': full[:, ::3000] = float('nan'); full[:, 1::3000] = float('inf')
                    for valid in (154856, 38710, 7):
                        with self.subTest(rows=rows, dtype=dtype, fixture=fixture, valid=valid):
                            packet = self.packet(full, 16, valid)
                            want = restore(packet, 154880).topk(16, dim=-1)
                            got = topk(packet, 154880, 16)
                            torch.testing.assert_close(got.values, want.values, rtol=0, atol=0, equal_nan=True)
                            self.assertTrue(torch.equal(got.indices, want.indices), (got.indices, want.indices))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'actual CUDA graph required')
    def test_graph_replays_moving_packets_without_stale_state_or_rng(self):
        from engine.kernels.common.vocab_merge import topk
        from engine.kernels.common.vocab_candidates import restore
        packet = torch.full((7, 64), -(2**63), dtype=torch.int64, device='cuda')
        for _ in range(3): topk(packet, 154880, 16)
        rng = torch.cuda.get_rng_state().clone()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph): got = topk(packet, 154880, 16)
            for offset in (0, 38708, 120000, 4):
                full = torch.full((7, 154880), -float('inf'), device='cuda')
                full[:, offset:offset+25] = 4
                packet.copy_(self.packet(full, 16))
                before = packet.clone()
                graph.replay()
                want = restore(packet, 154880).topk(16, dim=-1)
                self.assertTrue(torch.equal(got.values, want.values))
                self.assertTrue(torch.equal(got.indices, want.indices))
                self.assertTrue(torch.equal(packet, before))
            self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
        finally:
            graph.reset()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'actual CUDA selector required')
    def test_serving_greedy_and_t1_walk_q_and_block_verdict_are_identical(self):
        from types import SimpleNamespace as NS
        from engine.base.comm import Comm
        from engine.base.sampler import block_verify
        from engine.modules.vocab import CandidateBuffer
        from engine.profiles.glm53.drafter import Drafter
        generator = torch.Generator(device='cuda').manual_seed(17119)
        vocab, K, R = 154880, 7, 256
        logits = torch.randn(K, vocab, device='cuda', generator=generator).bfloat16()
        h = torch.zeros(K+1, R, device='cuda', dtype=torch.bfloat16)
        proj = torch.randn(K, R, device='cuda', generator=generator) * .03
        books = {name: torch.randn(vocab, R, device='cuda', generator=generator).bfloat16() * .03
                 for name in ('candidate_selector.predecessor_codebook', 'candidate_selector.successor_codebook')}
        drafter = NS(k=K, F=NS(mask_id=0, sel_top_k=16), p=books,
                     target=NS(head_local=lambda _: logits, comm=Comm(), rank=0, vp=vocab),
                     block=lambda *args: h, selector_projection=lambda _: proj, _packed_head=lambda: False,
                     diagnostics=None, decodable=154856, selector_alpha=(1.,)*K)
        anchor = torch.tensor([19], device='cuda')
        ring = torch.zeros(1, device='cuda')
        rng = torch.cuda.get_rng_state().clone()
        for tied in (False, True):
            if tied:
                logits.fill_(-1); logits[:, 1000:1032] = 5
                proj.zero_()  # exact selector ties make candidate order observable
            for uniforms in ([.01,.21,.41,.61,.81,.99,.5], [.99,.8,.6,.4,.2,.01,.7]):
                outputs = []
                for compact in (False, True):
                    drafter.candidate_buffer = CandidateBuffer(K, vocab, 16, 'cuda', compact=compact)
                    greedy = Drafter.propose_tensor(drafter, anchor, 2048, ring)
                    sampled, q = Drafter.propose_sampled_tensor(drafter, anchor, 2048, ring,
                                                               1., uniforms, vocab)
                    outputs.append((greedy, sampled, q))
                for actual, reference in zip(outputs[1], outputs[0]):
                    self.assertTrue(torch.equal(actual, reference))
                # A nontrivial target with overlapping and residual mass.
                p = torch.zeros(K+1, vocab, device='cuda')
                p[:K] = outputs[0][2] * .75
                p[:K, 154855] += .25
                p[K, 154855] = 1.
                verdicts = [block_verify(p, sampled.tolist(), q, uniforms+[.43])
                            for _, sampled, q in outputs]
                self.assertEqual(verdicts[0], verdicts[1])
        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
