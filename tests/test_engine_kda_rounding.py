"""SR distribution, recurrent drift, counter identity and actual graph stores."""
import unittest

import torch
import triton
import triton.language as tl

from engine.modules.kda_storage import FP16_FORMAT, philox_words, rounding_seed, store_state
from engine.kernels.kda.rounding import fp16_sr, _copy


@triton.jit
def _walk(OUT, STEPS: tl.constexpr, WIDTH: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    state = tl.full((BLOCK,), 1., tl.float32)
    for t in range(STEPS):
        value = state + 2**-12
        if SR:
            state = fp16_sr(value, t, i, WIDTH, 0).to(tl.float32)
        else:
            state = value.to(tl.float16).to(tl.float32)
    tl.store(OUT + i, state, i < WIDTH)


class RoundingContractTests(unittest.TestCase):
    def test_philox_known_answer_and_high_counter_word(self):
        words = philox_words(torch.tensor([0, 2**32], dtype=torch.int64), 0)
        self.assertEqual(int(words[0]), 0x6627E8D5)  # Random123 Philox4x32-10 zero key/counter
        self.assertNotEqual(int(words[0]), int(words[1]))
        self.assertNotEqual(rounding_seed(1, 0), rounding_seed(0, 1))
        self.assertNotEqual(rounding_seed(0, 0), rounding_seed(0, 1))

    def test_cpu_reference_unbiased_and_exact_values_unchanged(self):
        for value, spacing in ((1 + 2**-12, 2**-10), (-1 - 2**-12, 2**-10), (2**-25, 2**-24)):
            src = torch.full((1, 1, 32768), value)
            dst = torch.empty_like(src, dtype=torch.float16)
            store_state(dst, src, 32768, rounding_seed(3, 2))
            self.assertLess(abs(float(dst.double().mean()) - value), spacing * .02)
        src = torch.tensor([0., -0., 1., -2., float('inf'), -float('inf')]).reshape(1, 1, -1)
        dst = torch.empty_like(src, dtype=torch.float16)
        store_state(dst, src, 0, 0)
        self.assertTrue(torch.equal(dst.view(torch.int16), src.half().view(torch.int16)))

    def test_old_rtne_tier_is_not_promoted_into_sr(self):
        from engine.base.kv_tier import NvmeTier, SECTOR
        tier = NvmeTier.__new__(NvmeTier)
        tier.block_bytes, tier.state_format = SECTOR, FP16_FORMAT
        tier.index = {'1': dict(block_bytes=SECTOR, extra=SECTOR, bytes=2*SECTOR, at=1,
                               state_format='glm53-kda-fp16-v1'),
                      '2': dict(block_bytes=SECTOR, extra=SECTOR, bytes=2*SECTOR, at=2,
                               state_format=FP16_FORMAT)}
        self.assertEqual(tier.keys(), [2])
        self.assertFalse(tier.has(1))


@unittest.skipUnless(torch.cuda.is_available(), 'requires Blackwell CUDA')
class CudaRoundingTests(unittest.TestCase):
    def test_neighbors_mean_exact_values_and_nonfinite(self):
        for value, spacing in ((1 + 2**-12, 2**-10), (-1 - 2**-12, 2**-10),
                               (2**-25, 2**-24), (2**-14 + 2**-26, 2**-24)):
            src = torch.full((1, 1, 32768), value, device='cuda')
            dst = torch.empty_like(src, dtype=torch.float16)
            store_state(dst, src, 32768, rounding_seed(3, 2))
            self.assertEqual(dst.unique().numel(), 2)
            self.assertLessEqual(float((dst.float() - src).abs().max()), spacing)
            self.assertLess(abs(float(dst.double().mean()) - value), spacing * .02)
        src = torch.tensor([0., -0., 1., -2., 65504., float('inf'), -float('inf'), float('nan')], device='cuda').reshape(1, 1, -1)
        dst = torch.empty_like(src, dtype=torch.float16)
        store_state(dst, src, 9, 0)
        self.assertTrue(torch.equal(dst[..., :-1].view(torch.int16), src.half()[..., :-1].view(torch.int16)))
        self.assertTrue(torch.isnan(dst[..., -1]).all())

    def test_counter_layout_position_and_layer_domains(self):
        src = torch.full((2, 33, 17), 1 + 2**-12, device='cuda').transpose(1, 2)
        n = src.numel()
        dst = torch.empty(src.shape, device='cuda', dtype=torch.float16)
        seed = rounding_seed(4, 2)
        store_state(dst, src, 32768, seed)
        expected = dst.clone()
        # A different CTA size must use the same logical element stream.
        _copy[(triton.cdiv(n, 128),)](src, dst, 32768, *src.shape, *src.stride(), seed, 128)
        self.assertTrue(torch.equal(dst, expected))
        for position, other_seed in ((32769, seed), (32768 + 2**32, seed), (32768, rounding_seed(5, 2)), (32768, rounding_seed(4, 3))):
            store_state(dst, src, position, other_seed)
            self.assertFalse(torch.equal(dst, expected))

    def test_graph_ring_writes_match_eager_and_preserve_other_cells(self):
        from engine.kernels.state import write_ring
        src = torch.full((7, 2, 17, 33), 1 + 2**-12, device='cuda')
        ring = torch.full((4, 7, 2, 17, 33), -8., device='cuda', dtype=torch.float16)
        slot, context = torch.tensor([1], device='cuda'), torch.tensor(32768, device='cuda')
        seed = rounding_seed(3, 2)
        write_ring(src, ring, slot, context, round_seed=seed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_ring(src, ring, slot, context, round_seed=seed)
        try:
            for physical, position in ((1, 32768), (3, 32768), (2, 32769), (1, 65536)):
                ring.fill_(-8.)
                expected = ring.clone()
                for i in range(7):
                    store_state(expected[physical, (position+i)%7], src[i], position+i, seed)
                slot.fill_(physical); context.fill_(position)
                graph.replay()
                self.assertTrue(torch.equal(ring, expected))
        finally:
            graph.reset()

    def test_direct_recurrence_replay_after_rollback_and_slot_remap(self):
        from engine.kernels.kda.ring import recurrent_kda_ring
        from tests.test_engine_kda_ring import KdaRingTests
        args, _, ring = KdaRingTests().inputs(7, h=2, hv=2, k=33, v=17, state_dtype=torch.float16, cells=7)
        initial = ring[0].clone()
        seed = rounding_seed(4, 1)
        slot, context = torch.tensor([1], device='cuda'), torch.tensor([32768], device='cuda')
        recurrent_kda_ring(*args, ring, slot, context, -5., round_seed=seed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = recurrent_kda_ring(*args, ring, slot, context, -5., round_seed=seed)
        try:
            ring[1].copy_(initial)
            graph.replay()
            expected, expected_out = ring[1].clone(), output.clone()
            for physical in (2, 1):
                # Reject all speculative writes, restore the accepted state in
                # another slot, then replay the same absolute positions.
                ring[physical].copy_(initial); slot.fill_(physical)
                graph.replay()
                self.assertTrue(torch.equal(ring[physical], expected))
                self.assertTrue(torch.equal(output, expected_out))
        finally:
            graph.reset()

    def test_repeated_small_updates_do_not_stagnate(self):
        n, steps = 4096, 2048
        out = torch.empty(n, device='cuda')
        _walk[(triton.cdiv(n, 128),)](out, steps, n, False, 128)
        self.assertTrue(torch.equal(out, torch.ones_like(out)))
        _walk[(triton.cdiv(n, 128),)](out, steps, n, True, 128)
        self.assertLess(abs(float(out.double().mean()) - (1 + steps*2**-12)), .005)


if __name__ == '__main__':
    unittest.main()
