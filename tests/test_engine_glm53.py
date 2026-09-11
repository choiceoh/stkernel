"""GLM cache layout contracts plus small CUDA rollback/paging regressions."""
from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

from engine.profiles.glm53.caches import layout
from engine.profiles.glm53.facts import Facts

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


def tiny_facts():
    return Facts(max_position=1 << 20, hidden=128, layers=3, kinds=("kda", "dsa", "dsa"), dense=(0, 1, 2),
                 vocab=256, rms_eps=1e-5, kda_heads=4, kda_dim=8, conv=4, lower_bound=-5,
                 heads=4, qk_nope=16, v_dim=16, q_lora=16, kv_lora=16,
                 idx_heads=4, idx_dim=128, topk=8, kpool=4, experts=4,
                 topk_experts=2, moe_inter=64, dense_inter=64, routed_scale=1.,
                 swiglu_limit=10., hc=4, hc_eps=1e-6, sinkhorn=20, post_mult=2.,
                 block=16, spec_k=5)


class LayoutTests(unittest.TestCase):
    def test_block_layout_keeps_every_typed_view_aligned_and_disjoint(self):
        F = tiny_facts()
        p = layout(F, range(F.layers))
        self.assertEqual(p.block_bytes % 4096, 0)
        self.assertEqual(p.block_bytes % F.kv_lora, 0)
        self.assertEqual(p.block_bytes % (F.idx_dim + 4), 0)
        regions = []
        for L in F.dsa_layers:
            a, b = p.token_offsets[L], p.pool_offsets[L]
            self.assertEqual(a % F.kv_lora, 0)
            self.assertEqual(b % (F.idx_dim + 4), 0)
            regions.extend([(a, a + F.block * F.kv_lora),
                            (b, b + F.block // F.kpool * (F.idx_dim + 4))])
        regions.sort()
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(regions, regions[1:])))
        self.assertLessEqual(regions[-1][1], p.block_bytes)

    def test_position_rings_retain_history_after_rejecting_all_drafts(self):
        F = tiny_facts()
        fields = {(f.name, f.layer): f for f in layout(F, range(F.layers)).fields}
        self.assertEqual(fields["tail", 1].shape[0], F.kpool - 1 + F.spec_k)
        self.assertEqual(fields["conv", 0].shape[-1], F.conv - 1 + F.spec_k)
        self.assertEqual(fields["rec", 0].shape[0], F.spec_k + 1)

    def test_invalid_layers_and_partial_pools_fail_before_allocation(self):
        F = tiny_facts()
        for layers in ([], [0, 0], [-1], [F.layers]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                layout(F, layers)
        with self.assertRaises(ValueError):
            layout(replace(F, block=15), [1])


@unittest.skipUnless(torch is not None, "requires PyTorch")
class ExpertPreshardTests(unittest.TestCase):
    def test_fp4_midpoints_round_to_even_mantissas(self):
        from engine.modules.quant import _fp4_encode, FP4_TABLE
        values = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
        expected = torch.tensor([0., 1., 1., 2., 2., 4., 4.])
        for sign in (1, -1):
            actual = FP4_TABLE[_fp4_encode(sign*values).long()]
            self.assertTrue(torch.equal(actual, sign*expected))

    def test_each_rank_writes_up_then_gate_with_matching_folded_scales(self):
        from engine.profiles.glm53.specs import layer_specs
        from engine.modules.nvfp4_sf import unswizzle_sf
        F = replace(tiny_facts(), dense=(0, 1), moe_inter=512)
        specs = {s.name: s for s in layer_specs(F, 2)}
        source = {}
        for expert in range(F.experts):
            prefix = f"model.language_model.layers.2.mlp.experts.{expert}."
            rank_rows = torch.arange(F.moe_inter) // F.moe_inter_local
            for projection, value in (("up", 16), ("gate", 64)):
                key = prefix + projection + "_proj."
                source[key + "weight_packed"] = (value + rank_rows[:, None]).expand(-1, F.hidden//2).to(torch.uint8)
                source[key + "weight_scale"] = torch.full((F.moe_inter, F.hidden//16), value/16).to(torch.float8_e4m3fn)
                source[key + "weight_global_scale"] = torch.tensor(2.)
        for rank in range(4):
            packed = specs["L2.moe.w13"].build(source, rank, 4)
            sf = specs["L2.moe.w13_sf"].build(source, rank, 4)
            self.assertTrue(torch.all(packed[:, :F.moe_inter_local] == 16 + rank))
            self.assertTrue(torch.all(packed[:, F.moe_inter_local:] == 64 + rank))
            for expert in range(F.experts):
                plain = unswizzle_sf(sf[expert].view(torch.uint8), 2*F.moe_inter_local, F.hidden//16).view(torch.float8_e4m3fn).float()
                self.assertTrue(torch.all(plain[:F.moe_inter_local] == .5))
                self.assertTrue(torch.all(plain[F.moe_inter_local:] == 2.))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA PyTorch")
class CudaCacheTests(unittest.TestCase):
    def test_paired_cache_oracle_checks_inputs_before_isolating_expert_rounding(self):
        from engine.profiles.glm53.check import PairedMoe
        calls = []

        def expert(x, *args):
            calls.append(1)
            return x + len(calls)

        args = [torch.ones(2, 4, device="cuda") for _ in range(7)] + [10.0]
        oracle = PairedMoe(expert)
        first = oracle(*args)
        oracle.replaying = True
        self.assertTrue(torch.equal(oracle(*args), first))
        self.assertEqual(len(calls), 2)               # both passes execute the real lane
        self.assertGreater(oracle.max_rel, 0)
        for changed in range(3):                    # activations, expert ids, routing weights
            oracle.position = 0
            bad = list(args)
            bad[changed] = bad[changed] + 1
            with self.assertRaisesRegex(AssertionError, "paged cache changed"):
                oracle(*bad)
        self.assertEqual(len(calls), 2)

    def test_sparse_mla_padding_does_not_read_nan_from_an_unused_block(self):
        from engine.modules.sparse_attention import mla_sparse_mqa
        q = torch.ones(1, 1, 16, device="cuda", dtype=torch.bfloat16)
        cache = torch.full((8, 16), float("nan"), device="cuda").to(torch.float8_e4m3fn)
        cache[2] = torch.full((16,), 2., device="cuda").to(torch.float8_e4m3fn)
        slots = torch.tensor([[2, -1]], device="cuda", dtype=torch.int32)
        out = mla_sparse_mqa(q, cache, slots, torch.tensor([1], device="cuda"), 0.25, 1.)
        self.assertTrue(torch.equal(out, torch.full_like(out, 2.)))

    def test_step_rejects_gaps_and_reused_state_slots(self):
        from engine.profiles.glm53.net import Segment, Step
        ids = torch.arange(4, device="cuda")
        cases = [(Segment(0, 1, 0, 1, 4),),
                 (Segment(0, 1, 0, 0, 2),),
                 (Segment(0, 0, 0, 0, 4),),
                 (Segment(0, 1, 0, 0, 2), Segment(1, 1, 0, 2, 2)),
                 (Segment(0, 1, 0, 0, 2), Segment(0, 2, 2, 2, 2))]
        for segments in cases:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                Step(ids, segments)
        with self.assertRaises(ValueError):
            Step.decode([])

    def setUp(self):
        from engine.base.arena import Arena
        from engine.profiles.glm53.caches import Glm53Caches
        self.F = tiny_facts()
        self.p = layout(self.F, range(self.F.layers))
        self.arena = Arena(256 + self.p.nbytes(8, 4))
        self.prefix = self.arena.carve(256, "earlier arena owner").fill_(0xEE)
        self.c = Glm53Caches(self.arena, self.F, range(self.F.layers), 8, 4)

    def step(self, seq, length, ctx=0):
        from engine.profiles.glm53.net import Step
        slot = list(self.c.slots.owner).index(seq)
        st = Step.prefill(torch.zeros(length, dtype=torch.int64, device="cuda"), ctx, seq, slot)
        self.c.prepare(st)
        return st

    def test_views_use_one_arena_and_map_noncontiguous_blocks_per_layer(self):
        c, F = self.c, self.F
        for seq, tokens in [(1, 16), (0, 32), (2, 16)]:
            c.pool.reserve(seq, tokens)
        c.pool.release(1)
        c.pool.reserve(0, 16)
        self.assertEqual(list(c.pool.row(0))[:3], [1, 2, 0])
        c.slots.take(0)
        self.step(0, 48)
        positions = torch.arange(48, device="cuda")
        for L in F.dsa_layers:
            ids = c.token_slots(L, 0, positions).long()
            values = torch.full((48, F.kv_lora), float(L), device="cuda").to(torch.float8_e4m3fn)
            c.latent(L)[ids] = values
            for j, block in enumerate((1, 2, 0)):
                offset = block * self.p.block_bytes + self.p.token_offsets[L]
                raw = c.paged[offset:offset + F.block * F.kv_lora]
                self.assertTrue(torch.equal(raw, values[j * 16:(j + 1) * 16].view(torch.uint8).flatten()))
            pools = c.pool_slots(L, 0, torch.arange(12, device="cuda")).long()
            c.pool_keys(L)[pools] = torch.full((12, F.idx_dim), float(L), device="cuda").to(torch.float8_e4m3fn)
            c.pool_scales(L)[pools] = float(L + 10)
            self.assertTrue(torch.all(c.pool_scales(L)[pools] == L + 10))
            self.assertTrue(torch.all(c.latent(L)[ids].float() == L))
        self.assertEqual(self.arena.used, self.arena.nbytes)
        self.assertTrue(torch.all(self.prefix == 0xEE))
        self.assertTrue(c.latent(1).is_contiguous())
        self.assertEqual(c.latent(1).untyped_storage().data_ptr(), self.arena.buf.data_ptr())

    def test_resetting_one_state_slot_preserves_other_sequences(self):
        a, b = self.c.slots.take(0), self.c.slots.take(1)
        for L in self.F.dsa_layers:
            self.c.tail(L, a).fill_(3)
            self.c.tail(L, b).fill_(7)
        for t in self.c.kda(0, a):
            t.fill_(3)
        for t in self.c.kda(0, b):
            t.fill_(7)
        self.c.reset_slot(a)
        for L in self.F.dsa_layers:
            self.assertTrue(torch.all(self.c.tail(L, a) == 0))
            self.assertTrue(torch.all(self.c.tail(L, b) == 7))
        for t in self.c.kda(0, a):
            self.assertTrue(torch.all(t == 0))
        for t in self.c.kda(0, b):
            self.assertTrue(torch.all(t == 7))

    def test_drafter_context_is_in_the_same_slot_and_arena_budget(self):
        from engine.base.arena import Arena
        from engine.profiles.glm53.caches import Glm53Caches
        draft = (2, 8, 1, 4)
        p = layout(self.F, range(3), draft)
        arena = Arena(p.nbytes(2, 2))
        c = Glm53Caches(arena, self.F, range(3), 2, 2, draft=draft)
        c.draft_ring(1).fill_(3)
        c.draft_ring(2).fill_(7)
        c.reset_slot(1)
        self.assertTrue(torch.all(c.draft_ring(1) == 0))
        self.assertTrue(torch.all(c.draft_ring(2) == 7))
        self.assertEqual(c.draft_ring(1).untyped_storage().data_ptr(), arena.buf.data_ptr())
        self.assertEqual(arena.used, arena.nbytes)

    def test_prepare_rejects_wrong_slot_and_unreserved_positions(self):
        from engine.profiles.glm53.net import Step
        self.c.pool.reserve(0, 16)
        slot = self.c.slots.take(1)
        ids = torch.zeros(17, dtype=torch.int64, device="cuda")
        with self.assertRaises(ValueError):
            self.c.prepare(Step.prefill(ids[:16], 0, 0, slot))
        self.c.slots.give(slot)
        self.c.slots.take(0)
        with self.assertRaises(ValueError):
            self.c.prepare(Step.prefill(ids, 0, 0, slot))

    def test_prepare_publishes_growth_reuse_and_reset_without_stale_blocks(self):
        c = self.c
        c.pool.reserve(0, 16)
        c.slots.take(0)
        self.step(0, 6)
        self.assertEqual(c.block_table[0].tolist(), list(c.pool.row(0)))
        c.pool.reserve_to([0], [32])
        self.step(0, 6, 16)
        self.assertEqual(c.block_table[0].tolist(), list(c.pool.row(0)))
        old = c.block_table[0].clone()
        # Same-sized row, different physical blocks; its old epoch cannot win.
        c.pool.release(0)
        c.pool.reserve(1, 16)
        c.pool.reserve(0, 32)
        self.step(0, 6)
        self.assertEqual(c.block_table[0].tolist(), list(c.pool.row(0)))
        self.assertFalse(torch.equal(c.block_table[0], old))
        # A shorter new owner must clear the old suffix, not expose stale ids.
        c.pool.release(0)
        c.pool.reserve(0, 16)
        self.step(0, 6)
        self.assertEqual(c.block_table[0].tolist(), list(c.pool.row(0)))
        c.reset()                              # same owners/epochs, cleared device contents
        self.step(0, 6)
        self.assertEqual(c.block_table[0].tolist(), list(c.pool.row(0)))

    def test_unchanged_mapping_still_checks_ownership_and_reservation(self):
        from engine.profiles.glm53.net import Step
        c = self.c
        c.pool.reserve(0, 16)
        slot = c.slots.take(0)
        step = self.step(0, 6)
        c.prepare(step)
        c.pool.reserve_to([0], [15])            # rejected draft horizon reuses the same blocks
        c.prepare(step)
        c.slots.give(slot)
        c.slots.take(1)
        with self.assertRaisesRegex(ValueError, "does not own"):
            c.prepare(step)
        c.slots.give(slot)
        c.slots.take(0)
        with self.assertRaisesRegex(ValueError, "reserved context"):
            c.prepare(Step.prefill(step.ids, 15, 0, slot))

    def test_failed_table_upload_can_retry_the_same_mapping(self):
        from engine.profiles.glm53.net import Step
        from unittest.mock import patch
        c = self.c
        c.pool.reserve(0, 16)
        slot = c.slots.take(0)
        self.step(0, 6)
        for replacement in (False, True):
            with self.subTest(replacement=replacement):
                if replacement:
                    c.pool.release(0)
                    c.pool.reserve(1, 16)
                c.pool.reserve_to([0], [32])
                step = Step.prefill(torch.zeros(6, dtype=torch.int64, device="cuda"), 16, 0, slot)
                with patch.object(torch.Tensor, "copy_", side_effect=RuntimeError("injected upload failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected upload failure"):
                        c.prepare(step)
                c.prepare(step)
                self.assertEqual(c.block_table[0].tolist(), list(c.pool.row(0)))

    def test_cache_table_tracks_parking_and_failed_promotion_into_new_blocks(self):
        from engine.base.tiered_kv import TieredKV
        c = self.c
        class Tier:
            block_bytes = c.pool.block_bytes
            fail = False
            def demote(self, seq, storage, ids, tokens):
                self.data = torch.cat([c.pool.block(i) for i in ids]).clone()
                return self.data.numel()
            def promote(self, seq, storage, ids):
                for j, block in enumerate(ids):
                    c.pool.block(block).copy_(self.data[j * self.block_bytes:(j + 1) * self.block_bytes])
                    if self.fail:
                        raise OSError("injected promotion failure")
                return self.data.numel()
            def forget(self, seq):
                pass

        tier = Tier()
        kv = TieredKV(c.pool, tier)
        c.pool.reserve(0, 32)
        c.slots.take(0)
        self.step(0, 6)
        for i, block in enumerate(c.pool.blocks_of(0)):
            block.fill_(i + 7)
        old = c.block_table[0].clone()
        kv.park(0)
        tier.fail = True
        with self.assertRaisesRegex(OSError, "promotion failure"):
            kv.resume(0)
        c.pool.reserve(1, 16)                   # another row occupies a formerly used block
        tier.fail = False
        kv.resume(0)
        self.step(0, 6, 16)
        self.assertEqual(c.block_table[0].tolist(), list(c.pool.row(0)))
        self.assertFalse(torch.equal(c.block_table[0], old))
        self.assertTrue(torch.equal(torch.cat(c.pool.blocks_of(0)), tier.data))

    def test_partial_replacement_upload_retries_and_clears_stale_suffix(self):
        from engine.profiles.glm53.net import Step
        from unittest.mock import patch
        c = self.c
        c.pool.reserve(0, 32)
        slot = c.slots.take(0)
        self.step(0, 6)
        c.pool.release(0)
        c.pool.reserve(0, 16)
        step = Step.prefill(torch.zeros(6, dtype=torch.int64, device="cuda"), 0, 0, slot)
        copy = torch.Tensor.copy_
        def partial_copy(dst, src):
            copy(dst[:1], src[:1])
            raise RuntimeError("injected partial copy failure")
        with patch.object(torch.Tensor, "copy_", partial_copy):
            with self.assertRaisesRegex(RuntimeError, "injected partial copy failure"):
                c.prepare(step)
        self.assertEqual(c.block_table[0, 0].item(), c.pool.row(0)[0])
        c.prepare(step)
        self.assertEqual(c.block_table[0].tolist(), list(c.pool.row(0)))

    def test_incremental_publication_matches_full_rows_across_random_lifetimes(self):
        import random
        from engine.profiles.glm53.net import Step
        c = self.c
        slots = [c.slots.take(seq) for seq in range(3)]
        rng = random.Random(143)
        ids = torch.zeros(1, dtype=torch.int64, device="cuda")
        for tick in range(200):
            seq = rng.randrange(3)
            if rng.randrange(3) == 0:
                c.pool.release(seq)
            else:
                try:
                    c.pool.reserve(seq, rng.randrange(1, 18))
                except MemoryError:
                    pass
            if tick % 23 == 0:
                c.reset()
            for s in range(3):
                if c.pool.tokens[s]:
                    c.prepare(Step.prefill(ids, c.pool.tokens[s] - 1, s, slots[s]))
                    self.assertEqual(c.block_table[s].tolist(), list(c.pool.row(s)))

    def indexer(self):
        from engine.base.comm import Comm
        from engine.profiles.glm53 import lanes
        from engine.profiles.glm53.net import Glm53Net
        F = self.F
        net = Glm53Net(F, Comm(4, 0), lanes.reference(), layers=[1])
        g = torch.Generator(device="cuda").manual_seed(72)
        def rand(*shape):
            return torch.randn(*shape, generator=g, device="cuda")
        net.p = {"L1.idx.wq_b": rand(F.idx_heads * F.idx_dim, F.q_lora).to(torch.bfloat16),
                 "L1.idx.wk": torch.eye(F.hidden, device="cuda", dtype=torch.bfloat16),
                 "L1.idx.w_heads": rand(F.idx_heads, F.hidden),
                 "L1.idx.k_norm_w": torch.ones(F.idx_dim, device="cuda"),
                 "L1.idx.k_norm_b": torch.zeros(F.idx_dim, device="cuda"),
                 "L1.idx.gate": torch.zeros(F.idx_dim, F.hidden, device="cuda", dtype=torch.bfloat16),
                 "L1.idx.ape": torch.zeros(F.kpool, F.idx_dim, device="cuda")}
        return net, rand(40, F.hidden).to(torch.bfloat16), rand(40, F.q_lora).to(torch.bfloat16)

    def test_rollback_matches_clean_pools_for_every_boundary_and_accept_count(self):
        net, x, qr = self.indexer()
        c, F = self.c, self.F
        c.pool.reserve(0, 40)
        c.slots.take(0)
        for phase in range(F.kpool):
            ctx = 12 + phase
            for accepted in range(1, F.spec_k + 1):
                with self.subTest(phase=phase, accepted=accepted):
                    end = ctx + accepted + F.spec_k + 1
                    c.reset()
                    net._indexer(1, x[:ctx + accepted], qr[:ctx + accepted], self.step(0, ctx + accepted), c)
                    net._indexer(1, x[ctx + accepted:end], qr[ctx + accepted:end], self.step(0, 6, ctx + accepted), c)
                    ids = c.pool_slots(1, 0, torch.arange(end // F.kpool, device="cuda")).long()
                    expected_keys = c.pool_keys(1)[ids].view(torch.uint8).clone()
                    expected_scales = c.pool_scales(1)[ids].clone()
                    c.reset()
                    net._indexer(1, x[:ctx], qr[:ctx], self.step(0, ctx), c)
                    draft = x[ctx:ctx + 6].clone()
                    draft[accepted:] = x[30:30 + 6 - accepted]
                    net._indexer(1, draft, qr[ctx:ctx + 6], self.step(0, 6, ctx), c)
                    net._indexer(1, x[ctx + accepted:end], qr[ctx + accepted:end], self.step(0, 6, ctx + accepted), c)
                    self.assertTrue(torch.equal(c.pool_keys(1)[ids].view(torch.uint8), expected_keys))
                    self.assertTrue(torch.equal(c.pool_scales(1)[ids], expected_scales))

    def test_indexer_dispatches_quantization_and_pool_slots_through_lanes(self):
        from unittest.mock import Mock
        net, x, qr = self.indexer()
        quant = Mock(wraps=net.lanes.indexer_quant)
        finish = Mock(wraps=net.lanes.pool_slots)
        net.lanes = replace(net.lanes, indexer_quant=quant, pool_slots=finish)
        self.c.pool.reserve(0, 24)
        self.c.slots.take(0)
        for ctx, length in ((0, 12), (12, 6)):
            net._indexer(1, x[ctx:ctx + length], qr[ctx:ctx + length], self.step(0, length, ctx), self.c)
            rows = quant.call_args.args[0]
            self.assertEqual(rows.shape, (length * self.F.idx_heads, 128))
            self.assertEqual(rows.dtype, torch.bfloat16)
            self.assertTrue(rows.is_contiguous())
            pools, seq_lens, size = finish.call_args.args[:3]
            self.assertEqual(pools.shape, (length, self.F.topk // self.F.kpool))
            self.assertEqual(seq_lens.tolist(), list(range(ctx + 1, ctx + length + 1)))
            self.assertEqual(size, self.F.kpool)
        self.assertEqual((quant.call_count, finish.call_count), (2, 2))

    def test_indexer_lane_failure_propagates_without_reference_fallback(self):
        from unittest.mock import Mock
        net, x, qr = self.indexer()
        original = net.lanes
        self.c.pool.reserve(0, 16)
        self.c.slots.take(0)
        for name in ("indexer_quant", "pool_slots"):
            with self.subTest(lane=name):
                self.c.reset()
                net.lanes = replace(original, **{name: Mock(side_effect=RuntimeError("injected lane failure"))})
                with self.assertRaisesRegex(RuntimeError, "injected lane failure"):
                    net._indexer(1, x[:12], qr[:12], self.step(0, 12), self.c)

    def test_indexer_finalizes_each_segment_with_its_own_block_row(self):
        from engine.modules.sparse_indexer import select_with_tail
        from engine.profiles.glm53.net import Step
        from unittest.mock import Mock
        net, x, qr = self.indexer()
        finish = Mock(wraps=net.lanes.pool_slots)
        net.lanes = replace(net.lanes, pool_slots=finish)
        chunks = []
        for seq, length in ((2, 12), (0, 6)):
            self.c.pool.reserve(seq, length)
            slot = self.c.slots.take(seq)
            chunks.append((torch.zeros(length, device="cuda", dtype=torch.int64), 0, seq, slot))
        step = Step.decode(chunks)
        self.c.prepare(step)
        selected, counts = net._indexer(1, x[:18], qr[:18], step, self.c)
        self.assertEqual(finish.call_count, 2)
        for call, segment in zip(finish.call_args_list, step.segments):
            pools, lengths, pool_size, row, size, stride, offset, out, valid = call.args
            tokens = select_with_tail(pools, lengths, pool_size)
            self.assertEqual(row.data_ptr(), self.c.block_table[segment.seq].data_ptr())
            positions = tokens.sort(dim=1, descending=True).values
            expected = self.c.token_slots(1, segment.seq, positions.clamp_min(0).flatten()).view_as(positions)
            expected.masked_fill_(positions < 0, -1)
            sl = slice(segment.start, segment.start + segment.length)
            self.assertTrue(torch.equal(selected[sl], expected))
            self.assertTrue(torch.equal(counts[sl], (positions >= 0).sum(1).int()))

    def test_long_prefill_ring_contains_only_the_latest_positions(self):
        net, x, qr = self.indexer()
        self.c.pool.reserve(0, 40)
        slot = self.c.slots.take(0)
        net._indexer(1, x, qr, self.step(0, 40), self.c)
        width = self.F.kpool - 1 + self.F.spec_k
        expected = torch.nn.functional.layer_norm(x[-width:].float(), (128,), eps=1e-6).to(torch.bfloat16)
        positions = torch.arange(40 - width, 40, device="cuda")
        self.assertTrue(torch.equal(self.c.tail(1, slot)[positions % width, 0], expected))

    def runtime(self, *, eos_ids=()):
        from engine.base.scheduler import Contract
        from engine.profiles.glm53.runtime import Glm53Runtime
        F = self.F

        class NextTokenNet:
            layers = (0, 1, 2)
            def __init__(self):
                self.F = F
            def forward(self, step, caches, aux_layers=None):
                h = step.ids[:, None].float()
                return (h, None) if aux_layers is not None else h
            def head(self, hidden):
                logits = torch.full((len(hidden), F.vocab), -100., device=hidden.device)
                return logits.scatter_(1, (hidden.long() + 1) % F.vocab, 100.)
            def head_tokens(self, hidden, decodable=None):
                return self.head(hidden)[:, :decodable].argmax(-1)

        return Glm53Runtime(NextTokenNet(), self.c, Contract(4, 8, 0, 0., 2), eos_ids=eos_ids)

    def test_serving_adapter_stops_at_first_sample_and_clips_accepted_drafts(self):
        from engine.base.runner import Runner, STEP_RECORD
        from engine.base.record import Ring
        from engine.base.scheduler import Contract
        from engine.profiles.glm53.adapter import Glm53Engine
        from unittest.mock import patch

        class Draft:
            k = 2
            aux_layers = ()
            def propose(self, anchor, position, ring):
                return [anchor + 1, anchor + 2]

        net = self.runtime().net
        for limit, eos, expected in ((1, (), [5]), (9, (5,), [5]),
                                     (2, (), [5, 6]), (9, (6,), [5, 6])):
            with self.subTest(limit=limit, eos=eos):
                engine = Glm53Engine(net, self.c, self.F, Draft(), max_new=limit, eos_ids=eos)
                runner = Runner(engine, Contract(4, 8, 2, 0., 2), self.c.pool, self.c.slots, Ring(8, STEP_RECORD.size))
                engine.add(0, [4])
                runner.submit(0, 1, now=0)
                with patch.object(self.c, "draft_ring", return_value=None):
                    for tick in range(4):
                        if runner.step(now=tick + 1) is None:
                            break
                self.assertEqual(engine.generated(0), expected)
                self.assertEqual(self.c.pool.available, self.c.pool.num_blocks)
                self.assertEqual(self.c.slots.available, 4)

    def test_memory_prefill_preparation_covers_capacity_and_cleans_failed_warmup(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from engine.profiles.glm53.adapter import Glm53Engine
        net = self.runtime().net
        engine = Glm53Engine(net, self.c, self.F)
        engine.prefill_chunk = 20
        phases = []
        engine.memory = SimpleNamespace(checkpoint=phases.append)
        capacity = self.c.pool.num_blocks * self.F.block
        with patch.object(net, "forward", wraps=net.forward) as forward:
            engine._warmup_prefill_memory()
        self.assertEqual([(call.args[0].segments[0].ctx, call.args[0].ids.numel())
                          for call in forward.call_args_list], [(0, 20), (capacity-20, 20)])
        self.assertEqual(len(phases), 4)
        self.assertEqual(self.c.pool.rows_in_use, 0)
        self.assertEqual(self.c.slots.available, 4)
        self.assertTrue(torch.all(self.c.state == 0))
        with patch.object(net, "forward", side_effect=MemoryError("workspace exhausted")):
            with self.assertRaisesRegex(MemoryError, "workspace exhausted"):
                engine._warmup_prefill_memory()
        self.assertEqual(self.c.pool.rows_in_use, 0)
        self.assertEqual(self.c.slots.available, 4)
        self.assertTrue(torch.all(self.c.state == 0))

    def test_runtime_generates_exact_limits_and_releases_every_request(self):
        r = self.runtime()
        r.submit(0, torch.arange(20, device="cuda"), 3, now=0)
        r.submit(1, torch.arange(7, device="cuda"), 1, now=0)
        kinds = []
        for tick in range(20):
            step = r.step(now=tick + 1)
            if step is None:
                break
            kinds.append(step.kind)
        self.assertEqual(r.take_result(0), (20, 21, 22))
        self.assertEqual(r.take_result(1), (7,))
        self.assertEqual(kinds.count("prefill"), 4)
        self.assertEqual(kinds.count("decode"), 2)
        self.assertEqual(self.c.pool.available, self.c.pool.num_blocks)
        self.assertEqual(self.c.slots.available, 4)
        r.submit(0, torch.tensor([9], device="cuda"), 1, now=21)
        r.step(now=21)
        self.assertEqual(r.take_result(0), (10,))

    def test_flat_decode_preserves_ragged_drafts_contexts_and_auxiliary_rows(self):
        from engine.profiles.glm53.adapter import Glm53Engine
        from engine.profiles.glm53.net import Segment
        from unittest.mock import patch

        observed, steps = [], []
        class Draft:
            k = 2
            aux_layers = ()
            def propose(self, anchor, position, ring):
                return {5: [6, 7], 21: [99], 30: []}[anchor]
            def observe(self, ring, positions, aux):
                observed.append((ring, positions.tolist(), aux[:, 0].tolist()))

        net = self.runtime().net
        def forward(step, caches, aux_layers=None):
            steps.append(step)
            return step.ids[:, None].float(), torch.arange(len(step.ids), device="cuda")[:, None]

        engine = Glm53Engine(net, self.c, self.F, Draft())
        jobs = [(2, [4], 2), (0, [18, 20], 4), (1, [0, 1, 29], 2)]
        slots = []
        with patch.object(net, "forward", side_effect=forward), patch.object(self.c, "draft_ring", side_effect=lambda slot: slot):
            for seq, prompt, limit in jobs:
                engine.add(seq, prompt, max_new=limit)
                self.c.pool.reserve(seq, len(prompt) + 4)
                slot = self.c.slots.take(seq)
                slots.append(slot)
                engine.open(seq, slot)
                self.assertFalse(engine.prefill(seq, 0, len(prompt), None, slot))
            observed.clear()
            self.assertEqual(engine.decode([2, 0, 1], None, slots), [True, False, True])
        self.assertEqual(steps[-1].ids.tolist(), [5, 6, 7, 21, 99, 30])
        self.assertEqual(steps[-1].segments, (Segment(2, slots[0], 1, 0, 3),
                                            Segment(0, slots[1], 2, 3, 2),
                                            Segment(1, slots[2], 3, 5, 1)))
        self.assertEqual(observed, [(slots[0], [1], [0]), (slots[1], [2], [3]), (slots[2], [3], [5])])
        self.assertEqual([engine.context(s) for s in (2, 0, 1)], [2, 3, 4])
        self.assertEqual([engine.generated(s) for s in (2, 0, 1)], [[5, 6], [21, 22], [30, 31]])
        self.assertEqual((engine.accepted_total, engine.drafted_total), (1, 3))
        result = engine.generated(2)
        result.append(99)
        self.assertEqual(engine.generated(2), [5, 6])  # result collection still returns an independent copy

    def test_generation_count_resets_on_a_new_turn_after_long_history(self):
        from engine.profiles.glm53.adapter import Glm53Engine
        engine = Glm53Engine(self.runtime().net, self.c, self.F)
        engine.add(0, [1, 2])
        engine.tokens[0].extend([3] * 65536)
        engine.ctx[0] = len(engine.tokens[0]) - 1
        self.assertEqual(engine._generated_count(0), 65536)
        self.assertEqual(engine.extend(0, [4, 5], max_new=2), 3)
        self.assertEqual(engine._generated_count(0), 0)
        engine.tokens[0].append(6)
        self.assertEqual(engine._generated_count(0), 1)
        self.assertEqual(engine.generated(0), [6])

    def test_clipped_drafts_leave_the_last_emitted_token_pending_for_the_next_turn(self):
        from engine.base.record import Ring
        from engine.base.runner import Runner, STEP_RECORD
        from engine.base.scheduler import Contract
        from engine.profiles.glm53.adapter import Glm53Engine
        from unittest.mock import patch
        class Draft:
            k = 2
            aux_layers = ()
            def propose(self, anchor, position, ring):
                return [anchor + 1, anchor + 2]
        engine = Glm53Engine(self.runtime().net, self.c, self.F, Draft(), max_new=2)
        runner = Runner(engine, Contract(4, 8, 2, 0., 2), self.c.pool, self.c.slots,
                        Ring(8, STEP_RECORD.size), keep_idle=True)
        engine.add(0, [4]); runner.submit(0, 1)
        with patch.object(self.c, "draft_ring", return_value=None):
            while runner.step() is not None:
                pass
            self.assertEqual(engine.generated(0), [5, 6])
            self.assertEqual(engine.context(0), 2)
            self.assertEqual(engine.extension_tokens(0, [9]), 2)
            runner.extend(0, engine.extend(0, [9], max_new=1))
            while runner.step() is not None:
                pass
        self.assertEqual(engine.generated(0), [10])
        self.assertEqual(engine.context(0), len(engine.tokens[0]) - 1)
        runner.cancel(0)
        engine.forget(0)

    def test_http_engine_adapter_recycles_rows_and_releases_token_buffers(self):
        from engine.base.comm import Comm
        from engine.base.record import Ring
        from engine.base.runner import Runner, STEP_RECORD
        from engine.base.scheduler import Contract
        from engine.base.serve import Server
        from engine.profiles.glm53.adapter import Glm53Engine
        engine = Glm53Engine(self.runtime().net, self.c, self.F)
        runner = Runner(engine, Contract(4, 8, 0, 0., 2), self.c.pool, self.c.slots, Ring(8, STEP_RECORD.size))
        server = Server(engine, runner, Comm(1, 0))
        jobs = [server.submit([i], 2, 0) for i in range(25)]
        for _ in range(100):
            if not server.once() and not server._waiting:
                break
        for i, (request, event) in enumerate(jobs):
            self.assertTrue(event.is_set())
            self.assertEqual(server.take_result(request), [i + 1, i + 2])
        self.assertFalse(engine.tokens or engine.prompt_len or engine.limits or engine.ctx or engine.slot)
        self.assertEqual(self.c.pool.available, self.c.pool.num_blocks)
        self.assertEqual(self.c.slots.available, 4)

    def test_serving_adapter_extends_cached_context_and_cancel_releases_idle_state(self):
        from engine.base.comm import Comm
        from engine.base.record import Ring
        from engine.base.runner import Runner, STEP_RECORD
        from engine.base.scheduler import Contract
        from engine.base.serve import Server
        from engine.profiles.glm53.adapter import Glm53Engine
        engine = Glm53Engine(self.runtime().net, self.c, self.F)
        runner = Runner(engine, Contract(4, 8, 0, 0., 2), self.c.pool, self.c.slots,
                        Ring(8, STEP_RECORD.size), keep_idle=True)
        server = Server(engine, runner, Comm(1, 0))
        first, _ = server.submit([4], 2, 0)
        while server.once():
            pass
        self.assertEqual(server.take_result(first), [5, 6])
        row = server._conversations[first]
        self.assertEqual(engine.context(row), 2)
        second, _ = server.submit([9, 10], 2, 0, conversation=first)
        while server.once():
            pass
        self.assertEqual(server.take_result(second), [11, 12])
        self.assertEqual(engine.tokens[row], [4, 5, 6, 9, 10, 11, 12])
        self.assertEqual(engine.context(row), 6)
        server.alive = False
        server.once()
        self.assertFalse(engine.tokens or engine.ctx or engine.slot or runner.idle)
        self.assertEqual(self.c.pool.available, self.c.pool.num_blocks)
        self.assertEqual(self.c.slots.available, 4)

    def test_runtime_can_stop_on_the_first_eos_without_decode(self):
        r = self.runtime(eos_ids=(7,))
        r.submit(0, torch.arange(7, device="cuda"), 20, now=0)
        self.assertEqual(r.step(now=1).kind, "prefill")
        self.assertIsNone(r.step(now=2))
        self.assertEqual(r.take_result(0), (7,))

    def test_runtime_rejects_bad_input_before_reserving_memory(self):
        r = self.runtime()
        for ids in (torch.tensor([-1], device="cuda"), torch.tensor([256], device="cuda"),
                    torch.empty(0, dtype=torch.int64, device="cuda")):
            with self.assertRaises(ValueError):
                r.submit(0, ids, 1)
        self.assertEqual(self.c.pool.available, 8)
        self.assertEqual(self.c.slots.available, 4)


if __name__ == "__main__":
    unittest.main()
