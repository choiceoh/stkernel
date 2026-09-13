"""Execution order must preserve state, masks and the full-batch contract."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

import torch

from engine.base.arena import Arena
from engine.base.comm import Comm, LocalTP
from engine.profiles.glm53.caches import Glm53Caches, layout, snapshot_layout
from engine.profiles.glm53.execution import ExecutionPlan, SerialStreams, decode_overlap, prefill_layer_major, prefill_steps
from engine.profiles.glm53.lanes import reference
from engine.profiles.glm53.net import Glm53Net, Segment, Step
from tests.test_engine_glm53 import tiny_facts


class EagerStep(Step):
    def subset(self, start, end):
        segs = self.segments[start:end]
        offset = segs[0].start
        return Step(self.ids[offset:segs[-1].start + segs[-1].length],
                    tuple(replace(s, start=s.start-offset) for s in segs))


class EagerCaches:
    def __init__(self, real):
        self.real = real

    def __getattr__(self, name):
        return getattr(self.real, name)

    def subset(self, start, end):
        # Eager segments retain real sequence/slot IDs; GraphCaches instead
        # maps local row IDs through the device slot/block-table slices.
        return self


def model(kinds=("kda", "kda", "kda"), snapshots=4, comm=None):
    f = replace(tiny_facts(), kinds=kinds, block=64, spec_k=6)
    net = Glm53Net(f, comm or Comm(4, 0), reference())
    if comm is None:
        net.comm = Comm()  # CPU rank-arithmetic oracle, not a TP performance claim
    gen = torch.Generator().manual_seed(213)
    net.p = {s.name: (torch.randn(s.shape, generator=gen) * .04).to(s.dtype) for s in net.specs()}
    for key, value in net.p.items():
        if "norm" in key or key.endswith("o_norm"):
            value.fill_(1)
    p = layout(f, net.layers)
    snap = snapshot_layout(f, net.layers)[0]
    arena = Arena(p.nbytes(24, 4) + snapshots * snap + 4096, device="cpu")
    cache = Glm53Caches(arena, f, net.layers, 24, 4, snapshots=snapshots)
    return net, cache


class ExecutionPlanTests(unittest.TestCase):
    def test_only_c4_splits_and_invalid_budgets_fail(self):
        p = ExecutionPlan(overlap=True, prefill_tiles=4)
        self.assertEqual(p.groups(4), ((0, 2), (2, 4)))
        for n in (1, 2, 3):
            self.assertEqual(p.groups(n), ((0, n),))
        for kw in ({"prefill_tiles": 3}, {"prefill_tiles": True}, {"tile_rows": 2305}, {"overlap": 1}):
            with self.assertRaises(ValueError):
                ExecutionPlan(**kw)

    def test_tiles_rebase_patches_and_keep_only_internal_marks(self):
        values = torch.arange(5 * 128).reshape(5, 128).bfloat16()
        original = Step.prefill(torch.arange(150), 256, 2, 3,
                               ((torch.tensor([0, 63, 64, 127, 149]), values),),
                               ((32, 0), (64, 1), (128, 2)))
        tiles = list(prefill_steps(original, 64))
        self.assertEqual([s.segments[0].ctx for s in tiles], [256, 320, 384])
        self.assertEqual([s.marks for s in tiles], [((32, 0),), (), ()])
        self.assertEqual([s.patches[0][0].tolist() for s in tiles], [[0, 63], [0, 63], [21]])
        self.assertTrue(torch.equal(torch.cat([s.patches[0][1] for s in tiles]), values))

    def test_layer_order_matches_chunk_order_and_all_prefix_states(self):
        for kinds, context in ((("kda", "kda", "kda"), 0), (("kda", "dsa", "kda"), 64)):
            with self.subTest(kinds=kinds, context=context):
                self.check_layer_order(kinds, context)

    def check_layer_order(self, kinds, context):
        torch.set_num_threads(1)
        net, cache = model(kinds)
        slot = cache.slots.take(0)
        ids = torch.arange(192) % net.vp
        cache.pool.reserve(0, context + 192)
        if context:
            pre = Step.prefill(torch.arange(context) % net.vp, 0, 0, slot)
            cache.prepare(pre); net.forward(pre, cache)
        step = Step.prefill(ids, context, 0, slot, marks=((64, 0), (128, 1)))
        cache.prepare(step)
        before = cache.state.clone()
        paged_before = cache.paged.clone()
        expected, aux = [], []
        for start in (0, 64, 128):
            part = Step.prefill(ids[start:start+64], context + start, 0, slot)
            cache.prepare(part)
            h, a = net.forward(part, cache, aux_layers=[0, 2])
            expected.append(h); aux.append(a)
            if start < 128:
                cache.checkpoint(slot, context + start + 64, start // 64)
        want_state = cache.state.clone()
        want_paged = cache.paged.clone()
        want_marks = {k: v.clone() for k, v in cache._snap.items()}
        cache.state.copy_(before)
        cache.paged.copy_(paged_before)
        for v in cache._snap.values():
            v.zero_()
        h, a = prefill_layer_major(net, step, cache, NS(tile_rows=64, prefill_tiles=4), [0, 2])
        torch.testing.assert_close(h, torch.cat(expected), rtol=0, atol=0)
        torch.testing.assert_close(a, torch.cat(aux), rtol=0, atol=0)
        torch.testing.assert_close(cache.state, want_state, rtol=0, atol=0)
        torch.testing.assert_close(cache.paged, want_paged, rtol=0, atol=0)
        for k, v in cache._snap.items():
            torch.testing.assert_close(v[:2], want_marks[k][:2], rtol=0, atol=0)

    def test_c4_overlap_keeps_rows_state_and_auxiliary_order(self):
        for kinds in (("kda", "kda", "kda"), ("kda", "dsa", "kda")):
            with self.subTest(kinds=kinds):
                self.check_decode_order(kinds)

    def test_four_ranks_keep_collective_order_and_reduced_outputs(self):
        torch.set_num_threads(1)
        def rank(comm):
            return self.check_decode_order(("kda", "dsa", "kda"), comm=comm)
        results = LocalTP(4, timeout_s=15).run(rank)
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=0)

    def check_decode_order(self, kinds, comm=None):
        torch.set_num_threads(1)
        net, cache = model(kinds, comm=comm)
        chunks = []
        for seq, ctx in zip((2, 0, 3, 1), (0, 3, 9, 15)):
            slot = cache.slots.take(seq)
            cache.pool.reserve(seq, ctx + 7)
            if ctx:
                pre = Step.prefill(torch.arange(ctx) % net.vp, 0, seq, slot)
                cache.prepare(pre); net.forward(pre, cache)
            chunks.append(((torch.arange(7) + seq) % net.vp, ctx, seq, slot))
        raw = Step.decode(chunks)
        step = EagerStep(raw.ids, raw.segments)
        cache.prepare(step)
        before = cache.state.clone()
        paged_before = cache.paged.clone()
        h0, a0 = net.forward(step, cache, aux_layers=[0, 2])
        after = cache.state.clone()
        paged_after = cache.paged.clone()
        cache.state.copy_(before)
        cache.paged.copy_(paged_before)
        seen = []
        h1, a1 = decode_overlap(net, step, EagerCaches(cache), ExecutionPlan(overlap=True),
                                SerialStreams(), [0, 2], lambda a: seen.append(a.clone()))
        torch.testing.assert_close(h1, h0, rtol=0, atol=0)
        torch.testing.assert_close(a1, a0, rtol=0, atol=0)
        torch.testing.assert_close(cache.state, after, rtol=0, atol=0)
        torch.testing.assert_close(cache.paged, paged_after, rtol=0, atol=0)
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], a1, rtol=0, atol=0)
        return h1

    def test_oversized_window_refuses_before_mutating_state(self):
        net, cache = model()
        slot = cache.slots.take(0)
        step = Step.prefill(torch.zeros(129, dtype=torch.int64), 0, 0, slot)
        before = cache.state.clone()
        with self.assertRaisesRegex(ValueError, "activation window"):
            prefill_layer_major(net, step, cache, NS(tile_rows=64, prefill_tiles=2))
        self.assertTrue(torch.equal(before, cache.state))


if __name__ == "__main__":
    unittest.main()
