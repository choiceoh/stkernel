"""The probe's C=2 thread arms compute the batch forward's values (2026-09-16, the C=2 architecture question).

`probes/engine_prefill_chunk_profile.py --lanes threads` runs one 16-row C=2 step as two 8-row request threads on
one stream -- `execution.decode_overlap` with (0,1),(1,2) groups, and a variant that splits the attention side
only. Both must produce the batch forward's hidden rows, auxiliary features and cache state bit for bit on the CPU
reference lanes, or the probe would time a different computation.
"""
import unittest

import torch

from engine.profiles.glm53.execution import SerialStreams, decode_overlap
from engine.profiles.glm53.net import Step
from probes.engine_prefill_chunk_profile import ThreadPlan, decode_overlap_attn
from tests.test_engine_execution_plans import EagerCaches, EagerStep, model


class ThreadArmTests(unittest.TestCase):
    def test_thread_plan_splits_only_two_requests(self):
        plan = ThreadPlan(overlap=True)
        self.assertEqual(plan.groups(2), ((0, 1), (1, 2)))
        for n in (1, 3, 4):
            self.assertEqual(plan.groups(n), ((0, n),))
        self.assertEqual(ThreadPlan().groups(2), ((0, 2),))
        with self.assertRaises(ValueError):
            plan.groups(0)

    def test_both_sides_split_matches_the_batch(self):
        for kinds in (("kda", "kda", "kda"), ("kda", "dsa", "kda")):
            with self.subTest(kinds=kinds):
                self.check(kinds, decode_overlap)

    def test_attention_side_split_matches_the_batch(self):
        for kinds in (("kda", "kda", "kda"), ("kda", "dsa", "kda")):
            with self.subTest(kinds=kinds):
                self.check(kinds, decode_overlap_attn)

    def check(self, kinds, arm):
        torch.set_num_threads(1)
        net, cache = model(kinds)
        tokens = net.F.spec_k + 1
        chunks = []
        for seq, ctx in zip((1, 0), (3, 9)):          # two requests, out of slot order, different contexts
            slot = cache.slots.take(seq)
            cache.pool.reserve(seq, ctx + tokens)
            pre = Step.prefill(torch.arange(ctx) % net.vp, 0, seq, slot)
            cache.prepare(pre)
            net.forward(pre, cache)
            chunks.append(((torch.arange(tokens) + seq) % net.vp, ctx, seq, slot))
        raw = Step.decode(chunks)
        step = EagerStep(raw.ids, raw.segments)
        cache.prepare(step)
        before, paged_before = cache.state.clone(), cache.paged.clone()
        h0, a0 = net.forward(step, cache, aux_layers=[0, 2])
        after, paged_after = cache.state.clone(), cache.paged.clone()
        cache.state.copy_(before)
        cache.paged.copy_(paged_before)
        seen = []
        h1, a1 = arm(net, step, EagerCaches(cache), ThreadPlan(overlap=True), SerialStreams(), [0, 2],
                     lambda a: seen.append(a.clone()))
        torch.testing.assert_close(h1, h0, rtol=0, atol=0)
        torch.testing.assert_close(a1, a0, rtol=0, atol=0)
        torch.testing.assert_close(cache.state, after, rtol=0, atol=0)
        torch.testing.assert_close(cache.paged, paged_after, rtol=0, atol=0)
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], a1, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
