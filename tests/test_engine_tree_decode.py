"""Whole tiny GLM tree vs ordinary linear paths, including DSA and cache continuation."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from engine.base.comm import LocalTP
from engine.modules.speculative_tree import Tree
from engine.modules.sparse_indexer import topk_positions
from engine.profiles.glm53.net import Step
from engine.profiles.glm53.tree_decode import Verification
from tests.test_engine_execution_plans import model


class TreeDecodeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tree = Tree((1, 2, 3, 4, 5, 6), (-1, 0, 0, 1, 2, 4))

    def prepare(self, context, *, comm=None):
        net, cache = model(("kda", "dsa", "kda"), comm=comm)
        slot = cache.slots.take(0)
        cache.pool.reserve(0, context + 16)
        if context:
            pre = Step.prefill(torch.arange(context) % net.vp, 0, 0, slot)
            cache.prepare(pre)
            net.forward(pre, cache)
        return net, cache, slot

    def test_every_branch_matches_linear_target_with_prefix_and_pool_boundaries(self):
        for context in (0, 3, 7, 63, 129):
            with self.subTest(context=context):
                net, cache, slot = self.prepare(context)
                # Prepare before cloning the canonical state.
                run = Verification(net, cache, self.tree, seq=0, slot=slot, context=context)
                before, paged = cache.state.clone(), cache.paged.clone()
                actual = run.verify(aux_layers=(0, 2))
                self.assertTrue(torch.equal(before, cache.state))
                self.assertTrue(torch.equal(paged, cache.paged))
                run.abort()
                for node in range(len(self.tree.tokens)):
                    cache.state.copy_(before); cache.paged.copy_(paged)
                    tokens = torch.tensor([self.tree.tokens[i] for i in self.tree.path(node)])
                    step = Step.prefill(tokens, context, 0, slot)
                    cache.prepare(step)
                    expected, aux = net.forward(step, cache, aux_layers=(0, 2))
                    # BF16 projection batch shapes differ (tree vs one path).
                    torch.testing.assert_close(actual[node], expected[-1], rtol=.008, atol=.008)
                    torch.testing.assert_close(run.features[node], aux[-1], rtol=.008, atol=.008)
                    self.assertEqual(int(net.head_tokens(actual[node:node+1])[0]), int(net.head_tokens(expected[-1:])[0]))

    def test_commit_selected_state_boundary_and_next_step(self):
        for context, targets, budget in ((3, [3, 4, 5, 9, 6, 7], 8), (63, [3, 4, 5, 9, 6, 7], 2),
                                          (7, [22, 4, 5, 9, 6, 7], 8)):
            with self.subTest(context=context, budget=budget):
                net, cache, slot = self.prepare(context)
                run = Verification(net, cache, self.tree, seq=0, slot=slot, context=context)
                before, paged = cache.state.clone(), cache.paged.clone()
                run.verify(aux_layers=(0, 2))
                with patch.object(net, "head_tokens", return_value=torch.tensor(targets)):
                    result = run.commit(budget=budget)
                after, got_paged = cache.state.clone(), cache.paged.clone()
                next_step = Step.prefill(torch.tensor([result["tokens"][-1]]), result["context"], 0, slot)
                cache.prepare(next_step)
                got_next = net.forward(next_step, cache)
                cache.state.copy_(before); cache.paged.copy_(paged)
                path = result["path"]
                step = Step.prefill(torch.tensor([self.tree.tokens[i] for i in path]), context, 0, slot)
                cache.prepare(step); net.forward(step, cache)
                for L in net.layers:
                    if not net.F.is_dsa(L):
                        _, rec = cache.kda(L, slot)
                        for i in range(len(path)):
                            pos = context+i
                            if i == len(path)-1 or (pos+1) % net.F.block == 0:
                                # Arena view offsets: compare the selected ring cells, not abandoned history cells.
                                expected = rec[pos % net.rec_ring].clone()
                                cache.state.copy_(after)
                                torch.testing.assert_close(rec[pos % net.rec_ring], expected, rtol=0, atol=0)
                                cache.state.copy_(before)
                                net.forward(step, cache)
                self.assertTrue(torch.equal(cache.paged, got_paged))
                cache.prepare(next_step)
                torch.testing.assert_close(net.forward(next_step, cache), got_next, rtol=0, atol=0)
                self.assertEqual(result["features"].shape[0], len(path))
                self.assertIsNone(cache._tree_pending)

    def test_stale_cache_and_reused_owner_refused(self):
        net, cache, slot = self.prepare(3)
        run = Verification(net, cache, self.tree, seq=0, slot=slot, context=3)
        with self.assertRaisesRegex(RuntimeError, "already owns"):
            Verification(net, cache, self.tree, seq=0, slot=slot, context=3)
        run.verify()
        cache.state.add_(0)
        with self.assertRaisesRegex(RuntimeError, "cache changed"):
            run.commit(budget=1)
        run.abort()
        with self.assertRaises(RuntimeError):
            run.verify()

    def test_dsa_batches_queries_once_and_keeps_linear_topk_tie_sets(self):
        for context in (0, 3, 63, 129):
            net, cache, slot = self.prepare(context)
            calls, selected = [], []
            mla, compress, slots = (getattr(net.lanes, name) for name in ("mla_sparse", "kpool_compress", "pool_slots"))
            def score(q, keys, scales, w, ke):
                calls.append(("indexer", len(q), len(keys)))
                return torch.zeros((len(q), len(keys)), dtype=torch.float32)
            def attention(*args):
                calls.append(("mla", len(args[0])))
                return mla(*args)
            def pools(*args):
                calls.append(("compress", len(args[0])))
                return compress(*args)
            def positions(*args):
                selected.append(args[0].clone())
                return slots(*args)
            with patch.object(net, "lanes", replace(net.lanes, indexer_logits=score, mla_sparse=attention,
                                                   kpool_compress=pools, pool_slots=positions)):
                with Verification(net, cache, self.tree, seq=0, slot=slot, context=context) as run:
                    run.verify()
            self.assertEqual(sum(c[0] == "mla" for c in calls), 1)
            self.assertTrue(all(c[1] == len(self.tree.tokens) for c in calls if c[0] in ("mla", "indexer")))
            self.assertLessEqual(sum(c[0] == "compress" for c in calls), 1)
            for node, depth in enumerate(self.tree.depths):
                count, k = (context+depth+1)//net.F.kpool, net.F.topk//net.F.kpool
                expected = topk_positions(torch.zeros(1, count), k)
                torch.testing.assert_close(selected[0][node], expected[0], rtol=0, atol=0)

    def test_scratch_sampling_and_reservation_fail_before_state_mutation(self):
        net, cache, slot = self.prepare(0)
        before = cache.state.clone()
        for kw in ({"temperature": 1.}, {"max_scratch_bytes": 1}, {"context": 20}):
            args = dict(seq=0, slot=slot, context=0)
            args.update(kw)
            with self.assertRaises(ValueError):
                Verification(net, cache, self.tree, **args)
            self.assertTrue(torch.equal(before, cache.state))
        # A broad tree must not silently select DenseLinear's >32-row FP8
        # prefill weight lane instead of the target's W4A8 decode packs.
        broad = Tree(tuple(range(33)), (-1,)+(0,)*32)
        with self.assertRaisesRegex(ValueError, "W4A8 row"):
            Verification(net, cache, broad, seq=0, slot=slot, context=0)
        self.assertTrue(torch.equal(before, cache.state))

    def test_four_ranks_keep_tree_collectives_and_outputs_in_agreement(self):
        def rank(comm):
            net, cache, slot = self.prepare(3, comm=comm)
            with Verification(net, cache, self.tree, seq=0, slot=slot, context=3) as run:
                out = run.verify()
                result = run.commit(budget=3)
                return out, result["tokens"], result["path"]
        outputs = LocalTP(4, timeout_s=30).run(rank)
        for actual in outputs[1:]:
            torch.testing.assert_close(actual[0], outputs[0][0], rtol=0, atol=0)
            self.assertEqual(actual[1:], outputs[0][1:])


if __name__ == "__main__":
    unittest.main()
