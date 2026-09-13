"""Real cache/graph publication, including the aliased recurrent prefix stage."""
from dataclasses import replace
import unittest

import torch

from tests.test_engine_compact_serving import cache_for
from tests.test_engine_glm53 import tiny_facts
from tests.test_engine_kda_deferred_batch import fixture


@unittest.skipUnless(torch.cuda.is_available(), "requires admitted GB10")
class CompactServingCudaTests(unittest.TestCase):
    def exact(self, actual, expected):
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))

    def test_c1_c4_four_graph_commits_stage_restore_and_rebind(self):
        from engine.kernels.kda.ring import recurrent_kda_ring_rows
        torch.manual_seed(89513)
        for rows in (1, 4):
            with self.subTest(rows=rows):
                f = replace(tiny_facts(), layers=2, kinds=("kda", "kda"), block=768,
                            spec_k=7, kda_heads=64, kda_dim=128)
                base = cache_for(f, device="cuda")
                compact = cache_for(replace(f, kda_state_layout="committed_boundary"), device="cuda")
                args, _, _ = fixture(rows, 8, layers=2, width=8)
                slots = torch.arange(rows, 0, -1, device="cuda", dtype=torch.int64)
                # Near 32K/128K, cross actual 768-token prefix boundaries
                # (33,024 and 131,328), not power-of-two context buckets.
                initial_contexts = torch.tensor([766, 767, 33021, 131325][:rows], device="cuda")
                counts = torch.arange(1, rows+1, device="cuda", dtype=torch.int64)
                for L in range(2):
                    for slot, ctx in zip(slots.tolist(), initial_contexts.tolist()):
                        state = torch.randn_like(compact.kda(L, slot)[1][0])*.01
                        compact.kda(L, slot)[1][0].copy_(state)
                        base.kda(L, slot)[1][(ctx-1) % 8].copy_(state)
                    base._fields["conv", L].normal_()
                    compact._fields["conv", L].copy_(base._fields["conv", L])
                for slot, ctx in zip(slots.tolist(), initial_contexts.tolist()):
                    compact._fields["rec_meta", -1][slot, 0] = ctx
                owner = compact.deferred_batch(rows, 8)
                contexts = initial_contexts.clone()
                saved_state, saved_stage = compact.state.clone(), compact.stage_store.clone()

                def chain(caches, contexts, batch=None):
                    outputs = []
                    for iteration in range(4):
                        for L, inputs in enumerate(args):
                            if batch is None:
                                out = recurrent_kda_ring_rows(*inputs, caches._fields["rec", L],
                                                              slots, contexts, -5.)
                            else:
                                out = batch.verify(L, *inputs, slots, contexts, -5.)
                            outputs.append(out)
                        if batch is not None:
                            batch.commit(slots, contexts, counts)
                        caches.stage_boundaries(slots, contexts, counts)
                        contexts.add_(counts)
                    return outputs

                expected = chain(base, initial_contexts.clone())
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    chain(compact, contexts, owner)
                torch.cuda.current_stream().wait_stream(side)
                compact.state.copy_(saved_state)
                compact.stage_store.copy_(saved_stage)
                contexts.copy_(initial_contexts)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = chain(compact, contexts, owner)
                graph.replay()
                torch.cuda.synchronize()
                for a, b in zip(actual, expected):
                    self.exact(a, b)
                for slot, after in zip(slots.tolist(), contexts.tolist()):
                    boundary = (after // f.block)*f.block
                    self.assertEqual(compact._fields["rec_meta", -1][slot].tolist(), [after, boundary])
                    for L in range(2):
                        self.exact(compact.kda(L, slot)[1][0], base.kda(L, slot)[1][(after-1) % 8])
                        self.exact(compact._stage["rec", L][slot], base._stage["rec", L][slot])
                        self.exact(compact._stage["conv", L][slot], base._stage["conv", L][slot])
                    base.checkpoint_from_stage(slot, 0)
                    compact.checkpoint_from_stage(slot, 0, position=boundary)
                    for key in base._snap:
                        self.exact(compact._snap[key][0], base._snap[key][0])
                # Slot 1 is rebound after its graph/last reader completes.
                compact.reset_slot(1)
                compact.restore(1, boundary, 0)
                self.assertEqual(compact._fields["rec_meta", -1][1].tolist(), [boundary, 0])
                for L in range(2):
                    self.exact(compact.kda(L, 1)[1][0], compact._snap["rec", L][0])


if __name__ == "__main__":
    unittest.main()
