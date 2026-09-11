"""Sparse recurrent transfers preserve rollback history and arena ownership."""
import importlib.util
from dataclasses import replace
from types import SimpleNamespace
import unittest

torch = None
if importlib.util.find_spec('torch'):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires CUDA PyTorch')
class StateCacheTests(unittest.TestCase):
    def test_strided_arena_gather_reads_only_the_correct_predecessor(self):
        from engine.kernels.state_cache import gather_ring
        for width in (64, 1089, 16 * 128 * 128):
            # Nonzero storage offset and padding between physical slots.
            arena = torch.randn(7 * (6 * width + 64) + 64, device='cuda')
            source = arena.as_strided((7, 6, width), (6 * width + 64, width, 1), 64)
            slots = torch.tensor([5, 1, 3], device='cuda')
            for contexts in ([0, 1, 7], [5, 6, 32768]):
                ctx = torch.tensor(contexts, device='cuda')
                scratch = gather_ring(source, slots, ctx)
                for i, (slot, c) in enumerate(zip([5, 1, 3], contexts)):
                    self.assertTrue(torch.equal(scratch[i, (c-1) % 6], source[slot, (c-1) % 6]))

    def test_commit_never_copies_poisoned_unwritten_positions(self):
        from engine.kernels.state_cache import commit_ring
        for tokens in (1, 6):
            arena = torch.randn(5 * (6 * 1089 + 64) + 64, device='cuda')
            before = arena.clone()
            destination = arena.as_strided((5, 6, 1089), (6 * 1089 + 64, 1089, 1), 64)
            expected = before.as_strided(destination.shape, destination.stride(), 64)
            slots = torch.tensor([3, 1], device='cuda')
            ctx = torch.tensor([5, 32768], device='cuda')
            scratch = torch.full((2, 6, 1089), float('nan'), device='cuda')
            for i, (slot, c) in enumerate(zip([3, 1], [5, 32768])):
                for t in range(tokens):
                    row = (c+t) % 6
                    scratch[i, row].fill_(100*i + t + 1)
                    expected[slot, row].copy_(scratch[i, row])
            commit_ring(scratch, destination, slots, ctx, tokens)
            self.assertTrue(torch.equal(arena, before))

    def test_graph_replay_remaps_slots_and_preserves_rejected_future_states(self):
        from engine.base.arena import Arena
        from engine.profiles.glm53.caches import Glm53Caches, layout
        from engine.profiles.glm53.decode_graphs import GraphCaches
        from test_engine_glm53 import tiny_facts
        F = replace(tiny_facts(), kda_dim=33)
        draft = (1, 8, 1, 8)
        p = layout(F, range(F.layers), draft)
        arena = Arena(256 + p.nbytes(8, 4))
        prefix = arena.carve(256, 'unrelated owner').fill_(0xEE)
        caches = Glm53Caches(arena, F, range(F.layers), 8, 4, draft=draft)
        for value in caches._fields.values():
            value.copy_(torch.randn_like(value))
        initial = caches.state.clone()
        for tokens in (1, 6):
            seqs = torch.tensor([2, 0], device='cuda')
            slots = torch.tensor([3, 1], device='cuda')
            contexts = torch.tensor([7, 11], device='cuda')
            scratch = GraphCaches(caches, seqs, slots, 128,
                                  SimpleNamespace(contexts=contexts, tokens=tokens))

            def work():
                scratch.gather()
                for i in range(2):
                    conv, ring = scratch.kda(0, i)
                    prev = ring.index_select(0, ((contexts[i]-1) % 6).reshape(1))
                    prev = prev.masked_fill(contexts[i] <= 0, 0)
                    for t in range(tokens):
                        ring.index_copy_(0, ((contexts[i]+t) % 6).reshape(1), prev+t+1)
                    conv.add_(1)
                    scratch.tail(1, i).add_(2)
                scratch.commit()
                scratch.fields.clear()
                del scratch.block_table

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                work()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                work()
            caches.state.copy_(initial)
            # The next replay follows a rejected verify without clearing its
            # future state, then exercises request slot reversal and reuse.
            for slot_ids, ctx in [([3, 1], [0, 5]), ([3, 1], [1, 6]),
                                 ([1, 3], [7, 2]), ([4, 2], [32768, 1])]:
                slots.copy_(torch.tensor(slot_ids))
                contexts.copy_(torch.tensor(ctx))
                before = caches.state.clone()
                # Full-ring oracle executes the model's actual position rule.
                for slot, c in zip(slot_ids, ctx):
                    conv, ring = caches.kda(0, slot)
                    prev = ring[(c-1) % 6].clone() if c > 0 else torch.zeros_like(ring[0])
                    for t in range(tokens):
                        ring[(c+t) % 6].copy_(prev+t+1)
                    conv.add_(1)
                    caches.tail(1, slot).add_(2)
                expected = caches.state.clone()
                caches.state.copy_(before)
                graph.replay()
                self.assertTrue(torch.equal(caches.state, expected))
                self.assertTrue(torch.all(prefix == 0xEE))
            graph.reset()


if __name__ == '__main__':
    unittest.main()
