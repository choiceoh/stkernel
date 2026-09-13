"""A committed cell plus a boundary cell versus the full FP32 rollback ring."""
import unittest

import torch

from tests.test_engine_kda_deferred_batch import fixture


def compact_storage(rows, layers, h, k, v):
    cell = h*k*v
    layer_stride = 2*cell+64
    slot_stride = layers*layer_stride+64
    storage = torch.randn((rows+2)*slot_stride+64, device="cuda", dtype=torch.float32)*.1
    current, boundary = [], []
    for layer in range(layers):
        offset = 64+layer*layer_stride
        current.append(storage.as_strided((rows+2, 1, h, k, v), (slot_stride, cell, k*v, v, 1), offset))
        boundary.append(storage.as_strided((rows+2, h, k, v), (slot_stride, k*v, v, 1), offset+cell))
    return storage, current, boundary


def clone_views(storage, values):
    return [storage.as_strided(x.shape, x.stride(), x.storage_offset()) for x in values]


@unittest.skipUnless(torch.cuda.is_available(), "requires admitted GB10")
class CompactStateTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.kda.deferred import Batch
        from engine.kernels.kda.ring import recurrent_kda_ring_rows
        self.Batch, self.baseline = Batch, recurrent_kda_ring_rows
        torch.manual_seed(91341)

    def exact(self, actual, expected):
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))

    def reference(self, args, rings, storage, current, boundary, slots, contexts, counts, block):
        # An ordinary ring's committed entry is the compact cell's initial
        # value. Other ring positions are irrelevant until verification.
        for ring, state in zip(rings, current):
            for slot, context in zip(slots.tolist(), contexts.tolist()):
                ring[slot, (context-1) % ring.shape[1]].copy_(state[slot, 0])
        outputs = [self.baseline(*a, ring, slots, contexts, -5.) for a, ring in zip(args, rings)]
        expected = storage.clone()
        for ring, state, edge in zip(rings, clone_views(expected, current), clone_views(expected, boundary)):
            for slot, context, count in zip(slots.tolist(), contexts.tolist(), counts.tolist()):
                if count:
                    state[slot, 0].copy_(ring[slot, (context+count-1) % ring.shape[1]])
                    for i in range(count):
                        if (context+i+1) % block == 0:
                            edge[slot].copy_(ring[slot, (context+i) % ring.shape[1]])
        return outputs, expected

    def test_all_counts_preserve_outputs_current_boundary_and_padding_exactly(self):
        # K=5/6/7 widths exercise both recurrence tiles. The boundary and
        # final positions deliberately collide under a naive two-cell ring.
        for rows, tokens in ((1, 1), (1, 6), (1, 7), (1, 8), (4, 8)):
            with self.subTest(rows=rows, tokens=tokens):
                args, _, rings = fixture(rows, tokens, layers=2, width=tokens)
                storage, current, boundary = compact_storage(rows, 2, 16, 128, 128)
                original = storage.clone()
                owner = self.Batch(current, rows, tokens, block=768, boundaries=boundary)
                slots = torch.arange(rows, 0, -1, device="cuda")
                for start in (0, 767, 768, 131071):
                    contexts = torch.tensor([start, 766, 32767, 131071][:rows], device="cuda")
                    for count in range(tokens+1):
                        counts = (torch.arange(rows, device="cuda")+count) % (tokens+1)
                        storage.copy_(original)
                        want, expected = self.reference(args, rings, storage, current, boundary,
                                                        slots, contexts, counts, 768)
                        actual = [owner.verify(i, *a, slots, contexts, -5.) for i, a in enumerate(args)]
                        self.exact(storage, original)
                        for a, b in zip(actual, want):
                            self.exact(a, b)
                        owner.commit(slots, contexts, counts)
                        self.exact(storage, expected)

    def test_graph_replay_rebinds_slots_after_rejection_and_prefix_restore(self):
        for rows in (1, 4):
            args, _, rings = fixture(rows, 8, layers=2, h=2, k=33, v=17, width=8)
            storage, current, boundary = compact_storage(rows, 2, 2, 33, 17)
            owner = self.Batch(current, rows, 8, block=8, boundaries=boundary)
            slots = torch.arange(1, rows+1, device="cuda")
            contexts = torch.zeros(rows, device="cuda", dtype=torch.int64)
            counts = torch.ones_like(contexts)
            def step():
                out = [owner.verify(i, *a, slots, contexts, -5.) for i, a in enumerate(args)]
                owner.commit(slots, contexts, counts)
                return out
            step()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = step()
            try:
                for trial in range(24):
                    slots.copy_(slots.roll(1))
                    contexts.add_(counts)
                    counts.copy_((torch.arange(rows, device="cuda")+trial) % 9)
                    if trial in (3, 9):
                        # The future cache adapter restores directly into
                        # current, with no dependence on position modulo K.
                        contexts.fill_(768 if trial == 3 else 131072)
                        for value in current:
                            value.normal_(0, .1)
                    for a in args:
                        for value in a[:5]:
                            value.normal_()
                    want, expected = self.reference(args, rings, storage, current, boundary,
                                                    slots, contexts, counts, 8)
                    graph.replay()
                    for a, b in zip(actual, want):
                        self.exact(a, b)
                    self.exact(storage, expected)
            finally:
                graph.reset()

    def test_four_commits_in_one_graph_read_the_new_current(self):
        for rows in (1, 4):
            args, _, rings = fixture(rows, 8, layers=2, h=2, k=33, v=17, width=8)
            storage, current, boundary = compact_storage(rows, 2, 2, 33, 17)
            owner = self.Batch(current, rows, 8, block=768, boundaries=boundary)
            slots = torch.arange(1, rows+1, device="cuda")
            contexts = torch.full((rows,), 767, device="cuda", dtype=torch.int64)
            program = (torch.arange(4*rows, device="cuda").view(4, rows)*3+1) % 9
            initial = storage.clone()
            expected_outputs = []
            for counts in program:
                want, expected = self.reference(args, rings, storage, current, boundary,
                                                slots, contexts, counts, 768)
                expected_outputs.extend(want)
                storage.copy_(expected)
                contexts.add_(counts)
            expected_state, expected_contexts = storage.clone(), contexts.clone()
            def body():
                outputs = []
                for i in range(4):
                    outputs += [owner.verify(layer, *a, slots, contexts, -5.) for layer, a in enumerate(args)]
                    owner.commit(slots, contexts, program[i])
                    contexts.add_(program[i])
                return outputs
            def reset():
                storage.copy_(initial)
                contexts.fill_(767)
            reset()
            body()
            reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = body()
            try:
                reset()
                graph.replay()
                for a, b in zip(actual, expected_outputs):
                    self.exact(a, b)
                self.exact(storage, expected_state)
                self.exact(contexts, expected_contexts)
            finally:
                graph.reset()

    def test_commit_large_positions_and_invalid_counts(self):
        _, _, rings = fixture(4, 8, layers=2, h=2, k=33, v=17, width=8)
        source = self.Batch(rings, 4, 8, block=768)
        storage, current, boundary = compact_storage(4, 2, 2, 33, 17)
        owner = self.Batch(current, 4, 8, block=768, boundaries=boundary)
        for dst, src in zip(owner.factors, source.factors):
            src.normal_(0, .1)
            dst.copy_(src)
        original = storage.clone()
        slots = torch.tensor([4, 1, 3, 2], device="cuda")
        contexts = torch.tensor([0, 767, (1 << 32)-1, (1 << 48)-1], device="cuda")
        for count in range(9):
            counts = (torch.arange(4, device="cuda")+count) % 9
            storage.copy_(original)
            for ring, state in zip(rings, current):
                for slot, context in zip(slots.tolist(), contexts.tolist()):
                    ring[slot, (context-1) % 8].copy_(state[slot, 0])
            source.commit(slots, contexts, counts)
            owner.commit(slots, contexts, counts)
            for ring, state, edge, old_state, old_edge in zip(
                    rings, current, boundary, clone_views(original, current), clone_views(original, boundary)):
                for slot, context, count in zip(slots.tolist(), contexts.tolist(), counts.tolist()):
                    self.exact(state[slot, 0], ring[slot, (context+count-1) % 8] if count else old_state[slot, 0])
                    position = (context+count)//768*768
                    self.exact(edge[slot], ring[slot, (position-1) % 8] if position > context else old_edge[slot])
        for invalid in (-1, 9):
            storage.copy_(original)
            owner.commit(slots, contexts, torch.full_like(contexts, invalid))
            self.exact(storage, original)

    def test_compact_ownership_rejects_aliases_and_multiple_boundaries(self):
        storage, current, boundary = compact_storage(4, 2, 2, 33, 17)
        with self.assertRaisesRegex(ValueError, "at most one"):
            self.Batch(current, 4, 8, block=7, boundaries=boundary)
        with self.assertRaisesRegex(ValueError, "one boundary view"):
            self.Batch(current, 4, 8, block=768, boundaries=[])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.Batch(current, 4, 8, block=768, boundaries=[x[:, 0] for x in current])
        with self.assertRaisesRegex(ValueError, "arena"):
            self.Batch(current, 4, 8, block=768, boundaries=[x.clone() for x in boundary])
        # One-cell storage is never admitted to the ordinary rollback writer.
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.Batch(current, 4, 8, block=768)


if __name__ == "__main__":
    unittest.main()
