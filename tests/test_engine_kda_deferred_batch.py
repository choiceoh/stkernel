"""Accepted-only FP32 state materialization across rows and layer arena views."""
import unittest

import torch


def fixture(rows, tokens, layers=3, h=16, k=128, v=128, width=7):
    cell = h*k*v
    layer_stride = width*cell+64
    slot_stride = layers*layer_stride+64
    storage = torch.randn((rows+2)*slot_stride+64, device="cuda", dtype=torch.float32)*.1
    rings = [storage.as_strided((rows+2, width, h, k, v), (slot_stride, cell, k*v, v, 1),
                                64+i*layer_stride) for i in range(layers)]
    args = []
    for _ in range(layers):
        q, key, value = (torch.randn(1, rows*tokens, h, d*3, device="cuda", dtype=torch.bfloat16)[..., :d]
                          for d in (k, k, v))
        g = torch.randn(1, rows*tokens, h, k, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, rows*tokens, h*3, device="cuda", dtype=torch.bfloat16)[..., :h]
        args.append((q, key, value, g, beta, torch.randn(h, device="cuda")*.2,
                     torch.randn(h*k, device="cuda")*.1))
    return args, storage, rings


def views(storage, rings):
    return [storage.as_strided(r.shape, r.stride(), r.storage_offset()) for r in rings]


@unittest.skipUnless(torch.cuda.is_available(), "requires admitted GB10")
class DeferredBatchTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.kda.deferred import Batch
        from engine.kernels.kda.ring import recurrent_kda_ring_rows
        self.Batch, self.baseline = Batch, recurrent_kda_ring_rows
        torch.manual_seed(91317)

    def exact(self, actual, expected):
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))

    def expected(self, original, reference, rings, slots, contexts, counts, block=768):
        expected = original.clone()
        for out, source in zip(views(expected, rings), views(reference, rings)):
            for slot, context, count in zip(slots.tolist(), contexts.tolist(), counts.tolist()):
                for i in range(count):
                    pos = context+i
                    if i == count-1 or (pos+1) % block == 0:
                        out[slot, pos % out.shape[1]].copy_(source[slot, pos % out.shape[1]])
        return expected

    def test_all_counts_and_distinct_rows_keep_exact_outputs_states_and_padding(self):
        for rows, tokens in ((1, 6), (1, 7), (4, 7), (1, 8), (4, 8)):
            args, storage, rings = fixture(rows, tokens, width=max(7, tokens))
            original = storage.clone()
            batch = self.Batch(rings, rows, tokens, block=768)
            slots = torch.arange(rows, 0, -1, device="cuda")
            contexts = torch.tensor([0, 767, 32767, 131071][:rows], device="cuda")
            reference = original.clone()
            want = [self.baseline(*a, r, slots, contexts, -5.) for a, r in zip(args, views(reference, rings))]
            for count in range(tokens+1):
                counts = (torch.arange(rows, device="cuda")+count) % (tokens+1)
                storage.copy_(original)
                actual = [batch.verify(i, *a, slots, contexts, -5.) for i, a in enumerate(args)]
                self.exact(storage, original)
                for a, b in zip(actual, want):
                    self.exact(a, b)
                batch.commit(slots, contexts, counts)
                self.exact(storage, self.expected(original, reference, rings, slots, contexts, counts))

    def test_replay_rebinds_slots_and_counts_after_rollback_and_ring_wrap(self):
        for rows in (1, 4):
            args, storage, rings = fixture(rows, 8, h=2, k=33, v=17, width=8)
            batch = self.Batch(rings, rows, 8, block=4)
            slots = torch.arange(1, rows+1, device="cuda")
            contexts = torch.zeros(rows, dtype=torch.int64, device="cuda")
            counts = torch.ones_like(contexts)
            def step():
                out = [batch.verify(i, *a, slots, contexts, -5.) for i, a in enumerate(args)]
                batch.commit(slots, contexts, counts)
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
                    if trial == 3:
                        contexts.fill_(767)
                    if trial == 9:
                        contexts.fill_(131071)
                    for a in args:
                        for value in a[:5]:
                            value.normal_()
                    original, reference = storage.clone(), storage.clone()
                    expected = [self.baseline(*a, r, slots, contexts, -5.) for a, r in zip(args, views(reference, rings))]
                    graph.replay()
                    for a, b in zip(actual, expected):
                        self.exact(a, b)
                    self.exact(storage, self.expected(original, reference, rings, slots, contexts, counts, block=4))
            finally:
                graph.reset()

    def test_factor_and_ring_ownership_is_checked_before_writes(self):
        from engine.kernels.kda.deferred import verify_rows
        args, storage, rings = fixture(4, 7, layers=2, h=2, k=33, v=17)
        slots = torch.arange(1, 5, device="cuda")
        contexts = torch.zeros_like(slots)
        with self.assertRaisesRegex(ValueError, "FP32"):
            self.Batch([rings[0].half()], 4, 7, block=768)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.Batch([rings[0], rings[0]], 4, 7, block=768)
        batch = self.Batch(rings, 4, 7, block=768)
        factors = batch.layer_factors[0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_rows(*args[0], rings[0], slots, contexts, -5., factors=(factors[0], factors[0], factors[2]))
        with self.assertRaisesRegex(ValueError, "integer vectors"):
            batch.commit(slots, contexts, torch.zeros(8, device="cuda", dtype=torch.int64)[::2])

    def test_materialization_matches_flat_reference_at_large_contexts_and_every_boundary(self):
        # Test the changed commit directly: large contexts exercise int64
        # cursor setup without asking the unchanged verifier to consume them.
        for h, k, v, width, block in ((16, 128, 128, 7, 768), (16, 128, 128, 8, 768), (2, 33, 17, 7, 4),
                                     (1, 5, 9, 1, 1), (2, 17, 33, 8, 3)):
            with self.subTest(shape=(h, k, v), width=width, block=block):
                _, original, rings = fixture(4, width, layers=2, h=h, k=k, v=v, width=width)
                storage = [original.clone() for _ in range(5)]
                owners = [self.Batch(views(storage[0], rings), 4, width, block=block, tiled=False)]
                owners += [self.Batch(views(x, rings), 4, width, block=block, cells=cells)
                           for x, cells in zip(storage[1:], (1024, 2048, 4096))]
                owners.append(self.Batch(views(storage[4], rings), 4, width, block=block, cells=4096, warps=8))
                for src in owners[0].factors:
                    src.normal_(0, .2)
                for owner in owners[1:]:
                    for dst, src in zip(owner.factors, owners[0].factors):
                        dst.copy_(src)
                slots = torch.tensor([4, 1, 3, 2], device="cuda")
                contexts = torch.tensor([0, block-1, (1 << 32)+block-1, (1 << 48)+width-1], device="cuda")
                for count in range(width+1):
                    counts = (torch.arange(4, device="cuda")+count) % (width+1)
                    for x, owner in zip(storage, owners):
                        x.copy_(original)
                        owner.commit(slots, contexts, counts)
                    for x in storage[1:]:
                        self.exact(x, storage[0])
                for invalid in (-1, width+1):
                    counts.fill_(invalid)
                    for x, owner in zip(storage[1:], owners[1:]):
                        x.copy_(original)
                        owner.commit(slots, contexts, counts)
                        self.exact(x, original)

    def test_four_iteration_conditional_graph_commits_before_reusing_factors(self):
        from engine.kernels.bounded_graph import BoundedGraph
        for rows in (1, 4):
            args, storage, rings = fixture(rows, 8, h=2, k=33, v=17, width=8)
            batch = self.Batch(rings, rows, 8, block=4)
            slots = torch.arange(1, rows+1, device="cuda")
            contexts = torch.full((rows,), 767, dtype=torch.int64, device="cuda")
            program = (torch.arange(4*rows, device="cuda").view(4, rows)*3+1) % 9
            counter = torch.zeros(1, device="cuda", dtype=torch.int64)
            stop = torch.zeros_like(counter)
            log = torch.empty((4, len(args), 1, rows*8, 2, 17), device="cuda", dtype=torch.bfloat16)
            initial = storage.clone()
            expected_outputs = []
            for i in range(4):
                before, reference = storage.clone(), storage.clone()
                expected_outputs.append(torch.stack([self.baseline(*a, r, slots, contexts, -5.)
                                                     for a, r in zip(args, views(reference, rings))]))
                storage.copy_(self.expected(before, reference, rings, slots, contexts, program[i], block=4))
                contexts.add_(program[i])
            expected_state, expected_context = storage.clone(), contexts.clone()
            def body():
                counts = program.index_select(0, counter).squeeze(0)
                out = torch.stack([batch.verify(i, *a, slots, contexts, -5.) for i, a in enumerate(args)])
                batch.commit(slots, contexts, counts)
                contexts.add_(counts)
                log.index_copy_(0, counter, out.unsqueeze(0))
            storage.copy_(initial); contexts.fill_(767)
            body()  # compile every operation before capture
            storage.copy_(initial); contexts.fill_(767)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                body()
            loop = BoundedGraph(graph, counter, stop, 4, owners=(batch, slots, contexts, program, log))
            try:
                loop.replay()
                self.exact(log, torch.stack(expected_outputs))
                self.exact(storage, expected_state)
                self.exact(contexts, expected_context)
            finally:
                loop.close()
                graph.reset()


if __name__ == "__main__":
    unittest.main()
