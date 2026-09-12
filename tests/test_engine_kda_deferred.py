"""Deferred KDA versus the ordinary ring: outputs, acceptance and checkpoints."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and importlib.util.find_spec("triton"), "requires CUDA and Triton")
class DeferredKdaTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.kda.ring import recurrent_kda_ring
        from engine.kernels.kda.deferred import verify, commit
        self.baseline, self.verify, self.commit = recurrent_kda_ring, verify, commit
        torch.manual_seed(20260912)

    def inputs(self, t, h=16, d=128, v=128):
        # Production Q/K/V views share a wider conv output; beta is strided.
        q, k, vv = (torch.randn(1, t, h, n*3, device="cuda", dtype=torch.bfloat16)[..., :n]
                    for n in (d, d, v))
        g = torch.randn(1, t, h, d, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, t, h*3, device="cuda", dtype=torch.bfloat16)[..., :h]
        a, bias = torch.randn(h, device="cuda")*.2, torch.randn(h*d, device="cuda")*.1
        width = h*d*v
        storage = torch.randn(3*(8*width+64)+64, device="cuda")*.1
        ring = storage.as_strided((3, 8, h, d, v), (8*width+64, width, d*v, v, 1), 64)
        return (q, k, vv, g, beta, a, bias), storage, ring

    def exact(self, a, b):
        self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    def expected(self, baseline, original, slot, context, count, block):
        expected = original.clone()
        if count:
            for i in range(count):
                p = context+i
                if i == count-1 or (p+1) % block == 0:
                    expected[slot, p % 8].copy_(baseline[slot, p % 8])
        return expected

    def test_all_acceptance_lengths_prefix_boundaries_and_padding(self):
        for t in (1, 2, 6, 7, 8):
            args, backing, ring = self.inputs(t)
            seed = backing.clone()
            original = ring.clone()
            for context in (0, 1, 7, 767, 4095, 131071):
                baseline = original.clone()
                out = self.baseline(*args, baseline, 2, context, -5.)
                slot, ctx = (torch.tensor(x, device="cuda") for x in (2, context))
                for count in range(t+1):
                    with self.subTest(t=t, context=context, accepted=count):
                        backing.copy_(seed)
                        actual, factors = self.verify(*args, ring, slot, ctx, -5.)
                        self.exact(backing, seed)  # verification is read-only
                        self.exact(actual, out)
                        self.commit(factors, ring, slot, ctx, torch.tensor(count, device="cuda"), block=768)
                        expected = seed.clone()
                        view = expected.as_strided(ring.shape, ring.stride(), ring.storage_offset())
                        view.copy_(self.expected(baseline, original, 2, context, count, 768))
                        self.exact(backing, expected)

    def test_graph_replay_changes_slots_contexts_and_acceptance(self):
        for t in (1, 6, 7):
            args, backing, ring = self.inputs(t)
            slot, ctx, count = (torch.tensor(x, device="cuda", dtype=torch.int32) for x in (1, 0, t))
            self.verify(*args, ring, slot, ctx, -5.)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual, factors = self.verify(*args, ring, slot, ctx, -5.)
                self.commit(factors, ring, slot, ctx, count, block=768)
            try:
                for physical, context, accepted in ((2, 0, 1), (2, 1, t), (1, 767, 2 if t > 1 else 1),
                                                     (2, 8, 0), (1, 769, t), (2, 131071, t)):
                    for x in args[:5]:
                        x.normal_()
                    original, baseline = ring.clone(), ring.clone()
                    expected_out = self.baseline(*args, baseline, physical, context, -5.)
                    slot.fill_(physical); ctx.fill_(context); count.fill_(accepted)
                    graph.replay()
                    self.exact(actual, expected_out)
                    self.exact(ring, self.expected(baseline, original, physical, context, accepted, 768))
            finally:
                graph.reset()

    def test_tail_dimensions_and_zero_context_ignore_stale_nan(self):
        args, _, ring = self.inputs(7, 2, 33, 17)
        ring.fill_(float("nan"))
        baseline = ring.clone()
        expected = self.baseline(*args, baseline, 1, 0, -5.)
        slot, context, count = (torch.tensor(x, device="cuda") for x in (1, 0, 7))
        out, factors = self.verify(*args, ring, slot, context, -5.)
        self.commit(factors, ring, slot, context, count, block=4)
        self.exact(out, expected)
        self.exact(ring[1, 3], baseline[1, 3])
        self.exact(ring[1, 6], baseline[1, 6])


if __name__ == "__main__":
    unittest.main()
