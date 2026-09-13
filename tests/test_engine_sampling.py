"""Sampling contracts at the GLM serving boundary, with no model weights."""
from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@unittest.skipUnless(torch is not None, "requires PyTorch")
class SamplingTests(unittest.TestCase):
    def engine(self, **kwargs):
        from engine.profiles.glm53.adapter import Glm53Engine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return Glm53Engine(None, SimpleNamespace(device=device), SimpleNamespace(spec_k=5), **kwargs)

    def test_greedy_excludes_orphans_preserves_ties_and_does_not_modify_logits(self):
        engine = self.engine(decodable=4, top_p=0.1)
        # A strided view, tied maxima, and a larger orphan logit on each row.
        values = torch.tensor([[1., 3., 3., -2., 99., 5.],
                               [-3., -2., -1., -4., 99., 5.]], device=engine.caches.device)
        logits = values.repeat_interleave(2, dim=1)[:, ::2]
        before = logits.clone()
        self.assertEqual(engine._sample(logits, [0., 0.]).tolist(), [1, 2], "greedy needs no uniforms")
        self.assertTrue(torch.equal(logits, before))

    def test_greedy_with_unpadded_or_unbounded_vocabulary(self):
        for decodable in (None, 3, 100):
            with self.subTest(decodable=decodable):
                engine = self.engine(decodable=decodable)
                logits = torch.tensor([[1., 2., 3.]], device=engine.caches.device)
                self.assertEqual(engine._sample(logits, [0.]).item(), 2)

    def test_stochastic_and_mixed_steps_match_the_base_sampler_at_the_same_uniforms(self):
        from engine.base.sampler import sample
        for temps in ([0.7, 1., 1.2], [0., 0.7, 1.], [1e-50, 1., 0.]):
            for top_p in (1., 0.8):
                with self.subTest(temps=temps, top_p=top_p):
                    engine = self.engine(decodable=61, top_p=top_p, seed=42)
                    dev = engine.caches.device
                    logits = torch.randn(3, 64, device=dev,
                                         generator=torch.Generator(device=dev).manual_seed(7))
                    before = logits.clone()
                    masked = logits.clone()
                    masked[:, 61:] = float("-inf")
                    t = torch.tensor(temps, dtype=torch.float32, device=dev)
                    p = torch.full((3,), top_p, device=dev)
                    for step in range(3):
                        u = torch.rand(3, generator=torch.Generator(device=dev).manual_seed(100 + step), device=dev)
                        expected = sample(masked, t, p, u)
                        self.assertTrue(torch.equal(engine._sample(logits, temps, uniforms=u.tolist()), expected))
                    self.assertTrue(torch.equal(logits, before))
                    with self.assertRaisesRegex(ValueError, "one uniform a row"):
                        engine._sample(logits, temps)

    def test_a_row_s_draws_key_on_what_they_are_for_and_never_on_what_came_before(self):
        """The engine holds no stream: a uniform is a hash of (seed, the row's admission nonce, its generation
        count) with (purpose, position). Greedy steps, neighbours and earlier draws move nothing (base/draws)."""
        from engine.base import draws
        a, b = self.engine(seed=19), self.engine(seed=19)
        for e in (a, b):
            e.tokens, e.prompt_len, e.nonces = {0: [1, 2, 3, 4], 5: [1, 2]}, {0: 2, 5: 2}, {0: 7, 5: 8}
        logits = torch.arange(32, device=a.caches.device).float()[None, :] / 32
        for _ in range(5):
            a._sample(logits, [0.])
        self.assertEqual(a._uniforms(0, draws.PICK, 3), b._uniforms(0, draws.PICK, 3))
        self.assertEqual(a._uniforms(0, draws.PICK, 3), draws.uniforms(draws.row_key(19, 7, 2), draws.PICK, 3))
        self.assertNotEqual(a._uniforms(0, draws.PICK, 3), a._uniforms(5, draws.PICK, 3), "another row, another nonce")
        self.assertNotEqual(a._uniforms(0, draws.PICK, 3), a._uniforms(0, draws.DRAFT, 3), "another purpose")
        a.tokens[0].append(9)
        self.assertEqual(a._uniforms(0, draws.PICK, 3), draws.uniforms(draws.row_key(19, 7, 3), draws.PICK, 3), "one more generated")
        a.seeds[0] = 4
        self.assertEqual(a._uniforms(0, draws.PICK, 3), draws.uniforms(draws.row_key(4, 0, 3), draws.PICK, 3),
                         "a seeded request keys on its seed alone, whichever row it landed on")
        self.assertEqual(a._uniform_tensor(0, draws.RICH, 4, a.caches.device).tolist(), a._uniforms(0, draws.RICH, 4))
        self.assertEqual(a._pick_uniforms([SimpleNamespace(seq=0, length=2), SimpleNamespace(seq=5, length=1)]),
                         a._uniforms(0, draws.PICK, 2) + a._uniforms(5, draws.PICK, 1))
        with self.assertRaises(KeyError):
            a._uniforms(9, draws.PICK, 1)                            # a row the engine never admitted has no key


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class GraphSamplingTests(unittest.TestCase):
    def test_graph_matches_seeded_eager_sampling_and_preserves_greedy_rng(self):
        from engine.profiles.glm53.decode_graphs import SamplingGraphs
        from engine.profiles.glm53.adapter import Glm53Engine
        from engine.base.comm import Comm
        for top_p in (1., .8):
            outputs = {(n, 1): (None, None, torch.empty(n, 64, device="cuda")) for n in (1, 3)}
            target = SimpleNamespace(tokens=1, graphs=SimpleNamespace(outputs=outputs),
                                     net=SimpleNamespace(comm=Comm(), rank=0, vp=64))
            graphs = SamplingGraphs(target, 61, top_p)
            try:
                eager = Glm53Engine(None, SimpleNamespace(device="cuda"), SimpleNamespace(spec_k=5),
                                     decodable=61, top_p=top_p, seed=19)
                inputs = torch.Generator(device="cuda").manual_seed(73)
                for temperatures in ([0.], [1.], [0., 0., 0.], [0., .7, 1.], [.5, 1., 1.2], [1.]):
                    shape = (len(temperatures), 1)
                    logits = torch.randn(len(temperatures), 64, device="cuda", generator=inputs)
                    outputs[shape][2].copy_(logits)
                    uniforms = torch.rand(len(temperatures), device="cuda", generator=inputs).tolist()
                    actual = graphs.run(shape, temperatures, uniforms=uniforms)
                    expected = eager._sample(logits, temperatures, uniforms=uniforms)
                    self.assertTrue(torch.equal(actual, expected), (top_p, temperatures))
                if any(t > 0 for t in temperatures):
                    with self.assertRaisesRegex(ValueError, "one uniform a row"):
                        graphs.run(shape, temperatures)
            finally:
                graphs.close()


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class SamplerSharingTests(unittest.TestCase):
    """One sampler per logits buffer: a row's capacity buckets share it (boot-time study)."""

    def target(self, outputs):
        from engine.base.comm import Comm
        return SimpleNamespace(tokens=6, graphs=SimpleNamespace(outputs=outputs),
                               net=SimpleNamespace(comm=Comm(), rank=0, vp=32))

    def test_capacity_buckets_of_one_row_capture_a_single_sampler(self):
        from engine.profiles.glm53.decode_graphs import SamplingGraphs
        shared = {n: torch.empty(n * 6, 32, device="cuda") for n in (1, 2)}
        outputs = {(n, 6, cap): (None, None, shared[n])
                   for n in (1, 2) for cap in (4096, 8192, 16384)}
        graphs = SamplingGraphs(self.target(outputs), None, 1.)
        try:
            self.assertEqual(sorted(graphs.greedy.graphs), [(1, 6), (2, 6)])      # not six
            self.assertEqual(sorted(graphs.stochastic.graphs), [(1, 6), (2, 6)])
            for cap in (4096, 8192, 16384):                                       # every bucket reaches it
                shared[1].normal_()
                picked = graphs.run((1, 6, cap), [0.] * 6)
                self.assertTrue(torch.equal(picked, shared[1].float().argmax(-1)))
        finally:
            graphs.close()

    def test_a_target_that_does_not_share_its_buffer_is_refused(self):
        from engine.profiles.glm53.decode_graphs import SamplingGraphs
        outputs = {(1, 6, cap): (None, None, torch.empty(6, 32, device="cuda"))
                   for cap in (4096, 8192)}
        with self.assertRaisesRegex(ValueError, "must share one logits buffer"):
            SamplingGraphs(self.target(outputs), None, 1.)


if __name__ == "__main__":
    unittest.main()
