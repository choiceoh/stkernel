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
        state = engine.gen.get_state().clone()
        self.assertEqual(engine._sample(logits, [0., 0.]).tolist(), [1, 2])
        self.assertTrue(torch.equal(logits, before))
        self.assertTrue(torch.equal(engine.gen.get_state(), state))

    def test_greedy_with_unpadded_or_unbounded_vocabulary(self):
        for decodable in (None, 3, 100):
            with self.subTest(decodable=decodable):
                engine = self.engine(decodable=decodable)
                logits = torch.tensor([[1., 2., 3.]], device=engine.caches.device)
                self.assertEqual(engine._sample(logits, [0.]).item(), 2)

    def test_stochastic_and_mixed_steps_match_seeded_base_sampler(self):
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
                    ref_gen = torch.Generator(device=dev).manual_seed(42)
                    t = torch.tensor(temps, dtype=torch.float32, device=dev)
                    p = torch.full((3,), top_p, device=dev)
                    for _ in range(3):
                        expected = sample(masked, t, p, ref_gen)
                        self.assertTrue(torch.equal(engine._sample(logits, temps), expected))
                        self.assertTrue(torch.equal(engine.gen.get_state(), ref_gen.get_state()))
                    self.assertTrue(torch.equal(logits, before))

    def test_greedy_steps_do_not_shift_later_stochastic_draws(self):
        a, b = self.engine(seed=19), self.engine(seed=19)
        logits = torch.arange(32, device=a.caches.device).float()[None, :] / 32
        for _ in range(5):
            a._sample(logits, [0.])
        for _ in range(5):
            self.assertTrue(torch.equal(a._sample(logits, [1.]), b._sample(logits, [1.])))


if __name__ == "__main__":
    unittest.main()
