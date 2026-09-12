"""DFlash context retention and selection across the device/host boundary."""
import importlib.util
from types import SimpleNamespace
import unittest
from pathlib import Path

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@unittest.skipUnless(torch is not None, "requires PyTorch")
class DrafterTests(unittest.TestCase):
    def make_drafter(self):
        from engine.profiles.glm53.drafter import Drafter, DrafterFacts
        F = DrafterFacts(layers=1, hidden=16, heads=2, kv_heads=1, head_dim=4,
                         inter=16, rms_eps=1e-6, rope_theta=10000., window=8,
                         block=4, mask_id=20, conv_taps=2, conv_group=4,
                         sel_rank=4, sel_top_k=3, target_layers=(1,), k=3)
        return Drafter(F, SimpleNamespace(), 21)

    def test_long_observation_retains_the_newest_window(self):
        from engine.profiles.glm53.drafter import rmsnorm, rope
        d = self.make_drafter()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(6)
        rand = lambda *shape: torch.randn(*shape, device=dev, generator=gen).bfloat16()
        d.p = {"fc.weight": rand(16, 16), "hidden_norm.weight": torch.ones(16, device=dev, dtype=torch.bfloat16),
               "layers.0.self_attn.k_proj.weight": rand(4, 16),
               "layers.0.self_attn.k_norm.weight": torch.ones(4, device=dev, dtype=torch.bfloat16),
               "layers.0.self_attn.v_proj.weight": rand(4, 16)}
        ring = torch.zeros(1, 2, 8, 1, 4, device=dev, dtype=torch.bfloat16)
        positions = torch.arange(3, 38, device=dev)
        aux = rand(35, 16)
        d.observe(ring, positions, aux)
        # Compute the retained positions directly, without repeated scatter ids.
        h = rmsnorm(torch.nn.functional.linear(aux[-8:], d.p["fc.weight"]),
                    d.p["hidden_norm.weight"], d.F.rms_eps).bfloat16()
        key = torch.nn.functional.linear(h, d.p["layers.0.self_attn.k_proj.weight"]).view(8, 1, 4)
        key = rope(rmsnorm(key, d.p["layers.0.self_attn.k_norm.weight"], d.F.rms_eps),
                   positions[-8:], d.F.rope_theta)
        value = torch.nn.functional.linear(h, d.p["layers.0.self_attn.v_proj.weight"]).view(8, 1, 4)
        self.assertTrue(torch.equal(ring[0, 0, positions[-8:] % 8], key))
        self.assertTrue(torch.equal(ring[0, 1, positions[-8:] % 8], value))

    def test_device_selector_matches_the_scalar_greedy_walk(self):
        d = self.make_drafter()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(71)
        rand = lambda *shape: torch.randn(*shape, device=dev, generator=gen).bfloat16()
        hidden, logits = rand(4, 16), rand(3, 21).float()
        d.block = lambda *args: hidden
        from engine.base.comm import Comm
        d.target.head_local = lambda h: logits.clone()
        d.target.comm, d.target.rank, d.target.vp = Comm(), 0, 21
        d.target.head = lambda h: self.fail("drafter gathered full-vocabulary logits")
        prefix = "candidate_selector."
        d.p = {prefix + "hidden_projection.weight": rand(4, 16),
               prefix + "predecessor_codebook": rand(21, 4),
               prefix + "successor_codebook": rand(21, 4)}
        anchor = 7
        unary, cand = logits.topk(3, dim=-1)
        projection = torch.nn.functional.linear(hidden[1:], d.p[prefix + "hidden_projection.weight"]).float()
        expected, previous = [], anchor
        for step in range(3):
            pred = d.p[prefix + "predecessor_codebook"][previous].float()
            succ = d.p[prefix + "successor_codebook"][cand[step]].float()
            score = unary[step] + ((pred * projection[step])[None, :] * succ).sum(-1)
            previous = int(cand[step, score.argmax()])
            expected.append(previous)
        actual = d.propose_tensor(torch.full((1,), anchor, device=dev, dtype=torch.int64), 9,
                                   torch.empty(0, device=dev))
        self.assertEqual(actual.tolist(), expected)

    def test_the_sampled_walk_hands_back_the_distribution_it_drew_from(self):
        """The accept test divides by this, so it has to be the one the pick came out of."""
        d = self.make_drafter()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(71)
        rand = lambda *shape: torch.randn(*shape, device=dev, generator=gen).bfloat16()
        hidden, logits = rand(4, 16), rand(3, 21).float()
        d.block = lambda *args: hidden
        from engine.base.comm import Comm
        d.target.head_local = lambda h: logits.clone()
        d.target.comm, d.target.rank, d.target.vp = Comm(), 0, 21
        prefix = "candidate_selector."
        d.p = {prefix + "hidden_projection.weight": rand(4, 16),
               prefix + "predecessor_codebook": rand(21, 4),
               prefix + "successor_codebook": rand(21, 4)}
        ring = torch.empty(0, device=dev)
        ids, dists = d.propose_sampled(7, 9, ring, 0.8, torch.Generator(device=dev).manual_seed(5), 21)
        same, same_dists = d.propose_sampled_tensor(torch.full((1,), 7, device=dev, dtype=torch.int64), 9, ring,
                                                    0.8, torch.Generator(device=dev).manual_seed(5), 21)
        self.assertEqual(ids, same.tolist(), "the host walk is the device walk, read back once at the end")
        self.assertTrue(torch.equal(dists, same_dists))
        for step, token in enumerate(ids):
            self.assertGreater(float(dists[step, token]), 0.0, "the pick has mass in what the verifier is given")
            self.assertAlmostEqual(float(dists[step].sum()), 1.0, places=4)
            self.assertLessEqual(int((dists[step] > 0).sum()), 3, "and the mass sits only on the candidates")

    def test_the_host_walk_crosses_once_not_twice_a_position(self):
        source = (Path(__file__).resolve().parents[1] / "engine/profiles/glm53/drafter.py").read_text()
        body = source[source.index("    def propose_sampled(self"):source.index("    def propose_sampled_tensor(")]
        self.assertNotIn(".item()", body)
        self.assertIn("propose_sampled_tensor(", body)


if __name__ == "__main__":
    unittest.main()
