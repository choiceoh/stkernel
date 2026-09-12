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

    def make_full_drafter(self, seed=3):
        """The tiny drafter with every checkpoint tensor random, a vocabulary-shaped fake target and a draft field."""
        from engine.base.comm import Comm
        from engine.profiles.glm53.drafter import ring_cells, specs
        d = self.make_drafter()
        F = d.F
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(seed)
        rand = lambda *shape: (torch.randn(*shape, device=dev, generator=gen) * 0.3).bfloat16()
        d.p = {s.name: (torch.ones(*s.shape, device=dev, dtype=torch.bfloat16) if s.name.endswith("norm.weight") else rand(*s.shape))
               for s in specs(F)}
        vocab = 21
        table, head = rand(vocab, F.hidden), rand(vocab, F.hidden)
        d.target = SimpleNamespace(embed=lambda ids: table[ids], head_local=lambda h: torch.nn.functional.linear(h, head),
                                   comm=Comm(), rank=0, vp=vocab)
        field = rand(4, F.layers, 2, ring_cells(F), F.kv_heads, F.head_dim)
        return d, field, dev

    def test_rows_propose_matches_the_per_row_walk(self):
        """Three rows in three slots -- two with a context shorter than the window, one longer -- at once: the same
        drafts as three per-row walks, and the block's hidden within bf16 of the per-row block."""
        d, field, dev = self.make_full_drafter()
        F = d.F
        slots = torch.tensor([1, 3, 2], device=dev)
        ctx = torch.tensor([5, 12, 3], device=dev)
        anchors = torch.tensor([7, 3, 9], device=dev)
        expect = [d.propose_tensor(anchors[i:i + 1], int(ctx[i]), field[int(slots[i])].clone()) for i in range(3)]
        got = d.propose_rows(field, slots, anchors, ctx)
        self.assertEqual(got.tolist(), [e.tolist() for e in expect])
        t = F.k + 1
        ids = torch.cat([anchors.view(3, 1), torch.full((3, F.k), F.mask_id, device=dev)], 1).reshape(-1)
        pos = (ctx.view(3, 1) + torch.arange(t, device=dev)).reshape(-1)
        rows = d.block_rows(ids, pos, slots, ctx, field, 3, t).view(3, t, -1)
        for i in range(3):
            one = d.block(ids.view(3, t)[i], pos.view(3, t)[i], field[int(slots[i])], int(ctx[i]))
            torch.testing.assert_close(rows[i].float(), one.float(), atol=6e-2, rtol=6e-2)
        self.assertTrue(torch.isfinite(rows.float()).all())

    def test_rows_observe_writes_the_valid_cells_in_place(self):
        d, field, dev = self.make_full_drafter(seed=4)
        F = d.F
        t = F.k + 1
        slots = torch.tensor([2, 1], device=dev)
        positions = torch.tensor([[5, 6, 7, 8], [13, 14, 15, 16]], device=dev)
        valid = torch.tensor([3, 0], device=dev)
        aux = (torch.randn(2 * t, F.hidden, device=dev) * 0.3).bfloat16()
        expect = field.clone()
        for i in range(2):
            d.observe_masked(expect[int(slots[i])], positions[i], aux[i * t:(i + 1) * t], valid[i])
        d.observe_rows(field, slots, positions, aux, valid)
        torch.testing.assert_close(field.float(), expect.float(), atol=2e-2, rtol=2e-2)
        self.assertTrue(torch.equal(field[3], expect[3]), "an untouched slot is untouched")
        self.assertTrue(torch.equal(field[1], expect[1]), "a row with no valid position writes nothing")

    def test_rows_sampled_walk_keeps_greedy_rows_greedy_and_hands_back_distributions(self):
        d, field, dev = self.make_full_drafter(seed=5)
        F = d.F
        slots = torch.tensor([1, 2, 3], device=dev)
        ctx = torch.tensor([9, 4, 30], device=dev)
        anchors = torch.tensor([2, 7, 11], device=dev)
        temps = torch.tensor([0.8, 0.0, 0.5], device=dev)
        greedy = d.propose_rows(field, slots, anchors, ctx)
        drafts, dists = d.propose_rows(field, slots, anchors, ctx, temps=temps, generator=torch.Generator(device=dev).manual_seed(9), vocab=21)
        self.assertEqual(tuple(dists.shape), (3, F.k, 21))
        self.assertEqual(drafts[1].tolist(), greedy[1].tolist(), "a row at temperature 0 walks greedily")
        for r in range(3):
            for s in range(F.k):
                self.assertGreater(float(dists[r, s, drafts[r, s]]), 0.0)
                self.assertAlmostEqual(float(dists[r, s].sum()), 1.0, places=4)
                self.assertLessEqual(int((dists[r, s] > 0).sum()), F.sel_top_k)
        self.assertTrue(torch.equal(dists[1] > 0, torch.nn.functional.one_hot(greedy[1], 21).bool()), "one-hot on the greedy pick")

    def test_the_host_walk_crosses_once_not_twice_a_position(self):
        source = (Path(__file__).resolve().parents[1] / "engine/profiles/glm53/drafter.py").read_text()
        body = source[source.index("    def propose_sampled(self"):source.index("    def propose_sampled_tensor(")]
        self.assertNotIn(".item()", body)
        self.assertIn("propose_sampled_tensor(", body)


if __name__ == "__main__":
    unittest.main()
