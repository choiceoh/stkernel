"""Every draw is a function of what it is for, never of what came before (45차, 2026-09-13).

The host hash and the device hash must agree bit for bit; a row key must separate seeds, rows and
generation counts, and a word must separate purposes and positions; the step block the chain
computes must be exactly what the host path asks for; and the serving modules must hold no
generator at all -- the uniforms are inputs everywhere (base/draws), which is what keeps four
ranks on one number whatever path each took to the step.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.base import draws                                          # noqa: E402

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class KeyTests(unittest.TestCase):
    def test_uniforms_lie_in_the_unit_interval_and_are_float32_values(self):
        k = draws.row_key(0, 1, 0)
        us = draws.uniforms(k, draws.PICK, 1000)
        self.assertTrue(all(0.0 <= u < 1.0 for u in us))
        self.assertTrue(all(u == draws._float32(u) for u in us), "already rounded to what the device produces")
        self.assertEqual(draws.uniforms(k, draws.PICK, 3, start=5), [draws.uniform(k, draws.PICK, 5 + i) for i in range(3)])

    def test_every_component_separates_the_draws(self):
        first = lambda seed, nonce, gen, purpose=draws.DRAFT, pos=0: draws.uniform(draws.row_key(seed, nonce, gen), purpose, pos)  # noqa: E731
        seen = {first(19, 7, 12)}
        for args in ((20, 7, 12), (19, 8, 12), (19, 7, 13)):
            seen.add(first(*args))
        for purpose in (draws.PICK, draws.VERIFY, draws.FRESH, draws.RICH):
            seen.add(first(19, 7, 12, purpose))
        seen.add(first(19, 7, 12, draws.DRAFT, 1))
        self.assertEqual(len(seen), 9, "seed, nonce, generation, purpose and position each move the draw")
        k = draws.row_key(19, 7, 12)
        self.assertEqual(len({draws.uniform(k, p, i) for p, i in draws.step_layout(5)}), 11)

    def test_a_word_names_one_purpose_and_position_and_refuses_what_it_cannot_hold(self):
        self.assertEqual(draws.word(draws.VERIFY, 4), (draws.VERIFY << 32) | 4)
        for bad in ((0, 0), (1 << 16, 0), (1, -1), (1, 1 << 32)):
            with self.assertRaises(ValueError):
                draws.word(*bad)

    def test_the_hash_is_a_pure_function_with_no_stream_behind_it(self):
        k = draws.row_key(3, 4, 5)
        first = draws.uniforms(k, draws.RICH, 6)
        for _ in range(3):
            draws.uniforms(draws.row_key(9, 9, 9), draws.DRAFT, 100)     # whatever else was drawn meanwhile
        self.assertEqual(draws.uniforms(k, draws.RICH, 6), first)
        self.assertEqual(draws.uniform(k, draws.RICH, 5), first[5], "a position alone, without the ones before it")
        self.assertLess(draws.mix(0), 1 << 64)

    def test_the_distribution_is_close_to_uniform(self):
        k = draws.row_key(1, 2, 3)
        bins = [0] * 10
        for u in draws.uniforms(k, draws.PICK, 20000):
            bins[int(u * 10)] += 1
        self.assertTrue(all(abs(b - 2000) < 200 for b in bins), bins)

    def test_the_step_layout_is_the_walk_then_the_verification_then_the_correction(self):
        self.assertEqual(draws.step_layout(2), [(draws.DRAFT, 0), (draws.DRAFT, 1), (draws.VERIFY, 0), (draws.VERIFY, 1),
                                                (draws.FRESH, 0)])


@unittest.skipUnless(torch is not None, "requires PyTorch")
class TensorAgreementTests(unittest.TestCase):
    def test_the_tensor_hash_matches_the_host_hash_bit_for_bit(self):
        import random
        rng = random.Random(7)
        for _ in range(300):
            seed, nonce, gen = rng.randrange(0, 1 << 64), rng.randrange(0, 1 << 40), rng.randrange(0, 1 << 20)
            purpose, start = rng.choice([draws.DRAFT, draws.PICK, draws.VERIFY, draws.FRESH, draws.RICH]), rng.randrange(0, 5)
            host = draws.uniforms(draws.row_key(seed, nonce, gen), purpose, 7, start=start)
            keys = draws.row_keys(seed, torch.tensor([nonce]), torch.tensor([gen]))
            dev = draws.uniform_tensor(keys, purpose, 7, start=start)[0]
            self.assertEqual(dev.dtype, torch.float32)
            self.assertEqual(host, dev.tolist(), (seed, nonce, gen, purpose, start))

    def test_the_step_block_is_what_the_host_path_asks_for_row_by_row(self):
        nonces, gens = torch.tensor([3, 4, 3]), torch.tensor([10, 10, 11])
        for K in (1, 5, 7):
            block = draws.step_block(19, nonces, gens, K)
            self.assertEqual(tuple(block.shape), (3, 2 * K + 1))
            for i in range(3):
                k = draws.row_key(19, int(nonces[i]), int(gens[i]))
                self.assertEqual(block[i].tolist(), [draws.uniform(k, p, j) for p, j in draws.step_layout(K)])
                self.assertEqual(block[i, :K].tolist(), draws.uniforms(k, draws.DRAFT, K), "the walk's")
                self.assertEqual(block[i, K:].tolist(), draws.uniforms(k, draws.VERIFY, K) + draws.uniforms(k, draws.FRESH, 1),
                                 "and the verification's: exactly the host's VERIFY then FRESH")
            self.assertNotEqual(block[0].tolist(), block[2].tolist(), "the same row one token later draws afresh")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_the_device_agrees_with_the_host_on_cuda_too(self):
        nonces = torch.tensor([1, 2, 3, 1 << 38], device="cuda")
        gens = torch.tensor([0, 5, 99, 1 << 19], device="cuda")
        block = draws.step_block(7, nonces, gens, 5).cpu()
        for i in range(4):
            k = draws.row_key(7, int(nonces[i]), int(gens[i]))
            self.assertEqual(block[i].tolist(), [draws.uniform(k, p, j) for p, j in draws.step_layout(5)])

    def test_the_sampler_and_the_verifier_take_them_as_inputs(self):
        from engine.base.sampler import block_verify, block_verify_batch, draw, sample
        torch.manual_seed(0)
        logits = torch.randn(3, 50)
        u = torch.tensor(draws.uniforms(draws.row_key(0, 1, 0), draws.PICK, 3))
        self.assertTrue(torch.equal(sample(logits, torch.ones(3), torch.ones(3), u),
                                    sample(logits, torch.ones(3), torch.ones(3), u.clone())))
        with self.assertRaisesRegex(ValueError, "one uniform a row"):
            sample(logits, torch.ones(3), torch.ones(3), u[:2])
        self.assertEqual(draw(torch.softmax(torch.randn(50), -1), 0.0), 0)
        K, V = 3, 11
        target = torch.softmax(torch.randn(K + 1, V), -1)
        dr = torch.softmax(torch.randn(K, V), -1)
        block = draws.step_block(0, torch.tensor([1]), torch.tensor([0]), K)
        one, new = block_verify(target, [1, 2, 3], dr, block[0, K:].tolist())
        self.assertEqual(len(new), one + 1)
        cand = torch.arange(V).view(1, 1, V).expand(1, K, V).contiguous()
        many, _, _ = block_verify_batch(target.unsqueeze(0), torch.tensor([[1, 2, 3]]), cand, dr.unsqueeze(0), block[:, K:])
        self.assertEqual(int(many[0]), one, "the host path and the chain decide alike from the same keyed uniforms")


class ContractTests(unittest.TestCase):
    def test_no_serving_module_holds_a_generator(self):
        for rel in ("engine/base/sampler.py", "engine/profiles/glm53/adapter.py", "engine/profiles/glm53/drafter.py",
                    "engine/profiles/glm53/decode_graphs.py", "engine/profiles/glm53/pipeline.py",
                    "engine/profiles/glm53/burst_decode.py"):
            text = (ROOT / rel).read_text()
            body = text.split("def _selfcheck", 1)[0]                       # a self-check may seed its own fixture
            self.assertNotIn("torch.Generator(", body, rel)
            self.assertNotIn("torch.multinomial(", body, rel)
            self.assertNotIn("generator=", body, rel)
            self.assertNotIn("register_generator_state", body, rel)

    def test_the_engine_keys_every_draw_it_makes(self):
        adapter = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        for line in ("self.seed = int(seed)", "self.nonces, self.admissions = {}, 0", "def _admitted(self, seq: int) -> None:",
                     "return draws.row_key(self.seed, self.nonces[seq], self._generated_count(seq))",
                     "return draws.row_key(seed, 0, self._generated_count(seq))",
                     "self._uniform_tensor(seq, draws.DRAFT, self.drafter.k, ring.device)",
                     "self._uniform_tensor(seq, draws.RICH, count, device)",
                     "self._uniforms(seq, draws.VERIFY, k) + self._uniforms(seq, draws.FRESH, 1)",
                     "self._uniforms(seq, draws.PICK, 1)", "self._pick_uniforms(step.segments)",
                     "draws_seed=self.seed, vocab=self.F.vocab",
                     "SamplingGraphs(self.decode_graphs, self.decodable, self.top_p)"):
            self.assertIn(line, adapter, line)
        self.assertEqual(adapter.count("self._admitted(seq)"), 3, "add, extend and resume: every new turn is a new nonce")
        pipeline = (ROOT / "engine/profiles/glm53/pipeline.py").read_text()
        self.assertIn('b["alive"], b["nonce"], b["generated"])', pipeline)
        self.assertIn('block = draws.step_block(e.seed, b["nonce"], b["generated"], K)', pipeline)
        self.assertIn('block_verify_batch(probs, b["drafts"], b["qcand"], b["qprob"], b["draws"])', pipeline)
        self.assertIn('nonce=self._upload([e.nonces[s] for s in seqs], torch.int64)', pipeline)
        graphs = (ROOT / "engine/profiles/glm53/decode_graphs.py").read_text()
        self.assertIn('block = draws.step_block(draws_seed, inputs["nonce"], inputs["generated"], drafter.k)', graphs)
        self.assertIn("def run(self, shape, temperatures, top_k=None, top_p=None, uniforms=None):", graphs)


if __name__ == "__main__":
    unittest.main()
