"""The fused sampler kernel against the definition it implements (45차 §28).

`engine/base/sampler._rows_by_sorting` is the definition, written with a sort. `engine/kernels/sampler`
searches for the same threshold without one. These check that they are the same sampler: the same
truncation, the same distribution, the same law for the draw -- and they pin what "the same" can mean
for two fp32 summations that add the row up in different orders.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from engine.base.sampler import _rows_by_sorting, rows, threshold  # noqa: E402

CUDA = torch.cuda.is_available()


def sorted_reference(logits, temps, ks, ps, valid=None):
    """[M, V] distributions from the sorting definition, whatever device the logits are on."""
    out = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.float32, device=logits.device)
    _rows_by_sorting(logits, temps, ks, ps, torch.zeros(logits.shape[0], device=logits.device), valid, out)
    return out


def policy(M, device, temperature=1.0, top_k=0, top_p=1.0):
    return (torch.full((M,), temperature, dtype=torch.float32, device=device),
            torch.full((M,), top_k, dtype=torch.int32, device=device),
            torch.full((M,), top_p, dtype=torch.float32, device=device))


class ThresholdTests(unittest.TestCase):
    """The definition itself, on the host: what top-k and top-p keep."""

    def test_top_p_keeps_the_smallest_prefix_that_reaches_the_mass(self):
        w = torch.tensor([1.0, 0.5, 0.25, 0.125])          # mass 1.875
        self.assertEqual(threshold(w, None, 0.5), 1.0)      # the first alone is 53%
        self.assertEqual(threshold(w, None, 0.9), 0.25)     # 1 + 0.5 + 0.25 = 93%
        self.assertEqual(threshold(w, None, 1.0), 0.0)      # off

    def test_top_k_keeps_the_kth_largest_and_everything_tied_with_it(self):
        w = torch.tensor([1.0, 0.5, 0.5, 0.1])
        self.assertEqual(threshold(w, 2, None), 0.5)
        self.assertEqual(float((w >= threshold(w, 2, None)).sum()), 3.0, "a tie at the cut is kept, not broken")

    def test_top_p_sits_on_top_of_top_k(self):
        w = torch.tensor([1.0, 0.5, 0.25, 0.125])
        # top-k 2 leaves mass 1.5; 0.9 of that is 1.35, which the first two reach and the first does not
        self.assertEqual(threshold(w, 2, 0.9), 0.5)
        self.assertEqual(threshold(w, 2, 0.6), 1.0)

    def test_a_row_with_one_live_token_keeps_it(self):
        self.assertEqual(threshold(torch.tensor([1.0, 0.0, 0.0]), None, 0.9), 1.0)


@unittest.skipUnless(CUDA, "requires CUDA")
class KernelAgreesWithTheSortTests(unittest.TestCase):
    def blocks(self, M, V, scale=2.0, seed=0, dtype=torch.float32, **kw):
        g = torch.Generator(device="cuda").manual_seed(seed)
        logits = (torch.randn(M, V, device="cuda", generator=g) * scale).to(dtype)
        t, k, p = policy(M, "cuda", **kw)
        valid = kw.pop("valid", None)
        mine = torch.empty(M, V, dtype=torch.float32, device="cuda")
        rows(logits, t, k, p, torch.rand(M, device="cuda", generator=g), valid, mine)
        return logits, mine, sorted_reference(logits, t, k, p, valid)

    def assertSameSampler(self, mine, ref, label=""):
        """Two fp32 summations of the same row do not have to agree at the boundary -- but they may
        disagree only there. The sweep behind these bounds is 1,800 shapes (45차 §28)."""
        differ = (mine > 0) != (ref > 0)
        self.assertLessEqual(int(differ.sum(1).max()), 2, f"{label}: more than a boundary token apart")
        moved = torch.where(differ, torch.maximum(mine, ref), torch.zeros((), device=mine.device))
        self.assertLess(float(moved.sum(1).max()), 1e-4, f"{label}: the tokens they disagree on carry real mass")
        self.assertLess(float((mine - ref).abs()[~differ].max()), 1e-5, f"{label}: the shared tokens differ")

    def test_the_truncations_over_a_serving_shaped_block(self):
        for label, kw in (("top_p 0.9", dict(top_p=0.9)), ("top_p 0.1", dict(top_p=0.1)),
                          ("top_k 40", dict(top_k=40)), ("both", dict(top_k=40, top_p=0.8)),
                          ("neither", {}), ("top_k 1", dict(top_k=1))):
            with self.subTest(label):
                _, mine, ref = self.blocks(8, 154880, **kw)
                self.assertSameSampler(mine, ref, label)

    def test_flat_peaked_narrow_and_bfloat16_rows(self):
        for label, kw in (("flat", dict(scale=0.01, top_p=0.9)), ("peaked", dict(scale=30.0, top_p=0.9)),
                          ("bf16", dict(dtype=torch.bfloat16, top_p=0.9)),
                          ("short vocab", dict(V=777, top_p=0.9)), ("tiny vocab", dict(V=64, top_p=0.9))):
            V = kw.pop("V", 154880)
            with self.subTest(label):
                _, mine, ref = self.blocks(6, V, **kw)
                self.assertSameSampler(mine, ref, label)

    def test_a_greedy_row_is_one_hot_at_its_argmax_and_draws_nothing(self):
        g = torch.Generator(device="cuda").manual_seed(5)
        logits = torch.randn(4, 4096, device="cuda", generator=g)
        t, k, p = policy(4, "cuda", temperature=0.0)
        probs = torch.empty(4, 4096, dtype=torch.float32, device="cuda")
        ids = rows(logits, t, k, p, torch.rand(4, device="cuda", generator=g), None, probs)
        self.assertTrue(torch.equal(ids, logits.argmax(-1)))
        self.assertTrue(torch.equal(probs.sum(1), torch.ones(4, device="cuda")))
        self.assertTrue(torch.equal(probs.argmax(1), logits.argmax(-1)))

    def test_the_undecodable_tail_is_never_picked_or_given_mass(self):
        g = torch.Generator(device="cuda").manual_seed(6)
        logits = torch.randn(4, 512, device="cuda", generator=g)
        logits[:, 300:] += 20.0                                     # orphans the tokenizer never decodes
        t, k, p = policy(4, "cuda", top_p=0.9)
        probs = torch.empty(4, 512, dtype=torch.float32, device="cuda")
        ids = rows(logits, t, k, p, torch.rand(4, device="cuda", generator=g), 300, probs)
        self.assertTrue(bool((ids < 300).all()))
        self.assertEqual(float(probs[:, 300:].sum()), 0.0)

    def test_a_row_that_only_wants_the_distribution_does_not_draw(self):
        g = torch.Generator(device="cuda").manual_seed(7)
        logits = torch.randn(3, 2048, device="cuda", generator=g)
        t, k, p = policy(3, "cuda", top_p=0.9)
        probs = torch.empty(3, 2048, dtype=torch.float32, device="cuda")
        self.assertIsNone(rows(logits, t, k, p, None, None, probs))
        drawn = torch.empty(3, 2048, dtype=torch.float32, device="cuda")
        rows(logits, t, k, p, torch.rand(3, device="cuda"), None, drawn)
        self.assertTrue(torch.equal(probs, drawn), "the distribution must not depend on whether a draw followed")

    def test_the_draw_reproduces_the_truncated_law(self):
        g = torch.Generator(device="cuda").manual_seed(8)
        one = torch.randn(1, 512, device="cuda", generator=g) * 2
        n = 40000
        block = one.expand(n, -1).contiguous()
        t, k, p = policy(n, "cuda", top_p=0.9)
        ids = rows(block, t, k, p, torch.rand(n, device="cuda", generator=g))
        seen = torch.bincount(ids, minlength=512).float() / n
        want = sorted_reference(one, *policy(1, "cuda", top_p=0.9))[0]
        self.assertLess(float((seen - want).abs().max()), 0.01)
        self.assertEqual(float(seen[want == 0].sum()), 0.0, "a draw landed outside the nucleus")

    def test_the_same_uniform_gives_the_same_token_on_every_rank(self):
        g = torch.Generator(device="cuda").manual_seed(9)
        logits = torch.randn(8, 30000, device="cuda", generator=g) * 2
        t, k, p = policy(8, "cuda", top_p=0.9)
        u = torch.rand(8, device="cuda", generator=g)
        self.assertTrue(torch.equal(rows(logits, t, k, p, u), rows(logits, t, k, p, u)))

    def test_a_rows_pick_does_not_depend_on_who_shares_its_batch(self):
        """Rank agreement is per row: the same logits and the same uniform, in any company."""
        g = torch.Generator(device="cuda").manual_seed(10)
        logits = torch.randn(6, 30000, device="cuda", generator=g) * 2
        t, k, p = policy(6, "cuda", top_p=0.9)
        u = torch.rand(6, device="cuda", generator=g)
        together = rows(logits, t, k, p, u)
        for i in range(6):
            alone = rows(logits[i:i + 1], t[i:i + 1], k[i:i + 1], p[i:i + 1], u[i:i + 1])
            self.assertEqual(int(alone[0]), int(together[i]))

    def test_the_caller_s_logits_are_never_written(self):
        g = torch.Generator(device="cuda").manual_seed(11)
        logits = torch.randn(4, 8192, device="cuda", generator=g)
        before = logits.clone()
        t, k, p = policy(4, "cuda", top_k=50, top_p=0.9)
        rows(logits, t, k, p, torch.rand(4, device="cuda", generator=g), 8000)
        self.assertTrue(torch.equal(logits, before))

    def test_a_mixed_batch_is_one_launch(self):
        g = torch.Generator(device="cuda").manual_seed(12)
        logits = torch.randn(5, 20000, device="cuda", generator=g) * 2
        t = torch.tensor([0.0, 0.7, 1.0, 1.3, 0.0], dtype=torch.float32, device="cuda")
        k = torch.tensor([0, 40, 0, 5, 0], dtype=torch.int32, device="cuda")
        p = torch.tensor([1.0, 0.9, 0.5, 1.0, 1.0], dtype=torch.float32, device="cuda")
        probs = torch.empty(5, 20000, dtype=torch.float32, device="cuda")
        rows(logits, t, k, p, torch.rand(5, device="cuda", generator=g), None, probs)
        self.assertSameSampler(probs, sorted_reference(logits, t, k, p), "mixed")


@unittest.skipUnless(CUDA, "requires CUDA")
class NoHostCrossingTests(unittest.TestCase):
    def test_the_sampler_never_waits_for_the_device(self):
        """What `torch.multinomial` cost and vLLM works around with V exponentials a row: the whole
        sampler has to be queueable, or the decode pipeline's whole point is gone."""
        g = torch.Generator(device="cuda").manual_seed(13)
        logits = torch.randn(6, 30000, device="cuda", generator=g)
        t, k, p = policy(6, "cuda", top_k=50, top_p=0.9)
        u = torch.rand(6, device="cuda", generator=g)
        probs = torch.empty(6, 30000, dtype=torch.float32, device="cuda")
        rows(logits, t, k, p, u, None, probs)                       # compile before the mode is armed
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            rows(logits, t, k, p, u, None, probs)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_it_captures_into_a_cuda_graph(self):
        from engine.base.sampler import sample
        g = torch.Generator(device="cuda").manual_seed(14)
        logits = torch.randn(4, 16384, device="cuda", generator=g)
        t = torch.ones(4, device="cuda")
        p = torch.full((4,), 0.9, device="cuda")
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(g)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sample(logits, t, p, g)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            picked = sample(logits, t, p, g)
        graph.replay()
        torch.cuda.synchronize()
        first = picked.clone()
        graph.replay()
        torch.cuda.synchronize()
        self.assertFalse(torch.equal(first, picked) and bool((first == first[0]).all()),
                         "a replay that cannot move at all is not drawing")
        self.assertTrue(bool(((picked >= 0) & (picked < 16384)).all()))


if __name__ == "__main__":
    unittest.main()
