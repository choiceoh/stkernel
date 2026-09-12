"""The candidate selector's greedy walk, fused against the loop it replaces (45차 §86).

The walk carries one integer across K positions. Fused it is one launch; as a loop it was five rounds of
advanced indexing and argmax. The only thing that has to hold is that it is the same walk, ties included --
a different tie is a different draft, and a different draft is a different acceptance rate.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from engine.kernels.draft_select import _by_torch, greedy_walk  # noqa: E402

CUDA = torch.cuda.is_available()


class ContractTests(unittest.TestCase):
    def test_what_the_walk_refuses(self):
        scores = torch.zeros(2, 5, 16, 16)
        cand = torch.zeros(2, 5, 16, dtype=torch.int64)
        with self.assertRaises(ValueError):
            greedy_walk(scores[0], cand)                      # scores are [n, K, prev, cand]
        with self.assertRaises(ValueError):
            greedy_walk(scores, cand[..., :8])                # the candidates the scores name
        with self.assertRaises(ValueError):
            greedy_walk(torch.zeros(2, 5, 8, 16), cand)       # a step's predecessors are the last step's candidates

    def test_the_reference_carries_the_choice_forward(self):
        """Position 0 always starts from predecessor 0; after that the row read is the one just chosen."""
        scores = torch.full((1, 3, 4, 4), -1.0)
        scores[0, 0, 0, 2] = 1.0                              # step 0 picks candidate 2
        scores[0, 1, 2, 3] = 1.0                              # step 1 must read predecessor row 2
        scores[0, 2, 3, 1] = 1.0
        cand = torch.arange(12, dtype=torch.int64).view(1, 3, 4)
        self.assertEqual(_by_torch(scores, cand).tolist(), [[2, 7, 9]])


@unittest.skipUnless(CUDA, "the fused walk is the CUDA path")
class KernelTests(unittest.TestCase):
    def test_it_is_the_same_walk_including_ties(self):
        for n, K, C in ((1, 5, 16), (4, 5, 16), (8, 3, 4), (1, 7, 32), (32, 5, 16)):
            for trial in range(12):
                with self.subTest(n=n, K=K, C=C, trial=trial):
                    gen = torch.Generator(device="cuda").manual_seed(n * 100 + K * 10 + trial)
                    scores = torch.randn(n, K, C, C, device="cuda", generator=gen)
                    cand = torch.randint(0, 154_880, (n, K, C), device="cuda", generator=gen)
                    if trial % 3 == 1:
                        scores = scores.round()               # ties everywhere the rounding collides
                    if trial % 3 == 2:
                        scores = torch.zeros_like(scores)     # every candidate tied: the walk is index 0 throughout
                    self.assertTrue(torch.equal(_by_torch(scores, cand), greedy_walk(scores, cand)))

    def test_a_walk_over_no_rows_is_no_drafts(self):
        empty = greedy_walk(torch.zeros(0, 5, 16, 16, device="cuda"),
                            torch.zeros(0, 5, 16, dtype=torch.int64, device="cuda"))
        self.assertEqual(tuple(empty.shape), (0, 5))




@unittest.skipUnless(CUDA, "the fused selector is the CUDA path")
class ScoreWalkTests(unittest.TestCase):
    """The scores and the walk in one launch, against the form that built the scores first (45차 §87).

    The torch form materialised [n, K, candidates, rank] twice -- a predecessor and a successor codebook row
    for every candidate of every step -- to read each once. Fused, a step holds one [rank] vector against the
    step's candidate rows, and the scores never exist.
    """
    def rows(self, n, K, C, R, vocab, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn(n, K, C, device="cuda", generator=gen),
                torch.randint(0, vocab, (n, K, C), device="cuda", generator=gen),
                torch.randint(0, vocab, (n,), device="cuda", generator=gen),
                torch.randn(n, K, R, device="cuda", generator=gen),
                torch.randn(vocab, R, device="cuda", generator=gen).bfloat16(),
                torch.randn(vocab, R, device="cuda", generator=gen).bfloat16())

    def test_it_is_the_walk_the_scores_used_to_produce(self):
        from engine.kernels.draft_select import _scores_by_torch, walk_scores
        for n, K, C, R, vocab in ((1, 5, 16, 256, 38_720), (4, 5, 16, 256, 38_720),
                                  (2, 3, 4, 64, 512), (1, 5, 16, 256, 1024)):
            for trial in range(8):
                with self.subTest(n=n, K=K, C=C, R=R, trial=trial):
                    parts = self.rows(n, K, C, R, vocab, n + K + C + trial)
                    self.assertTrue(torch.equal(_scores_by_torch(*parts), walk_scores(*parts)))

    def test_the_first_step_reads_the_anchor_and_the_rest_read_the_last_choice(self):
        """Zero the projection and every score is its unary term, so the walk is a plain per-step argmax --
        which makes what each step CARRIES visible: the token, not the index."""
        from engine.kernels.draft_select import walk_scores
        unary = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [3.0, 0.0, 0.0, 0.0]]], device="cuda")
        cand = torch.arange(12, device="cuda").view(1, 3, 4)
        got = walk_scores(unary, cand, torch.zeros(1, dtype=torch.int64, device="cuda"),
                          torch.zeros(1, 3, 8, device="cuda"),
                          torch.zeros(16, 8, device="cuda", dtype=torch.bfloat16),
                          torch.zeros(16, 8, device="cuda", dtype=torch.bfloat16))
        self.assertEqual(got.tolist(), [[1, 6, 8]])


if __name__ == "__main__":
    unittest.main()
