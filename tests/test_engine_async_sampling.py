"""The device half of a decode batch running ahead of the host (45차 §23 B3): rejection sampling and the commit
clip over whole batches, judged against the scalar host versions in base/sampler and adapter._commit's rules."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from engine.base.sampler import commit_batch, speculative_pick_batch  # noqa: E402


class SpeculativePickBatchTests(unittest.TestCase):
    def test_identical_distributions_accept_every_draft_and_draw_the_bonus_from_the_target(self):
        g = torch.Generator().manual_seed(0)
        V, K, n = 50, 5, 3
        probs = torch.softmax(torch.randn(n, K + 1, V, generator=g), -1)
        drafts = torch.randint(0, V, (n, K), generator=g)
        accepted, tokens, count = speculative_pick_batch(probs, drafts, probs[:, :K], torch.rand(n, K + 1, generator=g))
        self.assertEqual(accepted.tolist(), [K] * n)
        self.assertEqual(count.tolist(), [K + 1] * n)
        self.assertTrue(torch.equal(tokens[:, :K], drafts))
        self.assertTrue(((tokens[:, K] >= 0) & (tokens[:, K] < V)).all())

    def test_disjoint_supports_reject_the_first_draft_and_recover_from_the_target(self):
        g = torch.Generator().manual_seed(1)
        V, K, n = 40, 4, 2
        target = torch.zeros(n, K + 1, V); target[:, :, :20] = 1 / 20                    # the target lives on 0..19
        draft = torch.zeros(n, K, V); draft[:, :, 20:] = 1 / 20                            # the drafter on 20..39
        drafts = torch.randint(20, V, (n, K), generator=g)
        accepted, tokens, count = speculative_pick_batch(target, drafts, draft, torch.rand(n, K + 1, generator=g))
        self.assertEqual(accepted.tolist(), [0, 0])
        self.assertEqual(count.tolist(), [1, 1])
        self.assertTrue((tokens[:, 0] < 20).all())                                          # recovered from the target's support

    def test_partial_acceptance_recovers_at_the_first_rejection(self):
        g = torch.Generator().manual_seed(2)
        V, K = 30, 3
        target = torch.full((1, K + 1, V), 1 / V)
        draft = torch.full((1, K, V), 1 / V)
        draft[0, 1] = 0; draft[0, 1, 5] = 1.0                                               # position 1: the drafter is sure of 5, the target is not
        drafts = torch.tensor([[7, 5, 9]])
        uniforms = torch.tensor([[0.5, 0.9, 0.5, 0.5]])                                     # position 1's uniform is above 1/30
        accepted, tokens, count = speculative_pick_batch(target, drafts, draft, uniforms)
        self.assertEqual(accepted.item(), 1)                                                # p/q = 1 at 0 accepts; at 1 p/q = 1/30 (u rejects)
        self.assertEqual(count.item(), 2)
        self.assertEqual(tokens[0, 0].item(), 7)
        self.assertNotEqual(tokens[0, 1].item(), 5)                                         # recovered from target - draft: never the draft

    def test_the_same_uniforms_give_the_same_picks_on_every_rank(self):
        """The uniforms are inputs (base/draws): what makes them the same on every rank is their key, not a
        stream every rank must have advanced alike."""
        V, K, n = 64, 5, 4
        probs = torch.softmax(torch.randn(n, K + 1, V, generator=torch.Generator().manual_seed(3)), -1)
        draft = torch.softmax(torch.randn(n, K, V, generator=torch.Generator().manual_seed(4)), -1)
        drafts = torch.randint(0, V, (n, K), generator=torch.Generator().manual_seed(5))
        uniforms = torch.rand(n, K + 1, generator=torch.Generator().manual_seed(9))
        a = speculative_pick_batch(probs, drafts, draft, uniforms)
        b = speculative_pick_batch(probs, drafts, draft, uniforms.clone())
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        with self.assertRaisesRegex(ValueError, "needs uniforms"):
            speculative_pick_batch(probs, drafts, draft, uniforms[:, :K])


class CommitBatchTests(unittest.TestCase):
    def test_greedy_acceptance_limit_and_end_tokens_clip_as_the_host_does(self):
        K = 3
        picks = torch.tensor([[1, 2, 3, 4],      # all drafts confirmed: 4 tokens
                              [1, 9, 3, 4],      # first draft confirmed, second corrected: 2 tokens
                              [1, 2, 3, 4],      # limit leaves room for 2
                              [1, 7, 3, 4],      # 7 is an end token at position 1: 2 tokens, done
                              [1, 2, 3, 4]])     # not alive: nothing
        drafts = torch.tensor([[1, 2, 3]] * 5)
        alive = torch.tensor([True, True, True, True, False])
        generated = torch.tensor([0, 0, 8, 0, 0])
        limit = torch.tensor([10, 10, 10, 10, 10])
        ends = torch.tensor([[7, -1]] * 5)
        count, done, accepted, tokens = commit_batch(picks, drafts, alive, generated, limit, ends)
        self.assertEqual(count.tolist(), [4, 2, 2, 2, 0])
        self.assertEqual(done.tolist(), [False, False, True, True, False])
        self.assertEqual(accepted.tolist(), [3, 1, 1, 1, 0])
        self.assertTrue(torch.equal(tokens, picks))

    def test_a_row_reaching_its_limit_exactly_is_done(self):
        picks = torch.tensor([[5, 6]]); drafts = torch.tensor([[5]])
        count, done, accepted, _ = commit_batch(picks, drafts, torch.tensor([True]), torch.tensor([3]), torch.tensor([5]), torch.tensor([[-1]]))
        self.assertEqual((count.item(), done.item(), accepted.item()), (2, True, 1))


if __name__ == "__main__":
    unittest.main()
