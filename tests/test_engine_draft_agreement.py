"""Replicated draft differences must not move TP contexts apart."""
import unittest
import torch

from engine.base.comm import LocalTP
from engine.base.sampler import block_verify_batch, commit_batch
from engine.modules.draft_agreement import agree_walk


class DraftAgreementTests(unittest.TestCase):
    def test_different_local_drafts_cannot_split_committed_context(self):
        picks = torch.tensor([[1, 2, 3, 4]])
        def commit(drafts):
            return commit_batch(picks, drafts, torch.tensor([True]), torch.tensor([500]),
                                torch.tensor([16384]), torch.tensor([[-1]]))
        def rank(comm):
            # Same target picks, different local walk: the live failure was
            # exactly a three-token difference at one committed dispatch.
            drafts = torch.tensor([[9, 2, 3] if comm.rank != 1 else [1, 2, 3]])
            before = int(commit(drafts)[0])
            return before, commit(agree_walk(comm, drafts))
        rows = LocalTP(4, timeout_s=20).run(rank)
        self.assertEqual([row[0] for row in rows], [1, 4, 1, 1])
        for _, result in rows:
            self.assertEqual(int(result[0]), 1)
            for got, expected in zip(result, rows[0][1]):
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_sampled_walk_keeps_the_distribution_that_generated_its_tokens(self):
        target = torch.softmax(torch.arange(4 * 17).reshape(1, 4, 17).float() / 17, -1)
        def rank(comm):
            generator = torch.Generator().manual_seed(50 + comm.rank)
            support = torch.stack([torch.randperm(17, generator=generator)[:8] for _ in range(3)]).unsqueeze(0)
            probabilities = torch.softmax(torch.randn(1, 3, 8, generator=generator), -1)
            choice = torch.multinomial(probabilities.reshape(3, 8), 1, generator=generator).view(1, 3, 1)
            drafts = support.gather(2, choice).squeeze(2)
            original = (drafts.clone(), support.clone(), probabilities.clone())
            shared = agree_walk(comm, drafts, support, probabilities)
            outcome = block_verify_batch(target, *shared, torch.tensor([[.13, .71, .49, .27]]))
            return original, shared, outcome
        rows = LocalTP(4, timeout_s=20).run(rank)
        self.assertFalse(torch.equal(rows[0][0][0], rows[1][0][0]))
        for original, shared, outcome in rows:
            for got, expected in zip(shared, rows[0][0]):
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
            for got, expected in zip(outcome, rows[0][2]):
                torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
