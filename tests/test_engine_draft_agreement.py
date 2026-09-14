"""Replicated draft differences must not move TP contexts apart."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from engine.base.comm import LocalTP
from engine.base.sampler import block_verify_batch, commit_batch
from engine.modules.draft_agreement import agree_verdict, agree_walk


class DraftAgreementTests(unittest.TestCase):
    def test_isolated_probe_comms_keep_their_local_proposal(self):
        from probes.engine_decode_graph_check import IsolatedRank as Replay
        from probes.engine_graph_profile import IsolatedRank as Profile
        from probes.engine_prefill_chunk_profile import IsolatedRank as Prefill
        for comm in (Replay(), Profile(), Prefill(3)):
            tokens = torch.tensor([[7, 11, 13]])
            self.assertIs(agree_walk(comm, tokens), tokens)
            support = tokens.unsqueeze(-1)
            probabilities = torch.ones_like(support, dtype=torch.float32)
            for got, expected in zip(agree_walk(comm, tokens, support, probabilities),
                                     (tokens, support, probabilities)):
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

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


    def test_each_rank_commits_rank_zeros_verdict_whatever_it_reached_alone(self):
        def rank(comm):
            accepted = torch.tensor([comm.rank, 2])
            tokens = torch.tensor([[40 + comm.rank, 41, 42], [50, 51, 52 + comm.rank]])
            return agree_verdict(comm, accepted, tokens)
        for got_accepted, got_tokens in LocalTP(4, timeout_s=20).run(rank):
            self.assertEqual(got_accepted.tolist(), [0, 2])
            self.assertEqual(got_tokens.tolist(), [[40, 41, 42], [50, 51, 52]])
            self.assertTrue(got_accepted.is_contiguous() and got_tokens.is_contiguous(),
                            'the commit kernel reads accepted at stride 1 and tokens at stride (t, 1)')
        from probes.engine_decode_graph_check import IsolatedRank
        accepted, tokens = torch.tensor([1]), torch.tensor([[3, 4]])
        self.assertIs(agree_verdict(SimpleNamespace(world_size=1), accepted, tokens)[1], tokens)
        for got, expected in zip(agree_verdict(IsolatedRank(), accepted, tokens), (accepted, tokens)):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            agree_verdict(IsolatedRank(), accepted.int(), tokens)

    def test_sampled_decode_chain_commits_one_verdict_when_ranks_verify_to_different_bits(self):
        """The 2026-09-14 failure on CPU: the ranks gather the same target rows and hold the same walk, but their
        verification is their own arithmetic. Rotating the rows a rank's verification sees stands in for bits that
        differ from rank to rank; before agreement the ranks committed four different tokens."""
        from engine.profiles.glm53 import pipeline
        from engine.profiles.glm53.pipeline import AsyncDecode
        from tests.test_engine_pipeline import BatchTransitionTests
        vocab, world = 32, 4
        full = torch.arange(vocab, dtype=torch.float32).mul(20).repeat(2, 1)   # a row a position, its mass on id 31
        here = threading.local()
        verified = [None] * world
        def verify(probs, *args):
            accepted, tokens, count = block_verify_batch(torch.roll(probs, here.rank, -1), *args)
            verified[here.rank] = (accepted.tolist(), tokens.tolist())
            return accepted, tokens, count
        def rank(comm):
            here.rank = comm.rank
            e = BatchTransitionTests().engine()
            e.limits[1] = (10, 1.0)
            e.decodable, e.net, e.note_ceilings = vocab, SimpleNamespace(comm=comm), lambda *args: None
            e.drafter.propose_rows = lambda *args, **kwargs: (
                torch.tensor([[9]]), torch.tensor([[[8, 9, 10, 11]]]), torch.full((1, 1, 4), 0.25))
            shard = full.chunk(world, -1)[comm.rank].contiguous()
            e.decode_graphs.run_inputs = lambda *args: (None, None, shard)
            p = AsyncDecode(e)
            p._build([1], [1])
            result = p.iterate((1, 2, 64), p.buf)
            return {k: result[k].tolist() for k in ('tokens', 'count', 'accepted')}
        with mock.patch.object(pipeline, 'block_verify_batch', verify):
            rows = LocalTP(world, timeout_s=20).run(rank)
        self.assertEqual(len({str(v) for v in verified}), world, 'each rank reached its own verdict')
        accepted, tokens = verified[0]
        for row in rows:
            self.assertEqual(row, rows[0])
        self.assertEqual(rows[0]['accepted'], accepted)
        self.assertEqual(rows[0]['tokens'][0][:rows[0]['count'][0]], tokens[0][:accepted[0] + 1])

    def test_rich_rows_commit_rank_zeros_verdict(self):
        """The host rich path verifies with lists; its verdicts meet in one packet sized by the agreed live spans."""
        from engine.profiles.glm53.adapter import Glm53Engine
        verdicts = {0: [(1, [7, 8]), (0, [5])], 1: [(2, [7, 9, 4]), (0, [6])],
                    2: [(0, [3]), (0, [5])], 3: [(1, [7, 8]), (0, [5])]}
        def rank(comm):
            e = SimpleNamespace(net=SimpleNamespace(comm=comm))
            return Glm53Engine._agree_verdicts(e, verdicts[comm.rank], [3, 1], torch.device('cpu'))
        for got in LocalTP(4, timeout_s=20).run(rank):
            self.assertEqual(got, verdicts[0])
        alone = SimpleNamespace(net=SimpleNamespace(comm=SimpleNamespace(world_size=1)))
        self.assertIs(Glm53Engine._agree_verdicts(alone, verdicts[1], [3, 1], torch.device('cpu')), verdicts[1])


if __name__ == '__main__':
    unittest.main()
