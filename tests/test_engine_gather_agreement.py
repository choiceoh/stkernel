#!/usr/bin/env python3
"""The one collective in a decode step whose size is data dependent.

`_gather` sends the rich rows, and when a grammar is live it sends each row's live span -- so the
number of rows is computed per rank from per-rank matcher state. `all_gather_into_tensor` requires
the same shape on every rank and does not check it. It waits.

2026-09-12 it waited forever: ranks 0 and 1 offered seven rows and ranks 2 and 3 offered six
(NumelIn 271,040 against 232,320 at vp 38,720). All four sat in _ALLGATHER_BASE, the GPUs read 0%,
requests queued behind them, and the only evidence was an NCCL watchdog dump twelve minutes later.
"""
from __future__ import annotations

import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Comm:
    """Four ranks. `all_reduce_host` sums the per-rank vectors, so a vote with one slot filled per
    rank comes back as every rank's value -- which is what the real one does on the control group."""

    def __init__(self, rank, offered, votes=()):
        self.rank, self.world_size, self.offered = rank, len(offered), list(offered)
        self.votes = [list(v) for v in votes]      # the answers to the follow-up votes, in order
        self.gathered = self.asked = 0

    def all_reduce_host(self, values):
        """Sum the per-rank vectors. The caller fills only its own slot, so what comes back is the
        vector of every rank's value -- `votes` and `peers` below are the other ranks' answers."""
        assert len(values) == self.world_size, values
        assert all(v == 0 for i, v in enumerate(values) if i != self.rank), values
        mine = values[self.rank]
        row = self.offered if self.asked == 0 else (self.votes.pop(0) if self.votes else [mine] * self.world_size)
        self.asked += 1
        return [mine if i == self.rank else row[i] for i in range(self.world_size)]

    def all_gather(self, t, dim=-1):
        self.gathered += 1
        return t


class Local:
    def __init__(self, rows):
        self.shape = (rows, 38720)


class GatherAgreementTests(unittest.TestCase):
    def gather(self, rank, offered, votes=()):
        from engine.profiles.glm53.adapter import Glm53Engine
        comm = Comm(rank, offered, votes)
        engine = types.SimpleNamespace(net=types.SimpleNamespace(comm=comm))
        engine._gather_divergence = types.MethodType(Glm53Engine._gather_divergence, engine)
        engine._draft_digest = Glm53Engine._draft_digest
        return types.MethodType(Glm53Engine._gather, engine), comm

    def test_agreement_lets_the_collective_through(self):
        call, comm = self.gather(0, [7, 7, 7, 7])
        call(Local(7))
        self.assertEqual(comm.gathered, 1)

    def test_disagreement_refuses_instead_of_hanging_and_names_every_rank(self):
        call, comm = self.gather(0, [7, 7, 6, 6])          # the shape of the 2026-09-12 hang
        with self.assertRaises(RuntimeError) as caught:
            call(Local(7))
        message = str(caught.exception)
        self.assertEqual(comm.gathered, 0, "the collective that can only hang is not entered")
        for fragment in ("rank0=7", "rank1=7", "rank2=6", "rank3=6"):
            self.assertIn(fragment, message, fragment)
        self.assertIn("grammar", message, "and it says where a divergence comes from")

    def test_the_message_names_the_row_and_says_which_half_diverged(self):
        """Seven against six is where the first occurrence stopped. The two candidates behind that
        number -- the live span a matcher computed, or the drafts it computed it over, which are drawn
        per rank -- cannot be told apart afterwards, so both are voted and the message says which."""
        from engine.profiles.glm53.adapter import Glm53Engine
        # seq 12's spans agree (3 each) but its drafts do not: an RNG divergence, not a grammar one
        call, _ = self.gather(0, [7, 7, 6, 6], votes=[[3, 3, 3, 3], [0xAA, 0xAA, 0xBB, 0xBB]])
        with self.assertRaises(RuntimeError) as caught:
            call(Local(7), detail=[(12, 3, 0xAA)])
        message = str(caught.exception)
        self.assertIn("seq 12: drafts differ", message)
        self.assertIn("0xbb", message.lower())

        # and when the spans are what differ, it says that instead
        call, _ = self.gather(0, [7, 7, 6, 6], votes=[[3, 3, 2, 2], [0xAA, 0xAA, 0xAA, 0xAA]])
        with self.assertRaises(RuntimeError) as caught:
            call(Local(7), detail=[(12, 3, 0xAA)])
        self.assertIn("seq 12: live spans differ", str(caught.exception))

    def test_a_digest_is_order_sensitive_and_small_enough_to_sum(self):
        from engine.profiles.glm53.adapter import Glm53Engine
        digest = Glm53Engine._draft_digest
        self.assertNotEqual(digest([1, 2, 3]), digest([3, 2, 1]))
        self.assertNotEqual(digest([1, 2]), digest([1, 2, 0]), "a trailing zero is a token, not nothing")
        self.assertEqual(digest(None), 0)
        self.assertLess(4 * digest([9] * 32), 2 ** 63, "four ranks of it still fit the vote")

    def test_one_rank_pays_nothing_for_the_vote(self):
        call, comm = self.gather(0, [5])
        call(Local(5))
        self.assertEqual(comm.gathered, 1)


if __name__ == "__main__":
    unittest.main()
