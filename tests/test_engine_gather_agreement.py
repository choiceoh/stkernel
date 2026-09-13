#!/usr/bin/env python3
"""The one collective in a decode step whose size is data dependent.

`_gather` sends the rich rows, and when a grammar is live it sends each row's live span -- so the
number of rows is computed per rank from per-rank matcher state. `all_gather_into_tensor` requires
the same shape on every rank and does not check it. It waits.

2026-09-12 it waited forever: ranks 0 and 1 offered seven rows and ranks 2 and 3 offered six
(NumelIn 271,040 against 232,320 at vp 38,720). All four sat in _ALLGATHER_BASE, the GPUs read 0%,
requests queued behind them, and the only evidence was an NCCL watchdog dump twelve minutes later.

The vote now rides the tripwire (base/tripwire): a fixed-shape exchange that also says which site
each rank is at, and the divergence report is ONE exchange of a fixed shape, never a loop whose
trip count is the very number that diverged.
"""
from __future__ import annotations

import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.base.tripwire import pack, unpack                          # noqa: E402

PAD = [-1, 0, 0]


class Comm:
    """Four ranks, every one running the same code with its own values: this rank's contribution
    arrives packed, the peers' are packed the way real ranks pack them, and the sum is what the
    control group would return. `offered` is each rank's row count; `details` each rank's
    (seq, span, digest) rows for the divergence report."""

    def __init__(self, rank, offered, details=None):
        self.rank, self.world_size, self.offered = rank, len(offered), list(offered)
        self.details = details                          # per rank: [(seq, span, digest), ...]
        self.gathered = self.asked = 0

    def all_reduce_host(self, values):
        tags, _ = unpack(values, self.world_size)
        seq, sid, count, exchange = tags[self.rank]
        self.asked += 1
        total = list(values)
        for r in range(self.world_size):
            if r == self.rank:
                continue
            theirs = [self.offered[r]] if self.asked == 1 else self.flat(self.details[r] if self.details else [])
            total = [a + b for a, b in zip(total, pack(self.world_size, r, seq, sid, theirs, bool(exchange)))]
        return total

    @staticmethod
    def flat(rows):
        from engine.profiles.glm53.adapter import Glm53Engine
        return [v for row in rows for v in row] + PAD * (Glm53Engine.DETAIL_ROWS - len(rows))

    def all_gather(self, t, dim=-1):
        self.gathered += 1
        return t


class Local:
    def __init__(self, rows):
        self.shape = (rows, 38720)


class GatherAgreementTests(unittest.TestCase):
    def gather(self, rank, offered, details=None):
        from engine.profiles.glm53.adapter import Glm53Engine
        comm = Comm(rank, offered, details)
        engine = types.SimpleNamespace(net=types.SimpleNamespace(comm=comm), DETAIL_ROWS=Glm53Engine.DETAIL_ROWS)
        engine._gather_divergence = types.MethodType(Glm53Engine._gather_divergence, engine)
        engine._draft_digest = Glm53Engine._draft_digest
        return types.MethodType(Glm53Engine._gather, engine), comm

    def test_agreement_lets_the_collective_through(self):
        call, comm = self.gather(0, [7, 7, 7, 7])
        call(Local(7))
        self.assertEqual(comm.gathered, 1)
        self.assertEqual(comm.asked, 1, "one exchange, no report")

    def test_disagreement_refuses_instead_of_hanging_and_names_every_rank(self):
        call, comm = self.gather(0, [7, 7, 6, 6])          # the shape of the 2026-09-12 hang
        with self.assertRaises(RuntimeError) as caught:
            call(Local(7))
        message = str(caught.exception)
        self.assertEqual(comm.gathered, 0, "the collective that can only hang is not entered")
        for fragment in ("rank0=7", "rank1=7", "rank2=6", "rank3=6"):
            self.assertIn(fragment, message, fragment)
        self.assertIn("grammar", message, "and it says where a divergence comes from")
        self.assertEqual(comm.asked, 2, "the report is one more exchange, of a fixed shape")

    def test_the_message_names_the_row_and_says_which_half_diverged(self):
        """Seven against six is where the first occurrence stopped. The two candidates behind that
        number -- the live span a matcher computed, or the drafts it computed it over, which are drawn
        per rank -- cannot be told apart afterwards, so both are voted and the message says which."""
        # seq 12's spans agree (3 each) but its drafts do not: an RNG divergence, not a grammar one
        rows = [[(12, 3, 0xAA)], [(12, 3, 0xAA)], [(12, 3, 0xBB)], [(12, 3, 0xBB)]]
        call, _ = self.gather(0, [7, 7, 6, 6], details=rows)
        with self.assertRaises(RuntimeError) as caught:
            call(Local(7), detail=[(12, 3, 0xAA)])
        message = str(caught.exception)
        self.assertIn("seq 12: drafts differ", message)
        self.assertIn("0xbb", message.lower())

        # and when the spans are what differ, it says that instead
        rows = [[(12, 3, 0xAA)], [(12, 3, 0xAA)], [(12, 2, 0xAA)], [(12, 2, 0xAA)]]
        call, _ = self.gather(0, [7, 7, 6, 6], details=rows)
        with self.assertRaises(RuntimeError) as caught:
            call(Local(7), detail=[(12, 3, 0xAA)])
        self.assertIn("seq 12: live spans differ", str(caught.exception))

    def test_ranks_that_disagree_which_rows_are_rich_are_named_without_a_second_divergence(self):
        """The report used to loop once per rich row with a collective inside: when the number of rich
        rows was what diverged, the report about the divergence diverged. One fixed-shape exchange."""
        rows = [[(12, 3, 0xAA), (15, 2, 0xCC)], [(12, 3, 0xAA), (15, 2, 0xCC)], [(12, 3, 0xAA)], [(12, 3, 0xAA)]]
        call, comm = self.gather(0, [7, 7, 6, 6], details=rows)
        with self.assertRaises(RuntimeError) as caught:
            call(Local(7), detail=[(12, 3, 0xAA), (15, 2, 0xCC)])
        message = str(caught.exception)
        self.assertIn("seq 15: gathered by rank0, rank1 only -- the ranks do not agree which rows are rich", message)
        self.assertNotIn("seq 12:", message, "the row every rank agrees on is not reported")
        self.assertEqual(comm.asked, 2)

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
        self.assertEqual(comm.asked, 0)


if __name__ == "__main__":
    unittest.main()
