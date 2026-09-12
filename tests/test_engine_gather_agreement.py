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

    def __init__(self, rank, offered):
        self.rank, self.world_size, self.offered = rank, len(offered), list(offered)
        self.gathered = 0

    def all_reduce_host(self, values):
        self.assert_own_slot(values)
        peers = [[n if i == r else 0 for i, n in enumerate(self.offered)] for r in range(self.world_size)]
        return [sum(column) for column in zip(*peers)]

    def assert_own_slot(self, values):
        assert len(values) == self.world_size and values[self.rank] == self.offered[self.rank], values
        assert all(v == 0 for i, v in enumerate(values) if i != self.rank), values

    def all_gather(self, t, dim=-1):
        self.gathered += 1
        return t


class Local:
    def __init__(self, rows):
        self.shape = (rows, 38720)


class GatherAgreementTests(unittest.TestCase):
    def gather(self, rank, offered):
        from engine.profiles.glm53.adapter import Glm53Engine
        comm = Comm(rank, offered)
        engine = types.SimpleNamespace(net=types.SimpleNamespace(comm=comm))
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

    def test_one_rank_pays_nothing_for_the_vote(self):
        call, comm = self.gather(0, [5])
        call(Local(5))
        self.assertEqual(comm.gathered, 1)


if __name__ == "__main__":
    unittest.main()
