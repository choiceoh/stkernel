"""Every rank settles every transfer, whether or not its own half of it ever began (45차, 2026-09-13).

Ranks 0 and 1 stood in _settle's host vote while ranks 2 and 3 -- whose prefix restore had failed
to begin and who had fallen back to prefill on their own -- ran the next step's one-shot all-reduce
and spun in it until its stall trap: Xid 43, "unspecified launch failure", twice in one night
(04:57, 05:24). A begin that fails on one rank is now a transfer that rank still votes on, and the
vote turns every rank the same way: dropped, prefilled, refused.
"""
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_engine_serve as T                                          # noqa: E402
from test_engine_tier import MemoryTier                                # noqa: E402
from engine.base.serve import RequestError                             # noqa: E402


class TwoRankComm(T.Comm):
    """This rank and one imagined peer that agrees with every flag it is shown (doubling the sums)."""
    world_size = 2

    def __init__(self):
        self.votes = []

    def all_reduce_host(self, values):
        self.votes.append([int(v) for v in values])
        return [int(v) * 2 for v in values]

    def all_reduce_max(self, t):
        return t


def settle(s, steps=200):
    """Run until nothing is in flight and nothing waits."""
    for _ in range(steps):
        ran = s.once()
        if not ran and not s._waiting and not s._retiring and not s._resuming and not s._restoring:
            return
        threading.Event().wait(0.001)
    raise AssertionError("the server did not settle")


class SettleSymmetryTests(unittest.TestCase):
    def test_a_park_that_cannot_begin_here_is_still_voted_and_dropped_everywhere(self):
        s = T.server(rows=2, keep_idle=True, tier=MemoryTier())
        real = s.runner.park_begin
        calls = []

        def refuse(seq, key=None):
            calls.append(seq)
            raise ValueError("no room on this rank's tier")
        s.runner.park_begin = refuse
        first, _ = s.submit([3], 1, 0)
        settle(s)                                                      # the turn finishes, _retire runs, the vote drops the row
        self.assertEqual(calls, [0])
        self.assertFalse(s._retiring or s._failed_begins)
        self.assertFalse(s.runner.is_parked(first), "not parked here, so forgotten everywhere")
        self.assertEqual(sorted(s._free_rows), [0, 1])
        self.assertFalse(s.runner.slot_of)
        s.runner.park_begin = real

    def test_the_vote_is_cast_even_when_this_rank_has_no_transfer_in_flight(self):
        """The old _settle returned early on `runner.transfers()` -- this rank's tier thread's view --
        and never voted; the ranks whose park had begun then waited for it in a vote it never joined."""
        comm = TwoRankComm()
        s = T.server(rows=2, keep_idle=True, tier=MemoryTier(), comm=comm)
        s.runner.park_begin = lambda seq, key=None: (_ for _ in ()).throw(ValueError("no tier here"))
        s.submit([3], 1, 0)
        settle(s)
        self.assertFalse(s.runner.transfers(), "nothing ever began on this rank")
        self.assertIn([1], comm.votes, "and it still voted 'done' for the row")
        self.assertIn([0, 0], comm.votes, "and 'not ok, not full' on the outcome")
        self.assertFalse(s._retiring or s._failed_begins)

    def test_a_restore_that_cannot_begin_here_turns_the_prompt_into_a_prefill_by_the_vote(self):
        """The crash's rank 2 and 3: the boundary was on the tier by every rank's word, the read
        could not start here. Before: a silent local prefill and a fleet stuck in two collectives."""
        s = T.server(rows=2, prefix=3, prefix_tier=True)
        s.runner.spill_low_water = 10                                  # spill leaves as soon as they exist
        prompt = [11, 12, 13, 14, 15, 16, 17, 18]
        first, _ = s.submit(prompt, 1, 0)
        settle(s)
        s.take_result(first)
        for _ in range(4):
            s.once()                                                   # the leaf (8) is written out to the prefix tier
        self.assertEqual(len(s.runner.prefix.tier_keys), 1)
        other, _ = s.submit([31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], 1, 0)   # evicts the memory copy: only the tier holds 8
        settle(s)
        s.take_result(other)
        real = s.runner.restore_begin
        refused = []

        def refuse(seq, h, tokens):
            refused.append((seq, tokens))
            raise OSError("this rank's copy is unreadable")
        s.runner.restore_begin = refuse
        again, _ = s.submit(prompt + [21, 22], 1, 0)
        settle(s)
        out = s.take_result(again)                                     # answered, by a plain prefill
        self.assertEqual(refused, [(0, 8)], "the restore was attempted once, for the 8-token boundary")
        self.assertFalse(s._restoring or s._failed_begins or s._waiting)
        self.assertEqual(s.runner.prefix_restores, 0)
        self.assertIsNotNone(out)
        s.runner.restore_begin = real

    def test_a_resume_that_cannot_begin_here_is_refused_everywhere_instead_of_killing_this_rank(self):
        s = T.server(rows=2, keep_idle=True, tier=MemoryTier())
        first, _ = s.submit([3], 1, 0)
        settle(s)
        s.take_result(first)
        self.assertTrue(s.runner.is_parked(first))
        s.runner.resume_begin = lambda seq, key=None: (_ for _ in ()).throw(OSError("the tier's read cannot start"))
        turn, _ = s.submit([9], 1, 0, conversation=first)
        settle(s)                                                      # no exception escapes once()
        with self.assertRaises(RequestError) as caught:
            s.take_result(turn)
        self.assertEqual(caught.exception.status, 503)
        self.assertFalse(s.runner.is_parked(first), "gone everywhere: the vote said a rank could not read it back")
        self.assertFalse(s._resuming or s._failed_begins)
        self.assertEqual(sorted(s._free_rows), [0, 1])

    def test_the_source_keeps_its_word(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        body = serve[serve.index("    def _settle(self):"):serve.index("    def _yield_asked(self)")]
        self.assertIn("rows = sorted(set(self._retiring) | set(self._resuming) | set(self._restoring))", body)
        self.assertNotIn("rows = self.runner.transfers()", body)
        self.assertNotIn("prefill it instead", serve, "no rank prefills on its own after a vote admitted a restore")
        self.assertIn("self._failed_begins.add(row)", serve)


if __name__ == "__main__":
    unittest.main()
