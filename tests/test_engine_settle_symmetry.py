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
from engine.base.tripwire import pack, site_id, unpack                  # noqa: E402


class TwoRankComm(T.Comm):
    """This rank and one imagined peer that stands where this rank stands and agrees with every value it
    is shown -- except at the sites in `answers`, where the peer's values are the ones given (by site id).
    The vectors are the tripwire's: the peer's contribution is packed the way a real rank packs it and
    summed, which is what the control group does."""
    world_size = 2

    def __init__(self, answers=None):
        self.votes = []                                   # (site id, this rank's values) as cast
        self.answers = dict(answers or {})

    def all_reduce_host(self, values):
        tags, region = unpack(values, self.world_size)
        seq, sid, count, exchange = tags[0]               # this rank is rank 0
        mine = region[:count]
        self.votes.append((sid, mine))
        theirs = self.answers.get(sid, mine)
        return [a + b for a, b in zip(values, pack(self.world_size, 1, seq, sid, theirs, bool(exchange)))]

    def all_reduce_max(self, t):
        return t

    def gather_objects(self, obj):
        return [obj, obj]                                 # the peer's tier holds what this rank's does

    def cast(self, site):
        """This rank's values at `site`, in the order they were cast."""
        return [values for sid, values in self.votes if sid == site_id(site)]


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
        self.assertIn([1], comm.cast("settle:done"), "and it still voted 'done' for the row")
        self.assertIn([0, 0], comm.cast("settle:outcome"), "and 'not ok, not full' on the outcome")
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

    def test_a_request_that_fits_here_waits_until_it_fits_everywhere(self):
        """`kv.available` is per rank (a prefix spill pins blocks on the rank whose tier thread got there):
        one rank admitting a request the others do not is the seven-rows-against-six step of 2026-09-12."""
        comm = TwoRankComm({site_id("admit:fits"): [0]})           # the peer says it does not fit
        s = T.server(rows=1, comm=comm)
        request, _ = s.submit([3], 1, 0)
        for _ in range(5):
            s.once()
        self.assertEqual(len(s._waiting), 1, "not admitted here either")
        self.assertEqual(sorted(s._free_rows), [0])
        self.assertEqual(comm.cast("admit:fits")[-1], [1], "this rank's own answer was yes")
        comm.answers.clear()                                        # the peer's spill ended
        settle(s)
        self.assertEqual(list(s.take_result(request)), [3])

    def test_a_prompt_one_rank_sees_in_flight_waits_on_every_rank(self):
        """The prefix cache is per rank; whether the same prompt is being prefilled beside this request,
        and how much of it is already cached, are agreed before the branch: one rank's 'in flight' makes
        every rank wait, and the boundary every rank has is the lowest."""
        comm = TwoRankComm({site_id("admit:prefix"): [0, 1]})      # the peer: nothing cached, the prompt is in flight
        s = T.server(rows=2, comm=comm, prefix=4)
        request, _ = s.submit([1, 2, 3, 4, 5, 6, 7, 8], 1, 0)
        for _ in range(5):
            s.once()
        self.assertEqual(len(s._waiting), 1, "deferred here too, though this rank sees nothing in flight")
        self.assertIn(request, s._deferred)
        self.assertEqual(comm.cast("admit:prefix")[-1][1], 0, "this rank's own answer was 'not in flight'")
        comm.answers.clear()
        settle(s)
        self.assertEqual(len(list(s.take_result(request))), 1)
        self.assertNotIn(request, s._deferred)

    def test_the_source_keeps_its_word(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        body = serve[serve.index("    def _settle(self):"):serve.index("    def _yield_asked(self)")]
        self.assertIn("rows = sorted(set(self._retiring) | set(self._resuming) | set(self._restoring))", body)
        self.assertNotIn("rows = self.runner.transfers()", body)
        self.assertNotIn("prefill it instead", serve, "no rank prefills on its own after a vote admitted a restore")
        self.assertIn("self._failed_begins.add(row)", serve)
        admit = serve[serve.index("    def _admit(self):"):serve.index("    def _transfer_done(self, row)")]
        self.assertIn('rows = self.tripwire.exchange("admit:prefix", [above, int(ahead is not None)])', admit)
        self.assertIn("above = min(row[0] for row in rows)", admit)
        self.assertIn("if any(row[1] for row in rows):", admit)
        self.assertNotIn("if ahead is not None:", admit, "this rank's prefix cache no longer decides the branch alone")
        self.assertIn('if self._votes([fits], "admit:fits")[0] < world:', admit)
        self.assertNotIn("if promised > self.runner.kv.available - future + resident:", admit)
        reorder = serve[serve.index("    def _reorder_waiting(self)"):serve.index("    def _admit(self):")]
        self.assertIn('first = min(row[0] for row in self.tripwire.exchange("admit:reorder", [first]))', reorder)


if __name__ == "__main__":
    unittest.main()
