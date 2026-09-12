"""A boot whose ranks disagree on the parked conversations drops them and boots (45차, 2026-09-13).

A rank that crashed parked nothing; the survivors parked their rows on the way down; the next boot
died on the skew, and the one after, and production stayed down for want of one parked chat probe.
The vote is the same on every rank, so the decision is too: every rank drops what it holds.
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:                                   # pragma: no cover -- the vote is a tensor
    torch = None

ROOT = Path(__file__).resolve().parents[1]


class FakeComm:
    """world_size ranks; `highest` is what the fleet's vote returns, `others_disagree` forces the flag."""

    def __init__(self, world_size=4, highest=None, others_disagree=False):
        self.world_size, self.highest, self.others_disagree, self.rank = world_size, highest, others_disagree, 0

    def all_reduce_max(self, t):
        if t.numel() == 3:                                         # the vote: (count, last key, checksum)
            return t.clone() if self.highest is None else torch.tensor(self.highest, dtype=t.dtype, device=t.device)
        return torch.ones_like(t) if self.others_disagree else t   # the disagree flag


@unittest.skipUnless(torch is not None, "the vote is a torch tensor")
class TierAgreementTests(unittest.TestCase):
    def setUp(self):
        from engine.base.serve import Server
        self.agree = Server._agree_on_parked
        self.dropped = []

    def forget(self, key):
        self.dropped.append(key)

    def test_ranks_that_agree_keep_what_they_hold(self):
        self.assertEqual(self.agree(FakeComm(), [0, 1, 2], forget=self.forget), 0)
        self.assertEqual(self.dropped, [])

    def test_ranks_that_disagree_drop_everything_and_the_boot_goes_on(self):
        out = io.StringIO()
        with redirect_stdout(out):
            n = self.agree(FakeComm(highest=[3, 2, 12345]), [0, 1], forget=self.forget)
        self.assertEqual((n, self.dropped), (2, [0, 1]))
        self.assertIn("tier skew", out.getvalue())
        self.assertIn("Dropped all 2 conversations", out.getvalue())

    def test_a_rank_that_matches_the_highest_still_drops_when_the_fleet_disagrees(self):
        """The decision is the vote's, not this rank's: rank 0 held the highest count at 06:06 and would
        have kept its parked row while ranks 2 and 3 held nothing."""
        with redirect_stdout(io.StringIO()):
            n = self.agree(FakeComm(others_disagree=True), [5], forget=self.forget)
        self.assertEqual((n, self.dropped), (1, [5]))

    def test_a_drop_that_fails_does_not_stop_the_boot(self):
        def bad(key):
            raise OSError("disk copy gone")
        with redirect_stdout(io.StringIO()) as out:
            n = self.agree(FakeComm(highest=[3, 2, 1]), [7], forget=bad)
        self.assertEqual(n, 1)
        self.assertIn("could not drop conversations 7", out.getvalue())

    def test_without_a_dropper_the_old_contract_holds(self):
        with self.assertRaisesRegex(RuntimeError, "Clear glm53-logs/st-tier on every node"):
            self.agree(FakeComm(highest=[3, 2, 1]), [0])

    def test_one_rank_has_nobody_to_disagree_with(self):
        self.assertEqual(self.agree(FakeComm(world_size=1, highest=[9, 9, 9]), [0, 1], forget=self.forget), 0)
        self.assertEqual(self.dropped, [])

    def test_the_boot_drops_conversations_and_prefix_boundaries_alike(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        boot = serve[serve.index("parked = sorted(runner.parked_keys())"):serve.index("self.next_seq, self.served")]
        self.assertIn("forget=runner.forget_parked", boot)
        self.assertIn("parked = sorted(runner.parked_keys())                        # dropped on every rank", boot)
        self.assertIn('forget=self._forget_prefix_boundary(runner), what="prefix boundaries"', boot)


if __name__ == "__main__":
    unittest.main()
