"""Every rank keeps the parked entries every rank holds alike, and drops the rest (45차, 2026-09-13).

2026-09-13 13:01:47 the fleet split during a health-check chat: ranks 0 and 1 finished the turn and
parked it, ranks 2 and 3 had not finished it and parked nothing. The tiers then held conversations
{0, 1, 2} on two nodes and {0, 1} on the other two, and six production launches died on the skew
at a boot check that still killed instead of recovering. The recovery before that dropped
EVERYTHING on any skew. Now the ranks exchange (key, digest) lists and keep the intersection:
conversations 0 and 1 survive on all four, conversation 2 goes from ranks 0 and 1.

Four ranks run as four threads over a fake host group, so the claim under test is the real one:
every rank reaches the same decision from its own tier.
"""
import io
import sys
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from engine.base.serve import Server                                   # noqa: E402


class Board:
    """A host group for threads: every rank posts an object, every rank gets the list back in rank order."""

    def __init__(self, world, timeout=5.0):
        self.world, self.lock, self.posted = world, threading.Lock(), {}
        self.arrive = threading.Barrier(world, timeout=timeout)
        self.leave = threading.Barrier(world, timeout=timeout)

    def gather(self, rank, obj):
        with self.lock:
            self.posted[rank] = obj
        self.arrive.wait()
        out = [self.posted[r] for r in range(self.world)]
        self.leave.wait()
        return out


class Rank:
    def __init__(self, board, rank):
        self.board, self.rank, self.world_size = board, rank, board.world

    def gather_objects(self, obj):
        return self.board.gather(self.rank, obj)


class Tier:
    """A rank's tier as `_parked_entries` reads it: a record and a block count per key."""

    def __init__(self, entries, unreadable=()):
        self.entries, self.unreadable = dict(entries), set(unreadable)
        self.forgotten = []

    def keys(self):
        return sorted(self.entries)

    def record(self, key):
        if key in self.unreadable:
            raise OSError("the record is torn")
        return self.entries[key][0]

    def blocks(self, key):
        return self.entries[key][1]

    def forget(self, key):
        self.forgotten.append(key)
        self.entries.pop(key)


def ping(answer):
    """A health-check chat as the tier recorded it (13:00:44 .. 13:01:47)."""
    return {"context": 17, "pending": 1, "tokens": [154822, 154824, 58694, 271] + answer, "prompt_len": 14,
            "limits": [4, 1.0], "min_new": 0, "options": {}, "media": []}


def reconcile(tiers, what="conversations"):
    """Every rank reconciles its own tier; returns (dropped per rank, what each tier holds after, the output)."""
    board, dropped, out = Board(len(tiers)), [None] * len(tiers), io.StringIO()

    def main(r):
        entries = Server._parked_entries(tiers[r], tiers[r].keys(), r)
        dropped[r] = Server._reconcile_parked(Rank(board, r), entries, forget=tiers[r].forget, what=what)
    with redirect_stdout(out):
        threads = [threading.Thread(target=main, args=(r,), daemon=True) for r in range(len(tiers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
    return dropped, [t.keys() for t in tiers], out.getvalue()


class ReconcileTests(unittest.TestCase):
    def test_the_13_01_47_skew_keeps_what_all_four_hold_and_drops_the_turn_half_of_them_parked(self):
        both = {0: (ping([40, 2776]), 1), 1: (ping([40, 2776]), 1)}
        tiers = [Tier({**both, 2: (ping([3838, 646]), 1)}), Tier({**both, 2: (ping([3838, 646]), 1)}),
                 Tier(both), Tier(both)]
        dropped, held, out = reconcile(tiers)
        self.assertEqual(dropped, [1, 1, 0, 0])
        self.assertEqual(held, [[0, 1]] * 4, "every rank now numbers conversations alike")
        self.assertEqual((tiers[0].forgotten, tiers[2].forgotten), ([2], []))
        self.assertIn("rank 0 dropped 1 conversations the other ranks do not hold alike: 2 (only rank 0,1)", out)
        self.assertIn("kept the 2 every rank holds alike", out)

    def test_ranks_that_agree_keep_everything_and_say_nothing(self):
        tiers = [Tier({0: (ping([1, 2]), 1), 5: (ping([3, 4]), 2)}) for _ in range(4)]
        dropped, held, out = reconcile(tiers)
        self.assertEqual((dropped, held, out), ([0] * 4, [[0, 5]] * 4, ""))

    def test_the_same_key_with_a_different_record_goes_everywhere(self):
        """A node's leftover under a key the fleet also uses (45th 21's shape): resuming it would mix two states."""
        tiers = [Tier({0: (ping([1, 2]), 1)}), Tier({0: (ping([1, 2]), 1)}),
                 Tier({0: (ping([1, 2]), 1)}), Tier({0: (ping([9, 9]), 1)})]
        dropped, held, out = reconcile(tiers)
        self.assertEqual((dropped, held), ([1] * 4, [[]] * 4))
        self.assertIn("0 (records differ)", out)

    def test_the_same_record_needing_different_blocks_is_not_the_same_entry(self):
        tiers = [Tier({3: (ping([1, 2]), 1)}), Tier({3: (ping([1, 2]), 2)})]
        dropped, held, _ = reconcile(tiers)
        self.assertEqual((dropped, held), ([1, 1], [[], []]))

    def test_a_record_one_rank_cannot_read_goes_everywhere_even_if_every_rank_fails_alike(self):
        tiers = [Tier({4: (ping([1, 2]), 1)}), Tier({4: (ping([1, 2]), 1)}, unreadable={4})]
        dropped, held, _ = reconcile(tiers)
        self.assertEqual((dropped, held), ([1, 1], [[], []]))
        both = [Tier({4: (ping([1, 2]), 1)}, unreadable={4}), Tier({4: (ping([1, 2]), 1)}, unreadable={4})]
        dropped, held, _ = reconcile(both)
        self.assertEqual(held, [[], []], "an unreadable digest names its rank, so two failures never match")

    def test_a_drop_that_fails_is_reported_and_the_boot_goes_on(self):
        tiers = [Tier({7: (ping([1, 2]), 1)}), Tier({})]

        def bad(key):
            raise OSError("disk copy gone")
        tiers[0].forget = bad
        dropped, _, out = reconcile(tiers)
        self.assertEqual(dropped, [1, 0])
        self.assertIn("rank 0 could not drop conversations 7: disk copy gone", out)

    def test_one_rank_has_nobody_to_disagree_with_and_asks_nobody(self):
        class Alone:
            world_size, rank = 1, 0

            def gather_objects(self, obj):
                raise AssertionError("a world of one exchanges nothing")
        self.assertEqual(Server._reconcile_parked(Alone(), [(0, "x")], forget=lambda k: self.fail("kept")), 0)

    def test_a_short_answer_from_the_group_is_refused(self):
        class Short:
            world_size, rank = 4, 0

            def gather_objects(self, obj):
                return [obj, obj]
        with self.assertRaisesRegex(RuntimeError, "2 ranks answered for a world of 4"):
            Server._reconcile_parked(Short(), [(0, "x")], forget=lambda k: None)

    def test_the_digest_ignores_what_differs_by_rank_and_names_what_must_not(self):
        a = Server._parked_entries(Tier({0: (ping([1, 2]), 1)}), [0], 0)
        b = Server._parked_entries(Tier({0: (ping([1, 2]), 1)}), [0], 3)
        self.assertEqual(a, b, "the rank is not part of a readable entry's digest")
        c = Server._parked_entries(Tier({0: (ping([1, 3]), 1)}), [0], 0)
        self.assertNotEqual(a, c)
        self.assertEqual(Server._parked_entries(None, [], 0), [])


class ServerBootTests(unittest.TestCase):
    """The real constructor, a real Runner and TieredKV over each rank's own memory tier, four ranks at once."""

    def test_four_servers_booting_from_skewed_tiers_number_conversations_alike(self):
        import test_engine_serve as T
        from test_engine_tier import MemoryTier
        storage = bytearray(range(64))

        def park(tier, seq, answer):
            tier.demote(seq, storage, [0], 17, record=ping(answer))
        tiers = []
        for r in range(4):
            tier = MemoryTier()
            park(tier, 0, [40, 2776])
            park(tier, 1, [40, 2776])
            if r < 2:
                park(tier, 2, [3838, 646])                 # the turn only ranks 0 and 1 finished (13:01:47)
            if r == 3:
                park(tier, 5, [1, 1])                      # and a leftover only rank 3 carries (45th 21)
            tiers.append(tier)
        board, out = Board(4), [None] * 4

        def main(r):
            s = T.server(comm=Rank(board, r), rows=2, keep_idle=True, tier=tiers[r])
            out[r] = (sorted(s.runner.parked_keys()), s.next_seq)
        with redirect_stdout(io.StringIO()) as said:
            threads = [threading.Thread(target=main, args=(r,), daemon=True) for r in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(20)
        self.assertEqual(out, [([0, 1], 2)] * 4, "the same conversations and the same next id on every rank")
        self.assertEqual([t.keys() for t in tiers], [[0, 1]] * 4, "and the disk copies that no rank can resume are gone")
        self.assertIn("rank 3 dropped 1 conversations the other ranks do not hold alike: 5 (only rank 3)", said.getvalue())


class ContractTests(unittest.TestCase):
    def test_both_boot_checks_reconcile_and_neither_kills(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        ctor = serve[serve.index("parked = sorted(runner.parked_keys())"):serve.index("self.next_seq, self.served")]
        self.assertIn('self._reconcile_parked(comm, self._parked_entries(getattr(runner, "tiered", None), parked, comm.rank),', ctor)
        self.assertIn("forget=runner.forget_parked", ctor)
        self.assertIn('forget=self._forget_prefix_boundary(runner), what="prefix boundaries"', ctor)
        self.assertNotIn("_agree_on_parked", serve)
        self.assertNotIn("Clear glm53-logs/st-tier on every node", serve, "no boot dies on a skew it can reconcile")
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn("Server._reconcile_parked(comm, Server._parked_entries(runner.tiered, sorted(runner.parked_keys()), comm.rank),", boot)
        self.assertNotIn("_agree_on_parked", boot)
        self.assertLess(boot.index("Server._reconcile_parked("), boot.index('with rec.phase("capture decode"):'),
                        "still before the capture: a skew costs seconds, not a capture")


if __name__ == "__main__":
    unittest.main()
