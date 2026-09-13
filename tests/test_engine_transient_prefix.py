"""A turn that will not be retained caches its boundaries for its own turn and takes none of them away (45차, 2026-09-13).

A D17 probe measures on the live production door after every deploy, and it used to POST
/v1/prefix/reset first -- every boundary in memory and every slot of the prefix tier, gone. It
needs nothing reset: its requests carry unique cache salts, so they cannot hit what production
cached, and they say `retain: false`. The rows those turns take are TRANSIENT here: their prefill
does the same work as any prompt's (the boundaries are computed and cached, so what is measured is
what production runs), but a transient boundary is never written to the prefix tier, it is the
first to give up its snapshot, and it leaves with its row -- unless another prompt adopted it or an
operator pinned it, which makes it an ordinary boundary. The marks ride the replicated options, so
every rank drops the same boundaries; a four-rank run pins that too.
"""
import importlib.util
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_engine_serve as T                                          # noqa: E402
from test_engine_prefix import BLOCK, CHUNK, run_to_end, runner         # noqa: E402
from test_engine_tier import MemoryTier                                # noqa: E402
from engine.base.kv import BlockPool                                   # noqa: E402
from engine.base.prefix import PrefixCache                             # noqa: E402

TRANSIENT = {"_transient": True}


def settle(s, steps=300):
    for _ in range(steps):
        ran = s.once()
        if not ran and not s._waiting and not s._retiring and not s._resuming and not s._restoring:
            return
        threading.Event().wait(0.001)
    raise AssertionError("the server did not settle")


def cache(snapshots=4):
    c = PrefixCache(BLOCK, CHUNK, snapshots)
    c.bind(BlockPool(16, BLOCK, 4, 16))
    return c


class CachePolicyTests(unittest.TestCase):
    def test_a_transient_leaf_is_no_spill_candidate_until_a_prompt_adopts_it(self):
        c = cache()
        prod, probe = c.chain(list(range(4)))[BLOCK], c.chain(list(range(100, 104)))[BLOCK]
        c.insert(prod, (0,), BLOCK, c.take_snapshot())
        c.insert(probe, (1,), BLOCK, c.take_snapshot(), transient=True)
        self.assertEqual(c.spill_candidates(4), [prod])
        tokens, entry, h = c.lookup_chain({BLOCK: probe}, BLOCK + 1)
        self.assertEqual((tokens, h, entry.transient), (BLOCK, probe, False), "adopted: it has served, and lives as any other")
        self.assertEqual(c.spill_candidates(4), [prod, probe])

    def test_an_operator_s_pin_makes_it_an_ordinary_boundary(self):
        c = cache()
        h = c.chain(list(range(4)))[BLOCK]
        c.insert(h, (0,), BLOCK, c.take_snapshot(), transient=True)
        self.assertEqual(c.pin([h]), 1)
        self.assertFalse(c.entries[h].transient)

    def test_a_transient_boundary_gives_up_its_snapshot_before_every_other_kind(self):
        c = PrefixCache(BLOCK, CHUNK, 3)
        c.bind(BlockPool(16, BLOCK, 4, 16))
        c.insert(b"spilled", (), 0, c.take_snapshot())         # zero-length entries: the policy is all that is tested
        c.entries[b"spilled"].spilled = True                   # unadopted with a copy on the tier: first of the old classes
        c.insert(b"older", (), 0, c.take_snapshot())
        c.insert(b"probe", (), 0, c.take_snapshot(), transient=True)
        snap = c.take_snapshot()
        self.assertEqual(c.last_fade, b"probe")
        self.assertEqual(set(c.entries), {b"spilled", b"older"})
        c.give_snapshot(snap)


class RunnerTests(unittest.TestCase):
    def test_a_transient_row_s_boundaries_leave_with_it_and_its_blocks_are_anonymous_again(self):
        r, c = runner(blocks=16, snapshots=4)
        r.submit(0, 9, now=0, ids=list(range(9)))              # boundaries at 4 and 8
        r.transient.add(0)                                     # the server marks the row before its first step
        r.step(now=0)
        self.assertEqual([e.transient for e in c.entries.values()], [True, True], "cached for the turn, as any prompt's")
        run_to_end(r, 0)
        self.assertEqual((c.entries, c.faded, r.transient, r.transient_dropped), ({}, {}, set(), 2))
        self.assertEqual((r.kv.cached, r.kv.anonymous, len(c.free_snaps)), (0, 16, 4))
        c.check()
        kept, kc = runner(blocks=16, snapshots=4)              # the same prompt unmarked keeps both, as before
        kept.submit(0, 9, now=0, ids=list(range(9)))
        run_to_end(kept, 0)
        self.assertEqual((len(kc.entries), kept.kv.cached, kept.transient_dropped), (2, 2, 0))

    def test_a_boundary_a_production_prompt_adopted_while_the_row_lived_stays(self):
        r, c = runner(blocks=32, snapshots=4)
        ids = list(range(9))
        r.submit(0, 9, now=0, ids=ids)
        r.transient.add(0)
        r.step(now=0)                                          # [0, 8): boundaries 4 and 8, the row still prefilling
        chain = c.chain(ids)
        r.submit(1, 10, now=0, ids=ids[:8] + [50, 51])         # production shares the first two blocks and adopts 8
        self.assertEqual(r.reused_tokens, 8)
        run_to_end(r, 0)
        run_to_end(r, 1)
        self.assertEqual(set(c.entries), {chain[8]}, "the adopted leaf stays; its unadopted parent left with the row")
        self.assertFalse(c.entries[chain[8]].transient)
        self.assertEqual((r.transient_dropped, c.spill_candidates(4)), (1, [chain[8]]))
        c.check()

    def test_a_long_transient_prompt_displaces_its_own_boundaries_and_not_production_s(self):
        r, c = runner(blocks=32, snapshots=2)
        prod = c.chain(list(range(9)))
        r.submit(0, 9, now=0, ids=list(range(9)))              # production: boundaries 4 and 8 hold both slots
        run_to_end(r, 0)
        r.submit(1, 24, now=0, ids=list(range(100, 124)))      # a probe: six boundaries against two slots
        r.transient.add(1)
        run_to_end(r, 1)
        # its first boundary needed a slot while it had none of its own; every later one took its own earlier one
        self.assertEqual(set(c.entries), {prod[8]})
        self.assertEqual((r.snapshot_self_evicts, r.transient_dropped, len(c.free_snaps)), (5, 1, 1))
        c.check()

    def test_a_row_that_parks_carries_no_mark_into_its_next_turn(self):
        s = T.server(rows=1, keep_idle=True, tier=MemoryTier())
        s.engine.add(0, [1, 2, 3], max_new=1, temperature=0)
        s.runner.submit(0, 3)
        while 0 not in s.runner.idle:
            s.runner.step()
        s.runner.transient.add(0)
        s.runner.park(0)
        self.assertEqual(s.runner.transient, set())


class ServerTests(unittest.TestCase):
    def test_a_turn_not_retained_leaves_no_boundary_and_the_next_turn_in_its_row_keeps_its_own(self):
        s = T.server(rows=1, prefix=4)
        probe, _ = s.submit(list(range(1, 10)), 1, 0, options=dict(TRANSIENT))    # BLOCK 4: boundaries 4 and 8
        settle(s)
        self.assertEqual(s.take_result(probe), [9], "the answer is unchanged")
        prefix = s.runner.prefix
        self.assertEqual((prefix.entries, prefix.faded, s.runner.transient, s.runner.transient_dropped), ({}, {}, set(), 2))
        self.assertIn('st:prefix_transient_dropped_total{engine="st"} 2\n', s.metrics())
        chat, _ = s.submit(list(range(1, 10)), 1, 0)          # the same row, the same prompt, an ordinary turn
        settle(s)
        self.assertEqual(s.take_result(chat), [9])
        self.assertEqual(prefix.hits, 0, "nothing the probe made was left to hit")
        self.assertEqual(sorted((e.tokens, e.transient) for e in prefix.entries.values()), [(4, False), (8, False)])
        self.assertEqual(s.runner.transient_dropped, 2)
        prefix.check()

    def test_a_turn_not_retained_that_reuses_production_s_boundaries_leaves_them_and_takes_only_its_own(self):
        """A health check or any retain-false client without a salt shares production's prompts: what it adopted
        was production's before it came and stays after it went."""
        s = T.server(rows=2, prefix=4)
        chat, _ = s.submit(list(range(1, 10)), 1, 0)
        settle(s)
        prefix = s.runner.prefix
        production = sorted(prefix.entries)
        probe, _ = s.submit(list(range(1, 10)) + [11, 12, 13, 14], 1, 0, options=dict(TRANSIENT))   # adopts 8, makes 12
        settle(s)
        self.assertEqual((s.take_result(chat), s.take_result(probe)), ([9], [14]))
        self.assertEqual((prefix.hits, s.runner.reused_tokens), (1, 8))
        self.assertEqual(sorted(prefix.entries), production)
        self.assertEqual(s.runner.transient_dropped, 1)
        prefix.check()

    def test_boundaries_its_answer_crossed_leave_too_when_the_idle_turn_is_released(self):
        tier = MemoryTier()
        s = T.server(rows=2, keep_idle=True, tier=tier, prefix=4)
        probe, _ = s.submit([1, 2, 3, 4, 5, 6], 4, 0, options=dict(TRANSIENT))    # 4 in the prompt, 8 in the answer
        settle(s)
        self.assertEqual(s.take_result(probe), [6, 6, 6, 6])
        self.assertEqual((s.runner.prefix.entries, tier.keys(), s.runner.transient_dropped), ({}, [], 2))
        self.assertEqual(s.turns_not_retained, {"asked": 1})

    def test_nothing_a_transient_turn_made_is_written_to_the_prefix_tier(self):
        s = T.server(rows=2, prefix=4, prefix_tier=True)
        s.runner.spill_low_water = 10                           # write leaves out as soon as they exist
        probe, _ = s.submit(list(range(1, 10)), 3, 0, options=dict(TRANSIENT))
        settle(s)
        for _ in range(4):
            s.once()
        self.assertEqual((s.runner.prefix_spills, s.runner.prefix_tier_keys(), s.runner.prefix.entries), (0, [], {}))
        chat, _ = s.submit(list(range(21, 30)), 3, 0)
        settle(s)
        for _ in range(4):
            s.once()
        self.assertEqual(s.runner.prefix_spills, 1, "an ordinary turn's leaf is written as before")
        self.assertEqual((s.take_result(probe), s.take_result(chat)), ([9, 9, 9], [29, 29, 29]))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch for LocalTP")
class FourRankTests(unittest.TestCase):
    def test_every_rank_drops_the_same_boundaries(self):
        from engine.base.comm import LocalTP

        def rank_main(comm, _):
            s = T.server(comm=comm, rows=2, prefix=4)
            if comm.rank == 0:
                s.submit(list(range(1, 10)), 2, 0, options=dict(TRANSIENT))
                s.submit(list(range(21, 30)), 2, 0)
            for _ in range(150):
                s.once()
                threading.Event().wait(0.001)
            if comm.rank == 0:
                s.alive = False
            s.once()
            prefix = s.runner.prefix
            return (sorted((e.tokens, e.transient) for e in prefix.entries.values()), s.runner.transient_dropped,
                    sorted(s.runner.transient), s.runner.kv.cached, len(prefix.free_snaps))
        out = LocalTP(4).run(rank_main, None)
        self.assertTrue(all(row == out[0] for row in out), out)
        self.assertEqual(out[0][:3], ([(4, False), (8, False)], 2, []))


class ContractTests(unittest.TestCase):
    def test_rows_are_marked_between_admission_and_the_step_and_only_what_they_insert_is_marked(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        once = serve[serve.index("    def once(self) -> bool:"):serve.index("    def _death_note(")]
        self.assertLess(once.index("self._admit()"), once.index("self.runner.transient.add(row)"))
        self.assertLess(once.index("self.runner.transient.add(row)"), once.index("step = self.runner.step()"))
        runner_src = (ROOT / "engine/base/runner.py").read_text()
        self.assertIn("self.prefix.insert(h, blocks, position, snap, transient=seq in self.transient)", runner_src)
        restore = runner_src[runner_src.index("    def restore_finish("):runner_src.index("    def restore_undo(")]
        self.assertNotIn("transient", restore, "a boundary read back from the tier is production's")
        release = runner_src[runner_src.index("    def _release("):runner_src.index("    def _drop_transient(")]
        self.assertLess(release.index("self._drop_transient(seq)"), release.index("self._chain.pop(seq, None)"),
                        "the chain names the row's boundaries: it is read before it goes")


if __name__ == "__main__":
    unittest.main()
