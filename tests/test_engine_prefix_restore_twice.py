"""One boundary, read back from the prefix tier into two rows at once (2026-09-19 22:14).

A boot found the boundaries the boot before it had written to the prefix tier, and took six prompts at once that
all began with the same one. Memory did not hold it yet, so each prompt was admitted to read it back into its own
row. The first read to land made it a memory entry; the second had nothing to add to one, and `insert` raised
inside `restore_finish` after the read -- with the row still holding the blocks the read had filled. `_settle`
took that for a failed restore and gave the row back to the free rows as it was, and the prompt, turned into a
plain prefill ahead of the queue, took that very row: `seq 1 already owns resident resources`, on all four ranks,
in 'settling transfers'.

A prompt whose boundary another row is reading back now waits for it and adopts it from memory, as it waits for a
prefill that will cache it (45차 §23 B). A read that lands on a boundary memory holds anyway -- a prefill admitted
while it was still in memory crossed it again -- adopts the entry and gives its own copy back.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_engine_serve as T                                          # noqa: E402
from test_engine_prefix import BLOCK, run_to_end, runner               # noqa: E402
from test_engine_settle_symmetry import settle                         # noqa: E402

PROMPT = [11, 12, 13, 14, 15, 16, 17, 18]


class TwoReadsOfOneBoundaryTests(unittest.TestCase):
    def tiered_only(self):
        """A server whose prefix tier holds PROMPT's 8-token boundary and whose memory cannot serve it."""
        s = T.server(rows=3, prefix=3, prefix_tier=True)
        s.runner.spill_low_water = 10                                  # spill leaves as soon as they exist
        first, _ = s.submit(PROMPT, 1, 0)
        settle(s)
        s.take_result(first)
        for _ in range(4):
            s.once()                                                   # the leaf (8) is written out to the prefix tier
        other, _ = s.submit([31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], 1, 0)   # its snapshots: memory lets 8 go
        settle(s)
        s.take_result(other)
        h8 = s.runner.prefix.chain(PROMPT)[8]
        self.assertIn(h8, s.runner.prefix.tier_keys)
        self.assertFalse(s.runner.prefix.has(h8))
        return s

    def test_two_prompts_that_begin_with_a_tiered_boundary_arrive_together_and_share_one_read(self):
        s = self.tiered_only()
        reads, promote = [], s.runner.prefix_tier.tier.promote
        s.runner.prefix_tier.tier.promote = lambda key, storage, ids, extra=None: (reads.append(key),
                                                                                  promote(key, storage, ids, extra))[1]
        a, _ = s.submit(PROMPT + [21, 22], 1, 0)
        b, _ = s.submit(PROMPT + [23, 24, 25], 1, 0)                   # the same broadcast: admitted in the same step
        settle(s)                                                      # the crash: ValueError out of once()
        self.assertEqual(list(s.take_result(a)), [22])
        self.assertEqual(list(s.take_result(b)), [25])
        self.assertEqual(len(reads), 1, "the boundary was read back once")
        self.assertEqual(s.runner.prefix_restores, 1)
        self.assertEqual(s.runner.dedup_waits, 1, "the second prompt waited for the first one's read")
        self.assertEqual(s.cached_tokens(a, b), 16, "and both started after the boundary")
        self.assertEqual(sorted(s._free_rows), [0, 1, 2])
        self.assertFalse(s.runner.slot_of)
        self.assertFalse(any(s.runner.kv.tokens[row] for row in range(3)), "no row went back holding blocks")
        s.runner.prefix.check()

    def test_a_row_whose_finish_raised_goes_back_to_the_free_rows_empty(self):
        """The other half of the crash: a finish that raised was a failed restore, and the row went back as it was.
        Whatever a finish leaves behind, the next prompt to take the row finds it empty."""
        s = self.tiered_only()
        finish = s.runner.restore_finish

        def lands_then_raises(seq):
            finish(seq)                                                # the row holds the boundary's blocks now
            raise RuntimeError("a finish that fails after the read")
        s.runner.restore_finish = lands_then_raises
        a, _ = s.submit(PROMPT + [21, 22], 1, 0)
        settle(s)                                                      # before: seq 0 already owns resident resources
        self.assertEqual(list(s.take_result(a)), [22], "answered by the plain prefill the vote turned it into")
        self.assertEqual(sorted(s._free_rows), [0, 1, 2])
        self.assertFalse(any(s.runner.kv.tokens[row] for row in range(3)))
        s.runner.prefix.check()

    def test_a_read_that_lands_on_a_boundary_memory_already_holds_adopts_it_and_gives_its_copy_back(self):
        """What waiting cannot cover: memory gets the boundary another way while the read is in flight."""
        from test_engine_tier import MemoryTier, Storage
        from engine.base.tiered_kv import TieredKV
        r, cache = runner(blocks=32, snapshots=3)
        r.kv.attach_storage(Storage(32 * 4), 4)
        r.prefix_tier = TieredKV(r.kv, MemoryTier())
        r.spill_low_water = 10
        r.submit(0, 12, now=0, ids=list(range(12)))
        run_to_end(r, 0)
        r.step(now=0); r.step(now=0)                                    # the leaf (12) is written out
        h12 = cache.chain(list(range(12)))[12]
        r.submit(1, 12, now=0, ids=list(range(50, 62)))                # its snapshot goes to a fresh prompt ...
        run_to_end(r, 1)
        r.kv.reserve(3, r.kv.available * BLOCK); r.kv.release(3)       # ... and its blocks to a prompt that needs them all
        self.assertFalse(cache.has(h12) or h12 in cache.faded)
        r.restore_begin(2, h12, 12)
        r.restore_begin(3, h12, 12)                                    # the same boundary into a second row
        tokens, snap = r.restore_finish(2)
        free = len(cache.free_snaps)
        again, shared = r.restore_finish(3)                            # memory holds it now: before, `insert` raised here
        entry = cache.entries[h12]
        self.assertEqual((again, shared), (12, entry.snap), "the second row starts from the entry the first read made")
        self.assertEqual(snap, entry.snap)
        self.assertEqual(tuple(r.kv.row(3))[:3], entry.blocks, "on the entry's blocks")
        self.assertEqual(len(cache.free_snaps), free + 1, "and its own snapshot went back")
        self.assertEqual(r.prefix_restores, 1, "one boundary came back from the tier")
        cache.check()
        r.submit(2, 15, now=0, ids=list(range(12)) + [7, 7, 7], prepared=(tokens, snap))
        r.submit(3, 14, now=0, ids=list(range(12)) + [8, 8], prepared=(again, shared))
        self.assertEqual((r.state.computed[2], r.state.computed[3]), (12, 12))
        run_to_end(r, 2)
        run_to_end(r, 3)


if __name__ == "__main__":
    unittest.main()
