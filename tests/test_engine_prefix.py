"""Prefix reuse (engine/base/prefix.py) over the real pool, scheduler and runner, with a fake model. The unit is the
pool's BLOCK (45차 §23): a prefill chunk holds two here (BLOCK 4, CHUNK 8), so the boundary inside a chunk is `marks`."""
from __future__ import annotations

import unittest

from engine.base import scheduler as sched
from engine.base.kv import BlockPool, SlotPool
from engine.base.prefix import PrefixCache
from engine.base.record import Ring
from engine.base.runner import Runner, STEP_RECORD
from engine.base.scheduler import Contract

BLOCK, CHUNK = 4, 8                     # two blocks per chunk
CONTRACT = Contract(chunk_align=BLOCK, token_budget=CHUNK, draft_slots=0, max_wait_s=20.0, max_running=4)


class Model:
    """Prefill/decode bookkeeping only; checkpoint/restore record what the runner asked."""
    def __init__(self):
        self.ctx, self.live, self.calls = {}, set(), []

    def open(self, seq, slot):
        self.live.add(seq)

    def close(self, seq):
        self.live.discard(seq)

    def horizon(self, seq):
        return self.ctx[seq] + 1

    def context(self, seq):
        return self.ctx[seq]

    def prefill(self, seq, start, tokens, blocks, slot, marks=None):
        self.calls.append(("prefill", seq, start, tokens))
        for position, snap in sorted((marks or {}).items()):
            self.calls.append(("mark", seq, position, snap))
        self.ctx[seq] = start + tokens
        return False

    def decode(self, seqs, blocks, slots):
        for seq in seqs:
            self.ctx[seq] += 1
        return [True] * len(seqs)           # one token, done

    def checkpoint(self, seq, position, snap):
        self.calls.append(("checkpoint", seq, position, snap))

    def restore(self, seq, position, snap):
        self.calls.append(("restore", seq, position, snap))
        self.ctx[seq] = position


def runner(blocks=16, snapshots=2):
    cache = PrefixCache(BLOCK, CHUNK, snapshots)
    r = Runner(Model(), CONTRACT, BlockPool(blocks, BLOCK, 4, blocks), SlotPool(5), Ring(16, STEP_RECORD.size), prefix=cache)
    return r, cache


def run_to_end(r, seq):
    while seq in r.state.waiting or seq in r.state.running:
        r.step(now=0)


class PoolOwnershipTests(unittest.TestCase):
    def test_adopt_pin_release_count_owners(self):
        pool = BlockPool(8, BLOCK, 2, 8)
        pool.reserve(0, 8)                                   # row 0: two blocks
        first = tuple(pool.row(0)[:2])
        pool.pin(first)
        self.assertEqual([pool.refs[b] for b in first], [2, 2])
        pool.release(0)                                      # the cache still holds them
        self.assertEqual([pool.refs[b] for b in first], [1, 1])
        self.assertEqual(len(pool.free), 6)
        pool.adopt(1, first, 8)
        self.assertEqual(list(pool.row(1)[:2]), list(first))
        self.assertEqual(pool.tokens[1], 8)
        pool.reserve(1, 3)                                   # its own block after the prefix
        self.assertEqual(pool.tokens[1], 11)
        with self.assertRaises(ValueError):
            pool.adopt(1, first, 8)                          # not empty
        with self.assertRaises(ValueError):
            pool.adopt(0, first, 7)                          # not whole blocks
        self.assertEqual(pool.unpin(first), 0)               # row 1 still uses them
        pool.release(1)
        self.assertEqual(len(pool.free), 8)
        with self.assertRaises(ValueError):
            pool.pin(first)                                  # dead blocks cannot be pinned

    def test_reservation_reclaims_through_the_cache(self):
        r, cache = runner(blocks=4, snapshots=2)             # 16 tokens of blocks
        r.submit(0, 9, now=0, ids=list(range(9)))            # block boundaries at 4 (marked) and 8 (checkpoint), then a token
        run_to_end(r, 0)
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual(r.kv.available, 4)                  # the cached blocks count as available ...
        self.assertEqual(len(r.kv.free), 2)                  # ... but only two are free right now
        r.submit(1, 13, now=0, ids=list(range(100, 113)))    # needs four blocks: the cache gives its two back
        self.assertEqual(len(cache.entries), 0)
        self.assertEqual(cache.evictions, 2)
        run_to_end(r, 1)
        self.assertEqual(r.kv.available, 4)


class PrefixCacheTests(unittest.TestCase):
    def test_chain_hashes_depend_on_the_prefix_only(self):
        c = PrefixCache(BLOCK, CHUNK, 2)
        a, b = c.chain(list(range(20))), c.chain(list(range(16)) + [99, 99, 99, 99])
        self.assertEqual(sorted(a), [4, 8, 12, 16, 20])                     # every block boundary, not only the chunk's
        self.assertEqual(a[8], b[8]); self.assertEqual(a[16], b[16]); self.assertNotEqual(a[20], b[20])
        self.assertNotEqual(a[8], c.chain([1] + list(range(1, 20)))[8])
        self.assertEqual(c.chain([1, 2, 3]), {})

    def test_salts_split_identical_ids_from_the_chunk_that_holds_them_onward(self):
        c = PrefixCache(BLOCK, CHUNK, 2)
        ids = list(range(24))
        plain, cat, dog = c.chain(ids), c.chain(ids, [(10, b"cat")]), c.chain(ids, [(10, b"dog")])
        self.assertEqual(plain[8], cat[8])                                    # before the picture: the same boundary
        self.assertNotEqual(plain[16], cat[16]); self.assertNotEqual(cat[16], dog[16]); self.assertNotEqual(cat[24], dog[24])
        self.assertEqual(c.chain(ids, [(10, b"cat")]), cat)                   # deterministic
        self.assertEqual(c.chain(ids, [(3, b"x"), (10, b"cat")])[8], c.chain(ids, [(10, b"cat"), (3, b"x")])[8])   # order-free

    def test_second_prompt_reuses_the_boundary_and_prefills_the_rest(self):
        r, cache = runner(snapshots=4)
        r.submit(0, 19, now=0, ids=list(range(19)))
        run_to_end(r, 0)
        prefills = [c for c in r.model.calls if c[0] == "prefill"]
        self.assertEqual(prefills, [("prefill", 0, 0, 8), ("prefill", 0, 8, 8), ("prefill", 0, 16, 3)])
        # the boundary inside each chunk is marked before the step (taken on the way), the one at its end copied after it
        self.assertEqual([c for c in r.model.calls if c[0] in ("mark", "checkpoint")],
                         [("mark", 0, 4, 0), ("checkpoint", 0, 8, 1), ("mark", 0, 12, 2), ("checkpoint", 0, 16, 3)])
        self.assertEqual(len(cache.entries), 4)
        r.model.calls.clear()
        r.submit(1, 20, now=0, ids=list(range(13)) + [7] * 7)                # shares 13 tokens: three whole blocks, not one chunk
        self.assertEqual(r.state.computed[1], 12)
        self.assertEqual(r.model.calls, [("restore", 1, 12, 2)])
        self.assertEqual(list(r.kv.row(1)[:3]), list(cache.entries[cache.chain(list(range(12)))[12]].blocks))
        run_to_end(r, 1)
        self.assertEqual([c for c in r.model.calls if c[0] == "prefill"], [("prefill", 1, 12, 8)])
        self.assertEqual(cache.hits, 1)
        self.assertEqual(r.kv.available, r.kv.num_blocks, "everything is reclaimable once both are done")
        self.assertEqual(r.state, sched.State())

    def test_boundary_at_the_prompt_end_is_cached_but_never_adopted_as_whole(self):
        r, cache = runner()
        r.submit(0, 8, now=0, ids=list(range(8)))            # exactly one chunk: boundaries at 4 (marked) and 8 (checkpoint)
        run_to_end(r, 0)
        self.assertEqual(len(cache.entries), 2)
        r.model.calls.clear()
        r.submit(1, 8, now=0, ids=list(range(8)))            # the same prompt: the whole-prompt boundary never serves, the block before it does
        self.assertEqual(r.state.computed[1], 4)
        self.assertEqual(cache.hits, 1)
        run_to_end(r, 1)
        r.submit(2, 9, now=0, ids=list(range(9)))            # one token longer: the boundary serves
        self.assertEqual(r.state.computed[2], 8)
        run_to_end(r, 2)

    def test_snapshots_evict_least_recently_used_and_unpin_blocks(self):
        r, cache = runner(blocks=32, snapshots=1)
        r.submit(0, 9, now=0, ids=list(range(9)))
        run_to_end(r, 0)                                      # the one snapshot: boundary 4 (marked), then 8 evicts it (checkpoint)
        self.assertEqual([c for c in r.model.calls if c[0] in ("mark", "checkpoint")], [("mark", 0, 4, 0), ("checkpoint", 0, 8, 0)])
        r.submit(1, 9, now=0, ids=list(range(50, 59)))       # another prompt's boundary: the only snapshot moves to it
        run_to_end(r, 1)
        self.assertEqual(len(cache.entries), 1)
        self.assertEqual(cache.evictions, 3)
        self.assertIn(cache.chain(list(range(50, 59)))[8], cache.entries)
        self.assertEqual(r.kv.available, 32)
        self.assertEqual(len(r.kv.free), 30)                # boundary 8 pins two blocks
        cache.clear()
        self.assertEqual(len(r.kv.free), 32)

    def test_failed_admission_returns_the_adopted_prefix(self):
        r, cache = runner(blocks=4)
        r.submit(0, 9, now=0, ids=list(range(9)))
        run_to_end(r, 0)
        r.model.open = lambda seq, slot: (_ for _ in ()).throw(RuntimeError("open failed"))
        with self.assertRaisesRegex(RuntimeError, "open failed"):
            r.submit(1, 9, now=0, ids=list(range(9)))
        self.assertEqual(r.kv.tokens[1], 0)
        self.assertEqual(len(cache.entries), 2)              # the cache keeps its boundaries
        self.assertEqual(r.kv.available, 4)

    def test_ids_must_match_the_prompt(self):
        r, _ = runner()
        with self.assertRaises(ValueError):
            r.submit(0, 5, now=0, ids=[1, 2])


if __name__ == "__main__":
    unittest.main()
