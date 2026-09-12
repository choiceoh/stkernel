"""Prefix reuse (engine/base/prefix.py) over the real pool, scheduler and runner, with a fake model. The unit is the
pool's BLOCK (45차 §23): a prefill chunk holds two here (BLOCK 4, CHUNK 8), so the boundary inside a chunk is `marks`."""
from __future__ import annotations

import unittest

from engine.base import scheduler as sched
from engine.base.kv import BlockPool, SlotPool
from engine.base.prefix import PrefixCache, tenant_salt
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
        self.snaps = getattr(self, "snaps", {}); self.snaps[snap] = bytearray([position % 256] * 4)

    def restore(self, seq, position, snap):
        self.calls.append(("restore", seq, position, snap))
        self.ctx[seq] = position

    def snapshot_bytes(self, snap):
        self.snaps = getattr(self, "snaps", {})
        return self.snaps.setdefault(snap, bytearray(4))


def runner(blocks=16, snapshots=2):
    cache = PrefixCache(BLOCK, CHUNK, snapshots)
    r = Runner(Model(), CONTRACT, BlockPool(blocks, BLOCK, 4, blocks), SlotPool(5), Ring(16, STEP_RECORD.size), prefix=cache)
    return r, cache


def run_to_end(r, seq):
    while seq in r.state.waiting or seq in r.state.running:
        r.step(now=0)
        r.prefix.check()                                     # the two resources' invariants hold at every step


class PoolOwnershipTests(unittest.TestCase):
    def test_a_claim_keeps_blocks_in_the_free_list_under_the_boundary_s_name(self):
        pool = BlockPool(8, BLOCK, 2, 8)
        pool.forget = lambda b: pool.disclaim((b,), pool.CACHED)     # a cache of one boundary
        pool.reserve(0, 8)                                   # row 0: two blocks
        first = tuple(pool.row(0)[:2])
        pool.claim(first, pool.CACHED)
        self.assertEqual([pool.refs[b] for b in first], [1, 1], "a claim is not an owner")
        pool.release(0)
        self.assertEqual([pool.refs[b] for b in first], [0, 0])
        self.assertEqual((pool.available, pool.anonymous, pool.cached), (8, 6, 2))
        pool.adopt(1, first, 8)                              # out of the free list, not copied (vLLM's touch)
        self.assertEqual(list(pool.row(1)[:2]), list(first))
        self.assertEqual((pool.tokens[1], pool.available, pool.cached), (8, 6, 0))
        pool.reserve(1, 3)                                   # its own block after the prefix
        self.assertEqual(pool.tokens[1], 11)
        with self.assertRaises(ValueError):
            pool.adopt(1, first, 8)                          # not empty
        with self.assertRaises(ValueError):
            pool.adopt(0, first, 7)                          # not whole blocks
        pool.release(1)
        self.assertEqual((pool.available, pool.cached), (8, 2), "the boundary still holds them")
        pool.disclaim(first, pool.CACHED)
        self.assertEqual((pool.anonymous, pool.cached), (8, 0))
        with self.assertRaises(ValueError):
            pool.adopt(0, first, 8)                          # nobody holds them: they are not a prefix any more

    def test_a_reservation_spends_anonymous_blocks_before_any_boundary_pays(self):
        r, cache = runner(blocks=8, snapshots=4)
        r.submit(0, 9, now=0, ids=list(range(9)))            # three blocks: boundaries at 4 and 8
        run_to_end(r, 0)
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual((r.kv.available, r.kv.anonymous, r.kv.cached), (8, 6, 2))
        r.submit(1, 20, now=0, ids=list(range(100, 120)))    # five blocks, and six are anonymous
        self.assertEqual(len(cache.entries), 2, "nothing was evicted to make room for a prompt the free blocks fit")
        self.assertEqual((cache.evictions, r.kv.anonymous, r.kv.cached), (0, 1, 2))

    def test_an_operator_s_pinned_boundary_is_the_last_block_anybody_takes(self):
        r, cache = runner(blocks=8, snapshots=4)
        r.submit(0, 5, now=0, ids=list(range(5)))            # a boundary at 4: one block
        run_to_end(r, 0)
        r.submit(1, 5, now=0, ids=list(range(50, 55)))       # another prompt's, another block
        run_to_end(r, 1)
        self.assertEqual((r.kv.anonymous, r.kv.cached), (6, 2))
        warm = cache.chain(list(range(5)))[4]
        self.assertEqual(cache.pin([warm]), 1)
        r.submit(2, 25, now=0, ids=list(range(100, 125)))    # seven blocks: six anonymous, then the boundary nobody pinned
        self.assertEqual(list(cache.entries), [warm])
        block = cache.entries[warm].blocks[0]
        self.assertEqual((r.kv.pins[block], r.kv.claims[block]), (1, 0))
        self.assertEqual(cache.unpin_all(), 1)
        self.assertEqual((r.kv.pins[block], r.kv.claims[block]), (0, 1), "an ordinary boundary's block again")

    def test_a_block_leaves_one_at_a_time_and_only_its_own_boundary_with_it(self):
        r, cache = runner(blocks=6, snapshots=4)
        r.submit(0, 17, now=0, ids=list(range(17)))          # five blocks: boundaries at 4, 8, 12 and 16
        run_to_end(r, 0)
        self.assertEqual(len(cache.entries), 4)
        self.assertEqual((r.kv.anonymous, r.kv.cached), (2, 4))   # the last block is nobody's: no boundary ends there
        r.submit(1, 9, now=0, ids=list(range(200, 209)))     # three blocks: both anonymous ones, then exactly one boundary's
        self.assertEqual(sorted(e.tokens for e in cache.entries.values()), [4, 8, 12],
                         "the longest boundary's tail block went, and the prefixes it shares stayed")
        self.assertEqual(cache.evictions, 1)

    def test_reservation_reclaims_through_the_cache(self):
        r, cache = runner(blocks=4, snapshots=2)             # 16 tokens of blocks
        r.submit(0, 9, now=0, ids=list(range(9)))            # block boundaries at 4 (marked) and 8 (checkpoint), then a token
        run_to_end(r, 0)
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual(r.kv.available, 4)                  # every block is free ...
        self.assertEqual((r.kv.anonymous, r.kv.cached), (2, 2))   # ... but two of them are a boundary's until one is needed
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

    def test_a_tenant_salt_separates_the_same_prompt_from_the_first_block_on(self):
        c = PrefixCache(BLOCK, CHUNK, 2)
        ids = list(range(20))
        plain, red, blue = c.chain(ids), c.chain(ids, [tenant_salt("red")]), c.chain(ids, [tenant_salt("blue")])
        self.assertEqual(sorted(red), sorted(plain))                       # the same boundaries, different names
        for t in plain:
            self.assertNotEqual(red[t], plain[t]); self.assertNotEqual(red[t], blue[t])
        self.assertEqual(c.chain(ids, [tenant_salt("red")]), red)          # deterministic: every rank agrees without a message
        self.assertEqual(red, c.chain(ids, [tenant_salt(b"red")]))         # a string or its bytes are one tenant
        salted = c.chain(ids, [tenant_salt("red"), (10, b"cat")])
        self.assertEqual(salted[8], red[8])                                # a picture still separates only from its block on
        self.assertNotEqual(salted[12], red[12])
        self.assertEqual(c.extend_chain(c.chain(ids[:9], [tenant_salt("red")]), ids[8:], [tenant_salt("red")], start=8),
                         c.chain(ids, [tenant_salt("red")]))               # the salt is inside the chain already

    def test_one_tenant_s_boundaries_are_not_another_s(self):
        r, cache = runner(blocks=32, snapshots=8)
        ids = list(range(19))
        red, blue = [tenant_salt("red")], [tenant_salt("blue")]
        r.submit(0, 19, now=0, ids=ids, salts=red, chain=cache.chain(ids, red))
        run_to_end(r, 0)
        self.assertEqual(len(cache.entries), 4)
        self.assertEqual(cache.peek(ids, blue), 0, "another tenant sees nothing of it")
        r.submit(1, 19, now=0, ids=ids, salts=blue, chain=cache.chain(ids, blue))
        self.assertEqual((r.state.computed[1], cache.hits), (0, 0))
        run_to_end(r, 1)
        r.submit(2, 19, now=0, ids=ids, salts=red, chain=cache.chain(ids, red))
        self.assertEqual((r.state.computed[2], cache.hits), (16, 1), "its own tenant's boundary serves it")

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

    def test_snapshots_take_the_least_recently_used_boundary_and_free_its_blocks(self):
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
        self.assertEqual((r.kv.anonymous, r.kv.cached), (30, 2))   # boundary 8 holds two blocks; with no tier, the rest are gone
        self.assertEqual(len(cache.faded), 0)
        cache.clear()
        self.assertEqual(r.kv.anonymous, 32)

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

    def test_boundaries_crossed_while_generating_serve_a_resent_conversation(self):
        class Generating(Model):
            """Knows its history: the prompt, then one generated token (200 + position) per decode step."""
            def __init__(self):
                super().__init__(); self.ids, self.left = {}, {}
            def submit_ids(self, seq, ids, left):
                self.ids[seq], self.left[seq] = list(ids), left
            def history(self, seq):
                return self.ids[seq]
            def decode(self, seqs, blocks, slots):
                out = []
                for seq in seqs:
                    self.ids[seq].append(200 + self.ctx[seq]); self.ctx[seq] += 1; self.left[seq] -= 1
                    out.append(self.left[seq] == 0)
                return out
        m = Generating()
        cache = PrefixCache(BLOCK, CHUNK, 8)
        r = Runner(m, CONTRACT, BlockPool(16, BLOCK, 4, 16), SlotPool(5), Ring(16, STEP_RECORD.size), prefix=cache)
        m.submit_ids(0, list(range(6)), 6)                    # 6 prompt tokens, then 6 generated: boundaries 8 and 12 fall in generation
        r.submit(0, 6, now=0, ids=list(range(6)))
        run_to_end(r, 0)
        checkpoints = [c for c in m.calls if c[0] == "checkpoint"]
        self.assertEqual([c[2] for c in checkpoints], [8, 12])
        self.assertEqual(sorted(e.tokens for e in cache.entries.values()), [4, 8, 12])
        history = list(m.history(0))                          # the conversation, resent after it was forgotten
        m.submit_ids(1, history[:11] + [9, 9], 1)
        r.submit(1, 13, now=0, ids=history[:11] + [9, 9])
        self.assertEqual(r.state.computed[1], 8)              # the answer's first block is reused
        self.assertEqual(cache.hits, 1)

    def test_boundaries_a_request_adopted_outlive_the_ones_nobody_asked_for(self):
        c = PrefixCache(BLOCK, CHUNK, 2)
        pool = BlockPool(16, BLOCK, 4, 16); c.bind(pool)
        c.insert(b"hot", (), 0, c.take_snapshot())            # zero-length entries: the policy is all that is tested
        c.tick += 1; c.entries[b"hot"].hits = 1              # adopted once
        c.insert(b"fresh", (), 0, c.take_snapshot())
        snap = c.take_snapshot()                              # room is needed: the fresh, never-adopted one leaves
        self.assertIn(b"hot", c.entries); self.assertNotIn(b"fresh", c.entries)
        c.give_snapshot(snap)

    def test_a_running_prefill_s_boundaries_are_visible_ahead_and_gone_once_cached(self):
        r, cache = runner(snapshots=8)
        ids = list(range(20))
        r.submit(0, 20, now=0, ids=ids)
        r.step(now=0)                                                   # computed 8: boundaries 12 and 16 still to come
        h16 = cache.chain(ids)[16]
        self.assertEqual(r.shared_ahead(ids, (), above=8), h16)
        self.assertIsNone(r.shared_ahead(ids, (), above=16))            # nothing beyond 16 inside a 20-token prompt
        self.assertIsNone(r.shared_ahead(list(range(100, 120)), ()))    # a different prompt shares nothing
        run_to_end(r, 0)
        self.assertIsNone(r.shared_ahead(ids, (), above=8))             # cached now: nothing to wait for

    def test_a_leaf_spills_to_the_prefix_tier_and_a_later_prompt_restores_it_with_its_snapshot(self):
        from test_engine_tier import MemoryTier, Storage
        from engine.base.tiered_kv import TieredKV
        r, cache = runner(blocks=32, snapshots=3)
        r.kv.attach_storage(Storage(32 * 4), 4)
        r.prefix_tier = TieredKV(r.kv, MemoryTier())
        r.spill_low_water = 10                                          # write leaves out as soon as they exist
        r.submit(0, 12, now=0, ids=list(range(12)))                     # boundaries 4 (mark), 8 (checkpoint), 12 (checkpoint)
        run_to_end(r, 0)
        r.step(now=0); r.step(now=0)                                    # idle steps: maintain lands the spill of the leaf (12)
        self.assertEqual(r.prefix_spills, 1)
        h12 = cache.chain(list(range(12)))[12]
        self.assertEqual(list(cache.tier_keys), [h12])
        self.assertTrue(cache.entries[h12].spilled)
        # three fresh boundaries take every snapshot: the spilled leaf leaves memory but not the tier
        r.submit(1, 12, now=0, ids=list(range(50, 62)))
        run_to_end(r, 1)
        self.assertNotIn(h12, cache.entries)
        self.assertIn(h12, cache.tier_keys)
        r.kv.reserve(3, r.kv.available * BLOCK); r.kv.release(3)   # and a prompt that needs every block takes its blocks too
        self.assertNotIn(h12, cache.faded)
        # a prompt that starts with the first one: the tier's copy is read into the row, snapshot and all
        ids = list(range(12)) + [7, 7, 7]
        self.assertEqual(cache.tier_lookup(ids, (), cache.peek(ids)), (12, h12))
        r.restore_begin(2, h12, 12)
        self.assertTrue(r.transfer_done(2))
        tokens, snap = r.restore_finish(2)
        self.assertEqual(tokens, 12)
        self.assertEqual(bytes(r.model.snaps[snap]), bytes([12] * 4))    # the boundary's snapshot bytes came back with it
        r.submit(2, 15, now=0, ids=ids, prepared=(tokens, snap))
        self.assertEqual(r.state.computed[2], 12)
        self.assertEqual(r.model.calls[-1], ("restore", 2, 12, snap))
        self.assertEqual((r.prefix_restores, r.reused_tokens), (1, 12))
        run_to_end(r, 2)
        self.assertEqual([c for c in r.model.calls if c[0] == "prefill"][-1], ("prefill", 2, 12, 3))

    def test_a_faded_boundary_keeps_its_blocks_and_comes_back_by_the_snapshot_alone(self):
        from test_engine_tier import MemoryTier, Storage
        from engine.base.tiered_kv import TieredKV
        r, cache = runner(blocks=32, snapshots=3)
        r.kv.attach_storage(Storage(32 * 4), 4)
        r.prefix_tier = TieredKV(r.kv, MemoryTier())
        r.spill_low_water = 10
        r.submit(0, 12, now=0, ids=list(range(12)))          # boundaries 4, 8, 12; the leaf (12) is written out
        run_to_end(r, 0)
        r.step(now=0); r.step(now=0)
        h12 = cache.chain(list(range(12)))[12]
        self.assertEqual(r.prefix_spills, 1)
        blocks = cache.entries[h12].blocks
        # every snapshot goes to a fresh prompt's boundaries. The spilled leaf gives its snapshot up first and keeps
        # its blocks -- nobody needed those -- so its KV is still exactly where it was.
        r.submit(1, 12, now=0, ids=list(range(50, 62)))
        run_to_end(r, 1)
        self.assertEqual((list(cache.faded), cache.faded[h12].blocks), ([h12], blocks))
        # only its last block is the faded one's alone: the two before it are still a live shorter boundary's,
        # which is a better claim, so they are graded by that one and leave even later
        self.assertEqual((r.kv.faded, r.kv.cached), (1, 2))
        self.assertFalse(cache.has(h12), "memory cannot serve it: the state has to be read back first")
        ids = list(range(12)) + [7, 7, 7]
        self.assertEqual(cache.tier_lookup(ids, (), cache.peek(ids)), (12, h12))
        seen, promote = [], r.prefix_tier.tier.promote
        r.prefix_tier.tier.promote = lambda seq, storage, ids_, extra=None: (seen.append(ids_), promote(seq, storage, ids_, extra))[1]
        r.restore_begin(2, h12, 12)
        self.assertEqual(list(r.kv.row(2))[:3], list(blocks), "the row adopted the blocks where they lay")
        self.assertTrue(r.transfer_done(2))
        tokens, snap = r.restore_finish(2)
        self.assertEqual(seen, [None], "only the snapshot was read; the KV under it was never touched")
        self.assertEqual((tokens, bytes(r.model.snaps[snap])), (12, bytes([12] * 4)))
        self.assertNotIn(h12, cache.faded)
        self.assertEqual(cache.entries[h12].blocks, blocks, "a whole boundary again, on the very blocks it always had")
        r.submit(2, 15, now=0, ids=ids, prepared=(tokens, snap))
        self.assertEqual(r.state.computed[2], 12)
        run_to_end(r, 2)

    def test_ids_must_match_the_prompt(self):
        r, _ = runner()
        with self.assertRaises(ValueError):
            r.submit(0, 5, now=0, ids=[1, 2])


class ChainTests(unittest.TestCase):
    def test_extend_chain_equals_a_fresh_chain_and_hashes_only_the_new_blocks(self):
        c = PrefixCache(BLOCK, CHUNK, 2)
        ids = list(range(23))
        chain = c.chain(ids[:9], [(2, b"x")])                                          # boundaries 4 and 8
        grown = c.extend_chain(chain, ids[8:], [(2, b"x"), (14, b"y")], start=8)     # only the tail from the last boundary is given
        with self.assertRaises(ValueError):
            c.extend_chain(chain, ids[9:], (), start=9)                                # a tail that starts past it cannot continue
        self.assertEqual(grown, c.chain(ids, [(2, b"x"), (14, b"y")]))
        self.assertEqual(c.extend_chain(c.chain(ids), ids[20:], (), start=20), c.chain(ids))   # nothing new: unchanged
        self.assertEqual(c.extend_chain({}, ids, ()), c.chain(ids))                  # from nothing: the whole chain

    def test_chain_lookups_take_a_chain_computed_once(self):
        r, cache = runner(snapshots=4)
        ids = list(range(12))
        r.submit(0, 12, now=0, ids=ids)
        run_to_end(r, 0)
        chain = cache.chain(ids + [1, 2])
        self.assertEqual(cache.peek_chain(chain, 14), 12)
        hits = cache.hits
        self.assertEqual(cache.lookup_chain(chain, 14)[0], 12)
        self.assertEqual(cache.hits, hits + 1)
        self.assertIsNone(cache.tier_lookup_chain(chain, 14, 12))


if __name__ == "__main__":
    unittest.main()
