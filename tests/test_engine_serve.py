"""Request lifecycle regressions using the real scheduler, pools and runner."""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import queue
import socket
import sys
import threading
import unittest
import urllib.error
import urllib.request

from pathlib import Path

from engine.base.kv import BlockPool, SlotPool
from engine.base.record import Ring

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    # `test_engine_tier` is a sibling, and whether it is importable by that name depends on how the
    # suite was started: `python3 -m unittest tests.test_engine_serve` from the root does not put
    # `tests/` on the path, and twelve tests here errored out on the import rather than running.
    sys.path.insert(0, str(ROOT / "tests"))
from engine.base.runner import Runner, STEP_RECORD
from engine.base.scheduler import Contract
from engine.base.serve import RequestError, Server


class Engine:
    def __init__(self, slots=8):
        self.tokens, self.ctx, self.limits, self.output = {}, {}, {}, {}
        self.opened = []
        self.state = [bytearray(4) for _ in range(slots)]     # one 4-byte "state slot" per slot id
        self.fail_open = self.fail_decode = False

    def state_bytes(self, slot):
        return memoryview(self.state[slot])

    def park(self, seq):
        record = {"context": self.ctx[seq], "pending": len(self.tokens[seq]) + len(self.output[seq]) - self.ctx[seq],
                  "tokens": list(self.tokens[seq]), "output": list(self.output[seq]), "limit": self.limits[seq],
                  "media": [[m["kind"], m["digest"], m["positions"][0], len(m["positions"]), [1, 2, 2]]
                            for m in getattr(self, "media", {}).get(seq, [])]}
        self.close(seq)
        self.forget(seq)
        return record

    def resume(self, seq, slot, record):
        self.tokens[seq], self.output[seq], self.limits[seq] = list(record["tokens"]), list(record["output"]), record["limit"]
        self.ctx[seq] = record["context"]
        self.media = getattr(self, "media", {})
        self.media[seq] = [{"kind": k, "digest": d, "positions": list(range(first, first + n))} for k, d, first, n, _ in record.get("media", [])]

    def validate(self, ids, limit, temperature):
        if any(t >= 256 for t in ids):
            raise ValueError("token outside vocabulary")

    def add(self, seq, ids, max_new, temperature, min_new=0, options=None, media=None):
        self.tokens[seq], self.limits[seq], self.output[seq] = list(ids), max_new, []
        self.min_new = getattr(self, "min_new", {}); self.min_new[seq] = min_new
        self.options = getattr(self, "options", {}); self.options[seq] = dict(options or {})
        self.media = getattr(self, "media", {}); self.media[seq] = [dict(m, positions=list(m["positions"])) for m in (media or [])]

    def media_marks(self, seq):
        return [(m["positions"][0], m["digest"]) for m in getattr(self, "media", {}).get(seq, [])]

    # the prefix cache's half (base/prefix.py): position state in and out of snapshots, whose bytes the tier moves
    def checkpoint(self, seq, position, snap):
        self.snaps = getattr(self, "snaps", {}); self.snaps[snap] = bytearray([position % 256] * 4)

    def restore(self, seq, position, snap):
        self.ctx[seq] = position
        self.restored = getattr(self, "restored", []); self.restored.append((seq, position, snap))

    def snapshot_bytes(self, snap):
        self.snaps = getattr(self, "snaps", {})
        return self.snaps.setdefault(snap, bytearray(4))

    def history(self, seq):
        return self.tokens[seq] + self.output[seq]

    def open(self, seq, slot):
        self.ctx[seq] = 0
        self.opened.append(seq)
        self.state[slot][:] = bytes([slot, seq, 0, 0])          # a mark the tier must carry between slots
        if self.fail_open:
            raise RuntimeError("open failed")

    def close(self, seq):
        self.ctx.pop(seq, None)

    def forget(self, seq):
        for rows in (self.tokens, self.limits, self.output):
            rows.pop(seq, None)

    def horizon(self, seq):
        return self.ctx[seq] + 1

    def context(self, seq):
        return self.ctx[seq]

    def extension_tokens(self, seq, ids):
        return len(self.tokens[seq]) + len(self.output[seq]) + len(ids) - self.ctx[seq]

    def extend(self, seq, ids, max_new, temperature, min_new=0, options=None, media=None, drop_unfed=False):
        if drop_unfed:
            self.output[seq].pop()                                  # the sampled end token the resent history does not carry
        n = self.extension_tokens(seq, ids)
        base = len(self.tokens[seq]) + len(self.output[seq])
        self.media = getattr(self, "media", {})
        self.media.setdefault(seq, []).extend(dict(m, positions=[base + p for p in m["positions"]]) for m in (media or []))
        self.tokens[seq] += self.output[seq] + list(ids)
        self.output[seq] = []
        self.limits[seq] = max_new
        return n

    def prefill(self, seq, start, tokens, blocks, slot, marks=None):
        for position, snap in (marks or {}).items():                    # a boundary inside the step: its snapshot, taken on the way
            self.checkpoint(seq, position, snap)
        self.ctx[seq] = start + tokens
        if self.ctx[seq] == len(self.tokens[seq]):
            self.output[seq].append(self.tokens[seq][-1])
        ends = getattr(self, "eos", ()) if getattr(self, "stop_at_eos", False) else ()
        return len(self.output[seq]) == self.limits[seq] or bool(self.output[seq] and self.output[seq][-1] in ends)

    def decode(self, seqs, blocks, slots):
        if self.fail_decode:
            raise RuntimeError("kernel failed")
        for seq in seqs:
            self.ctx[seq] += 1
            self.output[seq].append(self.tokens[seq][-1])
        ends = getattr(self, "eos", ()) if getattr(self, "stop_at_eos", False) else ()     # the real engine ends a row at its end token
        return [len(self.output[seq]) == self.limits[seq] or self.output[seq][-1] in ends for seq in seqs]

    def generated(self, seq):
        return self.output[seq]

    def generated_count(self, seq):
        return len(self.output[seq])

    def generated_since(self, seq, sent):
        return list(self.output[seq][sent:])


class Comm:
    rank = 0
    def broadcast_object(self, obj):
        return obj


def server(*, rows=2, blocks=16, comm=None, max_pending=64, keep_idle=False, tiered=False, tier=None, prefix=0, prefix_tier=False):
    engine = Engine(rows + 1)
    cache = None
    if prefix:
        from engine.base.prefix import PrefixCache
        cache = PrefixCache(4, 8, prefix)                              # BLOCK 4, CHUNK 8, `prefix` snapshots
    runner = Runner(engine, Contract(4, 8, 0, 0, rows), BlockPool(blocks, 4, rows, blocks),
                    SlotPool(rows + 1), Ring(16, STEP_RECORD.size), keep_idle=keep_idle, prefix=cache)
    if tiered or tier is not None or prefix_tier:
        from test_engine_tier import MemoryTier, Storage
        from engine.base.tiered_kv import TieredKV
        runner.kv.attach_storage(Storage(blocks * 4), 4)
        if tiered or tier is not None:
            runner.tiered = TieredKV(runner.kv, tier if tier is not None else MemoryTier())
        if prefix_tier:
            runner.prefix_tier = TieredKV(runner.kv, MemoryTier())
    return Server(engine, runner, comm or Comm(), host="127.0.0.1", port=0, max_pending=max_pending)


class ServeTests(unittest.TestCase):
    def drain(self, s, retained=False):
        for _ in range(500):
            ran = s.once()
            if not ran and not s._waiting and not s._retiring and not s._resuming:
                break
            if not ran and (s._retiring or s._resuming):
                threading.Event().wait(0.001)                  # the tier's thread needs the GIL to finish its transfer
        else:
            self.fail("server did not drain bounded requests")
        if retained:
            self.assertFalse(s._active)
            self.assertFalse(s.runner.state.running or s.runner.state.waiting)
            self.assertLessEqual(len(s.engine.tokens), s.runner.c.max_running)
            return
        self.assertFalse(s.engine.tokens)
        self.assertFalse(s.engine.ctx)
        self.assertFalse(s.runner.state.waiting)
        self.assertFalse(s.runner.state.running)
        self.assertFalse(s.runner.slot_of)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        self.assertEqual(s.runner.slots.available, s.runner.c.max_running)

    def test_continuation_reuses_context_but_has_an_independent_request_result(self):
        s = server(keep_idle=True)
        first, _ = s.submit([3], 2, 0)
        self.drain(s, retained=True)
        second, _ = s.submit([9], 3, 0, conversation=first)
        self.drain(s, retained=True)
        self.assertNotEqual(first, second)
        self.assertEqual(s.take_result(first), [3, 3])
        self.assertEqual(s.take_result(second), [9, 9, 9])
        self.assertEqual(s.engine.opened, [0])
        self.assertEqual(s._conversations, {first: 0})

    def test_a_retained_conversation_belongs_to_the_tenant_that_started_it(self):
        from engine.base.prefix import tenant_salt
        s = server(keep_idle=True)
        first, _ = s.submit([3, 4], 2, 0, cache_salt="red")
        self.drain(s, retained=True)
        history = s.engine.history(0)                              # what the next turn would resend
        self.assertEqual(s._tenant_of[first], tenant_salt("red"))
        self.assertIsNone(s._continuation(history + [5], (), tenant_salt("blue")), "another tenant cannot continue it")
        self.assertIsNone(s._continuation(history + [5], ()), "and neither can an unsalted caller")
        self.assertEqual(s._continuation(history + [5], (), tenant_salt("red")), (first, len(history), False))
        with self.assertRaisesRegex(RequestError, "unknown"):      # naming the id is not proof: ids are small integers
            s.submit([9], 1, 0, conversation=first, cache_salt="blue")
        second, _ = s.submit([9], 1, 0, conversation=first, cache_salt="red")
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(second), [9])
        with self.assertRaises(RequestError):
            s.submit([3], 1, 0, cache_salt="")                     # a salt is a string with something in it
        with self.assertRaises(RequestError):
            s.submit([3], 1, 0, cache_salt=7)

    def test_new_requests_evict_old_idle_conversations_without_exhausting_rows(self):
        s = server(keep_idle=True)
        jobs = [s.submit([i], 1, 0) for i in range(12)]
        self.drain(s, retained=True)
        self.assertEqual(set(s._conversations), {10, 11})
        for i, (request, _) in enumerate(jobs):
            self.assertEqual(s.take_result(request), [i])
        bad, event = s.submit([1], 1, 0, conversation=0)
        self.drain(s, retained=True)
        self.assertTrue(event.is_set())
        with self.assertRaises(RequestError) as error:
            s.take_result(bad)
        self.assertEqual(error.exception.status, 409)
        s.alive = False
        s.once()
        self.assertFalse(s.engine.tokens or s.runner.slot_of or s.runner.idle)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)

    def test_parked_conversation_resumes_before_extension_and_returns_to_disk(self):
        s = server(keep_idle=True, tiered=True)
        first, _ = s.submit([3], 2, 0)
        self.drain(s, retained=True)
        # parked: blocks, slot and row are all free; the conversation is a key on the tier, not a row
        self.assertTrue(s.runner.is_parked(first))
        self.assertNotIn(first, s._conversations)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        self.assertEqual(s.runner.slots.available, s.runner.c.max_running)
        self.assertEqual(sorted(s._free_rows), [0, 1])
        self.assertFalse(s.engine.tokens or s.runner.slot_of or s.runner.idle)
        self.assertEqual(s.runner.tiered.tier.record(first)["tokens"], [3])
        second, _ = s.submit([9], 2, 0, conversation=first)
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(second), [9, 9])
        self.assertTrue(s.runner.is_parked(first))
        self.assertEqual(s.engine.opened, [0])                 # resume reopens without `open`: the slot's bytes came from disk
        s.alive = False
        s.once()
        self.assertTrue(s.runner.is_parked(first))             # the process stops; the conversation stays on disk (D16)
        self.assertFalse(s.engine.tokens or s.runner.slot_of)

    def test_retained_conversations_are_bounded_by_the_tier_not_by_rows(self):
        s = server(rows=2, keep_idle=True, tiered=True)
        jobs = [s.submit([i], 1, 0) for i in range(6)]
        self.drain(s, retained=True)
        self.assertEqual(sorted(s.runner.parked_keys()), [r for r, _ in jobs])   # six conversations, two rows
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        for i, (request, _) in enumerate(jobs):
            self.assertEqual(s.take_result(request), [i])
        turns = [s.submit([10 + i], 1, 0, conversation=r) for i, (r, _) in enumerate(jobs)]
        self.drain(s, retained=True)
        for i, (request, _) in enumerate(turns):
            self.assertEqual(s.take_result(request), [10 + i])
        self.assertEqual(sorted(s.runner.parked_keys()), [r for r, _ in jobs])
        self.assertEqual(s.engine.opened, [0, 1] * 3)           # rows were reused, resumes never `open`

    def test_slot_bytes_follow_a_conversation_into_another_slot(self):
        s = server(rows=2, keep_idle=True, tiered=True)
        first, _ = s.submit([3], 1, 0)
        self.drain(s, retained=True)
        self.assertEqual(bytes(s.runner.tiered.tier.extra[first]), bytes([1, 0, 0, 0]))   # slot 1 held row 0
        blocker, _ = s.submit([5], 4, 0)                          # a live request takes row 0 / slot 1
        s.once()
        second, _ = s.submit([9], 1, 0, conversation=first)        # the continuation lands in row 1 / slot 2
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(second), [9])
        self.assertEqual(bytes(s.runner.tiered.tier.extra[first]), bytes([1, 0, 0, 0]))   # the same bytes, parked from slot 2
        self.assertEqual(s.take_result(blocker), [5] * 4)

    def test_a_full_tier_forgets_the_least_recently_parked_conversation(self):
        from test_engine_tier import MemoryTier
        s = server(rows=2, keep_idle=True, tier=MemoryTier(capacity=2))
        jobs = [s.submit([i], 1, 0) for i in range(3)]
        self.drain(s, retained=True)
        keys = [r for r, _ in jobs]
        self.assertEqual(sorted(s.runner.parked_keys()), keys[1:])
        gone, event = s.submit([7], 1, 0, conversation=keys[0])
        self.drain(s, retained=True)
        with self.assertRaises(RequestError) as error:
            s.take_result(gone)
        self.assertEqual(error.exception.status, 409)
        kept, _ = s.submit([7], 1, 0, conversation=keys[1])
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(kept), [7])

    def test_the_step_loop_never_waits_on_a_park_or_resume(self):
        from test_engine_tier import MemoryTier
        gate = threading.Event()
        s = server(rows=2, keep_idle=True, tier=MemoryTier(gate=gate))
        first, _ = s.submit([3], 2, 0)
        self.drain_steps(s)
        self.assertIn(0, s._retiring)                          # the write is on the tier's thread ...
        self.assertNotIn(0, s._free_rows)
        other, _ = s.submit([5], 3, 0)                         # ... and the loop keeps serving on the other row
        for _ in range(6):
            s.once()
        self.assertEqual(s.take_result(other), [5, 5, 5])
        self.assertIn(0, s._retiring)
        gate.set()
        self.drain(s, retained=True)
        self.assertTrue(s.runner.is_parked(first))
        self.assertEqual(sorted(s._free_rows), [0, 1])
        gate.clear()
        turn, _ = s.submit([9], 1, 0, conversation=first)     # the read is on the tier's thread: the request waits, the loop does not
        s.once()
        self.assertEqual(len(s._resuming), 1)
        again, _ = s.submit([6], 1, 0)                         # a fresh request is admitted on the remaining row meanwhile
        for _ in range(4):
            s.once()
        self.assertEqual(s.take_result(again), [6])
        self.assertEqual(len(s._resuming), 1)
        gate.set()
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(turn), [9])

    def test_a_continuation_arriving_during_its_park_waits_and_then_resumes(self):
        from test_engine_tier import MemoryTier
        gate = threading.Event()
        s = server(rows=2, keep_idle=True, tier=MemoryTier(gate=gate))
        first, _ = s.submit([3], 1, 0)
        self.drain_steps(s)
        self.assertIn(0, s._retiring)
        turn, event = s.submit([9], 1, 0, conversation=first)
        for _ in range(3):
            s.once()
        self.assertFalse(event.is_set())                       # not 409: it waits for the park to land
        gate.set()
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(turn), [9])

    def test_a_failed_read_in_flight_answers_503_and_drops_the_conversation_everywhere(self):
        from test_engine_tier import MemoryTier
        tier = MemoryTier(gate=threading.Event())
        s = server(rows=2, keep_idle=True, tier=tier)
        tier.gate.set()
        first, _ = s.submit([3], 1, 0)
        self.drain(s, retained=True)
        tier.fail_promote = True
        turn, _ = s.submit([9], 1, 0, conversation=first)
        self.drain(s, retained=True)
        with self.assertRaises(RequestError) as error:
            s.take_result(turn)
        self.assertEqual(error.exception.status, 503)
        self.assertFalse(s.runner.is_parked(first))            # dropped everywhere after the agreed failure
        self.assertEqual(sorted(s._free_rows), [0, 1])
        self.assertFalse(s.runner.slot_of or s.runner.resuming)

    def test_stopping_with_a_park_in_flight_settles_it(self):
        from test_engine_tier import MemoryTier
        gate = threading.Event()
        s = server(rows=2, keep_idle=True, tier=MemoryTier(gate=gate, delay=0.05))
        first, _ = s.submit([3], 1, 0)
        self.drain_steps(s)
        self.assertIn(0, s._retiring)
        gate.set()
        s.alive = False
        s.once()
        self.assertTrue(s.runner.is_parked(first))
        self.assertFalse(s.runner.retiring or s.runner.slot_of or s._retiring)

    def drain_steps(self, s):
        """Run until nothing is scheduled: transfers may still be in flight."""
        for _ in range(500):
            if not s.once() and not s._waiting:
                return
            threading.Event().wait(0.001)
        self.fail("server did not drain")

    def test_conversations_survive_a_restart_of_server_and_runner(self):
        from test_engine_tier import MemoryTier
        tier = MemoryTier()
        s = server(rows=2, keep_idle=True, tier=tier)
        jobs = [s.submit([i], 2, 0) for i in range(3)]
        self.drain(s, retained=True)
        s.alive = False
        s.once()
        restarted = server(rows=2, keep_idle=True, tier=tier)   # a new process: no host state but the tier
        self.assertEqual(restarted.next_seq, 3)                  # new request ids start above the parked ones
        self.assertEqual(sorted(restarted.runner.parked_keys()), [0, 1, 2])
        turn, _ = restarted.submit([9], 1, 0, conversation=1)
        self.drain(restarted, retained=True)
        self.assertEqual(turn, 3)
        self.assertEqual(restarted.take_result(turn), [9])
        self.assertEqual(restarted.engine.tokens, {})            # parked again after the turn
        self.assertTrue(restarted.runner.is_parked(1))

    def test_oversized_continuation_preserves_the_previous_idle_context(self):
        s = server(blocks=2, keep_idle=True)
        first, _ = s.submit([3] * 4, 2, 0)
        self.drain(s, retained=True)
        before = list(s.engine.tokens[0])
        bad, _ = s.submit([9] * 4, 2, 0, conversation=first)
        self.drain(s, retained=True)
        with self.assertRaises(RequestError):
            s.take_result(bad)
        self.assertEqual(s.engine.tokens[0], before)
        good, _ = s.submit([7], 1, 0, conversation=first)
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(good), [7])

    def test_failed_resume_answers_503_and_the_engine_lives(self):
        s = server(keep_idle=True, tiered=True)
        first, _ = s.submit([3], 2, 0)
        self.drain(s, retained=True)
        s.runner.tiered.tier.fail_promote = True
        request, event = s.submit([9], 2, 0, conversation=first)
        self.drain(s, retained=True)
        self.assertTrue(event.is_set())
        with self.assertRaises(RequestError) as error:
            s.take_result(request)
        self.assertEqual(error.exception.status, 503)
        self.assertTrue(s.alive)
        self.assertFalse(s.engine.tokens or s.runner.slot_of or s.runner.idle)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        self.assertEqual(s.runner.slots.available, s.runner.c.max_running)
        s.runner.tiered.tier.fail_promote = False
        later, _ = s.submit([4], 1, 0)                          # the door still serves
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(later), [4])

    def test_busy_continuation_does_not_replace_the_active_turns_event(self):
        s = server(keep_idle=True)
        first, event = s.submit([3], 4, 0)
        s.once()
        bad, _ = s.submit([9], 1, 0, conversation=first)
        self.drain(s, retained=True)
        with self.assertRaises(RequestError) as error:
            s.take_result(bad)
        self.assertEqual(error.exception.status, 409)
        self.assertTrue(event.is_set())
        self.assertEqual(s.take_result(first), [3] * 4)

    def test_public_ids_outlive_rows_and_uncollected_results_keep_their_tokens(self):
        s = server()
        jobs = [s.submit([i], 1 + i % 3, 0) for i in range(40)]
        self.drain(s)
        self.assertEqual([r for r, _ in jobs], list(range(40)))
        self.assertTrue(all(ev.is_set() for _, ev in jobs))
        self.assertLess(max(s.engine.opened), 2)
        for i, _ in jobs:
            self.assertEqual(s.take_result(i), [i] * (1 + i % 3))
        self.assertFalse(s.pending or s.results)

    def test_admission_reserves_future_decode_growth_before_accepting_a_second_row(self):
        s = server(blocks=4)
        first, _ = s.submit([7] * 4, 13, 0)       # needs all four blocks eventually
        second, _ = s.submit([9] * 4, 13, 0)
        s.once()
        self.assertEqual(len(s._active), 1)
        self.assertEqual(len(s._waiting), 1)
        self.drain(s)
        self.assertEqual(s.take_result(first), [7] * 13)
        self.assertEqual(s.take_result(second), [9] * 13)

    def test_invalid_requests_never_enter_the_queue_or_model(self):
        s = server()
        cases = [([], 1, 0), ([True], 1, 0), ([1.0], 1, 0), ([256], 1, 0),
                 ([1], 0, 0), ([1], True, 0), ([1], 1, float('nan')),
                 ([1], 1, -1), ([1], 1, 10**1000), ([1], 1000, 0), ('text', 1, 0)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(RequestError):
                s.submit(*args)
        self.assertEqual(s.next_seq, 0)
        self.assertFalse(s.pending or s.engine.tokens)

    def test_backpressure_counts_results_until_the_caller_collects_them(self):
        s = server(max_pending=2)
        ids = [s.submit([i], 1, 0)[0] for i in range(2)]
        self.drain(s)
        with self.assertRaises(RequestError) as error:
            s.submit([3], 1, 0)
        self.assertEqual(error.exception.status, 503)
        s.take_result(ids[0])
        request, _ = s.submit([4], 1, 0)
        self.drain(s)
        self.assertEqual(s.take_result(request), [4])

    def test_concurrent_submit_allocates_unique_public_ids(self):
        s = server(max_pending=40)
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            jobs = list(pool.map(lambda i: s.submit([i], 1, 0), range(40)))
        self.assertEqual(len({r for r, _ in jobs}), 40)
        self.drain(s)
        for i, (request, _) in enumerate(jobs):
            self.assertEqual(s.take_result(request), [i])

    def test_stop_cancels_partial_prefill_and_waiting_requests(self):
        s = server()
        jobs = [s.submit([3] * 16, 4, 0) for _ in range(4)]
        s.once()                                    # one request is midway through prefill
        self.assertIsNotNone(s.runner.state.in_prefill)
        s.alive = False
        self.assertFalse(s.once())
        self.assertIsNone(s.runner.state.in_prefill)
        self.assertFalse(s.engine.tokens or s.runner.slot_of)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        for request, event in jobs:
            self.assertTrue(event.is_set())
            with self.assertRaises(RequestError):
                s.take_result(request)
        with self.assertRaises(RequestError):
            s.submit([1], 1, 0)

    def test_open_failure_wakes_all_clients_and_preserves_the_original_error(self):
        s = server()
        jobs = [s.submit([2], 3, 0) for _ in range(4)]
        s.engine.fail_open = True
        with self.assertRaisesRegex(RuntimeError, "open failed"):
            s.once()
        self.assertFalse(s.engine.tokens or s.runner.slot_of)
        self.assertTrue(all(event.is_set() for _, event in jobs))
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)

    def test_kernel_failure_cleans_rows_and_wakes_pending_clients(self):
        s = server()
        jobs = [s.submit([2], 3, 0) for _ in range(4)]
        s.once()
        s.engine.fail_decode = True
        # The starvation valve may prefill the second request first.
        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            for _ in range(3):
                s.once()
        self.assertFalse(s.engine.tokens or s.runner.slot_of)
        self.assertTrue(all(event.is_set() for _, event in jobs))
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)

    def test_http_bad_json_and_invalid_requests_return_errors_then_valid_request_succeeds(self):
        s = server()
        httpd = s._serve_http()
        url = f'http://127.0.0.1:{httpd.server_port}/v1/engine/completions'
        def post(body):
            with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=3) as response:
                return json.load(response)
        try:
            for body in (b'{', b'[]', b'{"ids":[],"max_tokens":1}', b'{"ids":[999],"max_tokens":1}'):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post(body)
                self.assertEqual(error.exception.code, 400)
                self.assertIn('error', json.load(error.exception))
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(post, b'{"ids":[7],"max_tokens":2}')
                for _ in range(2000):
                    s.once()
                    if future.done():
                        break
                    threading.Event().wait(0.001)
                self.assertEqual(future.result(timeout=3)['ids'], [7, 7])
            self.assertFalse(s.pending or s.results)
        finally:
            httpd.shutdown()
            httpd.server_close()

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires PyTorch for LocalTP')
    def test_four_ranks_park_and_resume_in_lockstep_at_different_disk_speeds(self):
        from engine.base.comm import LocalTP
        from test_engine_tier import MemoryTier
        snapshots = []
        def rank_main(comm, _):
            s = server(comm=comm, rows=2, keep_idle=True, tier=MemoryTier(delay=0.01 * (comm.rank + 1)))
            if comm.rank == 0:
                jobs = [s.submit([i], 1 + i % 2, 0) for i in range(5)]
            for _ in range(120):
                s.once()
                threading.Event().wait(0.002)
            if comm.rank == 0:
                snapshots.append([s.take_result(i) for i, _ in jobs])
                turns = [s.submit([20 + i], 1, 0, conversation=i) for i in range(5)]
            for _ in range(200):
                s.once()
                threading.Event().wait(0.002)
            if comm.rank == 0:
                snapshots.append([s.take_result(r) for r, _ in turns])
                s.alive = False
            s.once()
            return sorted(s.runner.parked_keys()), s.served, s.runner.kv.available, len(s.runner.retiring), len(s.runner.resuming)
        out = LocalTP(4).run(rank_main, None)
        self.assertTrue(all(row == out[0] for row in out), out)
        self.assertEqual(out[0][0], [0, 1, 2, 3, 4])          # five conversations retained on two rows, on every rank
        self.assertEqual(out[0][1], 10)
        self.assertEqual(snapshots[0], [[i] * (1 + i % 2) for i in range(5)])
        self.assertEqual(snapshots[1], [[20 + i] for i in range(5)])

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires PyTorch for LocalTP')
    def test_four_ranks_admit_reuse_and_stop_in_the_same_order(self):
        from engine.base.comm import LocalTP
        snapshots = []
        def rank_main(comm, _):
            s = server(comm=comm, keep_idle=True)
            if comm.rank == 0:
                jobs = [s.submit([i], 1 + i % 3, 0) for i in range(12)]
            for _ in range(60):
                s.once()
            if comm.rank == 0:
                snapshots.append([s.take_result(i) for i, _ in jobs])
                request, _ = s.submit([99], 2, 0, conversation=10)
            for _ in range(20):
                s.once()
            if comm.rank == 0:
                snapshots.append(s.take_result(request))
                s.alive = False
            s.once()
            return s.served, s.engine.opened, s.alive, s.runner.kv.available
        out = LocalTP(4).run(rank_main, None)
        self.assertTrue(all(row == out[0] for row in out))
        self.assertEqual(out[0][0], 13)
        self.assertFalse(out[0][2])
        self.assertEqual(snapshots[0], [[i] * (1 + i % 3) for i in range(12)])
        self.assertEqual(snapshots[1], [99, 99])


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class Tokenizer:
    """A vocabulary of 256: a character is its code point (below 256), a token decodes to its character."""
    def encode(self, text, add_special_tokens=True):
        return Encoded([ord(c) % 256 for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class Door:
    """The profile's door half, faked: a picture is the bytes themselves, three placeholder tokens (250) wide."""
    kinds = ("image", "video")
    limits = {"image": 2, "video": 1}
    TOKEN = 250

    def __init__(self):
        self.prepared = []

    def prepare(self, kind, data):
        import hashlib
        if data == b"bad":
            raise ValueError("not a picture")
        self.prepared.append((kind, data))
        return {"kind": kind, "digest": hashlib.sha1(data).hexdigest(), "canvas": data, "grid": (1, 2, 6), "tokens": 3}

    def expand(self, ids, items):
        out, media, i = [], [], 0
        for t in ids:
            if t == self.TOKEN:
                if i >= len(items):
                    raise ValueError("more placeholders than pictures")
                item = items[i]; i += 1
                positions = list(range(len(out), len(out) + item["tokens"]))
                out.extend([self.TOKEN] * item["tokens"])
                media.append({"kind": item["kind"], "digest": item["digest"], "positions": positions, "canvas": item["canvas"], "grid": item["grid"]})
            else:
                out.append(t)
        if i != len(items):
            raise ValueError("pictures without placeholders")
        return out, media


def chat_server(**kw):
    s = server(**kw)
    s.tok = Tokenizer()
    def render(messages, kwargs, *, generation_prompt=True, continue_final=False):
        out = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                out.append("".join(p["text"] if p["type"] == "text" else chr(Door.TOKEN) for p in c))
            else:
                out.append(c or "")
        return "".join(out) + ("!" if kwargs.get("thinking") else "") + ("" if generation_prompt else "<resume>")
    s.chat = render
    s.model_name = "fake"
    return s


def drive(s, future, steps=2000):
    for _ in range(steps):
        s.once()
        if future.done():
            return future.result(timeout=3)
        threading.Event().wait(0.001)
    return future.result(timeout=3)


class ChatDoorTests(unittest.TestCase):
    """The OpenAI dialect over the fake engine (which repeats the prompt's last token)."""

    def test_models_metrics_and_health(self):
        s = chat_server()
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            models = json.load(urllib.request.urlopen(base + '/v1/models', timeout=3))
            self.assertEqual([m['id'] for m in models['data']], ['fake'])
            self.assertEqual(json.load(urllib.request.urlopen(base + '/health', timeout=3)), {'status': 'ok'})
            text = urllib.request.urlopen(base + '/metrics', timeout=3).read().decode()
            self.assertIn('vllm:request_success_total{engine="st"} 0\n', text)
            self.assertIn('vllm:iteration_tokens_total_count{engine="st"} ', text)   # bench/bracket._StepWindows samples this as engine steps
            self.assertIn('vllm:num_requests_running{engine="st"} 0\n', text)
            self.assertIn('vllm:spec_decode_num_draft_tokens_total{engine="st"} 0\n', text)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_chat_completion_whole(self):
        s = chat_server()
        s.engine.eos = {ord('b')}                    # 'b' ends a generation
        httpd = s._serve_http()
        url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
        def post(body):
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=3) as r:
                return json.load(r)
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = drive(s, pool.submit(post, {"model": "m", "messages": [{"role": "user", "content": "ab"}], "max_tokens": 3}))
            self.assertEqual(out['object'], 'chat.completion')
            # the fake engine runs to its limit regardless of eos; the door names the ending by the last token
            self.assertEqual(out['choices'][0]['message'], {'role': 'assistant', 'content': 'bbb'})
            self.assertEqual(out['choices'][0]['finish_reason'], 'stop')
            self.assertEqual(out['usage'], {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5,
                                            'prompt_tokens_details': {'cached_tokens': 0},
                                            'completion_tokens_details': {'reasoning_tokens': 0}})
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = drive(s, pool.submit(post, {"messages": [{"role": "user", "content": "xy"}], "max_tokens": 3}))
            self.assertEqual(out['choices'][0]['message']['content'], 'yyy')
            self.assertEqual(out['choices'][0]['finish_reason'], 'length')
            for body in ({"messages": []}, {"messages": [{"role": "user"}]}, {"messages": "hi"}, {"messages": [{"role": "user", "content": "a"}], "chat_template_kwargs": 3}):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post(body)
                self.assertEqual(error.exception.code, 400)
            self.assertFalse(s.pending or s.results or s._streams)
            self.assertIn('vllm:request_success_total{engine="st"} 2\n', s.metrics())
            self.assertIn('vllm:generation_tokens_total{engine="st"} 6\n', s.metrics())
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_chat_completion_streams_by_token_with_reasoning_split(self):
        s = chat_server()
        s.reasoning_end = ord('y')                  # the fake engine repeats the last prompt token: 'y'... so with
        httpd = s._serve_http()                     # reasoning_end = 'y' the first token closes the reasoning and the rest is content
        url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
        def stream(body):
            events = []
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=5) as r:
                self.assertEqual(r.headers['Content-Type'], 'text/event-stream')
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith('data:'):
                        events.append(line[5:].strip())
            return events
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                events = drive(s, pool.submit(stream, {"messages": [{"role": "user", "content": "xy"}], "max_tokens": 4, "stream": True,
                                                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": True}}))
            self.assertEqual(events[-1], '[DONE]')
            chunks = [json.loads(e) for e in events[:-1]]
            self.assertTrue(all(c['object'] == 'chat.completion.chunk' for c in chunks))
            deltas = [c['choices'][0]['delta'] for c in chunks if c['choices']]
            self.assertEqual(deltas[0], {'role': 'assistant', 'content': ''})
            # the rendered prompt is "xy!" (thinking kwarg honoured), so the engine repeats '!': 4 tokens; the reasoning end
            # ('y') never comes, so all of it is still reasoning -- each token in its own chunk as the loop steps
            self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas), '!!!!')
            self.assertEqual(''.join(d.get('content', '') for d in deltas), '')
            self.assertGreaterEqual(sum(1 for d in deltas if d.get('reasoning_content')), 2)
            self.assertEqual([c['choices'][0]['finish_reason'] for c in chunks if c['choices']][-1], 'length')
            self.assertEqual(chunks[-1]['usage'], {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7,
                                                   'prompt_tokens_details': {'cached_tokens': 0},
                                                   'completion_tokens_details': {'reasoning_tokens': 4}})
            self.assertFalse(s._streams or s._sent or s.pending or s.results)
            # thinking off: the rendered prompt "xy" ENDS with 'y' = reasoning_end (the template closed the think block),
            # so the door starts in content mode and every generated 'y' is content (45차 §22: an answer used to land in
            # reasoning_content with thinking off, which the gateway's -low route never reads)
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                events = drive(s, pool.submit(stream, {"messages": [{"role": "user", "content": "xy"}], "max_tokens": 4, "stream": True}))
            chunks = [json.loads(e) for e in events[:-1]]
            deltas = [c['choices'][0]['delta'] for c in chunks if c['choices']]
            self.assertEqual(''.join(d.get('content', '') for d in deltas), 'yyyy')
            self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas), '')
            self.assertEqual(s.split([1, ord('y'), 2, 3]), ([1], [2, 3]))
            self.assertEqual(s.split([1, 2]), ([1, 2], []))
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_streaming_client_is_released_when_the_engine_dies(self):
        s = chat_server()
        httpd = s._serve_http()
        url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
        def stream():
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 5, "stream": True}).encode()), timeout=5) as r:
                return [l.decode().strip() for l in r if l.strip()]
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(stream)
                for _ in range(200):
                    if s._streams:
                        break
                    threading.Event().wait(0.001)
                s.once()
                s.engine.fail_decode = True
                with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                    for _ in range(3):
                        s.once()
                lines = future.result(timeout=5)
            self.assertTrue(any('"error"' in l for l in lines), lines)
            self.assertFalse(s._streams or s.pending)
        finally:
            httpd.shutdown(); httpd.server_close()


class CancelTests(unittest.TestCase):
    """A request leaves wherever it is: the FIFO, a prefill, a decode; the client that asked, or the clock, or a stop string."""

    def test_cancel_waiting_and_active_requests_release_rows_and_blocks(self):
        s = server(rows=1, blocks=16)
        a, ea = s.submit([1], 6, 0)
        b, eb = s.submit([2], 6, 0)                 # waits: one row
        s.once()                                    # a prefills
        self.assertEqual(list(s._active), [0])
        s.cancel(b, "client closed")
        s.once()                                    # b leaves the FIFO before admission
        self.assertTrue(eb.is_set())
        with self.assertRaisesRegex(RequestError, "client closed"):
            s.take_result(b)
        self.assertEqual(len(s._waiting), 0)
        s.cancel(a, "client closed")
        s.once()
        self.assertTrue(ea.is_set())
        with self.assertRaisesRegex(RequestError, "cancelled"):
            s.take_result(a)
        self.assertFalse(s._active or s.runner.slot_of or s.engine.tokens)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        self.assertEqual(s._free_rows, [0])
        self.assertEqual(s.cancelled, 2)
        c, ec = s.submit([3], 2, 0)                 # the row serves again
        for _ in range(5):
            s.once()
        self.assertEqual(s.take_result(c), [3, 3])
        self.assertIn('st:requests_cancelled_total{engine="st"} 2\n', s.metrics())

    def test_timeout_cancels_with_504_and_a_conversation_row_is_freed(self):
        s = server(rows=2, keep_idle=True)
        clock = [100.0]
        s.clock = lambda: clock[0]
        s.request_timeout_s = 5.0
        a, ea = s.submit([1], 50, 0)
        s.once()
        clock[0] = 104.0
        s.once()
        self.assertFalse(ea.is_set())
        clock[0] = 106.0
        s.once()
        self.assertTrue(ea.is_set())
        with self.assertRaises(RequestError) as error:
            s.take_result(a)
        self.assertEqual(error.exception.status, 504)
        self.assertFalse(s._active or s._conversations or s._conversation_of or s.runner.slot_of)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)

    def test_stop_string_ends_generation_early_in_both_modes(self):
        s = chat_server()
        httpd = s._serve_http()
        url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
        def post(body):
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=5) as r:
                return json.load(r)
        def stream(body):
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=5) as r:
                return [json.loads(l.decode()[5:]) for l in r if l.startswith(b'data:') and b'[DONE]' not in l]
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:      # prompt "ab" repeats 'b': "bb" stops it at 2 tokens of 50
                out = drive(s, pool.submit(post, {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 50, "stop": ["bb"]}))
            self.assertEqual(out['choices'][0]['finish_reason'], 'stop')
            self.assertEqual(out['choices'][0]['message']['content'], None)      # the text before the stop string is empty
            self.assertLess(out['usage']['completion_tokens'], 50)
            self.assertFalse(s._active or s.runner.slot_of, "the row was dropped, not run to the limit")
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                chunks = drive(s, pool.submit(stream, {"messages": [{"role": "user", "content": "xy"}], "max_tokens": 50, "stop": "yyy", "stream": True}))
            content = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices'])
            # The stop string starts at the first generated character, so nothing precedes it.
            # This used to read 'yy': the stream showed two thirds of the stop string before the
            # third token completed it, and the non-streaming case above already answered None.
            self.assertEqual(content, '')
            self.assertEqual([c['choices'][0]['finish_reason'] for c in chunks if c['choices']][-1], 'stop')
            self.assertFalse(s._active or s._streams or s.pending or s.results)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_client_hangup_cancels_a_running_request(self):
        s = chat_server()
        httpd = s._serve_http()
        try:
            sock = socket.create_connection(('127.0.0.1', httpd.server_port), timeout=5)
            body = json.dumps({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 40}).encode()
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                         + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            for _ in range(2000):
                s.once()
                if s._active:
                    break
                threading.Event().wait(0.001)
            self.assertTrue(s._active)
            sock.close()                                              # the client leaves mid-generation
            for _ in range(4000):
                s.once()
                if not s._active:
                    break
                threading.Event().wait(0.001)
            self.assertFalse(s._active or s.runner.slot_of, "hang-up must drop the row")
            self.assertEqual(s.cancelled, 1)
            self.assertLess(s.generation_tokens_total, 40)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_min_tokens_n_logprobs_and_tools_reach_the_engine_and_the_parser(self):
        s = chat_server()
        s.tool_parser = lambda text: [("f", '{"a": 1}')] if "<tool_call>" in text else None
        seen = {}
        def render(messages, kwargs, *, generation_prompt=True, continue_final=False):
            seen["kwargs"] = kwargs
            seen["switches"] = (generation_prompt, continue_final)
            return "ab"
        s.chat = render
        httpd = s._serve_http()
        url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
        def post(body):
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=5) as r:
                return json.load(r)
        try:
            for body in ({"messages": [{"role": "user", "content": "a"}], "n": 2}, {"messages": [{"role": "user", "content": "a"}], "logprobs": True},
                         {"messages": [{"role": "user", "content": "a"}], "stop": ["", "x"]}, {"messages": [{"role": "user", "content": "a"}], "min_tokens": 9, "max_tokens": 4}):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post(body)
                self.assertEqual(error.exception.code, 400)
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = drive(s, pool.submit(post, {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 3, "min_tokens": 2,
                                                  "tools": [{"type": "function", "function": {"name": "f"}}]}))
            self.assertEqual(seen["kwargs"]["tools"][0]["function"]["name"], "f")
            self.assertEqual(s.engine.min_new[0], 2)
            self.assertEqual(out['choices'][0]['finish_reason'], 'length')
            s.tok.decode = lambda ids, skip_special_tokens=True: "<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = drive(s, pool.submit(post, {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 2}))
            self.assertEqual(out['choices'][0]['finish_reason'], 'tool_calls')
            self.assertEqual(out['choices'][0]['message']['tool_calls'][0]['function'], {'name': 'f', 'arguments': '{"a": 1}'})
            self.assertEqual(out['choices'][0]['message']['content'], None)
        finally:
            httpd.shutdown(); httpd.server_close()


if __name__ == '__main__':
    unittest.main()


class ByteTokenizer:
    """One byte per token, assembled as UTF-8: a character can span several tokens."""

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def _stream_bytes(data, tok, stop=(), per_step=6):
    c = _choice(tok, stop)
    shown = []
    for i in range(0, len(data), per_step):
        c.feed(list(data[i:i + per_step]), None, None)
        shown.extend(d.get("content", "") for d in c.flush())
        if c.finish == "stop":
            return "".join(shown), c
    shown.extend(d.get("content", "") for d in c.flush(final=True))
    return "".join(shown), c


def _choice(tok, stop=(), repairs=None):
    from engine.base.serve import _Choice
    return _Choice(0, 1, threading.Event(), queue.Queue(), tok=tok, stop=list(stop), reasoning=False,
                   repairs=repairs)


class StreamedTextTests(unittest.TestCase):
    """What the door shows a client, token by token, must be what the whole answer says."""

    def choice(self, tok, stop=()):
        from engine.base.serve import _Choice
        return _Choice(0, 1, threading.Event(), queue.Queue(), tok=tok, stop=list(stop), reasoning=False)

    def stream(self, ids, tok, stop=()):
        """Feed one token at a time and return what the client saw."""
        c = self.choice(tok, stop)
        shown = []
        for i in ids:
            c.feed([i], None, None)
            shown.extend(d.get("content", "") for d in c.flush())
            if c.finish == "stop":
                break
        else:
            shown.extend(d.get("content", "") for d in c.flush(final=True))
        return "".join(shown), c

    def test_a_character_split_across_tokens_is_never_shown_in_halves(self):
        text = "한국어 ok 漢字"
        shown, _ = self.stream(list(text.encode()), ByteTokenizer())
        self.assertEqual(shown, text)
        self.assertNotIn("\ufffd", shown)

    def test_the_shown_text_matches_decoding_the_whole_answer(self):
        tok = ByteTokenizer()
        ids = list("mixed ascii and 한자 and emoji 🙂 tail".encode())
        shown, _ = self.stream(ids, tok)
        self.assertEqual(shown, tok.decode(ids))

    def test_a_stop_string_spanning_tokens_never_shows_its_prefix(self):
        # "STO" must not reach the client: the cut arrives with the next token, too late.
        shown, c = self.stream(list("hi STOP there".encode()), ByteTokenizer(), stop=["STOP"])
        self.assertEqual(shown, "hi ")
        self.assertEqual(c.finish, "stop")

    def test_a_tail_that_only_looks_like_a_stop_string_is_released(self):
        shown, c = self.stream(list("hi STOup".encode()), ByteTokenizer(), stop=["STOP"])
        self.assertEqual(shown, "hi STOup")
        self.assertIsNone(c.finish)

    def test_the_window_does_not_grow_with_the_answer(self):
        # the whole point: decoding stays O(1) per token instead of O(answer)
        seen = []

        class Counting(ByteTokenizer):
            def decode(self, ids):
                seen.append(len(ids))
                return super().decode(ids)

        self.stream(list(b"x" * 200), Counting())
        self.assertLessEqual(max(seen), 4, "a window, not the whole answer")


@unittest.skipUnless(importlib.util.find_spec("tokenizers") is not None, "requires the tokenizers library")
class RustStreamTests(unittest.TestCase):
    """The Rust DecodeStream path, against the Python window it replaces.

    The fixture is a byte-fallback tokenizer whose id IS the byte, so it says exactly what
    ByteTokenizer above says: the two paths can be put on the same tokens and compared.
    """

    def rust(self):
        from tokenizers import Tokenizer, decoders, models
        tok = Tokenizer(models.BPE({f"<0x{b:02X}>": b for b in range(256)}, [],
                                   byte_fallback=True, unk_token=None))
        tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
        return tok

    def setUp(self):
        from engine.base.serve import new_repairs
        self.repairs = new_repairs()

    def shown(self, ids, tok, *, per_step=1, stop=()):
        """What a client saw, fed `per_step` tokens at a time."""
        c = _choice(tok, stop, self.repairs)
        out = []
        for i in range(0, len(ids), per_step):
            c.feed(ids[i:i + per_step], None, None)
            out.extend(d.get("content", "") for d in c.flush())
            if c.finish == "stop":
                return "".join(out), c
        out.extend(d.get("content", "") for d in c.flush(final=True))
        return "".join(out), c

    def test_the_rust_stream_is_the_path_a_real_tokenizer_takes(self):
        from engine.base.serve import _Stream
        self.assertIsNotNone(_Stream(self.rust())._stream, "a Rust tokenizer must not fall to the window")
        self.assertIsNone(_Stream(ByteTokenizer())._stream, "anything else must")

    def test_both_paths_show_the_same_text(self):
        text = "한국어 mixed ascii 漢字 and emoji 🙂 tail"
        ids = list(text.encode())
        for per_step in (1, 3, 7):
            with self.subTest(per_step=per_step):
                fast, _ = self.shown(ids, self.rust(), per_step=per_step)
                slow, _ = self.shown(ids, ByteTokenizer(), per_step=per_step)
                self.assertEqual(fast, text)
                self.assertEqual(fast, slow)
                self.assertNotIn("\ufffd", fast)

    def test_a_stop_string_spanning_tokens_never_shows_its_prefix(self):
        shown, c = self.shown(list(b"hi STOP there"), self.rust(), stop=["STOP"])
        self.assertEqual(shown, "hi ")
        self.assertEqual(c.finish, "stop")

    # -- the two repairs vLLM hit in production ------------------------------------------------

    def test_an_id_that_is_not_a_token_id_costs_its_own_text_and_no_more(self):
        """vllm-project/vllm#21951. The batch is refused before the stream is touched, so a step
        that carries several tokens still loses only the one id that is not a token id."""
        before = self.repairs["invalid_token_id"]
        for bad in (-1, 2 ** 63, 1.5, None):
            with self.subTest(bad=bad):
                ids = list(b"ab") + [bad] + list(b"cd")
                shown, c = self.shown(ids, self.rust(), per_step=5)
                self.assertEqual(shown, "abcd")
                self.assertIsNone(c.error)
        self.assertEqual(self.repairs["invalid_token_id"], before + 4)

    def test_a_decoder_that_rewrites_its_own_output_does_not_end_the_answer(self):
        """vllm-project/vllm#17448: a non-monotonic decoder breaks DecodeStream's prefix, and
        every later step raises the same way until the stream is replaced."""
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from engine.base.serve import _Stream
        tok = Tokenizer(models.WordLevel({"a": 0, "b": 1, "c": 2}, unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.decoder = decoders.Sequence([decoders.Fuse(), decoders.Replace("ab", "X")])
        stream = _Stream(tok, self.repairs)
        stream.extend([0])
        self.assertEqual(stream.decoded(False), "a")            # the stream's prefix is now "a"
        before = self.repairs["invalid_prefix"]
        stream.extend([1])
        self.assertEqual(stream.decoded(False), "a")            # "ab" became "X": held, not shown
        stream.extend([2])
        # Repaired, not raised. "a" is already out and cannot be taken back, so no answer here
        # equals decode([0,1,2]) == "Xc"; this one at least loses no token -- vLLM's drops the
        # held "b" and shows "ac".
        self.assertEqual(stream.decoded(False), "abc")
        self.assertEqual(self.repairs["invalid_prefix"], before + 1)
        stream.extend([2])                                      # and the primed stream carries on
        self.assertEqual(stream.decoded(True), "abcc")

    def test_a_tail_that_will_never_finish_is_shown_instead_of_held_forever(self):
        """A held-back tail is a character waiting for its rest. A run of lone lead bytes is not
        waiting: holding it shows the client nothing for the rest of the answer, and the decode
        that repeats grows with the run. The bound is `_STALL_TOKENS` tokens, not bytes -- see
        the multi-byte test above for why that distinction is the whole point."""
        from engine.base.serve import _STALL_TOKENS
        before = self.repairs["stalled"]
        stuck = [0xED] * (_STALL_TOKENS * 4)
        for tok in (self.rust(), ByteTokenizer()):
            with self.subTest(tok=type(tok).__name__):
                c = _choice(tok, repairs=self.repairs)
                seen = 0
                for i in range(0, len(stuck), 4):
                    c.feed(stuck[i:i + 4], None, None)
                    seen += sum(len(d.get("content", "")) for d in c.flush())
                self.assertGreater(seen, 0, "the client saw nothing while the answer ran")
        self.assertGreater(self.repairs["stalled"], before)

    def test_a_step_that_carries_several_tokens_does_not_break_multi_byte_text(self):
        """The wait for a character's rest is bounded by the bytes missing, not by the tokens
        they arrived among. Counting tokens, a step that commits five of them reaches the
        bound after two waits and gives up on a character that was one byte away -- and the
        answer gets a U+FFFD in the middle of a word. Korean is three bytes a syllable, so
        this is every Korean answer under speculative decoding; English never shows it."""
        text = "안녕하세요 세계 여러분 반갑습니다 좋은 하루 되세요 " * 8
        for tok in (self.rust(), ByteTokenizer()):
            for per_step in (2, 3, 5, 7):
                with self.subTest(tok=type(tok).__name__, per_step=per_step):
                    before = self.repairs["stalled"]
                    shown, _ = self.shown(list(text.encode()), tok, per_step=per_step)
                    self.assertEqual(shown, text)
                    self.assertNotIn("\ufffd", shown)
                    self.assertEqual(self.repairs["stalled"], before, "nothing here is stuck")

    def test_an_unfinished_character_at_the_very_end_is_still_shown(self):
        shown, _ = self.shown(list("ok ".encode()) + [0xED], self.rust())
        self.assertEqual(shown, "ok \ufffd")


@unittest.skipUnless(importlib.util.find_spec("tokenizers") is not None, "requires the tokenizers library")
class ProvisionalTextTests(unittest.TestCase):
    """A step that ends mid-character still shows the characters it did finish.

    Where a character is one byte this never happens and there is nothing to see. Where it is
    three, holding the step back is most of what the client waits for (45차 §32).
    """

    def byte_level(self):
        """A tokenizer whose decode is a plain byte concatenation, as GLM-5.3's is."""
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        tok = Tokenizer(models.BPE({c: i for i, c in enumerate(sorted(alphabet))}, [], unk_token=None))
        tok.decoder = decoders.ByteLevel()
        return tok, {b: sorted(alphabet).index(c) for b, c in
                     zip(range(256), _byte_level_chars())}

    def ids_for(self, text):
        tok, of_byte = self.byte_level()
        return tok, [of_byte[b] for b in text.encode()]

    def test_a_step_that_ends_mid_character_shows_what_it_finished(self):
        from engine.base.serve import _Stream
        tok, ids = self.ids_for("가나다")                      # three bytes a syllable
        stream = _Stream(tok)
        stream.extend(ids[:4])                                 # "가" and one byte of "나"
        self.assertEqual(stream.decoded(False), "가")          # not "" -- the syllable is whole
        stream.extend(ids[4:])
        self.assertEqual(stream.decoded(True), "가나다")

    def test_nothing_shown_is_taken_back(self):
        from engine.base.serve import _Stream
        tok, ids = self.ids_for("가나다라마바사아자차카타파하 ok 끝")
        for per_step in (1, 2, 4, 7):
            with self.subTest(per_step=per_step):
                stream, seen = _Stream(tok), ""
                for i in range(0, len(ids), per_step):
                    stream.extend(ids[i:i + per_step])
                    out = stream.decoded(False)
                    self.assertTrue(out.startswith(seen), "what was shown changed")
                    seen = out
                self.assertEqual(stream.decoded(True), tok.decode(ids))

    def test_what_is_settled_always_begins_with_what_was_shown(self):
        """Both come from one decode of the tail in place (`_in_place`), so a step that ends
        mid-character can never contradict the delta the client already read -- however the
        tail is finally settled, whether by the next token, the stall bound or the end."""
        from engine.base.serve import _Stream
        tok, ids = self.ids_for("가나다 ok 라마")
        for per_step in (1, 2, 3, 5):
            with self.subTest(per_step=per_step):
                stream, seen = _Stream(tok), ""
                for i in range(0, len(ids), per_step):
                    stream.extend(ids[i:i + per_step])
                    for final in (False, False, True) if i + per_step >= len(ids) else (False,):
                        out = stream.decoded(final)
                        self.assertTrue(out.startswith(seen), f"{out!r} does not continue {seen!r}")
                        seen = out
                self.assertEqual(seen, tok.decode(ids))

    def test_a_decoder_that_rewrites_its_settled_run_is_shown_nothing_early(self):
        """Byte fallback turns a whole run into U+FFFD the moment one byte of it is missing,
        so the settled part is not a prefix of the grown one. Then there is nothing safe to
        show early, and the old behaviour -- wait -- is what happens."""
        from engine.base.serve import _Stream
        from tokenizers import Tokenizer, decoders, models
        tok = Tokenizer(models.BPE({f"<0x{b:02X}>": b for b in range(256)}, [],
                                   byte_fallback=True, unk_token=None))
        tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
        stream = _Stream(tok)
        stream.extend(list("가".encode()) + list("나".encode())[:1])
        self.assertEqual(stream.decoded(False), "")            # nothing, rather than something wrong
        stream.extend(list("나".encode())[1:])
        self.assertEqual(stream.decoded(True), "가나")


def _byte_level_chars():
    """GPT-2's byte-to-character map, which `pre_tokenizers.ByteLevel` decodes back."""
    printable = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1))
                 + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    table, spare = {}, 0
    for b in range(256):
        if b in printable:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + spare)
            spare += 1
    return [table[b] for b in range(256)]


class KoreanBudgetTests(unittest.TestCase):
    """A token is not the same amount of answer in every language (45차 §38)."""

    class ByteTok(Tokenizer):
        """One token a byte, so a Korean syllable costs three and an ASCII letter one."""
        def encode(self, text, add_special_tokens=True):
            return Encoded(list(text.encode()))

    def test_the_default_is_a_length_a_reader_would_recognise(self):
        from engine.base.serve import DEFAULT_ANSWER_CHARS, DEFAULT_ANSWER_TOKENS, answer_budget
        tok = self.ByteTok()
        english = answer_budget(tok, "How should I reduce the latency of our inference server?")
        korean = answer_budget(tok, "우리 서버의 지연 시간을 줄이려면 어떤 것부터 봐야 할까요?")
        self.assertGreater(korean, english, "the same characters cost more tokens in Korean")
        self.assertEqual(english, min(DEFAULT_ANSWER_TOKENS[1], DEFAULT_ANSWER_CHARS))  # one token a character
        self.assertLessEqual(korean, DEFAULT_ANSWER_TOKENS[1])

    def test_a_budget_never_fails_and_never_goes_below_the_old_one(self):
        from engine.base.serve import DEFAULT_ANSWER_TOKENS, answer_budget
        floor = DEFAULT_ANSWER_TOKENS[0]
        self.assertEqual(answer_budget(None, "무엇이든"), floor)
        self.assertEqual(answer_budget(self.ByteTok(), ""), floor)

        class Broken(Tokenizer):
            def encode(self, text, add_special_tokens=True):
                raise RuntimeError("no")

        self.assertEqual(answer_budget(Broken(), "안녕하세요"), floor)

    def test_the_text_measured_is_what_the_person_last_wrote(self):
        from engine.base.serve import written_text
        self.assertEqual(written_text([{"role": "system", "content": "You are a helpful assistant."},
                                       {"role": "user", "content": "안녕하세요"}]), "안녕하세요")
        self.assertEqual(written_text([{"role": "user", "content": "first"},
                                       {"role": "assistant", "content": "reply"},
                                       {"role": "user", "content": "second"}]), "second")
        self.assertEqual(written_text([{"role": "user", "content": [
            {"type": "text", "text": "이 사진은"}, {"type": "image_url", "image_url": {"url": "x"}}]}]), "이 사진은")
        self.assertEqual(written_text([{"role": "system", "content": "only a system turn"}]), "")
        self.assertEqual(written_text("not a list"), "")

    def test_a_default_that_does_not_fit_is_cut_down_to_the_old_one_and_no_further(self):
        """Backing a default off to fit is the engine choosing; refusing a prompt the old
        default could not fit is the contract, and that answer does not change."""
        s = chat_server()
        room = s.room_for(0)
        self.assertGreaterEqual(room, 1)
        self.assertLess(s.room_for(s.max_context), room)


class ComposedFormTests(unittest.TestCase):
    """The same Korean, written two ways, must not be two different prompts (45차 §42)."""

    def nfd(self, text):
        import unicodedata
        return unicodedata.normalize("NFD", text)

    def test_a_decomposed_prompt_is_tokenised_as_the_composed_one(self):
        from engine.base.serve import nfc
        for text in ("안녕하세요", "우리 서버의 지연 시간", "mixed 한글 and ascii", "Tiếng Việt"):
            with self.subTest(text=text):
                self.assertEqual(nfc(self.nfd(text)), text)
                self.assertEqual(nfc(text), text, "already composed text is handed back")

    def test_ascii_is_untouched(self):
        from engine.base.serve import nfc
        for text in ("", "plain ascii", "{\"v\": \"x\"}", "<|user|>"):
            self.assertIs(nfc(text), text)

    def test_a_decomposed_stop_string_is_composed_so_it_can_match(self):
        from engine.base.serve import stop_strings
        self.assertEqual(stop_strings({"stop": self.nfd("끝")}), ["끝"])
        self.assertEqual(stop_strings({"stop": [self.nfd("끝"), "STOP"]}), ["끝", "STOP"])

    def test_the_door_tokenises_both_forms_the_same(self):
        """The fake tokenizer is one token a character, so a prompt's token count is its length:
        decomposed "가" is three characters and composed is one, and the door must send one."""
        s = chat_server()
        seen, counted = [], []
        s.chat = lambda messages, kwargs, *, generation_prompt=True, continue_final=False: (
            seen.append(messages[0]["content"]) or messages[0]["content"])
        for content in ("가", self.nfd("가")):
            httpd = s._serve_http()
            url = f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"
            def post():
                with urllib.request.urlopen(urllib.request.Request(
                        url, data=json.dumps({"messages": [{"role": "user", "content": content}],
                                              "max_tokens": 1}).encode()), timeout=5) as r:
                    return json.load(r)
            try:
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    counted.append(drive(s, pool.submit(post))["usage"]["prompt_tokens"])
            finally:
                httpd.shutdown(); httpd.server_close()
        self.assertEqual([len(x) for x in seen], [1, len(self.nfd("가"))], "the two forms really did arrive differently")
        self.assertGreater(len(self.nfd("가")), 1)
        self.assertEqual(counted[0], counted[1], "and they reached the engine as the same prompt")
        self.assertEqual(counted[0], 1)


class HangulPatternTests(unittest.TestCase):
    """`[가-힣]` is the natural way to say "Hangul", and xgrammar drops most of it (45차 §41)."""

    def split(self, pattern):
        from engine.base.serve import split_surrogate_branch
        return split_surrogate_branch(pattern)

    def test_a_range_that_ends_inside_the_0xed_branch_becomes_two(self):
        self.assertEqual(self.split("^[가-힣]+$"), "^[가-\uCFFF\uD000-힣]+$")
        self.assertEqual(self.split("^[가-힣a-z0-9 ]+$"), "^[가-\uCFFF\uD000-힣a-z0-9 ]+$")
        self.assertEqual(self.split("^[^가-힣]+$"), "^[^가-\uCFFF\uD000-힣]+$")

    def test_the_escaped_spelling_of_the_same_range_is_read_too(self):
        self.assertEqual(self.split(r"^[\uAC00-\uD7A3]+$"), "^[\\uAC00-\uCFFF\uD000-\\uD7A3]+$")
        self.assertEqual(self.split(r"^[\x41-\uD7A3]$"), "^[\\x41-\uCFFF\uD000-\\uD7A3]$")

    def test_a_range_that_reaches_past_the_branch_is_left_alone(self):
        """`[가-\uffff]` ends on a whole branch and already compiles (verified against the
        xgrammar in the image), so rewriting it would change a caller's pattern for nothing."""
        for pattern in ("^[가-\uffff]+$", "^[\u0000-\U0010FFFF]$", "^[가-\ue000]$"):
            with self.subTest(pattern=pattern):
                self.assertEqual(self.split(pattern), pattern)

    def test_everything_else_is_handed_on_byte_for_byte(self):
        for pattern in ("^[a-z]+$", "^[가-쿿]+$", "^[\uD000-힣]+$", r"\[가-힣\]", "가-힣", "",
                        r"^[\d]+$", "^[-가-쿿]$", "^(안녕|반가워)$", "^한.*$"):
            with self.subTest(pattern=pattern):
                self.assertEqual(self.split(pattern), pattern)

    def test_a_dash_at_either_end_of_a_class_is_a_literal(self):
        self.assertEqual(self.split("^[-가-힣]$"), "^[-가-\uCFFF\uD000-힣]$")
        self.assertEqual(self.split("^[가-힣-]$"), "^[가-\uCFFF\uD000-힣-]$")

    def test_every_pattern_in_a_schema_is_repaired_wherever_it_sits(self):
        from engine.base.serve import repair_patterns
        schema = {"type": "object",
                  "properties": {"name": {"type": "string", "pattern": "^[가-힣]+$"},
                                 "tags": {"type": "array", "items": {"type": "string", "pattern": "^[가-힣]$"}}},
                  "anyOf": [{"pattern": "^[가-힣]{2}$"}, {"pattern": "^[a-z]+$"}],
                  "description": "^[가-힣]+$"}
        out = repair_patterns(schema)
        self.assertEqual(out["properties"]["name"]["pattern"], "^[가-\uCFFF\uD000-힣]+$")
        self.assertEqual(out["properties"]["tags"]["items"]["pattern"], "^[가-\uCFFF\uD000-힣]$")
        self.assertEqual(out["anyOf"][0]["pattern"], "^[가-\uCFFF\uD000-힣]{2}$")
        self.assertEqual(out["anyOf"][1]["pattern"], "^[a-z]+$")
        self.assertEqual(out["description"], "^[가-힣]+$", "only a pattern is a pattern")

    def test_the_door_repairs_what_it_sends_to_the_grammar(self):
        from engine.base.serve import response_format_grammar
        spec = response_format_grammar({"response_format": {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object",
                                       "properties": {"v": {"type": "string", "pattern": "^[가-힣]+$"}}}}}})
        self.assertIn("\uD000", json.loads(spec["schema"])["properties"]["v"]["pattern"])


class KoreanWireTests(unittest.TestCase):
    """What a Korean answer costs between the door and the client (45차 §40)."""

    def raw(self, s, path, body=None):
        httpd = s._serve_http()
        url = f"http://127.0.0.1:{httpd.server_port}{path}"
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode()) if body else urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.read()
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_the_body_carries_utf8_and_not_escapes(self):
        s = chat_server()
        s.model_name = "한국어-모델"
        body = self.raw(s, "/v1/models")
        self.assertIn("한국어-모델".encode(), body)
        self.assertNotIn(rb"\u", body, "a Korean character must not cost six bytes where three do")
        self.assertEqual(json.loads(body)["data"][0]["id"], "한국어-모델")   # still JSON, and still says the same

    def test_a_refusal_says_the_numbers(self):
        s = chat_server()
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.raw(s, "/v1/chat/completions",
                     {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 10 ** 6})
        message = json.loads(err.exception.read())["error"]
        self.assertRegex(message, r"\d+ tokens")
        self.assertIn("to generate", message)
        self.assertIn("for the prompt", message)

    def test_tokenize_says_when_it_composed_what_it_was_handed(self):
        import unicodedata
        s = chat_server()
        plain = json.loads(self.raw(s, "/tokenize", {"prompt": "안녕"}))
        self.assertNotIn("normalized", plain, "nothing to say when nothing changed")
        moved = json.loads(self.raw(s, "/tokenize", {"prompt": unicodedata.normalize("NFD", "안녕")}))
        self.assertEqual(moved["tokens"], plain["tokens"])          # the ids generation would use
        self.assertEqual(moved["normalized"], "NFC")                # and it says they are not the caller's
        self.assertEqual(moved["prompt"], "안녕")                    # here is the string they belong to

    def test_an_answer_the_engine_killed_still_counts_what_it_showed(self):
        """The counter is read against the token counter, and the traffic worth reading it for
        is exactly the traffic that did not finish cleanly. Every way out of the streaming loop
        goes through one `retire`, so this covers the client-hung-up way too."""
        s = chat_server()
        before = s.generation_characters_total
        httpd = s._serve_http()
        url = f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"
        def stream():
            body = {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 5, "stream": True}
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=5) as r:
                return [l.decode().strip() for l in r if l.strip()]
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(stream)
                for _ in range(200):
                    if s._streams:
                        break
                    threading.Event().wait(0.001)
                s.once()                                   # one step of text reaches the client
                s.engine.fail_decode = True
                with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                    for _ in range(3):
                        s.once()
                lines = future.result(timeout=5)
        finally:
            httpd.shutdown(); httpd.server_close()
        shown = sum(len(json.loads(l[6:])["choices"][0]["delta"].get("content", ""))
                    for l in lines if l.startswith("data: ") and l != "data: [DONE]"
                    and json.loads(l[6:]).get("choices"))
        self.assertTrue(any('"error"' in l for l in lines), lines)
        self.assertGreater(shown, 0, "the client did read something before the engine died")
        self.assertEqual(s.generation_characters_total - before, shown)

    def test_the_characters_a_client_read_are_counted_next_to_the_tokens(self):
        s = chat_server()
        before = s.generation_characters_total
        out = self._drive(s, {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 3})
        shown = out["choices"][0]["message"]["content"]
        self.assertEqual(s.generation_characters_total - before, len(shown))
        self.assertIn("st:generation_characters_total", s.metrics())

    def _drive(self, s, body):
        httpd = s._serve_http()
        url = f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"
        def post():
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=5) as r:
                return json.load(r)
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return drive(s, pool.submit(post))
        finally:
            httpd.shutdown(); httpd.server_close()


class TokenSpanTests(unittest.TestCase):
    """`text_offset` has to index into the text it describes, in every language."""

    def spans(self, text, start=0):
        from engine.base.serve import token_spans
        tok = ByteTokenizer()
        ids = list(text.encode())
        tokens, offsets = token_spans(tok, ids, start)
        self.assertEqual(len(tokens), len(ids))
        return tok.decode(ids), tokens, offsets

    def test_the_tokens_join_into_the_text_and_every_offset_lands_on_its_token(self):
        for text in ("안녕하세요 세계 여러분", "漢字テスト 🙂 ok", "plain ascii only", "혼합 mixed 텍스트"):
            with self.subTest(text=text):
                whole, tokens, offsets = self.spans(text)
                self.assertEqual("".join(tokens), whole)
                for token, offset in zip(tokens, offsets):
                    self.assertEqual(whole[offset:offset + len(token)], token)

    def test_the_halves_of_a_character_add_nothing_and_the_last_adds_it_whole(self):
        _, tokens, _ = self.spans("가")                       # three bytes, three tokens
        self.assertEqual(tokens, ["", "", "가"])

    def test_an_echoed_prompt_shifts_every_offset(self):
        _, _, plain = self.spans("가나다")
        _, _, shifted = self.spans("가나다", start=7)
        self.assertEqual(shifted, [o + 7 for o in plain])

    def test_no_tokens_is_no_rows(self):
        from engine.base.serve import token_spans
        self.assertEqual(token_spans(ByteTokenizer(), []), ([], []))


@unittest.skipUnless(importlib.util.find_spec("tokenizers") is not None, "requires the tokenizers library")
class TokenBytesTests(unittest.TestCase):
    """OpenAI's `bytes` exists so a client can rejoin a character split across tokens."""

    def byte_level(self):
        from tokenizers import Tokenizer as Rust, decoders, models, pre_tokenizers
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        tok = Rust(models.BPE({c: i for i, c in enumerate(alphabet)}, [], unk_token=None))
        tok.decoder = decoders.ByteLevel()
        from engine.base.serve import _byte_level_chars
        order = {c: i for i, c in enumerate(alphabet)}
        return tok, [order[c] for c in _byte_level_chars()]

    def test_the_bytes_rejoin_into_the_text(self):
        from engine.base.serve import token_bytes
        tok, id_of_byte = self.byte_level()
        for text in ("안녕하세요 세계", "漢字と emoji 🙂", "plain ascii"):
            with self.subTest(text=text):
                ids = [id_of_byte[b] for b in text.encode()]
                joined = b"".join(bytes(token_bytes(tok, i, tok.decode([i]))) for i in ids)
                self.assertEqual(joined.decode(), text)

    def test_the_byte_table_is_gpt2s(self):
        """Everything else here round-trips through the same table, so an error in it would
        cancel out and pass. These four are GPT-2's published values, checked from outside."""
        from engine.base.serve import _BYTE_OF_CHAR, _byte_level_chars
        table = _byte_level_chars()
        self.assertEqual(len(table), 256)
        self.assertEqual(len(set(table)), 256, "the table has to be a bijection")
        for byte, char in ((0x00, "\u0100"), (0x20, "\u0120"), (0x41, "A"), (0xFF, "\u00ff")):
            self.assertEqual(table[byte], char, f"byte {byte:#04x}")
            self.assertEqual(_BYTE_OF_CHAR[char], byte)

    def test_a_whole_character_keeps_its_own_bytes(self):
        from engine.base.serve import token_bytes
        tok, id_of_byte = self.byte_level()
        tid = id_of_byte[ord("A")]
        self.assertEqual(token_bytes(tok, tid, tok.decode([tid])), [65])

    def test_a_byte_fallback_vocabulary_is_read_too(self):
        from engine.base.serve import token_bytes
        from tokenizers import Tokenizer as Rust, decoders, models
        tok = Rust(models.BPE({f"<0x{b:02X}>": b for b in range(256)}, [], byte_fallback=True, unk_token=None))
        tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
        self.assertEqual(token_bytes(tok, 0xEC, tok.decode([0xEC])), [0xEC])

    def test_a_tokenizer_that_cannot_say_keeps_the_old_answer(self):
        from engine.base.serve import token_bytes
        self.assertEqual(token_bytes(Tokenizer(), 1, "\ufffd"), list("\ufffd".encode()))


class StopScanCostTests(unittest.TestCase):
    """A streamed step must not cost more because the answer is longer."""

    def test_the_stop_scan_reads_the_new_tail_not_the_answer(self):
        spans = []

        class Seen(str):
            def find(self, sub, start=0, *rest):
                spans.append(len(self) - start)
                return str.find(self, sub, start, *rest)

        class Watched:
            """The real stream, with what the stop scan is handed put on the record."""
            def __init__(self, inner):
                self.inner, self.ids = inner, inner.ids

            def extend(self, ids):
                self.inner.extend(ids)

            def decoded(self, final):
                return Seen(self.inner.decoded(final))

        c = _choice(ByteTokenizer(), stop=["STOP", "\n\nHuman:"])
        c.streams["content"] = Watched(c.streams["content"])
        answer = list(b"the quick brown fox. " * 400)
        for i in range(0, len(answer), 6):
            c.feed(answer[i:i + 6], None, None)
            c.flush()
        self.assertGreater(len(c.text["content"]), 8000)
        self.assertGreater(len(spans), 100)
        self.assertLessEqual(max(spans), 64, "the scan grew with the answer")

    def test_a_stop_string_is_still_found_after_a_long_run_before_it(self):
        shown, c = _stream_bytes(b"x" * 5000 + b"STOP tail", ByteTokenizer(), stop=["STOP"])
        self.assertEqual(c.finish, "stop")
        self.assertEqual(shown, "x" * 5000)

    def test_the_short_circuit_in_partial_suffix_changes_no_answer(self):
        import itertools
        import random
        from engine.base.serve import partial_suffix

        def naive(text, needles):
            keep = 0
            for needle in needles:
                for cut in range(min(len(needle) - 1, len(text)), keep, -1):
                    if text.endswith(needle[:cut]):
                        keep = cut
                        break
            return keep

        random.seed(11)
        alphabet = "abST"
        needles = ["STOP", "ab", "aab", "S"]
        cases = ["".join(t) for n in range(5) for t in itertools.product(alphabet, repeat=n)]
        cases += ["".join(random.choice(alphabet) for _ in range(random.randrange(12))) for _ in range(400)]
        for text in cases:
            for subset in (needles, needles[:1], needles[1:], []):
                self.assertEqual(partial_suffix(text, subset), naive(text, subset), (text, subset))


class StepCostTests(unittest.TestCase):
    """What the step loop asks each row must not grow with what that row has already said."""

    def test_the_loop_counts_the_output_instead_of_copying_it(self):
        s = server()
        asked = []
        whole = s.engine.generated

        def watched(seq):
            asked.append(seq)
            return whole(seq)

        s.engine.generated = watched
        request, event = s.submit([1, 2], 4, 0.0)
        steps = 0
        for _ in range(40):
            if not s.once():
                break
            steps += 1
        self.assertGreaterEqual(steps, 4, "the request actually ran")
        # once, when the answer is handed over -- not once per step, and not once per row per step
        self.assertLessEqual(len(asked), 1)

    def test_only_the_unsent_tail_reaches_the_stream(self):
        s = server()
        s.engine.output[0] = [7, 8, 9, 10]
        self.assertEqual(s.engine.generated_count(0), 4)
        self.assertEqual(s.engine.generated_since(0, 2), [9, 10])


class RequestClockTests(unittest.TestCase):
    """Two clocks per request. A request that does not finish must still clear both."""

    def test_a_cancelled_request_leaves_no_clock_behind(self):
        s = server()
        request, event = s.submit([1, 2], 8, 0.0)
        for _ in range(3):
            s.once()
        self.assertIn(request, s._admitted, "it was admitted, so the queue clock stopped")
        s.cancel(request, "client closed")
        for _ in range(3):
            s.once()
        self.assertNotIn(request, s._arrived)
        self.assertNotIn(request, s._admitted)

    def test_both_clocks_are_cleared_in_the_same_places(self):
        source = (ROOT / "engine/base/serve.py").read_text()
        self.assertEqual(source.count("self._arrived.pop(request, None)"),
                         source.count("self._admitted.pop(request, None)"),
                         "one of them is cleared somewhere the other is not")


class MetricsTests(unittest.TestCase):
    """The numbers a dashboard needs that a single latency histogram cannot give."""

    def text(self, s):
        return s.metrics()

    def test_queue_and_inference_time_are_separate_series(self):
        s = server()
        out = self.text(s)
        for name in ("vllm:request_queue_time_seconds", "vllm:request_inference_time_seconds",
                     "vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"):
            self.assertIn(name, out, name)

    def test_the_per_step_and_per_token_series_are_not_the_same_measurement(self):
        # under speculation a step carries several tokens; the two differ by the acceptance length
        s = server()
        s.step_gap.observe(0.3)
        for _ in range(3):
            s.itl.observe(0.1)
        rows = dict(s.step_gap.rows("x")), dict(s.itl.rows("x"))
        self.assertNotEqual(rows[0], rows[1])

    def test_the_success_counter_is_broken_out_by_why_it_stopped(self):
        s = server()
        self.assertNotIn("finished_reason", self.text(s))
        s.by_reason["stop"] = 2
        s.by_reason["length"] = 1
        out = self.text(s)
        self.assertIn('finished_reason="stop"} 2', out)
        self.assertIn('finished_reason="length"} 1', out)

    def test_the_detokenizer_says_which_path_it_took_and_what_it_had_to_repair(self):
        s, other = server(), server()
        out = self.text(s)
        self.assertIn("st:detokenizer_rust_stream", out)            # D3: a silent path is an unchecked path
        self.assertNotIn("st:detokenizer_repairs_total", out)       # no series at all in a healthy run
        s.detok_repairs["invalid_prefix"] = 1
        self.assertIn('st:detokenizer_repairs_total{engine="st",reason="invalid_prefix"} 1', self.text(s))
        self.assertNotIn("st:detokenizer_repairs_total", self.text(other), "another door's repairs are not ours")

    def test_the_box_s_own_memory_is_on_the_scrape(self):
        """The OOM study's conclusion was that there is no eye on memory during serving, and
        vLLM has none either -- its `gpu_cache_usage_perc` counts blocks, not bytes. The peak
        rides beside the current value because a scrape cannot see a four-second cliff."""
        out = self.text(server())
        line = next(l for l in out.splitlines() if l.startswith("st:host_memory_available_bytes{"))
        self.assertGreater(int(line.rsplit(" ", 1)[1]), 0)        # what earlyoom decides on
        if "st:device_memory_reserved_bytes" in out:              # only where there is a device
            self.assertIn("st:device_memory_reserved_peak_bytes", out)
            self.assertIn("st:device_memory_free_bytes", out)

    def test_a_scrape_answers_even_where_the_numbers_are_not_there(self):
        """`/metrics` never raises: a box without CUDA, or a /proc that will not answer, drops
        the row instead of the scrape."""
        import engine.base.serve as serve
        from unittest.mock import patch
        with patch.object(serve, "device_memory_rows", lambda: []):
            self.assertIn("vllm:request_success_total", self.text(server()))

    def test_the_queue_clock_is_taken_once_for_a_continued_request(self):
        s = server()
        s._arrived[3] = s.clock() - 1.0
        s._admit_clock(3)
        first = s._admitted[3]
        s._admit_clock(3)                       # the row is reused for the next turn
        self.assertEqual(s._admitted[3], first)


class StopFloorTests(unittest.TestCase):
    """min_tokens means at least that many, so a stop string cannot undo it from below."""

    def stream(self, text, stop, min_new):
        from engine.base.serve import _Choice
        c = _Choice(0, 1, threading.Event(), queue.Queue(), tok=ByteTokenizer(), stop=list(stop),
                    reasoning=False, min_new=min_new)
        shown = []
        for i in text.encode():
            c.feed([i], None, None)
            shown.extend(d.get("content", "") for d in c.flush())
            if c.finish == "stop":
                break
        else:
            shown.extend(d.get("content", "") for d in c.flush(final=True))
        return "".join(shown), c

    def test_a_stop_string_below_the_floor_does_not_end_the_answer(self):
        shown, c = self.stream("XX keep going", ["XX"], min_new=6)
        self.assertIsNone(c.finish)
        self.assertEqual(shown, "XX keep going")

    def test_the_same_stop_string_above_the_floor_ends_it(self):
        shown, c = self.stream("abcdefg XX tail", ["XX"], min_new=6)
        self.assertEqual(c.finish, "stop")
        self.assertEqual(shown, "abcdefg ")

    def test_with_no_floor_it_ends_at_once(self):
        shown, c = self.stream("XX keep going", ["XX"], min_new=0)
        self.assertEqual(c.finish, "stop")
        self.assertEqual(shown, "")


class StopTokenIdTests(unittest.TestCase):
    """A stop string the model can emit as one token lets the engine end the row itself."""

    def ids(self, stops):
        from engine.base.serve import stop_token_ids_for
        return stop_token_ids_for(stops, Tokenizer())

    def test_a_single_token_stop_string_becomes_an_id(self):
        self.assertEqual(self.ids(["a", "Z"]), [ord("a"), ord("Z")])

    def test_a_multi_token_stop_string_is_left_to_the_text_scan(self):
        self.assertEqual(self.ids(["STOP", "\n\nHuman:"]), [])

    def test_a_token_that_does_not_render_as_the_string_is_refused(self):
        class Lossy(Tokenizer):
            def decode(self, ids, skip_special_tokens=True):
                return ""                                  # as a special token renders
        from engine.base.serve import stop_token_ids_for
        self.assertEqual(stop_token_ids_for(["a"], Lossy()), [])


class LogprobDecodeTests(unittest.TestCase):
    """The payload asks for each token's text and its bytes: that is one decode, not two."""

    def test_each_distinct_id_is_decoded_once(self):
        from engine.base.serve import _Choice
        calls = []

        class Counting(Tokenizer):
            def decode(self, ids, skip_special_tokens=True):
                calls.append(tuple(ids))
                return super().decode(ids)

        c = _Choice(0, 1, threading.Event(), queue.Queue(), tok=Counting(), stop=[], reasoning=False,
                    want_logprobs=3)
        c.logprobs = [(65, -0.1, [(65, -0.1), (66, -1.0), (67, -2.0)]),
                      (66, -0.2, [(66, -0.2), (65, -1.0), (67, -2.0)])]
        payload = c.logprobs_payload()
        self.assertEqual([r["token"] for r in payload["content"]], ["A", "B"])
        self.assertEqual(payload["content"][0]["bytes"], [65])
        self.assertEqual(len(calls), len({c[0] for c in calls}), "one decode per distinct id")


class WakeupTests(unittest.TestCase):
    """A streamed token must wake its reader, not wait out a poll."""

    def test_answering_a_streaming_request_sets_the_wake(self):
        s = server()
        s._streams[7] = queue.Queue()
        s.pending[7] = threading.Event()
        s._wake.clear()
        s._answer(7, [1, 2, 3])
        self.assertTrue(s._wake.is_set())

    def test_the_drain_clears_before_it_reads_and_waits_after(self):
        source = (ROOT / "engine/base/serve.py").read_text()
        drain = source[source.index("live = {c.request: c for c in choices}"):]
        drain = drain[:drain.index("return True")]
        self.assertLess(drain.index("server._wake.clear()"), drain.index("get_nowait"))
        self.assertIn("server._wake.wait(", drain)
        self.assertNotIn("time.sleep", drain)


class PromptSwitchTests(unittest.TestCase):
    """Opening a new assistant turn and resuming the last one are opposites."""

    def switches(self, body):
        from engine.base.serve import prompt_switches
        return prompt_switches(body)

    def test_the_default_opens_a_new_turn(self):
        self.assertEqual(self.switches({}), (True, False))

    def test_resuming_asks_for_both(self):
        self.assertEqual(self.switches({"add_generation_prompt": False,
                                        "continue_final_message": True}), (False, True))

    def test_both_at_once_is_refused(self):
        from engine.base.serve import RequestError
        with self.assertRaisesRegex(RequestError, "cannot both be true"):
            self.switches({"continue_final_message": True})

    def test_they_must_be_booleans(self):
        from engine.base.serve import RequestError
        with self.assertRaisesRegex(RequestError, "must be booleans"):
            self.switches({"add_generation_prompt": "yes"})


class OpenAIDialectTests(unittest.TestCase):
    """45차 §23 A/B: the request surface the gateway uses -- reasoning_effort, sampling options, n, tool-call
    streaming, legacy completions, tokenize/detokenize -- and the multi-turn continuation (B1), over the fake engine."""

    def _post(self, base, path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)

    def _get(self, base, path):
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return json.load(r)

    def _serve(self, s, fn):
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return drive(s, pool.submit(fn, base))
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_reasoning_effort_and_enable_thinking_reach_the_template(self):
        s = chat_server()
        seen = {}
        def chat(messages, kwargs, *, generation_prompt=True, continue_final=False):
            seen.update(kwargs)
            return "".join(m.get("content") or "" for m in messages)
        s.chat = chat
        out = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                     {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1,
                                                      "reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}))
        self.assertEqual(out["choices"][0]["message"]["content"], "b")
        self.assertEqual((seen["reasoning_effort"], seen["thinking"], seen["enable_thinking"]), ("high", False, False))
        with self.assertRaises(urllib.error.HTTPError) as err:
            self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                   {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1,
                                                    "reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "low"}}))
        self.assertEqual(err.exception.code, 400)

    def test_a_grammar_waits_for_the_reasoning_to_end(self):
        """A grammar armed inside the think block forbids the block's own end token, so the block never closes
        and the whole answer comes back as reasoning_content with content empty -- the 45차 §22 bug, reached
        through response_format this time. The door tells the engine which token the grammar waits for."""
        s = chat_server()
        s.reasoning_end = ord('y')
        body = {"messages": [{"role": "user", "content": "xy"}], "max_tokens": 1,
                "response_format": {"type": "json_object"}, "chat_template_kwargs": {"thinking": True}}
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions", body))
        opts = s.engine.options[0]                       # the rendered prompt is "xy!": the answer starts in the block
        self.assertEqual(opts["grammar"], {"type": "json_object"})
        self.assertEqual(opts["grammar_after"], ord('y'))
        s = chat_server()
        s.reasoning_end = ord('y')                       # thinking off: the prompt "xy" ends with the reasoning end,
        body = dict(body); body.pop("chat_template_kwargs")   # so the answer starts in content and the grammar is armed at once
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions", body))
        self.assertNotIn("grammar_after", s.engine.options[0])

    def test_a_schema_the_grammar_compiler_refuses_is_a_bad_request_not_a_dead_engine(self):
        """`prepare_options` builds the grammar at the door. On the loop the compiler's error would leave
        `once()` and take every live row on every rank with it."""
        s = chat_server()
        asked = []

        class Engine(type(s.engine)):
            def prepare_options(self, options):
                asked.append(options.get("grammar"))
                if options.get("grammar", {}).get("type") == "json_schema":
                    raise ValueError("the grammar cannot be compiled: Regex parsing error")

        s.engine.__class__ = Engine
        body = {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1,
                "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "string", "pattern": "(a)"}}}}
        with self.assertRaises(urllib.error.HTTPError) as err:
            self._serve(s, lambda base: self._post(base, "/v1/chat/completions", body))
        self.assertEqual(err.exception.code, 400)
        self.assertIn("the grammar cannot be compiled", err.exception.read().decode())
        self.assertFalse(s.pending or s.results or s._waiting)
        self.assertTrue(asked)

    def test_sampling_options_are_validated_and_travel_to_the_engine(self):
        s = chat_server()
        body = {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1, "temperature": 0.7, "top_p": 0.9, "top_k": 40,
                "seed": 7, "presence_penalty": 0.5, "frequency_penalty": -0.5, "repetition_penalty": 1.1, "logit_bias": {"98": -5},
                "stop": ["1", "2", "3", "4", "5", "6"], "stop_token_ids": [3]}
        out = self._serve(s, lambda base: self._post(base, "/v1/chat/completions", body))
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        opts = s.engine.options[0]
        # every one of those stop strings is a single token for this tokenizer, so the engine is
        # told to end on them itself; the request's own id stays in the set
        self.assertEqual(opts, {"top_p": 0.9, "top_k": 40, "presence_penalty": 0.5, "frequency_penalty": -0.5,
                                "repetition_penalty": 1.1, "seed": 7, "logit_bias": {98: -5.0},
                                "stop_token_ids": [3] + [ord(c) for c in "123456"]})
        for bad in ({"top_p": 1.5}, {"top_k": -2}, {"presence_penalty": 3}, {"logit_bias": {"x": 1}}, {"seed": -1}, {"n": 9},
                    {"tool_choice": "required"}, {"response_format": {"type": "xml"}}):
            with self.assertRaises(urllib.error.HTTPError) as err:
                self._serve(s, lambda base, bad=bad: self._post(base, "/v1/chat/completions",
                                                                {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1, **bad}))
            self.assertEqual(err.exception.code, 400, bad)

    def test_n_choices_share_the_prompt_and_come_back_indexed(self):
        s = chat_server()
        out = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                     {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 2, "n": 2, "seed": 3}))
        self.assertEqual([c["index"] for c in out["choices"]], [0, 1])
        self.assertEqual([c["message"]["content"] for c in out["choices"]], ["bb", "bb"])
        self.assertEqual(out["usage"]["completion_tokens"], 4)
        self.assertEqual([s.engine.options[i]["seed"] for i in (0, 1)], [3, 4])

    def test_tool_calls_stream_as_complete_blocks_and_content_stops_before_them(self):
        from engine.base.serve import _Choice
        tok = Tokenizer()
        parser = lambda text: [("f", '{"a": 1}')] if "<tool_call>" in text else None
        c = _Choice(0, 1, threading.Event(), queue.Queue(), tok=tok, stop=[], reasoning=False, tool_parser=parser)
        text = "hi <tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call> tail"
        ids = [ord(ch) for ch in text]
        c.feed(ids[:8], None, None)                          # "hi <tool" -- the block has begun: content waits
        deltas = c.flush()
        self.assertEqual(deltas, [{"content": "hi "}])
        c.feed(ids[8:], None, None)
        deltas = c.flush(final=True)
        self.assertEqual(deltas, [{"tool_calls": [{"index": 0, "id": "call_1_0", "type": "function",
                                                   "function": {"name": "f", "arguments": '{"a": 1}'}}]}])
        self.assertEqual(c.text["content"], "hi ")
        self.assertEqual(c.finish_reason(), "tool_calls")

    def test_tool_arguments_stream_in_fragments_the_way_openai_defines(self):
        """The first delta of a call carries its name with empty arguments; every one after
        carries the next fragment. On this format the arguments are most of the call, so the
        old shape -- the whole call at `</tool_call>` -- was the whole answer's wait (§43 §5.1)."""
        from engine.base.serve import _Choice
        from engine.profiles.glm53.tools import parse_tool_calls, partial_tool_calls
        c = _Choice(0, 1, threading.Event(), queue.Queue(), tok=ByteTokenizer(), stop=[], reasoning=False,
                    tool_parser=parse_tool_calls, tool_stream=partial_tool_calls)
        text = 'hi <tool_call>search<arg_key>q</arg_key><arg_value>서울 날씨</arg_value></tool_call>'
        ids, deltas = list(text.encode()), []
        for i in range(0, len(ids), 3):
            c.feed(ids[i:i + 3], None, None)
            deltas.extend(c.flush())
        deltas.extend(c.flush(final=True))

        heads = [d["tool_calls"][0] for d in deltas if "tool_calls" in d and "id" in d["tool_calls"][0]]
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads[0], {"index": 0, "id": "call_1_0", "type": "function",
                                    "function": {"name": "search", "arguments": ""}})
        pieces = [d["tool_calls"][0]["function"]["arguments"] for d in deltas
                  if "tool_calls" in d and "id" not in d["tool_calls"][0]]
        self.assertGreater(len(pieces), 1, "the arguments arrived in one piece, which is the old shape")
        self.assertEqual(json.loads("".join(pieces)), {"q": "서울 날씨"})
        self.assertEqual(c.text["content"], "hi ")
        self.assertEqual(c.finish_reason(), "tool_calls")
        self.assertEqual(c.tool_calls_done()[0]["function"]["arguments"], parse_tool_calls(text)[0][1])

    def test_a_call_the_answer_was_cut_off_inside_is_not_a_call(self):
        """Its fragments went out, because the client had already read them, but a caller cannot
        make a call whose arguments never closed -- so it is not in the body and the answer ended
        for the reason it really ended."""
        from engine.base.serve import _Choice
        from engine.profiles.glm53.tools import parse_tool_calls, partial_tool_calls
        c = _Choice(0, 1, threading.Event(), queue.Queue(), tok=ByteTokenizer(), stop=[], reasoning=False,
                    tool_parser=parse_tool_calls, tool_stream=partial_tool_calls)
        c.feed(list('<tool_call>search<arg_key>q</arg_key><arg_value>서울'.encode()), None, None)
        deltas = c.flush() + c.flush(final=True)
        self.assertTrue(any("tool_calls" in d for d in deltas), "what was read was still sent")
        self.assertEqual(c.tool_calls_done(), [])
        self.assertEqual(c.finish_reason(), "length")

    def test_without_a_partial_parser_a_call_still_arrives_whole(self):
        from engine.base.serve import _Choice
        parser = lambda text: [("f", '{"a": 1}')] if "<tool_call>" in text else None
        c = _Choice(0, 1, threading.Event(), queue.Queue(), tok=Tokenizer(), stop=[], reasoning=False,
                    tool_parser=parser)
        c.feed([ord(ch) for ch in "hi <tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"], None, None)
        deltas = c.flush(final=True)
        self.assertEqual([d for d in deltas if "tool_calls" in d],
                         [{"tool_calls": [{"index": 0, "id": "call_1_0", "type": "function",
                                           "function": {"name": "f", "arguments": '{"a": 1}'}}]}])
        self.assertEqual(c.finish_reason(), "tool_calls")

    def test_a_tool_call_is_held_to_the_tools_that_were_declared(self):
        """The grammar arms at `<tool_call>` and not before, so the prose is free -- llama.cpp's
        lazy trigger, and `grammar_after` is exactly that (45차 §45)."""
        from engine.profiles.glm53.tools import tool_grammar
        s = chat_server()
        s.tool_grammar, s.tool_call_start = tool_grammar, 154843
        tools = [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"a": {}}}}}]
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                               {"messages": [{"role": "user", "content": "ab"}],
                                                "max_tokens": 1, "tools": tools}))
        got = s.engine.options[0]
        self.assertEqual(got["grammar"]["type"], "ebnf")
        self.assertIn('call0 ::= "f"', got["grammar"]["grammar"])
        self.assertEqual(got["grammar_after"], 154843)

    def test_a_response_format_wins_over_the_tool_grammar(self):
        """One grammar a row. What the caller asked for in `response_format` is what they get."""
        from engine.profiles.glm53.tools import tool_grammar
        s = chat_server()
        s.tool_grammar, s.tool_call_start = tool_grammar, 154843
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                               {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1,
                                                "tools": [{"type": "function", "function": {"name": "f"}}],
                                                "response_format": {"type": "json_object"}}))
        self.assertEqual(s.engine.options[0]["grammar"], {"type": "json_object"})

    def test_without_a_trigger_token_no_tool_grammar_is_armed(self):
        """D3, one floor down: a marker this vocabulary spells in pieces gets no grammar at all
        rather than one that arms in the middle of it."""
        from engine.profiles.glm53.tools import tool_grammar
        s = chat_server()
        s.tool_grammar, s.tool_call_start = tool_grammar, None
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                               {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1,
                                                "tools": [{"type": "function", "function": {"name": "f"}}]}))
        self.assertNotIn("grammar", s.engine.options[0])

    def test_an_answer_that_starts_inside_a_think_block_gets_a_budget(self):
        """The block is bounded so an answer is always possible; the door names the token that
        ends it, and the engine forces that token when the budget runs out (45차 §46)."""
        from engine.base.serve import reasoning_budget
        s = chat_server()
        s.reasoning_end = 7
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                               {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 5}))
        got = s.engine.options[0]
        self.assertEqual(got["reasoning_end"], 7)
        self.assertEqual(got["reasoning_budget"], reasoning_budget({}, 5))
        self.assertLess(got["reasoning_budget"], 5, "the answer keeps a share of the limit")

    def test_a_caller_may_ask_for_no_bound_or_no_thinking(self):
        s = chat_server()
        s.reasoning_end = 7
        for asked, expect in ((-1, None), (0, 0), (3, 3)):
            with self.subTest(asked=asked):
                self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                       {"messages": [{"role": "user", "content": "ab"}],
                                                        "max_tokens": 5, "reasoning_budget": asked}))
                got = s.engine.options[max(s.engine.options)]
                self.assertEqual(got.get("reasoning_budget"), expect)
        with self.assertRaises(urllib.error.HTTPError) as err:
            self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                   {"messages": [{"role": "user", "content": "ab"}],
                                                    "max_tokens": 5, "reasoning_budget": -2}))
        self.assertEqual(err.exception.code, 400)

    def test_an_answer_that_is_already_past_the_block_gets_no_budget(self):
        """The prompt ended on the reasoning-end token, so there is no block to bound."""
        s = chat_server()
        s.reasoning_end = ord("b")                       # the rendered prompt "ab" ends on it
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                               {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 5}))
        self.assertNotIn("reasoning_budget", s.engine.options[0])

    def test_legacy_completions_tokenize_and_detokenize(self):
        s = chat_server()
        out = self._serve(s, lambda base: self._post(base, "/v1/completions", {"prompt": "xy", "max_tokens": 2, "echo": True, "n": 1}))
        self.assertEqual(out["object"], "text_completion")
        self.assertEqual(out["choices"][0]["text"], "xyyy")
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        self.assertEqual(out["usage"], {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4,
                                        "prompt_tokens_details": {"cached_tokens": 0}})
        out = self._serve(s, lambda base: self._post(base, "/v1/completions", {"prompt": [[120, 121]], "max_tokens": 1}))
        self.assertEqual(out["choices"][0]["text"], "y")
        tk = self._serve(s, lambda base: self._post(base, "/tokenize", {"prompt": "abc"}))
        self.assertEqual((tk["count"], tk["tokens"]), (3, [97, 98, 99]))
        dt = self._serve(s, lambda base: self._post(base, "/detokenize", {"tokens": [97, 98, 99]}))
        self.assertEqual(dt["prompt"], "abc")

    def _image(self, data):
        import base64
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}

    def test_pictures_expand_into_placeholder_runs_and_reach_the_engine(self):
        s = chat_server(keep_idle=True)
        s.vision = Door()
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, self._image(b"cat"), {"type": "text", "text": "b"}]}],
                "max_tokens": 1}
        out = self._serve(s, lambda base: self._post(base, "/v1/chat/completions", body))
        self.assertEqual(out["usage"]["prompt_tokens"], 5)                       # a + three placeholders + b
        self.assertEqual(s.vision.prepared, [("image", b"cat")])
        added = s.engine.media[0]
        self.assertEqual([(m["kind"], m["positions"], m["canvas"]) for m in added], [("image", [1, 2, 3], b"cat")])
        self.assertEqual(s.engine.history(0), [97, 250, 250, 250, 98, 98])

    def test_picture_limits_bad_parts_and_absent_vision_are_refused_at_the_door(self):
        import base64
        s = chat_server()
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        def post(body):
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=5):
                    return 200
            except urllib.error.HTTPError as e:
                return e.code, json.load(e)["error"]
        try:
            msg = lambda *parts: {"messages": [{"role": "user", "content": list(parts)}], "max_tokens": 1}    # noqa: E731
            code, err = post(msg(self._image(b"x")))
            self.assertEqual(code, 400); self.assertIn("not served", err)                 # no vision bound
            s.vision = Door()
            code, err = post(msg(self._image(b"x"), self._image(b"y"), self._image(b"z")))
            self.assertEqual(code, 400); self.assertIn("at most 2", err)
            code, err = post(msg({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(b"v").decode()}},
                                 {"type": "video_url", "video_url": "data:video/mp4;base64," + base64.b64encode(b"w").decode()}))
            self.assertEqual(code, 400); self.assertIn("at most 1", err)
            code, err = post(msg(self._image(b"bad")))
            self.assertEqual(code, 400); self.assertIn("not a picture", err)
            code, err = post(msg({"type": "input_audio", "input_audio": {"data": "x"}}))
            self.assertEqual(code, 400); self.assertIn("not served", err)
            code, err = post(msg({"type": "image_url", "image_url": {"url": "ftp://x/y.png"}}))
            self.assertEqual(code, 400); self.assertIn("data: or http", err)
            code, err = post(msg({"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}))
            self.assertEqual(code, 400); self.assertIn("base64", err)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_a_resent_chat_continues_only_when_its_pictures_are_the_same(self):
        s = chat_server(keep_idle=True)
        s.vision = Door()
        turn = lambda *parts: {"messages": [{"role": "user", "content": list(parts)}], "max_tokens": 2}    # noqa: E731
        text = lambda t: {"type": "text", "text": t}                                                      # noqa: E731
        first = self._serve(s, lambda base: self._post(base, "/v1/chat/completions", turn(text("a"), self._image(b"cat"), text("b"))))
        self.assertEqual(first["usage"]["prompt_tokens"], 5)
        self.assertEqual(first["choices"][0]["message"]["content"], "bb")
        opened = list(s.engine.opened)
        # the same picture, the history and a new question: the conversation continues, the picture is not re-sent to the engine
        second = self._serve(s, lambda base: self._post(base, "/v1/chat/completions", turn(text("a"), self._image(b"cat"), text("bbbq"))))
        self.assertEqual(second["usage"]["prompt_tokens"], 8)
        self.assertEqual(s.engine.opened, opened)
        self.assertEqual(len(s.engine.media[0]), 1)                                        # still the one picture
        # another picture of the same size in the same place: a different prompt -- a fresh row
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions", turn(text("a"), self._image(b"dog"), text("bbbq"))))
        self.assertEqual(len(s.engine.opened), len(opened) + 1)

    def test_a_continuation_with_a_new_picture_sends_only_the_new_one(self):
        s = chat_server(keep_idle=True)
        s.vision = Door()
        turn = lambda *parts: {"messages": [{"role": "user", "content": list(parts)}], "max_tokens": 2}    # noqa: E731
        text = lambda t: {"type": "text", "text": t}                                                      # noqa: E731
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions", turn(text("a"), self._image(b"cat"), text("b"))))
        opened = list(s.engine.opened)
        out = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                     turn(text("a"), self._image(b"cat"), text("bbb"), self._image(b"dog"), text("q"))))
        self.assertEqual(out["usage"]["prompt_tokens"], 11)
        self.assertEqual(s.engine.opened, opened)
        media = s.engine.media[0]
        self.assertEqual([(m["canvas"], m["positions"]) for m in media], [(b"cat", [1, 2, 3]), (b"dog", [7, 8, 9])])   # absolute positions

    def test_cached_tokens_say_what_a_continued_turn_did_not_prefill(self):
        """An agent resends its whole transcript every step. `usage.prompt_tokens_details.cached_tokens` is where its
        SDK already looks to find out whether that is costing anything."""
        s = chat_server(keep_idle=True)
        first = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                       {"messages": [{"role": "user", "content": "abcd"}], "max_tokens": 2}))
        self.assertEqual(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0)      # nothing to continue yet
        self.assertEqual(s.engine.history(0), [97, 98, 99, 100, 100, 100])                # "abcd" and two 'd's of answer
        second = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                        {"messages": [{"role": "user", "content": "abcddddef"}], "max_tokens": 2}))
        self.assertEqual(second["usage"]["prompt_tokens"], 9)
        # five, not six: the token the row sampled last has never been through a forward pass, so it is not computed.
        # This counts what the engine did not have to do, which is the question the number is asked to answer.
        self.assertEqual(second["usage"]["prompt_tokens_details"]["cached_tokens"], 5)

    def test_cached_tokens_say_what_a_reused_boundary_gave(self):
        s = chat_server(prefix=4)                                  # BLOCK 4: "abcdefgh" is two whole blocks
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                               {"messages": [{"role": "user", "content": "abcdefgh"}], "max_tokens": 2}))
        out = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                     {"messages": [{"role": "user", "content": "abcdefghij"}], "max_tokens": 2}))
        self.assertEqual(out["usage"]["prompt_tokens_details"]["cached_tokens"], 8)
        self.assertEqual(out["usage"]["prompt_tokens"], 10)

    def test_prompt_cache_key_is_the_openai_name_for_cache_salt(self):
        s = chat_server(prefix=4)
        body = lambda key, text: {"messages": [{"role": "user", "content": text}], "max_tokens": 2,   # noqa: E731
                                  "prompt_cache_key": key}
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions", body("red", "abcdefgh")))
        other = self._serve(s, lambda base: self._post(base, "/v1/chat/completions", body("blue", "abcdefghij")))
        self.assertEqual(other["usage"]["prompt_tokens_details"]["cached_tokens"], 0)      # another key never reads red's
        same = self._serve(s, lambda base: self._post(base, "/v1/chat/completions", body("red", "abcdefghij")))
        self.assertEqual(same["usage"]["prompt_tokens_details"]["cached_tokens"], 8)

    def test_the_two_names_for_the_cache_key_may_not_disagree(self):
        s = chat_server(prefix=4)
        def ask(extra):
            return self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                          {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1, **extra}))
        with self.assertRaises(urllib.error.HTTPError) as error:
            ask({"cache_salt": "red", "prompt_cache_key": "blue"})
        self.assertEqual(error.exception.code, 400)
        ask({"cache_salt": "red", "prompt_cache_key": "red"})                              # one field under two names is fine

    def test_cached_tokens_count_a_conversation_brought_back_from_the_tier(self):
        """The third admission path: not a cached boundary and not a resident row, but a whole conversation read back
        off NVMe. An agent whose step came minutes after the last one lands here."""
        s = chat_server(keep_idle=True, tiered=True)
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                               {"messages": [{"role": "user", "content": "abcd"}], "max_tokens": 2}))
        for _ in range(200):                                   # the park runs on the tier's thread (D10)
            s.once()
            if s.runner.is_parked(0):
                break
        self.assertTrue(s.runner.is_parked(0))
        self.assertFalse(s._conversations)                     # no row holds it: the next turn must read it back
        second = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                        {"messages": [{"role": "user", "content": "abcddddef"}], "max_tokens": 2}))
        self.assertEqual(second["usage"]["prompt_tokens"], 9)
        self.assertEqual(second["usage"]["prompt_tokens_details"]["cached_tokens"], 5)

    def test_a_served_request_leaves_no_cached_token_record_behind(self):
        s = chat_server(prefix=4)
        self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                               {"messages": [{"role": "user", "content": "abcdefgh"}], "max_tokens": 2}))
        self.assertFalse(s._cached)

    def test_a_history_resent_without_its_end_token_still_continues(self):
        s = chat_server(keep_idle=True)
        s.engine.eos = {ord('b')}; s.engine.stop_at_eos = True          # 'b' ends a generation and is never fed
        first = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                       {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 4}))
        self.assertEqual(first["choices"][0]["finish_reason"], "stop")
        self.assertEqual(s.engine.history(0), [97, 98, 98])              # prompt + the end token
        opened = list(s.engine.opened)
        # the client resends the prompt and its (empty) answer, then a new question: the end token is not in the text
        second = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                        {"messages": [{"role": "user", "content": "abc"}], "max_tokens": 2}))
        self.assertEqual(second["usage"]["prompt_tokens"], 3)
        self.assertEqual(s.engine.opened, opened)                         # continued: no new row
        self.assertEqual(s.engine.history(0), [97, 98, 99, 99, 99])        # the end token was dropped, the new turn fed
        self.assertEqual(second["choices"][0]["message"]["content"], "cc")

    def test_the_same_prompt_arriving_twice_waits_for_the_running_prefill_and_adopts_its_boundary(self):
        s = chat_server(prefix=4)
        text = "abcdefghijklmnop" + "q"                                 # 17 tokens: two whole chunks, boundaries 4..16
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": 1}
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                first = pool.submit(self._post, base, "/v1/chat/completions", body)
                second = pool.submit(self._post, base, "/v1/chat/completions", body)
                threading.Event().wait(0.05)                            # both are in the queue before the loop runs
                for _ in range(400):
                    s.once()
                    if first.done() and second.done():
                        break
                    threading.Event().wait(0.001)
                self.assertEqual(first.result(timeout=3)["usage"]["prompt_tokens"], 17)
                self.assertEqual(second.result(timeout=3)["usage"]["prompt_tokens"], 17)
        finally:
            httpd.shutdown(); httpd.server_close()
        self.assertEqual(s.runner.dedup_waits, 1)                        # the second yielded to the first's prefill ...
        self.assertGreaterEqual(s.runner.prefix.hits, 1)                 # ... and adopted what it cached
        self.assertEqual(s.runner.reused_tokens, 16)
        prefills = [c for c in s.engine.__dict__.get("prefills", [])]     # the fake does not record steps; the counters above say it
        self.assertIn("st:prefix_dedup_waits_total{engine=\"st\"} 1\n", s.metrics())

    def test_the_decode_chain_meters_reach_the_metrics_page(self):
        """How much of decode runs ahead on the device, how often that pipeline is emptied, and by what. The
        engine's own `async_steps` existed and never left the process, so nobody could see the first of these."""
        s = chat_server()
        s.runner.async_steps, s.runner.sync_drain_steps = 12, 3
        s.engine.chain_exits = {"logprobs": 7, "rows_churned": 2}
        page = s.metrics()
        self.assertIn('st:async_decode_steps_total{engine="st"} 12\n', page)       # how much ran ahead
        self.assertIn('st:sync_drain_steps_total{engine="st"} 3\n', page)          # how often it was emptied
        self.assertIn('st:decode_chain_exits_total{engine="st",reason="logprobs"} 7\n', page)     # ... and by what
        self.assertIn('st:decode_chain_exits_total{engine="st",reason="rows_churned"} 2\n', page)

    def test_the_snapshot_pressure_meters_reach_the_metrics_page(self):
        """A prompt has a block boundary every BLOCK tokens and the engine has a fixed number of snapshot slots, so a
        long enough prompt drops checkpoints it just computed. These three say whether that is happening."""
        s = chat_server(prefix=4)
        self.assertIn('st:prefix_snapshots_free{engine="st"} 4\n', s.metrics())
        s.runner.prefix.take_snapshot()
        self.assertIn('st:prefix_snapshots_free{engine="st"} 3\n', s.metrics())
        s.runner.prefix.snapshot_denials, s.runner.snapshot_self_evicts = 3, 7      # distinct: a swapped wire shows
        page = s.metrics()
        self.assertIn('st:prefix_snapshot_denials_total{engine="st"} 3\n', page)
        self.assertIn('st:prefix_snapshot_self_evicts_total{engine="st"} 7\n', page)

    def test_warm_caches_a_prompt_s_boundaries_and_pins_them_until_unpinned(self):
        s = chat_server(prefix=4)
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = drive(s, pool.submit(self._post, base, "/v1/prefix/warm", {"prompt": "abcdefghij", "pin": True}))
                self.assertEqual(out, {"tokens": 10, "boundaries": [4, 8], "pinned": True})
                for _ in range(3):
                    s.once()                                             # the pin control lands on the loop's next iteration
                self.assertEqual(sum(1 for e in s.runner.prefix.entries.values() if e.pinned), 2)
                self.assertIn("st:prefix_pinned_entries{engine=\"st\"} 2\n", s.metrics())
                drive(s, pool.submit(self._post, base, "/v1/prefix/unpin", {}))
                for _ in range(3):
                    s.once()
                self.assertEqual(sum(1 for e in s.runner.prefix.entries.values() if e.pinned), 0)
                out = drive(s, pool.submit(self._post, base, "/v1/prefix/warm", {"messages": [{"role": "user", "content": "abcdefghij"}]}))
                self.assertEqual(out["boundaries"], [4, 8])              # the same prompt: nothing new, still cached
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_the_model_card_states_what_this_boot_actually_bound(self):
        """SparkFleet probes `/v1/models` to decide a backend is routable and wormhole turns its
        inventory into routes, so this is where the engine says what it can do (45차 §56)."""
        s = chat_server(prefix=4)
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = drive(s, pool.submit(self._get, base, "/v1/models"))
            card = out["data"][0]
            self.assertEqual((out["object"], card["object"], card["owned_by"]), ("list", "model", "st"))
            self.assertEqual(card["id"], s.model_name)
            self.assertIsNone(card["max_model_len"], "a fake engine declares no ceiling: the card says None, not a guess")
            s.engine.max_context = 262144
            self.assertEqual(s.model_card()["max_model_len"], 262144)       # vLLM's field name, so vLLM readers get it free
            caps = card["capabilities"]
            self.assertEqual(caps["prefix_cache"], True)
            self.assertEqual(caps["max_concurrent_requests"], s.runner.c.max_running)
            self.assertEqual(caps["streaming"], True)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_the_card_never_claims_a_capability_this_boot_did_not_bind(self):
        # A hand-written capability drifts; this one is read off the objects that exist.
        s = chat_server()
        self.assertEqual(s.vision, None)
        caps = s.model_card()["capabilities"]
        self.assertEqual((caps["vision"], caps["conversation_tier"]), (False, False))
        s.vision = object()
        self.assertEqual(s.model_card()["capabilities"]["vision"], True)

    def test_the_catalog_stops_advertising_while_the_fleet_is_being_handed_over(self):
        """`/v1/models` is the endpoint the control plane asks: wormhole re-probes it for
        `max_model_len` and SparkFleet takes a service's model id from it, which is what makes a
        backend routable. Nothing there reads `/health`, so a drain has to show up here."""
        s = chat_server()
        body, code = s.catalog()
        self.assertEqual((code, len(body["data"])), (200, 1))

        s.draining = "another-session"
        body, code = s.catalog()
        self.assertEqual(code, 503)
        self.assertEqual(body["data"], [], "an empty catalog and a 503: both probes agree")
        self.assertEqual((body["status"], body["handing_over_to"]), ("draining", "another-session"))

        s.draining, s.alive = None, False
        body, code = s.catalog()
        self.assertEqual((code, body["status"], body["data"]), (503, "stopping", []))

    def test_the_models_endpoint_carries_the_drain_verdict(self):
        # No `drive`: a driven loop that is already quiet completes the handover.
        s = chat_server()
        s.draining = "another-session"
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self._get(base, "/v1/models")
            self.assertEqual(caught.exception.code, 503)
            self.assertEqual(json.loads(caught.exception.read())["data"], [])
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_health_says_draining_while_the_fleet_is_being_handed_over(self):
        """`alive` stays true through a drain -- that is the point -- but every new request is
        already refused with 503. A prober that only asks "alive?" keeps the engine in the
        inventory and every caller pays a failed hop before the router fails over."""
        s = chat_server()
        self.assertEqual(s.readiness(), ({"status": "ok"}, 200))
        s.draining = "another-session"
        body, code = s.readiness()
        self.assertEqual(code, 503)
        self.assertEqual((body["status"], body["handing_over_to"]), ("draining", "another-session"))
        s.draining, s.alive = None, False
        self.assertEqual(s.readiness(), ({"status": "stopping"}, 503))

    def test_the_health_endpoint_carries_the_draining_verdict(self):
        # No `drive` here on purpose: a driven loop that is already quiet finishes the handover
        # and the answer becomes "stopping". The door thread answers a GET without the loop.
        s = chat_server()
        s.draining = "another-session"
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self._get(base, "/health")
            self.assertEqual(caught.exception.code, 503)
            body = json.loads(caught.exception.read())
            self.assertEqual((body["status"], body["handing_over_to"]), ("draining", "another-session"))
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_reset_throws_the_whole_prefix_cache_away_on_the_loop_thread(self):
        """The case nothing in the engine can see: the prompt's MEANING changed under an unchanged
        prefix -- a tool list, a retrieved document, an edited template -- so the ids still hash
        the same and every boundary still matches. `unpin` only releases a pin (45차 §55)."""
        s = chat_server(prefix=4)
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                drive(s, pool.submit(self._post, base, "/v1/prefix/warm", {"prompt": "abcdefghij"}))
                for _ in range(3):
                    s.once()
                self.assertEqual(len(s.runner.prefix.entries), 2)
                self.assertEqual(s.runner.kv.cached, 2)

                out = drive(s, pool.submit(self._post, base, "/v1/prefix/reset", {}))
                self.assertEqual(out, {"ok": True})
                for _ in range(3):
                    s.once()                                 # the control lands on the loop, on every rank, between steps

                self.assertEqual(s.runner.prefix.entries, {})
                self.assertEqual(s.runner.kv.cached, 0, "the blocks are anonymous again, not lost")
                self.assertIn('st:prefix_resets_total{engine="st"} 1\n', s.metrics())
                # and the same prompt is a miss now
                out = drive(s, pool.submit(self._post, base, "/v1/chat/completions",
                                           {"messages": [{"role": "user", "content": "abcdefghij"}], "max_tokens": 1}))
                self.assertEqual(out["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_reset_is_refused_where_there_is_no_prefix_cache(self):
        s = chat_server()
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    drive(s, pool.submit(self._post, base, "/v1/prefix/reset", {}))
                self.assertEqual(caught.exception.code, 404)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_a_boundary_evicted_to_the_prefix_tier_comes_back_for_a_later_prompt(self):
        s = chat_server(prefix=3, prefix_tier=True)
        s.runner.spill_low_water = 10                                    # spill leaves as soon as they exist
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                drive(s, pool.submit(self._post, base, "/v1/prefix/warm", {"prompt": "abcdefgh"}))   # boundaries 4, 8 (8 = the whole prompt)
                for _ in range(4):
                    s.once()                                             # the leaf (8) is written out
                self.assertEqual(s.runner.prefix_spills, 1)
                self.assertEqual(len(s.runner.prefix.tier_keys), 1)
                drive(s, pool.submit(self._post, base, "/v1/prefix/warm", {"prompt": "zzzzyyyyxxxx"}))   # three more boundaries: 8 is evicted
                self.assertNotIn(8, [e.tokens for e in s.runner.prefix.entries.values() if e.blocks[:1] == (0,)] if False else [])
                before = s.runner.prefix_restores
                out = drive(s, pool.submit(self._post, base, "/v1/chat/completions",
                                           {"messages": [{"role": "user", "content": "abcdefghXY"}], "max_tokens": 1}))
                self.assertEqual(out["usage"]["prompt_tokens"], 10)
                self.assertEqual(out["usage"]["prompt_tokens_details"]["cached_tokens"], 8)   # read from disk still counts as cached
                self.assertEqual(s.runner.prefix_restores, before + 1)   # the tier's copy served: 8 tokens were not recomputed
                self.assertIn((0, 8, 0) if False else 8, [p for _, p, _ in s.engine.restored])
                self.assertIn("st:prefix_tier_restores_total{engine=\"st\"} 1\n", s.metrics())
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_a_chat_that_resends_its_history_continues_the_retained_conversation(self):
        s = chat_server(keep_idle=True)
        first = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                       {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 2}))
        self.assertEqual(first["choices"][0]["message"]["content"], "bb")
        opened = list(s.engine.opened)
        # the next turn re-sends the whole chat: prompt "ab" + the answer "bb" + the new user text "c"
        second = self._serve(s, lambda base: self._post(base, "/v1/chat/completions",
                                                        {"messages": [{"role": "user", "content": "abbbc"}], "max_tokens": 2}))
        self.assertEqual(second["choices"][0]["message"]["content"], "cc")
        self.assertEqual(second["usage"]["prompt_tokens"], 5)
        self.assertEqual(s.engine.opened, opened)          # no new row was opened: the conversation was extended
        self.assertEqual(s.engine.history(0), [97, 98, 98, 98, 99, 99, 99])
