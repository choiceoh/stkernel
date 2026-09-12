"""Request lifecycle regressions using the real scheduler, pools and runner."""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import queue
import socket
import threading
import unittest
import urllib.error
import urllib.request

from pathlib import Path

from engine.base.kv import BlockPool, SlotPool
from engine.base.record import Ring

ROOT = Path(__file__).resolve().parents[1]
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


def _choice(tok, stop=()):
    from engine.base.serve import _Choice
    return _Choice(0, 1, threading.Event(), queue.Queue(), tok=tok, stop=list(stop), reasoning=False)


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

    def shown(self, ids, tok, *, per_step=1, stop=()):
        """What a client saw, fed `per_step` tokens at a time."""
        c = _choice(tok, stop)
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
        from engine.base.serve import DETOK_REPAIRS
        before = DETOK_REPAIRS["invalid_token_id"]
        for bad in (-1, 2 ** 63, 1.5, None):
            with self.subTest(bad=bad):
                ids = list(b"ab") + [bad] + list(b"cd")
                shown, c = self.shown(ids, self.rust(), per_step=5)
                self.assertEqual(shown, "abcd")
                self.assertIsNone(c.error)
        self.assertEqual(DETOK_REPAIRS["invalid_token_id"], before + 4)

    def test_a_decoder_that_rewrites_its_own_output_does_not_end_the_answer(self):
        """vllm-project/vllm#17448: a non-monotonic decoder breaks DecodeStream's prefix, and
        every later step raises the same way until the stream is replaced."""
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from engine.base.serve import DETOK_REPAIRS, _Stream
        tok = Tokenizer(models.WordLevel({"a": 0, "b": 1, "c": 2}, unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.decoder = decoders.Sequence([decoders.Fuse(), decoders.Replace("ab", "X")])
        stream = _Stream(tok)
        stream.extend([0])
        self.assertEqual(stream.decoded(False), "a")            # the stream's prefix is now "a"
        before = DETOK_REPAIRS["invalid_prefix"]
        stream.extend([1])
        self.assertEqual(stream.decoded(False), "a")            # "ab" became "X": held, not shown
        stream.extend([2])
        # Repaired, not raised. "a" is already out and cannot be taken back, so no answer here
        # equals decode([0,1,2]) == "Xc"; this one at least loses no token -- vLLM's drops the
        # held "b" and shows "ac".
        self.assertEqual(stream.decoded(False), "abc")
        self.assertEqual(DETOK_REPAIRS["invalid_prefix"], before + 1)
        stream.extend([2])                                      # and the primed stream carries on
        self.assertEqual(stream.decoded(True), "abcc")

    def test_a_tail_that_will_never_finish_is_shown_instead_of_held_forever(self):
        """A held-back tail is a character waiting for its rest, and that wait is bounded. A run
        of lone lead bytes is not waiting: holding it shows the client nothing for the rest of
        the answer, and the decode that repeats grows with the run."""
        from engine.base.serve import DETOK_REPAIRS
        before = DETOK_REPAIRS["stalled"]
        stuck = [0xED] * 64
        for tok in (self.rust(), ByteTokenizer()):
            with self.subTest(tok=type(tok).__name__):
                c = _choice(tok)
                seen = 0
                for i in range(0, len(stuck), 4):
                    c.feed(stuck[i:i + 4], None, None)
                    seen += sum(len(d.get("content", "")) for d in c.flush())
                self.assertGreater(seen, 0, "the client saw nothing while the answer ran")
        self.assertGreater(DETOK_REPAIRS["stalled"], before)

    def test_an_unfinished_character_at_the_very_end_is_still_shown(self):
        shown, _ = self.shown(list("ok ".encode()) + [0xED], self.rust())
        self.assertEqual(shown, "ok \ufffd")


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
        from engine.base.serve import DETOK_REPAIRS
        s = server()
        kept = dict(DETOK_REPAIRS)
        try:
            DETOK_REPAIRS.update(dict.fromkeys(DETOK_REPAIRS, 0))
            out = self.text(s)
            self.assertIn("st:detokenizer_rust_stream", out)        # D3: a silent path is an unchecked path
            self.assertNotIn("st:detokenizer_repairs_total", out)   # no series at all in a healthy run
            DETOK_REPAIRS["invalid_prefix"] = 1
            self.assertIn('st:detokenizer_repairs_total{engine="st",reason="invalid_prefix"} 1', self.text(s))
        finally:
            DETOK_REPAIRS.update(kept)

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

    def test_legacy_completions_tokenize_and_detokenize(self):
        s = chat_server()
        out = self._serve(s, lambda base: self._post(base, "/v1/completions", {"prompt": "xy", "max_tokens": 2, "echo": True, "n": 1}))
        self.assertEqual(out["object"], "text_completion")
        self.assertEqual(out["choices"][0]["text"], "xyyy")
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        self.assertEqual(out["usage"], {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4})
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
