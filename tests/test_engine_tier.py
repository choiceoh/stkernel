"""Tier ownership and asynchronous error handling, without CUDA or real KV."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from engine.base.kv import BlockPool
from engine.base.kv_tier import NvmeTier, SECTOR, TierFull
from engine.base.tiered_kv import TieredKV


class Storage(bytearray):
    def numel(self):
        return len(self)


class MemoryTier:
    """Byte storage with failures after partial I/O, exercising real pool ownership.

    Mirrors NvmeTier's contract: blocks, optional slot bytes (`extra`), a host
    `record`, a capacity that raises TierFull, `keys()` and `oldest()`."""
    block_bytes = 4

    def __init__(self, capacity=None, delay=0.0, gate=None):
        self.index = {}
        self.data, self.extra, self.records = {}, {}, {}
        self.capacity = capacity                    # conversations, not bytes: enough for the policy
        self.order = []
        self.delay, self.gate = delay, gate         # a slow tier thread: sleep, or wait for an Event the test sets
        self.fail_demote = self.fail_promote = self.fail_forget = False

    def run_async(self, fn, *args):
        """Like NvmeTier.run_async: the transfer on its own thread. With no delay and no gate it
        completes before returning, which keeps the ownership tests sequential."""
        from concurrent.futures import Future
        future = Future()

        def work():
            future.set_running_or_notify_cancel()
            try:
                if self.gate is not None and not self.gate.wait(5):
                    raise TimeoutError("test did not open the tier gate")
                if self.delay:
                    time.sleep(self.delay)
                future.set_result(fn(*args))
            except BaseException as exc:            # noqa: BLE001
                future.set_exception(exc)

        if self.delay or self.gate is not None:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()
        return future

    def has(self, seq):
        return str(seq) in self.index

    def keys(self):
        return sorted(int(k) for k in self.index)

    def oldest(self):
        return self.order[0] if self.order else None

    def record(self, seq):
        return self.records.get(seq)

    def demote(self, seq, storage, ids, tokens, extra=None, record=None):
        from engine.base.kv_tier import TierFull
        if self.fail_demote:
            raise OSError("disk write failed")
        if self.capacity is not None and len(self.index) >= self.capacity:
            raise TierFull("memory tier full")
        data = b"".join(storage[i * 4:(i + 1) * 4] for i in ids)
        self.data[seq] = data
        if extra is not None:
            self.extra[seq] = bytes(extra)
        if record is not None:
            self.records[seq] = json.loads(json.dumps(record))    # what a JSON file would give back
        self.index[str(seq)] = {"tokens": tokens, "blocks": len(ids), "bytes": len(data) + len(self.extra.get(seq, b"")),
                                "extra": len(self.extra.get(seq, b""))}
        self.order.append(seq)
        return self.index[str(seq)]["bytes"]

    def promote(self, seq, storage, ids, extra=None):
        data = self.data[seq]
        if ids is not None and len(ids) != self.index[str(seq)]["blocks"]:
            raise ValueError("wrong block count")
        want = len(self.extra.get(seq, b""))
        if (len(extra) if extra is not None else 0) != want:
            raise ValueError("slot bytes on disk do not match the view given")
        if ids is None and not want:
            raise ValueError("a snapshot-only read would read nothing")
        for j, i in enumerate(ids or ()):                 # ids None: the blocks are already in memory, read the slot alone
            storage[i * 4:(i + 1) * 4] = data[j * 4:(j + 1) * 4]
            if self.fail_promote:
                raise OSError("disk read failed")
        if self.fail_promote and ids is None:
            raise OSError("disk read failed")
        if want:
            extra[:] = self.extra[seq]
        return want if ids is None else self.index[str(seq)]["bytes"]

    def forget(self, seq):
        if self.fail_forget:
            raise OSError("manifest write failed")
        self.index.pop(str(seq))
        self.data.pop(seq)
        self.extra.pop(seq, None)
        self.records.pop(seq, None)
        self.order.remove(seq)


def make_tier():
    pool = BlockPool(4, 16, 4, 4)
    pool.attach_storage(Storage(range(16)), 4)
    return TieredKV(pool, MemoryTier())


class TierOwnershipTests(unittest.TestCase):
    def test_failed_demote_keeps_resident_blocks_and_tokens(self):
        kv = make_tier()
        kv.pool.reserve(0, 17)
        before = (list(kv.pool.row(0)), bytes(kv.pool.storage), kv.pool.available)
        kv.tier.fail_demote = True
        with self.assertRaisesRegex(OSError, "write failed"):
            kv.park(0)
        self.assertEqual((list(kv.pool.row(0)), bytes(kv.pool.storage), kv.pool.available), before)
        self.assertEqual(kv.pool.tokens[0], 17)
        self.assertFalse(kv.is_parked(0))

    def test_partial_promote_returns_blocks_and_can_retry_into_different_blocks(self):
        kv = make_tier()
        kv.pool.reserve(0, 17)
        kv.park(0)
        saved = kv.tier.data[0]
        kv.tier.fail_promote = True
        with self.assertRaisesRegex(OSError, "read failed"):
            kv.resume(0)
        self.assertEqual(kv.pool.available, 4)
        self.assertEqual(kv.pool.tokens[0], 0)
        self.assertEqual(kv.pool.rows_in_use, 0)
        self.assertTrue(kv.is_parked(0))
        self.assertEqual(kv.parked[0], 17)
        kv.pool.reserve(1, 16)                 # another conversation takes a returned block
        kv.tier.fail_promote = False
        self.assertEqual(kv.resume(0), len(saved))
        self.assertEqual(b"".join(kv.pool.blocks_of(0)), saved)
        self.assertEqual(kv.pool.tokens[0], 17)
        self.assertFalse(kv.is_parked(0))

    def test_a_resume_given_up_after_its_read_returns_the_blocks_and_keeps_the_disk_copy(self):
        """The runner gives a resume up when the blocks came back but its record did not (`Runner.resume_finish`)."""
        kv = make_tier()
        kv.pool.reserve(0, 17)
        kv.park(0, key=5, record={"context": 16, "pending": 1})
        self.assertEqual(kv.read_record(5).result(), {"context": 16, "pending": 1})
        kv.resume_begin(1, key=5)
        self.assertEqual(kv.pool.available, 2)
        kv.resume_cancel(1)
        self.assertEqual((kv.pool.available, kv.pool.tokens[1], kv.inflight), (4, 0, {}))
        self.assertTrue(kv.is_parked(5))
        kv.resume(2, key=5)                                   # and it still comes back
        self.assertEqual(kv.pool.tokens[2], 17)
        self.assertFalse(kv.is_parked(5))

    def test_exhaustion_preserves_parked_state_for_retry(self):
        kv = make_tier()
        kv.pool.reserve(0, 17)
        kv.park(0)
        kv.pool.reserve(1, 48)
        with self.assertRaises(MemoryError):
            kv.resume(0)
        self.assertEqual(kv.parked, {0: 17})
        self.assertEqual(kv.pool.tokens[0], 0)
        self.assertEqual(kv.pool.available, 1)
        kv.pool.release(1)
        self.assertEqual(kv.resume(0), 8)

    def test_resume_after_restart_uses_manifest(self):
        kv = make_tier()
        kv.pool.reserve(0, 17)
        kv.park(0)
        restarted = TieredKV(kv.pool, kv.tier)
        self.assertTrue(restarted.is_parked(0))
        self.assertEqual(restarted.resume(0), 8)
        self.assertEqual(kv.pool.tokens[0], 17)

    def test_resume_rejects_a_row_reused_by_a_live_sequence(self):
        kv = make_tier()
        kv.pool.reserve(0, 17)
        kv.park(0)
        kv.pool.reserve(0, 1)
        before = (list(kv.pool.row(0)), kv.pool.available, bytes(kv.pool.storage))
        with self.assertRaises(ValueError):
            kv.resume(0)
        self.assertEqual((list(kv.pool.row(0)), kv.pool.available, bytes(kv.pool.storage)), before)
        self.assertEqual(kv.pool.tokens[0], 1)
        self.assertEqual(kv.parked, {0: 17})

    def test_failed_disk_cleanup_preserves_successfully_restored_memory(self):
        kv = make_tier()
        kv.pool.reserve(0, 17)
        kv.park(0)
        expected = kv.tier.data[0]
        kv.tier.fail_forget = True
        with self.assertRaisesRegex(OSError, "manifest"):
            kv.resume(0)
        self.assertEqual(kv.pool.tokens[0], 17)
        self.assertEqual(b"".join(kv.pool.blocks_of(0)), expected)

    def test_conversation_key_slot_bytes_and_record_travel_between_rows(self):
        kv = make_tier()
        kv.pool.reserve(3, 17)
        slots = [bytearray(b"....") for _ in range(3)]
        slots[2][:] = b"abcd"
        before = b"".join(kv.pool.blocks_of(3))
        kv.park(3, key=42, extra=memoryview(slots[2]), record={"context": 17, "pending": 1, "tokens": [1, 2]})
        self.assertTrue(kv.is_parked(42) and not kv.is_parked(3))
        self.assertEqual(kv.pool.tokens[3], 0)
        self.assertEqual(kv.record(42), {"context": 17, "pending": 1, "tokens": [1, 2]})
        self.assertEqual((kv.blocks(42), kv.keys(), kv.oldest()), (2, [42], 42))
        kv.pool.reserve(3, 5)                       # the old row is someone else's now
        slots[2][:] = b"zzzz"
        got = kv.resume(0, key=42, extra=memoryview(slots[1]))   # a different row, a different slot
        self.assertEqual(got, 8 + 4)
        self.assertEqual(b"".join(kv.pool.blocks_of(0)), before)
        self.assertEqual(bytes(slots[1]), b"abcd")
        self.assertEqual(kv.pool.tokens[0], 17)
        self.assertFalse(kv.is_parked(42) or kv.tier.index)

    def test_slot_bytes_on_disk_require_a_slot_view_to_resume(self):
        kv = make_tier()
        kv.pool.reserve(0, 17)
        kv.park(0, key=7, extra=memoryview(bytearray(b"wxyz")))
        with self.assertRaisesRegex(ValueError, "slot bytes"):
            kv.resume(0, key=7)
        self.assertTrue(kv.is_parked(7))
        self.assertEqual(kv.pool.available, 4)

    def test_park_halves_keep_blocks_reserved_until_the_write_is_done(self):
        gate = threading.Event()
        pool = BlockPool(4, 16, 4, 4)
        pool.attach_storage(Storage(range(16)), 4)
        kv = TieredKV(pool, MemoryTier(gate=gate))
        kv.pool.reserve(0, 17)
        kv.park_begin(0, key=9, record={"context": 17, "pending": 1})
        self.assertFalse(kv.done(0))
        self.assertEqual((kv.pool.tokens[0], kv.pool.available), (17, 2))   # still reserved: the write reads them
        with self.assertRaisesRegex(ValueError, "in flight"):
            kv.park_begin(0, key=10)
        gate.set()
        for _ in range(200):
            if kv.done(0):
                break
            time.sleep(0.005)
        self.assertTrue(kv.done(0))
        self.assertEqual(kv.park_finish(0), 8)
        self.assertEqual((kv.pool.tokens[0], kv.pool.available), (0, 4))
        self.assertTrue(kv.is_parked(9) and not kv.inflight)
        # resume halves: blocks reserved at begin, committed at finish
        gate.clear()
        kv.resume_begin(2, key=9)
        self.assertEqual((kv.pool.tokens[2], kv.pool.available), (17, 2))
        self.assertFalse(kv.done(2))
        gate.set()
        for _ in range(200):
            if kv.done(2):
                break
            time.sleep(0.005)
        self.assertEqual(kv.resume_finish(2), 8)
        self.assertFalse(kv.is_parked(9) or kv.inflight)
        self.assertEqual(kv.pool.tokens[2], 17)

    def test_a_key_cannot_be_parked_twice(self):
        kv = make_tier()
        kv.pool.reserve(0, 17)
        kv.pool.reserve(1, 3)
        kv.park(0, key=5)
        with self.assertRaisesRegex(ValueError, "already parked"):
            kv.park(1, key=5)
        self.assertEqual(kv.pool.tokens[1], 3)
        kv.forget(5)
        self.assertFalse(kv.is_parked(5))


class NvmeControlTests(unittest.TestCase):
    def tier(self, directory):
        tier = NvmeTier.__new__(NvmeTier)
        tier.dir = Path(directory)
        tier.block_bytes = SECTOR
        tier.manifest = tier.dir / "manifest.json"
        tier.index = json.loads(tier.manifest.read_text()) if tier.manifest.exists() else {}
        tier.lock, tier._transfer_lock = threading.Lock(), threading.Lock()
        return tier

    def test_failed_replacement_keeps_prior_generation_and_cleanup_reclaims_only_unpublished_files(self):
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            old = tier.dir / 'seq-0.kv'
            old.write_bytes(b'original')
            tier._save_manifest({'0': {'blocks': 1, 'tokens': 1, 'bytes': 8}})
            new = tier.dir / ('seq-0-' + 'a' * 32 + '.kv')
            new.write_bytes(b'new bytes')
            with patch('engine.base.kv_tier.os.replace', side_effect=OSError('manifest full')):
                with self.assertRaisesRegex(OSError, 'manifest full'):
                    tier._publish(0, new, 1, 2, 9)
            self.assertEqual(tier._path(0).read_bytes(), b'original')

            self.assertEqual(tier.index, json.loads(tier.manifest.read_text()))
            unrelated = tier.dir / 'seq-99.kv'
            unrelated.write_bytes(b'legacy, not in manifest')
            tier.cleanup()
            self.assertFalse(new.exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(tier._path(0).read_bytes(), b'original')

    def test_cleanup_preserves_a_foreign_layouts_files(self):
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            active = tier.dir / ('seq-0-' + 'a' * 32 + '.kv')
            retired = tier.dir / ('seq-0-' + 'b' * 32 + '.kv')
            active.write_bytes(b'active'); retired.write_bytes(b'retired')
            tier._save_manifest({'0': {'file': active.name, 'retired': [retired.name], 'block_bytes': 2 * SECTOR}})
            self.assertFalse(tier.has(0))
            self.assertEqual(tier.stale(), ['0'])
            tier.cleanup()
            self.assertEqual(active.read_bytes(), b'active')
            self.assertEqual(retired.read_bytes(), b'retired')

    def test_a_write_under_a_foreign_layouts_key_replaces_it(self):
        """GLM-5.3 and Qwen3.8 park in one directory (tiered_kv.TIER_ROOT), each numbering its conversations from its
        own parked set, so a key both used is expected. This write used to be refused -- dropping the live
        conversation to keep one no boot of this layout can read."""
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            tier.capacity_bytes, tier.reserve_bytes = None, 0
            theirs = tier.dir / ('seq-0-' + 'a' * 32 + '.kv')
            record = tier.dir / ('seq-0-' + 'a' * 32 + '.json')
            theirs.write_bytes(b'theirs'); record.write_text('{}')
            tier._save_manifest({'0': {'file': theirs.name, 'record': record.name, 'bytes': 6,
                                       'block_bytes': 2 * SECTOR}})
            with patch.object(tier, '_room', side_effect=TierFull('stopped before the device')):
                with self.assertRaises(TierFull):             # the room check is where this CPU test stops the write
                    tier.demote(0, None, [], 1)
            self.assertNotIn('0', tier.index, "the foreign conversation is forgotten before this one is written")
            self.assertFalse(theirs.exists() or record.exists())
            self.assertEqual(json.loads(tier.manifest.read_text()), {})

    def test_the_reserve_holds_under_a_declared_cap(self):
        """45차 §53 gave both tiers a cap, and from then on the reserve was not checked at all: a tier under its cap
        wrote on into a disk the checkpoints, images, dumps and logs were filling too. Both hold now."""
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            tier.capacity_bytes, tier.reserve_bytes = 64 << 30, 16 << 30
            with patch('engine.base.kv_tier.shutil.disk_usage', return_value=SimpleNamespace(free=20 << 30)):
                tier._room(3 << 30, 0)                             # leaves 17 GiB free: under the cap and the reserve
                with self.assertRaisesRegex(TierFull, 'reserve'):
                    tier._room(5 << 30, 0)                         # would leave 15 GiB, under the 16 kept free
            tier.capacity_bytes = 1 << 30
            with patch('engine.base.kv_tier.shutil.disk_usage', return_value=SimpleNamespace(free=1 << 40)):
                with self.assertRaisesRegex(TierFull, 'declared'):
                    tier._room(2 << 30, 0)                         # a roomy disk does not lift the cap

    def test_a_foreign_layout_is_forgotten_before_a_conversation_that_can_still_be_promoted(self):
        # It occupies the disk and counts against the cap, and no boot of this layout can ever
        # promote it. Until this order existed, a tier whose stale entries alone filled the cap
        # had nothing it was willing to give up and refused every park forever.
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            tier._save_manifest({
                "5": {"file": "seq-5.kv", "at": 100.0, "bytes": 7, "block_bytes": SECTOR},        # oldest, promotable
                "9": {"file": "seq-9.kv", "at": 300.0, "bytes": 11, "block_bytes": SECTOR},
                "7": {"file": "seq-7.kv", "at": 200.0, "bytes": 13, "block_bytes": 2 * SECTOR},   # foreign, newer
            })
            self.assertEqual(tier.oldest(), 7, "the unpromotable entry goes first, old or not")
            self.assertEqual(tier.stale_bytes(), 13)

            tier.forget(7)
            self.assertEqual(tier.oldest(), 5, "then the least recently parked one this layout can use")
            self.assertEqual(tier.stale_bytes(), 0)

    def test_a_tier_full_of_foreign_entries_can_still_make_room(self):
        # The deadlock declaring a capacity would otherwise have created: 85 GiB of one
        # checkpoint's parked conversations, a cap below that, and an LRU with nothing to give.
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            tier.capacity_bytes, tier.reserve_bytes = 20, 0
            tier._save_manifest({str(seq): {"file": f"seq-{seq}.kv", "at": float(seq), "bytes": 10,
                                            "block_bytes": 2 * SECTOR} for seq in (1, 2)})
            with self.assertRaises(TierFull):
                tier._room(10, 0)
            self.assertEqual(tier.keys(), [], "none of them is promotable")

            while tier.used_bytes() + 10 > tier.capacity_bytes:
                seq = tier.oldest()
                self.assertIsNotNone(seq, "the cap must never be unservable")
                tier.forget(seq)
            tier._room(10, 0)                                  # room, without a single promotable entry lost

    def test_a_deleting_entry_is_not_offered_twice(self):
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            tier._save_manifest({"3": {"file": "seq-3.kv", "at": 1.0, "bytes": 5,
                                       "block_bytes": 2 * SECTOR, "deleting": True}})
            self.assertIsNone(tier.oldest())
            self.assertEqual(tier.stale_bytes(), 0)

    def test_failed_unlink_keeps_a_restartable_cleanup_tombstone(self):
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            tier._path(0).write_bytes(b'snapshot')
            tier._save_manifest({'0': {'blocks': 1, 'tokens': 1, 'bytes': 8}})
            with patch('pathlib.Path.unlink', side_effect=PermissionError('busy')):
                with self.assertRaisesRegex(PermissionError, 'busy'):
                    tier.forget(0)
            self.assertFalse(tier.has(0))             # no longer promotable once deletion commits
            self.assertTrue(tier.index['0']['deleting'])
            self.assertEqual(tier._path(0).read_bytes(), b'snapshot')
            reopened = self.tier(d)
            reopened.cleanup()
            self.assertFalse(reopened.index)
            self.assertFalse(tier._path(0).exists())

    def test_final_manifest_failure_after_unlink_is_retryable_after_restart(self):
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            tier._path(0).write_bytes(b'snapshot')
            tier._save_manifest({'0': {'blocks': 1, 'tokens': 1, 'bytes': 8}})
            save = tier._save_manifest
            def fail_final(index):
                if not index:
                    raise OSError('final manifest failed')
                save(index)
            with patch.object(tier, '_save_manifest', side_effect=fail_final):
                with self.assertRaisesRegex(OSError, 'final manifest failed'):
                    tier.forget(0)
            self.assertFalse(tier._path(0).exists())
            self.assertTrue(tier.index['0']['deleting'])
            reopened = self.tier(d)
            reopened.forget(0)
            self.assertFalse(reopened.index)

    def test_retired_files_remain_discoverable_until_cleanup_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            old = tier._path(0)
            old.write_bytes(b'old')
            tier._save_manifest({'0': {'blocks': 1, 'tokens': 1, 'bytes': 3}})
            new = tier.dir / ('seq-0-' + 'b' * 32 + '.kv')
            new.write_bytes(b'new')
            tier._publish(0, new, 1, 2, 3)
            with patch('pathlib.Path.unlink', side_effect=PermissionError('busy')):
                with self.assertRaises(PermissionError):
                    tier.cleanup()
            self.assertTrue(tier.has(0))
            self.assertEqual(tier.index['0']['retired'], [old.name])
            self.assertEqual(tier._path(0).read_bytes(), b'new')
            tier.cleanup()
            self.assertFalse(old.exists())
            self.assertEqual(tier.index['0']['retired'], [])
            self.assertEqual(tier._path(0).read_bytes(), b'new')

    def test_async_transfers_cannot_use_shared_staging_concurrently(self):
        tier = NvmeTier.__new__(NvmeTier)
        tier._transfer_lock = threading.Lock()
        writing, release, reading, attempted = (threading.Event() for _ in range(4))

        def write(*args):
            writing.set()
            if not release.wait(2):
                raise TimeoutError("test did not release writer")
            return 8

        def read(*args):
            reading.set()
            return 8

        def promote():
            attempted.set()
            return tier.promote(0, None, [0, 1])

        tier._demote, tier._promote = write, read
        writer = tier.run_async(tier.demote, 0, None, [0, 1], 17)
        try:
            self.assertTrue(writing.wait(2))
            reader = tier.run_async(promote)
            self.assertTrue(attempted.wait(2))
            self.assertFalse(reading.wait(0.05))
        finally:
            release.set()
        self.assertEqual(writer.result(timeout=2), 8)
        self.assertEqual(reader.result(timeout=2), 8)

    def test_close_returns_the_staging_buffers_and_leaves_what_is_parked_on_disk(self):
        # A tier's pinned host staging and device scratch live OUTSIDE the arena, so the
        # engine's release cannot reach them: on a handover they are held for nothing.
        import torch
        tier = NvmeTier.__new__(NvmeTier)
        tier.stage_t = torch.empty(64, dtype=torch.uint8)
        tier.stage = memoryview(tier.stage_t.numpy())
        tier.scratch = torch.empty(32, dtype=torch.uint8)
        tier.stream = object()
        tier.index = {"7": {"file": "seq-7.kv"}}

        self.assertEqual(tier.close(), 96)
        self.assertIsNone(tier.stage)
        self.assertIsNone(tier.stage_t)
        self.assertIsNone(tier.scratch)
        self.assertIsNone(tier.stream)
        self.assertEqual(tier.index, {"7": {"file": "seq-7.kv"}})   # D16: parked conversations outlive the process
        self.assertEqual(tier.close(), 0)                           # idempotent

    def test_closing_a_tier_that_has_no_staging_of_its_own_is_nothing_rather_than_an_error(self):
        # Probes hand TieredKV a bare object; a shutdown must not care which kind it got.
        self.assertEqual(make_tier().close(), 0)

    def test_close_waits_for_a_transfer_still_holding_the_staging_it_is_about_to_free(self):
        kv = make_tier()
        started, release, done = threading.Event(), threading.Event(), []

        def work():
            started.set()
            release.wait(2)
            done.append("finished")
            return 8

        kv.inflight[0] = ("park", 11, 16, kv.tier.run_async(work))
        self.assertTrue(started.wait(2))
        threading.Timer(0.05, release.set).start()
        kv.close()
        self.assertEqual(done, ["finished"], "the staging went while a thread was still in it")
        self.assertFalse(kv.inflight)

    def test_a_failed_transfer_does_not_stop_the_shutdown(self):
        from concurrent.futures import Future
        kv = make_tier()
        failed = Future()
        failed.set_exception(OSError("the disk went away"))
        kv.inflight[0] = ("park", 11, 16, failed)
        self.assertEqual(kv.close(), 0)
        self.assertFalse(kv.inflight)

    def test_invalid_staging_geometry_fails_before_cuda_allocation(self):
        for block, stage in [(0, SECTOR), (-SECTOR, SECTOR), (1, SECTOR),
                             (SECTOR, 0), (SECTOR, SECTOR - 1)]:
            with self.subTest(block=block, stage=stage), self.assertRaises(ValueError):
                NvmeTier("unused", block, stage)

    def test_async_handle_reports_completion_and_result(self):
        tier = NvmeTier.__new__(NvmeTier)
        entered, release = threading.Event(), threading.Event()

        def work():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release worker")
            return 123

        future = tier.run_async(work)
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(future.done())
        finally:
            release.set()
        self.assertEqual(future.result(timeout=2), 123)
        self.assertTrue(future.done())

    def test_async_failure_reaches_the_caller(self):
        tier = NvmeTier.__new__(NvmeTier)

        def fail():
            raise OSError("disk disconnected")

        future = tier.run_async(fail)
        with self.assertRaisesRegex(OSError, "disk disconnected"):
            future.result(timeout=2)

    def test_failed_manifest_replacement_keeps_disk_and_memory_index(self):
        with tempfile.TemporaryDirectory() as d:
            tier = NvmeTier.__new__(NvmeTier)
            tier.dir = Path(d)
            tier.block_bytes = SECTOR
            tier.manifest = tier.dir / "manifest.json"
            tier.lock = threading.Lock()
            tier._transfer_lock = threading.Lock()
            tier.index = {"0": {"tokens": 17, "blocks": 2, "bytes": 8}}
            tier.manifest.write_text(json.dumps(tier.index))
            tier._path(0).write_bytes(b"saved kv")
            with patch("engine.base.kv_tier.os.replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    tier.forget(0)
            self.assertTrue(tier.has(0))
            self.assertEqual(json.loads(tier.manifest.read_text()), tier.index)
            self.assertEqual(tier._path(0).read_bytes(), b"saved kv")


class Index:
    """Just enough tier for the line the boot prints."""

    def __init__(self, index, block_bytes=SECTOR):
        self.index, self.block_bytes = index, block_bytes

    has = NvmeTier.has
    _compatible = NvmeTier._compatible
    keys = NvmeTier.keys
    stale = NvmeTier.stale
    stale_bytes = NvmeTier.stale_bytes
    used_bytes = NvmeTier.used_bytes


class TierBudgetTests(unittest.TestCase):
    """The disk is a declared budget too (D1), and the boot says where it stands."""

    def test_both_tiers_are_declared_and_the_conversations_get_the_larger_share(self):
        from engine.profiles.glm53 import boot
        # A boundary recomputes; a turn the user may come back to does not (D16).
        self.assertGreater(boot.TIER_GIB, boot.PREFIX_TIER_GIB)
        self.assertEqual((boot.TIER_GIB, boot.PREFIX_TIER_GIB, boot.TIER_RESERVE_GIB), (64.0, 16.0, 16.0))

    def test_the_boot_line_says_gibibytes_because_a_count_was_never_the_question(self):
        from engine.base.budget import GIB
        from engine.profiles.glm53 import boot
        tier = Index({"1": {"at": 1.0, "bytes": 3 * GIB, "block_bytes": SECTOR},
                      "2": {"at": 2.0, "bytes": 5 * GIB, "block_bytes": 2 * SECTOR}})
        line = boot.tier_line(tier, boot.TIER_GIB, "conversations")
        self.assertIn("1 conversations parked from before, 8.0 GiB of 64 GiB", line)
        self.assertIn("1 under another layout holding 5.0 GiB", line)
        self.assertIn("first thing forgotten when the cap bites", line)

    def test_a_tier_with_nothing_foreign_in_it_does_not_mention_foreign_layouts(self):
        from engine.base.budget import GIB
        from engine.profiles.glm53 import boot
        tier = Index({"1": {"at": 1.0, "bytes": 2 * GIB, "block_bytes": SECTOR}})
        self.assertNotIn("another layout", boot.tier_line(tier, boot.PREFIX_TIER_GIB, "prefix boundaries"))


class FakeNvme:
    """What `open_tiers` hands a tier class, kept: an NvmeTier allocates CUDA staging."""

    def __init__(self, directory, **kwargs):
        self.dir, self.kwargs, self.swept = Path(directory), kwargs, 0
        self.block_bytes = kwargs["block_bytes"]
        self.fail = None

    def cleanup(self):
        self.swept += 1
        if self.fail is not None:
            raise self.fail


class FleetTierTests(unittest.TestCase):
    """base/tiered_kv.open_tiers: a rank's two tiers in the directory every profile on the fleet shares."""

    def pool(self):
        pool = BlockPool(4, 16, 4, 4)
        pool.attach_storage(Storage(range(16)), 4)
        return pool

    def test_a_rank_gets_its_conversations_and_its_prefix_boundaries_under_the_fleets_caps(self):
        from engine.base import tenancy
        from engine.base.tiered_kv import open_tiers
        with tempfile.TemporaryDirectory() as d:
            conversations, prefix, left = open_tiers(self.pool(), 4, d, 2, state_format="fmt", mapped_staging=True,
                                                     prefix_cache_bytes=7, background=False, make=FakeNvme)
            self.assertIsNone(left)
            self.assertFalse((Path(d) / "rank2" / tenancy.MARKER).exists(), "no owner claims nothing")
            c, p = conversations.tier, prefix.tier
            self.assertEqual((c.dir, p.dir), (Path(d) / "rank2", Path(d) / "rank2" / "prefix"))
            self.assertEqual((c.kwargs["capacity_bytes"], c.kwargs["reserve_bytes"]), (64 << 30, 16 << 30))
            self.assertEqual((p.kwargs["capacity_bytes"], p.kwargs["reserve_bytes"], p.kwargs["stage_bytes"],
                              p.kwargs["snapshot_cache_bytes"]), (16 << 30, 16 << 30, 32 << 20, 7))
            self.assertTrue(all(t.kwargs["state_format"] == "fmt" and t.kwargs["mapped_staging"] and t.kwargs["block_bytes"] == 4
                                for t in (c, p)))
            self.assertEqual((c.swept, p.swept), (1, 1), "each swept of what a killed write left, once a boot")

    def test_an_owner_keeps_its_own_directory_and_a_change_of_hands_empties_it(self):
        from engine.base import tenancy
        from engine.base.tiered_kv import open_tiers
        with tempfile.TemporaryDirectory() as d:
            rank = Path(d) / "rank0"
            (rank / "prefix").mkdir(parents=True)
            (rank / "manifest.json").write_text("{}")
            tenancy.claim(rank, "production/srv2/4242")                   # what production parked here
            _, _, left = open_tiers(self.pool(), 4, d, 0, state_format="qwen38", owner="production/srv2/4242",
                                    background=False, make=FakeNvme)
            self.assertIsNone(left)
            self.assertTrue((rank / "manifest.json").exists(), "the same owner switching models keeps the directory")
            _, _, left = open_tiers(self.pool(), 4, d, 0, state_format="qwen38", owner="session/qwen38-window",
                                    background=False, make=FakeNvme)
            self.assertEqual(left, "production/srv2/4242")
            self.assertEqual(tenancy.held_by(rank), "session/qwen38-window")
            self.assertFalse((rank / "manifest.json").exists(), "a window does not inherit what production parked")

    def test_a_sweep_that_fails_does_not_fail_the_boot(self):
        from engine.base.tiered_kv import open_tiers

        class Refusing(FakeNvme):
            def cleanup(self):
                super().cleanup()
                raise OSError("read-only file system")

        with tempfile.TemporaryDirectory() as d:
            conversations, prefix, _ = open_tiers(self.pool(), 4, d, 1, state_format="fmt", background=False,
                                                  make=Refusing)
            self.assertEqual((conversations.tier.swept, prefix.tier.swept), (1, 1))

    def test_every_profile_parks_under_the_same_root_and_caps(self):
        """GLM-5.3 and Qwen3.8 share one tier root: the one root both launchers default to, which is the one under
        the directory their rank containers bind."""
        from engine.base import tiered_kv
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(tiered_kv.TIER_ROOT, "/home/choiceoh/glm53-logs/st-tier")
        self.assertEqual((tiered_kv.TIER_GIB, tiered_kv.PREFIX_TIER_GIB, tiered_kv.TIER_RESERVE_GIB), (64.0, 16.0, 16.0))
        for launcher in ("start-st-glm53.sh", "start-st-qwen38.sh"):
            text = (root / "launchers" / launcher).read_text()
            self.assertIn("MOUNTED_ROOT=/home/choiceoh/glm53-logs\n", text, launcher)
            self.assertIn("TIER_DIR=${ST_TIER_DIR:-$MOUNTED_ROOT/st-tier}", text, launcher)
            self.assertIn("-v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs ", text, launcher)
        for boot in ("engine/profiles/glm53/boot.py", "engine/profiles/qwen38/fleet.py"):
            self.assertIn('"--tier-dir", default=TIER_ROOT', (root / boot).read_text(), boot)


def _serve():
    """test_engine_serve's fake engine and `server()`: the real scheduler, runner and door over this file's tier."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import test_engine_serve
    return test_engine_serve


class HeldRecords(MemoryTier):
    """A tier whose next record read can be held once it has its bytes: the continuation scan's read on a request's
    thread (serve.py `_continuation`, off the server's lock), overtaken by the loop before it keeps what it read."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hold = None                            # an Event the next read waits on, its record in hand
        self.reading = threading.Event()            # that read has its record and is waiting
        self.files_go = False                       # ... or has only seen that the file exists, as NvmeTier.record does first

    def record(self, seq):
        record = super().record(seq)
        hold, self.hold = self.hold, None
        if hold is not None:
            self.reading.set()
            if not hold.wait(5):
                raise TimeoutError("the test never let the read go")
            if self.files_go and seq not in self.records:
                raise FileNotFoundError(f"seq-{seq}.json went before it was read")
        return record


class Scan:
    """`fn(*args)` on a thread of its own, as a request runs it, stopped inside its record read until `finish`."""

    def __init__(self, tier, fn, *args):
        self.result = self.error = None
        self.hold = threading.Event()
        tier.reading.clear()
        tier.hold = self.hold

        def run():
            try:
                self.result = fn(*args)
            except BaseException as exc:            # noqa: BLE001 -- `finish` raises it in the test
                self.error = exc
        self.thread = threading.Thread(target=run, daemon=True, name="scan")
        self.thread.start()
        if not tier.reading.wait(5):
            raise AssertionError("the scan never reached the tier's record")

    def finish(self):
        """Let the read land now; what the scan made of it."""
        self.hold.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise AssertionError("the scan did not finish")
        if self.error is not None:
            raise self.error
        return self.result


def quiet(s, steps=300):
    """The loop until nothing waits, runs or is on the tier's thread."""
    for _ in range(steps):
        ran = s.once()
        if not ran and not s._waiting and not s._retiring and not s._resuming and not s._restoring:
            return
    raise AssertionError("the server did not go quiet")


def until(s, done, steps=100):
    """The loop, a step at a time, until `done()`."""
    for _ in range(steps):
        if done():
            return
        s.once()
    raise AssertionError("the loop never got there")


class ParkedRecordRaceTests(unittest.TestCase):
    """The continuation scan reads parked conversations' records on the request's thread, off the server's lock, while
    the loop resumes, parks and forgets those same conversations. The runner keeps what a read brings back -- a few
    whole records and a digest for everybody -- so admission does not read the disk on the loop (D10). A read that
    lands after the loop moved its conversation brings back something that is no longer that conversation's record,
    and keeping it made rank 0 alone believe the conversation was still parked (2026-09-19, found reading the code)."""

    def served(self):
        tier = HeldRecords()
        return _serve().server(rows=2, keep_idle=True, tier=tier), tier

    def parked(self, s):
        """A conversation that finished its turn and parked, its record no longer one of the few held (this fleet parks
        about 280 and keeps 8): a scan reads it from the tier. Its digest is dropped too, so that is read from the tier
        as well: a park and the boot's read leave a digest for every parked conversation, and this is the read a bare
        runner still makes. It keeps the same rule."""
        first, _ = s.submit([3, 4, 5], 2, 0)
        quiet(s)
        self.assertEqual(s.take_result(first), [5, 5])
        self.assertTrue(s.runner.is_parked(first))
        s.runner.parked.pop(first)
        s.runner.digests.pop(first)
        return first

    def test_a_record_that_lands_after_its_conversation_resumed_is_not_kept(self):
        s, tier = self.served()
        first = self.parked(s)
        scan = Scan(tier, s.runner.parked_digest, first)          # the scan has the record in hand ...
        turn, _ = s.submit([6], 2, 0, conversation=first)
        until(s, lambda: first in s._conversations)                # ... when the loop resumes the conversation
        self.assertIsNone(scan.finish(), "a conversation that went while it was read has no digest")
        self.assertFalse(s.runner.is_parked(first))
        self.assertNotIn(first, s.runner.parked)
        self.assertNotIn(first, s.runner.digests)
        quiet(s)                                                   # the turn ends and the conversation parks again
        self.assertEqual(s.take_result(turn), [6, 6])
        self.assertEqual(s.runner.parked_keys(), [first], "not refused as 'already parked' and dropped")
        self.assertEqual(s.runner.parked_record(first)["tokens"], [3, 4, 5, 5, 5, 6])

    def test_a_record_that_lands_after_its_conversation_was_forgotten_is_not_kept(self):
        s, tier = self.served()
        first = self.parked(s)
        scan = Scan(tier, s.runner.parked_record, first)
        self.assertEqual(s.runner.forget_oldest_parked(), first)   # the tier needed room
        self.assertIsNone(scan.finish())
        self.assertFalse(s.runner.is_parked(first))
        self.assertNotIn(first, s.runner.parked)
        again, _ = s.submit([6], 1, 0, conversation=first)         # a turn naming it is refused, as every rank refuses it
        quiet(s)
        with self.assertRaises(_serve().RequestError) as refused:
            s.take_result(again)
        self.assertEqual(refused.exception.status, 409)

    def test_a_stale_record_does_not_replace_the_one_its_next_park_kept(self):
        """The widest window: a whole turn resumes, runs and parks again while one read of the old record is still out.
        Kept, the old record is what the next resume hands the model -- a history a turn shorter than the KV it comes
        back with -- on rank 0 alone, the only rank that scans."""
        s, tier = self.served()
        first = self.parked(s)
        scan = Scan(tier, s.runner.parked_digest, first)
        turn, _ = s.submit([6], 2, 0, conversation=first)
        quiet(s)
        self.assertEqual(s.take_result(turn), [6, 6])
        self.assertIsNone(scan.finish())
        self.assertEqual(s.runner.parked_record(first)["tokens"], [3, 4, 5, 5, 5, 6])
        self.assertEqual(s.runner.parked_digest(first)["tokens"], 6)
        third, _ = s.submit([7], 1, 0, conversation=first)
        quiet(s)
        self.assertEqual(s.take_result(third), [7])
        self.assertEqual(s.runner.parked_record(first)["tokens"], [3, 4, 5, 5, 5, 6, 6, 6, 7], "the second turn is in it")

    def test_a_park_does_not_keep_a_digest_from_before_it(self):
        s, tier = self.served()
        first = self.parked(s)
        before = s.runner.parked_digest(first)
        self.assertEqual(before["tokens"], 3)
        turn, _ = s.submit([6], 2, 0, conversation=first)
        until(s, lambda: first in s._conversations)
        s.runner.digests[first] = before                           # what a scan landing after the resume used to leave
        quiet(s)
        self.assertEqual(s.take_result(turn), [6, 6])
        self.assertEqual(s.runner.parked_digest(first)["tokens"], 6)

    def test_what_is_parked_is_the_tiers_word_not_the_record_cache(self):
        """Only rank 0 scans, so only rank 0's cache can hold a record the tier no longer has. Whether a conversation
        is parked decides admission and the park/resume guards on every rank, so it is what every rank knows alike."""
        s, tier = self.served()
        first = self.parked(s)
        prompt = s.runner.parked_record(first)["tokens"] + [7]
        self.assertIn(first, s.runner.parked)
        self.assertIsNotNone(s.runner.parked_digest(first))
        s.runner.tiered.forget(first)                              # gone from the tier, still in this rank's cache
        self.assertFalse(s.runner.is_parked(first))
        self.assertIsNone(s.runner.parked_record(first))
        # Its digest is read lock-free and can answer until the runner's own half of the move: a scan that listed the
        # conversation just before offers nothing on it, since the record behind the digest is the tier's word.
        with patch.object(s.runner, "parked_keys", return_value=[first]):
            self.assertIsNone(s._continuation(prompt))

    def test_a_read_is_not_kept_when_the_tier_let_its_conversation_go_meanwhile(self):
        s, tier = self.served()
        first = self.parked(s)
        scan = Scan(tier, s.runner.parked_record, first)
        s.runner.tiered.forget(first)                              # the tier's half of a resume, before the runner's
        self.assertIsNone(scan.finish())
        self.assertNotIn(first, s.runner.parked)

    def test_a_read_that_fails_because_its_conversation_went_is_a_miss_not_an_error(self):
        """NvmeTier.record sees the file, then reads it; the loop can forget the conversation, file and all, in
        between. The scan asked about a conversation that is gone, which is a miss. A read that fails while the
        conversation is still parked is still an error."""
        s, tier = self.served()
        tier.files_go = True
        first = self.parked(s)
        scan = Scan(tier, s.runner.parked_digest, first)
        s.runner.forget_parked(first)
        self.assertIsNone(scan.finish())
        second = self.parked(s)
        scan = Scan(tier, s.runner.parked_record, second)
        tier.records.pop(second)                                   # the file went, the conversation did not
        with self.assertRaises(FileNotFoundError):
            scan.finish()
        self.assertTrue(s.runner.is_parked(second))
        self.assertNotIn(second, s.runner.parked)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch for LocalTP")
    def test_four_ranks_keep_the_conversation_rank_0_was_reading_when_it_resumed(self):
        """Before: the read landed on rank 0 after the resume, rank 0 alone refused the next park ('already parked'),
        the vote dropped the conversation on every rank, and the turn after it went down the parked path on rank 0 and
        the 409 path on the other three -- rank 0 then died on the tier entry it did not have."""
        from engine.base.comm import LocalTP
        T = _serve()
        seen = {}

        def rank_main(comm, _):
            tier = HeldRecords()
            s = T.server(comm=comm, rows=2, keep_idle=True, tier=tier)
            first = turn = third = scan = None
            if comm.rank == 0:
                first, _ = s.submit([3, 4, 5], 2, 0)
            for _ in range(20):
                s.once()
            if comm.rank == 0:
                seen["first"] = s.take_result(first)
                s.runner.parked.pop(first)                         # no longer one of the few records held
                scan = Scan(tier, s._continuation, [3, 4, 5, 5, 5, 6], ())
                turn, _ = s.submit([6], 2, 0, conversation=first)
            for _ in range(30):
                s.once()
                if scan is not None and first in s._conversations:
                    seen["scan"] = scan.finish()                   # after the resume, before the turn parks again
                    scan = None
            if comm.rank == 0:
                seen["turn"] = s.take_result(turn)
                third, _ = s.submit([7], 1, 0, conversation=first)
            for _ in range(30):
                s.once()
            if comm.rank == 0:
                seen["third"] = s.take_result(third)
                s.alive = False
            s.once()
            return s.runner.parked_keys(), s.runner.parked_record(0)["tokens"], s.served

        out = LocalTP(4, timeout_s=30).run(rank_main, None)
        self.assertTrue(all(row == out[0] for row in out), out)
        self.assertEqual(out[0], ([0], [3, 4, 5, 5, 5, 6, 6, 6, 7], 3))
        self.assertEqual((seen["first"], seen["turn"], seen["third"]), ([5, 5], [6, 6], [7]))
        self.assertIsNone(seen["scan"])


class WatchedRecords(MemoryTier):
    """A tier that tells the record reads apart: on the tier's thread (a job `run_async` runs -- inline here, as
    MemoryTier's are without a gate), by the step loop itself (`in_loop`, set while `once()` runs), or by anybody else
    (the continuation scan in `submit`, the boot's reconciliation, the test)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reads, self.tier_reads, self.loop_reads = [], [], []
        self.in_loop = False
        self._tier_thread = threading.local()
        self.slow = None                            # an Event a read on the tier's thread waits for

    def run_async(self, fn, *args):
        def on_the_tier(*a):
            self._tier_thread.yes = True
            try:
                return fn(*a)
            finally:
                self._tier_thread.yes = False
        return super().run_async(on_the_tier, *args)

    def record(self, seq):
        self.reads.append(seq)
        if getattr(self._tier_thread, "yes", False):
            self.tier_reads.append(seq)
            if self.slow is not None and not self.slow.wait(5):
                raise TimeoutError("the test never let the record read go")
        elif self.in_loop:
            self.loop_reads.append(seq)
        return super().record(seq)


def looped(s, tier, steps=300):
    """`quiet`, with the loop's own record reads watched."""
    tier.in_loop = True
    try:
        quiet(s, steps)
    finally:
        tier.in_loop = False


class ParkedRecordBoundTests(unittest.TestCase):
    """`Runner.parked` holds a few whole records (PARKED_RECORDS_KEPT) -- a record is the conversation's whole token list,
    36 B a token -- and a digest for every parked conversation. Until 2026-09-19 a park put its record in untrimmed, so
    every rank held every conversation this process parked. The bound is only worth having if the loop never reads a
    record off the disk instead (D10): admission and the hint's re-check read the digest, a resume reads a record it
    does not hold on the tier's thread, and the boot's one read of each record leaves its digest behind."""

    def served(self, **kwargs):
        tier = WatchedRecords(**kwargs)
        return _serve().server(rows=2, keep_idle=True, tier=tier), tier

    def park(self, s, tier, prompt):
        request, _ = s.submit(prompt, 2, 0)
        looped(s, tier)
        self.assertEqual(s.take_result(request), prompt[-1:] * 2)
        self.assertTrue(s.runner.is_parked(request))
        return request

    def test_a_park_keeps_a_few_records_and_a_digest_for_every_conversation(self):
        s, tier = self.served()
        kept = s.runner.PARKED_RECORDS_KEPT
        keys = [self.park(s, tier, [3 + i, 4, 5]) for i in range(3 * kept)]
        self.assertEqual(s.runner.parked_keys(), keys)
        self.assertEqual(list(s.runner.parked), keys[-kept:], "the most recently parked, the oldest first to go")
        self.assertEqual(sorted(s.runner.digests), keys)
        for key in keys:
            record, summary = tier.records[key], s.runner.parked_summary(key)
            self.assertEqual((summary["context"], summary["pending"]), (record["context"], record["pending"]))
            self.assertEqual((summary["tokens"], summary["last"], summary["prev"]),
                             (len(record["tokens"]), record["tokens"][-1], record["tokens"][-2]))
        self.assertEqual(tier.reads, [], "a park reads nothing back")

    def test_a_conversation_whose_record_went_comes_back_without_the_loop_reading_the_disk(self):
        s, tier = self.served()
        first = self.park(s, tier, [3, 4, 5])
        for i in range(s.runner.PARKED_RECORDS_KEPT):
            self.park(s, tier, [9, 10 + i])
        self.assertNotIn(first, s.runner.parked)
        turn, _ = s.submit([6], 2, 0, conversation=first)                  # named: no scan read it for the loop
        looped(s, tier)
        self.assertEqual(s.take_result(turn), [6, 6])
        self.assertEqual(tier.loop_reads, [], "admission sized it from the digest")
        self.assertEqual(tier.tier_reads, [first], "the resume read its record beside the blocks")
        self.assertEqual(s.runner.parked_record(first)["tokens"], [3, 4, 5, 5, 5, 6])

    def test_a_hint_on_a_conversation_whose_record_went_is_checked_against_its_digest(self):
        s, tier = self.served()
        first = self.park(s, tier, [3, 4, 5])
        for i in range(s.runner.PARKED_RECORDS_KEPT):
            self.park(s, tier, [9, 10 + i])
        prompt = tier.records[first]["tokens"] + [7]
        again, _ = s.submit(prompt, 2, 0, continue_history=True)          # rank 0's scan reads the record on this thread
        self.assertEqual(tier.reads, [first])
        s.runner.parked.pop(first)                                         # as if eight more parks came first
        looped(s, tier)
        self.assertEqual(s.take_result(again), [7, 7])
        self.assertEqual(s.continuation_fallbacks, {}, "the hint held")
        self.assertFalse(s.runner.is_parked(again), "it continued `first` rather than starting its own")
        self.assertEqual(tier.loop_reads, [])
        self.assertEqual(tier.tier_reads, [first])
        history = s.runner.parked_record(first)["tokens"]                  # the second turn's, parked again
        self.assertEqual(history, [3, 4, 5, 5, 5, 7])
        s.runner.parked.pop(first)
        self.assertIsNone(s._stale_hint(first, history + [8], [], 6, False))
        other = [3, 9, 5, 5, 5, 7, 8]                                      # the same length and last two ids, another history
        self.assertEqual(s._stale_hint(first, other, [], 6, False), "changed")
        self.assertEqual(tier.reads, [first, first], "the re-checks read no record")

    def test_conversations_an_earlier_process_parked_are_read_once_at_boot_and_never_on_the_loop(self):
        s, tier = self.served()
        keys = [self.park(s, tier, [3 + i, 4, 5]) for i in range(3)]
        seen = len(tier.reads)
        s = _serve().server(rows=2, keep_idle=True, tier=tier)            # the next process, over the same tier
        self.assertEqual(sorted(tier.reads[seen:]), keys, "the reconciliation's read, once each")
        self.assertEqual(sorted(s.runner.digests), keys)
        self.assertEqual(len(s.runner.parked), 0, "no record held: the few are the most recently parked")
        seen = len(tier.reads)
        self.assertIsNone(s._continuation([99, 98, 97, 96]))
        self.assertEqual(tier.reads[seen:], [], "the first scan reads no record to build digests")
        turn, _ = s.submit([6], 2, 0, conversation=keys[0])
        looped(s, tier)
        self.assertEqual(s.take_result(turn), [6, 6])
        self.assertEqual(tier.loop_reads, [])
        self.assertEqual(tier.tier_reads, [keys[0]])

    def test_a_resume_whose_record_cannot_be_read_gives_everything_back(self):
        s, tier = self.served()
        first = self.park(s, tier, [3, 4, 5])
        for i in range(s.runner.PARKED_RECORDS_KEPT):
            self.park(s, tier, [9, 10 + i])
        free = (s.runner.kv.available, s.runner.slots.available, sorted(s._free_rows))
        tier.records.pop(first)                                            # the record file went, the blocks did not
        turn, _ = s.submit([6], 2, 0, conversation=first)
        looped(s, tier)
        with self.assertRaises(_serve().RequestError) as refused:
            s.take_result(turn)
        self.assertEqual(refused.exception.status, 503)
        self.assertFalse(s.runner.is_parked(first), "dropped everywhere: a rank could not read it back")
        self.assertNotIn(first, s.runner.digests)
        self.assertEqual((s.runner.kv.available, s.runner.slots.available, sorted(s._free_rows)), free)
        self.assertFalse(s.runner.resuming or s.runner._resume_reads or s.runner.tiered.inflight)

    def test_a_resume_is_not_done_before_its_record_is_read(self):
        """The loop does not wait for the read (D10): it votes the transfer not done and looks again next step."""
        opened = threading.Event()
        opened.set()
        s, tier = self.served(gate=opened)                                 # every tier job on a thread of its own

        def meanwhile(done, seconds=10):
            deadline = time.monotonic() + seconds
            while not done():
                if time.monotonic() > deadline:
                    raise AssertionError("the loop never got there")
                s.once()
                time.sleep(0.0005)

        first, _ = s.submit([3, 4, 5], 2, 0)
        meanwhile(lambda: first in s.results and not s._retiring)
        self.assertEqual(s.take_result(first), [5, 5])
        s.runner.parked.pop(first)
        tier.slow = threading.Event()
        turn, _ = s.submit([6], 2, 0, conversation=first)
        meanwhile(lambda: s._resuming and s.runner.tiered.done(next(iter(s._resuming))))
        row = next(iter(s._resuming))
        for _ in range(20):
            s.once()
        self.assertEqual(list(s._resuming), [row], "the blocks are back, the record is not: still resuming")
        self.assertFalse(s.runner.transfer_done(row))
        tier.slow.set()
        meanwhile(lambda: turn in s.results and not s._resuming and not s._retiring)
        self.assertEqual(s.take_result(turn), [6, 6])
        self.assertEqual(tier.tier_reads, [first])

    def test_the_digest_answers_a_hint_as_the_history_does(self):
        """`_offer_digest` is `_offer` with the history before its last token compared by its hash: the same answer for
        every prompt -- a proper prefix, one short of an end token, a changed middle token, pictures on either side of
        the cut."""
        import random
        from engine.base.runner import Runner
        Server = _serve().Server
        rng = random.Random(7)
        ends = {0}
        for _ in range(4000):
            history = [rng.randrange(4) for _ in range(rng.randrange(9))]
            tail = [rng.randrange(4) for _ in range(rng.randrange(4))]
            kind = rng.randrange(4)
            if kind == 0:
                ids = history + tail
            elif kind == 1:
                ids = history[:-1] + tail
            elif kind == 2 and history:
                ids = list(history) + tail
                ids[rng.randrange(len(history))] = rng.randrange(4)
            else:
                ids = [rng.randrange(4) for _ in range(rng.randrange(10))]
            held = [["image", rng.choice("ab"), p, 1, [1, 2, 2]] for p in sorted(rng.sample(range(9), rng.randrange(3)))
                    if p < len(history)]
            marks = sorted((p, rng.choice("ab")) for p in rng.sample(range(12), rng.randrange(3)))
            if rng.randrange(2):
                marks = sorted(set(marks) | {(r[2], r[1]) for r in held})
            record = {"context": len(history), "pending": 0, "tokens": history, "media": held}
            want = Server._offer(ids, marks, history, [(r[2], r[1]) for r in held], ends)
            digest = Runner._history(Runner._summary(record))
            got = None if digest is None else Server._offer_digest(ids, marks, digest, ends)
            self.assertEqual(got, want, (history, ids, marks, held))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch for LocalTP")
    def test_four_ranks_continue_a_conversation_whose_record_went_and_no_loop_reads_the_disk(self):
        """Ranks 1-3 never scan, so nothing ever put the record back in their memory: before, they read it on the loop."""
        from engine.base.comm import LocalTP
        T = _serve()
        from engine.base.runner import Runner
        kept = Runner.PARKED_RECORDS_KEPT
        seen = {}

        def rank_main(comm, _):
            tier = WatchedRecords()
            s = T.server(comm=comm, rows=2, keep_idle=True, tier=tier)
            first = again = None
            for i in range(kept + 1):
                if comm.rank == 0:
                    request, _ = s.submit([3, 4, 5] if i == 0 else [9, 10 + i], 2, 0)
                    first = request if i == 0 else first
                tier.in_loop = True
                for _ in range(20):
                    s.once()
                tier.in_loop = False
            if comm.rank == 0:
                again, _ = s.submit([3, 4, 5, 7], 2, 0, continue_history=True)
                seen["scan"] = list(tier.reads)
                s.runner.parked.pop(0)                                    # as if more parks came first
            tier.in_loop = True
            for _ in range(30):
                s.once()
            tier.in_loop = False
            if comm.rank == 0:
                seen["again"] = s.take_result(again)
                seen["fallbacks"] = dict(s.continuation_fallbacks)
                s.alive = False
            s.once()
            return (s.runner.parked_keys(), len(s.runner.parked) <= kept, sorted(s.runner.digests),
                    s.runner.digests[0]["tokens"], tier.loop_reads, tier.tier_reads)

        out = LocalTP(4, timeout_s=30).run(rank_main, None)
        self.assertEqual(seen["scan"], [0], "rank 0's scan read the record, on the request's thread")
        self.assertEqual((seen["again"], seen["fallbacks"]), ([7, 7], {}))
        self.assertTrue(all(row == out[0] for row in out), out)
        keys, bounded, digests, tokens, loop_reads, tier_reads = out[0]
        self.assertEqual(keys, list(range(kept + 1)), "conversation 0 continued and parked again, nothing new")
        self.assertTrue(bounded)
        self.assertEqual(digests, keys)
        self.assertEqual(tokens, 6, "the digest of the history the second turn left")
        self.assertEqual((loop_reads, tier_reads), ([], [0]), "every rank read the record on the tier's thread")


if __name__ == "__main__":
    unittest.main()
