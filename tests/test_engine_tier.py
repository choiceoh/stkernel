"""Tier ownership and asynchronous error handling, without CUDA or real KV."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.base.kv import BlockPool
from engine.base.kv_tier import NvmeTier, SECTOR
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

    def test_foreign_layout_cannot_be_replaced_and_cleanup_preserves_its_files(self):
        with tempfile.TemporaryDirectory() as d:
            tier = self.tier(d)
            active = tier.dir / ('seq-0-' + 'a' * 32 + '.kv')
            retired = tier.dir / ('seq-0-' + 'b' * 32 + '.kv')
            active.write_bytes(b'active'); retired.write_bytes(b'retired')
            tier._save_manifest({'0': {'file': active.name, 'retired': [retired.name], 'block_bytes': 2 * SECTOR}})
            self.assertFalse(tier.has(0))
            self.assertEqual(tier.stale(), ['0'])
            with self.assertRaisesRegex(ValueError, 'different block layout'):
                tier.demote(0, None, [], 1)
            tier.cleanup()
            self.assertEqual(active.read_bytes(), b'active')
            self.assertEqual(retired.read_bytes(), b'retired')

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


if __name__ == "__main__":
    unittest.main()
