"""Tier ownership and asynchronous error handling, without CUDA or real KV."""
from __future__ import annotations

import json
import tempfile
import threading
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
    """Byte storage with failures after partial I/O, exercising real pool ownership."""
    block_bytes = 4

    def __init__(self):
        self.index = {}
        self.data = {}
        self.fail_demote = self.fail_promote = self.fail_forget = False

    def has(self, seq):
        return str(seq) in self.index

    def demote(self, seq, storage, ids, tokens):
        if self.fail_demote:
            raise OSError("disk write failed")
        data = b"".join(storage[i * 4:(i + 1) * 4] for i in ids)
        self.data[seq] = data
        self.index[str(seq)] = {"tokens": tokens, "blocks": len(ids), "bytes": len(data)}
        return len(data)

    def promote(self, seq, storage, ids):
        data = self.data[seq]
        if len(ids) != self.index[str(seq)]["blocks"]:
            raise ValueError("wrong block count")
        for j, i in enumerate(ids):
            storage[i * 4:(i + 1) * 4] = data[j * 4:(j + 1) * 4]
            if self.fail_promote:
                raise OSError("disk read failed")
        return len(data)

    def forget(self, seq):
        if self.fail_forget:
            raise OSError("manifest write failed")
        self.index.pop(str(seq))
        self.data.pop(seq)


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


class NvmeControlTests(unittest.TestCase):
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
