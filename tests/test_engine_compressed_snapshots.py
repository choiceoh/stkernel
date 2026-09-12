"""Lossless bytes, bounded ownership, and the real tier I/O path on CPU tensors."""
from contextlib import ExitStack, nullcontext
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch
import zlib

from engine.base.compressed_snapshots import CHUNK_BYTES, CompressedSnapshots
from engine.base.kv_tier import NvmeTier, SECTOR


def pack(cache, key, raw):
    builder = cache.begin(len(raw))
    for off in range(0, len(raw), 170003):  # deliberately non-sector-aligned input windows
        builder.add(memoryview(raw)[off:off + 170003])
    result = builder.finish()
    cache.publish(key, result)
    return result


def unpack(snapshot):
    out, stage = bytearray(snapshot.raw_bytes), bytearray(CHUNK_BYTES)
    def write(off, n):
        out[off:off+n] = stage[:n]
    snapshot.restore(stage, write)
    return bytes(out)


class CompressionTests(unittest.TestCase):
    def test_roundtrip_preserves_every_bit_including_float_special_values(self):
        # Signed zero, infinities and distinct NaN payloads are bytes, never converted.
        raw = bytes.fromhex('00000000000000800000807f000080ff0100c07fff7f0080') * 100003
        cache = CompressedSnapshots(len(raw))
        entry = pack(cache, 'generation-a', raw)
        self.assertLess(cache.stored_bytes, len(raw))
        self.assertEqual(unpack(entry), raw)
        self.assertEqual(unpack(cache.get('generation-a')), raw)

    def test_incompressible_data_is_not_kept_as_a_raw_copy(self):
        cache = CompressedSnapshots(2 * CHUNK_BYTES)
        self.assertIsNone(pack(cache, 'noise', random.Random(51).randbytes(CHUNK_BYTES)))
        self.assertEqual((cache.stored_bytes, len(cache.entries), cache.rejections), (0, 0, 1))

    def test_restore_batches_chunks_into_the_shared_stage_before_upload(self):
        raw = b'z' * (2 * CHUNK_BYTES + 17)
        builder = CompressedSnapshots(len(raw)).begin(len(raw))
        builder.add(raw)
        out, stage, writes = bytearray(len(raw)), bytearray(2 * CHUNK_BYTES), []
        def upload(off, n):
            writes.append((off, n))
            out[off:off+n] = stage[:n]
        builder.finish().restore(stage, upload)
        self.assertEqual(writes, [(0, 2 * CHUNK_BYTES), (2 * CHUNK_BYTES, 17)])
        self.assertEqual(out, raw)

    def test_lru_and_pending_chunks_obey_the_same_cap(self):
        raw = b'abcde' * 5000
        probe = pack(CompressedSnapshots(len(raw)), 'probe', raw)
        cache = CompressedSnapshots(2 * probe.stored_bytes + 1)
        pack(cache, 'a', raw)
        pack(cache, 'b', raw)
        cache.get('a')
        builder = cache.begin(len(raw))
        builder.add(raw)
        self.assertLessEqual(cache.stored_bytes + builder.stored, cache.capacity_bytes)
        cache.publish('c', builder.finish())
        self.assertEqual(list(cache.entries), ['a', 'c'])
        self.assertEqual(cache.raw_bytes, 2 * len(raw))
        self.assertEqual(cache.evictions, 1)
        cache.discard('a')
        self.assertEqual(cache.raw_bytes, len(raw))
        self.assertGreater(cache.clear(), 0)
        self.assertEqual((cache.stored_bytes, cache.raw_bytes, cache.clear()), (0, 0, 0))

    def test_tiny_capacity_rejects_and_incomplete_input_cannot_publish(self):
        cache = CompressedSnapshots(600)
        self.assertIsNone(pack(cache, 'large', b'abc' * 5000))
        builder = CompressedSnapshots(10000).begin(9999)
        builder.add(b'a' * 2000)
        with self.assertRaisesRegex(ValueError, 'input length'):
            builder.finish()

    def test_corrupt_truncated_extra_and_oversized_payloads_are_rejected(self):
        entry = pack(CompressedSnapshots(100000), 'a', b'abcd' * 20000)
        n, data = entry.chunks[0]
        bad = [data[:-1], data + b'junk', bytes([data[0] ^ 0xff]) + data[1:], zlib.compress(b'x' * (n + 1))]
        for payload in bad:
            with self.subTest(payload=len(payload)), self.assertRaises((ValueError, zlib.error)):
                unpack(replace(entry, chunks=((n, payload),)))


@unittest.skipUnless(importlib.util.find_spec('torch') and hasattr(os, 'preadv'), 'needs CPU torch and Linux I/O')
class TierCompressionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        directory = self.context.enter_context(tempfile.TemporaryDirectory())
        self.context.enter_context(patch.object(os, 'O_DIRECT', 0, create=True))
        self.context.enter_context(patch('torch.cuda.current_stream', return_value=None))
        self.context.enter_context(patch('torch.cuda.stream', side_effect=lambda _: nullcontext()))
        self.tier = self.make_tier(directory)
        self.storage = torch.arange(4 * SECTOR, dtype=torch.int64).to(torch.uint8)
        self.raw = torch.tensor(list(bytes(range(256)) * 900 + b'end'), dtype=torch.uint8)
        self.keep = self.raw.clone()

    def make_tier(self, directory):
        from types import SimpleNamespace
        tier = NvmeTier.__new__(NvmeTier)
        tier.dir = Path(directory)
        tier.block_bytes = SECTOR
        tier.stage_bytes = 16 * SECTOR
        tier.per = 16
        tier.stage_t = self.torch.empty(tier.stage_bytes, dtype=self.torch.uint8)
        tier.stage = memoryview(tier.stage_t.numpy())
        tier.scratch = self.torch.empty_like(tier.stage_t)
        tier.stream = SimpleNamespace(device='cpu', wait_stream=lambda _: None, synchronize=lambda: None)
        tier.capacity_bytes, tier.reserve_bytes = 1 << 26, 0
        tier.manifest = tier.dir / 'manifest.json'
        import json
        tier.index = json.loads(tier.manifest.read_text()) if tier.manifest.exists() else {}
        tier.lock, tier._transfer_lock = threading.Lock(), threading.Lock()
        tier.bytes_read = tier.bytes_written = 0
        tier.snapshot_cache = CompressedSnapshots(1 << 20)
        self.addCleanup(tier.close)
        return tier

    def demote(self):
        return self.tier.demote(7, self.storage, [2, 0], 100, self.raw, {'hash': 'full-boundary-hash'})

    def test_faded_restore_uses_shared_buffer_and_opens_no_file(self):
        self.demote()
        self.raw.zero_()
        with patch('engine.base.kv_tier.os.open', side_effect=AssertionError('cache hit opened disk')):
            self.assertEqual(self.tier.promote(7, self.storage, None, self.raw), 0)
        self.assertTrue(self.torch.equal(self.raw, self.keep))
        self.assertEqual(self.tier.bytes_read, 0)
        self.assertEqual(self.tier.snapshot_cache.hits, 1)

    def test_async_worker_owns_compression_and_release_returns_the_cache(self):
        caller = threading.get_ident()
        threads = []
        begin = self.tier.snapshot_cache.begin
        def capture(n):
            threads.append(threading.get_ident())
            return begin(n)
        with patch.object(self.tier.snapshot_cache, 'begin', side_effect=capture):
            self.tier.run_async(self.demote).result(timeout=5)
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], caller)
        self.raw.zero_()
        self.tier.run_async(self.tier.promote, 7, self.storage, None, self.raw).result(timeout=5)
        self.assertTrue(self.torch.equal(self.raw, self.keep))
        self.assertGreater(self.tier.close(), 2 * self.tier.stage_bytes)
        self.assertEqual((self.tier.snapshot_cache.stored_bytes, self.tier.close()), (0, 0))

    def test_full_restore_reads_blocks_but_uses_compressed_snapshot(self):
        self.demote()
        expected = self.storage.view(-1, SECTOR)[[2, 0]].clone()
        self.storage.zero_()
        self.raw.zero_()
        self.assertEqual(self.tier.promote(7, self.storage, [1, 3], self.raw), 2 * SECTOR)
        self.assertTrue(self.torch.equal(self.raw, self.keep))
        self.assertTrue(self.torch.equal(self.storage.view(-1, SECTOR)[[1, 3]], expected))

    def test_eviction_restart_and_untiered_legacy_reads_remain_exact(self):
        wrote = self.demote()
        self.tier.snapshot_cache.clear()
        self.raw.zero_()
        self.assertEqual(self.tier.promote(7, self.storage, [1, 3], self.raw), wrote)
        self.assertTrue(self.torch.equal(self.raw, self.keep))
        reopened = self.make_tier(self.tier.dir)
        reopened.snapshot_cache = None
        self.raw.zero_()
        self.assertGreater(reopened.promote(7, self.storage, None, self.raw), 0)
        self.assertTrue(self.torch.equal(self.raw, self.keep))

    def test_replacement_forget_and_failed_publish_cannot_serve_stale_bytes(self):
        self.demote()
        old = self.tier._path(7).name
        self.raw.fill_(99)
        with patch.object(self.tier, '_publish', side_effect=OSError('manifest failed')):
            with self.assertRaisesRegex(OSError, 'manifest failed'):
                self.demote()
        self.tier.promote(7, self.storage, None, self.raw)
        self.assertTrue(self.torch.equal(self.raw, self.keep))
        self.raw.fill_(99)
        self.demote()
        self.assertNotIn(old, self.tier.snapshot_cache.entries)
        self.raw.zero_()
        self.tier.promote(7, self.storage, None, self.raw)
        self.assertTrue(bool((self.raw == 99).all()))
        self.tier.forget(7)
        self.assertEqual(self.tier.snapshot_cache.stored_bytes, 0)
        self.assertFalse(self.tier.has(7))


if __name__ == '__main__':
    unittest.main()
