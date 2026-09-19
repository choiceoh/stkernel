"""Lossless mapped-window I/O and unique-allocation accounting, CPU oracle."""
from contextlib import contextmanager, nullcontext
import os
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

from engine.base.kv_tier import NvmeTier, Segments


@contextmanager
def cpu_tier_io():
    """NvmeTier on the CPU with real O_DIRECT files: its pinned staging and device scratch are page-aligned host
    tensors, and its stream is one that is always done."""
    empty = torch.empty

    def aligned(n):
        raw = empty(n+4095, dtype=torch.uint8)
        start = (-raw.data_ptr()) % 4096
        return raw[start:start+n]

    def host_empty(*args, **kwargs):
        if kwargs.pop("pin_memory", False):
            return aligned(args[0])
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    def aliases(n):
        host = aligned(n)
        return host, host.view_as(host)
    stream = NS(wait_stream=lambda other: None, synchronize=lambda: None)
    with (patch("torch.empty", side_effect=host_empty),
          patch("torch.cuda.Stream", return_value=stream),
          patch("torch.cuda.current_stream", return_value=stream),
          patch("torch.cuda.stream", side_effect=lambda s: nullcontext()),
          patch("engine.kernels.mapped_staging.allocate", side_effect=aliases)):
        yield


def round_trip(case, tier, device):
    block = tier.block_bytes
    source = (torch.arange(9*block, device=device) % 251).to(torch.uint8)
    extra = (torch.arange(2*block+29, device=device) % 193).to(torch.uint8)
    expected_extra = extra.clone()
    before = source.clone().view(9, block)
    src_ids, dst_ids = [7, 1, 4, 0, 8], [2, 6, 3, 5, 1]
    tier.demote(12, source, src_ids, 133, extra, {"tokens": [7, 8, 9]})
    source.zero_(); extra.zero_()
    tier.promote(12, source, dst_ids, extra)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    for old, new in zip(src_ids, dst_ids):
        case.assertTrue(torch.equal(source.view(9, block)[new], before[old]))
    case.assertTrue(torch.equal(extra, expected_extra))
    case.assertEqual(tier.record(12), {"tokens": [7, 8, 9]})
    case.assertEqual(tier.index["12"]["tokens"], 133)
    # Restore slot-only state through both disk and the optional compressed cache.
    extra.zero_()
    tier.promote(12, source, None, extra)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    case.assertTrue(torch.equal(extra, expected_extra))


class MappedTierTests(unittest.TestCase):
    def test_shutdown_timeout_retains_the_worker_and_its_buffers(self):
        from concurrent.futures import Future
        from tests.test_engine_tier import make_tier
        kv = make_tier()
        pending = Future()
        pending.set_running_or_notify_cancel()
        kv.inflight[0] = ("park", 1, 16, pending)
        with patch.object(kv.tier, "close", create=True, return_value=0) as close:
            with self.assertRaisesRegex(TimeoutError, "still owns staging"):
                kv.close(timeout=.001)
            close.assert_not_called()
            self.assertIn(0, kv.inflight)
            pending.set_result(16)
            self.assertEqual(kv.close(), 0)
            close.assert_called_once()
            self.assertFalse(kv.inflight)

    @unittest.skipUnless(hasattr(os, "O_DIRECT") and hasattr(os, "pwritev"), "requires Linux O_DIRECT")
    def test_segments_are_written_back_to_back_and_read_back_into_their_views(self):
        """A parked conversation's live state (engine/modules/state_rings `live_bytes`) is many views of one slot: the
        file holds them as one byte range, and each staging window copies across their seams."""
        with cpu_tier_io(), tempfile.TemporaryDirectory() as directory:
            for mapped in (False, True):
                tier = NvmeTier(os.path.join(directory, str(mapped)), 4096, 8192, reserve_bytes=0, mapped_staging=mapped)
                try:
                    block = tier.block_bytes
                    source = (torch.arange(3*block) % 251).to(torch.uint8)
                    before = source.clone().view(3, block)
                    slot = (torch.arange(20000) % 197).to(torch.uint8)
                    views = [slot[100:3100], slot[5000:5017], slot[6000:12000], slot[19000:19029]]   # 9,046 B: two windows
                    expected = torch.cat(views)
                    tier.demote(3, source, [2, 0], 5, Segments(views), {"context": 5, "pending": 1})
                    self.assertEqual(tier.index["3"]["extra"], 9046)
                    flat = torch.zeros(9046, dtype=torch.uint8)
                    tier.promote(3, source, None, flat)                    # one view reads them back to back
                    self.assertTrue(torch.equal(flat, expected))
                    source.zero_(); slot.zero_()
                    tier.promote(3, source, [1, 2], Segments(views))        # blocks and the pieces, each into its view
                    self.assertTrue(torch.equal(torch.cat(views), expected))
                    self.assertTrue(torch.equal(source.view(3, block)[1], before[2]))
                    self.assertTrue(torch.equal(source.view(3, block)[2], before[0]))
                    self.assertEqual(int(slot.count_nonzero()), int(expected.count_nonzero()), "nothing else written")
                    with self.assertRaisesRegex(ValueError, "slot bytes on disk"):
                        tier.promote(3, source, None, Segments(views[:-1]))
                finally:
                    tier.close()

    @unittest.skipUnless(hasattr(os, "O_DIRECT") and hasattr(os, "pwritev"), "requires Linux O_DIRECT")
    def test_real_io_through_shared_cpu_aliases_and_ordinary_buffers(self):
        with cpu_tier_io():
            for mapped, compressed in ((False, 0), (True, 0), (True, 1 << 20)):
                with tempfile.TemporaryDirectory() as directory:
                    tier = NvmeTier(directory, 4096, 8192, reserve_bytes=0,
                                    mapped_staging=mapped, snapshot_cache_bytes=compressed)
                    try:
                        round_trip(self, tier, "cpu")
                        if tier.snapshot_cache is not None:
                            tier.snapshot_cache.clear()
                        self.assertEqual(tier.close(), 8192+4095 if mapped else 16384)
                        self.assertEqual(tier.close(), 0)
                    finally:
                        tier.close()

    def test_mapped_flag_and_allocator_geometry_reject_before_allocating(self):
        from engine.kernels.mapped_staging import allocate
        with self.assertRaises(ValueError):
            NvmeTier("unused", 4096, mapped_staging=1)
        for n in (0, -4096, 4095, True, 1.5):
            with self.assertRaises(ValueError):
                allocate(n)


if __name__ == "__main__":
    unittest.main()
