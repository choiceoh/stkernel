"""Real host mapping, alias lifetime and lossless NVMe round trips on GB10."""
import gc
import os
import tempfile
import unittest
import torch

from tests.test_engine_mapped_tier import round_trip


@unittest.skipUnless(torch.cuda.is_available() and hasattr(os, "O_DIRECT"), "requires admitted Linux GB10")
class MappedTierCudaTests(unittest.TestCase):
    def test_mapping_visibility_and_either_alias_owns_allocation(self):
        from engine.kernels.mapped_staging import allocate
        host, gpu = allocate(8192)
        self.assertTrue(host.is_pinned())
        self.assertEqual(host.data_ptr() % 4096, 0)
        self.assertEqual(gpu.device.type, "cuda")
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            gpu.fill_(73)
        side.synchronize()
        self.assertTrue(torch.all(host == 73).item())
        host[4096:] = 29
        ids = torch.tensor([1, 0], device="cuda")
        expected = host.view(2, 4096).flip(0).clone()
        del host
        gc.collect()
        out = torch.empty_like(gpu).view(2, 4096)
        torch.index_select(gpu.view(2, 4096), 0, ids, out=out)
        self.assertTrue(torch.equal(out.cpu(), expected))
        host, alias = allocate(4096)
        del alias
        gc.collect()
        host.fill_(11)
        self.assertTrue(torch.all(host == 11).item())
        for nbytes in (4096, 8192, 12288, 4096) * 4:
            host, gpu = allocate(nbytes)
            self.assertEqual(host.data_ptr() % 4096, 0)
            gpu.fill_(31)
            torch.cuda.synchronize()
            self.assertTrue(torch.all(host == 31).item())

    def test_real_nvme_permuted_blocks_extra_tail_and_compressed_restore(self):
        from engine.base.kv_tier import NvmeTier
        for mapped, compressed in ((False, 0), (True, 0), (True, 1 << 20)):
            with tempfile.TemporaryDirectory(dir="/cache") as directory:
                tier = NvmeTier(directory, 4096, 8192, reserve_bytes=0,
                                mapped_staging=mapped, snapshot_cache_bytes=compressed)
                try:
                    round_trip(self, tier, "cuda")
                    if tier.snapshot_cache is not None:
                        tier.snapshot_cache.clear()
                    self.assertEqual(tier.close(), 8192+4095 if mapped else 16384)
                    self.assertEqual(tier.close(), 0)
                finally:
                    tier.close()


if __name__ == "__main__":
    unittest.main()
