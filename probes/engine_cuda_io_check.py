"""Bounded ST arena/NVMe check, including producer-stream ordering and async I/O.

Uses 16 MiB of KV and 2 MiB staging; runs on a GB10 without loading a model.
The temporary directory must be on a filesystem that supports O_DIRECT.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.base.arena import Arena
from engine.base.kv import BlockPool
from engine.base.kv_tier import NvmeTier
from engine.base.tiered_kv import TieredKV


def main():
    block_bytes, blocks = 64 << 10, 256
    arena = Arena(block_bytes * blocks)
    pool = BlockPool(blocks, 16, 4, blocks)
    pool.attach_storage(arena.carve(arena.nbytes, "KV"), block_bytes)
    with tempfile.TemporaryDirectory(prefix="st-engine-io-", dir=str(Path.home())) as d:
        tier = NvmeTier(d, block_bytes, stage_bytes=2 << 20)
        kv = TieredKV(pool, tier)
        pool.reserve(0, 16 * 100)
        pool.reserve(1, 16 * 100)
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            # The tier's own stream must wait for these prior writes.
            torch.cuda._sleep(5_000_000)
            pool.storage.fill_(37)
            written = kv.park(0)
        pool.storage.zero_()
        promoted = kv.resume(0)
        assert written == promoted == 100 * block_bytes
        assert all(bool((block == 37).all()) for block in pool.blocks_of(0))
        assert all(bool((block == 0).all()) for block in pool.blocks_of(1))
        ids0 = [b for b in pool.row(0) if b >= 0]
        ids1 = [b for b in pool.row(1) if b >= 0]
        with torch.cuda.stream(producer):
            torch.cuda._sleep(5_000_000)
            pool.storage.view(-1, block_bytes)[ids0] = 73
            jobs = [tier.run_async(tier.demote, seq, pool.storage, ids, 1600)
                    for seq, ids in ((0, ids0), (1, ids1))]
        assert [job.result(timeout=30) for job in jobs] == [100 * block_bytes] * 2
        pool.storage.fill_(99)
        jobs = [tier.run_async(tier.promote, seq, pool.storage, ids)
                for seq, ids in ((0, ids0), (1, ids1))]
        assert [job.result(timeout=30) for job in jobs] == [100 * block_bytes] * 2
        assert all(bool((block == 73).all()) for block in pool.blocks_of(0))
        assert all(bool((block == 0).all()) for block in pool.blocks_of(1))
        print(json.dumps({"passed": True, "kv_bytes": arena.nbytes, "staging_bytes": tier.stage_bytes,
                          "bytes_written": tier.bytes_written, "bytes_read": tier.bytes_read,
                          "producer_stream": True, "async_producer_stream": True,
                          "concurrent_roundtrip": True}))


if __name__ == "__main__":
    main()
