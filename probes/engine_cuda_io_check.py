"""Bounded ST arena/NVMe check, including producer-stream ordering and async I/O.

Uses 16 MiB of KV and 2 MiB staging; runs on a GB10 without loading a model.
The temporary directory must be on a filesystem that supports O_DIRECT.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.base.arena import Arena
from engine.base.kv import BlockPool
from engine.base.kv_tier import NvmeTier
from engine.base.tiered_kv import TieredKV


def main():
    block_bytes, blocks = 64 << 10, 256
    slot_bytes = 4096 * 5 + 300                                    # odd-sized state slots: sector padding on disk
    arena = Arena(block_bytes * blocks + 3 * slot_bytes + 256)
    pool = BlockPool(blocks, 16, 4, blocks)
    pool.attach_storage(arena.carve(block_bytes * blocks, "KV"), block_bytes)
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
        # Replacing an existing snapshot must leave its published generation
        # readable through data-write, fsync and manifest failures.
        for operation in ("pwritev", "fsync", "replace"):
            before = dict(tier.index["0"])
            pool.storage.fill_(41)
            with patch(f"engine.base.kv_tier.os.{operation}", side_effect=OSError("injected replacement failure")):
                try:
                    tier.demote(0, pool.storage, ids0, 1600)
                except OSError as exc:
                    assert "injected replacement failure" in str(exc)
                else:
                    raise AssertionError(f"{operation} failure did not reach the caller")
            assert tier.index["0"] == before
            pool.storage.zero_()
            tier.promote(0, pool.storage, ids0)
            assert all(bool((block == 73).all()) for block in pool.blocks_of(0))
        pool.storage.fill_(105)
        tier.demote(0, pool.storage, ids0, 1600)
        pool.storage.zero_()
        tier.promote(0, pool.storage, ids0)
        assert all(bool((block == 105).all()) for block in pool.blocks_of(0))
        with patch("pathlib.Path.unlink", side_effect=PermissionError("injected cleanup failure")):
            try:
                tier.forget(0)
            except PermissionError:
                pass
            else:
                raise AssertionError("unlink failure did not reach the caller")
        assert tier.index["0"]["deleting"] and not tier.has(0)
        reopened = NvmeTier(d, block_bytes, stage_bytes=2 << 20)
        reopened.cleanup()
        assert "0" not in reopened.index and reopened.has(1)
        assert len(list(Path(d).glob("seq-*.kv"))) == 1
        # A conversation is blocks + its state slot + a host record, under its own key, into any row and slot.
        slots = arena.carve(3 * slot_bytes, "slots").view(3, slot_bytes)
        slots[1].copy_(torch.randint(0, 256, slots[1].shape, dtype=torch.uint8, device="cuda"))
        slot_before, blocks_before = slots[1].clone(), torch.cat(pool.blocks_of(1)).clone()
        wrote = kv.park(1, key=77, extra=slots[1], record={"context": 1599, "pending": 1, "tokens": list(range(1600))})
        assert kv.is_parked(77) and pool.tokens[1] == 0 and reopened.has(77) is False        # row 1's own key stays: its earlier direct demote is still published
        again = NvmeTier(d, block_bytes, stage_bytes=2 << 20)                                 # after a "reboot": key, record, sizes are on disk
        assert again.has(77) and again.record(77)["context"] == 1599 and again.index["77"]["extra"] == slots[1].numel()
        slots[1].zero_(); pool.storage.zero_()
        got = kv.resume(3, key=77, extra=slots[2])                                            # a different row and slot
        assert wrote == got and torch.equal(slots[2], slot_before) and torch.equal(torch.cat(pool.blocks_of(3)), blocks_before)
        assert not kv.is_parked(77) and not list(Path(d).glob("seq-77-*"))
        small = NvmeTier(d, block_bytes, stage_bytes=2 << 20, capacity_bytes=block_bytes)     # room for nothing real
        from engine.base.kv_tier import TierFull
        try:
            small.demote(78, pool.storage, [b for b in pool.row(3) if b >= 0], 1600)
        except TierFull:
            pass
        else:
            raise AssertionError("a full tier must refuse before writing")
        assert not list(Path(d).glob("seq-78-*"))
        print(json.dumps({"passed": True, "kv_bytes": block_bytes * blocks, "staging_bytes": tier.stage_bytes,
                          "bytes_written": tier.bytes_written, "bytes_read": tier.bytes_read,
                          "producer_stream": True, "async_producer_stream": True,
                          "concurrent_roundtrip": True, "replacement_failures_retryable": True,
                          "cleanup_after_restart": True, "keyed_slot_record_roundtrip": True, "tier_full_refused": True}))


if __name__ == "__main__":
    main()
