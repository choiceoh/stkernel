"""Park a sequence's KV on NVMe and bring it back (base): D16's verb pair.

    park(seq)    demote the sequence's blocks contiguously, then free them --
                 the arena gets its blocks back, the conversation keeps its KV.
    resume(seq)  reserve fresh blocks and read the file into them.

The tier proved it never blocks a running decoder (p50 ratio 1.000/1.001);
this file's job is the bookkeeping that keeps three things consistent: the
block table, the token count, and the manifest. Parking is per sequence and
the scheduler decides when (an idle turn, never a live one).
"""
from __future__ import annotations

from engine.base.kv import BlockPool, EMPTY
from engine.base.kv_tier import NvmeTier


class TieredKV:
    def __init__(self, pool: BlockPool, tier: NvmeTier):
        if pool.storage is None:
            raise ValueError("attach the pool's storage (arena) before tiering it")
        if pool.block_bytes != tier.block_bytes:
            raise ValueError(f"pool block {pool.block_bytes} B != tier block {tier.block_bytes} B")
        self.pool, self.tier = pool, tier
        self.parked = {}                      # seq -> tokens

    def park(self, seq: int) -> int:
        ids = [b for b in self.pool.row(seq) if b != EMPTY]
        tokens = self.pool.tokens[seq]
        if not ids:
            raise ValueError(f"seq {seq} holds no blocks")
        wrote = self.tier.demote(seq, self.pool.storage, ids, tokens)
        self.pool.release(seq)
        self.parked[seq] = tokens
        return wrote

    def resume(self, seq: int) -> int:
        """Restore into an empty row; a failed read keeps the disk copy retryable.

        Once promotion completes, the resident copy is committed. A subsequent
        failure to forget the disk copy must not release that restored memory.
        """
        self.pool.row(seq)                                # bounds before indexing tokens
        if self.pool.tokens[seq]:
            raise ValueError(f"seq {seq} already has resident KV")
        tokens = self.parked[seq] if seq in self.parked else self.tier.index[str(seq)]["tokens"]
        self.pool.reserve(seq, tokens)                     # MemoryError if the arena is full: no fallback
        ids = [b for b in self.pool.row(seq) if b != EMPTY]
        try:
            got = self.tier.promote(seq, self.pool.storage, ids)
        except BaseException:
            self.pool.release(seq)
            raise
        self.parked.pop(seq, None)
        self.tier.forget(seq)
        return got

    def is_parked(self, seq: int) -> bool:
        return seq in self.parked or self.tier.has(seq)


def _selfcheck() -> None:
    import tempfile
    import torch
    from engine.base.arena import Arena
    from engine.base.kv_tier import SECTOR

    block_bytes = 51 * SECTOR; n_blocks = 512
    arena = Arena(n_blocks * block_bytes + (8 << 20))
    pool = BlockPool(n_blocks, 16, max_seqs=4, max_blocks_per_seq=n_blocks)
    pool.attach_storage(arena.carve(n_blocks * block_bytes, "kv"), block_bytes)
    with tempfile.TemporaryDirectory(dir="/home/choiceoh") as d:
        kv = TieredKV(pool, NvmeTier(d, block_bytes, stage_bytes=8 << 20))
        pool.reserve(3, 16 * 300)                            # seq 3 holds 300 blocks
        for b in pool.blocks_of(3):
            b.copy_(torch.randint(0, 256, (block_bytes,), dtype=torch.uint8, device="cuda"))
        before = torch.cat(pool.blocks_of(3)).clone()
        free0 = pool.available
        wrote = kv.park(3)
        assert pool.available == free0 + 300 and kv.is_parked(3) and pool.tokens[3] == 0
        pool.reserve(1, 16 * 400)                            # someone else takes the space meanwhile
        pool.release(1)
        got = kv.resume(3)
        after = torch.cat(pool.blocks_of(3))
        assert wrote == got == 300 * block_bytes and torch.equal(before, after) and not kv.is_parked(3)
        assert pool.tokens[3] == 16 * 300
        print(f"  tiered_kv: park 300 blocks -> arena freed, resume into new blocks -> {wrote / 2**20:.0f} MiB byte-identical OK")


if __name__ == "__main__":
    _selfcheck()
