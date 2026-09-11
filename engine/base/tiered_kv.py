"""Park a conversation's KV on NVMe and bring it back (base): D16's verb pair.

    park(seq, key)    demote the row's blocks (and, given `extra`, its state
                      slot bytes and a host `record`) contiguously under `key`,
                      then free the blocks -- the arena gets them back, the
                      conversation keeps its KV.
    resume(seq, key)  reserve fresh blocks in an EMPTY row `seq` and read the
                      file `key` into them (and its slot bytes into `extra`).

`key` is the conversation's identity and `seq` is whichever row it occupies
this time: rows are few (max_seqs) and reused, conversations are many and
live on disk. The tier proved it never blocks a running decoder (p50 ratio
1.000/1.001); this file's job is the bookkeeping that keeps three things
consistent: the block table, the token count, and the manifest. Parking is
per conversation and the scheduler decides when (an idle turn, never a live one).
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
        self.parked = {}                      # key -> tokens, for conversations parked by this process

    def park(self, seq: int, key: "int | None" = None, extra=None, record=None) -> int:
        key = seq if key is None else key
        ids = [b for b in self.pool.row(seq) if b != EMPTY]
        tokens = self.pool.tokens[seq]
        if not ids:
            raise ValueError(f"seq {seq} holds no blocks")
        if self.is_parked(key):
            raise ValueError(f"conversation {key} is already parked")
        kwargs = {}
        if extra is not None:
            kwargs["extra"] = extra
        if record is not None:
            kwargs["record"] = record
        wrote = self.tier.demote(key, self.pool.storage, ids, tokens, **kwargs)
        self.pool.release(seq)
        self.parked[key] = tokens
        return wrote

    def resume(self, seq: int, key: "int | None" = None, extra=None) -> int:
        """Restore into an empty row; a failed read keeps the disk copy retryable.

        Once promotion completes, the resident copy is committed. A subsequent
        failure to forget the disk copy must not release that restored memory.
        """
        key = seq if key is None else key
        self.pool.row(seq)                                # bounds before indexing tokens
        if self.pool.tokens[seq]:
            raise ValueError(f"seq {seq} already has resident KV")
        tokens = self.parked[key] if key in self.parked else self.tier.index[str(key)]["tokens"]
        self.pool.reserve(seq, tokens)                     # MemoryError if the arena is full: no fallback
        ids = [b for b in self.pool.row(seq) if b != EMPTY]
        try:
            got = self.tier.promote(key, self.pool.storage, ids, **({"extra": extra} if extra is not None else {}))
        except BaseException:
            self.pool.release(seq)
            raise
        self.parked.pop(key, None)
        self.tier.forget(key)
        return got

    def is_parked(self, key: int) -> bool:
        has = getattr(self.tier, "has", None)                 # a bare tier (probes' fakes) knows only what this process parked
        return key in self.parked or (has is not None and bool(has(key)))

    def record(self, key: int) -> "dict | None":
        reader = getattr(self.tier, "record", None)
        return reader(key) if reader is not None else None

    def blocks(self, key: int) -> int:
        """Blocks a parked conversation will need back."""
        return int(self.tier.index[str(key)]["blocks"])

    def keys(self) -> "list[int]":
        lister = getattr(self.tier, "keys", None)
        if lister is not None:
            return lister()
        return sorted(int(k) for k in self.tier.index if self.tier.has(int(k)))

    def oldest(self) -> "int | None":
        finder = getattr(self.tier, "oldest", None)
        if finder is not None:
            return finder()
        keys = self.keys()
        return keys[0] if keys else None

    def forget(self, key: int) -> None:
        """The parked conversation is over: its disk copy goes."""
        self.parked.pop(key, None)
        self.tier.forget(key)


def _selfcheck() -> None:
    import tempfile
    import torch
    from engine.base.arena import Arena
    from engine.base.kv_tier import SECTOR

    block_bytes = 51 * SECTOR; n_blocks = 512
    arena = Arena(n_blocks * block_bytes + (8 << 20))
    pool = BlockPool(n_blocks, 16, max_seqs=4, max_blocks_per_seq=n_blocks)
    pool.attach_storage(arena.carve(n_blocks * block_bytes, "kv"), block_bytes)
    slots = arena.carve(4 * (SECTOR + 100), "slots").view(4, SECTOR + 100)    # odd-sized state slots
    with tempfile.TemporaryDirectory(dir="/home/choiceoh") as d:
        kv = TieredKV(pool, NvmeTier(d, block_bytes, stage_bytes=8 << 20))
        pool.reserve(3, 16 * 300)                            # row 3 holds 300 blocks of conversation 42
        for b in pool.blocks_of(3):
            b.copy_(torch.randint(0, 256, (block_bytes,), dtype=torch.uint8, device="cuda"))
        slots[2].copy_(torch.randint(0, 256, (SECTOR + 100,), dtype=torch.uint8, device="cuda"))
        before, slot_before = torch.cat(pool.blocks_of(3)).clone(), slots[2].clone()
        free0 = pool.available
        wrote = kv.park(3, key=42, extra=slots[2], record={"context": 4800, "pending": 1})
        assert pool.available == free0 + 300 and kv.is_parked(42) and not kv.is_parked(3) and pool.tokens[3] == 0
        assert kv.record(42) == {"context": 4800, "pending": 1} and kv.blocks(42) == 300 and kv.keys() == [42]
        pool.reserve(1, 16 * 400)                            # someone else takes the space meanwhile
        pool.release(1)
        slots[2].zero_()
        got = kv.resume(0, key=42, extra=slots[1])           # a different row and a different slot this time
        after = torch.cat(pool.blocks_of(0))
        assert wrote == got and torch.equal(before, after) and torch.equal(slots[1], slot_before) and not kv.is_parked(42)
        assert pool.tokens[0] == 16 * 300 and pool.tokens[3] == 0
        print(f"  tiered_kv: park 300 blocks + slot under key 42 -> arena freed, resume into row 0 / slot 1 -> {wrote / 2**20:.0f} MiB byte-identical OK")


if __name__ == "__main__":
    _selfcheck()
