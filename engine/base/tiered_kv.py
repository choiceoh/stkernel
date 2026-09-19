"""Park a conversation's KV on NVMe and bring it back (base): D16's verb pair.

    park(seq, key)    demote the row's blocks (and, given `extra`, its state
                      slot bytes and a host `record`) contiguously under `key`,
                      then free the blocks -- the arena gets them back, the
                      conversation keeps its KV.
    resume(seq, key)  reserve fresh blocks in an EMPTY row `seq` and read the
                      file `key` into them (and its slot bytes into `extra`).

`key` is the conversation's identity and `seq` is whichever row it occupies
this time: rows are few (max_seqs) and reused, conversations are many and
live on disk.

Each verb is two halves so a step loop never waits on the disk (D10):
`park_begin` hands the write to the tier's thread and keeps the row's blocks
reserved; `park_finish` -- once `done(seq)` -- returns them. `resume_begin`
reserves the blocks and hands the read over; `resume_finish` commits them or,
on a failed read, returns them and keeps the disk copy. `park`/`resume` are
the two halves back to back, for callers that may wait.

The tier proved it never blocks a running decoder (p50 ratio 1.000/1.001);
this file's job is the bookkeeping that keeps three things consistent: the
block table, the token count, and the manifest. Parking is per conversation
and the scheduler decides when (an idle turn, never a live one).

A fleet boot opens two of them per rank -- parked conversations and evicted
prefix boundaries -- in one directory every profile shares, under the fleet's
caps (`open_tiers`, TIER_ROOT).
"""
from __future__ import annotations

import functools
import threading
from concurrent.futures import Future
from pathlib import Path

from engine.base.kv import BlockPool, EMPTY
from engine.base.kv_tier import NvmeTier

GIB = 1 << 30
TIER_ROOT = "/home/choiceoh/glm53-logs/st-tier"
"""Where a fleet boot keeps its tiers, one `rank<N>` directory per rank: under the one host directory the rank
containers bind (the launchers' MOUNTED_ROOT). Every profile that serves on the fleet uses the same one -- GLM-5.3 and
Qwen3.8 never serve at once, and a directory each would hold a second budget of bytes for an engine that is not
running. A layout's files are foreign to the other (`NvmeTier.stale`): never promoted, counted against the cap, the
first forgotten when it bites, replaced when the other writes the same key. A change of hands still empties it
(base/tenancy); one owner switching models does not."""
TIER_GIB = 64.0
"""What a rank's parked conversations may occupy on NVMe, and its evicted prefix boundaries below.

Declared, because "the filesystem decides" is not a decision (D1). Until 45차 §53 neither tier
had a capacity at all, so the only brake was `reserve_bytes` -- one gigabyte of free space --
on a root that also carries the checkpoints, the images and the logs. It had eaten 75 GiB of a
disk that was 99% full, and nothing in the engine had ever deleted a byte of it.

A GLM-5.3 conversation is ~260 MiB here (one block plus its 247 MiB state slot, the size 45차
§49 left open), so 64 GiB is about 250 of them and 16 GiB is about 30 prefix boundaries; a
Qwen3.8 one is its 109 MiB state slot plus 10.4 MiB a 768-token block (~254 MiB at 10K tokens).
The prefix tier gets the smaller share on purpose: a boundary is a cache that recomputes, a
conversation is a turn the user may come back to (D16). Past the cap the LRU forgets, foreign
layouts first (`NvmeTier.oldest`). The caps are the fleet's, not a profile's: whichever engine
serves, the directory holds at most this much.
"""
PREFIX_TIER_GIB = 16.0
TIER_RESERVE_GIB = 16.0             # free space a tier leaves on the filesystem whatever its own cap allows
PREFIX_TIER_STAGE = 32 << 20        # the prefix tier's pinned staging + device scratch


class TieredKV:
    def __init__(self, pool: BlockPool, tier: NvmeTier):
        if pool.storage is None:
            raise ValueError("attach the pool's storage (arena) before tiering it")
        if pool.block_bytes != tier.block_bytes:
            raise ValueError(f"pool block {pool.block_bytes} B != tier block {tier.block_bytes} B")
        self.pool, self.tier = pool, tier
        self.parked = {}                      # key -> tokens, for conversations parked by this process
        self.inflight = {}                    # row -> (kind, key, tokens, future): a transfer the row waits on

    def close(self, timeout: float = 30.0) -> int:
        """The tier's staging buffers back; the index on disk and everything parked in it stay.

        A transfer in flight is reading or writing the very buffers this frees, and the tier's
        workers are daemon threads nobody joins, so this waits for them first. Their failures
        are not this call's to raise -- serving is already over and the row that cared is gone;
        what matters is that no thread is still holding the staging when it goes.
        """
        for entry in list(self.inflight.values()):
            try:
                entry[3].result(timeout=timeout)
            except BaseException:                  # noqa: BLE001 -- a shutdown never fails a shutdown
                if not entry[3].done():
                    # A timeout is not completion. The worker still owns the
                    # pinned allocation (and both aliases in mapped mode).
                    # Retain it and its future so shutdown can be retried.
                    raise TimeoutError("tier transfer still owns staging; retry close after it finishes")
        self.inflight.clear()
        close = getattr(self.tier, "close", None)
        return close() if close is not None else 0

    def _submit(self, fn, *args, **kwargs) -> Future:
        """The tier's own thread when it has one; a completed Future otherwise (probes' bare fakes)."""
        run_async = getattr(self.tier, "run_async", None)
        if run_async is not None:
            return run_async(functools.partial(fn, *args, **kwargs))
        future = Future()
        future.set_running_or_notify_cancel()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:          # noqa: BLE001 -- the caller reads it from the future, like the async path
            future.set_exception(exc)
        return future

    # -- park --------------------------------------------------------------------------------
    def park_begin(self, seq: int, key: "int | None" = None, extra=None, record=None) -> Future:
        """Hand the row's blocks (and slot bytes, record) to the tier under `key`; the blocks stay
        reserved until `park_finish`."""
        key = seq if key is None else key
        if seq in self.inflight:
            raise ValueError(f"row {seq} already has a transfer in flight")
        ids = [b for b in self.pool.row(seq) if b != EMPTY]
        tokens = self.pool.tokens[seq]
        if not ids:
            raise ValueError(f"seq {seq} holds no blocks")
        if self.is_parked(key) or any(k == key for _, k, _, _ in self.inflight.values()):
            raise ValueError(f"conversation {key} is already parked")
        kwargs = {}
        if extra is not None:
            kwargs["extra"] = extra
        if record is not None:
            kwargs["record"] = record
        future = self._submit(self.tier.demote, key, self.pool.storage, ids, tokens, **kwargs)
        self.inflight[seq] = ("park", key, tokens, future)
        return future

    def park_finish(self, seq: int) -> int:
        """The write is done (or failed): free the blocks, or keep them and raise."""
        kind, key, tokens, future = self.inflight.pop(seq)
        if kind != "park":
            self.inflight[seq] = (kind, key, tokens, future)
            raise ValueError(f"row {seq} is resuming, not parking")
        wrote = future.result()               # raises TierFull / OSError: the row keeps its resident blocks
        self.pool.release(seq)
        self.parked[key] = tokens
        return wrote

    def park(self, seq: int, key: "int | None" = None, extra=None, record=None) -> int:
        self.park_begin(seq, key, extra, record)
        return self.park_finish(seq)

    # -- resume ------------------------------------------------------------------------------
    def resume_begin(self, seq: int, key: "int | None" = None, extra=None) -> Future:
        """Reserve the blocks in the empty row `seq` and hand the read to the tier."""
        key = seq if key is None else key
        self.pool.row(seq)                                # bounds before indexing tokens
        if seq in self.inflight:
            raise ValueError(f"row {seq} already has a transfer in flight")
        if self.pool.tokens[seq]:
            raise ValueError(f"seq {seq} already has resident KV")
        tokens = self.parked[key] if key in self.parked else self.tier.index[str(key)]["tokens"]
        self.pool.reserve(seq, tokens)                     # MemoryError if the arena is full: no fallback
        ids = [b for b in self.pool.row(seq) if b != EMPTY]
        try:
            future = self._submit(self.tier.promote, key, self.pool.storage, ids,
                                  **({"extra": extra} if extra is not None else {}))
        except BaseException:
            self.pool.release(seq)
            raise
        self.inflight[seq] = ("resume", key, tokens, future)
        return future

    def resume_finish(self, seq: int) -> int:
        """The read is done: the resident copy is committed and the disk copy goes. A failed read
        returns the blocks and keeps the disk copy; a failed forget keeps the restored memory."""
        kind, key, tokens, future = self.inflight.pop(seq)
        if kind != "resume":
            self.inflight[seq] = (kind, key, tokens, future)
            raise ValueError(f"row {seq} is parking, not resuming")
        try:
            got = future.result()
        except BaseException:
            self.pool.release(seq)
            raise
        self.parked.pop(key, None)
        self.tier.forget(key)
        return got

    def resume_cancel(self, seq: int) -> None:
        """A resume that will not be committed, whatever its read did (the caller could not reopen the row): the
        blocks go back and the disk copy stays."""
        kind, key, tokens, future = self.inflight.pop(seq)
        if kind != "resume":
            self.inflight[seq] = (kind, key, tokens, future)
            raise ValueError(f"row {seq} is parking, not resuming")
        try:
            future.result()                               # the blocks are not the pool's again while a read writes them
        except BaseException:                             # noqa: BLE001 -- the resume is being given up either way
            pass
        self.pool.release(seq)

    def resume(self, seq: int, key: "int | None" = None, extra=None) -> int:
        self.resume_begin(seq, key, extra)
        return self.resume_finish(seq)

    def done(self, seq: int) -> bool:
        return self.inflight[seq][3].done()

    # -- the parked set ------------------------------------------------------------------------
    def is_parked(self, key: int) -> bool:
        has = getattr(self.tier, "has", None)                 # a bare tier (probes' fakes) knows only what this process parked
        return key in self.parked or (has is not None and bool(has(key)))

    def record(self, key: int) -> "dict | None":
        reader = getattr(self.tier, "record", None)
        return reader(key) if reader is not None else None

    def read_record(self, key: int) -> Future:
        """`record(key)` on the tier's thread: a resume whose record the runner no longer holds reads it beside the
        blocks, because reading a whole history is disk work (7.2 ms of json.loads per 100K tokens), not a step's
        (D10)."""
        return self._submit(self.record, key)

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


def open_tiers(pool: BlockPool, block_bytes: int, root, rank: int, *, state_format: str, owner: "str | None" = None,
               mapped_staging: bool = False, prefix_cache_bytes: int = 0, background: bool = True,
               make=NvmeTier) -> "tuple[TieredKV, TieredKV, str | None]":
    """One rank's two tiers under `root`/rank<N> (D16) -> (conversations, prefix boundaries, the owner the directory was
    taken from or None). What every fleet boot does the same way, so it is written once:

        claim       `owner` takes the rank's directory (base/tenancy): a restart keeps its conversations, a handover
                    does not. No owner (a bare boot) claims nothing.
        the tiers   the conversations at the directory's top, the prefix boundaries in `prefix/` below it, both in
                    `block_bytes` NVMe units under the fleet's caps and reserve (TIER_GIB, PREFIX_TIER_GIB,
                    TIER_RESERVE_GIB). `state_format` names the layout: another's files are foreign here.
        sweep       `NvmeTier.cleanup` on each: a generation a killed write left unpublished, and a forget a killed boot
                    left half done, are bytes on the disk no cap counts -- and nothing called it, so no boot ever took
                    them back. On a thread of its own (`background`): what it deletes differs by rank, and ranks that
                    finish their own work at different moments meet the next collective apart (tenancy.claim's lesson).

    `make`: the tier class (a test hands a fake -- an NvmeTier allocates CUDA staging)."""
    from engine.base import tenancy
    directory = Path(root) / f"rank{rank}"
    left = tenancy.claim(directory, owner) if owner else None
    conversations = make(directory, block_bytes=block_bytes, capacity_bytes=int(TIER_GIB * GIB),
                         reserve_bytes=int(TIER_RESERVE_GIB * GIB), state_format=state_format,
                         mapped_staging=mapped_staging)
    prefix = make(directory / "prefix", block_bytes=block_bytes, stage_bytes=PREFIX_TIER_STAGE,
                  capacity_bytes=int(PREFIX_TIER_GIB * GIB), reserve_bytes=int(TIER_RESERVE_GIB * GIB),
                  snapshot_cache_bytes=prefix_cache_bytes, state_format=state_format, mapped_staging=mapped_staging)
    for tier in (conversations, prefix):
        if background:
            threading.Thread(target=sweep, args=(tier,), daemon=True, name="tier-sweep").start()
        else:
            sweep(tier)
    return TieredKV(pool, conversations), TieredKV(pool, prefix), left


def sweep(tier) -> None:
    """`tier.cleanup()`, a failure printed rather than raised: the tier works without it, and the next boot's sweep
    takes what this one could not."""
    try:
        tier.cleanup()
    except Exception as exc:                                # noqa: BLE001 -- see above
        print(f"  tier sweep of {getattr(tier, 'dir', '?')} stopped: {type(exc).__name__}: {exc}", flush=True)


def tier_line(tier, cap_gib: float, what: str) -> str:
    """What is on the disk, in bytes -- the boot used to print counts and leave the size a mystery."""
    live = sum(1 for k in tier.index if tier.has(int(k)))
    stale, stale_bytes = len(tier.stale()), tier.stale_bytes()
    line = (f"  NVMe tier: {live} {what} parked from before, {tier.used_bytes() / GIB:.1f} GiB of "
            f"{cap_gib:.0f} GiB")
    if stale:
        line += (f"; {stale} under another layout holding {stale_bytes / GIB:.1f} GiB -- not resumable, "
                 f"and the first thing forgotten when the cap bites")
    return line


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
        future = kv.park_begin(3, key=42, extra=slots[2], record={"context": 4800, "pending": 1})
        assert pool.available == free0 and pool.tokens[3] == 16 * 300, "blocks stay reserved while the write is in flight"
        future.result()
        assert kv.done(3)
        wrote = kv.park_finish(3)
        assert pool.available == free0 + 300 and kv.is_parked(42) and not kv.is_parked(3) and pool.tokens[3] == 0
        assert kv.record(42) == {"context": 4800, "pending": 1} and kv.blocks(42) == 300 and kv.keys() == [42]
        pool.reserve(1, 16 * 400)                            # someone else takes the space meanwhile
        pool.release(1)
        slots[2].zero_()
        kv.resume_begin(0, key=42, extra=slots[1])           # a different row and a different slot this time
        assert pool.tokens[0] == 16 * 300, "resume reserves its blocks up front"
        got = kv.resume_finish(0)
        after = torch.cat(pool.blocks_of(0))
        assert wrote == got and torch.equal(before, after) and torch.equal(slots[1], slot_before) and not kv.is_parked(42)
        assert pool.tokens[0] == 16 * 300 and pool.tokens[3] == 0 and not kv.inflight
        print(f"  tiered_kv: park 300 blocks + slot under key 42 off-thread -> arena freed at finish, resume into row 0 / slot 1 -> {wrote / 2**20:.0f} MiB byte-identical OK")


if __name__ == "__main__":
    _selfcheck()
