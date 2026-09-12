"""The NVMe tier for cold KV (base). Contiguous per sequence, O_DIRECT, survives boots.

D16's second layer. A sequence that goes idle has its blocks written as ONE
contiguous run to a file named for it; a sequence that comes back has them
read straight into freshly reserved blocks. Two properties are load-bearing:

  contiguous     the drive does 5.1 GB/s sequential and ~0.6 GB/s random
                 (157K IOPS x 4 KiB). A 128K-token sequence is 1.59 GiB
                 (Qwen3.8); contiguous is 0.3 s, scattered is 2.7 s.
  O_DIRECT       the page cache is the same pool as the arena on this box.
                 Reading 1.6 GiB through it would evict something that
                 mattered; direct I/O touches only the staging buffer.

Both directions stage through one PINNED, page-aligned host buffer sized to
`stage_bytes` and reused, and every device-side move happens on the tier's
OWN CUDA stream: one gather kernel per staging window (index_select over the
block storage viewed as [num_blocks, block_bytes]) and one async copy. The
first version did per-block Python `copy_` calls on the default stream and
slowed a decode-shaped loop by 1.50x (42nd ledger) -- thousands of launches
competing for the GIL and the stream a decoder was using. The tier holds no
memory the budget did not declare: the staging buffer and the device scratch
are both fixed at construction. The prefix tier also has a byte-capped,
lossless RAM cache of its snapshot payloads. It compresses existing staging
windows on this worker and retains no raw host copy. The disk format stays
unchanged; a RAM eviction loses only an I/O shortcut.

A parked conversation is more than its paged blocks: the state slot (KDA
rings, indexer tails, the drafter ring -- 247 MiB on GLM-5.3) and a small
host record (token history, context) go with it. The file holds three
things in order: the gathered blocks, the slot bytes padded to a sector,
and -- beside it -- a JSON record. The manifest names all of them, so a
conversation parked before a reboot is still a conversation after it.

Capacity is the disk's (D16): a demote that would not leave `reserve_bytes`
free on the filesystem (or exceed a declared `capacity_bytes`) raises
TierFull before writing a byte. Whom to forget then is the caller's policy.

Not in this file, on purpose: WHEN to demote. That is the scheduler's call
(D10: never while a decoder would wait), and the tier only promises that a
demotion or promotion in flight never blocks a step -- it runs on its own
thread and the runner asks `done()`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

SECTOR = 4096
_GENERATION = re.compile(r"seq-\d+-[0-9a-f]{32}\.(kv|json)")


def _sectors(n: int) -> int:
    return -(-n // SECTOR) * SECTOR


class TierFull(MemoryError):
    """The disk cannot take this conversation; forget one or drop this one."""


class NvmeTier:
    def __init__(self, directory: "str | Path", block_bytes: int, stage_bytes: int = 64 << 20,
                 capacity_bytes: "int | None" = None, reserve_bytes: int = 1 << 30,
                 snapshot_cache_bytes: int = 0):
        if not isinstance(block_bytes, int) or block_bytes <= 0 or block_bytes % SECTOR:
            raise ValueError(f"block_bytes {block_bytes} must be a positive multiple of {SECTOR} for O_DIRECT")
        if not isinstance(stage_bytes, int) or stage_bytes < block_bytes:
            raise ValueError("stage_bytes must hold at least one complete block")
        if stage_bytes % block_bytes:
            stage_bytes = (stage_bytes // block_bytes) * block_bytes
        if capacity_bytes is not None and (not isinstance(capacity_bytes, int) or capacity_bytes <= 0):
            raise ValueError("capacity_bytes must be a positive integer or None (the filesystem decides)")
        if type(snapshot_cache_bytes) is not int or snapshot_cache_bytes < 0:
            raise ValueError("snapshot cache must be nonnegative bytes")
        from engine.base.compressed_snapshots import CompressedSnapshots
        self.snapshot_cache = CompressedSnapshots(snapshot_cache_bytes) if snapshot_cache_bytes else None
        import torch

        self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True)
        self.block_bytes, self.stage_bytes = block_bytes, stage_bytes
        self.capacity_bytes, self.reserve_bytes = capacity_bytes, reserve_bytes
        self.per = stage_bytes // block_bytes
        # pinned host staging (cudaHostAlloc is page-aligned, which O_DIRECT needs)
        self.stage_t = torch.empty(stage_bytes, dtype=torch.uint8, pin_memory=True)
        self.stage = memoryview(self.stage_t.numpy())
        # device scratch for one window's gather, and the tier's own stream
        self.scratch = torch.empty(stage_bytes, dtype=torch.uint8, device="cuda")
        self.stream = torch.cuda.Stream()
        self.manifest = self.dir / "manifest.json"
        self.index = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        self.lock = threading.Lock()
        self._transfer_lock = threading.Lock()    # one staging buffer/stream, even across async callers
        self.bytes_written = self.bytes_read = 0

    def _path(self, seq: int) -> Path:
        return self.dir / self.index.get(str(seq), {}).get("file", f"seq-{seq}.kv")

    def _record_path(self, seq: int) -> "Path | None":
        name = self.index.get(str(seq), {}).get("record")
        return self.dir / name if name else None

    def has(self, seq: int) -> bool:
        meta = self.index.get(str(seq))
        return (meta is not None and not meta.get("deleting", False)
                and meta.get("block_bytes", self.block_bytes) == self.block_bytes)

    def keys(self) -> "list[int]":
        """Every parked conversation this layout can promote."""
        return sorted(int(k) for k in self.index if self.has(int(k)))

    def stale(self) -> "list[str]":
        """Foreign block layouts stay on disk and cannot be promoted."""
        return [k for k, meta in self.index.items()
                if meta.get("block_bytes", self.block_bytes) != self.block_bytes]

    def used_bytes(self) -> int:
        return sum(int(meta.get("bytes", 0)) for meta in self.index.values() if not meta.get("deleting"))

    def oldest(self) -> "int | None":
        """The next conversation to forget when the tier is full, or None.

        A foreign block layout goes FIRST. It sits on the disk and counts against the capacity,
        and no boot of this layout can ever promote it (`stale`) -- so it is not a conversation,
        it is bytes. Until this order existed a tier whose stale entries alone filled the cap
        had nothing it was willing to give up and refused every park forever, which is exactly
        what 85 GiB of one checkpoint's parked conversations would have done to the other's
        (45차 §53).

        Then the least recently parked conversation this layout CAN promote, which is the LRU
        the callers have always assumed.
        """
        foreign = [(meta.get("at", 0.0), int(k)) for k, meta in self.index.items()
                   if not meta.get("deleting")
                   and meta.get("block_bytes", self.block_bytes) != self.block_bytes]
        if foreign:
            return min(foreign)[1]
        live = [(meta.get("at", 0.0), int(k)) for k, meta in self.index.items() if self.has(int(k))]
        return min(live)[1] if live else None

    def stale_bytes(self) -> int:
        """What the foreign layouts occupy: the number the boot line owed the operator."""
        return sum(int(meta.get("bytes", 0)) for meta in self.index.values()
                   if not meta.get("deleting")
                   and meta.get("block_bytes", self.block_bytes) != self.block_bytes)

    def record(self, seq: int) -> "dict | None":
        path = self._record_path(seq)
        return json.loads(path.read_text()) if path is not None and path.exists() else None

    def _room(self, nbytes: int, replacing: int) -> None:
        """Refuse before writing: a declared capacity, else the filesystem's free space minus a reserve."""
        if self.capacity_bytes is not None:
            if self.used_bytes() - replacing + nbytes > self.capacity_bytes:
                raise TierFull(f"tier: {nbytes / 2**20:.0f} MiB would exceed the declared {self.capacity_bytes / 2**30:.2f} GiB "
                               f"({self.used_bytes() / 2**30:.2f} used)")
            return
        free = shutil.disk_usage(self.dir).free
        if nbytes + self.reserve_bytes > free:
            raise TierFull(f"tier: {nbytes / 2**20:.0f} MiB would leave {(free - nbytes) / 2**30:.2f} GiB on {self.dir}, "
                           f"under the {self.reserve_bytes / 2**30:.2f} GiB reserve")

    def _sync_directory(self) -> None:
        fd = os.open(self.dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _save_manifest(self, index: dict) -> None:
        tmp = self.manifest.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(index, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.manifest)
        self.index = index                       # publish only after the disk commit succeeds
        self._sync_directory()

    def _publish(self, seq, path, blocks, tokens, written, extra: int = 0, record: "str | None" = None):
        old = self.index.get(str(seq))
        retired = []
        if old:
            retired = list(old.get("retired", ())) + [self._path(seq).name]
            if old.get("record"):
                retired.append(old["record"])
        self._sync_directory()                   # persist the new generation before referring to it
        meta = {"file": path.name, "blocks": blocks, "tokens": tokens, "bytes": written,
                "block_bytes": self.block_bytes, "at": time.time(), "retired": retired, "extra": extra}
        if record:
            meta["record"] = record
        self._save_manifest({**self.index, str(seq): meta})

    def _prune_retired(self, seq):
        meta = self.index[str(seq)]
        for name in meta.get("retired", ()):
            (self.dir / name).unlink(missing_ok=True)
        if meta.get("retired"):
            self._save_manifest({**self.index, str(seq): {**meta, "retired": []}})

    def demote(self, seq: int, storage, block_ids: "list[int]", tokens: int, extra=None, record=None) -> int:
        """Write blocks `block_ids` of `storage` ([num_blocks * block_bytes] uint8)
        contiguously, then `extra` (a contiguous device uint8 view: the state
        slot) padded to a sector, and `record` (JSON-serialisable) beside them.
        Returns bytes. Device work runs on the tier stream only."""
        with self._transfer_lock:
            return self._demote(seq, storage, block_ids, tokens, extra, record)

    def _demote(self, seq: int, storage, block_ids: "list[int]", tokens: int, extra=None, record=None) -> int:
        if str(seq) in self.stale():
            raise ValueError(f"seq {seq} belongs to a different block layout")
        import torch

        extra_bytes = int(extra.numel()) if extra is not None else 0
        cache = getattr(self, "snapshot_cache", None)
        builder = cache.begin(extra_bytes) if cache is not None and extra_bytes else None
        old_file = self._path(seq).name
        total = len(block_ids) * self.block_bytes + _sectors(extra_bytes)
        self._room(total, int(self.index.get(str(seq), {}).get("bytes", 0)))
        table = storage.view(-1, self.block_bytes) if block_ids else None
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=table.device) if block_ids else None
        device = table.device if table is not None else extra.device
        self.stream.wait_stream(torch.cuda.current_stream(device))
        # The published generation stays untouched through write/fsync/manifest
        # failures. The manifest rename is the sole publication point.
        generation = f"seq-{seq}-{uuid4().hex}"
        path = self.dir / f"{generation}.kv"
        record_path = self.dir / f"{generation}.json" if record is not None else None
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_DIRECT, 0o644)
        written = 0
        try:
            for i in range(0, len(block_ids), self.per):
                n_blk = min(self.per, len(block_ids) - i); n = n_blk * self.block_bytes
                with torch.cuda.stream(self.stream):
                    torch.index_select(table, 0, ids[i:i + n_blk],
                                       out=self.scratch[:n].view(n_blk, self.block_bytes))
                    self.stage_t[:n].copy_(self.scratch[:n], non_blocking=True)
                self.stream.synchronize()
                written += self._write_window(fd, n, written)
            for off in range(0, extra_bytes, self.stage_bytes):
                n = min(self.stage_bytes, extra_bytes - off)
                with torch.cuda.stream(self.stream):
                    self.stage_t[:n].copy_(extra[off:off + n], non_blocking=True)
                self.stream.synchronize()
                written += self._write_window(fd, _sectors(n), written)     # the sector tail is stale staging bytes, never read back
                if builder is not None:
                    builder.add(self.stage[:n])
            compressed = builder.finish() if builder is not None else None
            os.fsync(fd)
            os.close(fd)
            fd = None
            if record_path is not None:
                with record_path.open("w") as f:
                    json.dump(record, f, separators=(",", ":"))
                    f.flush()
                    os.fsync(f.fileno())
            with self.lock:
                self._publish(seq, path, len(block_ids), tokens, written, extra_bytes,
                              record_path.name if record_path is not None else None)
        except BaseException:
            # A directory fsync may fail after the manifest was already
            # replaced. Never delete a generation which is now published.
            if self.index.get(str(seq), {}).get("file") != path.name:
                for unpublished in (path, record_path):
                    try:
                        if unpublished is not None:
                            unpublished.unlink(missing_ok=True)
                    except OSError:
                        pass                      # unreferenced generation; prior snapshot is intact
            raise
        finally:
            if fd is not None:
                os.close(fd)
        if cache is not None:
            cache.discard(old_file)
            cache.publish(path.name, compressed)   # only a committed generation may enter the RAM cache
        try:
            with self.lock:
                self._prune_retired(seq)
        except OSError:
            pass                                  # retired filenames remain in the manifest for retry
        self.bytes_written += written
        return written

    def _write_window(self, fd, n: int, at: int) -> int:
        off = 0
        while off < n:
            got = os.pwritev(fd, [self.stage[off:n]], at + off)
            if got <= 0:
                raise OSError(f"short write at {at + off}")
            off += got
        return n

    def _read_window(self, fd, n: int, at: int) -> None:
        off = 0
        while off < n:
            got = os.preadv(fd, [self.stage[off:n]], at + off)
            if got <= 0:
                raise OSError(f"short read at {at + off}")
            off += got

    def promote(self, seq: int, storage, block_ids: "list[int]", extra=None) -> int:
        """Read the sequence back into blocks `block_ids` of `storage` and its
        slot bytes into `extra` (required iff the file carries them). Returns
        disk bytes read; a compressed snapshot-only hit returns zero.

        `block_ids` None: the blocks are already in memory and only `extra` is
        read -- a faded prefix boundary (base/prefix.py) whose KV was never
        handed out needs its state back, not its bytes."""
        with self._transfer_lock:
            return self._promote(seq, storage, block_ids, extra)

    def _promote(self, seq: int, storage, block_ids: "list[int]", extra=None) -> int:
        import torch

        meta = self.index[str(seq)]
        if meta.get("deleting"):
            raise ValueError(f"seq {seq} is pending file cleanup, not promotion")
        if meta.get("block_bytes", self.block_bytes) != self.block_bytes:
            raise ValueError(f"seq {seq} was parked with {meta['block_bytes']} B blocks; this layout has {self.block_bytes}")
        if block_ids is not None and len(block_ids) != meta["blocks"]:
            raise ValueError(f"seq {seq}: {meta['blocks']} blocks on disk, {len(block_ids)} given")
        extra_bytes = int(meta.get("extra", 0))
        given = int(extra.numel()) if extra is not None else 0
        if given != extra_bytes:
            raise ValueError(f"seq {seq}: {extra_bytes} slot bytes on disk, a view of {given} given")
        if block_ids is None and not extra_bytes:
            raise ValueError(f"seq {seq} carries no slot bytes: a snapshot-only read would read nothing")
        table = storage.view(-1, self.block_bytes) if block_ids else None
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=table.device) if block_ids else None
        device = table.device if table is not None else extra.device
        self.stream.wait_stream(torch.cuda.current_stream(device))
        cache = getattr(self, "snapshot_cache", None)
        compressed = cache.get(self._path(seq).name) if cache is not None and extra_bytes else None
        if compressed is not None and compressed.raw_bytes != extra_bytes:
            raise ValueError("compressed snapshot does not match its committed generation")
        # A faded boundary with a RAM copy needs no file I/O at all. Full
        # restores still read their paged KV, then reuse the compressed state.
        fd = os.open(self._path(seq), os.O_RDONLY | os.O_DIRECT) if block_ids or compressed is None else None
        at = 0 if block_ids is not None else int(meta["blocks"]) * self.block_bytes   # what memory already holds is not read again
        read = 0
        try:
            for i in range(0, len(block_ids or ()), self.per):
                n_blk = min(self.per, len(block_ids) - i); n = n_blk * self.block_bytes
                self._read_window(fd, n, at)
                with torch.cuda.stream(self.stream):
                    self.scratch[:n].copy_(self.stage_t[:n], non_blocking=True)
                    table.index_copy_(0, ids[i:i + n_blk], self.scratch[:n].view(n_blk, self.block_bytes))
                self.stream.synchronize()
                at += n; read += n
            def upload(off, n):
                with torch.cuda.stream(self.stream):
                    extra[off:off + n].copy_(self.stage_t[:n], non_blocking=True)
                self.stream.synchronize()
            if compressed is not None:
                started = time.perf_counter()
                try:
                    compressed.restore(self.stage, upload)
                finally:
                    cache.restore_seconds += time.perf_counter() - started
            else:
                for off in range(0, extra_bytes, self.stage_bytes):
                    n = min(self.stage_bytes, extra_bytes - off)
                    self._read_window(fd, _sectors(n), at)
                    upload(off, n)
                    at += _sectors(n); read += _sectors(n)
        finally:
            if fd is not None:
                os.close(fd)
        self.bytes_read += read
        return read

    def forget(self, seq: int) -> None:
        with self._transfer_lock:
            self._forget(seq)

    def _forget(self, seq):
        with self.lock:
            meta = self.index.get(str(seq))
            if meta is None:
                return
            if not meta.get("deleting"):
                self._save_manifest({**self.index, str(seq): {**meta, "deleting": True}})
            cache = getattr(self, "snapshot_cache", None)
            if cache is not None:
                cache.discard(self._path(seq).name)
            # The tombstone survives partial unlink and final manifest
            # failures. A retry (including after restart) finishes it.
            self._path(seq).unlink(missing_ok=True)
            record = self._record_path(seq)
            if record is not None:
                record.unlink(missing_ok=True)
            for name in meta.get("retired", ()):
                (self.dir / name).unlink(missing_ok=True)
            index = dict(self.index)
            index.pop(str(seq), None)
            self._save_manifest(index)

    def cleanup(self) -> None:
        """Retry recorded deletions and reclaim unpublished generations.

        Call off the decode path, including after reopening this tier. This
        directory belongs to one NvmeTier; transfers serialize with cleanup.
        Legacy files without a manifest entry are deliberately left alone.
        """
        with self._transfer_lock:
            for seq, meta in list(self.index.items()):
                if meta.get("deleting") and seq not in self.stale():
                    self._forget(int(seq))
            with self.lock:
                for seq in list(self.index):
                    if seq not in self.stale():
                        self._prune_retired(int(seq))
                referenced = {self._path(int(seq)).name for seq in self.index}
                referenced.update(meta["record"] for meta in self.index.values() if meta.get("record"))
                referenced.update(name for meta in self.index.values() for name in meta.get("retired", ()))
                for path in list(self.dir.glob("seq-*.kv")) + list(self.dir.glob("seq-*.json")):
                    if path.name not in referenced and _GENERATION.fullmatch(path.name):
                        path.unlink(missing_ok=True)
                self._sync_directory()

    def close(self) -> int:
        """Give the staging buffers and compressed copies back after transfers stop.

        A tier is pinned host DRAM plus device scratch that live OUTSIDE the arena, so the
        engine's release cannot reach them and `empty_cache` will not take them while this
        object holds them. Two tiers a rank, four ranks: on a handover that is most of a
        gigabyte the next holder would otherwise be waiting for (45차 §51).

        What is on disk is untouched -- conversations parked here outlive this process, which
        is the point of the tier (D16). This frees the buffers unconditionally, so the caller
        owes it a quiet tier: `TieredKV.close` waits out whatever was in flight first.
        Idempotent, and it returns the bytes.
        """
        given = 0
        stage = getattr(self, "stage", None)
        if stage is not None:
            stage.release()                       # the memoryview holds the pinned pages open
            self.stage = None
        for name in ("stage_t", "scratch"):
            buf = getattr(self, name, None)
            if buf is not None:
                given += buf.numel() * buf.element_size()
                setattr(self, name, None)
        self.stream = None
        cache = getattr(self, "snapshot_cache", None)
        if cache is not None:
            given += cache.clear()
        return given

    def run_async(self, fn, *args) -> Future:
        """Off-thread I/O. Poll `.done()`, then `.result()` to surface failures.

        Transfers on this tier serialize on its one staging buffer. Running
        decoders do not acquire that lock or wait on these futures.
        Preserve the caller's CUDA stream in the worker so a transfer waits
        for its producer even when submission came from a nondefault stream.
        """
        future = Future()
        context = nullcontext
        if hasattr(self, "stream"):
            import torch
            producer = torch.cuda.current_stream(self.stream.device)
            context = lambda: torch.cuda.stream(producer)

        def work():
            if not future.set_running_or_notify_cancel():
                return
            try:
                with context():
                    result = fn(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        threading.Thread(target=work, daemon=True).start()
        return future


def _selfcheck() -> None:
    import tempfile
    import torch

    GIB = 1 << 30
    block_bytes = 51 * SECTOR
    n_blocks = (1 * GIB) // block_bytes
    storage = torch.randint(0, 256, (n_blocks * block_bytes,), dtype=torch.uint8, device="cuda")
    keep = storage.clone()
    slot = torch.randint(0, 256, (3 * SECTOR + 1234,), dtype=torch.uint8, device="cuda")   # an odd-sized state slot
    slot_keep = slot.clone()
    ids = list(range(n_blocks))[::-1]               # scattered order: the gather must honour it
    with tempfile.TemporaryDirectory(dir="/home/choiceoh") as d:
        tier = NvmeTier(d, block_bytes)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        wrote = tier.demote(7, storage, ids, tokens=n_blocks * 16, extra=slot, record={"context": 5, "pending": 1, "tokens": [1, 2, 3]})
        t_w = time.perf_counter() - t0
        storage.zero_(); slot.zero_()
        t0 = time.perf_counter(); got = tier.promote(7, storage, ids, extra=slot); torch.cuda.synchronize(); t_r = time.perf_counter() - t0
        assert wrote == got == n_blocks * block_bytes + _sectors(slot.numel()) and torch.equal(storage, keep) and torch.equal(slot, slot_keep), "round trip must be exact"
        assert tier.record(7) == {"context": 5, "pending": 1, "tokens": [1, 2, 3]} and tier.keys() == [7] and tier.oldest() == 7
        tier2 = NvmeTier(d, block_bytes); assert tier2.has(7) and tier2.record(7)["context"] == 5
        try:
            tier2.promote(7, storage, ids); raise AssertionError("a file with slot bytes must be promoted with a slot view")
        except ValueError:
            pass
        other = NvmeTier(d, block_bytes * 2); assert not other.has(7) and other.stale() == ["7"]
        small = NvmeTier(d, block_bytes, capacity_bytes=wrote)                   # exactly one conversation fits
        try:
            small.demote(8, storage, ids[:1], tokens=16); raise AssertionError("a full tier must refuse before writing")
        except TierFull:
            pass
        assert not list(Path(d).glob("seq-8-*")), "a refused demote leaves no file"
        tier2.forget(7); assert not tier2.has(7) and not list(Path(d).glob("seq-7-*"))
        print(f"  kv_tier v3: {wrote / GIB:.2f} GiB demote {t_w:.2f}s ({wrote / GIB / t_w:.2f} GiB/s), "
              f"promote {t_r:.2f}s ({got / GIB / t_r:.2f} GiB/s), scattered ids + odd slot bytes + record exact, "
              "capacity refused, forget removes every file OK")


if __name__ == "__main__":
    _selfcheck()
