"""Bounded, lossless RAM copies of snapshots whose durable copy is on NVMe.

Only the tier worker touches entries. Rank-local compression ratios affect
disk reads, never prefix ownership or eviction decisions across TP ranks.
Chunks reuse the tier's host staging; no whole raw snapshot is allocated.
The cap includes compressed chunks being built and conservative object overhead.
"""
from collections import OrderedDict
from dataclasses import dataclass
import time
import zlib

CHUNK_BYTES = 1 << 20
WORKSPACE_BYTES = 4 << 20  # one input/output chunk, zlib state and Python metadata
ENTRY_OVERHEAD = 512
CHUNK_OVERHEAD = 192


@dataclass(frozen=True)
class Snapshot:
    chunks: tuple
    raw_bytes: int
    stored_bytes: int

    def restore(self, stage, write):
        """Decode through shared staging; `write(offset, n)` must fence its read."""
        off = filled = 0
        for n, packed in self.chunks:
            if not 0 < n <= min(CHUNK_BYTES, len(stage)) or off + n > self.raw_bytes:
                raise ValueError("invalid compressed snapshot chunk length")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(packed, n + 1)
            if len(raw) != n or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("invalid compressed snapshot payload")
            if filled + n > len(stage):
                write(off - filled, filled)
                filled = 0
            stage[filled:filled + n] = raw
            filled += n
            off += n
        if off != self.raw_bytes:
            raise ValueError("incomplete compressed snapshot")
        if filled:
            write(off - filled, filled)


class CompressedSnapshots:
    def __init__(self, capacity_bytes):
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("compressed snapshot capacity must be positive bytes")
        self.capacity_bytes = capacity_bytes
        self.entries = OrderedDict()
        self.stored_bytes = self.raw_bytes = 0
        self.hits = self.misses = self.evictions = self.rejections = 0
        self.compress_seconds = self.restore_seconds = 0.0

    def begin(self, raw_bytes):
        return Builder(self, raw_bytes)

    def _room(self, pending):
        while self.entries and self.stored_bytes + pending > self.capacity_bytes:
            self.discard(next(iter(self.entries)))
            self.evictions += 1

    def publish(self, key, snapshot):
        self.discard(key)
        if snapshot is not None:
            self._room(snapshot.stored_bytes)
            self.entries[key] = snapshot
            self.stored_bytes += snapshot.stored_bytes
            self.raw_bytes += snapshot.raw_bytes

    def get(self, key):
        snapshot = self.entries.get(key)
        if snapshot is None:
            self.misses += 1
        else:
            self.hits += 1
            self.entries.move_to_end(key)
        return snapshot

    def discard(self, key):
        snapshot = self.entries.pop(key, None)
        if snapshot is not None:
            self.stored_bytes -= snapshot.stored_bytes
            self.raw_bytes -= snapshot.raw_bytes

    def clear(self):
        released = self.stored_bytes
        self.entries.clear()
        self.stored_bytes = self.raw_bytes = 0
        return released


class Builder:
    def __init__(self, cache, raw_bytes):
        self.cache, self.raw_bytes = cache, raw_bytes
        self.chunks = []
        self.seen = 0
        self.stored = ENTRY_OVERHEAD
        self.rejected = False

    def add(self, data):
        if self.rejected:
            return
        start = time.perf_counter()
        try:
            for off in range(0, len(data), CHUNK_BYTES):
                part = data[off:off + CHUNK_BYTES]
                packed = zlib.compress(part, level=1)
                self.seen += len(part)
                self.stored += len(packed) + CHUNK_OVERHEAD
                # Incompressible data stays only on disk, never as a raw RAM copy.
                if self.stored >= min(self.raw_bytes, self.cache.capacity_bytes):
                    self.chunks.clear()
                    self.rejected = True
                    self.cache.rejections += 1
                    return
                self.cache._room(self.stored)
                self.chunks.append((len(part), packed))
        finally:
            self.cache.compress_seconds += time.perf_counter() - start

    def finish(self):
        if self.rejected:
            return None
        if self.seen != self.raw_bytes:
            raise ValueError("compressed snapshot input length mismatch")
        return Snapshot(tuple(self.chunks), self.raw_bytes, self.stored)
