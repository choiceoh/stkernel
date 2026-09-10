"""Aligned batched reads of engram rows out of per-rank shard files.

This is the half of the engram demotion that has nothing to do with the model.
It answers one question -- can a rank pull its rows off the NVMe inside a decode
step -- and it answers it standalone, which is why it is a separate file from
the layer that calls it.

Why not mmap. A row is fetched by hash, the hashes are spread over prime-sized
buckets by construction, and there are 12 of them per token per rank. Demand
paging serializes one fault per row: 384 rows for a batch-32 step against srv4's
measured QD1 latency of 62 us is 24 ms, on a step that wants to be 20. The same
drive served 157,357 IOPS at QD32 (O_DIRECT, 4 KiB random) -- 2.4 ms for the
same 384 rows. The gap is entirely the issue path, so the issue path is the
thing this file implements: a fixed pool of threads, each holding its own fd and
its own aligned buffer, kept full.

Alignment. A row is EMB_ROW_BYTES = 256 and O_DIRECT wants offsets and lengths
on the logical block size, which is 512 on this fleet's drives (`nvme list`
reports `512 B + 0 B`). Row r starts at r * 256, so it lies entirely inside the
single 512-byte sector at (r * 256) & ~511 -- an odd row occupies the sector's
upper half, an even row its lower half, and no row ever straddles. One aligned
sector read per row is therefore both correct and the smallest legal read, which
is where the 2x read amplification comes from rather than the 16x a 4 KiB page
would cost. READ_BYTES is still a parameter because that claim is worth
measuring rather than believing.
"""

from __future__ import annotations

import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# One table row: engram_head_dim 256 stored F8_E4M3, so one byte per element.
EMB_ROW_BYTES = 256
# The scale row alongside it, [.., 8] F8_E8M0. Scales are NOT read from disk --
# see the note on resident scales in dsv41_engram.py -- and the constant is here
# only so the shard builder and the reader agree on what was left behind.
SCALE_ROW_BYTES = 8

_LOGICAL_BLOCK = 512


def _align_down(value: int, to: int) -> int:
    return value & ~(to - 1)


class Gather:
    """Reads in flight. `wait()` collects them, in the order they were asked for.

    Deliberately not a Future: a step submits one of these and waits once, and
    exposing the underlying futures would invite waiting on them piecemeal --
    which reintroduces exactly the per-item bookkeeping the striding avoids.
    """

    __slots__ = ("_rows", "_futures", "_result")

    def __init__(self, rows, futures) -> None:
        self._rows = rows
        self._futures = futures
        self._result = None

    def wait(self) -> list:
        if self._result is None:
            merged = {}
            for future in self._futures:
                merged.update(future.result())
            self._result = [merged[row] for row in self._rows]
        return self._result

    @property
    def done(self) -> bool:
        """True when no wait would block. For instrumentation, not control."""
        return all(f.done() for f in self._futures)

    def __len__(self) -> int:
        return len(self._rows)


class ShardReader:
    """One rank's slice of one engram table, read row-wise with O_DIRECT.

    `path` holds the rows of the (n-gram size, head) bucket ranges this rank
    owns, densely and in the order the builder wrote them; the caller maps a
    global hash id onto a local row before asking for it. Keeping that mapping
    outside means this class never has to know about primes.
    """

    def __init__(self, path: str, *, queue_depth: int = 32,
                 read_bytes: int = _LOGICAL_BLOCK) -> None:
        if read_bytes < EMB_ROW_BYTES or read_bytes % _LOGICAL_BLOCK:
            raise ValueError(
                f"read_bytes must be a multiple of {_LOGICAL_BLOCK} and at least "
                f"{EMB_ROW_BYTES}, got {read_bytes}")
        self.path = path
        self.read_bytes = read_bytes
        self.queue_depth = max(1, int(queue_depth))
        self.n_rows = os.path.getsize(path) // EMB_ROW_BYTES
        # Each worker owns an fd and a page-aligned buffer for its whole life.
        # A shared fd would still be correct -- pread carries its own offset --
        # but per-thread fds keep the kernel's per-file lock off the hot path.
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(
            max_workers=self.queue_depth,
            thread_name_prefix="engram-io",
            initializer=self._init_thread,
        )
        self._closed = False

    def _init_thread(self) -> None:
        self._local.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
        # mmap of an anonymous region is page-aligned, which satisfies
        # O_DIRECT's buffer requirement without ctypes/posix_memalign.
        self._local.buf = mmap.mmap(-1, self.read_bytes)

    def submit(self, rows) -> "Gather":
        """Issue the reads and return WITHOUT waiting for them.

        This is the API the layer actually wants. The hash ids for a whole step
        are known before layer 0 runs -- the reference `NgramHashState.forward`
        takes token ids and no hidden state -- so the reads belong at the top of
        the step and the wait belongs at the engram layer. Calling `gather` and
        measuring how long it blocks answers a question nobody asked: what
        matters is what is LEFT after the compute in between, and a blocking API
        cannot express that, let alone measure it.
        """
        rows = list(rows)
        if not rows:
            return Gather(rows, [])
        windows = {}
        for row in rows:
            if not 0 <= row < self.n_rows:
                raise IndexError(f"row {row} outside shard of {self.n_rows} rows")
            windows.setdefault(_align_down(row * EMB_ROW_BYTES, self.read_bytes),
                               []).append(row)

        # ONE task per thread, not one per read. This is the whole difference
        # between 52.6K and the drive's 157K IOPS: a step asks for ~384 rows, and
        # dispatching 384 futures through the pool spends more time in the GIL
        # and in future bookkeeping than in the kernel. Handing each thread a
        # stride of the window list lets it run the same tight preadv loop the
        # raw O_DIRECT benchmark ran, and the queue stays just as full.
        items = list(windows.items())
        stride = min(self.queue_depth, len(items))
        futures = [self._pool.submit(self._read_chunk, items[i::stride])
                   for i in range(stride)]
        return Gather(rows, futures)

    def gather(self, rows) -> list:
        """Rows in the order asked for. Duplicates are read once.

        Coalescing is by aligned window rather than by row: two rows of the same
        sector are one read, which is not a rare case here because the builder
        writes a bucket range contiguously and consecutive n-gram hits inside one
        range land near each other more often than a uniform hash would suggest.
        """
        return self.submit(rows).wait()

    def _read_chunk(self, windows) -> dict:
        fd, buf, width = self._local.fd, self._local.buf, self.read_bytes
        out = {}
        for base, rows in windows:
            got = os.preadv(fd, [buf], base)
            if got <= 0:
                raise OSError(f"O_DIRECT read at {base} returned {got}")
            for row in rows:
                start = row * EMB_ROW_BYTES - base
                # The buffer is reused, so a short tail read would otherwise hand
                # back whatever the previous read left in those bytes.
                if start + EMB_ROW_BYTES > got:
                    raise OSError(
                        f"row {row} wants bytes {start}..{start + EMB_ROW_BYTES} "
                        f"of a {got}-byte read at {base}")
                out[row] = buf[start:start + EMB_ROW_BYTES]
        return out

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True)

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
