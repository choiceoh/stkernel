"""Giant embedding tables that do not fit in the box, read row-wise from NVMe
(module: DSv4.1 engram, Qwen3.8 PLE).

A gather is one `np.take` over a read-only mapping of the rank's file -- a C
loop of row copies with the GIL released -- and the page cache is what makes a
repeated n-gram free. That is the form engine/profiles/qwen38/ple_table.py
serves Qwen3.8's PLE table with; this is the same form under the engram's rows,
which is what engine/profiles/dsv41/engram.py reads through. The rows are bytes
here: what they mean -- the scale, the dtype, which vocabulary range a rank
holds -- belongs to the profile, beside the checkpoint that states it.

What it replaces: `dsv41_engram_io.ShardReader` (O_DIRECT, one aligned sector a
row, a fixed pool of threads each holding its own fd and aligned buffer). That
file left the tree with the vLLM overlay (#1152) and nothing took its place, so
`engram.attach` died with FileNotFoundError before a single row was read.

The two forms were argued on different boxes, and neither argument is a
measurement of the other:

    the retired reader   "demand paging serializes one fault per row: 384 rows
                         for a batch-32 step against srv4's measured QD1
                         latency of 62 us is 24 ms, on a step that wants to be
                         20. The same drive served 157,357 IOPS at QD32
                         (O_DIRECT, 4 KiB random)". Its thread pool was the
                         issue path that filled that queue, and 256-byte rows
                         never straddle a 512-byte sector, so a row was one
                         sector and the amplification 2x.
    ple_table's record   O_DIRECT sector reads were 3-11x slower than the
                         buffered reads that replaced them on srv2's NVMe
                         (2026-09-18), and the take's threads overlap their
                         page faults where a `pread` loop's threads queue on
                         the GIL: cold, 8 threads read 20,000 rows in 552 ms
                         against 857 ms and the take keeps scaling to 32
                         threads (183 ms); warm, 131,072 random rows in 1.4 ms
                         against 4.6 s (development PC, 2026-09-19 -- not a
                         fleet number, and 160-byte PLE rows, not these).

UNMEASURED: the engram's own rows (256 bytes, the 12 a token takes per rank) off
srv4's drive through either form. Nothing here claims a step-time number, and
the ledger has no DSv4.1 serving entry to put one beside.
"""
from __future__ import annotations

import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


THREADS = 8                     # ple_table's value: srv2's best for the pread loop it replaced, unswept for the take
SPLIT_AT = 64                   # rows below which one thread reads them all (a decode step's rows are few)


class MappedTable:
    """One rank's table file: `gather` reads rows by local id.

    `rows` is read off the file's size when it is not given -- the shard files are the rows and nothing else -- and a
    file that does not hold whole rows is refused rather than rounded down.
    """

    def __init__(self, path: "str | Path", *, width: int, rows: "int | None" = None, threads: int = THREADS):
        self.path = Path(path)
        self.width = int(width)
        if self.width <= 0:
            raise ValueError(f"a row is {self.width} bytes wide")
        size = self.path.stat().st_size
        self.rows = size // self.width if rows is None else int(rows)
        if size != self.rows * self.width:
            raise ValueError(f"{self.path.name}: {size:,} bytes, not {self.rows:,} rows of {self.width}")
        self.fd = os.open(self.path, os.O_RDONLY)
        self._map = self._rows = None
        if size:
            self._map = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
            if hasattr(self._map, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                self._map.madvise(mmap.MADV_RANDOM)       # a row's neighbours are other n-grams: no readahead
            self._rows = np.frombuffer(self._map, dtype=np.uint8).reshape(self.rows, self.width)
        self.threads = max(1, int(threads))
        self._pool = None
        self.reads = self.rows_read = 0                   # counters: gathers, and rows read

    def gather(self, rows) -> np.ndarray:
        """uint8 [N, width]: the rows (int64 [N], local ids, repeats allowed) as stored -- a copy, not a view."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        n = rows.shape[0]
        out = np.empty((n, self.width), dtype=np.uint8)
        if n == 0:
            return out
        lo, hi = int(rows.min()), int(rows.max())
        if lo < 0 or hi >= self.rows:
            raise IndexError(f"table rows {lo}..{hi} outside 0..{self.rows - 1}")
        # mode="clip": the ids were just checked, and "raise" makes numpy buffer `out`
        if n < SPLIT_AT or self.threads == 1:
            np.take(self._rows, rows, axis=0, out=out, mode="clip")
        else:
            # the take releases the GIL, so the threads' page faults overlap: what a cold cache waits for
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.threads)
            per = -(-n // self.threads)
            list(self._pool.map(lambda at: np.take(self._rows, rows[at:at + per], axis=0, out=out[at:at + per],
                                                   mode="clip"), range(0, n, per)))
        self.reads += 1
        self.rows_read += n
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._rows = None                                 # the view holds the mapping's buffer: it goes first
        if self._map is not None:
            self._map.close()
            self._map = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "MappedTable":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class SSDEngramLookup:
    """One table's rows, fetched per step instead of held.

    A row is e4m3 values with one e8m0 scale per `block_size` of them, and the scales stay resident: the same
    dequantisation modules/ngram_embedding.block_fp8_rows does over a resident table, with the bytes coming off the
    mapping instead of out of the arena (the vendor casts to bf16 there, so the engram's value path is bf16 whatever
    the activations are).
    """

    def __init__(self, weight_path: "str | Path", scale, block_size: int, *, row_bytes: int,
                 threads: int = THREADS):
        self.table = MappedTable(weight_path, width=row_bytes, threads=threads)
        self.row_bytes = self.table.width
        self.scale = scale                      # resident, [rows, dim/block]
        self.block_size = int(block_size)
        self.rows_read = 0
        self.calls = 0

    @property
    def n_rows(self) -> int:
        return self.table.rows

    def rows(self, local_indices):
        """[N, dim] bf16 for a flat int64 tensor of local row ids. Repeats are read once."""
        import torch

        flat = local_indices.reshape(-1)
        uniq, inverse = torch.unique(flat, return_inverse=True)
        raw = self.table.gather(uniq.detach().cpu().numpy())
        self.rows_read += raw.shape[0]
        self.calls += 1
        table = (torch.from_numpy(raw)                    # the gather copied: nothing here holds the mapping
                 .to(local_indices.device, non_blocking=False)
                 .view(torch.float8_e4m3fn))
        scales = self.scale.index_select(0, uniq)
        vals = (table.float().unflatten(-1, (-1, self.block_size))
                * scales.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        # the width is named, not inferred: a step with no rows has nothing for a -1 to be read off
        return vals.index_select(0, inverse).view(*local_indices.shape, self.row_bytes)

    def close(self) -> None:
        self.table.close()


__all__ = ["THREADS", "SPLIT_AT", "MappedTable", "SSDEngramLookup"]
