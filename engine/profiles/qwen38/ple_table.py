"""Qwen3.8-Flash-Next's PLE table served off the SSD (profile): the rank's vocabulary range of the hashed n-gram table
as one raw file beside its rank file, rows read by id when a step needs them, never held in the arena.

The table is 47.68 GiB of e4m3 rows (320,001,536 rows of 160 bytes under one scalar scale), vocabulary-parallel over
the four ranks: 11.92 GiB a rank, the second-largest resident item after the experts (plan.py's census). A token
reads (ngram_size - 1) * heads_per_ngram = 16 rows; a decode step of 4 rows x 2 tokens reads 128, of which this rank
holds about a quarter; a 32K-token prefill chunk about 131K. The operator's decision (2026-09-18): the table lives on
the SSD, not in memory.

    ple-r{r}of4.weight   rows [32 shards x 2,500,012, 160] e4m3: the checkpoint's `ngram_embedding.shard_N` tensors
                         32r..32r+31 back to back (engine/profiles/qwen38/preshard.py writes it; specs.ple_shards
                         names them), so a row's byte offset is row * 160 and the file is the shards' bytes exactly
    ple-r{r}of4.json     what the file is: rows, width, shards, the scale, the layout marker, its sha256

Reads are buffered `pread`s from a thread pool -- measured on srv2's NVMe (2026-09-18, this file's rank-0 table):
128 random rows in 1.0 ms with 8 threads, 20,000 rows in 71 ms; O_DIRECT sector reads were 3-11x slower, and the
page cache is what makes a repeated n-gram free. A captured decode step cannot read the host: its rows are gathered
before the replay into a staging buffer that is the graph's static input (PLEStaging; net.stage_ple from
decode_graphs.TargetGraphs.run), and the graph reads the rows from there. An eager step (prefill) gathers into a
fresh tensor (net._ple_embed). Rows of other ranks are zero bytes -- e4m3 zero -- so the step's all-reduce sums the one
rank that holds each row, as the arena-resident table's gather did.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from engine.profiles.qwen38 import facts

THREADS = 8                     # the srv2 measurement's best for a decode step's rows; more threads lost to the GIL
SPLIT_AT = 64                   # rows below which one thread reads them all (a decode step's rows are few)


def local_rows(rows: np.ndarray, rank: int, per_rank: int) -> "tuple[np.ndarray, np.ndarray]":
    """Table rows (any shape, int64) -> (this rank's row ids, the mask of rows this rank holds): the table is
    vocabulary-parallel, rank r holding rows [r * per_rank, (r + 1) * per_rank)."""
    local = rows - rank * per_rank
    mine = (local >= 0) & (local < per_rank)
    return np.where(mine, local, 0), mine


class PLETable:
    """One rank's table file: `gather` reads rows by local id."""

    def __init__(self, path: "str | Path", *, rows: int, width: int, scale: float, threads: int = THREADS):
        self.path = Path(path)
        self.rows, self.width, self.scale = int(rows), int(width), float(scale)
        size = self.path.stat().st_size
        if size != self.rows * self.width:
            raise ValueError(f"{self.path.name}: {size:,} bytes, not {self.rows:,} rows of {self.width}")
        self.fd = os.open(self.path, os.O_RDONLY)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.threads = max(1, int(threads))
        self._pool = None
        self.reads = self.rows_read = 0                   # counters: gathers, and distinct rows read

    @classmethod
    def open(cls, ranks_dir: "str | Path", rank: int, F: facts.Facts, *, world: int = facts.TP,
             threads: int = THREADS) -> "PLETable":
        """The rank's table beside its rank file, its sidecar checked against what this profile expects (D3): the
        layout marker, the rank, the row count of the rank's vocabulary range, the row width, the shards in order."""
        from engine.profiles.qwen38 import specs
        ranks_dir = Path(ranks_dir)
        sidecar = json.loads((ranks_dir / facts.ple_sidecar(rank, world)).read_text())
        L = F.ple_layers[0]
        expect = dict(layout=F.weight_layout, rank=rank, world=world, rows=F.ple_rows_per_rank, width=F.ple_head_dim,
                      dtype="F8_E4M3", shards=specs.ple_shards(F, L, rank, world))
        for key, want in expect.items():
            if sidecar.get(key) != want:
                raise ValueError(f"{facts.ple_sidecar(rank, world)}: {key} is {sidecar.get(key)!r}, this profile expects "
                                 f"{want!r}; regenerate the table files with engine/profiles/qwen38/preshard.py")
        return cls(ranks_dir / facts.ple_file(rank, world), rows=sidecar["rows"], width=sidecar["width"],
                   scale=sidecar["scale"], threads=threads)

    def gather(self, rows: np.ndarray) -> np.ndarray:
        """uint8 [N, width]: the rows (int64 [N], local ids, repeats allowed) as stored."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = np.empty((rows.shape[0], self.width), dtype=np.uint8)
        if rows.shape[0] == 0:
            return out
        uniq, inverse = np.unique(rows, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.rows:
            raise IndexError(f"table rows {int(uniq[0])}..{int(uniq[-1])} outside 0..{self.rows - 1}")
        got = np.empty((uniq.shape[0], self.width), dtype=np.uint8)
        n = uniq.shape[0]
        if n < SPLIT_AT or self.threads == 1:
            self._read(uniq, got, 0, n)
        else:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.threads)
            per = -(-n // self.threads)
            list(self._pool.map(lambda lo: self._read(uniq, got, lo, min(n, lo + per)), range(0, n, per)))
        self.reads += 1
        self.rows_read += n
        out[:] = got[inverse.reshape(-1)]
        return out

    def _read(self, uniq, got, lo: int, hi: int) -> None:
        fd, width = self.fd, self.width
        for i in range(lo, hi):
            raw = os.pread(fd, width, int(uniq[i]) * width)
            if len(raw) != width:
                raise OSError(f"{self.path.name}: short read at row {int(uniq[i])}")
            got[i] = np.frombuffer(raw, dtype=np.uint8)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class PLEStaging:
    """The rows a captured step reads: [capacity, heads, width] uint8 on the device -- the graphs' static input, every
    graph reading its own leading rows -- and the pinned host mirror `fill` writes before `upload` copies it in."""

    def __init__(self, capacity: int, heads: int, width: int, device):
        import torch
        self.capacity, self.heads, self.width = int(capacity), int(heads), int(width)
        pin = torch.device(device).type == "cuda"
        self.host = torch.zeros(self.capacity, self.heads, self.width, dtype=torch.uint8, pin_memory=pin)
        self.device = torch.zeros(self.capacity, self.heads, self.width, dtype=torch.uint8, device=device)
        self.filled = 0

    def fill(self, table: PLETable, local: np.ndarray, mine: np.ndarray) -> int:
        """The rows of `local` [n, heads] this rank holds (`mine`) read into the mirror; the rest zero."""
        n = local.shape[0]
        if n > self.capacity or local.shape[1] != self.heads:
            raise ValueError(f"a staged step has at most {self.capacity} rows of {self.heads} table rows")
        host = self.host[:n].numpy()
        host[...] = 0
        if mine.any():
            host[mine] = table.gather(local[mine])
        self.filled = n
        return n

    def upload(self) -> None:
        """The filled rows to the device, on the current stream (before the replay that reads them)."""
        n = self.filled
        self.device[:n].copy_(self.host[:n], non_blocking=True)


def write_sidecar(path: "str | Path", **fields) -> None:
    Path(path).write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n")


__all__ = ["THREADS", "PLETable", "PLEStaging", "local_rows", "write_sidecar"]
