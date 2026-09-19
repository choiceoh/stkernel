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

Reads go through a read-only mapping of the file: a gather is one `np.take` over it -- a C loop of row copies with
the GIL released -- and the page cache is what makes a repeated n-gram free, as it was for the buffered `pread`s this
replaced (O_DIRECT sector reads were 3-11x slower on srv2's NVMe, 2026-09-18). The `pread` form read a row a Python
iteration: 3.5 us a row on srv2 whatever the cache held (20,000 rows in 71 ms with 8 threads, so a 32K-token chunk's
131K rows about half a second of host time a rank), and past SPLIT_AT its threads queued on the GIL. Measured on the
development PC (WSL2 ext4, a 2 GiB synthetic table of 160-byte rows, 2026-09-19 -- not a fleet number): a warm cache
gathers 131,072 random rows in 1.4 ms (10 ms on one thread) where the `pread` loop took 4.6 s, 128 rows in 0.38 ms
(the pool's dispatch; 7 us on one thread) against 4.2 ms, 32 rows in 3 us against 30; with it a 32,256-token chunk's
whole lookup on the host -- hash, split by rank, gather, scatter into the step's rows -- is about 10 ms. Cold (the
file's pages evicted, a process a method) both forms wait on the disk at one thread, 8 threads read 20,000 rows in
552 ms against 857 ms, and the take keeps scaling (32 threads 183 ms) where the loop had stopped at 8. THREADS and
SPLIT_AT are still srv2's numbers for the loop; sweeping them for the take is a fleet box's measurement.

A mapping turns a failed read into SIGBUS rather than an OSError: a table whose disk returns an error kills the
process without a Python traceback. The file's size is checked when it is opened and the file is never written while
it is served.

A captured decode step cannot read the host: its rows are gathered before the replay into a staging buffer that is the
graph's static input (PLEStaging; net.stage_ple from decode_graphs.TargetGraphs.run), and the graph reads the rows from
there. An eager step (prefill) gathers into a fresh tensor (net._ple_embed). Rows of other ranks are zero bytes -- e4m3
zero -- so the step's all-reduce sums the one rank that holds each row, as the arena-resident table's gather did.
"""
from __future__ import annotations

import json
import os  # noqa: F401 -- the reader is lookup_table's; tests patch its `os` through this name
from pathlib import Path

import numpy as np

from engine.modules.lookup_table import SPLIT_AT, THREADS, MappedTable  # noqa: F401 -- SPLIT_AT: the split gather uses
from engine.profiles.qwen38 import facts


def local_rows(rows: np.ndarray, rank: int, per_rank: int) -> "tuple[np.ndarray, np.ndarray]":
    """Table rows (any shape, int64) -> (this rank's row ids, the mask of rows this rank holds): the table is
    vocabulary-parallel, rank r holding rows [r * per_rank, (r + 1) * per_rank)."""
    local = rows - rank * per_rank
    mine = (local >= 0) & (local < per_rank)
    return np.where(mine, local, 0), mine


class PLETable(MappedTable):
    """One rank's PLE table file: `gather` and `close` are the lookup-table module's (a row by local id off a
    read-only mapping); this adds the scale the rows are stored under and the sidecar `open` checks."""

    def __init__(self, path: "str | Path", *, rows: int, width: int, scale: float, threads: int = THREADS):
        super().__init__(path, width=width, rows=rows, threads=threads)
        self.scale = float(scale)

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
