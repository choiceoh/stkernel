"""The PLE n-gram table on SSD: one rank's row block, read row-wise with O_DIRECT.

Qwen3.8-Flash-Next's PLE table is 320,001,536 rows x 160 bytes of F8_E4M3 --
47.7 GiB -- behind one global scale. Serving keeps it in `VocabParallelEmbedding`,
so at TP=4 a rank already holds only its contiguous quarter (11.9 GiB). This
module takes that quarter off the device too: the builder in
tools/qwen38_ple_shard.py writes each rank's block to a raw file, and the
embedding method reads the rows a step needs -- 16 per token (`ngram_size` 3,
`heads_per_ngram` 8) -- through the O_DIRECT reader the DeepSeek-V4.1 engram
path already measured (dsv41_engram_io; 160-byte rows straddle sectors, which
`min_read_bytes` handles). Nothing else changes: `VocabParallelEmbedding.forward`
still masks the ids this rank does not own and all-reduces, and the layer still
applies the global scale afterwards.

Why the partition is not chosen here either: `VocabParallelEmbedding` hands
rank r the rows `[r * per, (r + 1) * per)` with `per = padded_vocab / tp_size`,
and refuses a vocab the tp_size does not divide. The checkpoint's 128 shards of
2,500,012 rows make 320,001,536, which 2 and 4 divide -- a rank's block is
exactly 32 whole checkpoint shards at TP=4 -- and `PleSsdConfig` repeats that
arithmetic so the builder cannot drift from the layer.

Reads this skips: a masked id arrives as row 0 (that is what the masking does),
so a decode step whose 32 slots are 3/4 foreign collapses to 8 owned rows plus
one read of row 0 -- `gather` reads each distinct row once.

    DENEB_PLE_SSD=1            take the table off the device
    DENEB_PLE_SSD_DIR=<dir>    where tools/qwen38_ple_shard.py put ple-r{r}of{W}.weight
    DENEB_PLE_SSD_QD=32        reader threads (each its own fd)
"""

from __future__ import annotations

import os

PLE_ROW_BYTES = 160


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


class PleSsdConfig:
    """The profile's knobs plus VocabParallelEmbedding's row partition."""

    def __init__(self, *, rank: int = 0, world_size: int = 1,
                 num_embeddings: int = 0, row_bytes: int = PLE_ROW_BYTES) -> None:
        self.enabled = _env("DENEB_PLE_SSD", "0") == "1"
        self.shard_dir = _env("DENEB_PLE_SSD_DIR", "")
        self.queue_depth = int(_env("DENEB_PLE_SSD_QD", "32"))
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside world_size {world_size}")
        if num_embeddings % world_size:
            raise ValueError(
                f"{num_embeddings} rows do not divide over {world_size} ranks; "
                f"VocabParallelEmbedding would have refused this too")
        self.rank = rank
        self.world_size = world_size
        self.num_embeddings = int(num_embeddings)
        self.row_bytes = int(row_bytes)
        if self.enabled and not self.shard_dir:
            raise ValueError(
                "DENEB_PLE_SSD=1 needs DENEB_PLE_SSD_DIR: the table is 47.7 GiB "
                "and this rank's block has to come from somewhere")

    @property
    def per_partition(self) -> int:
        return self.num_embeddings // self.world_size

    @property
    def vocab_start_idx(self) -> int:
        return self.rank * self.per_partition

    @property
    def vocab_end_idx(self) -> int:
        return self.vocab_start_idx + self.per_partition

    def rows_on_disk(self) -> int:
        return self.per_partition

    def shard_path(self) -> str:
        return os.path.join(self.shard_dir,
                            f"ple-r{self.rank}of{self.world_size}.weight")

    def describe(self) -> str:
        return (f"ple ssd rank={self.rank}/{self.world_size} "
                f"rows=[{self.vocab_start_idx},{self.vocab_end_idx}) "
                f"({self.rows_on_disk() * self.row_bytes / 2**30:.2f} GiB) "
                f"qd={self.queue_depth} file={self.shard_path()}")


class PleSsdTable:
    """`gather(local_ids) -> [n, 160] uint8`: this rank's rows, from its file.

    `local_ids` are what `VocabParallelEmbedding.forward` hands the quant
    method: already shifted to this rank's block, with the ids it does not own
    replaced by 0 (and zeroed again afterwards, so what row 0 holds does not
    matter). Distinct rows are read once; the staging buffer is pinned and
    grows to the largest batch seen.
    """

    def __init__(self, cfg: PleSsdConfig, reader) -> None:
        self.cfg = cfg
        self.reader = reader            # dsv41_engram_io.ShardReader
        self.stats = {"calls": 0, "rows": 0, "distinct": 0}
        self._staging = None
        if reader.n_rows != cfg.rows_on_disk():
            raise ValueError(
                f"{cfg.shard_path()} holds {reader.n_rows} rows; rank "
                f"{cfg.rank} of {cfg.world_size} owns {cfg.rows_on_disk()}")

    def gather(self, local_ids):
        import torch
        flat = local_ids.reshape(-1)
        n = int(flat.numel())
        ids_cpu = flat.to("cpu", dtype=torch.int64)
        uniq, inverse = torch.unique(ids_cpu, return_inverse=True)
        u = int(uniq.numel())
        self.stats["calls"] += 1
        self.stats["rows"] += n
        self.stats["distinct"] += u
        buf = self._staging
        if buf is None or buf.shape[0] < u:
            buf = torch.empty((max(u, 1024), self.cfg.row_bytes), dtype=torch.uint8,
                              pin_memory=torch.cuda.is_available())
            self._staging = buf
        rows = self.reader.gather(uniq.tolist())
        view = buf[:u]
        for i, blob in enumerate(rows):
            view[i] = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        dev_rows = view.to(flat.device, non_blocking=True)
        return dev_rows[inverse.to(flat.device)].reshape(*local_ids.shape, self.cfg.row_bytes)

    def close(self) -> None:
        self.reader.close()
