"""Engram conditional memory with the two lookup tables on SSD.

The two tables are `layers.{1,14}.engram.embed.weight`, each [384,006,168 x 256]
F8_E4M3, and together they are 188.8 GiB -- 39.7% of a checkpoint that is 475.2
GiB against a fleet holding ~480 GiB. Keeping them resident is not a tuning
choice this profile declines; it is arithmetic the fleet loses. What makes
demoting them viable rather than merely necessary is that the model card calls
them "sparsely accessed via token-based lookup", and it is: 24 rows per token
per table, 48 across both.

## The partition is DeepSeek's, not one of ours

An earlier draft of this module split the tables by the 24 disjoint prime bucket
ranges `EngramLayout` hands out -- six per rank, a fixed six lookups each. That
was wrong in the way that matters: the reference `ParallelEngramEmbedding`
shards by CONTIGUOUS ROW BLOCKS,

    part = ceil(num_embeddings / world_size)
    rank r owns [r * part, (r + 1) * part)

masks the ids outside its block, looks up the rest, zeroes the masked rows and
`dist.all_reduce`s the result. A checkpoint split any other way cannot be fed to
that module at all. So this file follows it exactly, and two differences from
the bucket-range scheme change the I/O model rather than just the bookkeeping:

  - a rank's per-token read count is no longer a fixed 6. It is Binomial(24,
    1/4) per table -- mean 6, sd 2.1. At batch 32 over two tables a step is
    Binomial(1536, 1/4), mean 384 and sd 17, so the measured 384 rows/step
    stands and its spread is small; a per-TOKEN bound does not exist.
  - the rows a rank does not own cost it nothing here. The reference still runs
    `F.embedding` over all 24 with the out-of-range ones pointed at row 0 and
    then zeroed; reading those off an SSD would be 4x the I/O for bytes that get
    multiplied by zero, so this path skips them and fills zeros. The result is
    identical because the reference zeroes them too.

## Scales stay resident

`embed.scale` is [rows, 8] F8_E8M0 -- 256 / fp8_block_size 32 -- 2.86 GiB per
table, 1.43 GiB per rank for both. Reading them from disk would double the IOPS
for 3% of the bytes, so the builder leaves them behind. Dequantization is the
reference's: `values.float().unflatten(-1, (-1, 32)) * scales.float()...`.

## Latency

A rank needs ~384 rows for a batch-32 step. Demand paging costs one fault each,
serialized: 384 x 62 us measured at QD1 is 24 ms on a ~20 ms step. Batched
O_DIRECT through dsv41_engram_io measured 4.32 ms at QD32 on srv4 (88,790
IOPS/rank).

That blocking figure is not what a step pays, because the reads need not block.
The hash ids depend on no hidden state -- the reference
`NgramHashState.forward(input_ids, start_pos, token_mask)` takes token ids only
-- so they are issued at layer 0 and collected at the layer. The RESIDUAL stall
is 0.07 ms p95 at layer 14, which has 13 of 40 layers of cover. Layer 1 has one
layer and does not clear it: 4.19 ms p95 is left over, so its rows want fetching
during the PREVIOUS step off the DSpark draft block (dspark_block_size 5), whose
ids already determine the hashes. Measured that way both tables cost 0.09 ms
p95 -- 0.5% of a 20 ms step -- at the price of discarding 26% of the layer-1
reads at 75% draft acceptance. ENGRAM_PREFETCH_LAYERS names the layer whose
issue point is early enough to self-cover; anything earlier is the drafter's.

## What is established

`probes/dsv41_engram_diff.py` builds a synthetic table, runs DeepSeek's own
`ParallelEngramEmbedding` against it at world_size 4, runs this path against the
same table sharded to files, and requires the two readouts to be bit-identical.
That is a correctness result about this file -- not about serving. Nothing here
has run inside a model, because no model file exists yet: `deepseek_v41` is not
registered on this fleet. That is a thing to WRITE (new files plus a .pth
calling `ModelRegistry.register_model`), not a thing to wait for -- see
profiles/dsv41.env.
"""

from __future__ import annotations

import os

# (engram_max_ngram_size 4 - 1) n-gram sizes x engram_n_heads 8, per table.
HASH_COLS = 24
# engram_head_dim, stored F8_E4M3.
EMB_ROW_DIM = 256
# weight_block_size [32, 32]; a scale row is EMB_ROW_DIM // this wide.
FP8_BLOCK = 32
SCALE_COLS = EMB_ROW_DIM // FP8_BLOCK


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


class EngramConfig:
    """The profile's engram knobs plus the reference's row-block partition.

    Reading the knobs here rather than at each call site is what makes them
    profile keys with a reader -- tests/test_logic.py holds every profile key to
    having one, on the grounds that an unread key is either dead config or a
    carrier nobody wrote.
    """

    def __init__(self, *, rank: int = 0, world_size: int = 1,
                 num_embeddings: int = 0) -> None:
        self.backend = _env("ENGRAM_BACKEND", "ssd").strip().lower()
        if self.backend not in ("ssd", "device"):
            raise ValueError(
                f"ENGRAM_BACKEND must be 'ssd' or 'device', got {self.backend!r}")
        self.shard_dir = _env("ENGRAM_SHARD_DIR", "")
        self.queue_depth = int(_env("ENGRAM_QUEUE_DEPTH", "32"))
        self.prefetch_layer = int(_env("ENGRAM_PREFETCH_LAYERS", "14"))
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside world_size {world_size}")
        self.rank = rank
        self.world_size = world_size
        self.num_embeddings = int(num_embeddings)
        if self.backend == "ssd" and not self.shard_dir:
            raise ValueError(
                "ENGRAM_BACKEND=ssd needs ENGRAM_SHARD_DIR; the tables are "
                "188.8 GiB and there is nowhere else on this fleet to put them")

    # -- the reference's partition, arithmetic included --------------------
    @property
    def part_num_embeddings(self) -> int:
        """ceil(rows / world_size). The LAST rank's block runs past the table.

        The reference allocates the same padded block on every rank, so the
        overhang exists there too and is simply never indexed -- no hash id can
        reach it, because ids are taken modulo a prime below the table size.
        Keeping the same arithmetic is what makes a shard built here loadable
        there.
        """
        return (self.num_embeddings + self.world_size - 1) // self.world_size

    @property
    def vocab_start_idx(self) -> int:
        return self.rank * self.part_num_embeddings

    @property
    def vocab_end_idx(self) -> int:
        return self.vocab_start_idx + self.part_num_embeddings

    def owns(self, index: int) -> bool:
        return self.vocab_start_idx <= index < self.vocab_end_idx

    def local_row(self, index: int) -> int:
        """Global hash id -> row inside this rank's shard. Caller checks owns()."""
        return index - self.vocab_start_idx

    def rows_on_disk(self) -> int:
        """Rows this rank stores: its block, clipped to the real table.

        The padded overhang is not written; asking for it would be asking for a
        row the reference never indexes either.
        """
        return max(0, min(self.vocab_end_idx, self.num_embeddings)
                   - self.vocab_start_idx)

    def shard_path(self, layer_id: int) -> str:
        return os.path.join(
            self.shard_dir,
            f"engram-l{layer_id}-r{self.rank}of{self.world_size}.weight")

    def scale_path(self, layer_id: int) -> str:
        return os.path.join(
            self.shard_dir,
            f"engram-l{layer_id}-r{self.rank}of{self.world_size}.scale")

    def describe(self) -> str:
        return (f"engram backend={self.backend} "
                f"rank={self.rank}/{self.world_size} "
                f"rows={self.rows_on_disk()}/{self.num_embeddings} "
                f"[{self.vocab_start_idx},{self.vocab_end_idx}) "
                f"qd={self.queue_depth} prefetch_from_layer={self.prefetch_layer}")


class ShardEmbedding:
    """The SSD-backed equivalent of the reference `ParallelEngramEmbedding`.

    Same partition, same dequantization, same zeroing -- the only difference is
    where the rows come from and that the rows this rank does not own are never
    read. The reference reads them (pointed at row 0) and then overwrites them
    with zeros, so skipping the read is not an approximation.

    The caller still owes the cross-rank sum. The reference ends in
    `dist.all_reduce(values)`; this returns one rank's contribution, zeros
    elsewhere, which is exactly what that all-reduce expects to add up.
    """

    def __init__(self, cfg: EngramConfig, reader, scale) -> None:
        self.cfg = cfg
        self.reader = reader          # dsv41_engram_io.ShardReader
        self.scale = scale            # [rows_on_disk, SCALE_COLS] e8m0, resident

    def _owned(self, flat_ids):
        """(position, local row) for the ids this rank holds."""
        cfg = self.cfg
        return [(i, cfg.local_row(v)) for i, v in enumerate(flat_ids)
                if cfg.owns(v)]

    def submit(self, flat_ids):
        """Issue this rank's reads for `flat_ids` and return without waiting."""
        owned = self._owned(flat_ids)
        return owned, self.reader.submit(row for _, row in owned)

    def readout(self, pending, shape):
        """(owned, gather) -> [*shape, EMB_ROW_DIM] bf16, zero where not owned.

        `shape` is the hash-id tensor's shape, so the result matches the
        reference's `self.embed(hash_ids)` before its `.flatten(-2)`.
        """
        import torch

        owned, gather = pending
        rows = gather.wait()
        n = 1
        for d in shape:
            n *= d
        raw = torch.zeros(n, EMB_ROW_DIM, dtype=torch.uint8)
        idx = torch.zeros(n, dtype=torch.long)
        keep = torch.zeros(n, dtype=torch.bool)
        for (pos, row), data in zip(owned, rows):
            raw[pos] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            idx[pos] = row
            keep[pos] = True

        values = raw.view(torch.float8_e4m3fn)
        scales = self.scale[idx]
        # the reference's dequantization, unchanged
        out = values.float().unflatten(-1, (-1, FP8_BLOCK)) * scales.float().unsqueeze(-1)
        out = out.flatten(-2).to(torch.bfloat16)
        out = out.masked_fill(~keep.unsqueeze(-1), 0)
        return out.reshape(*shape, EMB_ROW_DIM)
