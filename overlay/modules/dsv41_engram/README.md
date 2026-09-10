# dsv41_engram

Engram conditional memory with the two lookup tables on SSD.

Not an optimization. The checkpoint is 475.2 GiB against a fleet holding ~480
GiB, and 189.1 GiB of it is `layers.{1,14}.engram.embed.weight` -- two
[384,006,168 x 256] F8_E4M3 tables. Demoting them is what makes the remaining
286.1 GiB (71.5 GiB/node at TP=4) a number the fleet can hold at all.

## Rows

| | |
|---|---|
| per token | 48 (3 n-gram sizes x 8 heads x 2 tables) |
| per token per rank | 12 |
| row | 256 B weight + 8 B scale |
| scales | stay resident: 5.72 GiB total, 1.43 GiB/rank |
| SSD per node | 47.3 GiB |

The split is the reference's: `ParallelEngramEmbedding` shards by CONTIGUOUS
ROW BLOCKS of `ceil(num_embeddings / world_size)`, masks ids outside its block,
zeroes them and `dist.all_reduce`s. An earlier version of this file said the
split followed the 24 disjoint prime bucket ranges `EngramLayout` hands out --
it does not, and a checkpoint split that way could not be fed to the reference
module at all. The I/O consequence is that a rank's per-token read count is
Binomial(24, 1/4), mean 6 and sd 2.1, rather than a fixed 6; at batch 32 over
two tables that is mean 384 and sd 17.

**Verified bit-identical.** `probes/dsv41_engram_diff.py` extracts
`ParallelEngramEmbedding` verbatim from the checkpoint's `inference/model.py`,
runs it over a synthetic table at world_size 4, runs this module's SSD path over
the same table sharded to files, and requires `torch.equal` on the summed
readouts -- not a tolerance, because a wrong row in a hash table is uncorrelated
with the right one and would pass any tolerance loose enough to be useful.
393,216 elements, 390,198 non-zero, exact. It covers the block boundaries, the
ragged last block, and ids a rank does not own.

This does NOT solve srv1 by itself -- see below.

## Measured (srv4, 2026-09-10, no GPU, no model)

`probes/dsv41_engram_probe.py`, 150 steps, batch 32 = 384 rows/rank/step,
512 B O_DIRECT reads, QD32. Drive `ESL04TBTLCZ`, 4 TB, 512 B format.

Two shards were measured. The issue-path table below is from a 4 GiB synthetic
file; everything after it was re-run on the **real** rank-0 layer-1 shard --
22.89 GiB, 96,001,542 rows, written by `tools/dsv41_engram_shard.py build` from
the downloaded checkpoint and verified against it (1,028 rows, 0 problems).
That matters because a 4 GiB file touches 0.1% of the drive: the FTL can hold
its whole mapping in SRAM, and a random read never misses it. The real shard
spans 5.7x more LBA space, which is the case a rank actually runs.

**Issue path first.** Same drive, same reads; only the dispatch changed.

| dispatch | blocking fetch | IOPS/rank |
|---|---|---|
| one pool task per read | 7.30 ms | 52,607 |
| one task per thread (strided) | **4.32 ms** | **88,790** |
| the same, on the real 22.89 GiB shard | 4.35 ms | 88,198 |

The 5.7x larger address space costs 0.7%. Random 512 B O_DIRECT on this drive
is dispatch-bound, not mapping-bound, so the synthetic figure was not
flattering itself.

A raw tight-loop O_DIRECT benchmark on this drive reaches 157,357 IOPS at QD32,
so Python dispatch -- not the NVMe -- was two thirds of the original cost. mmap
demand paging is worse again: 384 serialized faults at the measured QD1 latency
of 62 us is 24 ms, and the step is 20.

**Then the number that decides it.** The blocking figure above is not what a
step pays. The hash ids depend only on token ids, so the reads are issued at
layer 0 and collected at the engram layer; what the step pays is the residual.

Measured on the real 22.89 GiB shard:

| | median | p95 | verdict |
|---|---|---|---|
| layer 14, 7.0 ms cover | 0.07 ms | 0.08 ms | fits |
| layer 14, pipelined steady state | 0.05 ms | 0.06 ms | fits |
| layer 1, 0.5 ms cover | 3.37 ms | 3.99 ms | **over by 3.0 ms** |
| **both, layer 1 drafted a step early** | **0.08 ms** | **0.09 ms** | **fits** |

Layer 14 sits 13 of 40 layers after the submit and is free. Layer 1 sits one
layer in, so nothing inside the step can cover it -- but DSpark drafts 5 tokens
(`dspark_block_size`) and a draft token's hash ids are computable from the draft
ids alone, so step N can issue step N+1's layer-1 rows. That closes it: both
tables together cost **0.09 ms of a 20 ms step, 0.5%**.

The price is waste, not latency: at 75% draft acceptance 24% of the layer-1
reads are discarded. The IOPS budget absorbs it -- the step uses 89K of the
drive's 157K.

The 4 GiB numbers were taken WHILE the 475 GiB checkpoint was downloading to
the same drive, so they were the contended case. The real-shard re-run is on an
idle drive, and lands within 1% of them -- so neither the contention nor the
shard size moves this verdict.

**The drafted layer-1 row assumes a loop, not a model.** The DRAFTER is not in
question: V4.1 ships DSpark, and the `mtp.*` namespace is where it is stored.
The checkpoint carries `mtp.N.markov_head.{embed,head}.weight`,
`mtp.N.confidence_head.proj.weight` and `mtp.N.main_proj.weight` -- tensors
that exist for no other reason -- across 3 stages, and `config.json` sets
`dspark_block_size 5`, `dspark_target_layer_ids [37, 38, 39]`,
`dspark_n_routed_experts 128`. So a draft token's ids ARE available a step
early in principle.

What is unwritten is the plumbing. The reference implements the block's
forward and says so in its own comment -- "Only the forward pass is implemented
here, nothing calls forward_spec" -- so even the reference does not drive a
speculative loop, and vLLM has no V4.1 path at all. Until something hands step
N the draft ids for step N+1, the layer-1 line is the 3.99 ms p95 above and
does not fit; layer 14 fits either way. What closes layer 1 is a step of cover
from any source -- draft ids, chunked prefill lookahead, or moving the table --
and the first of those now has a drafter behind it.

## What is not established

Everything involving the model. Nothing here has run against a checkpoint, a
rank, or a step -- there is no `deepseek_v41` model file yet. That is writable
without an image (`ModelRegistry.register_model` plus a .pth, the pattern
`boot_stamps` already uses); it is just unwritten.
The drafted layer-1 path is measured in the probe but not implemented in the
layer: nothing here computes hash ids, maps a global id to a local row, or
hands rows back to a forward pass. What the measurements establish is that the
I/O budget is not the obstacle -- 0.09 ms p95 -- and that a blocking fetch or a
per-read dispatch would have made it one.
