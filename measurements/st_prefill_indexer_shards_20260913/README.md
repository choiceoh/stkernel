# Replicated indexer query work — after PR #876

PR #876 merged as `504681a4ce60aa9bdb14c3e2d45886031e725640`; final-head engine CI passed.
This follow-up is a new, default-off CPU-qualified candidate. No GPU queue or baseline engine boot occurred.

## Implemented

The indexer's weights are replicated (`specs.py` uses `_whole`), but every rank previously projected, quantized,
scored and selected every query. Each rank now owns one contiguous query range. Cache-writing key/gate rows,
pool completion and tail-ring updates remain full-length on each rank. Only selected pool IDs are gathered;
each rank resolves those IDs against its own page map.

If `floor((context + row + 1)/pool) <= topk/pool`, all complete pools must be selected. Those queries skip
projection, head-gate, quantization, scoring and top-k. When the whole step is covered, no new collective runs.
In native GLM geometry this includes the first 2,051 positions. Small scored tails are padded to 64 rows for
query-only projection so they cannot accidentally enter the <=32-row W4 decode path. Padding never enters a cache.

Pool IDs use lossless 16-bit transport when the candidate count is at most 65,535. Two IDs travel in one NCCL
int32 lane; 65,535 denotes -1. Larger contexts or odd selection widths retain int32. Tests cover 32,767/32,768,
65,534, the invalid marker, and the wide fallback. Decode/capture and multi-segment steps retain their prior path.

Opt-in: `STK_prefill_indexer_shards=1` in an experimental boot. Production declares 0 and rejects overrides.
`ExecutionPlan.prefill_indexer_shards` and the runtime's per-DSA-layer execution markers expose the actual choice.
The planned shared-KV dense-prefix MLA kernel is not part of this implementation.

## CPU and upgraded Oracle evidence

Source `227003ef`: **42 tests passed, none skipped**, 14.70 seconds in the existing ST image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`.
Docker used runc, CPU=2, memory/swap=4 GiB, network=none, no NVIDIA devices, and a read-only source mount.
The suite was:

```text
tests.test_prefill_indexer_shards
tests.test_engine_knobs
tests.test_engine_execution_plans
tests.test_engine_prefill_tiles
tests.test_engine_prefill_outputs
tests.test_engine_native_execution
```

It compared the actual indexer slots, valid counts, pool bytes and tail state; full TP4 CPU model hidden/aux,
KDA/KV state and next decode also matched exactly in ordinary and layer-major execution. It includes 128–132
rows, partial pools, nonzero context, no-query ranks and the unchanged small-step path. The earlier uncompressed
candidate passed 40 tests (`cpu-r1.log`); `cpu-r2.log` includes the two transport checks.

The upgraded Oracle from **PR #875, `e2bfbb9a`** ran against this candidate with its explicit execution setting:

```sh
python3 /path/to/pr875/bench/storacle.py compare --tree /path/to/this/tree \
  --base 504681a4 --candidate 227003ef --ctx 2000,32000,128000 --width 1 \
  --set prefill_indexer_shards=1 --json
```

`oracle-pr875.json` confirms the new setting is active, chunk=32,256, and resident cache/state bytes unchanged.
**Total timing delta is null**: the changed indexer/communication has no matching GPU profile. The identical
historical base/candidate modeled times are not a 0% measured result. `paired-profile-template.json` freezes
both source/settings fingerprints; its empty durations and zero samples deliberately cannot be consumed.
`oracle-pr876-before-followup.json` retains the first requested #875 comparison, of PR #876 against its main base.

## Exact work counts, not speed estimates

`work_budget.py` uses the actual `QueryShard` ownership rules and the Oracle's chunk size.
The counts below sum across all four ranks **per DSA layer**; they do not represent end-to-end time saved.

| Input tokens | Query rows scored before | After | Reduction | Added remote ID bytes/rank/layer |
|---:|---:|---:|---:|---:|
| 2,000 | 8,000 | 0 | 100% | 0 |
| 2,672 (historical 2K prompt) | 10,688 | 621 | 94.2% | 2,052,096 |
| 32,000 | 128,000 | 29,949 | 76.6% | 24,576,000 |
| 128,000 | 512,000 | 125,949 | 75.4% | 98,304,000 |

The old replicated algorithm needed no query-ID exchange; this candidate adds one per active, non-covered
step/layer. The 16-bit form halves that new traffic relative to the first candidate. At 32,256 rows the gather
buffer is 31.5 MiB per rank/layer and the owned packet is 7.875 MiB. These are transient buffers, outside the
Oracle's resident layout accounting. Extra communication, smaller GEMMs and allocation cost can offset saved
query work. Runtime pack equality across ranks, GPU numerical/quality/acceptance, fixed decode, and profiler-off
no-cache C=1 2K/128K TTFT still require an authorized GPU run. No tok/s or target attainment is claimed.
