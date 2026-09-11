# Incremental GLM block-table publication — 2026-09-11

Baseline: PR #539 merge `0af79ecc2996304933b726310ffbe1fe7fc8add9`.
Measured implementation: `c28b047672be0477d9bdedf39a8a5184dc91d172`.
This second optimization removes repeated full-capacity host block-table
construction and CUDA copies from `Glm53Caches.prepare`.

## Implementation and invariants

`BlockPool` owns the mapping. A row can only append blocks until `release`
increments its epoch. The exposed table, row and epoch views are read-only;
callers must change mappings through `reserve*` and `release`. This closes the
untracked-write path that would make incremental publication unsound.

The profile remembers the published epoch and block count for each row:

- The same epoch and block count require no tensor construction or CUDA work.
- Growth within the same epoch uploads only newly allocated block IDs.
- A new epoch republishes the active prefix, even if the block count matches
  the old owner's count. This covers row reuse and NVMe park/resume.
- A shorter replacement includes the host row's `-1` suffix up to the old
  published length in the same copy. No extra fill kernel is needed.
- `reset()` invalidates publication bookkeeping when it clears the arena.
- Publication metadata commits after the copy succeeds, so a failed or partial
  copy can retry the entire affected range.

Every call still validates row bounds, state-slot ownership and the reserved
context before considering whether an upload can be skipped. Reservation
failure changes no mapping or epoch. A successful reservation followed by
cancellation advances the epoch when releasing its blocks, even if the row
ends empty again.

The GPU table and KV/state arena sizes and addresses are unchanged. Additional
host bookkeeping uses 20 bytes of flat array payload per sequence: an allocator
epoch, the published epoch and the published block count, plus Python headers
and read-only views. Model kernels, collective calls and sampling are unchanged.

## Measurements

srv1 NVIDIA GB10; torch `2.13.0+cu130`; five alternating rounds of 80 timed
calls per implementation, after 20 warmup calls per implementation. These are
**block-table publication component measurements**, not model ITL or throughput.

| Table capacity per row | Requests | Transition | Baseline µs | Optimized µs | Reduction |
|---|---:|---|---:|---:|---:|
| 512 blocks | 1 | Mapping unchanged | 40.35 | 8.35 | 79.3% |
| 512 blocks | 4 | Mapping unchanged | 137.81 | 9.39 | 93.2% |
| 4,096 blocks | 1 | Mapping unchanged | 210.33 | 8.45 | 96.0% |
| 4,096 blocks | 4 | Mapping unchanged | 794.53 | 9.31 | 98.8% |
| 4,096 blocks | 4 | Grow from 8 to 9 blocks each | 792.65 | 44.13 | 94.4% |
| 4,096 blocks | 4 | Reuse 32-block rows with 8 blocks each | 787.16 | 49.87 | 93.7% |

The table reports median wall microseconds including event submission and
waiting for the end event. The unchanged four-row path itself returns in a
median **1.49 µs** (`host_call`), with the remaining time belonging to measurement
instrumentation. CUDA stream timings include host launch gaps, not just kernels.
Raw medians, p95 and per-round wall summaries for all twelve cases are retained
in `overhead.json`.

Operator traces confirm the work reduction. With 4,096 entries per row and
four requests:

- Unchanged mapping: four 4,096-element copies become zero copies.
- One-block growth: four 4,096-element copies become four one-element copies,
  reducing int32 payload from **65,536 bytes to 16 bytes** per prepare call.
- Shorter row reuse: four 32-element copies replace four 4,096-element copies;
  the 128-byte prefix per row includes the new IDs and stale-entry clearing.

The probe uses a real `Glm53Caches` arena and allocator with a tiny one-layer
DSA layout. It calls the exported baseline prepare method and the new prepare
method against the current allocator, isolating publication costs. Allocator
reservation/reuse setup, arena reset/allocation and model execution are outside
timing. Both implementations receive equivalent mapping histories. Full GPU
rows, including unused suffixes, are compared with host rows after warmup,
every measured round, and operator collection.

The host's service configuration and clocks were not controlled. Results show
component behavior under this run's environment, not a fleet performance gate.
No full 45-layer quality, DFlash acceptance, or four-server ITL claim is made.

## Validation

`engine-tests.log`: **94 tests passed, zero skips**, including eight new tests:

- Allocator epochs stay stable during append/no-op/failed reservations and
  invalidate reuse; exposed mapping and epoch views reject direct writes.
- Growth, same-sized reuse into different physical blocks, shorter reuse and
  cache reset publish exactly the current host mapping.
- An unchanged row still rejects incorrect ownership or unreserved positions.
- Failed growth/replacement uploads retry; a partially copied replacement
  still clears its stale suffix on retry.
- Parking, a partially failed promotion, allocation by another sequence and
  successful resume restore both the GPU mapping and original KV bytes.
- A seeded 200-transition trace checks all live GPU rows against full host rows
  through interleaved growth, release, exhaustion, reuse and resets.

The existing CUDA indexer rollback, noncontiguous paging, sampling, request
lifecycle, LocalTP and HTTP regressions also pass. The LocalTP test here is the
existing four-logical-rank serving test, not a new four-server model run.

`nvme-io.log`: the real O_DIRECT probe passes with 16 MiB KV and 2 MiB staging,
including producer-stream ordering, concurrent Futures, replacement failure
recovery and restart cleanup. It reports 26,214,400 bytes in completed demotions
and 45,875,200 bytes in completed promotions. This probe validates the modified
allocator with actual I/O; these byte counters are not throughput measurements.
Its final sources were unchanged by the last cache-publication-only refinement.
`kv-selfcheck.log` also passes.

`source-sha256.json` records 84 Python files, checked against the submitted local
source and baseline commit. `environment.log` captures host/image metadata after
the run. Log cleanup removes trailing whitespace without changing measurements.

## Reproduction

```bash
git show 0af79ecc:engine/profiles/glm53/caches.py > baseline_caches.py
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
python3 -m engine.base.kv
python3 probes/engine_cuda_io_check.py
python3 probes/engine_cache_overhead.py \
  --baseline baseline_caches.py --output overhead.json
```

The isolated srv1 checkout was `/home/choiceoh/st-engine-f4d7-cache`, in image
`glm53:v13-b12x-it`, ID
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
Disposable containers used `--gpus all --memory=4g --memory-swap=4g --cpus=2
-e OMP_NUM_THREADS=2`, with the checkout mounted at `/repo`. The I/O probe also
mounted the checkout's isolated `scratch` directory at `/root`, placing temporary
files on the host filesystem that supports O_DIRECT.
