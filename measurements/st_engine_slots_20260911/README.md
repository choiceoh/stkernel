# GLM indexer slot finalization — 2026-09-11

Baseline: `c1d8357aac6f00b01f3b16ffddfdea1a16a18e17` (merged PR #540).
Measured implementation: `eb63e89e` (PR #543); source hashes are pinned to that revision.
The integrated comparison executes the exact baseline `net.py` alongside the
changed `_indexer` in the same process. Its source SHA256 is recorded in
`slots.json` and checked against `git show`. All other kernels, real weights,
inputs, runtime and incremental block tables are shared between the paths.

## Change and contract

`Lanes.indexer_slots` retains PyTorch's descending integer sort, then runs a
single native Triton kernel to count valid positions, map their block addresses,
mask padding and write the final output views. For six or more rows, the whole
finalizer goes from **17 kernels and five device copies to four kernels and
two copies**. Sorting accounts for the remaining kernels/copies. For a single
row it goes from 16 kernels/six copies to three kernels/three copies.

The lane writes every slot and count, so `_indexer` allocates outputs with
`empty` and avoids their separate fill/zero kernels. The slot buffer is reused
across the step's segments; each segment writes only its own slice. No sorted
indices, mask, clamped positions or mapped-slot tensor is materialized after
the sort. The lane follows the existing LocalTP main-thread dispatch and
propagates errors without a reference fallback.

Caches expose `token_map(layer, seq)`: the selected GPU block-table row, token
block size, block stride and layer offset, all measured in latent rows. The
chain-check cache returns a None table for identity mapping. Token writes and
pool addressing keep their existing APIs. Custom caches must implement this
new method, and custom lane tables must supply `indexer_slots`.

Valid positions must belong to the reserved row. Negative values are padding;
output padding is always -1. Duplicate positions and exact descending position
order are preserved even when physical block order differs. The kernel masks
padding before reading the block table. Input/output views may be strided and
must not overlap. Empty shapes are supported.

PyTorch's optimized integer sorting remains in place because an initial
single-program bitonic sort over the padded 4,096-column vector regressed
prefill. The final implementation fuses only the subsequent integer operations.

## Complete indexer measurements

srv1 GB10, standalone `st-engine:9391`, torch `2.13.0+cu130`. Five alternating
AB/BA rounds of 50 calls per path follow ten warmup calls. The probe loads seven
real L3 indexer tensors (15,207,424 bytes) with seeded synthetic activations.

| Complete L3 indexer call | Baseline | Optimized | Time reduction |
|---|---:|---:|---:|
| Decode/verify, 6 tokens at context 2,048 | 2.272 ms | 1.932 ms | 15.0% |
| Prefill, 256 tokens | 2.381 ms | 2.050 ms | 13.9% |

These times cover the indexer component, not a full attention block or model.
The host was not isolated: existing workloads were left running, and clocks
were not locked. Absolute latencies should not be compared with earlier runs;
only paired results within this run are used above. Full-model quality,
four-server ITL/throughput and DFlash acceptance were not requalified here.

The integrated call's additional peak PyTorch CUDA allocation decreased from
472,576 to 389,632 bytes for decode, and from 19,059,200 to 14,875,648 bytes for
256-token prefill. The allocation figures are component peaks, not total model
memory usage.

## Finalizer measurements

The helper comparison supplies the same preallocated output views to the
PyTorch reference and served lane, with 2,051 columns, a shuffled block map,
duplicated positions and trailing padding. Five alternating rounds of 100
calls follow 40 warmup calls. The integrated baseline above also includes the
original output initialization.

| Rows | Reference µs | Optimized µs | Time reduction |
|---:|---:|---:|---:|
| 1 | 433.29 | 138.07 | 68.1% |
| 6 | 437.34 | 140.75 | 67.8% |
| 24 | 437.14 | 141.23 | 67.7% |
| 256 | 455.98 | 191.88 | 57.9% |
| 512 | 686.62 | 332.11 | 51.6% |

At 512 rows, additional peak CUDA allocation fell from **25,202,688 bytes to
12,618,240 bytes (49.9%)**. Sort workspace remains; the final map/count kernel
uses the caller's output tensors and allocates no additional tensors.

All timings finish before CUDA profiling initializes. Wall samples include end
event submission and synchronization; stream spans include host launch gaps.
Setup, checkpoint loading and JIT warmup are excluded. Raw per-round medians,
p95, stream spans, peak allocations and operator traces are in `slots.json`.

## Correctness and evidence

- **110 engine tests passed on CUDA, zero skips** (`engine-tests.log`). This
  includes non-power-of-two widths through 4,099; duplicate and large integer
  positions; all-padding and empty inputs; strided input/table/output/count
  views with sentinel guards; paged and identity maps; and graph replay after
  changing both inputs and block mappings.
- Multiple request segments use their own block row and output slice. Lane
  failures propagate. Four LocalTP logical ranks produce exact slots/counts.
- The exact baseline and current indexer match selected slots, valid counts,
  pooled KV/scales and tail-ring bytes for prefill at contexts 63/256/2,048,
  six-position draft verification and rollback accepting two positions. The
  comparisons pass again after timing. No tolerance is used or relaxed.
- The probe asserts that `vllm` is neither installed nor loaded. It loads current
  source from `/repo`, with no overlay mounts, through the native ST runtime.
- `source-sha256.json` hashes **145 engine/test/probe Python files**, verified
  against the submitted source. The indexer-only weight file is the same seven
  tensors recorded in
  [the prior weight manifest](../st_engine_indexer_20260911/weights-sha256.json).
  The guarded profile loader remains active; no expert layout check is bypassed.
- `slots.json`, `slots.log`, `engine-tests.log`, `environment.log` retain the
  final run's evidence. Logs only have trailing whitespace removed.

The isolated directory is `/home/choiceoh/st-engine-f4d7-slots` on srv1.
Runtime image ID:
`sha256:0d781f0a8f77d4735d0d09d57b081c9489b3a46131dc72773fd40fcbf267b446`.
The suite used a 6 GiB container, then the probe used an 8 GiB container, both
with two CPUs, `OMP_NUM_THREADS=2` and a separate writable JIT cache. Existing
services, production checkpoints and previous test directories were unchanged.

## Reproduction

Prepare model config and a compatible aligned rank file, or the byte-identical
indexer-only subset documented above. Export the comparison source from git:

```bash
git show c1d8357a:engine/profiles/glm53/net.py > /absolute/path/to/inputs/baseline-net.py
```

Mount this checkout at `/repo` read-only, the config/rank/baseline directory at
`/inputs` read-only, an isolated writable JIT directory at `/cache`, and a writable
output directory at `/evidence`. Use the runtime/resource limits above and
`PYTHONPATH=/repo`, then run:

```bash
python3 /repo/probes/engine_indexer_slots.py \
  --baseline-net /inputs/baseline-net.py \
  --checkpoint /inputs --rank-file /inputs/indexer-only.safetensors \
  --output /evidence/slots.json
```

For the suite, set the working directory to `/repo` and run
`python3 -m unittest discover -s tests -p 'test_engine_*.py' -v`.
