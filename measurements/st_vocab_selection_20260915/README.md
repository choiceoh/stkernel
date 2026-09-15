# Drafter candidate selection in one warp — 2026-09-15

`select_logits` and the subsequent packed-key reduction now each use one warp
instead of Triton's four-warp default. The same segment geometry, 64-bit keys,
selection loop, sentinel padding and dense CUDA top-k tie ordering remain.
The drafter's greedy, sampled and candidate-tree calls all use this path.

## Why this change

Each segment picks 16 winners through dependent integer maxima. Four warps
exchange partial maxima through shared memory and synchronize for every winner.
One warp finishes those reductions with registers and shuffle instructions.
On the RTX build, each of the two kernels has 47 `bar.sync` occurrences in PTX
with four warps and zero with one. Shared storage is 32 → 0 bytes per CTA;
both versions have zero register spills. The first stage uses 80 → 168 registers
per thread, with 128 → 32 threads per CTA.

Parallel bitonic top-k and separate score/ID reductions were also tried locally;
the original integer selection loop with one warp was faster. They are not
included in the serving code.

## Local component evidence

Base: `9fe30cc9` (includes PR #1001 and #1002). Exact source hashes, runtime
versions and all samples are in `rtx5050.json`. NVIDIA RTX 5050, PyTorch
2.13.0+cu132, Triton 3.7.1; shared development GPU, 512 MiB allocator cap.

Each input has 38,720 BF16 logits per rank and selects 16 candidates. Production
C=1/C=2 at draft K=7 gives 7/14 selection rows. Timing is B/A/A/B ×4, eight
samples per arm; each sample replays 32 CUDA graphs containing 16 calls each.
Both arms use the same kernel source with only the warp count changed.

| Component | Shape | Four warps | One warp | Reduction |
|---|---|---:|---:|---:|
| Local candidate packet | C=1, 7 rows | 12.624 µs | 8.053 µs | 36.2% |
| Local candidate packet | C=2, 14 rows | 18.822 µs | 12.720 µs | 32.4% |
| Simulated peer merge + dense top-k | C=1, 7 rows | 95.947 µs | 91.669 µs | 4.5% |
| Simulated peer merge + dense top-k | C=2, 14 rows | 197.402 µs | 189.367 µs | 4.1% |

The merged measurement uses three precomputed peer packets and a local concat;
it includes the current dense CUDA top-k over 154,880 columns. It includes
neither real communication nor a model forward. These are component timings,
not engine tok/s or step/s. Rows 1 and 28 are additional non-production checks.

## Correctness

- Relevant CPU suite: 47 tests, 30 passed, 17 CUDA skips (`cpu-tests.log`).
- Actual CUDA packet/top-k suite: all 11 tests passed (`rtx5050.log`).
- Exact int64 candidate packets and final token IDs, including tied scores,
  NaNs, infinities, signed zeros, ragged vocabulary ends and strided logits.
- Four changing-input graph cases for each of rows 1/7/14/28: maxima move
  between segments and simulated ranks, cutoff ties and invalid vocabulary
  tails. Both arms match the independent dense CUDA top-k, and source bytes
  are unchanged. Reused merge storage clears previous candidates correctly.
- No floating-point arithmetic, communication payload or sampler policy change.

## GB10 and engine validation

GB10 component timing and the four-node full onepass have not completed.
The optimization is enabled by default under D11; these local results do not
establish a production engine throughput improvement. The bounded component
probe is available through the canonical queue entry below.

```sh
OMP_NUM_THREADS=2 python probes/engine_kernel_check.py \
  --lanes vocab_selection --output /tmp/vocab-selection.json

# From a frozen checkout on the srv2 controller:
ST_PROBE_GIB=1 bash bench/fleet.sh run --gpu --detach vocab-warp-950f 5 \
  'Candidate packet keys and graph replay; four vs one warp, C1/C2; no model boot' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
    --lanes vocab_selection --output /cache/vocab-warp-950f.json
```
