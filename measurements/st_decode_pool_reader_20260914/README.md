# Decode reader follow-up to #914, #919 and #921

Base: `119f881d90554fbe16eca707e69ce9bc41ef03e4` (includes #920 and #921).

Review followed the shared query packs, smoothed readers, MLA weight views,
pool-cache completion and indexer finalization. Two remaining materializations
are removed:

| Change | Old captured path | New captured path | Source launch reduction |
|---|---|---|---|
| Pool input reader | Mapped window -> compression -> cache update | Direct compression -> cache update | 11 / target forward |
| Top-k ids | int64 winners -> int32 tensor -> finalization | Finalization reads int64 winners | 11 / target forward |

At K=7, the total is **22 fewer launches per target forward**, at C=1 and C=4.
These are source counts, not measured latency. Per DSA layer and active
sequence, the change removes two 8x128 BF16 windows (8 KiB) and an 8x512
int32 winner copy (16 KiB). Graph-pool allocation reuse is not measured.

Both are default paths. Direct compression extends the existing
`decode_dsa_inputs=1` experimental/production default; its experimental
rollback remains `STK_decode_dsa_inputs=0`. The int64 reader replaces the
redundant copy in captured indexer selection without another knob.

## Contracts preserved and tightened

- The pool reader reuses the same return-only Triton kernel: four-slot
  softmax order, one-warp Hadamard, both BF16 rounding points and FP8 scale
  calculation are unchanged. Only source addresses differ. Pool compression
  finishes before the separate cache-update kernel writes tail cells.
- Int64 winners are range-masked **before** narrowing. Invalid values such
  as `2**32 + 1` cannot wrap into valid ids. Sorting/scanning remains int32;
  duplicate multiplicity, descending token order and padding stay the same.
- Every bound layer/width must execute the complete pool path for boot proof.
- The pooling reference selector also replaces the captured compressor,
  preserving `reference_for=("kpool_compress",)` bisection semantics.
- Query weight ownership, smoothing groups, MLA contractions, head-gate
  precision, recurrent state, KV capacity and K=7 are unchanged by this PR.

## Completed CPU and compiler evidence

All containers used runc, no network, NVIDIA_VISIBLE_DEVICES=void and
CUDA_VISIBLE_DEVICES= on the pinned Linux/arm64 image
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`.
Torch 2.13.0+cu130 / Triton 3.7.1 / CUDA 13.0; no GPU was accessed.

| Artifact | Result |
|---|---|
| cpu-tests.log | 104 tests: 92 pass, 12 GPU/interpreter skips |
| cpu-ids-tests.log | 34 tests: 28 pass, 6 GPU skips |
| cpu-final-tests.log | 17 tests: 14 pass, 3 GPU/interpreter skips |
| interpreter-tests.log | 17 tests: 16 pass, 1 GPU-only skip |
| interpreter-ids-tests.log | 6 tests: 5 pass, 1 GPU-only skip |

The direct-reader interpreter test executes the actual Triton body with
FP32 diagnostic destinations: quantized values **before the FP8 store**
and scales match materialized-window execution exactly at C=1/2/3/4,
all leading-pool offsets, 32K/128K positions, rollback, changed slots,
strided tail/current/bias views and magnitudes 0 through 100.
This is not evidence for the real GPU FP8 cast or compiler scheduling.

Pool-id interpreter checks use an independent expanded-token Torch oracle,
including large invalid int64 values, duplicates, strided buffers and maps.

`compile.json` binds source hashes and records SM121 PTXAS success for:
the direct reader; the existing BF16/FP32 score paths; the masked legacy
cache writer; int32/int64 pool-id finalization; and the prior DSA bundle.
Both id widths keep the same 2,048-byte shared allocation. Direct pooling
uses zero shared memory. Native dense cache `527d3941349c914d1ea68fbb` is reused.

PR875 Oracle head `95c27e6340159413c44a38eef9e54a84b13f9fc5` audits actual
defaults at 32K/128K and C=1/C=4. All four decode deltas remain null:
changed costs are unpriced, and the paired profile contains no measurements.

## Queued GPU scope

The canonical `engine_kernel_check.py --lanes dsa_inputs` now compares
three pool-completion paths using all 11 real layer biases: the original
path, #921's mapped windows, and the direct reader. It checks FP8 bytes,
scales and complete cache/tail storage after poisoned, reordered replays
with changed ids/pages/contexts. Pool timings now include actual compression.
A separate captured comparison qualifies top-k int64 reads versus the
previous int32-copy path over all 11 layer offsets.

Warm/evicted B/A/A/B compares both the incremental reader and the combined
pool changes. This extends the existing short reservation with
`--replaces`; no full-model baseline boot is requested. GPU results,
consumer step/s, tok/s and acceptance remain pending. Final consumer coverage
is still 32K/128K, C=1 twice and C=4 once.
