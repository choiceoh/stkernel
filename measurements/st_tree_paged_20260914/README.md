# Tree verification: direct paged/private KV

The tree experiment no longer materializes an FP8 `[nodes, selected, latent]`
cache at each DSA layer. One integer kernel expands/orders selected pools and
addresses the canonical cache or the node's ancestors. The existing MLA kernel
reads the two banks with unchanged attention arithmetic. Small-row absorb
contractions write token-major output directly, eliminating both layout copies.
W4A8 weights, FP32 KDA, target routing and commit ownership remain unchanged.

This is the **tree experiment's default**, not production graph/scheduler
integration. No GPU, fleet queue, boot or deployment was used. There is **no
production tok/s or acceptance verdict** here.

## Storage and operations removed

FP8 latent width 512, selected width 2051, per DSA layer:

| Nodes | Final KV staging removed | Minimum staging read/write removed | New slot tensor | Layout copies removed |
|---:|---:|---:|---:|---:|
| 8 | 8,400,896 B | 16,801,792 B | 65,632 B | 196,608 B |
| 15 | 15,751,680 B | 31,503,360 B | 123,060 B | 368,640 B |
| 32 | 33,603,584 B | 67,207,168 B | 262,528 B | 786,432 B |

The old path additionally allocated/gathered/masked full-width tensors. The
table counts only one final staging copy, not their larger combined traffic.
These are exact shape/operation counts; memory bandwidth and latency are unmeasured.

## Verification

- Engine CPU gate: **1,883 tests, zero failures, 372 skipped**. Final focused
  checks: **31 tests, zero failures, 7 skipped**. The full gate predates the
  review correction; final focused checks cover the tree-specific cluster
  capacity and capture admission. CUDA tests remain skipped.
- Native Triton CPU interpreter: tree addresses across 66 pool/context/width
  combinations (1/4/8 pooling, 9/512 selected pools, contexts through 131069),
  including duplicates, future/overflow ids and paged boundaries. The unchanged
  ordinary pool-slot interpreter suite also passes: 10 tests total, one GPU
  graph test skipped.
- Extracted native async-copy CPU model: **180 slices × 256 threads** covering
  canonical-only, private-only and mixed banks, partial tiles and empty splits.
  The missing-wait negative control is rejected.
- CUDA **13.2.86** offline SM121 compilation: both tree MLA instantiations and
  nine address/absorb variants have **zero spills**. Ordinary and cluster MLA
  retain **identical encoded instruction words** relative to merged #952.
  Tree ordinary/cluster register counts are 100/64 versus 97/62. Additional
  int32/int64 ordinary slot specializations compile successfully.
- The actual complete CUDA translation unit, including Torch bindings, compiles
  to an object. This uses Torch 2.14 CPU headers, C++20 and c10's CUDA generated
  export-header opt-out; it is not linked against CUDA or qualified as a serving
  runtime. Device bodies are also compiled separately with C++17. No CUDA
  context is initialized.

## Matched CPU comparison

Baseline: `c58a8eb534ee5bf712f09f1ab68891ce02edcb85` (merged #951 plus #952).
The same tiny three-layer target and inputs run full **verify + greedy commit**;
proposal is excluded from both. Cache restoration is outside timing. B/A/A/B is
repeated twice, with 3 warmups and 80 samples per arm. No concurrent local
compile/interpreter job ran during the final timing.

| Nodes | Context | Before ms | After ms |
|---:|---:|---:|---:|
| 8 | 0 | 1.860313 | 1.879521 |
| 8 | 129 | 1.851520 | 1.855520 |
| 8 | 1024 | 1.848291 | 1.875541 |
| 15 | 0 | 2.364167 | 2.382042 |
| 15 | 129 | 2.404354 | 2.415229 |
| 15 | 1024 | 2.425584 | 2.447563 |

Output, auxiliary features, canonical state, paged-cache bytes, emitted tokens
and committed paths match exactly in all six cases. CPU time is approximately
unchanged (-0.22% faster to 1.47% slower); this does **not** demonstrate a speedup.
The CPU oracle still gathers rows for its tensor attention and does not execute
the GPU copy-free reader or native absorb kernels.

Reproduction commands and boundaries are in `engine/EXPERIMENT_TREE_DATAFLOW.md`.
`identity.json` and probe JSON files retain exact source hashes. Native numerical
equivalence/replay, real reasoning quality/acceptance and a same-build full
serving tok/s comparison remain unmeasured.
