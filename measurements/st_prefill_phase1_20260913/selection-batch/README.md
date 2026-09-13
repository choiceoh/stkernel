# Post-J4 prefill batch: GPU proof pending

Engine `2c2ee77f`, remote candidate-v25; controller `de67dc0e`.
Target: approximately 15% higher prefill throughput, with the absolute goal
still 3,300 input tok/s at 32K and 128K. This is an optimization estimate,
not a measured speedup. No baseline engine is booted.

The batch replaces the prefill sparse indexer's dense mask/Torch top-k/
int64 padding chain with valid-prefix radix selection. The CUDA core is
adapted from the repository's former Apache-2.0 SGLang-derived kernel;
its previous unmeasured status is not treated as evidence. The new ST
wrapper returns 512 pool IDs, validates extents/devices, uses the current
CUDA stream, owns its build hash, and leaves decode and graph selection
on the original path. The existing pool_slots lane restores position order.
Equal scores choose lower pool IDs; near-boundary quality requires live proof.

Two accompanying changes use 64x64 instead of 32x64 router tiles with adjacent
expert tiles sharing the activation working set, and 64x64 instead of 32x32
calibration Gram output tiles. Router inputs remain BF16 and accumulation /
output FP32. Gram sums retain the original 256-row partial boundaries.

J4's 128K diagnostic has 3.131 s in top-k, 1.567 s in dense masking,
1.938 s in indexer scoring and 1.209 s in router GEMM. The new selector
principally targets the first two categories and their temporary copies;
it does not remove all 6.636 s of the indexer path. The wider Gram targets
first-request work and is not assumed to accelerate already-complete sums.
The component GPU timing and subsequent consumer requests decide the benefit.

CPU proof: 20 tests pass in 4.590 s; the native selection extension and
SM121 router/Gram kernels compile with CUDA uninitialized. `compile-pass.log`
retains this evidence. `compile-missing-header.log` retains the initial
build-only failure: a broad ATen CUDA header required absent cusparse headers.
The fixed source uses only c10 CUDA stream/guard headers. No GPU was used
for either compilation attempt.

GPU workflow: ordinary FIFO ticket `prefill3300-selectionk`, all four nodes.
First validate selected score multisets, bounds, uniqueness, deterministic
ties, unchanged input, changed pointers/values, NaNs/infinities, non-default
stream and capture fallback. Measure original Torch selection and original
Gram tile as kernel controls. Recheck calibration/KDA and actual router
weights, then boot the candidate with KV 2 GiB/rank and FP32 KDA state.

Consumer: frozen harness 40, profiler off, cache reuse zero, 2K/32K/128K,
three fixed 1,024-token decode requests, quality and raw Korean evidence.
Also run fresh 32K/128K requests later on the same build with the profiler
off, to resolve J4's first-versus-later discrepancy. Any later CUPTI capture
remains separately labeled. Save evidence and stop only owned containers.

## GPU gate checkpoint

Ticket acquired the fleet at 10:16:19 KST. Native selection passed all 28
shape/distribution cases, including the 262,144-column bound and special
values. Selected score multisets are exact and no selected ID-set differences
were observed in this fixture. Input immutability, changed values/lengths,
non-default stream and capture fallback passed. This does not establish that
all possible tied inputs have the same ID ordering as Torch.

For 1,024 rows, selection timing changed 1.290144 to 0.318752 ms at 8,064
columns and 5.230464 to 0.955776 ms at 32,256 columns. The 32,256x4,096
Gram changed 39.851841 to 16.525600 ms with exact original outputs in that
fixture. These are component controls, not full-engine speedups.

The broader KDA/calibration gate then passed (36.853 s), preserving peak
values and row counts exactly, Gram error under 3e-6 against the mathematical
reference, and the original KDA snapshots. Actual-weight router gates also
passed: all changed selections remain within measured FP32 rounding bounds.
The candidate now proceeds to four-node boot and consumer scoring.
