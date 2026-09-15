# KDA prefix-boundary staging — 2026-09-15

KDA prefix staging now uses one kernel and at most eight CTAs per layer and
request. Each CTA checks whether the committed tokens crossed a 768-token
prefix boundary. A non-crossing request returns immediately; a crossing
request's CTAs copy the recurrent cell and convolution history in disjoint
stripes. The drafter ring staging remains a separate launch.

The old path launched one CTA per 1024 recurrent elements and one per 256 conv
channels on every decode step, including steps that did not cross a boundary.
In GLM's production shape, the KDA staging grid changes from 9,520 → 272 CTAs
for C=1 and 19,040 → 544 for C=2, with two launches → one. Copy addresses,
source precision, stage layout and the boundary condition are unchanged.
The source rings are read-only; distinct CTAs own distinct destination cells.

## Local component comparison

Base: `da1c326d`. The two control kernel definitions in
`probes/engine_boundary_stage.py` have the same AST as the original functions,
apart from their names. Both arms execute in the same runtime on the same
source tensors. Hashes and all samples are in `rtx5050.json`.

- NVIDIA RTX 5050; torch 2.13.0+cu132, Triton 3.7.1, CUDA 13.2.
- Production geometry: 34 KDA layers, 16 local heads, 128×128 FP32 state,
  K=7, 6,144 BF16 convolution channels, 10 ring cells and 3 history taps.
- The model-free fixture has two slots and a 768 MiB allocator cap; peak
  reserved memory was 722 MiB. GPU development workloads may share the card.
- B/A/A/B ×4, eight samples per arm. Each sample replays 16 CUDA graphs,
  each containing eight staging calls. Device event timings exclude setup.
- The new kernel uses 34 registers per thread, no spills and no shared memory.

| Shape / boundary event | Original | Fused bounded grid | Change |
|---|---:|---:|---:|
| C=1, no crossing | 33.734 µs | 1.561 µs | −95.4% |
| C=2, no crossing | 66.548 µs | 1.650 µs | −97.5% |
| C=1, one crossing | 274.890 µs | 279.358 µs | +1.6% |
| C=2, one crossing | 277.834 µs | 281.903 µs | +1.5% |
| C=2, both crossing | 552.420 µs | 562.525 µs | +1.8% |

This trades roughly 4–10 µs on a crossing event for 32–65 µs on each
non-crossing step in this component comparison. It does not make the required
state copy smaller. A decode commits at most eight tokens against a 768-token
prefix block, so ordinary continuous generation has many non-crossing steps
between crossings. The measured percentages apply only to KDA boundary staging,
not the whole decode step or engine tok/s.

## Correctness

- Relevant CPU suite: 99 tests, 94 passed, five CUDA skips (`cpu-tests.log`).
- Boundary/restore suite with CUDA: all four tests passed. The CUDA test covers
  FP32 and FP16 recurrent storage and both small and multi-iteration odd
  dimensions, including a masked final tile and multiple conv iterations.
- Actual arena bytes contain arbitrary floating bit patterns, including NaN
  payloads and padding. Staging, checkpoint and restore match the CPU reference.
- Twelve production-shape graph cases cover C=1/C=2, crossing/no crossing,
  zero committed tokens, reordered slots, wrapped ring positions and contexts
  above 2^40. Every staged word matches both the original kernels and independent
  tensor indexing. Non-crossing slots retain their sentinel bytes.
- SHA-256 of every source-ring byte is unchanged after the probe.
- `git diff --check` and Python compilation passed.

The first probe attempt reached its own allocator cap during a byte-wise
comparison, before timing. Comparing integer words retains the same bitwise
gate while reducing the temporary allocation; the recorded run completed within
the unchanged 768 MiB cap.

## Adoption and limits

The change is on by default with no tuning flag. Per the operator's instruction,
no fleet reservation was registered; GB10 component and TP4 full onepass were
not run. This record establishes local component results and exact copy/restore
behavior, not a measured production engine throughput improvement.

## Reproduce

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -m unittest \
  tests.test_engine_boundary_stage tests.test_engine_state tests.test_engine_kda_storage \
  tests.test_engine_prefix tests.test_engine_transient_prefix tests.test_engine_pipeline \
  tests.test_engine_deferred_state_contracts tests.test_engine_charter -q

OMP_NUM_THREADS=2 python probes/engine_kernel_check.py \
  --lanes boundary_stage --output /tmp/boundary-stage.json
```
