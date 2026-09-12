# One-token packed mHC probe — not enabled

The frozen four-rank consumer remains `8c8b031b94bb80175cfccd49cfb6329d18cdecae`.
This separate source only adds an explicit `single_token_grid=True` probe entry.
Serving uses the existing false default. No running source, cache or boot was changed.

For seven verifier tokens, 16 hidden chunks per token give 112 CTAs instead of
48 CTAs looping over three token groups. The probe streams the mixing coefficients,
removes next-token state, and reads tail residuals after Sinkhorn to reduce register
pressure. The host queries this exact kernel's residency on the current device and
refuses a grid that cannot fit. No persistent-grid deadlock is accepted as a timing result.

The previous isolated profile priced 89 packed mHC calls at 2.779 ms. That is a
component budget, not current engine profiling or a predicted speedup. More concurrent
weight reads and reduced all-reduce overlap may lose; the comparison decides.

Validation prepared:

- Full native CUDA/Torch extension compile and load without CUDA initialization:
  `CUDA_VISIBLE_DEVICES= python3 -m probes.engine_mhc_single_compile --build-root /out/build --output /out/compile.json`.
- The admitted `probes/engine_kernel_check.py --lanes mhc_single` checks 1/6/7/8 rows,
  changed inputs, alternating graphs and rearmed native counters against the current
  packed consumer. Every output must match exactly; skipped tests fail the GPU gate.
- Only after those checks, 89 distinct coefficient packs are compared in B/A/A/B
  graph replay order. This excludes model traffic and RDMA overlap and is not onepass.

The full native CUDA/Torch compile and load passed on `a508f86b` without opening
a GPU, during the second consumer's preparation. `compile-a508f86b.json` records
80 registers and 28,736 shared bytes for the candidate, versus 128 registers for
the existing packed consumer. The runtime occupancy guard still has to qualify
the actual grid on GB10; compiler resources alone do not prove a safe launch or gain.

The first attempt compiled successfully but its resource reader expected an older
cuobjdump header and falsely reported a missing symbol. The corrected reader has
two CPU tests and the complete compile gate was rerun successfully; the CUDA source
is unchanged. `cpu-controller-a508f86b.json` records the image, CPU limits and phase.

Local Python syntax and whitespace checks pass. The two GPU tests skip on macOS.
GPU numerics and timing remain pending. Neither probe is in the live `8c8b031b`
consumer; 22 step/s has not been reached.
