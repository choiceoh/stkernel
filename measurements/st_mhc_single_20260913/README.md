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

Local Python syntax and whitespace checks pass. The two GPU tests skip on macOS.
Full compilation, GPU numerics and timing are pending. Do not adopt this probe on
those local checks or imply that 22 step/s has been reached.
