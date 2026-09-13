# Direct decode terminal feature writer

The target produces five drafter features and one final hidden mean from four MHC channels. Bind the existing FP32-mix/BF16-channel-exact terminal kernel into direct decode: each feature writes its final column, removing intermediate post tensors, casts, means and concat. The residual carry is retained for later layers; the final output still passes through the existing RMSNorm. Prefill is unchanged.

`terminal_mhc` is an explicit experiment, default off, requiring direct MHC. Feature order, repeated feature layers, early-observation callbacks, C=1/C=4 rank agreement and cache state have a CPU oracle; missing layers are rejected before reading uninitialized columns. The CUDA kernel suite adds the C=4 28-row shape to the existing exact rounding, cancellation, strided-output and replay checks.

CPU on the same ST image: 35 tests, 30 passed and five GPU-only skips in 7.776 seconds. GPU kernel timing and candidate-only onepass remain required before promotion.
