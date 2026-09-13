# Direct decode terminal feature writer

The target produces five drafter features and one final hidden mean from four MHC channels. Bind the existing FP32-mix/BF16-channel-exact terminal kernel into direct decode: each feature writes its final column, removing intermediate post tensors, casts, means and concat. The residual carry is retained for later layers; the final output still passes through the existing RMSNorm. Prefill is unchanged.

`terminal_mhc` is an explicit experiment, default off, requiring direct MHC. Feature order, repeated feature layers, early-observation callbacks, C=1/C=4 rank agreement and cache state have a CPU oracle; missing layers are rejected before reading uninitialized columns. The CUDA kernel suite adds the C=4 28-row shape to the existing exact rounding, cancellation, strided-output and replay checks.

CPU on the same ST image: 35 tests, 30 passed and five GPU-only skips in 7.776 seconds. GPU kernel timing and candidate-only onepass remain required before promotion.

## First GPU gate and repair under test

`st-terminal-write0913v1` ran on the same ST image at 18:15 KST. Four test methods passed; exact comparisons failed at rows 28, 65, 1728 and 6912 within the remaining method. No timing ran, and this result does not qualify the terminal path. `gpu-v1.log` retains it.

CPU inspection of the compiled Triton PTX found unqualified `mul.f32` instructions at the post-product boundary. The repair disables compiler FP fusion while retaining explicit ordered `tl.fma` operations, making separate multiplication rounding explicit (`mul.rn.f32`). Both actual SM121 stride variants compile without a CUDA context, recorded in `compile-no-fusion.json`. This is a repair hypothesis until the exact GPU gate passes. Failed synthetic cases now retain the served post channels and an independent host `fmaf` oracle so a remaining discrepancy can be located without changing the tolerance.
