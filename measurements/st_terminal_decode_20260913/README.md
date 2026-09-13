# Decode and prefill terminal feature writer

The target produces five drafter features and one final hidden mean from four MHC channels. Bind the existing FP32-mix/BF16-channel-exact terminal kernel into direct decode: each feature writes its final column, removing intermediate post tensors, casts, means and concat. The residual carry is retained for later layers; the final output still passes through the existing RMSNorm. The same terminal kernel is also bound into ordinary and layer-major prefill; ordinary prefill writes each local feature into its final column before the existing lossless SP gather.

`terminal_mhc` is an explicit experiment, default off, requiring direct MHC. Feature order, repeated feature layers, early-observation callbacks, C=1/C=4 rank agreement and cache state have a CPU oracle; missing layers are rejected before reading uninitialized columns. The CUDA kernel suite adds the C=4 28-row shape to the existing exact rounding, cancellation, strided-output and replay checks.

CPU on the same ST image: 35 tests, 30 passed and five GPU-only skips in 7.776 seconds. GPU kernel timing and candidate-only onepass remain required before promotion.

## First GPU gate and repair under test

`st-terminal-write0913v1` ran on the same ST image at 18:15 KST. Four test methods passed; exact comparisons failed at rows 28, 65, 1728 and 6912 within the remaining method. No timing ran, and this result does not qualify the terminal path. `gpu-v1.log` retains it.

CPU inspection of the compiled Triton PTX found unqualified `mul.f32` instructions at the post-product boundary. The repair disables compiler FP fusion while retaining explicit ordered `tl.fma` operations, making separate multiplication rounding explicit (`mul.rn.f32`). Both actual SM121 stride variants compile without a CUDA context, recorded in `compile-no-fusion.json`. This is a repair hypothesis until the exact GPU gate passes. Failed synthetic cases now retain the served post channels and an independent host `fmaf` oracle so a remaining discrepancy can be located without changing the tolerance.

Before the second GPU admission, CPU compilation of the actual served TileLang kernel located a more specific difference: NVCC starts each channel with `comb[0] * residual[0]`, then fuses `post * x` into that rounded product. The candidate had those products reversed. Its new implementation follows the served PTX order and keeps explicit product rounding. A deterministic cancellation canary produces BF16 `8.630752563476562e-5` in the served order versus `8.678436279296875e-5` in the old order. It now exercises every output cell, so the regression no longer depends on a large random sample. The second queued ticket was paused before admission to include this correction without wasting another GPU run.


## Corrected GPU gate and prefill integration

`st-terminal-write0913v2` retained ticket `17892916293354350` and admitted revision 4, source `93197d54`, at 18:46:41 KST. All five exact CUDA tests passed without skips in 4.318 seconds, including C=4 rows 28, larger rows through 6912, column guards, changed-input replay and the served-order canary. All eight paired timing cases completed with matching output hashes.

| Local rows / features | Warm baseline -> terminal | 64 MiB evicted baseline -> terminal |
| --- | --- | --- |
| 7 / 1 | 12.480 -> 7.152 us | 20.128 -> 8.192 us |
| 7 / 5 | 53.056 -> 12.064 us | 69.280 -> 26.272 us |
| 1728 / 1 | 2255.424 -> 393.168 us | 2265.248 -> 436.912 us |
| 1728 / 5 | 11582.144 -> 1916.448 us | 11883.248 -> 1985.840 us |

The payload subsequently failed while hashing a relative test path from the image's working directory. `gpu-v2.log` retains the failure and all completed metrics; `gpu-v2-recovered.json` recovers the printed medians and immutable checkout hashes. The 20 individual paired samples per case were not written before this reporting failure and cannot be recovered. The probe now resolves source files relative to its own file and compiles with the actual serving flags; running its compile-only mode from `/tmp` succeeds without a CUDA context (`compile-rooted-report.json`). Reporting repair does not require another GPU run.

Prefill now uses the same qualified kernel, including ragged SP rows and the final token's local-row slice. CPU integration passes 38 tests with five CUDA-only skips (`cpu-prefill.log`), covering repeated feature layers, same-SP hidden/features, KDA/KV state, prefix snapshots and the following decode. The existing unsharded-versus-SP CPU oracle fails at 130/131 rows in this ST image on both unmodified source `93197d54` and the candidate; both logs are retained, and that preexisting test is unchanged. The new terminal comparisons keep SP execution identical on both sides rather than treating an unrelated SP change as part of this optimization.

The kernel binary and source are unchanged from the completed CUDA gate. Full-model decode, acceptance, prefill and resident workspace still require the candidate-only onepass. `make_arm.py` creates exactly one measurement arm with both facts enabled; it creates no baseline or GPU reservation and must not be used to promote its measurement commit into main.
