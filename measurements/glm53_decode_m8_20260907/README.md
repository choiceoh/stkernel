# C=1 MK-GEMM decode experiments, 2026-09-07

**Serving follow-up:** no reproducible C=1 speedup. The clean 2K-context
comparison was 75.074 → 73.153 tok/s (-2.56%), with all onepass quality gates
passing. Both options remain off. See [serving results and raw evidence](serving/README.md).

Base checkout: `de2a0b3`. All runtime arms in each round are built from the
same composed CUDA source in an isolated checkout, in `glm53:v13-b12x-it`,
on srv2 / NVIDIA GB10 / SM121a. Torch 2.13.0+cu130, CUDA 13.0, PDL enabled.
The fleet queue grants the GPU window; these probes do not restart serving.
Both profile options remain off; the serving follow-up above did not show a win.

## Round 1: paired conversion and transposed MMA

Source SHA256: `b69850f9f77268d2a602b292df3e6bb5a6d700fa90334ad2e5be89badecc37ea`.
Fleet session `glm53pack2`, started at 00:18:49 KST.

- FP8 conversion: 1,114,880 inputs, zero byte differences. Includes all BF16
  bit patterns, rounding boundaries, random FP32 bits, infinities and NaNs.
- Both candidates pass the independent GEMM oracle and SMLP2 gate.
- All 12 geometry cases, warm/cold regimes and graph replays are bit-equal.
- Six execution-order permutations, 12 repetitions; warm graphs contain
  50 GEMMs, cold graphs contain one GEMM after the existing L2 flush/drain.
  Captured external CUDA events exclude the flush and host submission gap.
- Raw results: [round1.log](round1.log), [round1.json](round1.json).
  Resource dumps confirm 112 total registers for both baseline and mode 1;
  the smaller accumulator alone did not reduce total register allocation.

C=1 M=6, baseline → pack2 + transpose mode 1, microseconds:

| N × K | Warm | Change | Cold weights | Change |
|---|---:|---:|---:|---:|
| 6416 × 4096 | 41.361 → 39.905 | −3.52% | 74.432 → 73.088 | −1.81% |
| 4096 × 2048 | 14.023 → 13.544 | −3.41% | 31.392 → 31.504 | +0.36% |
| 1024 × 4096 | 10.978 → 10.670 | −2.80% | 21.504 → 21.456 | −0.22% |
| 4096 × 512 | 6.822 → 6.638 | −2.70% | 14.560 → 14.368 | −1.32% |
| 6144 × 4096 | 32.531 → 31.664 | −2.66% | 76.816 → 75.696 | −1.46% |

Cold results include outliers and changes near the noise floor. These are
kernel microbenchmarks, not a C=1 tokens/s or end-to-end latency verdict.

## Round 2: compact M<=8 allocation

Mode 2 uses eight-row activation stages, a three-block launch bound and
an occupancy-aware split planner. It preserves the original allocation
and planner for low-rank correction and larger M. The independent device
compile reports 78 registers and zero spill loads/stores; this is a compile
result, not runtime occupancy or latency proof. See
[compact_device_compile.log](compact_device_compile.log).

Fleet `glm53m8` started at 00:48:50 KST, with source SHA256
`2862763f5fb3dd8af957f55f1a5d42b501deba2ec123ae1b171fc0bf4d913662`.
Actual occupancy was three blocks/SM. The FP8 byte, GEMM/SMLP2 oracle and
replay gates all passed. The different split order changed some BF16 bits;
all elements remained within the existing FP32 oracle tolerance.
See [round2.log](round2.log) and [round2.json](round2.json).

M=6 baseline → unrestricted compact mode, microseconds:

| N × K | Warm | Change | Cold weights | Change |
|---|---:|---:|---:|---:|
| 6416 × 4096 | 41.364 → 42.222 | +2.07% | 74.768 → 73.728 | -1.39% |
| 4096 × 2048 | 14.056 → 13.134 | -6.56% | 31.664 → 29.696 | -6.22% |
| 1024 × 4096 | 10.955 → 12.709 | +16.01% | 21.504 → 21.504 | +0.00% |
| 4096 × 512 | 6.852 → 8.723 | +27.30% | 14.640 → 14.656 | +0.11% |
| 6144 × 4096 | 32.516 → 29.895 | -8.06% | 76.512 → 71.888 | -6.04% |

The broad compact policy is rejected: reducing K splits and increasing
occupancy did not guarantee lower latency. The final selector admits only
M=6 with (N,K)=(4096,2048) or (6144,4096); all other shapes use the regular
two-block transposed kernel. The final dispatch and compact SMLP2 handoffs passed round 3 below.


## Round 3: final selective dispatch, PASS

Fleet `glm53m8sel`, started 01:00:25 KST. Source SHA256:
`4cabebb9919abbddb778d3c6addec09afb16133c6c10bf3cd43acaee2cb936d9`.
Raw results: [round3.log](round3.log), [round3.json](round3.json).

- All 12 geometries pass the independent FP32 oracle (highest matmul
  precision); worst relative error 6.214e-5, zero elements beyond the
  existing oracle tolerance. The compact paths can differ in BF16 bits.
- Every shape outside the compact selector is bit-equal to the baseline.
  All graph outputs are stable across replays. Actual occupancy is three
  blocks only for the two admitted shapes and two elsewhere.
- Compact SMLP2 producer and consumer graph cases pass 12 replays each:
  intermediate 3072 relative error 2.569e-6; 2048 relative error 1.914e-7.
- Final source and both composed snapshots have the identical SHA above.
  Local checks: 6,587 passed, including 30 megakernel regressions. Local
  Torch-dependent checks skipped because that interpreter has no Torch;
  numerical validation was performed by the GPU probes.

M=6 baseline → final selector, microseconds:

| N × K | Warm | Change | Cold weights | Change |
|---|---:|---:|---:|---:|
| 6416 × 4096 | 41.534 → 40.152 | -3.33% | 76.832 → 75.936 | -1.17% |
| 4096 × 2048 | 13.986 → 13.109 | -6.26% | 31.808 → 31.456 | -1.11% |
| 1024 × 4096 | 10.985 → 10.731 | -2.32% | 22.272 → 22.192 | -0.36% |
| 4096 × 512 | 6.923 → 6.716 | -2.98% | 14.752 → 14.544 | -1.41% |
| 6144 × 4096 | 32.961 → 30.240 | -8.26% | 71.888 → 71.552 | -0.47% |

The warm improvements on the two compact shapes repeat at 6.3–8.3%.
Their round-2 cold gains of about 6% did **not** repeat: final cold changes
are about 0.5–1.1%, with other shapes around 0.4–1.4%. Treat the larger cold
claim as unconfirmed. There is no measured end-to-end C=1 decoding gain.
Both profile options remain 0; the implementations are available for a
matched serving bracket without changing the parallelism or quantization.
