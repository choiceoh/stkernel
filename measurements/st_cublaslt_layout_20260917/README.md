# cuBLAS FC layout and epilogue improvement — 2026-09-17

The default GLM FC decode reader now pads each of its five weight partitions'
physical row pitches by 384 bytes, then fuses the FP32 partial sum with RMS
normalization. Logical K, every FP8 weight/activation value, accumulation type,
BF16 rounding before normalization, optional FP32 bias, and normalization's
rounding boundaries are preserved. The vocabulary head and FC prefill retain
their matrix layouts.

**On the operator-authorized RTX 5050, actual FC decode + RMS calls are
5.2–6.5% faster than the preceding main reader (PR #1071).** This is component
latency, not engine step/s or acceptance. GB10/TP4 execution remains unmeasured;
no queue, deployment, service stop or restart was used.

## Matched final comparison

The harness loads `baseline_reader.py`, frozen byte-for-byte from main
`b41efc7d` / PR #1071, and calls both readers through `FP8Linear`. For FC decode,
the baseline includes the existing separate RMS kernel; the candidate returns
the same normalized result directly. Both use identical real PackStore weights
and changed, seeded synthetic inputs. These are not captured model activations.

Two warm B/A/A/B brackets per cell include activation quantization, MX padding,
GEMM, FP32 reduction and RMS (with bias where specified). The table averages all
four measurements per arm, after an independent warm bracket for each arm.

| FC decode rows | Previous main ms | Candidate ms | Latency change |
|---|---:|---:|---:|
| 1 | 0.32205 | 0.30303 | -5.91% |
| 7 | 0.32872 | 0.30743 | -6.48% |
| 8 | 0.32813 | 0.30789 | -6.17% |
| 16 | 0.33592 | 0.31536 | -6.12% |
| 24 | 0.34193 | 0.32096 | -6.13% |
| 32 | 0.34833 | 0.32690 | -6.15% |
| 64 | 0.37409 | 0.35461 | -5.21% |

With FP32 correction bias, the same seven widths improve 5.96–6.32%.
FC prefill differences are -0.28% to +0.11%; head differences are -0.35% to
+0.57%. Neither is a demonstrated execution-speed gain. The earlier ~3% large
FC prefill disadvantage against DeepGEMM is **not fixed** by this change.

All **28 cells** are **bit-identical** to previous main, including two
changed-input CUDA graph replays each and zero additional Torch allocation
bytes during replay. FC covers M=1/7/8/16/24/32/64 with and without bias and
prefill M=1/128/512/2304; head covers M=1/7/8/14/16/24/32/56/64/256.
[FC receipt](rtx5050/layout-final-fc.json), [head receipt](rtx5050/layout-final-head.json).

## Preparation and memory

The native plan first validates the already declared ID 70 / tile 20 / stages
36 implementation. A supported zero-workspace, <=16-byte-aligned implementation
needs one AlgoCheck and no catalog/heuristic enumeration. If the device cannot
use it, the existing **boot-only** enumeration still supplies a valid cuBLAS
choice; no forward-path tuning or DeepGEMM fallback is added. Derived row plans
preserve the physical pitch and validate the selected algorithm as before.

Warm host preparation averages (six samples per arm, interleaved B/A/A/B):

| Plan | Previous enumeration ms | Direct validation ms |
|---|---:|---:|
| Head | 0.184190 | 0.009940 |
| FC direct | 0.187848 | 0.011670 |
| FC split | 0.237386 | 0.011549 |

This reduces **algorithm preparation** by about 94–95%; the absolute saving is
less than 1 ms across these three plans. It is not a 94% boot-speed claim.

The new physical pitch costs **7.5 MiB/rank** for the FC split pack. Arena
admission and its independent profile declaration include it: total extra
cuBLAS weights/scales are now **99.734375 MiB/rank** with separate FC decode
calibration or **97.234375 MiB/rank** with shared calibration. No new runtime
workspace is added. Fused RMS removes one intermediate BF16 output and one
launch per FC decode call. Native execution proof requires the fused path to
have executed; phase, calibration observer/mask and bias selection remain explicit.

## Checks and runtime

- Nine focused CPU/native-GPU checks pass, including independent graph buffers,
  alias rejection, inference tensors, fused dispatch and padded logical-output
  refusal. Actual serving receipts establish the larger real-pack cases.
- Related CPU suite: **101 passed, 14 GPU/interpreter checks skipped** (115
  total). Native build/dlopen and **107 SM121 Triton variants** pass offline.
  Offline compilation is not SM121 execution proof. Receipts are beside the GPU files.
- GPU runtime: RTX 5050 / SM120 / 20 SMs, driver 595.79,
  Torch 2.13.0+cu132, CUDA 13.2, cuBLASLt 130400. Immutable image:
  `sha256:9f496f0dabe3a7b495d9b97181913cc20be1e4b3d3fcf2694407e34f24b3981b`.
- Real head and FC packs are the same pinned artifacts as
  [PR #1071's serving comparison](../st_cublaslt_serving_20260917/README.md).
  Full pack identities and source hashes are in each receipt.
- `run_5050.sh` uses the owned direct-host lock, two CPUs and 4 GiB; it cleans up
  only its own named container. The lock excludes cooperating probes, not
  arbitrary Windows desktop activity. Before/after utilization receipts are kept.

The initial CPU scratch lacked the bench modules required by a broader draft
precision test. They were synced before the final run. The initial offline
compile rule treated every new kernel as an FP8 producer and rejected RMS's
existing IEEE division; the rule now keeps that restriction on producers and
records RMS division without changing its arithmetic. Both initial logs are retained.

## Alternatives tested and rejected

- Directly reading strided partitions of the original FC weight saves 80 MiB,
  but is about 5–11% slower at common decode widths. The runtime change was
  removed (`rejected-strided-weight.patch`, `compare_split_layout.py`).
- P=2/4/5/8/10/16 partition sweeps: some narrow cells improve with P=16, but
  wider decode and prefill regress. Five remains the common decode geometry.
  This initial sweep stopped after 47 recorded cells on a CUDA `device not
  ready` error; its incomplete receipt is not a final pass. Later fresh owned
  processes completed the final 28-cell comparison.
- Larger producer tiles and reversed prefill operands did not improve the
  2304-row FC pipeline. `compare_prefill.py` and `prefill-layout.json` retain
  the measurements. A supported internal cuBLAS split-K helps M=128/512 but
  regresses the large shape; no shape-dependent serving switch was added.
- Weight pitch padding was screened over 0/64/128/192/256/320/384/448/512/768/
  1024/1280/1536/2048/3072/4096 bytes. 384 was selected and then measured through
  the final independent serving brackets. Adding whole padded rows between
  batches did not reproduce the gain (`rejected-row-padding.patch`).
- The fused epilogue alone passes bit equality with/without bias; its small
  timing samples are exploratory. The final conclusion uses whole FC + RMS
  calls, not epilogue-only percentages.
- The original FP32 block-128 scale modes are listed for compute capability
  9.0 in NVIDIA's [CUDA 13.2 scaling support table](https://docs.nvidia.com/cuda/archive/13.2.1/cublas/index.html#narrow-precision-data-types-usage).
  They were not introduced as an alternative backend on SM12.

The `compare_*.py` files are historical probes, not runtime imports. To replay
an experiment, copy it to its recorded `probes/engine_cublaslt_*_check.py` path
in an isolated scratch, with the corresponding native patch. Source hashes
identify the exact version used; `rejected-strided-weight.patch` applies to
`b41efc7d:engine/kernels/dense/cublaslt.cpp`. The maintained final check is
`probes/engine_cublaslt_reform_check.py` and requires the frozen baseline explicitly.

## Main integration validation

GPU receipts remain tied to `d35047b4`. Integration with #1068 at `d855cbc4`
preserves W4 `producer_pack` and FP8 `normalization` in `DenseLinear`. The
related CPU suite passed 97 tests with 14 skips (111 total), and GitHub engine
CI passed on that integration commit. The subsequent #1075 merge adds router
research and resolves an appended measurement-record conflict.
`integration-audit.json` verifies the cuBLAS source files and `FP8Linear` AST
are unchanged from the GPU-tested source; this is source evidence, not another
GPU run. Raw receipts retain their original identities.
