# Default head/FC replacement — 2026-09-17

Operator instruction: **“head랑 fc에서 deepgemm 바로 대채ㅔ”**. Native GLM boot
now installs cuBLAS readers for the shared target/draft vocabulary head, FC
prefill pack and committed-decode FC pack. FC decode uses five strided batches
with FP32 partials and one BF16 final store. Other dense readers keep their
existing implementation. This default is explicitly operator-selected;
**GB10 runtime/one-pass speed and live acceptance remain unmeasured.** No fleet
queue, production restart or deployment is part of this change.

## Serving behavior

- `FP8Linear` dispatches these prepared readers without importing/calling
  DeepGEMM. Calibration observers, padded head columns, explicit `out`, and the
  prequantized FP8 interface retain their contracts. Shared-calibration FC
  receives the explicit decode phase as well as separately calibrated FC.
- cuBLAS builds with the other native extensions before rank collectives. The
  implementation is fixed at weight preparation. A new eager row count only
  derives and validates layouts from that implementation; there is no heuristic
  enumeration, timing or fallback in the forward. Capture rejects an unwarmed
  row count. Graph pools own activation/output/partial allocations, and replay
  allocates zero additional Torch bytes.
- MX scales and split weight bytes are copied into a declared arena region after
  the original head/FC packs reach their final addresses. Boot admission and the
  budget report include this region: **92.234375 MiB/rank** with separate decode
  calibration, **89.734375 MiB/rank** with a shared FP8 FC pack. This includes head
  scales, direct FC scales and the 82.5 MiB split FC pack. Partial outputs belong
  to the existing graph/workspace ceiling, not this weight region.
- Native execution proof names each cuBLAS reader, selected algorithm, warmed
  rows, resident bytes and executed phase. A configured but unused reader fails
  the boot gate. Explicit W4 baseline policy still uses W4 for decode; its FC
  FP8 prefill reader and head use cuBLAS.

## Actual serving-call validation on the authorized RTX 5050

The probe calls `FP8Linear`, not the earlier manually bound comparison wrapper.
It uses the same real packed weights and pinned CUDA 13.2/Torch/cuBLAS/DeepGEMM
runtime as [the preceding comparison](../st_cublaslt_reform_20260917/README.md).
Inputs are seeded synthetic BF16, not captured model activations. It rejects
any DeepGEMM call from the replacement and changes input values for two graph
replays per cell. Every replay records zero extra Torch allocation bytes.

**23 cells pass** numerics (`rtol=.01, atol=.001`) and changed-input graph replay:
head M=1/7/8/14/16/21/24/28/32/56/64/256; FC decode
M=1/7/8/16/24/32/64; FC prefill M=1/128/512/2304.

| Reader | M | DeepGEMM ms | cuBLAS ms | Latency change |
|---|---:|---:|---:|---:|
| Head | 7 | 0.679617 | 0.593520 | −12.67% |
| Head | 8 | 0.681398 | 0.595760 | −12.57% |
| Head | 14 | 0.682396 | 0.598096 | −12.35% |
| Head | 16 | 0.682879 | 0.599136 | −12.26% |
| Head | 32 | 0.691177 | 0.606544 | −12.24% |
| Head | 256 | 0.951876 | 0.877912 | −7.77% |
| FC decode | 8 | 0.344555 | 0.322783 | −6.32% |
| FC decode | 16 | 0.352138 | 0.329351 | −6.47% |
| FC decode | 32 | 0.351014 | 0.344736 | −1.79% |
| FC decode | 64 | 0.362913 | 0.365542 | **+0.72%** |
| FC prefill | 1 | 0.333467 | 0.355616 | **+6.64%** |
| FC prefill | 128 | 0.428691 | 0.405040 | −5.52% |
| FC prefill | 512 | 1.008703 | 1.018360 | **+0.96%** |
| FC prefill | 2304 | 4.608929 | 4.739640 | **+2.84%** |

These are means from one warm B/A/A/B bracket per cell, including producer and
output work. No engine step/s conclusion follows. Unlike the earlier bound
microbenchmark, the serving graph owns its temporary tensors and initializes
MX padding in the producer; its measured FC improvement is about 6%, not 8–9%.
The table deliberately retains the losing cells. The operator requested the
whole reader replacement; the result is not a claim that every shape is faster.

A further algorithm/producer search found a better internal split-K option at
M=512, but not at M=2304. A 512-row tiled FC implementation still measured about
4.73 ms at M=2304 and did not fix the regression, so it was removed. The final
reader was restored byte-for-byte to the above 23-cell source. The unsuccessful
experiment remains as `rejected-prefill-tiles.patch` plus
`rtx5050/serving-fc-tile-trial.json`; the search is
`rtx5050/serving-prefill-search.json`.

## Other checks and reproduction

- Eight focused checks pass, including native GPU scale/stride validation,
  inference tensors, two independent changed-input graphs, output alias rejection,
  default dispatch, FC phase forwarding and resident/proof contracts.
- Related CPU suite: **54 passed, 9 GPU/interpreter checks skipped** (63 total).
  Covers native builds/cache, boot paths, execution proof and drafter storage.
- GPU-hidden native build/dlopen and **93 SM121 Triton variants** compile,
  including graph-owned split padding and the prequantized scale converter.
  This does not establish GB10 execution.
- Final GPU/compile receipts match their recorded current source hashes.

Raw evidence lives in `rtx5050/`. `run_5050.sh` uses the owned direct-host probe
lock, immutable image, 2 CPUs/4 GiB and the existing two packed weights. It never
uses the fleet queue or stops production. The earlier tests initially lacked
launcher files in the scratch checkout; those were supplied before the passing
CPU suite. A test's broad module mock initially disturbed Triton's extension
imports; dependencies now load before that mock. The native/GPU checks then
passed. The offline AST compiler requires explicit values for default constexpr
arguments, so split `PAD=False` is explicit in that harness.
