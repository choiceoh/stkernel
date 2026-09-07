# Warp-independent input reuse: development and acceptance

The user requested a substantially larger gain than the earlier input-reuse
prototype, followed by default promotion. This remains conditional on the
actual serving-source and C=1 step/output gates; the new profile flag is 0.

## Completed private-source sweep

Source `9c3efdc`, generated CUDA SHA-256
`d97d65b84600bd9612f347d0430061c6f030238edce9c30b4496001a9c5e3277`,
GB10/SM121a, image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`,
PyTorch 2.13.0+cu130. The canonical fleet session `inputwarp0907` ran the
probe after an explicit idle-service shutdown and restored public defaults
at 16:18:26 KST. The next queued prefill session then took the fleet.
The probe and restoration both returned 0; see [receipt](gpu/receipt.json).

Each warp stages only the 16 weight rows it consumes. Its K loop uses warp
synchronization rather than a block barrier. Prepacked X goes directly into
MMA registers. A second layout coalesces the MMA lane reads and packs the
M<=8 groups into 32 KiB with contiguous group scales. Both warp variants
compile to 78 registers, no local spills, three blocks per SM and 28,672 B
dynamic shared memory.

Seven geometries, five changing-input cases and multiple split plans pass
**290 numerical gates**, including exact equality with the baseline at the
same split, finite values and the independent FP32 quantized-weight oracle.
There are **102 timing rows**, each with 24 alternating samples per arm.
[Raw results](gpu/result.json) and [compiled source](gpu/candidate.cu).

Selected coalesced routes (microseconds; all preparation included):

| M,N,K / split | Cache | Baseline | Candidate | Latency reduction |
|---|---|---:|---:|---:|
| 6,6528,4096 / 8 | Warm | 42.752 | 32.512 | 23.95% |
| 6,6528,4096 / 8 | Cold | 132.768 | 132.080 | 0.52% |
| 6,4096,512 / 1 | Warm | 9.680 | 7.936 | 18.02% |
| 6,4096,512 / 1 | Cold | 20.400 | 20.128 | 1.33% |

N6144 and N4096/K4096 show warm gains with cold regressions; shared-expert
N1024 is slower warm. These are excluded from the serving candidate.
M1/M8 remain on the existing path. Background and low-rank calls also retain
the original path. Thus a winning standalone K512 result does not establish
that the background shared-expert join improves.

The sweep's cold fixture writes 64 MiB before timing. Existing benchmark
notes identify residual dirty writeback as a source of inflated cold latency.
These cold rows are retained, not replaced. The serving-source gate uses
read-only eviction instead. Do not compare the two fixtures' absolute times
or convert these microseconds to model step/output gains.

## Serving implementation and pending gates

The production candidate uses invocation-owned packed storage, retained by
the CUDA graph pool, instead of the prototype's global shared-expert scratch.
It is restricted to the two M6 foreground geometries above, with the original K splits
8 and 1. These preserve the baseline reduction order and exact output bits.
The marginally faster split 4 (24.10% warm) is not selected. A startup numerical/replay gate disables only input reuse on failure,
keeping the existing GEMM available. Startup captures cannot emit a serving
receipt. The profile flag `VLLM_GLM53_MK_INPUT_REUSE` is currently **0**.

`probes/gemm_input_serving_gate.py` checks the actual production source,
fallback shapes/background calls, changing captured inputs, allocation
poisoning, alternating live graphs and read-eviction/warm timing. The reserved
serving runner also checks the new kernels with Compute Sanitizer racecheck
and memcheck before deployment.

The subsequent matched B/A/A/B uses the standard Korean onepass 2K/32K/128K
ladder plus five fixed 2,048-token C=1 responses per arm, a loopback endpoint,
and all-rank receipts collected **before** the chain's failure judge. A
snapshot of approved main supplies the unconditional public recovery path.

Actual production-source GPU results, racecheck/memcheck, paired C=1 rates,
quality and default promotion remain pending. No serving gain is claimed yet.

## First actual-source run: incomplete graph-lifetime gate

Fleet `inputserve0907` started at 16:49:10 KST on source `5e63ad7`.
CUDA SHA-256 `0cbbe7c96d3815de307cf1d5d56fe7fab1fac25a41c5b430b5230aca31603baf`.
All 90 individual numerical rows and the baseline bit comparisons passed.
The subsequent retained-graph test failed at its first iteration
(`relative=1.0`, 24,570 values over the ULP limit). The runner aborted before
sanitizers or any serving arm. Approved defaults were restored at 17:02:24 KST,
with health 200 checked at 17:03. The next fleet boot job then took ownership.
The JSON still says RUNNING because the assertion interrupted it; the
[failure log](failed-lifetime/production-gate.log) is the terminal result.

Inspection found that the test retained X, reference weights and graphs,
but released the original packed-weight tensors between shapes. Those
external CUDA graph arguments must stay alive. The retry retains each pack
with its graphs and replays the baseline alongside the candidate. This
diagnosis still requires the corrected GPU test to pass.

The interrupted run's timings below are preliminary only, not acceptance:

| M,N,K / split | Cache fixture | Baseline us | Candidate us | Reduction |
|---|---|---:|---:|---:|
| 6,6528,4096 / 8 | Warm | 42.752 | 30.528 | 28.59% |
| 6,6528,4096 / 8 | Read eviction | 77.568 | 76.544 | 1.32% |
| 6,4096,512 / 1 | Warm | 9.504 | 7.936 | 16.50% |
| 6,4096,512 / 1 | Read eviction | 14.224 | 11.968 | 15.86% |

[Raw result](failed-lifetime/production-gate.json). No default was changed.
