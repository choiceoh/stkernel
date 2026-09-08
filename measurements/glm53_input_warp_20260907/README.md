# Warp-independent input reuse: development and acceptance

The GLM53 profile now defaults `VLLM_GLM53_MK_INPUT_REUSE` to **1**, following
the operator's explicit promotion request on 2026-09-07 after the results
below were reported. Set it to **0** to restore the original GEMM. The startup
numerical/replay gate still disables only input reuse if its checks fail.

The real-width kernel now measures **24.10% lower warm latency** and **3.22%
lower read-evicted latency**, with numerical, graph and sanitizer gates passed.
Three real serving boots completed. Window medians were essentially unchanged;
A2 triggered the existing Korean gate on two `Halvorsen博士` expressions.
The gate failure and incomplete B/A/A/B remain recorded. Default promotion
is an operator decision; it does not establish a stable whole-model speedup
or resolve the mixed-script quality finding. Historical raw summaries retain
their original default-off and acceptance status.

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

## Serving implementation and validation scope

The production candidate uses invocation-owned packed storage, retained by
the CUDA graph pool, instead of the prototype's global shared-expert scratch.
It now selects the real M6/N6416/K4096 foreground projection, with the original
K split 8. N6528 is its padded storage width, not its logical output width.
The final tile has only 16 live rows: seven warps skip weight reads and MMA,
and the reducer skips padded columns. K512 calls are background calls and
remain on the existing path. The original split preserves reduction order.
The marginally faster split 4 (24.10% warm) is not selected. A startup numerical/replay gate disables only input reuse on failure,
keeping the existing GEMM available. Startup captures cannot emit a serving
receipt. The profile flag `VLLM_GLM53_MK_INPUT_REUSE` is now **1**.

`probes/gemm_input_serving_gate.py` checks the actual production source,
fallback shapes/background calls, changing captured inputs, allocation
poisoning, alternating live graphs and read-eviction/warm timing. The reserved
serving runner also checks the new kernels with Compute Sanitizer racecheck
and memcheck before deployment.

The subsequent matched B/A/A/B uses the standard Korean onepass 2K/32K/128K
ladder plus five fixed 2,048-token C=1 responses per arm, a loopback endpoint,
and all-rank receipts collected **before** the chain's failure judge. A
snapshot of approved main supplies the unconditional public recovery path.

The sections below retain each stage's result. Actual-source GPU gates are
complete. Balanced serving acceptance remains unresolved after the operator
promoted the default.

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
with its graphs and replays the baseline alongside the candidate. The
corrected GPU test below confirmed that this fixes the failure.

The interrupted run's timings below are preliminary only, not acceptance:

| M,N,K / split | Cache fixture | Baseline us | Candidate us | Reduction |
|---|---|---:|---:|---:|
| 6,6528,4096 / 8 | Warm | 42.752 | 30.528 | 28.59% |
| 6,6528,4096 / 8 | Read eviction | 77.568 | 76.544 | 1.32% |
| 6,4096,512 / 1 | Warm | 9.504 | 7.936 | 16.50% |
| 6,4096,512 / 1 | Read eviction | 14.224 | 11.968 | 15.86% |

[Raw result](failed-lifetime/production-gate.json). No default was changed.

## Corrected actual-source gate

Fleet `inputserve20907`, source `36b71f6`, started at 17:31:44 KST.
The CUDA source is unchanged from the first attempt. Retaining the external
packed weights fixes the graph test: all 90 numerical rows, exact baseline
bits, 40 alternating candidate graph replays and the startup gate pass.
Compute Sanitizer racecheck reports **0 hazards, 0 errors, 0 warnings**;
memcheck reports **0 errors**. See the [GPU result](inactive-shapes/production-gate.json),
[racecheck log](inactive-shapes/racecheck.log) and [memcheck log](inactive-shapes/memcheck.log).

N6528/K4096 again improves: warm 42.560 -> 30.464 us (-28.42%), read-eviction
77.360 -> 75.472 us (-2.44%). N4096/K512 read-eviction improves
15.920 -> 11.952 us (-24.92%), but its short warm samples are noisy and
regress 11.072 -> 12.000 us (+8.38%). Retain these contrary samples; do not
claim a stable warm improvement for K512 from the corrected run.

No serving arm ran in this attempt. Main advanced to `619cfec` (benchmark
orchestration PR #443) during the queue wait. The deployment ancestry guard
rejected both the old candidate base and its pinned `8476c15` restoration
checkout after the GPU test had stopped the service. A first current-main
recovery verified all 56 overlays on four nodes, but its shell runner was
modified during execution and failed before boot. A separate immutable
recovery runner then started the verified current-main service at 17:42:27.
Recovery finished at 17:49:22 with quality 3/3, Korean corruption 0/3, and
health 200 checked before the next campaign. See [recovery records](recovery-current-main/records.jsonl)
and the preserved [stale deployment rejection](stale-deploy/deploy.log).
These failures are orchestration failures, not GPU numerical failures.

The serving runner now checks current-main ancestry **before stopping the
service**, and refreshes its approved-main recovery checkout at cleanup.
`reuse_input_gpu_evidence.py` allows gate reuse only with byte-identical
CUDA, Python driver, fixture and profile, and valid PASS/sanitizer receipts.

## Runtime routing correction

Fleet `inputserve30907`, source `aaaa9c2`, reused the preceding identical GPU
source. Its B1 measured 21.732 pooled step/s, 21.826 median window step/s and
69.395 output tok/s; quality was 24/24 and Korean corruption 0/10. The A1
startup numerical gate passed but all four ranks had **no input-reuse capture**.
The model captures logical N6416 (padded to N6528); the old predicate selected
logical N6528. The other selected K512 calls were background calls. Thus A1
did not execute the new kernel and cannot establish a serving gain. The
wrapper rejected it and the remaining arms were not run. Preserve its raw
[records](inactive-shapes/records.raw.jsonl) and four-rank receipts as a no-op
control, not candidate acceptance. A1 measured 21.744 step/s and 70.541 output
tok/s: even without executing the new kernel, output tok/s moved +1.65% while
step/s moved only +0.05%. This is a concrete reason to keep the two metrics
separate and use balanced independent boots.

Approved main `e00df24` was restored at 18:30:34 KST with health 200. The next
candidate routes logical N6416 and skips the padded tail warps. It requires
fresh GPU numeric/replay/sanitizer gates. Both baseline and candidate must
prove the real capture on all four ranks **before** running onepass traffic.
Short-kernel timings now initialize events first and replay 16 warm-ups before
the measured window; this addresses a possible CPU enqueue gap without
discarding the older noisy K512 samples.

## Real-width production GPU gate

Fleet `inputserve40907` started at 18:34:51 KST on source `e997de1`, containing
approved main `0d82b7c`. CUDA SHA-256:
`8bb94129d5cf910b0e2cbe94a3f082e4da478d7490d6f215c6bc439f100a35c7`.
The 100 numerical rows, exact baseline bits, 40 alternating graph replays
and startup gate pass. Racecheck reports 0 hazards/errors/warnings and
memcheck reports 0 errors.

| Actual shape M6/N6416/K4096, split 8 | Baseline us | Candidate us | Latency reduction | Faster pairs |
|---|---:|---:|---:|---:|
| Warm | 42.624 | 32.352 | 24.10% | 32/32 |
| Read eviction | 78.000 | 75.488 | 3.22% | 30/32 |

The warm paired reductions range from 22.85% to 34.39%, with a median of
23.97%. The timing includes input preparation. These are kernel results;
the incomplete serving measurements below do not establish a stable model gain.
See [actual-source GPU evidence](serving/production-gate.json).

CPU validation on the candidate passes 6,685 logic checks, 30 megakernel
regressions, 73 fleet regressions, four driver behavior tests and nine
compile-cache lifecycle tests. The CPU contract audit hash was refreshed
after reviewing the three changed megakernel source-count assertions; the
audited math/layout/dispatch functions and loader are unchanged.

## Three real serving boots; Korean gate unresolved

| Boot | Mode | Pooled step/s | Window median step/s | Output tok/s | Facts | Korean gate |
|---|---|---:|---:|---:|---:|---:|
| IR6416B1 | Original | 21.641 | 21.865 | 69.558 | 24/24 | 0/10 |
| IR6416A1 | Input reuse | 21.724 | 21.853 | 71.147 | 24/24 | 0/10 |
| IR6416A2 | Input reuse | 21.870 | 21.855 | 69.959 | 24/24 | 2/10 |

All boots have five fixed 2,048-token C=1 responses, the same 2K/32K/128K
workload and image, no foreign requests, and unchanged four-rank receipts
before/after traffic. Both candidates prove the real 6416-wide capture on
all four nodes. The first pair changes pooled step/s by +0.385%, output tok/s
by +2.285%, and the window median by -0.055%. The larger standalone kernel
gain is not a comparable whole-model gain.

A2's existing gate fails on `Halvorsen博士` in fixed responses 2 and 4:
four Han characters total, no replacement characters, welded jamo or control
characters. It is a meaningful Doctor title in an English sentence, but the
record does not preserve whether it came from reasoning or final content.
The gate is **not** relabeled as passed. The chain stopped before B2; one
baseline and two candidate boots do not establish a balanced serving result.
See the [partial summary](serving/partial-summary.json), [raw records](serving/records.raw.jsonl),
[verdicts](serving/verdicts.jsonl) and [printed excerpts](serving/runner.log).

Approved main `757ea2b` was restored at 19:26:50 KST, with health 200 verified
at 19:28. The outer runner exited 4 because the quality gate failed;
`restore.status` is `restored`.

The diagnostic `probes/input_reuse_channels.py` retains the unchanged
onepass requests, combined text and existing gates while recording separate
SSE content/reasoning fields. It verifies request/output hashes and writes
after the timed request. Three transport tests pass. Its follow-up uses
pre/post four-rank capture proof and the normal chain/judge/restore flow.

Follow-up fleet session `inputchan0907` was queued at 19:33 on immutable source
`6560212`, with order IRCHANB1/IRCHANA1/IRCHANB2. Preflight passed. At submission
it was second in the queue, after a 40-minute running startup campaign and
a queued 45-minute GPU campaign (estimated start 20:52, not a deadline).
Output directory: `/home/choiceoh/glm53-logs/INPUTCHANNELS0907`. It reuses
the exact CUDA/driver/fixture/profile GPU evidence only after byte comparisons,
checks current-main ancestry before stopping service, and restores approved
main on exit. It does not automatically change the default or erase the
preceding failed gate.

The follow-up started at 20:00 on that immutable source. Its first baseline
passed facts 24/24 and the Korean check 0/10, but recorded **zero decode
windows**, failing the minimum-20-window measurement contract. The chain
stopped before its candidate, so it supplies no additional candidate quality
or performance verdict. Approved main was restored at 20:25:15 KST;
`restore.status` was `restored` and health 200 was verified after recovery.
This diagnostic failure does not change the earlier raw gate results.

After merging main's fleet update, a worker-completion race was reproduced:
`ensure_worker` could overwrite a terminal result as `interrupted` after
the worker finished between two reads. A post-lock terminal check fixes it;
the regression exercises every terminal state without launching a process.
The added test uses the existing temporary Store fixture, so its reviewed
fleet dependency hash was updated. The final combined CPU run passes
**6,685 logic checks, 30 megakernel regressions and 85 fleet regressions**.
