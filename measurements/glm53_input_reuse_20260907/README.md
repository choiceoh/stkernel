# One-time FP8 input packing for decode GEMMs

A bounded follow-up after the previous three candidates failed to establish
an output-speed gain. This experiment changes input preparation for MK GEMM;
production kernel sources, profiles and overlays are unchanged. A later
service recovery is recorded separately below.

The retained September 7 trace has 283 complete steps of the modal kernel
count. Within that trace, MoE takes 22.860 ms/42 launches, MK GEMM takes
9.228 ms/237 launches, and the step spans 47.75 ms. CUPTI times identify
relative costs; they are not unprofiled latency or a new matched baseline.
The trace predates the latest default bundle, so it is a prioritization clue.
Source trace: `dp0_pp0_tp0_dcp0_ep0_rank0.1788735581053433310.pt.trace.json.gz`.

Current GEMM blocks independently quantize the same input for every output
tile. The prototype packs each row/128-column group once, then uses the
existing `a_ready` GEMM consumer. FP8 conversion, scaling, weight packs,
MMA reduction and output rounding are preserved. An extra kernel and scratch
reads cost time; timing includes both the pack and the consumer.

The generator injects the prototype into a private copy of the approved-main
CUDA source. Both arms use the same compiled extension. Current pack2,
transpose, compact-M8 and M8-fastpath compiler choices are enabled. The
prototype uses the existing single-stream SMLP2 scratch; it is not a serving
implementation and must not be enabled for overlapping consumers.

`probes/run_gemm_input_reuse.py` requires a fleet **probe** hold and either an
idle healthy server or an already stopped server whose state is preserved.
It creates only its own container, uses CPUs 14–17 and a 10 GiB memory cap,
and stops that container if serving traffic appears. The corrected runner
requires 16 GiB available host memory and stops below 12 GiB.
A six-minute timeout bounds compilation plus GPU work. It neither restarts
the server nor writes serving overlays. Before/after container identity,
traffic counters and half-second traffic samples are retained.

Six geometries include M6/M8 decode, wide 4096/6144/6528 output projections,
and small shared-expert controls. Five changing-input graph replays per shape
check bitwise output equality, exact FP8 bytes/scales and the independent
FP32 quantized-weight oracle. Cold and warm timing each use 24 samples per
arm in alternating order. Cold timing flushes 64 MiB outside the measured
region; the pack kernel remains inside it.

## Result

The successful probe ran on srv2/GB10 at 15:34:31–15:35:08 KST, September 7,
with source `c475b65c46aef1df06f178156a523ce09529f86d`, image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`,
PyTorch 2.13.0+cu130 and CUDA 13.0. The original/generated CUDA hashes and
compiler flags are in [result.json](gpu/result.json). Thirty changed-input
numerical/replay cases passed: exact BF16 output, exact FP8 bytes/scales,
finite output and the independent quantized-weight FP32 oracle. There were
no differences beyond the oracle ULP allowance.

Medians in microseconds; positive reduction means faster. Each row has 24
samples per arm. The candidate includes its extra packing kernel.

| M,N,K | Cache | Baseline | Candidate | Reduction | Paired wins |
|---|---|---:|---:|---:|---:|
| 6,4096,4096 | Cold | 90.784 | 91.792 | -1.11% | 7/24 |
| 6,4096,4096 | Warm | 24.336 | 24.032 | +1.25% | 22/24 |
| 6,6144,4096 | Cold | 125.600 | 126.624 | -0.82% | 9/24 |
| 6,6144,4096 | Warm | 32.736 | 30.496 | +6.84% | 24/24 |
| 6,6528,4096 | Cold | 133.104 | 131.824 | +0.96% | 18/24 |
| 6,6528,4096 | Warm | 42.720 | 38.480 | +9.93% | 24/24 |
| 8,4096,4096 | Cold | 90.736 | 91.408 | -0.74% | 6/24 |
| 8,4096,4096 | Warm | 24.320 | 24.320 | 0.00% | 13/24 |
| 6,1024,4096 | Cold | 33.408 | 32.416 | +2.97% | 19/24 |
| 6,1024,4096 | Warm | 13.760 | 13.888 | -0.93% | 8/24 |
| 6,4096,512 | Cold | 20.128 | 21.152 | -5.09% | 7/24 |
| 6,4096,512 | Warm | 9.488 | 9.712 | -2.36% | 5/24 |

**Retain the wide-projection candidate; do not apply it universally.** The
operator correctly noted that a warm-cache win is useful even if the cold
gain is small. M6/N6528/K4096 is the strongest retained case: warm latency
falls 9.93%, with a 0.96% cold reduction as well. M6/N6144/K4096 gains 6.84%
warm with a 0.82% cold regression, so its use needs workload evidence. Small
regressing shapes should keep the original path. No measured full-model
gain is required to acknowledge these specific kernel improvements.

The 64 MiB flush is a cache-pressure fixture, not a reproduction of the full
model's cache/DRAM schedule. Neither these microseconds nor the earlier
CUPTI times establish a model step/output gain. The next acceptance question
is how often the retained projection sees useful cache residency during
decoding, followed by a matched step/output measurement. Full-step latency
and output tokens/s for this candidate remain unmeasured; no serving default
has been changed.

## Attempts and service interruption

1. `inputreuse0907`, 15:25:56–15:26:11, source `195fb97`: compilation failed
   at the existing single-context launch helper, before GPU execution.
   [Raw compiler log](compile-failure/probe.log) and
   [unchanged-service receipt](compile-failure/admission.json) are retained.
2. `inputreuse20907`, 15:27:05–15:27:29, source `4a32ec6`: the public head
   exited while compilation was running, before timing. The probe guard
   stopped its own container on connection refusal. The old runner's final
   traffic collection then also failed, leaving no admission JSON and an
   empty [probe log](interrupted/probe.log). This failure is not a GPU result.
3. `inputreuse30907`, source `c475b65`: the successful run above preserved
   the already stopped service, with no guard issues and return code 0.
   [Admission and memory samples](gpu/admission.json) record that state;
   they do not establish public service availability.

The head container exited at 15:27:28 KST with exit code 0 and
`OOMKilled=false`. Its log contains a graceful API/engine shutdown. Kernel
logs inspected during the incident did not show OOM/Xid events. The user
confirmed they had not intentionally shut it down. The initiating cause
remains unresolved; exit code 0 does not explain who or what sent the signal.
Only the probe's own named container was targeted by its stop command.

The initial runner nevertheless admitted compilation with only about 9 GiB
available host memory, falling to about 6.8 GiB during compilation. That
admission was inadequate. Source `c475b65` added the 16/12 GiB guards and
failure-safe final evidence collection. Do not rule out resource pressure
or attribute the shutdown to another operator without further evidence.

At the user's clarification, fleet was free and restoration was submitted
through the canonical fleet runner as `inputrestore0907`. It boots the
existing deployed overlays with the canonical production profile and no
candidate override. The launcher source was `e1a88d7`; the existing deployed
build remained `0aca81454720` / `6f797df`. No deploy was performed.

Recovery completed at 15:50:50 KST. The six 2K/4K/8K prefill warmup requests
completed, followed by the canonical onepass 2K Korean workload under its
own fleet probe hold at 15:51:05–15:51:24. It passed **retrieval 3/3, Korean
corruption 0/3**, with 1,200 generated tokens and no exclusive-traffic issue.
Its ten decode windows have median 21.8 step/s; this small defaults-only
recovery sample is not a candidate benchmark or a matched baseline.

The [live receipt](recovery/live.json) verifies **health 200**, four running
nodes, the same image and all 56 mounted overlay-file hashes. All nodes use
the approved decode defaults pack2/transpose/fastpath/MHC-BF16 = **1/2/1/1**;
NVFP4 static scale is **0**. Fleet was released with no queued work. Retained
records include the [restore log](recovery/restore.log),
[warmup log](recovery/prefill-warmup.log),
[onepass record](recovery/onepass.jsonl),
[onepass log](recovery/onepass.log), compressed pre-recovery head/worker logs
and the [incident memory samples](recovery/memory-incident.txt).

Local closure verification recomputed all 12 medians, checked all 30 gates
and both original/generated source hashes, and passed `git diff --check`.
No new PR, merge or candidate default promotion was performed.
