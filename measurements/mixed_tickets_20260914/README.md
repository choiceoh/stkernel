# M2: served shared readers and fence-owned layer tickets

The mixed FFN component now binds the GLM profile's actual router, prepared
ModelOpt weight views and shared DenseLinear readers. Its layer scheduler owns
rank agreement, bounded cold dispatch, output sums and cancellation retirement.
Ordinary request scheduling and forward graphs still do not select mixed work.

Final implementation: `ccc253c15c127f42328152977ef945f9c347a291`, including
main `b02bff42` (#914 DSA inputs) and `96af4dbb` (#916). Conflicts in lane,
execution and knob registration retain both sets of features. C1 SF6 word expansion and
intact eight-byte quantized activation stores remain in both ordinary and
prepared kernels. The ordinary kernel AST is compared against that reviewed
main body, with only the private frontend guard removed.

## Ownership and execution

- `Glm53Net.prepare_mixed_ffn` binds one layer's already replicated, normalized
  decode/prefill FFN inputs. It executes the actual route selector and uses the
  same weight views and ModelOpt quantizer divisors as the ordinary MoE lane.
- `BoundMixedShared` keeps C1 SharedOverlap and its fused W4 shared MLP. Wider
  decode and prefill retain the existing DenseLinear W4/FP8 selection. Prefill
  shared GEMMs never run inside the C1 routed callback. Actual packs are retained;
  reader/storage replacement and tracked tensor mutation are refused. Native
  inference packs have no version counters and retain the dense lane's immutable
  storage contract.
- `MixedLayerScheduler` admits at most four tickets on one eager stream. Request,
  layer, serial, slot/source generation, descriptor digest, hot/cold quotas and
  phase must agree across ranks. One owner cannot be lent to two live tickets.
  An owner advanced outside its ticket is refused.
- Outputs use one fixed process-group sum: NCCL in native TP4, Gloo in the CPU
  process test, identity in one-rank GPU qualification. Pending packet consumers
  and output shape/dtype disagreements are voted before the data collective.
  This API does not select one-shot transport or LocalTP.
- Each `advance` dispatches one cold window. `finish` refuses unfinished windows.
  The first window also packs all cold routes: the quota bounds MMA task count,
  not latency or preemption. Decode and prefill results carry post-reduction events.
- Cancellation stops future dispatch and retains sources, partials and outputs
  until every rank's reader and last borrowed-output consumer fences finish.
  Failed local dispatch may leave different owner cursors; cancellation still
  drains each rank. A retired handle cannot target a reused slot. Lost processes
  or an unrecoverable CUDA/communicator failure require upper-level recovery.

## CPU and native compilation

- [cpu_postmerge.json](cpu_postmerge.json): **44 modules, 357 discovered, 332 passed,
  25 CUDA skips**. [cpu_postmerge_runner.py](cpu_postmerge_runner.py) isolates each
  module and caps CPU thread counts. This includes both new DSA components and
  the cleanup repair. The earlier [cpu.json](cpu.json) remains the separate
  pre-merge result; repeated tests are not added to the final count.
  No GPU was visible and CUDA was not initialized.
- The real four-process Gloo test covers local preparation failure, descriptor
  and collective order mismatch, partial cold dispatch failure, event recording
  failure, pending packet consumers, output shape mismatch, decode/prefill sums,
  and retirement delayed independently by reader and consumer ranks. GPU launches
  are simulated only in this host protocol test.
- [compile.json](compile.json): **eight actual CuTe/PTXAS/TVM-FFI builds** on
  SM121 metadata without initializing CUDA: hot/cold producers and
  ordinary/prepared M16, M32 and M128. The 9240/32768-row requests reuse the same
  dynamic handle. This is compiler proof, not device execution or boot timing.
- CPU/compiler image:
  `sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`;
  Torch `2.13.0+cu130`, CUDA `13.0`.
  [postmerge_source_continuity.json](postmerge_source_continuity.json) verifies
  the final CPU/GPU manifests and unchanged expert-compiler inputs. The updated
  dense extension is covered by the post-merge GPU run.
- The exact implementation passed [CI](https://github.com/choiceoh/stkernel/actions/runs/34796519386/job/103830570625).

## GPU qualification

Canonical probe: [engine_mixed_tickets_check.py](../../probes/engine_mixed_tickets_check.py).
It uses one real L3 TP4 weight shard on one GB10, D=8/32 and P=9240/32768,
the actual profile router on synthetic activations, and the same DenseLinear
W4/FP8 packs in native and mixed arms. Shared weights use RTN from checkpoint
BF16; no production GPTQ calibration store is assumed. Split/mixed samples
alternate order, create fresh routes/storage, and include preparation/admission
in wall time. Stage synchronizations make readiness observable. These are
component timings, not a serving-performance comparison.

The gate compares complete outputs against the ordinary FFN, records native
repeat error, exercises a foreign-stream consumer, checks queued/decode/cold
cancellation and stale handles, and hashes unchanged sources/shared packs.
Its budget is 8 GiB, enforced by the canonical single-GPU lane's room checks.

Reservation `st-mixed-tickets0914v4`, ticket `178934993625946`, admitted frozen
`ccc253c1` on srv4 with image
`sha256:8190d08e822e1f9d18dda5a127a5d9a9e8c53ef4b5136d1154f9ea4b7727f1ed`.
That image reports Torch `2.13.0+cu130`, CUDA `13.0`, Triton `3.7.1`; its ID is
distinct from the CPU/compiler image and is recorded separately.
[gpu_postmerge_admission.json](gpu_postmerge_admission.json) records **success**,
125.5 seconds in the payload. [gpu_postmerge.json](gpu_postmerge.json) retains
all 32 complete samples, source/weight/pack hashes and output errors;
[gpu_postmerge_summary.json](gpu_postmerge_summary.json) summarizes them.
All decode outputs have zero measured maximum and RMS error. Mixed prefill's
worst relative maximum error is 0.881%, RMS 0.237%, within the fixed 2%/0.4%
component gate. Native repeated prefill also has nonzero BF16 scatter error.
These tolerances do not establish generation quality or acceptance.

| Decode rows | Prefill rows | Cold windows | Mixed prefill max RMS error | Warm prepare/admit median | Warm full completion median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 9240 | 15 | 0.215% | 164.0 ms | 212.6 ms |
| 32 | 9240 | 15 | 0.237% | 118.8 ms | 142.8 ms |
| 8 | 32768 | 46 | 0.153% | 452.7 ms | 516.9 ms |
| 32 | 32768 | 46 | 0.146% | 282.2 ms | 360.7 ms |

Warm medians use the three mixed samples after the conservatively marked first
sample. Fresh preparation/admission still costs 119–453 ms in these fixtures;
that cost cannot be put on the live decode critical path. Samples ran beside
production, include measurement synchronizations, and have no matched native
serving-timing baseline. Do not interpret the table as a speedup. Peak Torch
allocation was 3.874 GiB, including weights and outputs, within the 8 GiB budget.
Queued/decode/cold cancellation was exercised in the D=8/P=9240 case;
foreign-stream consumers and final retirement were exercised in all four cases.

The first successful pre-merge run remains in [gpu.json](gpu.json), with its
own [admission](gpu_admission.json) and [summary](gpu_summary.json). The two
beside-production runs do not measure the DSA merge's performance impact.

The v2 reservation ended before GPU computation because its pinned image was
absent on srv4; it supplies no numerical evidence. Earlier S/P/M1 reservations
retain their separately frozen source identities and are not replaced by this
one-rank M2 result.

An unrelated CI cleanup poll could see procfs ESRCH after a child was reaped.
The test now treats that as stopped; its existing three process-cleanup tests
passed on macOS and in the GPU-hidden Linux image. No probe process handling or
serving code changed for this repair.

## Remaining gate

M3 must connect real request arrival/cancellation, layer pause/resume, cache and
prefix ownership, decode continuation and graph/TP4 collective ordering. It must
then compare matched 32K/128K C=1/C=4 onepass TTFT, output tok/s, acceptance and
quality against the production schedule. M has no serving selector or knob;
S/P defaults stay off. No serving speedup or default promotion is established.
