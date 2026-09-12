# ST native execution qualification — 2026-09-12

Candidate: `codex/st-performance-parity`, based on `cc6163c78613777225a4e48ac4ae7ac78871f97b`.
The fixed production release remains `5734b29fde84` until full-model gates pass.
Component results below establish arithmetic, state and replay correctness; they are not serving throughput measurements.

## Scope

| Adopted vLLM feature | ST implementation |
| --- | --- |
| One-shot AR | Owned TP4 RoCE transport, lockstep binary/rank-table agreement, NCCL arithmetic self-test, graph replay; other declared shapes and integer collectives use NCCL |
| MK GEMM / PDL | Native GB10 W4A8 dense kernels; fixed PDL, 96 CTA grid, compact/transposed M8; MK post/pre MHC fusion with model-owned scratch and AR consumer PDL |
| FP8 / NVFP4 dense prefill | W4 for M≤32, block-128 FP8 for 33≤M<1024, NVFP4 for M≥1024; vocabulary head stays FP8 |
| Prefill SP and FP8 AG/RS | Equal token shards around residual/MHC; full token order for attention and cache writes; FP8 packets at 4096+ full rows, BF16 below; non-divisible tails retain ordinary TP |
| GPTQ drafter | TP-sharded QKV, gate/up, output/down; 31 native linears (35 packs including the 5 FC K tiles), merged BF16 context KV; direct circular GQA reads and accepted-position writes |
| b12x static v2 | `t,r,sf6,q0`; implemented the missing FP32 route-scatter ABI while retaining each contribution's BF16 rounding |

ST has 202 target dense projections. The old 213 count includes 11 MLA `kv_b` matrices; ST's absorbed attention uses those as absorption weights rather than an equivalent standalone linear. They remain in their existing BF16 path.

Packs reuse the retired source BF16 arena regions; SF6 scales reuse their retired raw-scale regions. All moves happen before capture, with capacity, overlap and ownership checks. BF16 reference execution requires a fresh load after retirement. NVFP4 weight packs remain separate. The drafter's original arena reservation is retained, but its large matrix computation is TP-sharded and its packed copies use that reserved space.

GPTQ caches bind weight bytes, exact Hessian bytes, calibration namespace, layout, row scaling and packing code. Historical GPTQ files without a calibration digest are rebuilt; the legacy builder could silently save an RTN fallback under a GPTQ filename. Each rank prepared 237 packs: 215 GPTQ built from the current calibration and 22 RTN target projections without calibration (11 MLA q_b and 11 indexer wq_b). All 31 drafter linears, including the 5 FC K tiles, use GPTQ. Missing calibration is explicitly counted as RTN. New GPTQ failures abort preparation.

## Verified component gates

All distributed gates used the real rank order srv2/srv1/srv3/srv4, TP4, four GB10 SM121 devices. Private resource-limited containers ran beside the existing service, so elapsed times in those logs cannot qualify performance.

- Real target layers 0 and 3 plus full DFlash2: all four ranks passed contexts 0, 17, 128, 256, 4096 and 4097. Eager and captured outputs were identical (relative error 0); recurrent and paged caches were byte-identical. Retired arena storage was enabled.
- Drafter: contexts 0, 1, 2047, 2048 and 2057; proposal tokens agreed across all ranks and with eager execution. Observe counts 1, 3 and 6 gave identical ring states.
- One-shot: 1, 6, 12, 24 and 32 rows, two chained reductions, five changed-input replays per shape. FP8 AG/RS: full-row counts 128, 2128, 4096, 4100 and 6912 against an independent quantization/collective twin. All four ranks passed. Final transport build removes periodic phase tracing but retains stall diagnostics.
- MoE independent reciprocal/dequant oracle: two seeds, shared/mixed routing, 1/6/12/24 tokens, eight repeats; maximum relative error 0.0064935065, repeat and replay spread 0. Final log: `components/rank3/moe-static-oracle.log`.
- Dense CUDA tests: independent quantized arithmetic and BF16 error bounds, 1023/1024 dispatch boundary, changed-input replay, calibrated pack improvement, stale/unverified cache refusal, retired arena canaries and graph addresses. Three tests passed.
- Direct-ring CUDA attention: startup, wrap, poisoned unused entries, GQA, changing device slot/layer offsets and untouched arena regions. Two tests passed.
- MK MHC CUDA test: independent mHC algebra, lossless/non-lossless coefficient cases, M=1/6/12/24/32 and replay. Passed.
- CPU regression: 287 tests, 81 CUDA-dependent skips, no failures.

`model-storage2-rank*.log` is the successful extended storage test. Earlier component failure logs are retained for the FP32 scatter ABI, arena slot stride and retired-head metadata fixes; their results are not merged with successful runs.

## Full-model procedure

`run-exclusive.sh` is run on srv2 after all pack jobs and component containers finish. It drains active work, suspends supervisor-generated traffic, stops the current fleet through its owning launcher, and starts the candidate on port 8001 with a private tier and dump directory. An EXIT trap collects evidence and restores the pinned production release and supervisor on either outcome. The installed production env file is not edited.

The candidate must pass the existing arena plus **12 GiB workspace / 4 GiB OS reserve** gate, largest 6912-token prefill at both ends of KV capacity, and full decode capture with **KV 8.73 GiB / 4 sequences / 1,048,576 context ceiling**. The boot proof additionally rejects unexecuted target W4/NVFP4, drafter W4/context FP8, vocabulary FP8, MK MHC and FP8 prefill collectives.

The onepass workload, Korean document, request limits, seed, quality gates and interior-window filters remain unchanged. `run_onepass_st.py` adapts only ST identity and step telemetry. `GLM53_API_PORT=8001` selects the private endpoint. Final-answer channels and truncation must be reported separately from the original retrieval gate, which includes reasoning text.

Candidate image: `st-engine:perf-f4d7-a`. All four image manifests report source SHA-256 `cbf41bab9dec49a7dc8ca38e83d5f01abf0edc4ebef1c930eb43cb4a5fc8e1e1`, matching dependency ABIs, and no installed vLLM.

### Candidate A: failed quality; investigation in progress

Full preparation passed on all four ranks. Rank 0 reserved 56.001 GiB arena, measured 6.073 GiB peak workspace and 24.998 GiB minimum immediately free memory across 159 preparation phases. Every required native execution marker fired. This established coverage and memory fit, not model quality.

The exact five onepass request hashes match the stock ST run. The 2K requests produced sensible Korean and passed 3/3 retrieval checks, with decode 56.93 / 58.99 / 60.47 tokens/s. Both 32K and 128K requests emitted repeated `!`, failed all six retrieval checks and accepted no draft tokens. The aggregate result is **3/9, failed**. The existing character-contamination scanner reported 0/5 but did not detect punctuation repetition; it does not establish valid output. The candidate is being repaired for adoption. Its failed long-context speeds cannot qualify a performance improvement.

Artifacts: `onepass-a/`. The image/source identity above belongs only to candidate A. Subsequent memory, preparation and diagnostic changes use separately mounted source and must receive a new image identity before another serving measurement.

A full-model diagnostic of the exact 32K message completed with finite values through all 45 layers in each explicit combination of SP on/off and NVFP4/FP8 prefill. All four combinations selected token 785 (`The`) at the last prompt position. Each operation was synchronized for that diagnostic; it therefore does not exclude a lifetime or ordering error. The next diagnostic uses production graph capture and the original three 2K requests before 32K, with checks only at step boundaries.

Restart investigations also found that faulting only the free-memory shortfall did not evict UMA file cache. Admission now faults the desired free extent, bounded by MemAvailable minus the existing 16 GiB headroom. Completed dense calibration/cache reads explicitly return their clean pages. Independent external model downloading on srv4 can refill cache during preparation; rollback attempts and verified API recovery must be recorded separately from launcher success.


### Root cause and candidate B

The bounded production-capture replay and intermediate-state bisection are in `diagnostic-controlled/`. No prefix cache was active in the original replay. Reusing dirty KV pages was not sufficient to reproduce the error. A chunk-boundary synchronization sometimes restored valid text but did not establish deterministic arithmetic; it is not the final fix.

A fresh, identical 6912-token input reproduced the first divergence at **L0.kda.in_proj**, before attention state or drafter observation: maximum output difference **10,376,640,987,136**, with input difference exactly zero. The actual fault is the pinned FlashInfer NVFP4 quantizer's PDL contract: both ordinary and TMA kernels load `SFScaleVal = *SFScale` before `griddepcontrol.wait`. ST computes that scale dynamically immediately before quantization. An early launch can therefore read the allocation's previous contents, even though the later input loads wait correctly. Subsequent normalization can turn this into finite, incorrect states, so a final `isfinite` test alone cannot detect it.

Both weight and activation NVFP4 quantization now explicitly use ordinary stream ordering (`enable_pdl=False`). This fixes the unsafe consumer boundary without a host/device synchronization in the inference path. Native MK GEMM, MHC and one-shot AR retain their PDL implementation. The regression test supplies a scale from a deliberately delayed PDL producer and checks exact GEMM equality across eight repetitions. The unmodified candidate A fails that test; the repaired implementation passes all eight repetitions. Logs: `components/nvfp4-order-old.log` and `components/nvfp4-order.log`.

Candidate B additionally releases calibration/pack file pages, repairs UMA admission without relaxing budgets, warms up the FP8 and NVFP4 shape ranges, and requires every target FP8 projection to execute before readiness. All four B image manifests report source SHA-256 `395e889612afd568582876cf8f10ddde76e40ff2f2e736e5df841fbd07846c8f`, no vLLM, and the pinned dependency ABIs. B passed native preparation and finite prefill gates, then rank 3 failed the unchanged 4 GiB immediately free OS reserve during decode warmup while an independent model download refilled the page cache. No B onepass ran. The pinned service was restored and its API verified; evidence is in `boot-b/`. Preparation now reclaims clean cache through the existing bounded admission mechanism at checkpoints, retaining the 12 GiB workspace and 4 GiB OS limits. This is included in the next integrated build.


### Integrated native defaults

The branch now integrates main `887f2678`, including asynchronous decode, vision, API sampling/tool handling and block-boundary prefix snapshots. Production fixes native execution and `t,r,sf6,q0`; obsolete execution/MoE/eager/reference knobs are rejected. Main's explicit 24 GiB KV and 24 prefix snapshots are retained. The workspace and OS limits remain 12 GiB and 4 GiB. A final memory checkpoint follows image/video and serving-shape qualification.

The merged CPU suite passed 411 tests with 113 CUDA/environment skips. The direct-ring GPU suite passed all three tests, including changing device acceptance counts and untouched cache regions. The native drafter's masked observation now shares the calibrated TP context projection and writes only accepted positions directly into the arena. Its graph pool is checked and released along with the other drafter graphs.

Candidate C's real-model probe exposed a missing prefill-only field in `DeviceStep` before target capture completed (`components/integrated-c/`). Device decode now explicitly declares empty image patches and prefix marks. The regression runs the real model embedding prologue with eager and device inputs; all 33 graph-contract tests pass (3 CUDA skips). Candidate D has source SHA-256 `85f781fb303a9be3cb0e990ebc220a31e0ae10397905164d189231b4c0235df7` on every node. Integrated model and full serving qualification are in progress; C did not run onepass.
