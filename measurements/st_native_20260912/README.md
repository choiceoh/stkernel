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


### Candidate D and the final integrated memory layout

All four D component ranks passed the real target contexts 0/17/128/256/4096/4097 with exact eager/replay outputs and byte-identical caches. Drafter contexts 0/1/2047/2048/2057 passed proposals, ordinary observation and device-counted masked observation for 0/1/3/6 accepted positions. Logs: `components/integrated-d/`. D's full-model boot stopped before allocation: rank 3 had 80.89 GiB immediately free and 96.21 GiB available against a 73.49 GiB arena plus 16 GiB headroom. The anonymous reclaim route required another complete 16 GiB floor, so it refused. No D onepass ran. The pinned API and supervisor were restored (`boot-d/`).

The model volume's clean download pages can be returned directly with `POSIX_FADV_DONTNEED`, without a large temporary allocation. On srv4, advising 33 completed/incomplete files returned 10,002,923,520 immediately free bytes; file contents and the active download were preserved. Admission now tries this for the explicitly supplied model directories before anonymous reclaim, and repeats it only when preparation lacks physical headroom. The existing final arena/workspace/OS checks remain unchanged.

Main `20db7f1d` adds 768-token prefix blocks, 96 protected snapshots, generated-boundary staging and JSON fleet leases. These are integrated. Native drafter caches now contain this rank's two KV heads; retaining eight would waste 30 MiB in every snapshot, or 2.8125 GiB across 96 snapshots. The original BF16 drafter weight reservation is still retained. The launcher and supervisor resolve the owning container for both old and JSON leases, preserve other owners, and execute the head's lease operations locally when already on srv2.

Candidate E uses **KV 16 GiB, 96 snapshots, TP4 KV shards, 12 GiB workspace and 4 GiB OS reserve** alongside the fleet's existing services. The explicit launch budget is `ST_KV_GIB=16`; the profile default remains 24 GiB. CPU regression: 441 tests passed, 113 environment/GPU skips. Source SHA-256 on every E node: `af86a057c318a0e6ab6bdbd195e816ab9f31c68fed9c9cf28c8f37f8404104d3`. GPU model and full serving qualification are in progress.

All four E component ranks subsequently passed the six real target contexts and five full-drafter contexts above. Both ordinary observation and device-counted masked observation (0/1/3/6) agree exactly with eager execution, with two KV heads per rank (`components/integrated-e/`). The three real-metadata budget tests also pass, including the exact 2.8125 GiB snapshot saving and redistribution of the smaller live state slots into paged KV under the fixed budget (`components/native-e-budget-cpu.log`).

Two drain attempts (60 and 600 seconds) ended before starting a candidate because existing production traffic kept arriving. The final exclusive wrapper temporarily rejects only new external TCP connections to the old HTTP/1.0 port; established responses finish normally. Its EXIT cleanup reopens ingress after restoring the pinned service, and an independent 60-minute system timer also removes the rule after an abrupt wrapper termination. The candidate remains on private port 8001. After onepass, bounded API checks exercise seeded two-choice sampling, JSON-schema output and a real red-image request before rollback.

### Candidate E serving transition failure and F repair

E passed full preparation, all native execution markers, the largest image/video shapes, and every rank's final `production/ready` memory checkpoint. Arena: 68.1419 GiB; maximum workspace: 9.6159 GiB; the lowest immediately free memory across ranks was 5.3752 GiB. The first 2K request produced 400 coherent tokens and retrieved 8127. The second request emitted its initial prefill token, then crashed before its first decode. This is **an incomplete, failed onepass**, not a quality or performance pass (`onepass-e/`).

`AsyncDecode.launch` rejected a new row even after all pending readbacks had drained, because the old batch identity remained nonempty. It now rebuilds the device view from host state whenever there is no pending readback. Joining or invalidating a batch with pending work still fails; shrinking a batch retains its device progress. The regression executes the real pipeline launch, commit and readback with only model kernels replaced. The old image fails three transition cases; the repaired code passes all five pipeline tests. Boot now additionally launches and resolves the full asynchronous depth at each batch width 1/2/3/4, after the synchronous warmup, so this path executes before readiness.

E's initial rollback also failed old-release memory admission on srv4; its old anonymous reclaim did not return enough model file cache. After returning clean model-volume pages directly, the pinned service booted and answered the capital-city request with `서울` (`onepass-e/restored-chat.json`). The final wrapper includes that cache return in rollback too; model files and unrelated services are preserved.

Candidate F retains E's budgets and native kernels and adds the batch transition repair and asynchronous boot exercise. Source SHA-256 on all four nodes: `e24ea2a1152984f4b11168bcc99c0ab7fb5a4815f9e94848107711284ef193c4`. The complete CPU suite ran 445 tests successfully, with 114 environment/GPU skips (`components/native-f-cpu.log`). An earlier test invocation lacked the bench/probe fixture directories; its six import/file errors are preserved separately and were resolved by supplying those directories, without changing the test gates.
