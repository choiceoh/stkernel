# Native ST decode: target 22 step/s

Candidate branch: `codex/st-decode-22step`. Consumer result pending. No 22 step/s claim and no production promotion.

The candidate groups small calibration rows into 256-row Gram updates, fuses their masks/counts/peaks, and charges the staging buffers to the arena budget. Declared DenseLinear inputs stage their original BF16 values; generic observers retain FP32. It also binds native GLM clamped SwiGLU, target RMSNorm and indexer LayerNorm, fuses the router's surrounding arithmetic while retaining torch.topk, and removes redundant FP32 conversions from the BF16 expert join. CPU/reference lanes retain their original forms.

A read-only sample from the prior `main-ff728f43` boot recorded 34 stage observations: forward 51.79 ms, observe 15.84 ms, propose 16.85 ms. This is diagnostic context, not our baseline or a speedup result. That boot had 0 GPTQ / 231 RTN packs and self-calibration collecting 20 blobs. Its wide draft fc Hessian is 20480 squared FP32 entries (1.5625 GiB); the original small-row path updated it on every observation.

Validation so far:

- No-device Triton SM121 compile: 15 production/small-row/flush/pointwise specializations passed; no CUDA device initialized. Reproducer: `compile_gram.py` in the pinned ST runtime with no GPU attached.
- CPU runtime: 62 tests run, 24 passed, 38 GPU-only skips. Modules: dense calibration, buffered calibration, GLM53, dense smoothing, KDA norm.
- Fleet helper regressions cover pinned lease dependencies, exact local/SSH argument delivery, and returning a heartbeat PID without blocking before launching a probe.
- GPU numerical, changed-input graph replay and scoped B/A/A/B component measurements are in `probes/engine_kernel_check.py --lanes calibration,pointwise`.

Fleet receipts:

- `stdecode22gram3` was paused and cancelled before GPU execution after no-device compilation exposed an invalid Triton global constexpr reference.
- `stdecode22bundle1` reached GO but never launched a GPU container: the shell heartbeat kept the command-substitution pipe open. Its own supervisor was cancelled and its unused lease released. The helper now closes that pipe, with a regression test.
- `stdecode22bundle2` was rejected before admission because main gained gather-divergence guards. They were merged into the candidate.
- `stdecode22bundle3`: ticket `17892218341766139`, candidate `b0910bbe`, queued with a corrected pinned controller. Passed: eight numerical/graph tests and the scoped component comparisons. Full log and JSON are alongside this file.

The previous holder stalled in an NCCL collective. The user explicitly authorized preserving its logs and stopping it. All four logs and the lease receipt were saved under `/home/choiceoh/glm53-logs/st-decode22-prior-timeout`; the official launcher stopped all four nodes and released the lease. No other owner's reservation was cancelled.

Consumer comparison plan: baseline main including the same boot readiness and admission corrections, candidate, baseline again; two canonical onepass runs per boot. Before each pass reset the private engine's prefix cache. Use the same frozen pack/cache seed, TP4, native execution, spec K=6, KV=7 GiB, contexts 2K/32K/128K and seed=7. Both arms use the current canonical harness 42: 2400 tokens per individual certificate, 7200 per combined certificate, and three fixed 7200-token decode repetitions. The earlier 400/1024 budgets belonged to an older workload and were replaced before any consumer launch. C=1, C=4 and separate GPU diagnostics all run. Preserve quality, Korean corruption, request/output hashes, real output tok/s, window step/s and pooled fixed step/s separately; a failed quality/evidence gate closes adoption even when a raw rate exists.

`run_onepass.py` invokes the unmodified `bench/onepass.py` CLI between all-rank ST identity receipts. The canonical script's old overlay labels are not ST identity; the sidecars record actual image IDs, native source hashes, boot IDs, commands, mounts and lease owners before and after. It does not replace the workload, sampler or counter implementation.

GPU result (b0910bbe): 7-row wide calibration observer B=15.3298/15.3226 ms, A=1.5221/1.5274 ms, including partial flush. Shared activation approximately 18.5→4.0 us; router post-projection 30.8→20.5 us; RMSNorm 20.6→4.1 us; indexer norm 10.3→4.1 us; expert join 12.4→5.0 us. These are single-GPU captured components, not consumer speed. The first 1-row candidate interval includes first-use flush compilation and is not a steady-state estimate.

Additional pre-onepass improvements (after the user requested more work):

- Prepare the 42 FP32 router matrices once in the arena; rank-file binding remains BF16. The extra 189 MiB per rank is included in arena admission and the budget table. The projection has exactly the old converted values.
- Fuse all five drafter context layers' RMSNorm, rotary and committed KV writes. Ten launches become one, with no normalized-key staging planes. Slots retain their arena stride; rejected tokens and ghost rows do not write.
- CPU suite: 21 tests, 11 passed and 10 CUDA-only skipped. SM121 no-device compilation: three new specializations passed. The FP32 interpreter indexing/mask check passed. The BF16 interpreter attempt produced NaNs; it is not BF16 proof. Exact BF16 differential and changed-input graph replay are required in the new `residency` GPU lane before any consumer run.
- PR #760: tool source fixed at 2f4d7063, its 13 tests passed; base/candidate CPU simulation B/A/A/B gave 0.071/0.073/0.071/0.077 ms decode host medians. This is macOS with a null device, not serving speed. `pr760/` holds the receipts and a replay of historical onepass H (16.94 window step/s, quality 6/9, no adoption case).
- Current consumer sources are baseline `019425f8` (engine/launchers identical to main `21cb0539`) and candidate `ccbea87c`. Both include PR #762 admission, PR #768 boot rendezvous and PR #751 common KDA source. Earlier 93cdcaf9/1eb3a2c5/f5568254 brackets never launched a model. The canonical harness is identical across the current arms.
- The old B1 waiter was cancelled before it launched a model. `run_bracket.py` now requires explicit baseline/candidate source identities and respects the existing prefill priority deadline across retries.

BF16 calibration follow-up: DenseLinear now exposes its already-enforced BF16 input contract. Those observers stage the exact BF16 values (half the staging bytes) and use BF16 tensor-core products with FP32 accumulation. Generic observers retain FP32 staging and tf32x3. A BF16 observer rejects FP32 inputs or non-boolean masks before losing precision. Updated CPU calibration suite: 17 tests, 12 passed and five GPU-only skips. SM121 no-device compilation: 19 variants passed; BF16 Gram shared memory is 4 KiB versus 16 KiB for the FP32 specialization. GPU accuracy/timing remains pending. Force-flush compilation is now warmed while disarmed before all component timings.

Queue/controller follow-through: `stdecode22bundle5` retained its ticket through edit and resume, but its older pinned canonical probe runner could not accept the new runner introduced by fleet rules 2. It was cancelled before GPU execution and replaced through the current official controller with `stdecode22bundle6`, ticket `17892266482454447`, source `72013bcc`, explicit `--fleet` because these checks require GB10 SM121. The separate RTX 5050 lane cannot establish GB10 correctness. The existing prefill campaign released its priority window before this submission. Post-merge CPU readiness/residency/drafter/budget suite: 29 tests, 17 passed, 12 GPU-only skips, including real two-process Gloo rendezvous fixtures.

Expanded GPU result: `stdecode22bundle6` passed exact draft-write and router checks, then the BF16 calibration oracle rejected an FP32-versus-BF16 peak dtype mismatch. The saved peaks are contractually FP32; correcting the expected peaks to FP32 retained zero numeric tolerance and the dtype check. `stdecode22bundle7` was refused before any GPU hold because main advanced. After merging the required shared sources into both arms, `stdecode22bundle8` (ticket `17892270522509140`, source `ccbea87c`, runtime `st-engine:main-ff728f43`) passed all 12 numerical/graph tests and all component comparisons in 32.0 seconds. No engine math was weakened to pass the oracle.

| Captured component | B then B (ms) | A then A (ms) |
|---|---:|---:|
| BF16 wide Gram observer, 1 row, partial flush included | 15.3102 / 15.3023 | 0.1025 / 0.1055 |
| BF16 wide Gram observer, 7 rows, partial flush included | 15.4532 / 15.4056 | 0.6693 / 0.6705 |
| All-layer draft KV write, 7 rows | 0.02058 / 0.02050 | 0.00517 / 0.00514 |
| Resident router projection, 7 rows | 0.04732 / 0.04718 | 0.03494 / 0.03486 |

The actual order was B/A/A/B. The Gram fixture's Hessian, row count and peaks also matched the baseline after both timed arms. Logs and parsed checks are `gpu-kernels-ccbea87c.log/json`. These remain single-GPU component results; the B/A/B consumer is the adoption gate. Its driver started on srv2 at 00:32 KST, PID 2517966, with artifacts under `/home/choiceoh/glm53-logs/st-decode22-consumer-v3`.

PR #760 follow-through: found and fixed label mixing in `step_peek` histogram buckets; main independently gained that fix in PR #751. The candidate additionally requires matching bucket boundaries before computing quantiles, so a changed schema cannot invent a p50/p95. The merged 16-test tool suite passes. These fixes concern diagnostic correctness and are not decode speedups.
