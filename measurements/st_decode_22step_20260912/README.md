# Native ST decode: target 22 step/s

Candidate branch: `codex/st-decode-22step`. Consumer result pending. No 22 step/s claim and no production promotion.

The candidate groups small calibration rows into 256-row Gram updates, fuses their masks/counts/peaks, and charges the FP32 staging buffers to the arena budget. It also binds native GLM clamped SwiGLU, target RMSNorm and indexer LayerNorm, fuses the router's surrounding arithmetic while retaining torch.topk, and removes redundant FP32 conversions from the BF16 expert join. CPU/reference lanes retain their original forms.

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

Consumer comparison plan: baseline main including the same gather guards, candidate, baseline again; two canonical onepass runs per boot. Before each pass reset the private engine's prefix cache. Use the same frozen pack/cache seed, TP4, native execution, spec K=6, KV=7 GiB, contexts 2K/32K/128K, seed=7, and three fixed 1024-token decode repetitions. Preserve quality, Korean corruption, request/output hashes, real output tok/s, window step/s and pooled fixed step/s separately.

`run_onepass.py` invokes the unmodified `bench/onepass.py` CLI between all-rank ST identity receipts. The canonical script's old overlay labels are not ST identity; the sidecars record actual image IDs, native source hashes, boot IDs, commands, mounts and lease owners before and after. It does not replace the workload, sampler or counter implementation.

GPU result (b0910bbe): 7-row wide calibration observer B=15.3298/15.3226 ms, A=1.5221/1.5274 ms, including partial flush. Shared activation approximately 18.5→4.0 us; router post-projection 30.8→20.5 us; RMSNorm 20.6→4.1 us; indexer norm 10.3→4.1 us; expert join 12.4→5.0 us. These are single-GPU captured components, not consumer speed. The first 1-row candidate interval includes first-use flush compilation and is not a steady-state estimate.

Additional pre-onepass improvements (after the user requested more work):

- Prepare the 42 FP32 router matrices once in the arena; rank-file binding remains BF16. The extra 189 MiB per rank is included in arena admission and the budget table. The projection has exactly the old converted values.
- Fuse all five drafter context layers' RMSNorm, rotary and committed KV writes. Ten launches become one, with no normalized-key staging planes. Slots retain their arena stride; rejected tokens and ghost rows do not write.
- CPU suite: 21 tests, 11 passed and 10 CUDA-only skipped. SM121 no-device compilation: three new specializations passed. The FP32 interpreter indexing/mask check passed. The BF16 interpreter attempt produced NaNs; it is not BF16 proof. Exact BF16 differential and changed-input graph replay are required in the new `residency` GPU lane before any consumer run.
- PR #760: tool source fixed at 2f4d7063, its 13 tests passed; base/candidate CPU simulation B/A/A/B gave 0.071/0.073/0.071/0.077 ms decode host medians. This is macOS with a null device, not serving speed. `pr760/` holds the receipts and a replay of historical onepass H (16.94 window step/s, quality 6/9, no adoption case).
- Baseline 1eb3a2c5 is main 93cdcaf9 plus the common admission correction from PR #762. Both consumer arms receive that correction. The workload remains the identical frozen canonical onepass at 93cdcaf9; later onepass changes are not silently mixed into this bracket.
- The old B1 waiter was cancelled before it launched a model. `run_bracket.py` now requires explicit baseline/candidate source identities and respects the existing prefill priority deadline across retries.
