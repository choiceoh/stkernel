# Batched accepted-only FP32 KDA state (2026-09-13)

Status: candidate; production default remains ordinary FP32 rings. No new consumer speed verdict.

The target occupies 45.05 ms of a prior 48.55 ms bounded decode iteration. This candidate extends the existing single-row deferred-state experiment to C=1/C=4 and commits every KDA layer in one coalesced launch after acceptance. It does not change KDA state precision, gate arithmetic, or sampling.

- Verification records separate FP32 key/decay/update factors for every row and layer; the canonical ring is read-only until acceptance.
- Commit writes the accepted final state and crossed prefix boundaries. EOS/length clipping uses committed counts; finished rows do not write. The host sampling path materializes before returning to the runner.
- Factor storage is shared across serialized context-capacity graphs, keyed by rows and token width. For four captured widths 1..4 at seven positions, factors occupy about 55.78 MiB per rank.
- C=4, 34 layers, seven positions: ordinary state writes are 952 MiB. Away from a prefix boundary, final-state plus factor writes are 158.31 MiB. Materialization adds a 136 MiB initial-state read; these byte counts are not a latency claim.

Validation before GPU admission:

- ST image Python 3.12 / Torch 2.13.0+cu130: 142 tests, 129 passed, 13 CUDA skips. Includes device-chain ordering, host sampling with clipped counts, slot/layer ownership, boot policy, sampling and existing pipeline contracts.
- 19 SM121 Triton compile variants passed with no CUDA context. The compile harness now explicitly supplies RING_INDEX_STRIDE, which its older standalone AST invocation omitted.
- New GPU gate covers exact outputs and whole arena bytes, all accepted counts, row/slot changes, rollback, ring wrap, prefix boundaries, non-square tails and four conditional iterations before factor reuse. Timing compares the complete 34-layer verifier plus commit, under equal state and warm/evicted cache regimes.

The preceding PR849/PR848 short GPU ticket completed 2026-09-13: 12 of 14 tests passed, including MAX tails, actual-vocabulary selection and reusable buffers. Two toy serving graph tests failed at sampler warmup with CUDA invalid argument, before a consumer boot. This candidate refreshes the final attribution frontier instead of querying a possibly null graph handle obtained before the first node. CPU lazy-handle fixtures pass; the two failed GPU tests are included in this admission to judge that repair.

Pending: admitted component results, then a candidate-only fleet onepass at 32K/128K, C=1 twice and C=4 once, in one boot. Raw answer grades are observations; speed and acceptance are the requested performance metrics.

## GB10 result (17:58 KST, 2026-09-13)

The official single-GPU ticket `st-kda-batch0913v4` ran source `d02185a4` on Torch 2.13.0+cu130 / CUDA 13.0, passed all nine KDA and toy serving-graph tests with no skips or attribution errors, and completed in 135.73 seconds. The engine graph fix is identical to PR859; the probe is the unchanged canonical main version while GitHub's merge API returned 502. `gpu-v4.json` binds every source digest and keeps all samples.

Across 34 layers and seven verified positions, at three accepted positions away from a prefix boundary:

| Rows | Ordinary | Deferred plus commit | Change |
| --- | ---: | ---: | ---: |
| C=1 warm | 1.654 ms | 1.064 ms | -35.65% |
| C=1 64 MiB evicted | 1.671 ms | 1.088 ms | -34.90% |
| C=4 warm | 6.324 ms | 3.377 ms | -46.60% |
| C=4 64 MiB evicted | 6.342 ms | 3.395 ms | -46.47% |

All accepted lengths 1/3/7 and boundary/no-boundary cases favored deferred state. C=1 reductions range 23.33%-37.31%; C=4 settled cases range 34.81%-48.66%. The first C=4 warm case was substantially noisier (13.99 ms ordinary versus 5.82 ms deferred); retain it but do not use its 58.38% reduction as the headline. These are KDA component latencies, not full-model decode speed or an achieved 22-step result.

The earlier packet fusion also completed: original versus fused selection plus reuse was 127.39/115.60 us at C=1 warm, 151.46/152.90 us at C=1 evicted, 343.97/224.62 us at C=4 warm, and 379.42/335.86 us at C=4 evicted. The small C=1 evicted regression remains visible; this synthetic gather-copy measurement excludes the NIC.
