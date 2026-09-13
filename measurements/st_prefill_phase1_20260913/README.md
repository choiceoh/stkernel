# ST prefill phase 1 landing

The K campaign completed on 2026-09-13 using engine `2c2ee77f`, controller `de67dc0e`, GLM-5.3-Flash on 4 GB10 nodes, C=1, KV 2 GiB/rank, FP32 KDA state and six speculative tokens. Both scoring stages used frozen harness 40, zero reused prompt tokens and no profiler.

| Request | First request tok/s | Later same-boot tok/s | Later TTFT |
| --- | ---: | ---: | ---: |
| 32K (32,545 actual tokens) | 2695.852 | 3412.559 | 9.536831 s |
| 128K (128,559 actual tokens) | 3318.997 | 3360.426 | 38.256753 s |

See [full result](K/FULL_RESULTS.md), raw request records and output hashes in `K/1/` and `K/steady-unprofiled/`, four-rank source manifests, and the post-score diagnostic in `K/profile-only/`. First-pass quality was 18/18 and Korean corruption 0/8; later long requests were 6/6 and 0/2. Fixed 1024-token decode completed three times, pooled 49.951264 tok/s. Different output trajectories prevent a decode nonregression verdict. C=4 was not measured for this candidate; the later two long requests are not a second full C=1 pass. The user directed candidate-only testing, so no new baseline engine was run.

This PR ports the active prefill changes onto main `724b4ef6`: 32,256-token chunks with aligned tails, bounded native Top-512, long SF6 route/scale reuse, FP32 router accumulation, tiled calibration, reusable KDA/mHC metadata and final-hidden-row contraction. The new and modified kernel bodies retain the campaign implementation. Current main's BF16 collective agreement, small-row calibration buffering, execution plans, decode pointwise lanes, drafter storage and early observation are preserved. A pending small-row Gram tail is flushed before the new long-row observer.

The campaign's inactive project-before-gather experiment is omitted: actual FP8 calibration differed between ranks, so the measured build never enabled it. Campaign fleet backports and proposal-replication workaround are omitted; this landing retains current main's collective agreement implementation. Current main already has the fused SwiGLU and BF16 expert join.

**The merged source is not the frozen K runtime.** K numbers are historical campaign evidence, not a claim that this reconciled main has reproduced them. First 32K still missed 3300. Exact tie IDs from the new selector can differ from Torch; selected score sets, bounds, uniqueness, stream behavior and real consumer quality were checked. Native selection/Gram/router numerical gates and timings are retained under `selection-batch/`; earlier accepted component gates are under `J4-gates/`.

Phase 2 uses this landing as the starting source and targets C=1 **2K >= 3300 tok/s and 128K >= 4000 tok/s**, actual request tokens divided by TTFT. It retains zero prefix reuse, profiler-off scoring, quality/output records and fixed decode coverage. Against the K later results (~1920 at 2K, 3360 at 128K), this needs about 72% and 19% more throughput respectively. Those are required gains, not predicted gains.

## Reconciled-source CPU validation

ST image `st-engine:prefill3000-a890dac2`, CPU-only `runc` container with no GPU or network, 2 CPUs/6 GiB and one BLAS thread. The full engine suite ran 1,119 tests in 105 files: 103 files passed; the two failures were a readiness mock lacking the new snapshot/last-row contract and an existing supervisor assertion that a real-clock delay remained exactly five seconds. Both fixtures were corrected. The focused rerun of both files plus four onepass contract modules passed **78 tests**. Full-suite GPU-dependent skips: 206. See `landing-cpu-first.log` and `landing-cpu-fixed.log`. This is CPU integration evidence; no merged-runtime GPU result is claimed.
