# Phase 2 prefill and onepass result

GLM-5.3-Flash, 4×GB10 TP4, FP32 KDA, spec6, KV5.5. Scored requests used fresh salted prefixes, cached_tokens=0 and profiler OFF. Rates use each request's actual input tokens / client TTFT.

Measured engine: `3c7bcc0a08842d59a91e3f5166f5d5227a9e5cd6`; controller: `fac92ef980a5ba78861bc9c145c99fa0cf11e80c`; run: `20260913T065924-4fefee2be2e1`. This is the pre-merge run, not a measurement of later main integration, the tail scheduling fix or the revised preparation budget.

| C=1 request | Actual input | Prefill tok/s | TTFT s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 2K ledger | 2672 | 2413.70 | 1.1070 | 90.42 |
| 2K portfolio | 2632 | 2378.49 | 1.1066 | 72.23 |
| 2K logic | 2618 | 1740.65 | 1.5040 | 73.01 |
| 32K combined | 33817 | 3386.70 | 9.9852 | 87.04 |
| 128K combined | 129775 | 3305.71 | 39.2578 | 84.67 |

C1 context-window medians were about 19.91 steps/s. Fixed1024 completed exactly 1024 tokens at 69.66 / 86.09 / 76.89 tok/s, with a window median of 16.946 steps/s (pooled 15.934). Raw acceptance was 53.538%, or 4.2123 output tokens/step. The accepted window median remains null because quality failed.

| C=4 request group | Aggregate output tok/s | Per-request decode tok/s | TTFT range s | Group completion s | Quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K ledger | 151.23 | 40.58–44.06 | 1.12–7.07 | 48.02 | 1/4 |
| 2K portfolio | 121.71 | 32.46–35.92 | 1.15–6.99 | 174.06 | 1/4 |
| 2K logic | 110.45 | 30.43–37.12 | 1.58–7.32 | 232.20 | 1/4 |
| 32K combined | 113.69 | 31.86–47.76 | 10.07–69.42 | 461.91 | 9/12 |

C4 aggregate rate is sum(output tokens) / group wall time, including prefill. C4 TTFT includes queueing and interleaved decode; it is not the isolated C1 prefill rate. C4 128K was removed by the user, and its active request group was cancelled. The original partial checkpoint is retained without claiming completion.

C1 natural answers passed 3/9 certificates; fixed1024 passed 0/9 with all three answers truncated by the fixed budget. C4 passed 12/24. Korean corruption was 0/8 C1 and 0/16 C4. Natural errors include malformed JSON and wrong reasoning certificates; quality thresholds were not relaxed. The full campaign is incomplete: second C1 and diagnostic replays did not run. The 2K 3300 and 128K 4000 targets are not achieved, and this report makes no matched-baseline speedup or decode nonregression claim.

## Changes submitted

- Preserve sequence-parallel projection for arbitrary real prefill lengths by padding communication rows and cropping before attention/KDA/cache updates. Remove the obsolete whole-model tail split; ongoing decode keeps its interleave budget.
- Reuse two stream-ordered projection transport buffers; allow the measured FP8 transport path from 2048 rows. Use the existing MLA tile32 kernel from 128 rows and the SF6 word producer for 65–8192-row Q0 prefill.
- Retain committed host state in existing latency records for rank agreement diagnosis.
- Exclude C4 128K from preparation, scoring and diagnostics. Keep C1 2K/32K/128K and C4 2K/32K. Bound ungraded preparation to 64 output tokens at spec6 with full input prompts; measured quality and fixed1024 budgets and specialization checks remain intact.

Main already contains the native draft agreement and bounded gather repairs. The temporary raw-scale expansion prototype remains outside this change.

## Evidence and limits

`gpu-L7-qualification/` retains actual-weight Q0, short MLA, projection and TP4 bounded-walk gate logs, all-rank source verification, C1 timing/quality summary, and cancellation proof. GPU component checks and full serving boot passed for the measured source. No new engine run is claimed for the tail fix or integrated main. `L8-tail-cpu-proof.json` records the previously failing lengths and scheduler CPU checks.

The compressed `operator-stop.tar.gz` contains unmodified original workload, all 52 completed raw requests (including unscored preparation), quality records, log and the checkpoint at cancellation. SHA256: `7b55e97aeba0ddcda785533673c00114146a1e23c05cfe7246e931c716add669`.
