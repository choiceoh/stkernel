# GLM CPU renderer warmup overlap

Runtime source: `c001cb9e335db488390125347b133b80f0e60de2`; benchmark source: `c001cb9e335db488390125347b133b80f0e60de2`.

PRIME refreshes artifacts and compilation. Timed order is BASE1, FAST1, FAST2, BASE2 with the same code/image/profile. Only EARLY_MM_WARMUP changes (BASE=0, FAST=1). W4 SHA256/fast IO and warm rank/FP8 caches remain enabled. All boots use PREFILL_WARMUP=0 and the canonical Korean onepass workload at 2K/32K.

| Arm | Health s | Head model s | Profile s | Both MM warmups s | Early complete / reuse | Quality / corruption |
|---|---:|---:|---:|---:|---|---|
| EARLYMMPRIME | 314 | 135.3 | 90.9 | 9.784 | [9.863] / [0.0, 0.0] | 6/6; 0/4 |
| EARLYMMBASE1 | 218 | 78.3 | 36.4 | 6.901 | [] / [] | 6/6; 0/4 |
| EARLYMMFAST1 | 213 | 78.9 | 36.6 | 9.900 | [9.902] / [0.0, 0.0] | 6/6; 0/4 |
| EARLYMMFAST2 | 211 | 79.6 | 36.5 | 8.185 | [8.254] / [0.0, 0.0] | 6/6; 0/4 |
| EARLYMMBASE2 | 221 | 80.3 | 36.5 | 9.685 | [] / [] | 6/6; 0/4 |

Two warm samples per arm; PRIME excluded:

| Metric | BASE mean [range] | FAST mean [range] | FAST minus BASE |
|---|---:|---:|---:|
| health_s | 219.500 [218.000, 221.000] | 212.000 [211.000, 213.000] | -7.500 |
| model_s | 79.300 [78.300, 80.300] | 79.250 [78.900, 79.600] | -0.050 |
| profile_s | 36.450 [36.400, 36.500] | 36.550 [36.500, 36.600] | +0.100 |
| mm_warmup_s | 8.293 [6.901, 9.685] | 9.043 [8.185, 9.900] | +0.750 |

All-rank cache receipts:

| Arm | Node | Rank artifact | FP8 hit/miss/error | W4 SHA/fast/legacy |
|---|---|---|---|---|
| EARLYMMPRIME | srv1 | hit 53.239s | 244/0/0 | 255/255/0 |
| EARLYMMPRIME | srv2 | hit 46.624s | 244/0/0 | 255/255/0 |
| EARLYMMPRIME | srv3 | hit 47.884s | 244/0/0 | 255/255/0 |
| EARLYMMPRIME | srv4 | hit 48.102s | 244/0/0 | 255/255/0 |
| EARLYMMBASE1 | srv1 | hit 53.486s | 244/0/0 | 255/255/0 |
| EARLYMMBASE1 | srv2 | hit 48.304s | 244/0/0 | 255/255/0 |
| EARLYMMBASE1 | srv3 | hit 48.115s | 244/0/0 | 255/255/0 |
| EARLYMMBASE1 | srv4 | hit 47.716s | 244/0/0 | 255/255/0 |
| EARLYMMFAST1 | srv1 | hit 53.337s | 244/0/0 | 255/255/0 |
| EARLYMMFAST1 | srv2 | hit 46.448s | 244/0/0 | 255/255/0 |
| EARLYMMFAST1 | srv3 | hit 49.548s | 244/0/0 | 255/255/0 |
| EARLYMMFAST1 | srv4 | hit 47.975s | 244/0/0 | 255/255/0 |
| EARLYMMFAST2 | srv1 | hit 53.256s | 244/0/0 | 255/255/0 |
| EARLYMMFAST2 | srv2 | hit 45.303s | 244/0/0 | 255/255/0 |
| EARLYMMFAST2 | srv3 | hit 50.830s | 244/0/0 | 255/255/0 |
| EARLYMMFAST2 | srv4 | hit 48.219s | 244/0/0 | 255/255/0 |
| EARLYMMBASE2 | srv1 | hit 54.244s | 244/0/0 | 255/255/0 |
| EARLYMMBASE2 | srv2 | hit 47.142s | 244/0/0 | 255/255/0 |
| EARLYMMBASE2 | srv3 | hit 49.435s | 244/0/0 | 255/255/0 |
| EARLYMMBASE2 | srv4 | hit 49.005s | 244/0/0 | 255/255/0 |

Generated response matches against BASE1 (separate from exact CPU preprocessing checks):

- EARLYMMPRIME: `{'baseline': 'EARLYMMBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 0}`
- EARLYMMBASE1: `{'baseline': 'EARLYMMBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 4}`
- EARLYMMFAST1: `{'baseline': 'EARLYMMBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 0}`
- EARLYMMFAST2: `{'baseline': 'EARLYMMBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 0}`
- EARLYMMBASE2: `{'baseline': 'EARLYMMBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 0}`

Serving POST completions in the preserved logs:

- EARLYMMPRIME: `{'loopback': 4, 'non_loopback': 6, 'completed_before_first_health': 0}`
- EARLYMMBASE1: `{'loopback': 4, 'non_loopback': 1, 'completed_before_first_health': 0}`
- EARLYMMFAST1: `{'loopback': 4, 'non_loopback': 10, 'completed_before_first_health': 0}`
- EARLYMMFAST2: `{'loopback': 4, 'non_loopback': 2, 'completed_before_first_health': 0}`
- EARLYMMBASE2: `{'loopback': 4, 'non_loopback': 0, 'completed_before_first_health': 0}`

Non-loopback serving traffic overlapped the quality workload. Onepass throughput, TTFT and acceptance counters are therefore not a matched performance comparison. The startup endpoint is the first HTTP health 200 on a new container; logs place that health response before the recorded POST completions.


Host memory and disk sampled every 10 seconds; these are OS samples, not CUDA peak allocations:

| Node | Samples | Minimum available RAM GiB | Minimum disk GiB | Net swap growth MiB |
|---|---:|---:|---:|---:|
| srv1 | 164 | 13.88 | 321.48 | 0.0 |
| srv2 | 164 | 8.46 | 902.98 | 0.0 |
| srv3 | 164 | 9.20 | 1735.98 | 0.0 |
| srv4 | 164 | 11.00 | 2082.14 | 0.0 |

Trial exit: `0` (null means ongoing).

Raw logs, actual environment snapshots, exact response files and source hashes are retained beside this report. Two warm samples per arm do not establish broad throughput or full-context quality. Warmup still runs both stock processors and clears their caches; the candidate moves those CPU operations before engine readiness.
