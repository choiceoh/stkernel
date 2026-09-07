# GLM prefill comparison after disk space recovery

**The matched bracket completed: quality passed, but no substantial prefill
gain was demonstrated.** Against the mean of two independent defaults boots,
32K throughput changed +0.74% and 128K +0.27%, both smaller than the observed
baseline variation. 8K was 1.75% slower. The original 40% target is not met,
and this result does not justify promoting the new options.

The user requested capacity recovery and resumption after the earlier disk
refusal. A fresh read found srv1 already had about **279 GiB free**, with two
rank-cache artifacts instead of five. This space was recovered outside this
run; this agent did not delete those files. All other nodes also had more
than the new 128 GiB disk reserve. `disk-before.json` preserves the observed
state. The numerical kernels remain those already checked on GPUs.

Fleet `spfrt30907` acquired the slot at **14:32:36 KST**. It deploys immutable
source `6f797df28c29e7e4cb606724e2416dbc1c5dcfcc` (main `0b6dc75` included)
on image `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
All 56 overlays and the manifest were verified across four nodes. The
manifest stamp is `0aca81454720`.

The fresh bracket is `SPFR30907B1` / `SPFR30907A` / `SPFR30907B2`, using
one dedicated ledger, the same requested 2K/4K/8K/32K/128K documents,
`KV_TOKENS=524288`, `MAX_LEN=262144`, block override 415, and exclusive
loopback port 18000. The all-node memory guard reserves 10 GiB while its
own client runs. This is reduced-capacity measurement, not full production
capacity acceptance. Startup time is separate from request TTFT.

The candidate enables `VLLM_GLM53_PREFILL_SP_FUSE_MHC=1`, AG threshold 2048,
and RS threshold 4096. Direct PyNCCL transport remains off following its
previous measured size cliffs. Defaults use no fusion and inherit the shared
4096 threshold for both collectives.

The corrected attestation was checked with synthetic baseline/candidate
fixtures and an altered-overlay rejection before this run. Actual post-leg
attestation matches source, image, every mounted overlay and manifest on all
four nodes, selected options, capacity, endpoint, head boot identity and
workload/quality/exclusivity gates. The original failed scripts and incidents
remain in `../glm53_prefill_retry_20260907/`.

## Matched request results

All three arms completed with **15/15 retrieval, 0/11 Korean corruption,
11 requests and no traffic issues each** (45/45, 0/33 in total). All post-leg
attestations passed. Request body SHA-256 values and actual prompt-token counts
match for all 11 requests, as do every node's image, full serve arguments,
manifest and mounted source hashes. All records have `cold_compile=false`.
The candidate's 3/3 proof includes the executed FP8 unpack/MHC consumer;
its absence of a large gain is not an unapplied-option result.

| Context | Baseline 1 TTFT | Candidate TTFT | Baseline 2 TTFT | Throughput change vs baseline mean | Baseline spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K | 1.938 s | 1.917 s | 1.963 s | +1.74% | 1.32% |
| 4K | 1.603 s | 1.550 s | 1.562 s | +2.09% | 2.60% |
| 8K | 2.966 s | 2.993 s | 2.915 s | -1.75% | 1.76% |
| 32K | 10.679 s | 10.654 s | 10.787 s | +0.74% | 1.01% |
| 128K | 41.228 s | 41.258 s | 41.508 s | +0.27% | 0.68% |

Throughput change is `mean(baseline TTFT) / candidate TTFT - 1`; prompt
counts match, so this also compares prompt tokens per first-content second.
Baseline spread is `max(TTFT) / min(TTFT) - 1`, a descriptive variation range,
not a confidence interval. This is two independent baseline boots and one
candidate boot. A small relative gain from the single candidate is not
strong evidence of a stable improvement. 128K changes sign depending on
which baseline is used (-0.07% versus +0.61%).

The short-context warm minima (with prefix caching enabled) were:

| Context | Baseline 1 | Candidate | Baseline 2 | Candidate throughput change vs mean |
| --- | ---: | ---: | ---: | ---: |
| 2K | 849.2 ms | 833.8 ms | 844.5 ms | +1.56% |
| 4K | 854.9 ms | 895.5 ms | 904.6 ms | -1.76% |
| 8K | 766.9 ms | 754.0 ms | 751.9 ms | +0.72% |

These are minima of two later requests, not independent cold-prefill samples.
All request TTFTs remain in `comparison.json` and `onepass.jsonl`; no minimum
is substituted for the first-request column. The first-content TTFT includes
request handling and first decode, not just the isolated prefill kernel.
No boot time or failed request time is included in these speed ratios.

The automated judge evaluates decode window step/s, which is a different
metric; its raw verdicts are retained separately. The prefill conclusion
above uses direct request timings. Neither that judge nor a per-kernel
microbenchmark provides evidence of a 40% prefill gain.

## Memory, disk and production recovery

All client memory observations passed. `memory-summary.json` preserves
minimum MemAvailable on every node for each arm; each head minimum exceeded
20 GiB. There were no cancelled requests, no memory-guard errors and no
workload-time worker termination in this bracket. This validates the
reduced-capacity workload; the earlier full-capacity 128K failure is not
resolved by this evidence.

Candidate startup generated a new cache artifact; srv1 still had 239 GiB
free during publication. A later read found 315 GiB free as external cleanup
continued. This agent did not delete those files. No disk admission or write
failure occurred. The candidate cache setup time was outside request timing.

The bracket completed at **14:58:36 KST**. Since no boot successor was queued,
the runner began restoring the public 8000 endpoint and full production
capacity on the existing profile defaults. Recovery completed at **15:03:13 KST**, and read-only verification confirmed
all four nodes on the pinned image and expected defaults, public
`0.0.0.0:8000`, max length 1,048,576 and block override 1,056. Both srv2 and
srv1 returned **health 200** from the public endpoint. The fleet runner
exited 0. The final srv1 disk read had **360.71 GiB free**, with the two
remaining rank-cache artifacts totaling about 90 GiB. The existing repeat
automation remains paused; this completed comparison is not rerun.

Raw runners, admission request, fleet log, onepass rows, per-arm node
attestations, memory traces, boot logs and automated verdicts are retained
here. Run `python3 analyze.py` to recheck comparability and regenerate the
per-request summary. Device code and its earlier numerical evidence are
unchanged; no additional GPU microprobe was run.
