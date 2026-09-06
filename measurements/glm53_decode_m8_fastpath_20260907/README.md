# M8 exact fastpath and fixed-window follow-up

Status: complete. Kernel latency improved; serving shows a small positive
step-rate signal but remains **inconclusive**. Candidate retained, defaults off.

Four independent B/A/A/B boots on one deployed build produced 147 eligible
fixed-response windows: 74 baseline, 73 candidate. Each fixed request used
2,257 input tokens and exactly 2,048 output tokens, with matching request
hashes and seeds. The ordinary 2K/32K/128K quality ladder also ran per boot.

| Metric, median across two boots per arm | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Pooled interior-window step/s | 21.800487 | 21.872279 | +0.33% |
| Pooled fixed-response output tok/s | 70.517759 | 71.338854 | +1.16% |
| Fixed-response TPOT, ms | 14.194733 | 14.017761 | -1.25% |

Pooling is performed within each boot; the arm estimate is the median of
the two boot estimates (equal to their arithmetic mean at n=2). Per-request
medians are retained in [summary.json](summary.json) as a separate statistic.
The output-speed median-of-medians would read +4.04%; it is not the pooled
response-rate result reported above.

The forward and reverse pairs give **+0.60% / +0.06%** for engine step/s,
but **-1.58% / +4.09%** for pooled output tok/s. Baseline output speed varies
by **6.26%** across its boots, versus the average +1.16% change. Pooled step
spread is 0.23% for baseline and 0.30% for candidate; with only two boots per
arm these observed ranges are not confidence intervals. Same-setting
repeated boots match generated output hashes 0/3 in both arms, even though
their request hashes match 3/3. Acceptance varies too. The evidence supports
retaining a small step-rate improvement hypothesis, not claiming a stable
large C=1 speedup or rejecting the candidate.

All four boots pass **18/18 retrieval, 0/8 Korean corruption**, totaling
72/72 and 0/32. Every run has exactly its own eight completed requests and
no sampled concurrency or traffic issues. Both candidates prove all three
paths, including actual M=6 fastpath capture and both compact geometries.
The final server is healthy, all three flags are 0 and the fleet is released:
[final state](final-state.txt). The stale canonical fleet checkout labels the
already-enabled cache paths and SPEC_K alias as non-default; the served
three experiment flags above and the current harness's empty baseline
`knobs` are the authoritative result.

Reproduce with `python3 analyze.py`: it verifies source/settings, all gates,
archived boot/capture proof and paired inputs before writing the summary.
See [raw records](records.raw.jsonl), [completed resume log](serving-resume.log)
and the per-node runtime identity files for the source/model/image evidence.

The prior candidate remains retained and inconclusive. This follow-up adds
an opt-in reduction/scale-load path without changing its arithmetic result,
then increases the serving evidence beyond the prior 5/6 short-context windows.

## Candidate

- `VLLM_GLM53_MK_FP8_PACK2=1`
- `VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2`
- `VLLM_GLM53_MK_M8_FASTPATH=1`

The third flag replaces five shuffle/max pairs with a full-warp unsigned
maximum over nonnegative FP32 bit patterns. Each lane also reads just its
two FP4 scale bytes. The RQ=2/4 and low-rank paths retain their original code.
All three profile defaults remain 0. Implementation commit: `0421328`.

CPU validation: `tests/test_logic.py` passed 6,663 checks, including 30
megakernel and 20 fleet regressions. `tests/test_onepass_measurements.py`
passed nine contamination/window/timing fixtures. Both composed overlays
match their source, Python compilation and source/documentation
`git diff --check` passed. Archived `.log` files preserve the logger's
original carriage returns and trailing whitespace and are excluded from
that whitespace check.

## GPU gate and measurement protocol

`probes/mk_fp8_pack_bench.py --fastpath --reps 18` builds baseline, previous
M8 and new M8 from the same source. It checks FP8 conversion bytes, special
FP32/BF16 activation maxima, the independent GEMM oracle, SMLP2 and repeated
CUDA graphs. New M8 must be bit-identical to previous M8 on every tested
shape/replay. Twelve geometries have separate cold-weight and warm timings,
with all six execution orders balanced. The container is pinned to image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.

The serving [runner](runner.sh) uses four independent boots in B/A/A/B order
on one deployed build. Every arm runs the existing Korean 2K/32K/128K
onepass ladder plus three 2K-document responses with `min_tokens=max_tokens`
set to 2,048. Seeds, prompts, SPEC_K=5 and all other settings match.

Primary engine rate is total steps divided by total elapsed time in complete
fixed-response windows, with one-second edge margins. Report request tok/s
and TPOT alongside it. Each boot must have at least 20 eligible fixed windows.
The quality requirement becomes 18/18 retrieval checks and 0/8 corrupted
responses. Extra completed requests, observed concurrency/queueing, absent
counters, counter reset or incorrect output length invalidate the run.

The [analyzer](analyze.py) requires all four valid boots and candidate proof
3/3 before producing the comparison. It reports each boot, reversed pairs
and between-boot spread. Multiple windows within a boot are correlated and
are not counted as independent boot replicates. These fixed-output records
must not be pooled with the older variable-length workload.

## Queue issue observed before GPU execution

The canonical fleet accepted session `m8nextprobe` at 06:37 KST. At 06:38,
`prefillserve` yielded, but `legacy_busy()` continued counting its paused
`bash bench/chain.sh` parent as active benchmark traffic. Fleet was FREE,
serving healthy and idle, yet the probe could not acquire its turn. This is
a scheduler blockage before kernel execution, not a failed GPU test.
This task did not remove a hold or process. The preceding session was
cancelled outside this task; the canonical fleet granted this probe at 06:45:54
and released it at 06:47:54. The serving follow-up entered the queue at
06:49, behind the resumed prefill experiment.

## GPU result, first round

[Raw log](probe-round1.log) and [parsed records](probe-round1.json) preserve
all 18 samples per variant/regime and all gates. The source SHA-256 is
`b3b8c7f891b09792200c477b38b247ed9c720c0469b1b34957ac30f9dcde2cdd`;
the three builds use that identical source on GB10, Torch 2.13.0+cu130.
Both conversion and warp-amax gates passed 1,114,880 inputs with zero bit
mismatches. All 12 shapes passed the independent GEMM oracle and replay
checks; new M8 was bit-identical to previous M8. SMLP2 producer/consumer
graph checks passed 18 replays each for intermediate widths 2,048 and 3,072.

The [previous](previous.resources) and [new](fastpath.resources) binaries
retain 78 registers for compact M8 and 112 for ordinary M8, with zero local
memory/stack on those lanes. Runtime occupancy and split plans match.

| M=6 (N,K) | Warm baseline / previous / new, us | New vs previous | Cold baseline / previous / new, us | New vs previous |
|---|---:|---:|---:|---:|
| (6416,4096) | 41.640 / 40.641 / 38.176 | -6.06% | 77.024 / 77.552 / 76.368 | -1.53% |
| (4096,2048) | 14.068 / 13.110 / 12.609 | -3.82% | 31.536 / 31.200 / 30.576 | -2.00% |
| (1024,4096) | 11.107 / 10.789 / 10.330 | -4.26% | 22.096 / 21.472 / 20.896 | -2.68% |
| (4096,512) | 6.819 / 6.616 / 6.178 | -6.62% | 14.624 / 14.464 / 14.336 | -0.88% |
| (6144,4096) | 32.601 / 29.844 / 28.401 | -4.84% | 72.704 / 71.696 / 72.128 | +0.60% |

Negative deltas mean lower kernel latency. Warm gains are consistent in this
round; cold changes are smaller and one shape is 0.6% slower than previous
M8. These are single-GPU microbenchmarks and do not establish a serving gain.

## Serving provenance and storage recovery

Serving source is `b13dc8b`, deployed stamp `83bc6639a0a2`. Before the first
request, the host-only measurement code was finalized at `b319173`: stalled
interior windows count toward elapsed time, and a decreasing step counter
invalidates the record. Model overlays did not change. Every arm uses that
same finalized harness. Both median request speed and aggregate
`sum(completion_tokens-1)/sum(decode_seconds)` are reported; the primary
engine metric remains pooled interior-window steps/second.

`M8NEXTB1` finished at 07:11:20 KST: 36 fixed windows, **21.775064 step/s**,
request speed median **70.962593 tok/s**, quality **18/18**, Korean **0/8**.
Its eight completed requests exactly match the counter delta; no traffic
issues were observed. See [raw records](records.raw.jsonl) and the
[first chain log](serving.log).

The first A1 launch stopped before container teardown while staging the
chat template: srv1 had zero space available to the service account. B1
stayed healthy with all three candidate flags at 0; the failed A1 produced
no performance record. The cache identity includes all VLLM environment
flags, so this build's FP8 cache entries were new and the previous entries
were incompatible.

857 older generated FP8 cache files (8,590,997,853 bytes), all predating this
experiment's first boot, were archived to
`srv2:/home/choiceoh/glm53-logs/M8NEXT/cache-archive-srv1`. Each backup and
original was checked against SHA-256 before unlinking the original. The
manifest is [cache-archive-manifest.json](cache-archive-manifest.json), also beside that remote archive.
Current B1 cache entries were retained. This freed 6.3 GiB available on
srv1; [the archive helper](cache-archive.py), [audit](cache-removed.json) and
[space check](srv1-space-after.txt) are retained. The original files can be
restored from the archive; the archive is intentionally kept off Git.

[resume.sh](resume.sh) verifies the existing deployment and archive, then
resumes A1/A2/B2 without redeploying or repeating the valid B1. The resumed
chain ran from 07:20:03 to 07:44:30 KST. These are four independent boots in
B/A/A/B order, with a storage-recovery gap after B1. No invalid performance
record or contaminated arm was substituted into the completed comparison.
