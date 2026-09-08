# GLM rank checkpoint restore pipeline — no startup improvement

`VLLM_GLM53_RANK_CACHE_PIPELINE` remains **0**. In the four-node fleet bracket `rankpipe0908v3`, the candidate increased warm startup from **217.5 to 236 seconds on average: 18.5 seconds / 8.5% slower**. The exact-byte CUDA and output checks passed, but the performance gate failed. This experiment does not establish a faster boot.

| Warm arm | BASE1 | FAST1 | FAST2 | BASE2 |
|---|---:|---:|---:|---:|
| Pipeline | 0 | 1 | 1 | 0 |
| Health wall, seconds | 216 | 238 | 234 | 219 |
| Head rank restore, seconds | 44.560 | 66.425 | 64.756 | 44.749 |

Measured source is `82a97dbf5d05286ca0b9670d38463ad84953988d`. The 407-second source/cache-creation PRIME is excluded. Health wall covers the container restart through its first HTTP health response with one-second polling; it excludes the official overlay deployment before PRIME. Timed arms use populated artifact and compiler caches, with no intervening deployment or source/Ninja changes. This is two boots per policy, not a general estimate across hardware or a cold-install result.

The candidate eliminated the extra host staging copy and reduced head GPU-copy waits from about **1.25 to 0.06 seconds**. However, direct pinned-buffer reads took **41–43 seconds**, followed by **22 seconds** of SHA256 on the same reader thread. The baseline's combined mmap page-fault/read/hash time was about **41 seconds**. Reading and hashing are still sequential within the candidate reader, and their added wall time outweighed the saved copy waits. This measurement rejects this particular direct-read/CUDA-overlap implementation; it does not prove that every prefetch or read/hash scheduling strategy is exhausted. SHA256 validation remains mandatory.

The candidate overlaps checked disk reads with CUDA copies, bounded to two 64 MiB pinned buffers.

The serial path retains mmap → SHA256 → pinned staging → copy → stream synchronization → page eviction. The candidate uses one reader thread to read directly into the next pinned slot, validates the exact buffer that will be copied, then enqueues the copy on the caller's CUDA stream. A per-slot event is completed before the reader can overwrite that slot. Every exit, including a late read/checksum failure, drains pending copies before releasing buffers. There is no source-loader fallback after partial restoration. The only fallback is before publication when pinned allocation fails or state is not on one CUDA device. Rank aliases, checkpoint identity, the all-rank readiness vote, and post-load hooks are unchanged.

The transport flag is excluded from rank/FP8 artifact environment identity; implementation file hashes still invalidate previous-source artifacts. PRIME creates the current-source caches and is excluded. Timed order: BASE1, FAST1, FAST2, BASE2, with only pipeline 0/1 changed. CPU-vote remains 0 in both arms. All source hashes, runtime image/environment, source mtimes, Ninja build records, rank/FP8/W4 hits, required model/MM profiling, real graph capture and kernel warmup, first text/image/video, Korean onepass and host memory are retained. The API binds to loopback during the bracket to exclude external clients; normal fleet supervision owns restoration or acknowledged handoff.

Timers separate context construction, manifest validation, readiness vote, restoration, and checkpoint recheck. Serial `mapped_hash_s` includes mmap page faults and is not a pure hash timer. Candidate `read_s` and `hash_s` run on its reader thread; `reader_wait_s` and copy waits can overlap them, so do not sum these as sequential phase costs. The decision uses complete health wall and matched model-load time, not a sum of nested timers.

## Reproduction

From a clean checkout based on current main on srv2:

```sh
export REPO=$PWD
export HEAD_URL=http://127.0.0.1:8000
export STARTUP_CACHE_EVIDENCE=/home/choiceoh/glm53-logs/rank-pipeline-TRIAL
bash bench/fleet.sh run --gpu SESSION 40 "rank restore B/A/A/B" -- bash bench/run_startup_rank_pipeline.sh
```

The runner stops previous serving only after the normal fleet idle/current-main guards, runs bounded CPU and exact-byte CUDA checks in the immutable serving image, deploys the canonical source, then runs the five-boot bracket. It refuses unmanaged restoration. No throughput or readiness saving is claimed.


## Correctness and resource evidence

All five boots passed first text/image/video checks and manual response review, quality **6/6** and Korean corruption **0/4**. Each timed boot had four rank hits, **976 FP8 hits / zero misses/errors**, and **258 cumulative W4 SHA256 hits per rank**, without source fallback, repacking or alias errors. Every head log has exactly seven loopback POST completions, zero external POST completions and zero before health. Required encoder/model profiling, graph-memory profiling, real graph capture and kernel warmup ran in every arm.

The immutable serving image was `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`, Torch 2.13.0+cu130, CUDA 13.0, NVIDIA GB10. The device-free CPU gate passed **37 tests**, no skips and no CUDA initialization. The actual CUDA gate compared **268,435,517 bytes**, including full chunks, a tail, BF16, scalar/empty tensors, offset views and aliases in policy order 0/1/0/1. Delayed real copies stressed slot reuse; all four runs were exact and drained before return. A later corrupt chunk was rejected before publication while previous copies completed. Its synthetic delay timings are not startup speed measurements.

Each node has **167** host-memory samples, no collection errors, and maximum sample gap 12 seconds. Available host RAM never fell below **7.51 GiB**; peak swap usage did not increase. The candidate allocates 128 MiB of pinned staging versus 64 MiB for serial restoration. These are sampled host-memory observations, not exact peak CUDA allocations.

Before the fleet run, local validation passed 37 cache/pipeline tests, 71,059 core checks, 38 megakernel regressions and 115 fleet tests; the fleet suite was rerun after main #471 and passed 120 tests. Boot receipt tests passed 11 cases. The verifier reconstructs runtime identity from the measured Git revision, checks all four nodes' canonical sources and unchanged inode/mtime/Ninja receipts, then checks cache modes, output and memory records.

## Excluded attempts and completion

The first queued attempt was canceled before admission when main advanced. The second attempt stopped at the idle-entry guard before any candidate deployment or GPU gate because the predecessor's metrics lacked the running-request field. Its supervisor restored approved production at 13:43:20 KST before the recorded v3 trial began. Neither attempt is in the performance comparison.

The five-boot v3 payload and `run_startup_rank_pipeline.sh` completed with exit 0 at **14:14:59 KST**. Its subsequent supervisor handoff was not accepted: the default public health URL could not observe the loopback-only experiment server. A separate normal fleet recovery session, `rankpiperecover0908`, was admitted at **14:18:04 KST** with `HEAD_URL=http://127.0.0.1:8000` and restored the approved production revision `926239e915dda5906d21b447f08245a71ca6945c`. No reservation or idle guard was bypassed. The final harness now requires that URL to be exported on the outer fleet invocation, before any service changes, so the supervisor can see the experimental server during recovery. This admission-only fix was added after measurement and does not alter measured runtime bytes.

The initial fleet supervisor ultimately exited **1** after its five-minute reclaim timeout; this is distinct from the successful timed payload. The recovery supervisor exited **0**, public health responses are preserved in `recovery-head.log`, and its handoff was accepted at **14:23:30 KST**. The recovery boot is not in the performance comparison. No GPU requests were made after that handoff.

## Integration and evidence bundle

Main `306511a` was integrated after measurement. The rank-cache and startup-cache implementation bytes remain identical to measured `82a97db`; upstream scheduling/attention changes are not part of this comparison. The integrated CPU gate passed **71,087 core checks, 38 megakernel regressions and 120 fleet regressions**. The outer health-URL admission was checked to reject a missing URL before holder inspection or any service action. Composition, syntax and whitespace checks passed. The candidate remains opt-in and the PR remains a draft experiment because it failed the performance gate.

`raw-evidence.tar.gz` preserves the captured boot logs, before/after source and Ninja receipts, source/runtime identities, all response transcripts, host samples, exact-image CPU/CUDA gates, driver logs, execution exits and the filtered fleet lifecycle. `raw-file-sha256.json` identifies each archived file. Readable [timings](report.md), [verification](validation.json), [full report](report.json), first requests and onepass records are retained beside it. To reconstruct and verify without contacting a GPU or service:

```sh
mkdir -p /tmp/glm53-rank-pipeline-evidence
tar -xzf measurements/glm53_rank_pipeline_20260908/raw-evidence.tar.gz -C /tmp/glm53-rank-pipeline-evidence
python3 measurements/glm53_rank_pipeline_20260908/report.py /tmp/glm53-rank-pipeline-evidence . --verify
```
