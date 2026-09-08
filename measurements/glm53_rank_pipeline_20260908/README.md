# GLM rank checkpoint restore pipeline

Candidate `VLLM_GLM53_RANK_CACHE_PIPELINE=1` overlaps checked disk reads with CUDA copies, bounded to two 64 MiB pinned buffers. Default remains **0** pending the exact-image GPU gate and a matched warm B/A/A/B startup bracket.

The serial path retains mmap → SHA256 → pinned staging → copy → stream synchronization → page eviction. The candidate uses one reader thread to read directly into the next pinned slot, validates the exact buffer that will be copied, then enqueues the copy on the caller's CUDA stream. A per-slot event is completed before the reader can overwrite that slot. Every exit, including a late read/checksum failure, drains pending copies before releasing buffers. There is no source-loader fallback after partial restoration. The only fallback is before publication when pinned allocation fails or state is not on one CUDA device. Rank aliases, checkpoint identity, the all-rank readiness vote, and post-load hooks are unchanged.

The transport flag is excluded from rank/FP8 artifact environment identity; implementation file hashes still invalidate previous-source artifacts. PRIME creates the current-source caches and is excluded. Timed order: BASE1, FAST1, FAST2, BASE2, with only pipeline 0/1 changed. CPU-vote remains 0 in both arms. All source hashes, runtime image/environment, source mtimes, Ninja build records, rank/FP8/W4 hits, required model/MM profiling, real graph capture and kernel warmup, first text/image/video, Korean onepass and host memory are retained. The API binds to loopback during the bracket to exclude external clients; normal fleet supervision owns restoration or acknowledged handoff.

Timers separate context construction, manifest validation, readiness vote, restoration, and checkpoint recheck. Serial `mapped_hash_s` includes mmap page faults and is not a pure hash timer. Candidate `read_s` and `hash_s` run on its reader thread; `reader_wait_s` and copy waits can overlap them, so do not sum these as sequential phase costs. The decision uses complete health wall and matched model-load time, not a sum of nested timers.

## Reproduction

From a clean checkout based on current main on srv2:

```sh
export REPO=$PWD
export STARTUP_CACHE_EVIDENCE=/home/choiceoh/glm53-logs/rank-pipeline-TRIAL
bash bench/fleet.sh run --gpu SESSION 40 "rank restore B/A/A/B" -- bash bench/run_startup_rank_pipeline.sh
```

The runner stops previous serving only after the normal fleet idle/current-main guards, runs bounded CPU and exact-byte CUDA checks in the immutable serving image, deploys the canonical source, then runs the five-boot bracket. It refuses unmanaged restoration. No throughput or readiness saving is claimed until this trial completes.
