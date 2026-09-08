# GLM53 CPU renderer warmup overlap

Matched warm restarts improved from **219.5 s to 212.0 s** on the four GB10
nodes: **7.5 s / 3.4%**. Both candidate boots were faster than both controls.
The GLM profile enables `VLLM_GLM53_EARLY_MM_WARMUP=1`; `=0` restores stock
warmup scheduling. This is a two-sample-per-arm startup result, not a throughput
or full-context quality claim.

PR #452 was merged first as `bb123cfa1fc47e2b9e87fde3b55eb4400412785d`.
The follow-up runtime and benchmark source is
`c001cb9e335db488390125347b133b80f0e60de2`, based on main `4b0f1d1`.
Later documentation/default changes preserve the tested runtime Python bytes.

## What changes

The normal and read-only CPU multimodal processors still perform their stock
image/video warmups sequentially and clear their caches. The existing renderer
MM executor starts this work after input/output processor construction and
before the spawned engine starts. Normal renderer warmup joins it before chat
template processing and HTTP readiness; successful work is consumed once by
processor identity, and failures retry through the original method. Shutdown
also joins before the processor cache closes.

The placement after `InputProcessor` matters: its `MultiModalBudget` also changes
process-wide Torch thread settings. Starting before it can overlap those guards.
The selected placement is exercised by a regression test using the actual
AsyncLLM initialization statements. The following EngineCore client/launch
code was also inspected for another parent-side thread guard.

Guards restrict this optimization to GLM5-next, a single API process, spawn,
CPU multimodal processing, no GPU video backend, no GPU IPC cache and no device
normalization. ChatParams and the chat-template warmup retain their original
position. The flag is excluded from backend artifact identity because it only
changes frontend scheduling. New renderer files are pinned in the overlay
manifest; DSV4 is unchanged.

## Matched fleet result

`earlymm0908e` ran PRIME, BASE1, FAST1, FAST2, BASE2. All timed boots use the same
runtime, image, profile and warm rank/FP8/W4/compile artifacts; only the renderer
flag changes. Health wall starts before the launcher and ends at the first
HTTP 200 from a new container, sampled once per second. `PREFILL_WARMUP=0` is
constant; the canonical Korean onepass workload runs at 2K/32K after readiness.

| Run | Early warmup | Health s | Head model s | Memory profile s | Quality | Corrupt replies |
|---|---:|---:|---:|---:|---:|---:|
| PRIME, excluded | 1 | 314 | 135.3 | 90.9 | 6/6 | 0/4 |
| BASE1 | 0 | 218 | 78.3 | 36.4 | 6/6 | 0/4 |
| FAST1 | 1 | 213 | 78.9 | 36.6 | 6/6 | 0/4 |
| FAST2 | 1 | 211 | 79.6 | 36.5 | 6/6 | 0/4 |
| BASE2 | 0 | 221 | 80.3 | 36.5 | 6/6 | 0/4 |

Head model loading is essentially unchanged (79.30 -> 79.25 s), as is memory
profiling (36.45 -> 36.55 s). Controls run 6.901/9.685 s of CPU MM warmup after
engine initialization. Candidates perform the same operations earlier, complete
before model loading finishes, and reuse both results with `join_s=0.000`.
PRIME refreshes compilation and is excluded from the four-boot comparison.

Every timed boot has four rank-cache hits, 976 FP8 hits with zero misses/errors,
and 255 cached SHA256 W4 packs per rank with zero repacks/fallbacks/alias errors.
All node source/image hashes match. Final control had early warmup off, HTTP 200
and successful quality/state snapshots. The reservation was released at
02:41:53 KST; the next fleet owner acquired it at 02:41:58. The later 02:43:27
read-only check in `final-control.txt` observes that owner's stopped container,
not a failure of this trial. The new profile default takes effect on the next
deployment; no changes were made to the next owner's service.

Non-loopback API traffic overlapped the quality workload. Its counts and the
exact generated-response comparisons are retained in [the report](report.md).
Throughput, TTFT and acceptance counters are not a matched serving comparison.
The logged first health response precedes the logged POST completions in every
arm. OS RAM/disk samples and swap deltas are reported separately from CUDA peak
memory; no claim of identical generated answers or broad quality equivalence
is made.

## Validation and reproduction

- Pinned-image real processor check: 6 assertions, 20 CPU tensor fields. Stock
  and early paths have identical image/video dummy preprocessing and identical
  normal/read-only image chat inputs, including tokens, tensor bytes, dtype,
  shape, stride and tensor-valued slice indices. Thread settings are restored.
- Focused tests: 9 renderer lifecycle/guard/order tests and 4 receipt tests.
- Full local Torch-enabled gate: 71,014 logic checks, 30 megakernel regressions
  and 107 fleet regressions. See [validation.json](validation.json).
- Official composition/deployment verifies 58 overlays on all four nodes.

The recorded [fleet driver](fleet-driver.sh) performs the real processor check,
deployment, resource sampling and `bench/startup_cache_boots.sh` with
`STARTUP_CACHE_MODE=renderer-warmup`. Run it through the normal reservation from
the matching checkout, with `REPO=$PWD FLEET_AUTO_RESTORE=1 bash bench/fleet.sh
run --gpu SESSION 40 "renderer warmup startup bracket" -- bash DRIVER`.

Raw logs remain at `/home/choiceoh/glm53-logs/early-mm-20260908-v2` on srv2 and
`runs/early-mm-20260908-v2` in the task worktree. Rebuild the report with
`python renderer-report.py RAW_DIR`, then verify/copy receipts with
`python collect_receipts.py RAW_DIR REPO_DIR`. The adjacent file hash manifest
identifies the preserved raw logs, environment/state snapshots and responses.

Two initial probe attempts exposed a validation serializer omission for tensors
inside multimodal slice indices; it was fixed before any timing comparison.
The earlier pre-budget placement was stopped during PRIME after the thread-guard
review, and its incomplete logs remain separately under
`early-mm-20260908` on srv2 / `runs/early-mm-20260908-aborted` locally. It provides
no performance evidence. Signal handling now retains a failing exit code and
restores control after the foreground boot returns.
