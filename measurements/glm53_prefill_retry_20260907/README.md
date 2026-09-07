# Scheduled GLM prefill retry, 2026-09-07

The reduced-capacity defaults workload passed through 128K, but **no
candidate/baseline comparison completed**. The first bracket stopped in its
post-leg attestation helper. The corrected bracket was refused before deploy
because srv1 had only **18.9 GiB** available disk, below the 32 GiB test floor.
The original 40% prefill target remains unproven; PR #439 stays draft and
all new options retain their existing defaults.

## Source and workload

- Source `6f797df28c29e7e4cb606724e2416dbc1c5dcfcc`, including main `0b6dc75`.
- Immutable image `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
- Actual deployed manifest stamp `0aca81454720`; 56 source files were deployed
  and verified on all four nodes. Device code is unchanged from the previously
  numerically tested helper.
- Identical planned arms: `KV_TOKENS=524288`, `MAX_LEN=262144`, block override
  415, private loopback port 18000, prefix caching on, warmup off, exclusive
  onepass. These settings differ from production capacity and cannot establish
  production-capacity acceptance.
- The opt-in client memory guard samples all nodes, cancels only its client
  below 10 GiB or on missing observations, and never changes earlyoom.

## Completed baseline, not a speedup result

Fleet `spfrt0907` admitted at 13:56:11; `SPFRT0907B1` booted at 13:57:10.
The onepass workload ran 14:08:17–14:10:44. It completed all 11 requests,
retrieval **15/15**, and Korean corruption **0/11**. Traffic counters show
11 completed requests and none remaining, with no exclusivity issues.

| Requested context | First-request TTFT (s) | Recorded warm TTFT (s) |
| --- | ---: | ---: |
| 2K | 2.381 | 0.849 |
| 4K | 1.542 | 0.893 |
| 8K | 2.990 | 0.753 |
| 32K | 10.691 | No separate warm request |
| 128K | 41.401 | No separate warm request |

The first arm is marked `cold_compile=true`. Short-context warm values use
prefix reuse and are the minimum of later requests, not matched uncached
prefill samples. The long contexts have one combined request each. None of
these numbers establishes a candidate gain.

Across 106 memory observations, minimum available GiB was **20.700 / 26.239 /
21.785 / 24.250** for srv2/srv1/srv3/srv4. No guard issue was recorded.
This workload did not reproduce the earlier srv2 long-context RAM termination.

## Harness correction and disk refusal

The original post-leg helper tried to read
`/home/choiceoh/glm53-cache/.overlay-sha` on workers. That stamp is head-only,
so the helper failed with `FileNotFoundError`; chain recovery began before
any candidate boot. The original helper and failure log are retained.

The corrected helper validates the deployed manifest and all 56 mounted
source hashes on every node against the pinned composed tree, checks the
cache stamp only on the head, and binds the head container ID/start time to
the measured record. A read-only recovery-container preflight passed. This
later observation is **not** an attestation of the prior baseline container.

The fresh bracket `spfrt20907` was queued at 14:14:26. After defaults recovery
finished at 14:15:20, its waiter initially checked public port 8000 although
the restored test endpoint was loopback 18000. That owned waiter was cancelled
and requeued with the supported `HEAD_URL` set to the actual endpoint; queue
and preflight checks remained intact. It admitted at 14:17:16 and refused
before deploy because srv1 had 19,798,864 KiB (18.9 GiB) available disk.
No candidate or second-baseline workload ran.

A privileged read-only inventory found five rank-cache directories of
48,094,192,078 bytes each, **224 GiB total**, under srv1's `glm53-cache`.
Their latest writes were 01:42, 13:18, 13:33, 13:51 and 14:03 KST. The whole
cache directory occupied about 264 GiB. No cache or other user data was
removed. This disk-admission failure is distinct from the earlier 12:25
srv2 earlyoom termination and the unrelated 07:11 srv1 disk incident.

## Recovery and evidence

The first defaults recovery retained reduced capacity/private port because
the second bracket was queued to boot next. When that bracket refused before
deploy, a separate official fleet recovery was required to restore public
port 8000 and production capacity. An initial recovery wrapper quoting error
failed before boot; its log is retained beside the corrected wrapper.
The corrected `SPFR20907PUBLIC` recovery completed at **14:25:02 KST**.
Read-only verification confirmed all four nodes on the pinned image and
expected overlay hashes, new feature flags at their defaults, max length
1,048,576, block override 1,056 and public `0.0.0.0:8000`. Health returned
**200 from srv2 and srv1**. The fleet release warning still queried the old
private port 18000; the separately captured public-endpoint checks establish
restored availability. No additional inference workload was run during
recovery. The retry automation is paused because disk admission remains
blocked.

`first_retry/` and `second_retry/` preserve exact runner scripts, requests,
fleet logs and memory evidence. `boot-SPFRT0907B1.log` is the measured head
log. `recovery-container-preflight.json` captures only the later recovery
containers. `summary.json` records the result and limitations. The existing
local logic gate log is included; no extra GPU probe was run for this retry.
