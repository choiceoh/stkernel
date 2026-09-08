# CUDA binding diagnostic submission evidence

The binding GPU diagnostic did **not run in v1, v2, v3 or v4**. V3 received GO at
2026-09-08 17:35:38 KST but its mixed incoming state failed the strict lifecycle
guard before the GPU payload. This archive contains submission and
recovery evidence; it provides no new GPU sanitizer, numerical, performance,
or serving acceptance result.

| Attempt | Frozen source | Observed outcome |
| --- | --- | --- |
| v1 | `ee14d3090f5906f3ab1bc3ab0b5ecd4f7fab4e39` | The declared CPU gate was rejected by the fleet classifier at 16:34:46, exit 5. The driver stopped after that phase; its subsequent GPU command was never submitted. |
| v2 | `7254422f044ab3c5d042f32ee7f33add7baf5e00` | Normal GPU reservation received GO at 16:44:07. The no-device sanitizer version preflight passed, then the incoming-state guard rejected four present but stopped containers before the diagnostic GPU cell. Final exit 1. The outer supervisor completed public-default restoration and normal release at 16:49:14. |
| v3 | `7254422f044ab3c5d042f32ee7f33add7baf5e00` | Normal preflight passed at 16:50:14 and GO arrived at 17:35:38. The incoming head was stopped while three workers were running. The no-device sanitizer preflight passed, then the strict lifecycle guard refused the GPU payload. Final exit 1; restoration responsibility was handed to the next boot. |
| v4 | `7254422f044ab3c5d042f32ee7f33add7baf5e00` | Four healthy, idle public containers were verified before submission with the latest main scheduler. GO at 17:55:30. All four incoming containers were running, but a request was active at the mandatory idle check. GPU payload refused, exit 1. Supervisor restoration separately failed its CPU regression gate, then released at 17:56:37. |

V3's [completed originals](v3/completed/source-manifest.json) preserve all
11 job files, rechecked byte-for-byte against the remote source. Its
[completion](v3/completed/capture/completion.json) has no GPU cell or sanitizer
result and retains `restored_original: false`. The normal supervisor accepted
restore responsibility from `attr0908` after that payload failed, then passed
it to `vllmprs0908b` after the diagnostic's refusal. A separately captured
[later public state](v3/completed/later-public-state.json) found four running
containers and HTTP 200. These were different container identities; that
later health observation does not establish exact restoration by v3.

V4's [completed originals](v4/completed/source-manifest.json) preserve and
recheck all 14 job files. The preparation's empty driver was corrected before
any fleet submission: a read-only SSH subprocess consumed the preparation's
stdin; the driver was subsequently transferred through an encoded argument.
The original PID, empty-file checks and correction are recorded. No duplicate
GPU job was queued. Source 7254422 and the CPU8 receipt remained unchanged.

The v4 [scheduler receipt](v4/completed/scheduler.json) records main revision
`0d8ce2cef9a0991075329f4d8957091ddc43fb58` and the pre-existing `prodrec0908`
restore-debt record. Actual frozen runner hashes are in its
[lifecycle excerpt](v4/completed/lifecycle.relevant.jsonl). The strict
all-running/idle checks were not relaxed. The [fleet log](v4/completed/fleet.log)
records the supervisor's failed restoration: the approved-main gate reported
192 tests in two shards failed, including fleet asynchronous submission
contracts. This is not a candidate kernel failure or successful restoration.
An immediate read-only status showed health 200; the durable
[later snapshot](v4/completed/later-public-state.json) was taken after
`arconsumer0908v11b` acquired the lane and began its own boot, and must not be
used to infer our restoration outcome. Further retries need a stable idle
boundary and a passing normal restoration path.

The v2 runner's [completion receipt](v2/capture/completion.json) truthfully
retains `restored_original: false`: its incoming-state validation failed
before the runner entered its pause/restore scope. The later outer fleet
supervisor recovery is a separate event, recorded in the complete
[v2 fleet log](v2/fleet.log) and the
[fleet history excerpt](fleet-history/log.relevant.snapshot.20260908T075317Z.txt).
The [event excerpt](fleet-history/events.log.relevant.snapshot.20260908T075317Z.txt)
independently records GO and normal release. This archive does not contain
a separate four-node or HTTP health snapshot.

The [incoming snapshot](v2/capture/before.json) records `running: false`
for all four nodes with the pinned image. The
[preflight receipt](v2/capture/sanitizer-preflight.json) records version
`2025.3.1.0`, executable SHA-256
`7a7fcdefb67042731daf021478176f4919e1843d0b10cb697af28a7d8a3d108b`, exit 0,
and `cuda_devices_exposed: false`. It only ran `--version` in a bounded
no-device container. [Resource availability](v2/capture/resources-before.json)
was captured before the incoming-state rejection.

Both v2 and v3 preserve their submission, source freeze and CPU receipt.
Their `cpu-evidence.json` files are byte-identical to
[CPU8's result](../cpu8/local/result.json): 48 tests, zero failures/errors/skips,
`cuda_initialized: false`, and 24 admitted Triton remap compilation variants.
The freeze receipts bind the diagnostic source to these CPU checks; they do
not establish a GPU diagnostic result.

The historical v3 [status snapshot](v3/status.snapshot.20260908T075507Z.json),
[complete fleet-log snapshot](v3/fleet.snapshot.20260908T075507Z.log),
[holder snapshot](v3/fleet-state/holder.snapshot.20260908T075507Z.txt), and
[queue snapshot](v3/fleet-state/queue.snapshot.20260908T075507Z.txt) are fixed
copies of mutable live sources. The earlier 16:53:17 snapshot is retained;
the additional waiting job subsequently changed position 1/1 to 1/2.
These snapshots must not be read as current status after their stated time.

[source-manifest.json](source-manifest.json) records every copied source
path, byte length, mtime and SHA-256. Fleet history files are exact matching
line excerpts; both full-source and excerpt hashes are recorded. Immutable
sources were read again and matched all 20 archived originals.
[verification.json](verification.json) records those remote rechecks and
local byte verification, including expected later changes to mutable files.
`before.json` stores hashes of container configuration, host configuration
and mounts; no raw container environment was collected. No remote jobs,
queue entries, containers or services were changed while archiving.

Verify this directory with `shasum -a 256 -c SHA256SUMS`.
