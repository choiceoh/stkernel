# CUDA binding diagnostic submission evidence

The binding GPU diagnostic did **not run in v1 or v2**. At the latest
read-only snapshot, **2026-09-08 16:55:07 KST**, v3 remained queued behind
`attr0908`, first of two waiting jobs. This archive contains submission and
recovery evidence; it provides no new GPU sanitizer, numerical, performance,
or serving acceptance result.

| Attempt | Frozen source | Observed outcome |
| --- | --- | --- |
| v1 | `ee14d3090f5906f3ab1bc3ab0b5ecd4f7fab4e39` | The declared CPU gate was rejected by the fleet classifier at 16:34:46, exit 5. The driver stopped after that phase; its subsequent GPU command was never submitted. |
| v2 | `7254422f044ab3c5d042f32ee7f33add7baf5e00` | Normal GPU reservation received GO at 16:44:07. The no-device sanitizer version preflight passed, then the incoming-state guard rejected four present but stopped containers before the diagnostic GPU cell. Final exit 1. The outer supervisor completed public-default restoration and normal release at 16:49:14. |
| v3 | `7254422f044ab3c5d042f32ee7f33add7baf5e00` | Requeued normally at 16:50:14. Preflight passed. At 16:55:07 the submission driver PID 1808590 existed, `attr0908` still held the lane, queue position was 1/2, and neither `capture/` nor `exit.json` existed. No diagnostic result was available. |

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

The latest v3 [status snapshot](v3/status.snapshot.20260908T075507Z.json),
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
