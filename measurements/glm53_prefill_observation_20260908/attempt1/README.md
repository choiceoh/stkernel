# First observation attempt: configuration check failure

Session `glm53observe0908v1`, source `75686447d5cca72904b7050e333f8c319884a28d`.
Normal GO 2026-09-08 10:23:42 KST; approved-default boot completed at 10:34:29.
Outer exit 1 at 2026-09-08T10:34:33.620778+09:00.

All four clone preparations rejected Docker's null-to-false OomKillDisable
representation. Originals had not been paused and clones never started.
`cleanup.json` proves owned clones were removed; the supervisor also emitted
`FLEET_OBSERVATION_CLONES_REMOVED`, confirmed healthy approved public defaults,
and released the hold. `post-original-check.json` verifies all four original
IDs/config/source identities are unchanged and running, with public health true.
`restored_original=false` in completion means no pause/restart occurred.

The approved launcher chose GMU0.6429 after its ordinary memory preflight, retaining
KV2000000/maxlen1048576/1056blocks. This is recorded runtime context, not a matched
performance comparison. No observation request, TTFT, quality, profiler trace or
routing report was collected. Performance acceptance remains false.
