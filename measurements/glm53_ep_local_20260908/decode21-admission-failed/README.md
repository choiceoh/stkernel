# CPU21: memory admission refused before compilation

Normal fleet CPU session `epdecodecpu0909v21` attempted the no-device runner on
srv1 through srv2. The recorded outer interval is 2026-09-09 12:17:51.120947–12:17:51.736944 KST
(0.615997 seconds); driver PID115895. Head job:
`/tmp/glm53-ep-decode-cpu-0909-21`. Frozen source:
`028f98167376f0a0857c20c7ec89a3505c1a000f` at
`/home/choiceoh/stkernel-ep-onepass-0909-21`.

**Payload and outer return codes are2; copy return code is null.** The original
fleet log records refusal by the unchanged12GiB `MemAvailable` guard. Its exact
available-memory value was not recorded at the failed check, so this archive
makes no claim about that value. This is an admission failure, not a kernel or
numerical failure and not a completed CPU test run.

The actual frozen runner source is preserved and SHA-bound to the original
submission. Its memory rejection precedes output-directory creation and the
Docker invocation. The worker output path
`/home/choiceoh/glm53-ep-cpu-0909-21/evidence` was absent in both collection
snapshots. No container, compiler, test result, PTX, cubin, or GPU measurement
was produced by this attempt. No result or artifact archive is fabricated.

All five original job files are preserved under `head/`: submission, complete
fleet log, terminal exit, driver source and PID. Their10,171 original bytes and
hashes remained unchanged across collection. Both source checkouts were clean
and at the same frozen revision during collection. `frozen-runner.py` is an
additional source witness, not a new execution. `capture.json` records the two
read-only snapshots. The pure local `verify.py` checks bytes and guard ordering;
its `verification.json` PASS means archive integrity only.

The parent separately prepares a fresh CPU21b job on srv4 with the same frozen
source and unchanged limits. Its outcome belongs to separate evidence. This
archive neither retries nor claims success for that attempt, and it did not
change memory limits, reclaim caches, stop services, or launch workloads.

`SHA256SUMS` covers every file here except itself. No prior failure archive was
modified.
