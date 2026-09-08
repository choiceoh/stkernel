# Admission evidence for the host-memory experiment

Session `glm53observemem0908v1`, frozen source
`37db5eb1fb17b1ed03d5116f02e317fc64ffee4c`, all four hosts at
`/home/choiceoh/stkernel-prefill-observation-memory-0908-1`.
All 69 CPU-evidence source hashes match on each clean clone; 34 CPU tests passed.
Normal preflight passed, then the job waited for public requests to finish.
GO was 2026-09-08 11:36:19 KST. The detached driver PID was 3839722.
The source is frozen for the entire run and must not be updated in place.

The runner explicitly requests `--reclaim-host-memory`, followed by the unchanged
12 GiB guarded PRIME/baseline/profile/routes sequence if memory permits. The head
job is `/tmp/glm53-prefill-observation-memory-0908-1`, with `capture/` evidence
and eventual `exit.json`. The fleet owns clone cleanup and public restoration.
This folder records admission and preparation only, not successful reclamation,
model request execution, TTFT, quality or restoration of this new attempt.
