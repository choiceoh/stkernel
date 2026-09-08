# Second observation attempt: private boot passed, request guard refused

Session `glm53observe0908v2`, source
`cfd69dd5b7ad89847fabaa3639dbf8c99215c08f`, frozen on all four hosts under
`/home/choiceoh/stkernel-prefill-observation-0908-2`.

Normal fleet GO was 2026-09-08 10:54:57 KST. All four clone preparations and
post-start configuration identities passed. Private head application startup
completed at 10:58:56; health and idle observer status returned HTTP 200.
The observer had not begun and no model request was sent.

`capture/PRIME.memory.jsonl` records the request guard's first sample:

| Host | Available KiB | Available GiB | 12 GiB guard |
|---|---:|---:|---|
| srv2 (head/local) | 9,540,720 | 9.10 | refused |
| srv1 | 15,934,636 | 15.20 | passed |
| srv3 | 11,081,804 | 10.57 | refused |
| srv4 | 16,072,312 | 15.33 | passed |

The guard exited 3 before spawning the PRIME client. The lifecycle then removed
all owned clones and restarted the exact original containers. `completion.json`
correctly records collection incomplete and original restoration successful.
No TTFT, quality, profiler trace, model routing or performance acceptance exists.
The actual compiled hook, layer mapping and trace schema remain untested.

Outer fleet exit was 1 at 11:01:25 KST. The supervisor independently confirmed
clone removal and healthy approved public defaults. `post-original-check.json`
compares every original immutable identity, verifies all four are running,
checks public health true/private health false, records zero remaining session
clones, and retains the fleet release record. Holder and queue were empty at
that read-only check. The approved model capacity remained unchanged:
max length 1,048,576, 1,056 GPU blocks, max batched tokens 8,192, max sequences 4,
and launcher-selected GPU memory utilization 0.6429.

## Post-restoration resource census

`read_memory.py` reads procfs and Docker metadata only. It collects process names,
PSS/RSS, cgroups and resource counters, excluding arguments and environment.
`post-memory.json` preserves all four host results with timestamps. Processes
below 10 MiB RSS are excluded; counters are not an atomic snapshot. PSS and
cgroup usage are distinct accounting views and must not be added together.

Head still had 8.583 GiB available, with only the original GLM container running.
Its GLM API process had 4.995 GiB PSS (4.816 GiB anonymous); the worker had
9.569 GiB PSS and engine 1.210 GiB. Other observed head processes were individually
below 0.15 GiB PSS. No abandoned observation container accounts for the shortage.
This identifies retained serving memory for investigation, not a proven leak,
allocator cache or observer overhead.

Srv3 had 10.788 GiB available. Its pre-existing PaddleOCR container used
3.964 GiB of cgroup memory. That service was not stopped or modified. Srv1 and
srv4 had 15.609 and 15.357 GiB available. Recorded running-container cgroup OOM
and OOM-kill counters were zero; that does not establish a host-wide OOM history.
The boot's disk reserve check passed on every host (at least 301.57 GiB free).

Do not repeat this capture unchanged, reduce model capacity or weaken the guard.
Investigate recoverable serving memory before another normal-queue attempt.
No model or serving behavior is changed by this evidence update.

`remote-source-hashes.json` verifies all 26 archived remote files byte-for-byte.
`SHA256SUMS` covers the archive plus this read-only follow-up evidence.
