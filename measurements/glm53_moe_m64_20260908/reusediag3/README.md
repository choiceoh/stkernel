# Completed-fixture release diagnostic, 2026-09-08

This diagnostic failed and the M64 experiment is now deprioritized. It is not a
serving or numerical acceptance result. No further automatic M64 retry is planned.

Frozen source: `2afb45e3dcc56b7a156f13177b134d10a4b62287`, all four nodes.
Normal fleet session `moem64reuse30908` received GO at 08:07:53 KST. The probe ran
08:09:05–08:12:07. Exact incoming four-node container/configuration recovery and
endpoint health completed before outer exit 1 at **08:14:45 KST**.

All three intentional-error detector controls passed. Memcheck recorded 40/48
trials and 274,432 row-trials per arm with zero candidate/control failures, then
exited 15 before the 8192/concentrated case ran. The report says zero sanitizer
errors in the executed portion, but no completion marker exists. Racecheck was
not reached. Earlier independent-control and candidate failures remain valid.

Case inputs/outputs were released only after the original lifetime checks and
synchronization. Live CUDA allocated bytes after each completed case nevertheless
grew from 19,125,155,840 to 36,307,220,480, 55,637,047,296, 74,966,874,112 and
97,876,293,632. The last observed CUDA free/total bytes were
14,269,620,224 / 128,520,081,408. The release did not resolve allocation accumulation;
the exact allocations retained by the runtime/diagnostic are not identified.

640 resource samples show container peak **2,902,802,432 bytes** against a
17,179,869,184-byte limit, `OOMKilled=false`, and cgroup max/oom/oom_kill counters
zero. Minimum observed host MemAvailable was **6,683,123,712 bytes**. Cgroup readings
alone do not cover GPU driver allocations. The retained privileged kernel-journal
query found the two earlier `NV_ERR_NO_MEMORY` events, but no new one for this run.
The specific exit cause is unresolved; do not claim a newly proven driver OOM.

`analyze.py` verifies the ordered partial plan, all retained BF16 payloads and
failure math, detector controls, the 16-record memory prefix, process evidence
and exact incoming recovery. `summary.json` always leaves numerical/serving
acceptance false. `SHA256SUMS` covers the archived files. The snapshot intentionally
retains raw logs and the exact probe source, including the ineffective release.
