# MLA serving5: completed B1 and A; B2 pending

Snapshot: 2026-09-07 23:45 KST. **The bracket is incomplete.** B1 completed
at 23:34:52, candidate A at 23:45:05, and B2 began booting at 23:45:05.
No final speedup, production-capacity acceptance or cumulative 40% result
is established by this snapshot. Public default restoration is still pending.

Session `mlaprefill50907` received the normal fleet GPU hold at 23:20:50.
The frozen source is `/home/choiceoh/stkernel-mla-prefill-serving5-0907`,
revision `6e8d1a9e8438d456b43b682abc19789869a5e395`, based on main `944f65c`.
The remote job is `/tmp/glm53-mla-prefill-serving5-0907`. Build stamp is
`dbae9aaeb4f9`; all 56 overlays and the manifest were hash-verified on all
four ranks. The original request, submission, source contract and gate
identity are retained beside the arm artifacts.

The unchanged MLA CUDA source was validated at GPU revision
`3eb219dd2d938479326a5a6704f3789d854367dd`: all 11 numerical/changed-input
graph cases, 7 memcheck cases and 7 racecheck cases passed, with zero errors,
hazards or warnings and exact original-container recovery. Full gate evidence
is retained in the parent measurement directory as `serving3-gpu-*`.

## Preliminary request times

Each arm first ran excluded priming, then five measured requests with new
cache salts. These are request-to-first-content TTFTs, not isolated kernel
times. The 2K row is the mean of three questions; 32K and 128K are one
combined request each. Actual prompt lengths are 2121, 2128, 2128, 32545,
and 128559 tokens. The first baseline is the only comparator available yet.

| Context | B1 TTFT (s) | A TTFT (s) | Preliminary latency reduction vs B1 |
| --- | ---: | ---: | ---: |
| 2K | 0.879747 | 0.864658 | 1.72% |
| 32K | 10.474451 | 10.009738 | 4.44% |
| 128K | 41.584510 | 39.171938 | 5.80% |

All four completed phases (B1/A priming and measured) passed retrieval 9/9,
Korean corruption 0/5, fresh-cache and exclusive-traffic checks. The preserved
arm phase validator reports no issues across 20 unique request salts.
All four candidate ranks have the actual post-call
`[megakernel] mla prefill32 LAUNCHED` marker in nonempty archived bind-file
logs. The baseline correctly has no candidate launch marker. This resolves
the empty-Docker-log evidence collection failure of serving3.

`GMU=0.6229`, `MAX_BATCHED=8192`, `MAX_SEQS=4`, `GRAPH_CAP=32` and
`CG_UTIL_DELTA=0` are frozen from B1. Measurement uses private port 18000,
524288 KV tokens, max length 262144 and 415 blocks. The runner must still
complete B2, the strict three-arm comparison and public default restoration
(port 8000, 2000000 KV tokens, max length 1048576 and 1056 blocks).
`VLLM_GLM53_MK_MLA_PREFILL32` remains default-off.

The legacy client table's "warm" label denotes later requests here, not
prefix reuse: every request has an independent cache salt. A single
candidate boot and this reduced capacity limit any eventual result.
`progress-summary.json` contains the machine-readable incomplete status.
