# MLA serving5: matched bracket and public restoration complete

The same-source B1/A/B2 bracket completed on 2026-09-07 at 23:52:49 KST.
Public default restoration and the supervisor completed at 23:57:26 with
exit code 0. The strict comparator reports no issues; rerunning it locally
on the preserved raw arms produces the exact same comparison JSON.

| Context | B1 TTFT (s) | A TTFT (s) | B2 TTFT (s) | Latency reduction | Prompt tokens/TTFT gain | Baseline spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 0.879747 | 0.864658 | 0.881688 | 1.82% | +1.86% | 0.22% |
| 32K | 10.474451 | 10.009738 | 10.477625 | 4.45% | +4.66% | 0.03% |
| 128K | 41.584510 | 39.171938 | 41.394429 | 5.59% | +5.92% | 0.46% |

The reduction and reciprocal rate gain use the mean of the two baselines.
The 2K row averages three questions; each long context is one combined
request per boot. Actual token lengths are 2121, 2128, 2128, 32545 and
128559. All measured requests and excluded priming requests have unique
cache salts, with zero prefix-hit changes and exclusive-traffic checks.
All six phases pass retrieval 9/9 and Korean corruption 0/5 (54/54 and
0/30 including priming; 27/27 and 0/15 for measured requests alone).

The candidate was faster than both baselines on the observed 32K and 128K
requests, by more than their baseline spread. This is one candidate boot,
not a confidence interval. TTFT includes request processing and first
content generation; prompt tokens/TTFT is not complete-request throughput.
The 2K kernel route remains unchanged, so its small observed shift is not
attributable to the large-prefill kernel alone. Reduced measurement capacity
does not establish production-capacity candidate acceptance, and this
bracket does not prove cumulative improvement from the original campaign
baseline or achievement of the 40% target. The candidate remains off by default.

## Source and GPU validation

Session `mlaprefill50907` received the normal fleet GPU hold at 23:20:50.
Frozen source: `/home/choiceoh/stkernel-mla-prefill-serving5-0907`, revision
`6e8d1a9e8438d456b43b682abc19789869a5e395`, based on main `944f65c`.
Remote job: `/tmp/glm53-mla-prefill-serving5-0907`. Build stamp:
`dbae9aaeb4f9`. All 56 overlays and the manifest were hash-verified on all
four ranks. Complete arm records retain source/image/model metadata,
hardware, environment, input hashes, actual token counts and boot identity.

The unchanged MLA CUDA source was validated at GPU revision
`3eb219dd2d938479326a5a6704f3789d854367dd`: all 11 numerical/changed-input
graph cases, 7 memcheck cases and 7 racecheck cases passed, with zero errors,
hazards or warnings and exact original-container recovery. Full gate evidence
is retained in the parent measurement directory as `serving3-gpu-*`.

All four candidate ranks have actual post-call
`[megakernel] mla prefill32 LAUNCHED` markers in nonempty archived bind-file
logs. Both baselines correctly have no candidate launch marker. This fixes
the empty-Docker-log evidence collection failure of serving3.

## Controls and restoration

`GMU=0.6229`, `MAX_BATCHED=8192`, `MAX_SEQS=4`, `GRAPH_CAP=32` and
`CG_UTIL_DELTA=0` are frozen from B1. Measurement uses private port 18000,
524288 KV tokens, max length 262144 and 415 blocks. At 23:57:26 the runner
confirmed public port 8000, max length 1048576, 2000000 KV tokens and 1056
blocks, source/image/overlay/model/hardware identity on all four nodes,
`VLLM_GLM53_MK_MLA_PREFILL32=0` and health 200. This recovery evidence is
historical: the next queued MoE probe acquired the fleet at 23:57:28.

The archived generic chain judge discusses decode step/s, whose medians
were approximately 21.86 in all arms; it is not the prefill verdict. The
prefill result is `comparison.json`. Legacy client "warm" columns denote
later requests here, not prefix reuse. The complete raw arms, client/fresh/
memory logs, rank logs, controls, comparison and restoration evidence are
retained beside this document, with checksums in `sha256.json`.
