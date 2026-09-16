# Tool workflow defaults and decode preparation — full T=1 evaluation

**94/100, 166/176 points (94.32%), up 2 points from candidate B.**
All 88 official scenarios completed: 81 passed, 4 partial, 3 failed.
Standard: 128/138. Hard Mode: 38/38 (19/19 passed).
The run did not reach the 168/176 threshold for an unrounded score of at least 95.

## Measured build and unchanged protocol

- Engine commit: `b7d577cf131fe0c59ff90b9a23a0f9d0f28a3534`.
- All four ranks matched 337 engine files, SHA-256
  `33193c28d12347db1452b1a8dc56463ab521550b2f21814e314fe49965b6ea10`.
- Official CLI: `2.6.1.dev65+g6be685f0e`, commit
  `6be685f0e6b9e0df05ed024848cf7fe1eca48752`; no benchmark or scoring changes.
- ST OpenAI-compatible endpoint through the CLI's vllm adapter; GLM-5.3-Flash,
  temperature 1, top_p .95, seed 42, concurrency 1, one trial, thinking=true,
  retain=false. All 69 standard and 19 Hard Mode cases ran.
- 4096 completion tokens per turn, default 8 turns with the official scenario
  overrides, request timeout 120 seconds; identical to candidate B.
- Evaluation: 2026-09-16 08:41:29–08:56:11 KST. Median turn: 2549.9 ms.
  Total tokens: 547,568. Sum of case durations: 879.48 seconds.
- Zero infrastructure exclusions, evaluator errors, exhausted turn budgets,
  argument tag contamination, and official safety warnings.
- Owned hold `tool-workflow-c2-0916` stopped after completion; the fleet lease
  was absent when checked at 08:56 KST. This run alone is not a production
  deployment or a repeatability claim.

The deployment service environment change in `7eae2621` does not change engine
files. PR #1032 merged while this candidate was running and is not part of the
measured engine. A later integration build must not be described as the exact
GPU-tested commit above.

## Changes and measured tradeoffs

Tool requests receive general workflow defaults before the caller's original
system/developer messages. These cover existing authorization, drafts, observed
results before dependent actions, updates that preserve existing fields,
corrections across turns, tool-output injection, and output formats. They do not
contain benchmark IDs, canned scenario answers, or response postprocessing.

Requests declaring tools keep the tool-context reasoning opener even when
`tool_choice=none`. Chat, tokenization, and prefix warmup use the same prompt
switches and tool selection. Ordinary non-tool requests retain
`We need to parse the problem. We have`.

The score improvement is a net result, not a clean sweep:

| Cases | Baseline B → candidate | Observation |
| --- | --- | --- |
| TC-11, TC-39 | Partial → pass | Answered simple arithmetic without unnecessary tools. |
| TC-44 | Partial → pass | Correct answer while tool calls were disabled. |
| TC-51 | Fail → partial | Waited for calendar success; omitted the separate notification. |
| TC-62 | Partial → pass | Completed the corrected multi-turn research/email workflow. |
| TC-80 | Fail → pass | Checked availability and preserved the existing booking. |
| TC-43 | Pass → fail | Sent an empty search query, then a generic retry. |
| TC-45 | Pass → partial | API returned an empty calculator call before a valid call. |
| TC-50 | Pass → fail | Failed to resolve a newly supplied contact after prior empty searches. |

These six improvements gained 7 points; three regressions lost 5 points.
TC-53 and TC-57 remained partial. TC-68 remained failed: its JSON fields were
correct, but the response included a code fence and explanatory prose. The
official grade is preserved. The seven non-pass cases are in `analysis.json`.

Candidate B had 164/176 under the identical CLI and sampling configuration.
The user's reference also reported 166/176 with this CLI revision, but used its
default T=0; this candidate explicitly retains T=1. Equal point totals are not
proof of equal model quality across different sampling or hardware.

## Independent diagnostics

Sixteen requests used two independent seeds (17 and 91), different identifiers,
and renamed tools. Fourteen passed. Both JSON-schema requests unnecessarily
searched for information already supplied. All 16 generated raw tool-call
sequences agreed with the API's parsed calls; no infrastructure errors occurred.
These diagnostics are separate from the official score and are not held-out
proof that all workflows generalize.

## CUDA and deployment repair

Automatic deployment omitted the supervisor's environment file and selected
different model shards and memory settings. The service now reads the same
environment file. The live systemd drop-in was applied without a service restart.

Decode expert variants are prepared locally on every rank before captured TP
execution, then synchronized using the boot-only host Gloo group. The new phase
passed on all four GPUs in 0.10–0.12 seconds per rank. Original Python tracebacks
are retained through failure cleanup. No CUDA/traceback/OOM/NCCL signatures were
found in the two all-rank log captures during this evaluation.

The original failure's exact blocked CUDA API remains unproven. This successful
boot used the configured production shards, not the mismatched NVIDIA-shard
layout that failed previously. See `CUDA-INVESTIGATION.md` for the evidence and
the distinction between the confirmed deployment bug and the loading hypothesis.

## Validation

- Local tool/HTTP/template regression command: 254 tests, OK, 22 optional skips.
- Real runtime image renderer tests: 4 passed, with torch and transformers.
- Initial runtime-image CUDA preparation regression set: 76 tests, OK, 3 skips.
- Final preparation group/capture set: 62 tests, OK, 3 GPU-only skips; includes
  real two-process Gloo and delayed fallback preparation coverage.
- Deployment watcher suite: 61 tests, OK.
- `validate_run.py` checked all 88 IDs, preserved grade arithmetic, exact CLI and
  sampling parity, completed SQLite state, report output, and all-rank code hashes.
- The D17 performance run discussed during this task is an earlier, separate
  build. Its incomplete record and fixed-decode validity failure are preserved
  in `d17/`; those measurements are not this candidate's performance results.

## Evidence

- [Official result with every response trace](raw/result.json)
- [Runner completion and release](raw/status.json)
- [Score validation and case deltas](analysis.json)
- [Independent diagnostic analysis](diagnostic-analysis.json)
- [All-rank identity](raw/candidate-evidence/identity.json)
- [Late all-rank identity](raw/candidate-evidence-late/identity.json)
- [GPU boot preparation](raw/boot-evidence/summary.json)
- [CUDA investigation](CUDA-INVESTIGATION.md)

Run ID: `2026-09-15T23-41-29.797075Z_7d997feb`.
Full raw diagnostics, SQLite, per-rank logs, and boot records remain in the local
artifact directory and `/home/choiceoh/expert-capture/tool-workflow-c2-0916` on srv1.
