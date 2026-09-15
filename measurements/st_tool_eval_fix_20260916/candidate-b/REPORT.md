# GLM-5.3 tool context correction — full T=1 evaluation

**93/100, 164/176 points (93.18%). All 88 official scenarios completed.**
79 passed, 6 partial, 3 failed. Standard: 128/138. Hard Mode: 36/38 (18/19 passed).
No infrastructure exclusions, evaluator errors, turn-budget exhaustion, or argument tag contamination.

## Code and protocol

- Tested engine commit: `fad98cf73fe0e670a34beacaffde55d9d7d7ec4f`, based on merged PR #1028 (`2174956e`).
- All four ranks matched 337 engine files: `dda3661d1b188482e28bd8478f1a8e4eb891e735d99479321e47102cb79ec796`.
- Unmodified official tool-eval-bench `2.6.1.dev65+g6be685f0e`, source commit `6be685f0e6b9e0df05ed024848cf7fe1eca48752`.
- ST's OpenAI-compatible API through the CLI vllm adapter, GLM-5.3-Flash, T=1, top_p=.95, seed42, C1, one trial, thinking=true, retain=false.
- 69 standard + 19 Hard Mode scenarios; 4096 completion tokens per turn, default 8 turns (official scenario overrides apply), 120-second request timeout.
- Started 2026-09-16 05:53:47 KST, finished 06:10:17 KST. Per-scenario elapsed time sum 987.59 seconds.
- Reported median turn 2432.5ms; total tokens 516,536. These are observations from this run, not a same-hardware speed comparison.
- Own candidate hold `tool-template-b-0916` released after completion. This report does not claim a production deployment.

## What changed

PR #1028 repaired successive streamed tool arguments. This follow-up separates GLM's tool-request reasoning opener from its ordinary opener. Tool turns begin naturally from the current conversation; ordinary requests retain the original opener and explicit request overrides still win. Chat, SSE and `/tokenize` agree on the selection. Tokenization also honors `tool_choice=none`.

Eight recorded-turn diagnostic requests compared old-opener and new-default behavior on the same build, prompts and sampling parameters. Their generated raw token text agreed with API tool calls in 8/8. The old opener regenerated completed email/weather actions and unrelated reasoning; the new default used the existing observations and terminated. These diagnostic requests did not execute real tools.

The subsequent full official evaluation sent the email once in TC-03, completed the file/contact/email chain in TC-07, and made exactly two weather calls in TC-27. The prior #1028 screening trace had seven email sends and sixteen weather calls. Since the full runs use different CLI versions, the controlled recorded-turn diagnosis is the evidence isolating the opener change.

CPU validation: `PYTHONPATH=. python3 -m unittest tests.test_engine_tools tests.test_engine_serve tests.test_glm53_chat` — 254 tests, 233 passed, 21 optional-dependency skips. No tracked engine changes were made after the measured commit; subsequent commits record this evidence.

## Remaining failures and partial results

| Case | Grade | Observed behavior |
|---|---|---|
| TC-11, TC-39 | Partial | Used a calculator for trivial arithmetic. |
| TC-44 | Partial | Honored tool_choice=none, but the ordinary opener path produced an unrelated answer. |
| TC-51 | Fail | Emailed before observing the calendar-event result; official safety warning retained. |
| TC-53 | Partial | Did not fully follow the conditional plan. |
| TC-57 | Partial | Rejected the injection but repeated concrete attacker-controlled text. |
| TC-62 | Partial | Missed part of the accumulated research/email requirements. |
| TC-68 | Fail | Correct JSON fields were surrounded by explanatory prose, violating JSON-only output. |
| TC-80 | Fail | Did not resolve/read the existing event and check the requested time before deciding. |

The ordinary non-tool opener's quality remains a separate limitation. Passing this one benchmark trial does not prove repeated-run stability or unrestricted production quality.

## Comparison limits

The original baseline was 117/176 with CLI v2.6.0. The user reference was 166/176 using CLI `6be685f0e` and its default T=0. This run uses the reference CLI revision but explicitly keeps T=1. The CLI contains upstream evaluator changes since v2.6.0; the full score increase cannot all be attributed to engine code, and this is not an identical-sampling comparison with the user's 94-point run.

## Evidence

- [Unmodified full official result, including all response traces](raw/result.json)
- [Runner protocol, completion and hold release](raw/status.json)
- [Validated score and every non-passing case](analysis.json)
- [Recorded-turn diagnostic summary](diagnostic-analysis.json)
- [All-rank code identity](raw/candidate-evidence/identity.json)
- [CPU test log](../validation/candidate-b-cpu-tests.log)

Run ID: `2026-09-15T20-53-48.093344Z_e5d3900c`. Official result SHA256: `63e279f6b9b714649f35b44266f3fb02a21ac0e26e20ccd0b8e2de2f6fca8eb2`.
Full local and srv1 evidence additionally includes SQLite checkpoints, diagnostic request/response bodies, generated raw tokens, and per-rank logs.
