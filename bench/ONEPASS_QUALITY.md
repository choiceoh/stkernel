# Onepass reasoning quality — harness 44 / ko-reasoning-v2

> 살아 있는 참조 — **원패스가 무엇을 묻고 어떻게 채점하는지. 하니스가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

Canonical onepass now asks for verifiable reasoning certificates instead of
checking whether three retrieved names/numbers occur somewhere in the answer.
The question families are deliberately harder and self-contained. They measure
specified reasoning and instruction-following skills, **not a general intelligence
score or a subjective assessment of prose style**. No external judge model is used.

| Case | Required reasoning | Certificate checked |
| --- | --- | --- |
| `ledger` | Both effective and recording cutoffs; replacement revisions; cancellation; unit conversion; one final rounding; reservation and decision threshold | Selected transaction rows, received units, net/after-loss/available quantities, decision, changed reservation and shortage |
| `portfolio` | Exactly three of five projects; budget, staff, directional dependency, incompatibility; worst aggregate scenario; risk penalty; tie-breaking | All ten candidate rows with cost, staff and every violation, scores for the feasible rows (an infeasible row's score is `null`), optimal/runner-up choices and margin, changed-budget feasible set and optimum |
| `logic` | Implication across five Boolean constraints; universal truth versus underdetermination; counterexamples; contradiction minimality | All possible worlds, four ordered truth statuses and valid witnesses, all minimal unsatisfiable rule sets after a new observation |

Numbers, variable roles, query order and evidence placement vary deterministically
with `seed + ctx`. Each case's rules and data are scattered through its Korean
background document. At 2K, three individual requests each carry their own dossier;
at 32K and 128K, a combined request carries all three dossiers. `ctx` remains an
approximate **document** size: question/schema/template overhead is additional.
The server's actual `prompt_tokens` is the measurement denominator. Small context
overrides never remove evidence to meet a nominal size.

ko-reasoning-v2 (harness 44) lowers the load a little without dropping a skill or
a certificate dimension. Harness 43's answers showed where the reasoning went:
the 2K ledger answer appended a Markdown table and a paragraph after its JSON
("설명 대신 계산표·반례 증명서를 작성한다" read as an invitation) and failed to parse;
three combined answers wrote explanations instead of record IDs into `evidence`;
`received` was read three different ways; one logic answer wrote `unknown`/`true`
instead of `판단불가`/`참`; the 128K stream solved the wrong worlds because the
shuffled variable names and the alphabetical bit order were only stated, never
shown. So: the answer instruction now forbids anything outside the JSON, the
schema spells out enumerated choices (`전량승인|보류`, `참|거짓|판단불가`, `6자리
비트열|null`, `integer|null`) and marks every evidence array as `기록 ID`; each
ledger derivation field is defined by its formula and the L5 scope is explicit;
the ledger loss rate is 5/10/15% instead of 7/9/13%; the portfolio scores only
feasible combinations (infeasible rows still list cost, staff and violations,
with `score: null`); the logic domain record carries an encoding example, and
each case names which record groups belong in which evidence field. Unchanged:
both cutoffs, revision replacement and cancellation; all ten candidate rows and
the four violation kinds; all worlds, ordered statuses, witnesses and every
minimal core; and every grading rule below.

The model submits one JSON object in **visible final content**, with the requested
case IDs and structured calculations, evidence references and witnesses. A single
JSON code fence is accepted; anything after the JSON fails parsing. Set order is
immaterial; query statuses/witnesses retain Q1–Q4 order. Any valid witness is
accepted, not just the first oracle witness.
Missing/extra case IDs, extra fields, duplicate JSON keys, wrong types, nonfinite
values, incomplete candidate/world coverage, incorrect citations and invalid
witnesses fail the relevant checks. Correct text in `reasoning` or
`reasoning_content` cannot rescue a wrong or absent final answer.

Each case reports `format`, `completion`, `result`, `derivation`, `evidence` and
`counterfactual` checks; logic also reports `witnesses`. `failures` preserves
expected and submitted values for every failed certificate dimension. Evidence
fields enumerate the complete relevant source groups prescribed by the task;
they are source-coverage checks, not free-form semantic citation grading.
`score/max_score` provides partial rubric credit, while `quality.ok/total` counts
**fully passed cases**. Every check must pass for a case to pass. This strict
quality gate continues to invalidate performance acceptance; a higher partial
score alone is not an accepted speed improvement. C=1 and C=4 summaries remain
separate (`quality`, `quality_c4`). Standalone runs and `ONEPASS_RUN_INDEX=1`
cover 9 and 36 case results, respectively, excluding preparation/diagnostic
replays. On the same boot, run 2 repeats C=1 only: C=4 preparation, measurement
and diagnostic requests are omitted. `concurrency_coverage` records that omission,
`c4` is empty and `quality_c4` is null, rather than a passing 0/0 result. This
implements the operator's 2026-09-13 policy: C=1 twice, C=4 once. Optional
fixed-decode repetitions add three cases each to C=1.

Regular answers require `finish_reason=stop`. A length-truncated answer fails the
completion check even if some facts are correct. Explicit fixed-length requests
may end with `length`, but still need a complete, correct JSON certificate. Small
user-supplied fixed budgets can fail; no fallback silently restores easy questions.

Default total completion budgets are **16,384 individual / 49,152 combined** tokens;
reasoning caps are half the individual budget (**8,192**) / **24,576** combined.
Explicit fixed-length requests also reserve half their budget for visible content.
Harness 43 (8,192/4,096 and 24,576/12,288) ended every measured reasoning stream
at its cap mid-sentence (`measurements/st_decode_forward_20260913/consumer-v5-incomplete`):
the three 2K answers used 4,336–4,598 completion tokens against a 4,096 reasoning
cap, the combined answers 14,022–14,294 against 12,288, and the 128K stream was
still on the ledger case when the cap closed it, so two cases were answered with
no reasoning at all. One of nine cases passed. Harness 44 doubles both budgets
and asks the ko-reasoning-v2 questions above. Thinking remains enabled. A budget
or question change changes the performance workload, so harness 44 cannot reuse
a harness 43 baseline. Quality at the new budget has not yet been measured; this
change makes no GPU speed or quality claim.

Before preparation, `workloads.json` stores exact prompts, schemas, evidence,
oracles and valid witness sets. **Only prompts are sent to the model**. The record's
`quality_protocol` identifies the version and complete workload hash. Every stream
is retained in `requests.jsonl`; deferred grading appends and fsyncs `quality.jsonl`,
joined by salted request hash. Grade records include context, case, phase, client,
final-content hash and oracle hash. An interrupted run retains completed streams,
available grades and the answer keys for offline inspection. Preparation and GPU
diagnostic replays retain raw responses but do not enter measured quality totals.

CPU verification:

```sh
python3 -m unittest tests.test_onepass_quality
python3 -m unittest discover -s tests -p 'test_onepass*.py'
```

The tests independently replay transactions, search portfolio bitmasks and solve
the named-variable constraints/core minimality from generated evidence. Adversarial
answers exercise lucky correct results with false derivations, missing alternatives,
invented evidence, reasoning-only answers and truncated/malformed JSON. A streamed
C=4 fixture verifies shared workload identity, unique salt hashes and durable grades.
The standalone historical `check-quality.py` retrieval probe remains available for
old investigations; canonical onepass uses only its Korean filler/model resolver.
