# Onepass reasoning quality — harness 43 / ko-reasoning-v1

> 살아 있는 참조 — **원패스가 무엇을 묻고 어떻게 채점하는지. 하니스가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

Canonical onepass now asks for verifiable reasoning certificates instead of
checking whether three retrieved names/numbers occur somewhere in the answer.
The question families are deliberately harder and self-contained. They measure
specified reasoning and instruction-following skills, **not a general intelligence
score or a subjective assessment of prose style**. No external judge model is used.

| Case | Required reasoning | Certificate checked |
| --- | --- | --- |
| `ledger` | Both effective and recording cutoffs; replacement revisions; cancellation; unit conversion; one final rounding; reservation and decision threshold | Selected transaction rows, received units, net/after-loss/available quantities, decision, changed reservation and shortage |
| `portfolio` | Exactly three of five projects; budget, staff, directional dependency, incompatibility; worst aggregate scenario; risk penalty; tie-breaking | All ten candidate rows including every violation, optimal/runner-up choices and margin, changed-budget feasible set and optimum |
| `logic` | Implication across five Boolean constraints; universal truth versus underdetermination; counterexamples; contradiction minimality | All possible worlds, four ordered truth statuses and valid witnesses, all minimal unsatisfiable rule sets after a new observation |

Numbers, variable roles, query order and evidence placement vary deterministically
with `seed + ctx`. Each case's rules and data are scattered through its Korean
background document. At 2K, three individual requests each carry their own dossier;
at 32K and 128K, a combined request carries all three dossiers. `ctx` remains an
approximate **document** size: question/schema/template overhead is additional.
The server's actual `prompt_tokens` is the measurement denominator. Small context
overrides never remove evidence to meet a nominal size.

The model submits one JSON object in **visible final content**, with the requested
case IDs and structured calculations, evidence references and witnesses. A single
JSON code fence is accepted. Set order is immaterial; query statuses/witnesses
retain Q1–Q4 order. Any valid witness is accepted, not just the first oracle witness.
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
separate (`quality`, `quality_c4`). Default coverage is 9 and 36 case results,
respectively, excluding preparation/diagnostic replays. Optional fixed-decode
repetitions add three cases each to C=1.

Regular answers require `finish_reason=stop`. A length-truncated answer fails the
completion check even if some facts are correct. Explicit fixed-length requests
may end with `length`, but still need a complete, correct JSON certificate. Small
user-supplied fixed budgets can fail; no fallback silently restores easy questions.

Default total completion budgets are **8,192 individual / 24,576 combined** tokens;
reasoning caps are half the individual budget (**4,096**) / **12,288** combined.
Explicit fixed-length requests also reserve half their budget for visible content.
Harness 42's three observed 2K preparation requests all reached their 800-token
reasoning cap mid-sentence, before completing the calculations. The larger budget
addresses that limit; it does not change the questions, oracles or grading rules.
Thinking remains enabled. A budget change changes the performance workload, so
harness 43 cannot reuse a harness 42 baseline. Higher-budget quality has not yet
been measured; this change makes no GPU speed or quality claim.

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
