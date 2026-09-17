# Final answers, certificate errors, and prompt ambiguity

The 282-unit ledger case excluded symmetrically, baseline and `ss1` both have
**32/32 correct final answers** across the full campaign and separate 2K recheck
(repeated cases included). The original strict certificate scores are
**191/205 (93.17%) versus 199/205 (97.07%)**, an 8-point / 3.90-percentage-point
candidate advantage. These scores still deduct transcription errors; no score
was silently forgiven. Fully correct certificates are 24/32 versus 28/32.
The two campaigns' separate totals remain in `quality-excluding-order282.json`.

`quality-error-audit.json` covers **all 12 non-ledger failing responses** from
both arms and all six completed records. It preserves the original case text,
grades, request hash, raw-artifact hashes and narrowly scoped response excerpts.
All 12 pass the `result` dimension. They must not be described as 12 incorrect
final judgments.

| Observed issue | Responses | Assessment |
|---|---:|---|
| Correct named variable assignments encoded as incorrect bit strings | 10 | Confirmed transcription errors. Every invalid reported world has a corresponding correct named assignment in the recorded response. Invalid witness encodings can fail the witness check as well. |
| U6 omitted from the minimal inconsistent set | 1, baseline | Prompt ambiguity: the response explicitly considers whether U6 is a fixed scenario assumption or a rule to include in the set. |
| Portfolio ABE staff written as 8 instead of 6 | 1, baseline | Intermediate-field error; A=3, B=1, E=2. The selected optimum, runner-up, scores, feasibility and changed-budget result are correct. The trace does not establish that this was merely an output typo. |

For the candidate's 2K recheck, the response derives
`A=0, B=1, C=0, D=0, E=1, F=1`, then writes `010001` instead of `010011`.
For its 128K repeat it derives `A=1, B=1, C=1, D=0, E=0, F=0`, then writes
`110100` instead of `111000`. Both have correct final truth classifications.
This is stronger evidence than simply assuming any wrong bit is a typo.

The original ledger prompt's L4 can be read as approving an order of at least
282 units instead of comparing available inventory to the 282-unit order.
Both failed full-campaign candidate responses computed available inventory 279
correctly and explicitly chose the order-threshold interpretation. The 2K
recheck also has a different ledger certificate error: releasing 9 of 44
reserved units was interpreted as leaving 9 reserved, giving 314 instead of
288. The whole-case exclusion removes that error too; it is not selectively
forgiving only the candidate's two final-decision failures.

## Prompt audit and version 3 changes

- **Ledger:** L4 now states the order quantity and explicitly compares available
  inventory to it. L5 explicitly adds/subtracts a delta from existing reserved
  inventory. Base conditions and counterfactual conditions remain separate.
- **Logic core:** U6 now states only the additional assignment. The question
  selects a subset of U1–U6, applies only the rules inside that subset and keeps
  only U0 fixed. Each selected set must be inconsistent, with every single-rule
  removal restoring consistency. This removes the fixed-U6 interpretation.
- **Logic worlds and witnesses:** ABCDEF bit order and its example were already
  explicit. The observed encoding errors do not reveal an alternative valid
  order. The question now also explicitly requires each witness to satisfy all
  base constraints and the corresponding true/false query.
- **Portfolio:** No ambiguity explaining the observed staff sum was found.
  The exact-three rule, inclusive cost/staff limits, dependency direction,
  exclusion, aggregate worst-case score, tie-break and budget-only change are
  explicit. Its wording, numeric data, oracles and grading are unchanged.

Independent tests replay transaction revisions, enumerate portfolio combinations
and enumerate all 64 Boolean assignments / rule subsets. They now include the
actual 2007, 32007 and 128007 seeds in addition to the previous 30 seeds. The
oracle answers and witness sets are unchanged by version 3.

Harness 47 / `ko-reasoning-v3` identifies the revised prompts. The bracket's
sample-count and judge paths require the current harness and quality version;
the judge also compares the complete quality protocol, workload and budgets.
Old v2 grades and throughput remain historical evidence, not v3 measurements.

The user explicitly adopted `ss1` after reviewing these distinctions. The GLM
production recipe therefore adds `ss1`, with top-8 routing unchanged. This is a
source/default change; no new GPU result or live-process restart is claimed by
this update.
