# GLM 5.3 Flash required-first request ordering evaluation

The experimental `glm53_required_first` request option remains off until an
actual serving comparison supports changing the default. Parser replay checks
establish transport correctness, not generated-argument completeness.

## Run design

`probes/glm53_required_first_ab.py` runs 30 pairs on one already-running server:
24 automatic tool calls (four preference questions × low/high/max reasoning ×
streaming/nonstreaming), then six job/literal-content guardrails (required,
named, none × streaming/nonstreaming). Required fields occur after optional
fields in the input schemas. Half the preference cases wrap their question in
an array item (as in the upstream report); the other half use a flat object.
The job schema also includes a nested location. Job calls request
a 250-word description and ten responsibilities. The fixtures are synthetic;
returned tools are never executed.

Both arms send identical prompts, tools, tool choice, temperature zero, seed,
reasoning effort, and token budget; only the explicit boolean
`glm53_required_first` differs. AB/BA alternates within each four-case block.
The primary endpoint is successful completion with a valid, complete tool
schema. Missing required fields, invalid JSON, schema errors, wrong/missing
calls, length finishes and HTTP/stream failures are recorded separately. Each
response retains final content/tool arguments and usage, but omits reasoning
text and credentials. A transport failure ends the probe to avoid stacking a
request behind a possibly still-running generation.

The fixed-sample screen requires all 24 automatic pairs and all guardrails,
a stable container/image/command/parser/middleware/overlay identity, a one-sided
paired sign-test p ≤ 0.05, at least five percentage points of success gain,
fewer required-field omissions, no increase in other error categories, no
candidate guardrail failures, and median latency ratio ≤ 1.20. Passing supports
promotion review, not automatic deployment. Equal perfect results do not pass.
These are diagnostic synthetic cases, not independent samples of production
traffic; the p-value is a screening statistic, not a population guarantee.
No optional stopping for a favorable result is allowed. The original recorded
screen counted transport pairs separately; post-review scoring collapses both
transport modes into 12 topic/effort groups before the sign test and requires
both modes to pass within each group. It also checks the requested minimum
250-word description in required/named cases (even though the schema property
is optional). The original plan and scores are preserved in the raw evidence;
the stricter post-review result is recorded separately below.

The fleet CPU prerequisite runs `tests/test_glm53_required_first_ab.py` before
a `kind: probe` generation reservation. Manifests pin the image, hardware and
model metadata, and hash the runtime identity and model/config/overlay files.
The runner checks the container before every request and after the run. Fleet
reservation protects worker changes; the harness directly attests the head
container only. Existing weights are identified by their config/index hashes,
not a new full hash of hundreds of gigabytes. The serving process is not rebooted.
The default budget is 4096 output tokens/request, 180 seconds/request and 30
minutes for the probe. Generic fleet probe exit 0 remains `incomplete` until
these quality records are reviewed.

## Motivation

Upstream [vLLM PR #55558](https://github.com/vllm-project/vllm/pull/55558)
reports missing required fields when optional choices precede a required tag,
and improved native-FP8 results after reordering. That is evidence for testing,
not evidence that this NVFP4 server benefits. The forced-call checks also cover
the long-output failure described in
[vLLM issue #55541](https://github.com/vllm-project/vllm/issues/55541).

## Measured decision — 2026-09-07

**Keep the option off by default.** The complete 30-pair run did not show a
benefit. It does not establish that ordering is generally harmful either.

| Metric | Default order (`false`) | Required-first (`true`) |
| --- | ---: | ---: |
| Schema-complete responses, all cases | 29/30 (96.7%) | 28/30 (93.3%) |
| Schema-complete automatic tool calls | 23/24 | 22/24 |
| Missing required fields | 0 | 1 |
| Other schema errors | 1 | 1 |
| Schema-only required/named/none checks | 6/6 | 6/6 |
| Invalid JSON / length / HTTP or stream errors | 0 / 0 / 0 | 0 / 0 / 0 |
| Median request latency | 1.876 s | 2.040 s |

There were zero candidate-only successes and one candidate-only failure in the
24 automatic pairs (one-sided sign-test p = 1.0). Both arms returned an object
instead of the required `questions` array for the low-effort, nonstreaming
laptop question. The candidate additionally omitted `/questions/0/tag` for the
low-effort streaming laptop question. High/max cases and all six guardrails
passed the original schema-only checks in both arms. The candidate's median latency was 8.7% higher in this
sample; output lengths differ, so this is not a controlled throughput verdict.
"Schema-complete" measures response termination and the declared tool schema,
not prose quality or exact compliance with the requested description length.

The reorder was effective: required/named job calls changed their emitted key
order from `description, responsibilities, location, title, tag` to
`tag, title, location, responsibilities, description`. Automatic preference
calls frequently retained `choices` first even in the candidate. These
observations support keeping a per-request compatibility/experiment option,
not a global default change.

The probe ran from **06:25:37 to 06:29:03 KST**, taking 205.8 seconds, on the same
four-GB10 TP4 NVFP4 serving process, image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`,
with DFlash k=5 and overlay
`1af1a93f83b5148156125a1c2058b6c5888b56bf021f49bf8482202693852243`.
The head container identity, command hash and loaded parser/middleware hashes
remained unchanged throughout. The launcher had verified 56 overlay preimages
and the staged template digest
`0b76cfb35dfd15db1109043c77e99b807bf95024b5b71ca8b435e1c3a844fb42`.
No candidate was deployed and no default was changed after this result.

Before generation, the old serving process was stopped by a separate fleet
probe. A normal fleet boot (`ORDERABRESTORE`) restored serving, then the new
container identity was pinned. The initial 30-minute queued submission
`c4bd265877774d92873d` was cancelled **before any model request**. It was replaced
with a hard 900-second budget and 15-minute estimate to use the fleet's bounded
probe lane while another boot was queued. The sample and decision rules stayed
fixed, and all 30 pairs completed within that budget.

Evidence:

- CPU prerequisite: `833d550082064d43abf2`, seven tests passed on host Python
  3.12.3 with jsonschema 4.10.3.
- Live probe: `94cfe5a3443141bd8741`, source
  `88d0a518f3f1f1cd833a4feb706a4cfa0d48a578`.
- [All 60 synthetic responses, plan and summary](evidence/glm53-required-first-20260907.jsonl).
- [Fleet manifest, input hashes and timing](evidence/glm53-required-first-20260907.meta.json).
- Offline replay first revalidated all 60 request hashes and original
  classifications, the original fixture hash and summary, and before/after
  identity. The final harness still reproduces every sent request hash.

Post-run review tightened the exhausted-budget check and made final identity
read failures emit an unstable summary instead of losing the diagnostics.
It also identified two weaknesses in the original promotion screen: short
optional descriptions could evade the intended long-output guardrail, and
streaming/nonstreaming variants were counted as separate sign-test evidence.
The final harness requires 250 description words and collapses transport
variants before that test. These changes strengthen future promotion checks.

[Rescoring the saved responses](evidence/glm53-required-first-20260907.review.json)
with those stricter rules yields **25/30 baseline versus 24/30 candidate**.
All four required/named descriptions in each arm were under 250 words
(baseline 176–230; candidate 222–234), so neither arm passes the strengthened
long-description guardrail. Across the 12 topic/effort groups there are zero
candidate-only successes or failures: the extra streaming omission belongs to
the laptop/low group that already fails the nonstreaming array-shape check in
both arms. The promotion screen still fails (p = 1.0), so the default-off
decision is unchanged. This is explicitly a post-review analysis, not a claim
that the revised rules were the original preregistration.

Twelve focused CPU tests plus the five existing acceptance tests pass. Tests
cover fractional and expired deadlines, cleanup failure, short/missing
optional descriptions and transport duplicates creating false significance.
No second GPU sample was needed: the original 60 responses supply the evidence
for both recorded analyses.
