# GLM 5.3 Flash required-first request ordering evaluation

The experimental `glm53_required_first` request option remains off until an
actual serving comparison supports changing the default. Parser replay checks
establish transport correctness, not generated-argument completeness.

## Predeclared run

`probes/glm53_required_first_ab.py` runs 30 pairs on one already-running server:
24 automatic tool calls (four preference questions × low/high/max reasoning ×
streaming/nonstreaming), then six job/literal-content guardrails (required,
named, none × streaming/nonstreaming). Required fields occur after optional
fields in the input schemas, including a nested job location. Job calls request
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
No optional stopping for a favorable result is allowed.

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
