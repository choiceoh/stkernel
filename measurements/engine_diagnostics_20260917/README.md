# Deneb engine diagnostics — 2026-09-17

Base: `ed4202f7`; branch: `codex/engine-diagnostics-0917`.

## Scope

Host-only Prometheus instrumentation for Deneb Android statistics. No model,
precision policy, graph, kernel, collectives or scheduling decisions change.
The existing telemetry endpoint is extended; no additional device synchronization
or GPU profiling is requested. Host overhead is unmeasured; this is not a speed claim.

- `st:runtime_info`: build revision (or explicit `ST_BUILD_SHA`), boot UUID, K,
  drafter FC precision. A Git revision alone does not certify a dirty build.
- `st:condition_{steps,seconds,tokens,drafted,accepted}_total`: actual joint cohort,
  bounded sequence count, maximum starting context, request cache reuse and timing
  scope. Async residency can overlap and must not be compared with device burst
  time as the same quantity. Burst counts use completed iterations, not launches.
  Ghost/finished rows are excluded from cohort comparisons.
- `st:prefill_{computed_tokens,compute_seconds}_total`: chunks actually computed,
  with their host prefill span; cached prompt tokens do not inflate the numerator.
- `st:response_tokens_{sum,count,bucket}` and `st:response_finished_total`: completed
  chat choices, reasoning vs body length and finish reason. Body excludes tool
  calls. Failed/cancelled requests are not successful completion samples.

No request text, token IDs, or request identifiers enter these metrics. Histograms
are cumulative. A count of zero is retained as a measured zero.

## Validation

On the existing ost-97x CPU environment, using `stk-venv/bin/python`:

```sh
OMP_NUM_THREADS=1 python -m unittest tests.test_engine_diagnostic_metrics \
  tests.test_engine_serve tests.test_engine_runner_async tests.test_engine_burst_decode -v
```

Initial full focused suite: 227 tests, 209 passed and 18 skipped (57.149 seconds).
After strengthening ghost-row exclusion: four diagnostics cases plus eight runner
async cases passed (12 tests). See `final-tests.log`.

Deneb's isolated synthetic fixture separately passed scrape → persistence →
authenticated RPC and native UI rendering/30m–24h switching. Synthetic rates are
functional assertions, not a measurement of model performance.

Validation used no fleet queue, GPU benchmark, production restart or deployment.
Merging this host telemetry does not certify a GPU performance change.

PR merge gate: Prometheus HELP/TYPE and engine labels are emitted for every family. Response histogram buckets are numeric and close at +Inf. `exposition-tests.log` records 24 passing diagnostics and serving-metrics tests after the CI-discovered format correction.
