# Reporting future tool evaluations

Keep quality and performance in the same candidate boot, and retain the original
CLI console output as well as the official machine-readable results.

`evaluate_with_perf.py` runs the pinned reference CLI, all 88 scenarios, and its
official `--perf` integration. It requires an existing canonical fleet hold for
the supplied candidate; it does not acquire GPU resources itself. It verifies
the hold owner and release mount, then releases only that session in `finally`.

```sh
python3 evaluate_with_perf.py --sha FULL_ENGINE_SHA --session OWNED_FLEET_SESSION \
  --out ABSOLUTE_ARTIFACT_DIRECTORY --protocol reference_t1
```

The fixed protocol is T=1, top_p=.95, seed=42, thinking=true, retain=false,
quality concurrency 1 and one trial. The official performance integration uses
pp=2048, tg=512, depths 0/4096/8192, concurrency 1/2 and three runs per cell.
Its sampling settings are passed explicitly through `--benchy-args`; quality
sampling flags alone do not establish the performance subprocess's settings.

Do not pass `--json-file` to this workflow: this CLI revision treats it as
machine-output mode and suppresses its original Rich console panels. The runner
instead exports the completed official SQLite row without changing its grades.
`stdout.log` is the original console output. `progress.jsonl` is the historical
stderr filename and is not necessarily JSONL in console mode.

Every report should include:

- Actual CLI performance table and final `Benchmark Complete` panel, with a link
  to the complete captured stdout and official Markdown report.
- Exact engine commit, all-rank identity, CLI and llama-benchy versions, sampling,
  input/output lengths, cache policy, concurrency and number of trials.
- Prefill/TTFT, decode throughput, and concurrent aggregate versus per-request
  throughput. Do not mix measurements from a different build or temperature.
- Preserved failures, partial results, infrastructure errors and metric limits.
  A rerender of saved results must be labelled as a rerender, not original stdout.

## llama-benchy 0.4.0 prefill limitation

The ST streaming endpoint sends an empty assistant role chunk before generation.
llama-benchy 0.4.0 computes individual prompt processing time from the first
response chunk minus estimated latency, rather than the first content token.
Consequently, its C1 `pp t/s` and C2 per-request `pp_req_throughput` are not valid
prefill speed measurements for this endpoint. Keep the raw values intact and
report actual `e2e_ttft` beside them. C2 aggregate prompt throughput uses a
different first-token denominator; it cannot be compared to the faulty C1 value
as a scaling ratio. TTFT includes queueing, prompt work and the first generation
step and is not a pure kernel prefill timer.

Sources for this finding: the measured engine's `engine/base/serve.py` streaming
role chunk, llama-benchy 0.4.0 `client.py` first-response/first-token distinction,
and `results.py` individual versus batch throughput formulas. The evaluation
does not patch the official benchmark or alter the server to improve this table.
