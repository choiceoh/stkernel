# Fleet CPU turnaround, fair admission and unused reservations

The six-job CPU artifact chain completed in a median **2.267 s → 1.573 s
(30.6% less wall time)** over five alternating baseline/candidate rounds on the
same macOS host. Four deliberately heterogeneous fixture tests completed in
**1.286 s → 0.789 s (38.7% less)** when their recorded durations balanced two
shards. These are isolated CPU workflow measurements, not production GPU queue
latency, fleet p95, GPU numerical validation or serving throughput.

## Implementation

- Persistent FIFO admission prevents later one-slot jobs from overtaking an
  older multi-slot request. The pool can intentionally leave capacity idle while
  that request drains the pool. Dead, retired and over-capacity waiters are
  pruned; timeout/retirement cleanup removes only the job's own queue/lease rows.
- Definitive pair completion or retirement releases its synthetic baseline
  subscription. Only an unstarted, unheld baseline with no remaining subscribers
  or dependents is retired and dequeued. Shared/manual/incomplete demand,
  already-started work, unrelated queue entries and live holders remain intact.
  Workers check legacy orphan demand before preflight; stored results are kept.
- Complete fleet runs learn per-test timings. Subsequent runs use longest-first
  assignment to the least-loaded shard. History separates host/runtime/worker
  count and test-source hashes. Unusable history falls back to round robin;
  every shard and the parent independently verify exact test-ID coverage.
- CPU waiters poll at 50 ms; only the eligible head probes available RAM. Total
  physical memory is cached. RSS monitoring selects the owned process group
  (Linux session plus PGID filter) and process-exit waits return before the old
  fixed 100 ms delay. Existing time/RAM budgets and descendant cleanup remain.

## Measurement method

`compare.py` uses real subprocesses and the repository's private-checkout,
manifest, CPU-check, artifact-validation and dependency machinery. The fleet,
serving launcher and deployment identity are local fakes. No GPU is held or booted.
The first job executes one counted CPU test; five dependent preparation jobs
verify and extend an artifact from `1` to `5`. Both arms use fresh databases,
identical commands, budgets and fixture dependencies; no cache hit is accepted.
The helper records source hashes for the five runner files replaced between
baseline `bb123cfa1fc47e2b9e87fde3b55eb4400412785d` and candidate. Candidate hashes
match implementation commit `df20601`.

| CPU chain metric (median, five rounds per arm) | Baseline | Candidate |
| --- | ---: | ---: |
| All six results available | 2.267 s | 1.573 s |
| First CPU result available | 0.735 s | 0.689 s |
| Sum of five dependency handoff gaps | 0.707 s | 0.693 s |
| Plan command returns | 0.347 s | 0.347 s |

The chain gain is primarily shorter CPU command completion overhead. Its
handoff gaps and plan-return time changed little; this does not establish a
new dependency-polling gain. Each of ten runs verified all six results, one
executed check, final artifact `5`, zero cache hits and no fleet holder/boot.

`sharding.py` uses four actual unittest cases with deliberate service-time sleeps
of 0.60, 0.05, 0.55 and 0.05 seconds. After one untimed-for-comparison learning
run, five alternating rounds compare no history with the same fixed learned
profile. Both arms execute the same four IDs exactly once using two workers.
This fixture demonstrates scheduling under unequal case costs; it is not a
claim that the real fleet suite is 38.7% faster. Learning adds no extra test run
to normal operation: the first complete run records history for the next run.

Raw results: [CPU DAG](cpu-dag-macos.json), [sharding](sharding-macos.json).

## Validation

The fleet gate covers existing submission/evidence/retirement behavior plus 15
new tests, including real concurrent one-slot/two-slot workers, failed
supersession transaction rollback, preserving incomplete/shared/live baseline
demand, and rejecting a duplicate/missing test assignment before execution.
The real RAM-overrun and orphan-child cleanup regressions pass on macOS and
Linux; an unrelated process group survives cleanup.

See [validation receipt](validation.json) for the exact runs, executed IDs and
runner hashes. The cold macOS gate executed 106 cases; after adding the live
worker regression, the learned scheduler executed all 107 with 92 valid timing
hints. Linux also executes the complete gate in an isolated temporary clone
with GPUs hidden. Timing history is scheduling evidence only; skipped, failed,
missing or duplicated tests never satisfy the CPU gate.

## Reproduce

On macOS put modern Bash on PATH. Run each timing command without another local
benchmark concurrently; the helpers themselves alternate their two arms.

```bash
PATH=/opt/homebrew/bin:$PATH CUDA_VISIBLE_DEVICES='' \
  python3 measurements/fleet_runtime_20260908/compare.py --output /tmp/cpu-dag.json
CUDA_VISIBLE_DEVICES='' \
  python3 measurements/fleet_runtime_20260908/sharding.py --output /tmp/sharding.json
FLEET_EXPERIMENT_ROOT=/tmp/fleet-runtime-validation CUDA_VISIBLE_DEVICES='' \
  python3 bench/cpu_unittest.py 'tests/test_fleet*.py' /tmp/fleet-tests.json
```

Existing in-flight workers keep their original runner. Use the current runner
and one shared experiment root per host for FIFO participation and timing reuse.
No running production service or shared GPU reservation was changed for this work.
