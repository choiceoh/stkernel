Faster CPU dependency handoff

Completed checks and preparation stages now unlock the next stage with a 50 ms
poll interval. Each poll reads indexed job IDs/states together, without decoding
every dependency's pinned payload. Worker recovery uses its own one-second
cadence and skips completed jobs. Missing dependencies fail closed, incomplete
evidence is refreshed before judgment, and retirement exits before execution.

The matched comparison runs a real six-stage CPU workflow: one counted unittest
gate followed by five preparation commands. Each preparation consumes its
predecessor's verified artifact and produces the next value. The final artifact
must equal `5`; every job must pass without evidence-cache reuse. All source,
artifact, resource, and result checks remain active.

Five alternating rounds per version use one host and Python environment, with
fresh temporary repositories, databases, and CPU pools. Only
`bench/experiments.py` differs between the versions. The baseline is merged
revision `767469f613c2f8b1a5a366626f66c97ac4fbd14a`. Each variant submits and runs
with its own identical runner bytes, preserving the runner attestation guard.
There is no injected workload delay. The fleet boundary uses local fixtures;
no GPU, serving deployment, Docker daemon, or SSH is used by the comparison.

| Median over five rounds | Baseline | Candidate |
| --- | ---: | ---: |
| First CPU result | 0.74 s | 0.74 s |
| All six CPU results | 5.72 s | 2.26 s |
| Sum of five handoff gaps | 4.16 s | 0.71 s |
| Plan submission returns | 0.34 s | 0.34 s |

The median workflow completion time decreases by 60.5%. A handoff gap runs from
the predecessor's recorded completion to the successor's recorded start, so it
also includes verification, artifact transfer, and CPU admission. The unchanged
first result and submission timings isolate the benefit to the dependent stages.
This measures a short CPU workflow, not production GPU queue or serving latency.
The five-round sample does not establish production tail behavior.

Waiting workers now perform up to 20 state reads per second, batched in groups
of at most 256 IDs. Full payload decoding and worker recovery are excluded from
that fast loop. Database overhead under large production DAGs is not measured
by this small workflow.

All 92 fleet regressions pass on macOS/Python 3.14 in two isolated shards, with
exact coverage and no failures, errors, or skips. Seven focused tests also pass
on Linux/Python 3.12 with GPUs hidden: prompt dependency completion, throttled
recovery, incomplete-result refresh, batched state reads, retirement, failed CPU
gates blocking GPU execution, and verified artifact transfer. Two new behavioral
regressions fail on the baseline. Python syntax and diff checks pass.

[Raw samples and exact worker hashes](comparison.json) and the
[reproduction script](compare.py) are included. From the repository root:

```sh
CUDA_VISIBLE_DEVICES='' python3 measurements/fleet_cpu_handoff_20260907/compare.py \
  --output /tmp/fleet-cpu-handoff-comparison.json
FLEET_CPU_SLOTS=2 CUDA_VISIBLE_DEVICES='' python3 bench/cpu_unittest.py \
  'tests/test_fleet*.py' /tmp/fleet-cpu-handoff-tests.json --jobs 2
```
