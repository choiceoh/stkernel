CPU feedback during deployment preparation

Plans now save CPU job IDs and start their workers before checking the GPU
deployment. Agents can read those IDs from the saved plan and consume CPU
results through `result`, `wait`, or the session inbox while deployment
attestation is pending. Every GPU prerequisite and explicit CPU preparation
dependency still applies. A deployment error preserves the running CPU work.

The controlled comparison below isolates that ordering change. Both versions
use the same host, Python, temporary-repository fixtures, CPU checks, dependent
preparation command, resource pool, and worker implementation. The baseline
loads `experiment_plan.py` from merged revision
`69ea76f6d6edf704d5d55a7f9bf366a0d3f4fe84`; the candidate loads the current file.
Each workflow uses a fresh database and two uncached CPU jobs. Five alternating
rounds per version inject the same two-second delay into deployment attestation,
then return a controlled deployment error. CPU result latency is measured from
the caller's start to the actual result publication timestamp in the database.

| Median over five rounds | Baseline | Candidate |
| --- | ---: | ---: |
| First CPU result published | 2.65 s | 0.61 s |
| Both CPU results published | 3.55 s | 1.50 s |
| Plan call returns | 2.20 s | 2.17 s |
| CPU IDs visible in saved plan during attestation | 0/5 | 5/5 |

The injected delay is removed from the CPU result path. The synchronous plan
call still waits for deployment attestation; earlier feedback is consumed
through the saved IDs or inbox. These are CPU fixture measurements under a
controlled delay, not measurements of production GPU queues or serving speed.
No real fleet hold, deployment, SSH, or Docker daemon is used by the comparison.
The small sample describes this fixture, not production tail latency.

[Raw samples and exact plan-file hashes](comparison.json) and the
[reproduction script](compare.py) are included. From the repository root:

```sh
CUDA_VISIBLE_DEVICES='' python3 measurements/fleet_eager_cpu_20260907/compare.py \
  --output /tmp/fleet-eager-cpu-comparison.json
```

Validation passed all 87 fleet regression tests in two isolated shards on
macOS/Python 3.14, with exact coverage, zero failures, errors, or skips. Five
focused integration cases also passed on Linux/Python 3.12 with GPUs hidden:
CPU completion and persisted IDs before both deployment attestations,
independent preparation with a failing CPU gate blocking GPU execution,
explicit artifact dependencies, preserved CPU work after missing deployment,
and recovery from a transient worker-launch error. A job is marked launched only
after `ensure_worker` returns; the final launch pass can recover a failed start.
The new early-feedback regression fails on the baseline because the persisted
plan has no CPU submission IDs yet. The launch-recovery regression also fails
before the launch-bookkeeping fix. Python syntax and diff checks pass.

```sh
FLEET_CPU_SLOTS=2 CUDA_VISIBLE_DEVICES='' python3 bench/cpu_unittest.py \
  'tests/test_fleet*.py' /tmp/fleet-eager-cpu-tests.json --jobs 2
```
