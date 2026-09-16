# CUDA boot failure: deployment configuration and local expert preparation

## Confirmed findings

The failed automatic deployment used `023af8230e80ad061b11f121915419f3abb8d528`.
It started at 08:01 KST on 2026-09-16, before this task's candidate boot.
Rank 1 (srv1) reported `CUDA error: unspecified launch failure` at 08:04:33;
the kernel journal recorded Xid 43 at the same time. The containers were not
reported as Docker OOM kills. This does not exclude every possible memory fault.

Three independently saved failed boots have the same communication signature:

| Forensics directory | First stalled rank | Sequence | Missing peers | Per-rank scale fallback |
| --- | --- | --- | --- | --- |
| 20260916-063455 | 1 / srv1 | 1183 | all three (`0x7`) | ranks 0, 2, 3 only |
| 20260916-071938 | 1 / srv1 | 1183 | all three (`0x7`) | ranks 0, 2, 3 only |
| 20260916-080441 | 1 / srv1 | 1183 | all three (`0x7`) | ranks 0, 2, 3 only |

The final boot's complete supervisor forensics are retained in `cuda-forensics/`.
Rank 1's transmit and acknowledgement counters were both 1183, while the three
received flags for that ring slot remained at 1179. All ranks had announced the
packed M16 expert kernel; the other ranks had not announced the M16 raw fallback
kernel. The one-shot kernel contains a bounded-spin `__trap()` after its stall
deadline. Event/allocator destruction errors after this are secondary evidence,
not an identification of the original failing kernel.

### A confirmed deployment configuration mismatch

The installed `st-glm53.service` read `/home/choiceoh/.config/st-glm53.env`.
The installed `st-deploy-watch.service` had no `EnvironmentFile`.

The failed automatic deployment therefore used launcher defaults:

- `/home/choiceoh/models/st-glm53-nvidia-tp4-9391`
- no explicit `--production` or `--kv-gib 7.0`

The supervisor's successful recovery used the configured production values:

- `/home/choiceoh/models/st-glm53-9391-up-gate-full`
- `--production --kv-gib 7.0`

This explains why deployment and recovery entered different weight/scale and
memory configurations. It is a confirmed bug independent of the exact CUDA
instruction that triggered the final device error.

## CUDA mechanism and scope of the fix

The logs support a rank-asymmetric first-use loading problem: only three ranks
need the raw-scale expert variant, while rank 1 can advance to the next
collective. CUDA module loading can synchronize a context and deadlock with
work that is waiting for another kernel. See NVIDIA's
[lazy loading documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html).

The old death notes did not retain Python tracebacks, so the exact blocked API
on the three peers was not captured. The module-loading mechanism is an
evidence-backed hypothesis, not a completed instruction-level diagnosis.

Commit `b7d577cf131fe0c59ff90b9a23a0f9d0f28a3534` prepares every bound expert
locally at every declared decode row width before target graph capture. This
runs both packed and raw fallback layers when present, without any model TP
collective. Device work finishes before a host Gloo rendezvous; the preparation
group remains alive through this phase. The existing stall deadline is unchanged.
Death notes now retain the original exception traceback through cleanup errors.

Commit `7eae2621` makes automatic deployment read the same environment file as
the supervisor. A matching systemd drop-in was installed on srv2 at 08:27 KST;
`daemon-reload` applied it without restarting the service or interrupting a GPU
job. The receipt is retained as `deploy-env-repair.json` on srv2.

## Validation

- Initial runtime-image regression pass: 76 tests, OK, 3 GPU-only skips.
- Final preparation-group and capture regressions: 62 tests, OK, 3 GPU-only skips.
  This includes real two-process Gloo tests and a delayed fallback worker.
- Deployment watcher suite: 61 tests, OK.
- The bundled candidate reached API readiness at 08:40:46 KST. All four ranks
  passed the production memory gate and the new `decode experts` phase. That
  phase took 0.1185 / 0.1006 / 0.1003 / 0.1061 seconds on srv1 / srv2 / srv3 /
  srv4. The full boot records are captured in `raw/boot-evidence/`.
- The 08:45:53 capture confirmed 337 engine files with identical expected
  SHA-256 `33193c28d12347db1452b1a8dc56463ab521550b2f21814e314fe49965b6ea10`
  on all four ranks, with no CUDA/traceback/OOM/NCCL error signatures.
- The unchanged 88-scenario benchmark completed at 94/100 (166/176), with no
  infrastructure errors. This successful production
  boot uses `st-glm53-b12x-up-gate-v1`; it does not reproduce the old deployment's
  different NVIDIA-shard scale layout or establish the exact blocked API.

No benchmark scoring, failure exclusion, sampling temperature, or tool-call
postprocessing was changed to obtain a result.
