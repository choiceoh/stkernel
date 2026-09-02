# glm53_async_dflash

Async scheduling for the DFlash2 drafter, behind `VLLM_GLM53_ASYNC_DFLASH=1`.

## Why every dflash boot runs synchronously today

`vllm/config/vllm.py` resolves `async_scheduling=None` (the launcher passes no
flag) by an allowlist of speculative-method NAMES: eagle types, ngram GPU,
`draft_model`, `dspark`. `dflash` is not on it, so the config logs "Async
scheduling not supported with dflash-based speculative decoding and will be
disabled" and the engine runs the synchronous scheduler. Upstream main has the
same list (checked 2026-09-02).

The omission is by name, not by mechanism:

- `DSparkSpeculator` subclasses `DFlashSpeculator` and inherits `propose()`,
  the only speculator flow async scheduling touches. dspark is on the list and
  the dsv4 stack has served with `--async-scheduling` on it since August.
- The V2 model runner has no method-specific async branch (its request-state
  mirrors are optimistic upper bounds by design); the scheduler side is
  `AsyncScheduler`'s `[-1]` draft placeholders, which the worker overwrites
  with the real ids inside `combine_sampled_and_draft_tokens`.
- No mounted glm53 overlay reads draft ids on the scheduler side.

## What it is worth

The 2026-09-01 trace (rank 3, 229 steps) shows the GPU idle for 8.9 ms of a
72 ms profiled step (12%; nvidia-smi says 7% without the profiler): the host's
input preparation (~5.7 ms), the 1.43 ms `cudaGraphLaunch` of the 1,640-node
target graph, and the step turnaround. With the synchronous scheduler all of
it sits on the critical path. Async scheduling overlaps step N+1's scheduling,
preparation and graph submission with step N's GPU work, so the ceiling is the
whole idle share, 7-12% of the step. `glm53_prep_fused` shrinks the same host
time; the two are independent and can be armed together.

## What it changes

Two conditions in `VllmConfig.__post_init__` gain
`and not _deneb_dflash_async_ok(method)`; the helper returns True only for
method `dflash` with `VLLM_GLM53_ASYNC_DFLASH=1` and logs
`[async-dflash] ... whitelisted` once. With the knob unset the file behaves as
shipped. The whole file is mounted (preimage pinned in the manifest) because
the check runs in the front-end and engine-core processes before any model
module is imported, where no hook of ours exists.

## Effects to watch on the first boot

- `max_concurrent_batches` becomes 2 under async with the V2 runner, so the
  KV-cache in-flight reserve doubles: read the KV-cache line and the
  memfree preflight before comparing step/s.
- Structured-output requests copy draft ids back to the scheduler
  (`DraftTokensHandler`); plain generation does not.
- Gates: quality 9/9, Korean 0/16, pos-1 acceptance within 2 pct, then the
  C=1 step/s bracket (RUNBOOK EXP-8). Acceptance must not move: async
  scheduling changes when things run, not what is sampled.

## Arming

```
VLLM_GLM53_ASYNC_DFLASH=0   # default: stock verdict (synchronous)
VLLM_GLM53_ASYNC_DFLASH=1 bash launchers/start-glm53-nvfp4-tp4.sh   # profile-declared key: caller env, not EXTRA_ENV
```

Rollback is the env line. `ASYNC_SCHED=0` still forces the synchronous
scheduler regardless. Base contract from `glm53:v13-b12x`.
