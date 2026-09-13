# Boot memory guard repair, 2026-09-13

The run `st-decode-batch-consumer0913v2` (source `2461caa1`) failed during
boot qualification, before any onepass request. Rank 3 failed at
`warm kernels [1, 8, 64, 512, 4095]`; the other ranks stopped on its failed vote.
These are original failure ledgers, not measurements of the repaired engine.

| Rank 3, failing checkpoint | GiB |
| --- | ---: |
| Allocated | 56.640 |
| Reserved | 66.059 |
| Reserved minus allocated | 9.418 |
| Immediately free | 7.146 |
| MemAvailable | 24.880 |
| Previous immediate-free floor | 8.000 |
| Updated immediate-free floor | 7.000 |

The recorded page reclamation was zero. The 9.418 GiB reservation gap is not
a claim that all those bytes are releasable. The copied `memory-rank3.json`
came from srv4's dump; `memory-rank0.json` came from the head's collection.
The original dump directory was
`/home/choiceoh/glm53-logs/st-bracket-dumps/st-decode-batch-consumer0913v2-decode-batch`.

## Changes

- Return inactive CUDA allocator blocks at prefill/kernel warmup boundaries
  and final readiness, before the physical-memory guard. Other boot checkpoints
  attempt this when their immediate-free floor is threatened.
- Keep live tensors, graph ownership, the allocator's 12 GiB workspace ceiling,
  cumulative peaks, and the TP failure vote. A release failure also casts the vote.
- Record reservation and immediate-free headroom before reclamation, actual
  allocator bytes returned, separate host/device free bytes afterwards, and the
  failure reason in each memory ledger row.
- Lower the GLM boot OS margin from SIGTERM + 2 GiB to SIGTERM + 1 GiB
  (8 to 7 GiB on this fleet). Host earlyoom configuration is unchanged;
  SIGTERM remains 6 GiB and SIGKILL 4.5 GiB. The temporary anonymous-page
  reclamation operation retains its separate 2 GiB safety margin.
- Bound prefill memory qualification and short-kernel warmup by the configured
  served context. The failed boot qualified a 32,256-token chunk at context
  973,056. An explicitly smaller serving ceiling now bounds this preparation,
  too; the default production serving ceiling is not shortened.

## Validation

112 distinct CPU tests passed; 44 tests requiring CUDA or model fixtures were
skipped. The first 156-test invocation had one source-copy fixture error:
`probes/run_engine_probe.sh` was absent from the isolated test directory.
After copying the unchanged `probes/` directory, that one test passed.
Both the initial output (`cpu-regressions.log`) and the correction
(`cpu-fixture-recheck.log`) are retained.

The tests cover release-before-guard ordering, unchanged cumulative peaks,
rejection when ownership prevents reclamation, failure voting, context-bound
warmups, and replay of the recorded rank 3 numbers. That replay assumes **zero**
allocator reclamation: the 8 GiB floor rejects it and the 7 GiB floor passes.
This proves the guard decision on the recorded state, not a successful live boot.

Linux CPU tests used engine image
`sha256:3d94faa68269f643eeef37d69e02c290ef14201ea46dc0eb27e4ce66eeeb8e55`,
with the modified source mounted at `/work`, `--runtime=runc`, no network,
4 CPUs, 4 GiB memory, no swap, `CUDA_VISIBLE_DEVICES=` and
`NVIDIA_VISIBLE_DEVICES=void`. No GPU was exposed and no serving container
was restarted. The test modules were:

```text
tests.test_engine_runtime_memory
tests.test_engine_prefill_outputs
tests.test_engine_native_execution
tests.test_engine_budget
tests.test_engine_glm53
tests.test_engine_tripwire
tests.test_engine_boot_readiness
tests.test_engine_bootpaths
tests.test_engine_graph_resources
tests.test_engine_graph_contracts
tests.test_engine_knobs
```

A new full-model boot, the actual reclaimed GiB, and onepass throughput/quality
have not been measured for this change.
