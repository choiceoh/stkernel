# Captured decode replay lifetime fix — 2026-09-12

The full GLM-5.3 target graph stalled on its first replay because the
functional MoE workspace cache replaced allocations still referenced by
captured CUDA graphs. Retaining the workspace owners until graph teardown
removed the reproduced stall on all four GB10 ranks.

## Controlled reproduction

- Baseline: `b7a6e44c870f7f134499801a943deb00ead45573` (merged PR #565).
- Hardware: four GB10 nodes, real NCCL TP4 over IB/RoCE; rank order
  srv2, srv1, srv3, srv4. No RTX 5050 measurements.
- Runtime: `st-engine:9391`, native ST kernels, no vLLM import or overlay.
- Weights: `/home/choiceoh/models/st-glm53-9391-up-gate-full`, all 45 layers,
  plus `/home/choiceoh/models/GLM-5.3-Flash-DFlash2`.
- Both A/B runs: KV 2 GiB, max sequences 4, context ceiling 4096,
  prefill memory warmup 128, 24 generated tokens, graph-family synchronization
  enabled solely to locate the stalled boundary.
- The candidate changed only external workspace ownership in the graph
  capture path. No kernel arithmetic, collective recipe, sampling rule,
  eager fallback, or per-replay synchronization was added to the engine.

The capture warmups allocated static MoE workspaces of 48, 96, 144 and 192
routed rows. Previously each growth removed the prior owner. The first real
prefill grew the cache again to 240 rows. The CUDA graph still contained the
old addresses, including grid barrier counters and TMA scratch storage.

| Observation | Baseline | Fixed |
|---|---:|---:|
| First drafter proposal completed | 4/4 ranks | 4/4 ranks |
| First target replay completed | 0/4 ranks | 4/4 ranks |
| Completed target replays per rank | 0 | 6 |
| Target time, synchronized diagnostic | stalled | 67–70 ms |
| Small workspace owners freed | during later capture warmups | after graph teardown |

The baseline GPU utilization was 96% on every node. The captured process
stack was waiting in the target graph's completion synchronization. We
collected the evidence and stopped only this run's four private containers.
See [baseline-stall.txt](baseline-stall.txt), `full-baseline-rank*.log`,
`full-fixed-rank*.log`, and [comparison.json](comparison.json).

**The fixed A/B run is a replay result, not a complete quality PASS.** Its
initial probe used the checkpoint's default template, which entered thinking
mode. Generation completed 24 tokens of coherent reasoning and then the
probe's Seoul-string assertion failed because that budget ended before the
answer. The final probe uses the same explicit template as the serving boot.

The smaller controls did not reproduce the full-model stall: the weight-free
TP4 lifecycle test passed on all ranks, and a two-layer real-kernel run also
completed despite the workspace owners being freed. Those controls do not
establish full-model replay safety.

## Ownership contract

`DecodeGraphs(resources=...)` snapshots external workspace owners after each
capture, before another shape can grow the eager cache. Owners are deduplicated
within the graph family. `close()` resets all CUDA graphs before releasing
these references, including failure during later capture/memory qualification.

The GLM lane table exposes `cached_workspace_owners()` from the MoE dispatcher.
The ordinary eager cache retains its existing growth policy; the graph family
owns the older generations its kernels still need. Graph inputs, outputs and
the existing separation between target/drafter/sampler pools keep their roles.

## Regression checks

- [CPU suite](cpu-tests.log): 265 tests, 188 passed, 77 skipped for unavailable
  GPU/checkpoint requirements. Includes three new lifetime tests: cache growth
  after capture, shared-owner deduplication, and failure teardown ordering.
- `probes/engine_graph_lifecycle_check.py`: 36 synthetic target graphs with
  90 all-reduces per target, interleaved eager collectives, drafter top-k,
  greedy and stochastic sampler graphs; four ranks passed.
- `probes/engine_decode_replay_check.py`: real 45-layer runner, target and
  DFlash graph families. `--trace` adds diagnostic synchronization; omit it
  for the normal asynchronous serving path.

Final serving-sized validation uses the following arguments in four private
containers with the same rank/network setup:

```sh
python3 -u /repo/probes/engine_decode_replay_check.py \
  --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full \
  --metadata /metadata \
  --drafter-dir /home/choiceoh/models/GLM-5.3-Flash-DFlash2 \
  --full --kv-gib 8.73 --ceiling 1048576 --prefill-warmup 6912
```

The multi-request phase injects a scheduler clock past the existing starvation
deadline and suppresses EOS until the test budget, so actual decode widths
1, 2, 3 and 4 must all execute. The probe asserts those widths and the Seoul
answer; `--trace` is absent. Memory qualification uses the full legal prefill
chunk and both ends of KV capacity.

**Result: all four ranks passed.** Each captured 36 target graphs and completed
14 decode steps over widths 1, 2, 3 and 4, finishing five requests and 99 tokens.
The token ID arrays for every request were identical across ranks. The ordinary
single request generated `[26612, 139966, 154827]`: `서울` plus the configured
end token. The four-request stress phase deliberately held off end tokens to
keep growing batches alive; its forced continuations are not a quality score.

See `full-serving-rank*.log` and the `full-serving` row in
[comparison.json](comparison.json). This exercises the real runner and graph
families; it does not claim an HTTP deployment, long-context answer quality,
or a sustained throughput benchmark. All nine capacity buckets were captured;
the actual short prompts selected the 4096 bucket.

All 159 preparation checkpoints per rank passed. Peak allocator reservation
was 61.578 GiB; peak workspace above the declared arena was 5.575 GiB, below
the 12 GiB workspace ceiling. See [memory-summary.json](memory-summary.json)
and `memory-rank*.json` for each checkpoint. Five external workspace owners
were held by the target family (the warmed prefill workspace and four static
decode generations); repeated capacity buckets did not duplicate ownership.

The private containers were removed and the run's fleet lock was released.
