# Recurrent graph-state transfers — 2026-09-11

**Historical proposal, superseded during this task.** PR #549 reached main
with a better direct-state architecture that also removes the full scratch
rings. The final branch preserves that implementation and removes this
proposal's production kernel and adapter changes. The measurements below
belong to `7068978c`, not the current main implementation. Historical probes
are under `historical/`; they require that measured checkout.

GLM target graphs now gather only the KDA predecessor state and commit only the
states produced by this step. The transfer workload improves substantially;
the shared-fleet measurements do **not** establish an overall TP4 or full-model
speedup. All raw runs, including noisy and negative comparisons, are retained.

Implementation: `7068978c13f064c23f1ee2de3c0a1e0356c3b912`.
Baseline: main `b18c97f4`, after PR #547/#548. The paired probe imports an exact
`git show b18c97f4:engine/profiles/glm53/decode_graphs.py` export; its SHA-256 is
recorded in every paired report. `summarize.py` checks the measured engine Python
hashes against that immutable Git revision, the runtime manifest, and evidence completeness.

## Change and preserved contracts

- `GraphCaches` binds the captured contexts and token count. For each recurrent
  ring it gathers `(context-1) % 6`, and commits `context ... context+tokens-1`.
  The former `index_select`/`index_copy_` pair moved all six states each way.
- `engine/kernels/state_cache.py` copies contiguous state tiles through the
  arena's explicit physical-slot stride. Device slot/context values remain
  replay inputs. No CPU readback, runtime tuning switch or fallback was added.
- Unwritten scratch positions are unspecified. `_kda` reads only the gathered
  predecessor; commit never copies other scratch positions into the arena.
  Existing rejected-future history, inactive owners, padding and drafter state
  remain intact. Convolution and indexer-tail handling is unchanged.
- State values are copied in their existing FP32 form. Model arithmetic,
  rounding, ring layout, speculative width and cache ownership are unchanged.
- Scratch allocation still reserves the full ring shape so the model's existing
  positional indexing remains valid. This is a **traffic reduction**, not a
  claim of smaller graph workspace or arena allocation.

A local KDA state is 16 × 128 × 128 × FP32 = 1 MiB. Including reads and writes,
one sequence/layer now transfers 4 MiB instead of 24 MiB for one-token decode
(83.33% fewer algorithmic bytes), or 14 MiB instead of 24 MiB for six-token
verification (41.67% fewer). These byte counts are not hardware DRAM counters.

## Transfer measurements

Three independent processes, five alternating AB/BA rounds per process, 30
samples per variant/round. Both variants use the real GLM arena strides and
the same data. Warmup and correctness comparisons are outside timed spans.
The 34-layer case allocates the actual complete-model cache layout but executes
**only recurrent transfers**, with no model weights, attention, MLP or network.

The times below are the median across processes of each process's **sum of
separately measured gather and commit medians**. They are not a whole-step time.

| KDA layers | Active sequences | Tokens/sequence | Baseline | Optimized | Reduction across the three processes |
|---:|---:|---:|---:|---:|---:|
| 34 | 1 | 1 | 3.730 ms | 0.724 ms | 80.54–80.79% |
| 34 | 1 | 6 | 3.723 ms | 2.203 ms | 40.71–41.08% |
| 34 | 2 | 1 | 7.304 ms | 1.284 ms | 82.40–82.42% |
| 34 | 2 | 6 | 7.298 ms | 4.258 ms | 41.65–41.73% |
| 34 | 4 | 1 | 14.535 ms | 2.500 ms | 82.73–82.83% |
| 34 | 4 | 6 | 14.524 ms | 8.410 ms | 42.04–42.10% |

Sources: `transfer.json`, `transfer-2.json`, `transfer-3.json`, `summary.json`.
Single-layer cases are also retained. Their working set can fit in cache, so
their absolute times should not be substituted into the complete-model table.

## Real-weight graph comparisons and limits

The paired probe loads real rank weights for layers 0 and 3, including KDA,
dense MLP, DSA, NVFP4 MoE, embedding and vocabulary head. Each variant captures
the same forward path and differs only in `GraphCaches`. Actual 256-token
prefill chunks generate the initial states/KV, including the 32K context.
The inputs and state/KV snapshots are identical between variants; reset copies
are queued **outside** each timed span. Graph samples are queued in batches
before synchronization to reduce host entry jitter.

Three independent isolated-rank processes, five AB/BA rounds, 30 samples/round:

| Context | Tokens | Observed graph-time reduction across processes |
|---:|---:|---:|
| 256 | 1 | 1.38–1.79% |
| 256 | 6 | 1.00–1.26% |
| 32,768 | 1 | −2.63–1.74% — mixed, not qualified |
| 32,768 | 6 | 1.11–1.56% |

`isolated-paired-{1,2,3}.json` executes rank 0 arithmetic with identity
collectives. It is neither TP4 execution nor full-model ITL. A context of
32,768 plus the new token(s) selects the existing 65,536 capacity bucket.

TP4 was measured on all four nodes in six independent fleet starts. Runs 1–3
used two CPUs/container; run 3 showed large excursions, including steps over
30 ms. Runs 4–6 used four CPUs/container and retained cgroup CPU counters;
timed batches recorded **zero CPU-throttled microseconds**. Their improvement
ranges across runs/ranks still cross zero (roughly −2.38% to +3.01%). The cause
of the variance was not established. No positive-only filtering is applied,
and this dataset does **not qualify an overall TP4 speedup or regression**.
These were not exclusive fleet runs. Post-run lifecycle inspection also found
the existing GLM service exiting and being recreated during the measurement
period: rank 0 exited with code 0 at 12:20:28 UTC and a new container started at
12:26:03 UTC. `service-events.json` preserves the available daemon events. The
cause was not established; this dataset must not assume stable service residency
or attribute the timing variance to one specific cause.

## Baseline attribution and next targets

`engine_graph_profile.py` first times an uninstrumented graph, then a separate
graph containing external CUDA timing events around nested stages. Child times
are subtracted from parents so the recorded stage values are disjoint. The
instrumented/uninstrumented ratio is retained for every condition. These are
diagnostic event spans, not Nsight hardware utilization or roofline results.

- Isolated baseline: recurrent/cache gather+commit around 106–112 µs for this
  two-layer slice, about 2.5–3% of its graph time. Vocabulary projection was
  around 1.2–1.3 ms and is performed only once per step; two-layer proportions
  cannot be extrapolated to the complete 45-layer chain.
- TP4 baseline: collective spans, **including rank arrival/wait**, around
  0.9–1.4 ms per slice step. These do not measure network bandwidth alone.
- The separate 256-token eager prefill profile records about 15.1/16.1 ms on
  rank 0 at context 0/32K. Its collective spans are about 4.6/5.7 ms, KDA about
  2.8/2.6 ms, and MoE about 2.6/2.1 ms. Eager event spans include CPU dispatch
  gaps; their interpretation differs from captured-graph kernel time.

The next measurement priority is communication and rank waiting under an idle
fleet window, followed by long-context indexer gathers and KDA intermediate
state/layout traffic. Additional tuning of small pooling arithmetic is lower
priority. The benchmark dispatcher now refuses active serving/ST containers
and acquires the launcher's shared `/home/choiceoh/st-fleet.lock` exclusively.
It releases only its own lock, after every rank has exited. No service is
stopped by the dispatcher.

## Correctness and environment

- `gpu-tests.log`: **145 GPU-enabled engine tests passed, zero skips**.
- `focused-tests.log`: three new tests cover nonzero arena offsets, padded slot
  strides, masked final tiles, predecessor selection, poisoned unwritten
  positions, remapped/recycled slots, rollback, inactive owners and draft state.
- `st-state-graph-check-f4d7-rank{0,1,2,3}.log`: **17 real-weight conditions on
  every rank**, including two active sequences, 1/6 tokens, context boundaries,
  reversed slots and rejected-future overwrite. Eager/graph relative difference
  was zero; state and paged bytes matched exactly.
- All six paired TP4 starts and all three isolated starts compare hidden,
  logits, full state and paged bytes exactly before timing every condition.
- `cpu-tests.log`: 145 discovered, 87 passed, 58 skipped because the local
  environment lacks PyTorch/CUDA; the GPU suite above covers those paths.
- `runtime-final.json`: pinned native ABI, SM121 GB10, CUDA 13.0,
  torch 2.13.0+cu130, Triton 3.7.1, all 879 DeepGEMM provenance files verified,
  no vLLM in the validation image. `environment.json` records per-node image
  IDs, NCCL configuration, resource limits and post-run GPU/service snapshots.
- Optional Compute Sanitizer was unavailable in the image (`memcheck.log`).
  The initial final-run container therefore exited 127 **after** the GPU suite
  passed; the remaining benchmarks ran successfully in a separate container.
  No sanitizer-clean result is claimed.
- Only bounded private containers were used (4/8 GiB for focused/other probes).
  Forty-one finished private containers and the task's launch lock were removed
  (`cleanup.log`). Cleanup targets only those named containers and the owned
  lock; it does not delete serving containers or rank files.

This work does not validate full-model generation quality, DFlash2 acceptance,
NVMe interference, throughput, final fleet admission or full-model ITL.

## Reproduction

Restore the measured revision and copy the historical probes into its `probes/`
directory before reproducing this superseded proposal:

```bash
git worktree add /tmp/st-state-706 7068978c
cp measurements/st_engine_graph_state_20260911/historical/*.py /tmp/st-state-706/probes/
```

Export the baseline, mount that checkout at `/repo`, native cache at `/cache`,
rank files at `/ranks`, metadata at `/meta`, and a writable `/evidence`:

```bash
git show b18c97f4:engine/profiles/glm53/decode_graphs.py > st-graph-state-baseline.py
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
python3 probes/engine_state_transfer.py --ckpt-meta /meta --output /evidence/transfer.json
python3 probes/engine_state_cache_bench.py --ranks /ranks --ckpt-meta /meta \
  --baseline /repo/st-graph-state-baseline.py --output /evidence/paired.json
python3 probes/engine_graph_profile.py --ranks /ranks --ckpt-meta /meta \
  --baseline /repo/st-graph-state-baseline.py --output /evidence/baseline-profile.json
python3 probes/engine_graph_profile.py --ranks /ranks --ckpt-meta /meta \
  --prefill-only --contexts 0 32768 --samples 20 --output /evidence/prefill.json
```

Repeat paired/transfer commands in three independent processes. In an idle
fleet window, `fleet_probe.py start --name NAME --program probes/engine_state_cache_bench.py
--baseline /repo/st-graph-state-baseline.py` dispatches the four ranks from srv1;
use its `wait`, `logs`, and `remove-finished` modes with the same name. The fixed
source/metadata/rank/cache mounts are listed in the dispatcher. Once collected,
run `python3 measurements/st_engine_graph_state_20260911/summarize.py` locally.
