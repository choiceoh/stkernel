# GB10 execution plans

Three independent experiments for GLM53 on four GB10s. Default execution and
FP32 KDA storage remain unchanged. These are not measured speedups.

| Arm | Decode TP overlap | Early DFlash2 observe | Prefill window |
| --- | --- | --- | --- |
| baseline | off | off | one native tile |
| tp-overlap | C=4 as 2+2 | off | one native tile |
| early-observe | off | on | one native tile |
| layer-prefill | off | off | two native tiles |
| combined | C=4 as 2+2 | on | two native tiles |

## Ordered compute and communication

`execution.decode_overlap` captures one target graph. Compute stays on one
stream because the existing native kernels share scratch. A second stream
reduces the first group's output while the second group computes. Every rank
enqueues collectives in `(layer, attention/ffn, group)` order. Each reduction
owns its input until the NIC and consumer finish. Events protect consumers and
join both streams before graph capture returns.

The full batch still samples and commits once, using the existing RNG and row
lifecycle. C=1/2/3 keep their original target batch. C=4 changes GEMM row shape
and doubles per-layer reduction count; communication overlap may fail to pay
for those costs. Numerical CPU agreement alone is not adoption evidence.

## Early DFlash2 context preparation

Target auxiliary layers are zero-based `[4, 13, 23, 32, 41]`. Once layer 41
finishes, the ordinary context projection can run while target layers 42--44
and the vocabulary head finish. Proposal generation still waits for its anchor.

The native W4 kernel originally used process-global partial sums and arrival
counters. The early FC projection owns a separate initialized workspace;
the same packs, K-slice order and arithmetic use those pointers. Shared-MLP
pair state and low-rank correction are not admitted on the private entry point.
The existing fused `write_context` commits only the final retained count,
including EOS/limit trimming. Calibration observes these committed rows once;
tentative projection never updates calibration or the drafter ring.

The target graph rejoins the observation stream before returning. Prepared
context lives in the target graph pool and is consumed before its next replay.
The commit/calibration writer has a separate captured graph pool, so moving
the projection does not introduce eager compute dispatch after sampling.
Synchronous rich sampling also consumes the prepared context.

## Layer-major prefill

A scheduler window contains two or four existing 9,216-token tiles. At each
layer all tiles run attention, then all tiles run FFN, in token order. This
retains the existing per-tile kernel shapes, SP transport and numeric guards
while improving the opportunity to reuse that layer's weights in the 24 MiB
GB10 L2. No claim is made that a layer or its experts fit in L2.

Only the idle-prefill budget grows. With live decoders the 2,304-token prefill
budget and alternating decode step remain unchanged. Prefill and decode never
share a model step. The existing 12 GiB runtime workspace ceiling still applies;
additional carries and auxiliary features must fit it during qualification.

Per-layer KDA rings advance in token order. Interior marks use the existing
kernel outputs. A mark at a tile end is copied before that layer's next tile
overwrites its ring. The runner publishes a prefix only when the whole window
finishes, so publication may be delayed relative to one-tile execution.

## Selection and evidence

Non-production knobs expire on 2026-09-30:

```
STK_execution_overlap=1
STK_early_observe=1
STK_prefill_tiles=2
```

Production rejects knob overrides and fixes all three experiments off. The
selected plan is included in `st:lane_info`; private workspace bytes are
reported at capture. All experiment combinations require FP32 KDA state.

CPU tests compare real reference KDA/DSA paths: changed C=4 grouping, physical
request order, resumed prefill, output/auxiliary values, state/KV bytes and
prefix snapshots. The focused CUDA gate exercises concurrently running native
W4 GEMMs with distinct inputs and repeated changed-input graph replay.

The canonical fleet gate is a committed-arm `st-chain` with a same-code
baseline and each experiment separately, followed by the combined arm only
when individual arms qualify. Following current D17, each boot runs C=1 twice
and C=4 once over 2K/32K/128K, with prepared kernels, fresh prefixes and
separate diagnostics. Keep a same-code baseline even if a historical floor
can be borrowed: this change also modifies the default W4 kernel ABI layout.
Judge actual per-request TTFT/ITL, output tok/s, acceptance, quality, verbosity
and workspace peaks. Do not infer an engine win from compilation or the
focused scratch test.

## PR760 simulation and CPU results

`python3 measurements/st_execution_plans_20260913/simulate.py` reuses the real
`bench.step_sim` Runner with the production alignment (2,304), baseline token
budget (10,240), K=6, four resident rows and the live-decoder prefill budget.
`simulation.json` retains 54 host-only runs: C=1/C=4, 2K/32K/128K, 512 generated
positions, three alternating-order repetitions. Acceptance 0.5 is an explicit
workload assumption; no language is generated. GPU cost is zero.

| Prompt | C=1 prefill steps, 1/2/4 tiles | C=4 prefill steps, 1/2/4 tiles |
| --- | --- | --- |
| 2K | 1 / 1 / 1 | 4 / 4 / 4 |
| 32K | 4 / 2 / 1 | 46 / 44 / 43 |
| 128K | 14 / 7 / 4 | 182 / 175 / 172 |

The smaller live-decoder budget dominates C=4 after the first prompt enters
decode. At 128K/512 generation, four submitted requests never produce a width-4
decode step in this simulation: earlier requests finish before all prompts
enter decode. The overlap arm must therefore be judged against actual width-4
step counts, not the client's concurrency label. Host-only medians are
0.003--0.006 ms/decode step here; these omit CUDA dispatch and network work.
Fewer prefill steps do not by themselves establish faster GPU prefill.

`historical-replay.txt` re-derives PR760's saved ST record. `historical-fit.txt`
runs the existing fit-and-validate path against that same old record, preserving
its K=5. The six unfitted prediction rows have at most 5.9% residual in this
run. Fitted step rate and prefill throughput are inputs, not independent
predictions. This older record lacks a git revision and verified prefix-cache
control; it is a simulator exercise, not the baseline for these candidates.
Neither PR760's model nor this harness predicts subgroup GEMM latency, stream
contention or L2 reuse; those fields remain explicitly unmodeled.

The PR760 tool suite passes 23 tests. Reference execution tests include real
four-rank LocalTP reductions, row remapping, mixed KDA/DSA, resumed prefixes,
patched embeddings and byte-exact state/KV/snapshot comparisons. SM121a CUDA
compilation passes without creating a device context (`compile.json`), with a
3,343,616-byte private workspace.

The initial 1,095-test CPU pass (`cpu-full.txt`, ccec5bd84) found a supervisor
fixture timeout unrelated to these engine changes. Main subsequently fixed
the fixture; after merging that fix and the rank-consistent one-shot sums,
the 124-test integration slice passes (5 CUDA skips). Subsequent focused
results and fleet qualification are recorded separately below.

After adding the captured commit writer and exact subgroup-capacity checks,
77 focused tests pass (5 CUDA skips). `arms.json` names the same-code baseline
and one-fact-only measurement branches; `make_arms.py` reproduces those trees
without modifying the current checkout. They are experimental arm commits,
not production-default changes.

`st-gb10-scratch0913v2` passed the native private-workspace GPU comparison:
40 changed-input graph replays, exact outputs and rearmed counters. The
payload subsequently exited 1 while reading relative source paths for its
JSON report. `gpu-scratch.log` preserves that complete result and
`gpu-scratch-recovery.json` explicitly records the distinction. The report
now resolves paths from `__file__`, checked on CPU from another directory;
the numerical test was not repeated for this reporting-only fix. This
qualifies private W4 scratch, not the full target/drafter execution order.

The first full-bracket admission stopped before taking GPUs because overlay
composition changed the stale tracked `build/glm53/glm53_chat.py`. Regenerating
it with `launchers/compose-overlays.sh glm53` incorporates main's existing
max-to-high normalization and makes the controller checkout reproducible.
The replacement arm set includes that generated-file correction. No full
onepass or serving speed verdict exists yet.
