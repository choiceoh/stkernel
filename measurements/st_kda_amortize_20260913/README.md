# KDA tile follow-up and operator-selected K=7

The first tiled commit passed all five GPU state/graph tests but had mixed
latency at K=6: roughly +2% for one accepted token or three accepted tokens
crossing a prefix boundary, versus -7% for three without a boundary and up
to -19% for seven. `prior-k6-gpu.json` contains all raw samples from the
admitted `st-kda-cursor0913v2` run on `9ad895621d92ef562a12a75efd35b1246e899f97`.
This is a component result, not a consumer speed or acceptance measurement.

The follow-up keeps factor broadcasting and removes avoidable fixed work:
one ring modulo locates the initial and final states, the final write is
outside the recurrence, and only intermediate prefix snapshots write inside
it. The original FP32 multiply-then-FMA order, accepted prefix, and untouched
arena bytes remain part of the gate. Increasing the cell tile shares the
metadata cost across more state values. No runtime acceptance-dependent
kernel selection is introduced.

The operator requested draft K=7 during this work. `facts.SPEC_K` is now 7;
the drafter, graph token width, state ring, and planning budget derive from
that fact. The measured kernel-shape descriptor remains a historical K=6
record rather than being relabeled as new evidence. New GPU timing uses K=7
(eight verified positions), and must not be compared directly to the old K=6
timing as if only the implementation changed.

CPU validation: 28 layout, shape, deferred-state binding, execution-plan and
state-budget tests passed in 4.314 seconds. Thirty-six SM121 commit configurations
compiled without a CUDA context. At the 1024-cell tile, final-store hoisting
reduces physical registers from 48 to 40; the 2048/4096-cell variants use
64/128 registers; a separate eight-warp 4096-cell configuration uses 106.
All have zero local memory, stack and shared-memory usage.
The latter is a resource cost, not an assumed speedup. `compile.json` and
`cpu.log` preserve these results.

The short GPU gate compares six implementations at identical addresses,
FP32 factors and initial state: flat, the first tiled implementation, and
hoisted 1024/2048/4096-cell tiles, plus eight warps at 4096 cells.
It covers C=1/C=4, accepted counts 1/3/8,
boundary/no-boundary and warm/64MiB-evicted cache. Order rotates and reverses;
restoring the entire initial arena is outside the timed region. Each variant
must match the flat reference byte-for-byte, including untouched padding.
GPU correctness also covers every count, rejected counts, large int64 contexts,
odd dimensions, ring wrap, rollback and four-iteration conditional graphs.

GPU results and the full-model consumer remain pending at this commit.
Production `deferred_kda=0` and FP32 state remain unchanged; the new tiled
implementation is confined to the existing explicit deferred experiment.
