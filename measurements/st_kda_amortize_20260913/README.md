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

The owner also proves the alignment of every layer offset, slot stride and
ring position before capture. Passing that guarantee to the compiler enables
128-bit state loads/stores in the candidate; odd shapes retain their actual
smaller alignment. Flat and first-tile probe references disable this new hint.
The old scalar state writes are a concrete improvement opportunity, while
their share of the earlier ~2% regression still needs the GPU ablation.

The operator requested draft K=7 during this work. `facts.SPEC_K` is now 7;
the drafter, graph token width, state ring, and planning budget derive from
that fact. The measured kernel-shape descriptor remains a historical K=6
record rather than being relabeled as new evidence. New GPU timing uses K=7
(eight verified positions), and must not be compared directly to the old K=6
timing as if only the implementation changed.

The ST Oracle now also distinguishes the K=6 fleet width coefficient from
other widths. K=7 uses the component-derived width cost and labels it as an
estimate in JSON. Historical coefficient tests explicitly request K=6;
changing the production fact must not relabel those old measurements.

Integration includes main through `7fd05905` (#863, #870 and #871 were
already merged there): its DFlash serving policy is FP8/auto calibration
with acceptance diagnostics. Those are separate main changes, not gains
attributable to this KDA patch. A later consumer must record the resolved
calibration policy and pack identity along with K=7.

CPU validation: 28 layout, shape, deferred-state binding, execution-plan and
state-budget tests passed in 4.314 seconds. Six execution-order tests then
passed in 4.069 seconds after changing their decode rows to eight tokens,
including the four-rank CPU arithmetic oracle (not NIC/GPU proof).
Forty-two SM121 commit configurations compiled without a CUDA context.
Offline compilation now includes the same 16-byte pointer specialization as
the Batch allocation. Earlier resource figures without that specialization
are superseded: at eight tokens the flat/first-tile variants use 48/38
registers; aligned hoisted 1024/2048/4096-cell variants use 40/48/80 registers
and 1024 bytes of shared memory. The eight-warp 4096-cell variant uses 56
registers and no shared memory. All have zero stack/local memory. These are
resource costs, not assumed speedups. `compile.json`, `cpu.log`, and
`cpu-k7-execution.log` preserve the results.

The short GPU gate compares seven implementations at identical addresses,
FP32 factors and initial state: flat, the first tiled implementation, and
hoisted 1024/2048/4096-cell tiles, plus eight warps at 4096 cells and a scalar
1024-cell hoisted control to separate loop changes from alignment effects.
It covers C=1/C=4, accepted counts 1/3/8,
boundary/no-boundary and warm/64MiB-evicted cache. Order rotates and reverses;
restoring the entire initial arena is outside the timed region. Each variant
must match the flat reference byte-for-byte, including untouched padding.
GPU correctness also covers every count, rejected counts, large int64 contexts,
odd dimensions, ring wrap, rollback and four-iteration conditional graphs.

GPU results and the full-model consumer remain pending at this commit.
Production `deferred_kda=0` and FP32 state remain unchanged; the new tiled
implementation is confined to the existing explicit deferred experiment.
