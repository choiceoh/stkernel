# C1 MoE FC1 scale staging and scatter-cache reduction

Implementation: `3f69b9a8`, base `32fb2892`. The existing `t,r,sf6` C1
geometry enables this change at 1–8 rows, including K=7 verification at M8.
This is an enabled implementation with CPU/compiler evidence, not a measured
step/s or output tok/s improvement. No GPU job was queued or run.

## Mechanism

FC1 previously DMA-loaded each packed SF6 block into its expanded scale
buffer. The 128 MMA threads needed a barrier after retaining the packed
bytes, then another after writing expanded bytes. The new packed FC1 ring
is disjoint from every expanded scale stage. It removes the first barrier
and retains the publication barrier. Each pipeline slot owns both buffers
until consumer release. The integer reconstruction, 1552-byte DMA payload,
pipeline transaction accounting, MMA order and rounding are unchanged.

For the GLM TP4 hidden dimension 4096, FC1 consumes 32 stages per work item
(16 K256 tiles × gate/up), removing 32 128-thread barriers per item. This
count is not a latency estimate. FC2 and wider rows retain the in-place
helper and both barriers.

The C1 output-scatter cache held 128 token/weight entries although consumers
only index their M16 tile. Binding it to 16 entries removes 112 unnecessary
initializations in each of two arrays: 224 32-bit shared stores per item.
It also recovers header space for the packed ring. Both arrays still publish
through the existing consumer synchronization before scatter reads them.

The two-stage packed ring holds 3104 bytes. Cache reduction and header
padding limit the total estimated shared-memory increase to 2048 bytes,
from 98304 to 100352. Native resources retain 126 registers, zero stack
and zero local bytes for C1. The binary separately reports 1024 static
shared bytes; this is distinct from the kernel's staged allocation estimate.

The initial two-ring FC1+FC2 design exceeded the 101376-byte staged-memory
limit and was rejected during CPU compilation. The final implementation
only separates FC1 input. C1 already needed more than half the SM's shared
memory before this change; this is not a claim about measured occupancy
or cache behavior.

The compile-time control `sf6_separate=False` retains the prior C1 helper
and scatter-cache geometry for a future same-source kernel comparison.
Normalization is idempotent, and this control has distinct in-memory and
disk cache identities. Enabled handles include `fc1sep` in the existing
lane-selection log. It is not an additional public serving knob. Existing
decode fastpaths, FP32 KDA state, SPEC_K=7 and prefill defaults are retained.

## Validation without a GPU

Existing image:
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`.
Each check used runc, CUDA hidden, no network, at most two CPUs/four GiB,
one build worker and an explicit Python/shell entrypoint. No image build,
model boot, CUDA context or service restart was issued.

- `focused-tests.log`: eight tests pass. Tests execute the production SF6
  method one load/store at a time across 128 randomly scheduled threads,
  covering all 256 bases, all 64 codes, byte wraparound, stage reuse and
  surrounding-byte preservation. The in-place forms retain both barriers.
  The actual scatter initialization runs for every valid M16 row count,
  changed routes and guarded allocations; dispatch/cache controls are checked.
- `native-compile.json`: six real CuTe/PTXAS/TVM-FFI kernels pass:
  separate M1/M7/M8, in-place M8 control, and unchanged M16/M32 geometry.
  Native binary hashes, register/local/stack resources, layout checks and
  exact engine-source SHA256 are retained. Binary extraction uses the
  compiler's retained LLVM fatbin without loading a CUDA library.
- `helper-compile.json`: both in-place and separate production helpers
  compile with the repeated-stage graph fixture. GPU execution is pending.
- `git diff --check` passes. The PR's engine check covers the complete CPU
  suite and the onepass recording/consumer contracts.

Reproduce in the pinned CPU image with `CUDA_VISIBLE_DEVICES=`:

```sh
python3 -m unittest -v tests.test_engine_moe_sf6_staging tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --output /out/native-compile.json
python3 probes/engine_moe_sf6_check.py --cpu --output /out/helper-compile.json
python3 tools/check.py --list --jobs 1
python3 -m unittest discover -s tests -p 'test_onepass*.py'
```

`engine_moe_sf6_check.py --gpu` is prepared, not submitted. It exercises
both helpers with changed packed data, all bases/codes, exact whole-buffer
canaries, ring reuse and 64 CUDA graph replays each. It does not reserve a
GPU or replace real-weight full-MoE, serving quality/acceptance and matched
consumer measurements. The operator's no-queue/no-baseline-engine constraint
remains in force.

## Upgraded Oracle

`st-oracle-pr875` at `e2bfbb9afdcc6e8fe1e5fe47ddddd23180b278b3` compares
base `32fb2892` with implementation `3f69b9a8`, using the retained actual
checkpoint configuration at C1 2K/32K/128K. `--acc 0` is a timing-only
scenario, not an acceptance forecast. Both source and configuration
identities are preserved in `oracle-c1.json` and the empty paired template.

The changed MoE kernel has no paired timing coefficient, so total decode
delta is **null** for every context. The unchanged modeled subtotal is not
0% measured benefit. The Oracle also conservatively leaves prefill delta
unknown because both files contain shared dispatch/kernel code, although
the new branch is restricted to M≤8. No improvement percentage is assigned
from the barrier/store count or from the older profiled MoE duration.
