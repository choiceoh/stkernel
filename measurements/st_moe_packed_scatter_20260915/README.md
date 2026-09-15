# C1 MoE packed shared-output loads

Implementation and native compilation record, 2026-09-15. This follows
[PR #1002](https://github.com/choiceoh/stkernel/pull/1002), merged as `9fe30cc9`.
The default C1 SF6 staged epilogue now reads each BF16x8 output span with one
128-bit shared load and consumes it in the existing two FP32 vector REDs.
GPU numerical/replay checks now pass. The added instruction/resource reduction
does not establish a material component-latency gain in the measured bracket.
Full-model acceptance and tok/s results remain unmeasured.

## Implementation

The previous epilogue loaded eight BF16 scalars, formed eight FP32 weighted
contributions and passed pairs of four to the vector RED helper. The new
side-effecting helper performs one `ld.shared.v4.u32`, widens each BF16 value,
multiplies each by the original FP32 route weight, applies the same saturated
BF16 rounding and emits the two existing `red.relaxed.gpu.global.add.v4.f32`
operations. The prefill helper cannot be reused: it first rounds the route
scale to BF16, which is a different decode arithmetic contract.

The load stays inside the side-effecting scatter, so it cannot be reused
across different tile publications. Both publication and retirement barriers
are retained. The BF16 shared output uses S<3,4,3> on byte addresses. At native
compilation, every one of its 512 aligned BF16x8 spans is checked against
CuTe's ordinary composed layout, including full coverage and non-overlap.
The existing shared allocation is 1024-byte aligned. No buffers or launches
are added.

`scatter_packed_load` defaults on only with the staged SF6 reform tile's
`scatter_vec4`. The served C1 M1..8 path selects it; the direct-register C2
path retains its existing implementation. `scatter_packed_load=False` keeps
the PR #1002 vector RED with scalar shared loads in a separate cached handle.

## Native results

Actual CuTe lowering, PTXAS, TVM-FFI and SASS disassembly on srv2, with no GPU
context. Image: `st-engine:bracket-3acae0170bb2`
(`sha256:1eccea22a434b863c5f146e8616f18971f92e64f516cfbb0b7dc091ea402ef9a`).
Container: runc, network disabled, two CPUs, 4 GiB and CUDA devices hidden.
`native-compile.json` records the source and native binary hashes, resources
and opcode counts for all seven handles.

| Handle | Packed load | Registers | CTA shared bytes | Static instructions |
|---|---|---:|---:|---:|
| M8 default | on | 96 | 91,136 | 3,336 |
| M8 PR #1002 control | off | 113 | 91,136 | 3,539 |
| M8 pre-#1002 pairwise RED control | off | 113 | 91,136 | 3,547 |
| M7 default | on | 96 | 91,136 | 3,336 |
| M16 production direct scatter | off | 96 | 100,352 | 4,013 |
| M16 staged-output comparison | on | 96 | 91,136 | 3,336 |
| M16 staged-output scalar-load control | off | 113 | 91,136 | 3,539 |

All seven compiled, with zero stack/local usage. Against PR #1002, M8 loses
203 static instructions (5.74%) and 17 registers (15.04%). The output section
replaces 32 scalar shared loads with four `LDS.128` instructions across the
unrolled loop. The complete kernel's `LDS.U16` count falls from 72 to 40;
the other 40 belong to the scale operands. It retains eight F32x4 REDs.
These are instruction/resource changes, not measured latency or tok/s gains.

## GPU component results

Canonical fleet session `moe-maturity-load-a2bc` completed on 2026-09-15 at
14:18 KST on NVIDIA GB10, running the exact implementation commit `e1de84d6`
with the image above. Both `moe_pair_packed_load` and `moe_pair_vec4` ran in
the same invocation, with 2.1 seconds queued and 100.8 seconds in the payload.
`gpu-pair.json` retains the packed-load comparison; `gpu-vec4-control.json`
retains the independent RED-width comparison. All 14 recorded source hashes
in each artifact match the tested implementation. The actual L3 rank-weight
hashes and folded scale contract are retained in both files.

Each comparison passes 22 numerical cells, including changed activations and
routes, poisoned outputs and real-router fixtures. All recorded relative
errors and replay spreads are zero: 44 numerical cells across both axes.
M8 selects the production default in the packed-load comparison. M16
explicitly selects staged output to test the change; production M16 continues
to use the direct-register kernel. The configurations differ in exactly the
named axis at each row count. Neither arm of the RED-width comparison is the
new production default because packed loads are deliberately disabled.

In `engine_decode_batch.timing`, **B is the baseline (index 0) and A is the
candidate (index 1)**. The values below are arithmetic means of each arm's
two samples in one B/A/A/B sequence. Each sample is the median of 32 captured
replays. Negative change means lower candidate latency. The FFN includes
routed/shared experts and output finalization; router work is outside timing.

| M8 fixture | Cache | Scalar-load baseline B, microseconds | Packed-load candidate A, microseconds | Candidate latency change |
|---|---|---:|---:|---:|
| 8 unique experts | warm | 144.144 | 147.104 | +2.05% |
| 8 unique experts | evicted | 228.312 | 227.008 | -0.57% |
| Actual L3 router, correlated rows, 9 experts | warm | 184.376 | 183.752 | -0.34% |
| Actual L3 router, correlated rows, 9 experts | evicted | 245.552 | 245.032 | -0.21% |
| Actual L3 router, independent rows, 55 experts | warm | 884.488 | 884.424 | -0.01% |
| Actual L3 router, independent rows, 55 experts | evicted | 938.600 | 937.200 | -0.15% |

The other synthetic M8 cells range from -0.33% to +0.47%. The M16 staged
compatibility cells range from -0.98% to +2.18%; those do not represent the
served C2 implementation. This one bracket establishes numerical equivalence
on the exercised fixtures and fewer compiled instructions/registers, but not
a material additional latency gain. The M8/U8 warm regression is retained
above and in the raw artifact, rather than inferred away as measurement noise.
Full-model TP4 acceptance and serving speed have not been measured here.

The RED-width comparison repeats the earlier PR #1002 check with the corrected
M16 controls. At M8/U8, vector RED takes 145.192 vs 152.744 microseconds warm
(-4.94%) and 231.360 vs 232.976 microseconds evicted (-0.69%). With the actual
L3 router's correlated M8 fixture, it takes 184.472 vs 189.224 microseconds
warm (-2.51%) and 249.552 vs 249.912 microseconds evicted (-0.14%). These
smaller differences than the earlier run show why its largest observed gain
should not be treated as a stable serving-speed claim.

## Focused checks

41 CPU tests passed. The staged-output test executes the production epilogue
for pairwise RED, vector RED with scalar loads, and the new packed load. It
compares every address and weighted contribution against an independent
enumeration with partial tiles, duplicate destinations, zero/negative route
weights and first/last output tiles. It checks equal byte coverage, eightfold
fewer shared-load operations, exact vector alignment, default dispatch bounds,
control cache isolation and constructor exclusions.

The pair probe also addresses PR #1002's M16 review finding: both staged
comparisons explicitly disable the M16 direct-register path, and the test
requires their normalized configurations to differ in exactly the named
axis at both M8 and M16. Per-row identity metadata records whether a candidate
is actually the production default. The original vec4 probe pins packed loads
off, so its comparison continues to isolate the RED width.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest -q \
  tests.test_engine_moe_vector_scatter tests.test_engine_moe_sync_cleanup \
  tests.test_engine_moe_batch_reform tests.test_engine_moe_sf6_staging \
  tests.test_engine_moe_scatter_config tests.test_engine_moe_register_scales \
  tests.test_engine_moe_compact_staging tests.test_engine_moe_fc1_reuse \
  tests.test_engine_moe_activation_store
# Existing ST image, no GPU:
CUDA_VISIBLE_DEVICES= PYTHONPATH=/repo python3 probes/engine_moe_sf6_compile.py \
  --scatter-packed-load --sass --output /out/native-compile.json
# Through the canonical fleet queue only:
python3 probes/engine_kernel_check.py --lanes moe_pair_vec4,moe_pair_packed_load \
  --ranks EXACT_CONSUMER_RANK_FILE --output /cache/moe-packed-load.json
```
