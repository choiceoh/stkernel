# NVFP4 activation scale search: adoption review, 2026-09-17

**The default-off GPU prototype is implemented and passes its numerical and
kernel-cost gates. Consumer qualification is still pending.** Enable `ss1`
(three candidates) or `ss2` (five) in the explicit static MoE recipe to test
FC2 input quantization. The production recipe is unchanged on this branch.

The isolated consumer arm `151fcf3c4b291dff592ce2b0588866a34d493ab9` changes
only that recipe to add `ss1`; its same-build baseline is
`10eddbdedb12c533be43d11031d86b4d6e28b9d7`. The consumer experiment is separate
from deployment/default adoption.

The user contract remains unchanged: top-8 routing and model quality are
preserved; C=1 throughput may fall at most 5%; the C=2 aggregate/C=1 throughput
target is approximately 1.7. This change already starts from W4A4: it saves no
weight bytes and removes no matrix operations. Any net throughput benefit
would require a measured downstream effect, such as better accepted-token
yield, large enough to repay the additional quantization work.

## What was implemented and measured

`probes/nvfp4_scale_search_review.py` is an independent, standard-library CPU
reference. It compares round-to-nearest-even E2M1 reconstruction under the
max-based E4M3 scale with adjacent scale codes, preserving the per-tensor global
scale. It does not import or patch a serving kernel. `--input` accepts saved
16-element activation groups and their global scales for subsequent replay.

The study contains 4,096 groups per distribution, 16,384 groups / 262,144 values
in total. All fixtures are synthetic and BF16-rounded, including the SiLU
fixtures; they are not activations captured from GLM. The native kernel rounds
the post-SiLU intermediate to BF16 before reading it as FP32 for packing.
Actual activation distributions and weight sensitivity need separate evaluation.
Random seed 917 and source/input hashes are retained in the JSON receipts.

| Synthetic distribution | MSE reduction, 3 candidates | MSE reduction, 5 candidates | 3-candidate share of 5-candidate gain |
|---|---:|---:|---:|
| Normal | 16.42% | 17.12% | 95.91% |
| SiLU × up | 13.32% | 13.34% | 99.86% |
| Clamped SiLU × up | 15.45% | 15.51% | 99.55% |
| One large outlier | 0.45% | 0.46% | 99.13% |

MSE reduction is `1 - sum(search squared errors) / sum(base squared errors)`
over each distribution, NOT a model accuracy percentage. Both arms use the
same independent mathematical encoder; they are not a bit-exact replay of the
CUDA fast reciprocal path. Keeping the baseline among the candidates ensures
non-increasing group SSE in this reference. The benefit is distribution
dependent: the outlier fixture barely improves.

Lower SSE does not ensure smaller maximum error or smaller projection error.
In 348 / 186 / 180 / 203 groups respectively, the five-candidate result has a
larger maximum absolute error. A regression test gives a concrete example:
`[0.85] * 15 + [6]` prefers a smaller scale by SSE, but reconstructs the final
value less accurately. A projection concentrating on that final component
therefore becomes worse. Answer quality must remain a separate gate.

Validation: 12 CPU tests pass, including code boundaries, midpoint ties,
signed zero, scale convention, nested candidate sets, and the projection
counterexample. All 127 positive finite E4M3 encodings, all 126 E4M3 midpoints,
and seven E2M1 midpoints also agree with independent PyTorch / engine CPU
reference checks in the pinned serving image. The macOS and Linux CPU study
results match exactly. No GPU was exposed to the checking container.

## Actual integration points and implementation constraints

The serving container was inspected read-only during this review:

- Image: `sha256:e9e80b94d41277b171cef5785483989acd1c91d450d0d1068269e99f60bc75bd`.
- Runtime: PyTorch `2.13.0+cu132`.
- Installed `flashinfer/cute_dsl/fp4_common.py` SHA-256:
  `a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1`.
- Served `moe_static_kernel_v4.py` SHA-256:
  `c6320447ee8fba3d563cf988991a6f58c859201391c7278206cb933ef748359e`.

The served v4 kernel still imports `quantize_block_fp4[_fast]` from FlashInfer;
its additional input-pack branches and its SiLU-output branch use the same
max-based quantizer. V5 inherits the v4 body. The served source has advanced
beyond this research branch: rebase the eventual kernel prototype onto that
source before choosing every input-pack call site.

1. Start with the SiLU-output / FC2-input pack, with three candidates: baseline
   E4M3 code and its immediate neighbours. Retain the five-candidate option as
   an accuracy/cost comparison. Test input / FC1 packing separately, then both.
   This ordering isolates effects; it does not assert that FC2 is safe.
2. Respect the FlashInfer global-scale convention: reconstruction is
   `fp4 * e4m3_scale * global_scale`, and the initial block scale is
   `amax / (6 * global_scale)`. The similarly named local W4A16 helper uses a
   multiplier when deriving scales; transplanting that formula would be wrong.
3. Keep experts, router, route weights, checkpoint weights/scales, SF6 weight
   storage, activation/clamp semantics and FP32 scatter unchanged. Alter only
   selection of the activation block scale and its corresponding FP4 bytes.
4. Compile the experiment out when disabled. Include the option and helper
   source in native cache identity; do not mutate the installed FlashInfer
   package or monkey-patch shared production imports.
5. Start from the actual baseline packed bytes and scale, score their real
   reconstruction, and choose a candidate only on strict improvement. Preserve
   existing exceptional-input / zero-global-scale behavior by falling back to
   the baseline. The offline reference intentionally rejects nonfinite values
   and nonpositive global scales; it is not the production fallback contract.
6. Additional candidate rounding, reconstruction and SSE accumulation cost
   ALU instructions/registers. No measured cost is available. Three candidates
   score 48 scalar reconstructions per group, five score 80; these counts are
   not a kernel or end-to-end slowdown estimate. Fuse with existing packing
   instead of adding a separate launch and activation memory round trip.

## Remaining adoption gates

- Native compile plus exact baseline-off parity; independent GPU quantizer
  oracle for scale boundaries, midpoint neighbours, global scales, poisoned
  graph outputs and replay. Inspect registers/spills and code size.
- Captured real activation and real-weight projection checks, including both
  reconstruction and projection errors; synthetic group MSE alone cannot pass.
- Same-build baseline/candidate/baseline kernel and consumer measurements at
  C=1 and C=2, matched contexts and generation budgets, including 32K and 128K.
  Record output tok/s, TTFT, per-request latency, batch width, acceptance,
  output hashes and the existing answer-quality checks.
- Reject if C=1 throughput is below 95% of the matched baseline or quality
  regresses. Report C=2 absolute throughput as well as its ratio; a smaller
  denominator does not establish progress toward 1.7.

At the initial review, the single-GPU lane had insufficient memory. Following
the user's implementation request, the native experiment ran through the
canonical fleet reservation instead; no single-GPU memory guard was relaxed.

## Native implementation and GPU results

`fp4_scale_search.py` scores the actual packed E2M1 reconstruction. It starts
with the original result and retains it for invalid global scales, nonfinite
inputs or unordered SSE comparisons. Selection is fused into the existing
static FC2 pack. The disabled branch is eliminated at compile time; `ss1`
and `ss2` participate in both the in-process and persistent kernel identity.
Dynamic prefill and FC1 input quantization retain their original quantizers.

The `fp4_scale_search_compile` probe compiled 13 kernels without exposing a
GPU: four quantizer variants and three MoE variants at 8, 16 and 32 rows. All
passed CuTe/NVVM/PTXAS on sm_121a. The disabled quantizer and original have
identical cubin bytes (SHA-256
`37467c7e3cf3e88f492b4e0690bf5467d9d18581c4c4fc7e555d7b342e167633`).
C=1/C=2 MoE registers remain 96 with all three options; local/stack bytes are
zero. At 32 rows the counts are 117 / 120 / 118, with no local/stack use.

GPU ticket `fp4-scale-search17`, source `10eddbde`, completed successfully in
73.5 seconds on GB10/rank 0. Peak tensor allocation was 1,932,266,496 bytes.
The 151,322 finite blocks include every finite BF16 encoding, mixed blocks,
FP4 midpoint neighbours, wide exponents, and synthetic normal/SiLU inputs.
Independent FP64 PyTorch decoding found no worsened reconstruction SSE beyond
the declared 2e-5 relative tolerance. Invalid-scale/nonfinite fallback and
disabled-mode byte parity passed, as did poisoned-output graph replay.

Real layer-3 MoE timing uses identical weights, routes, inputs and runtime,
four B/A/A/B brackets and 8/16 verify rows. Positive numbers mean slower:

| Arm | C=1 warm | C=1 evicted | C=2 warm | C=2 evicted |
|---|---:|---:|---:|---:|
| ss1, 3 candidates | -0.077% | -0.045% | +0.103% | +0.093% |
| ss2, 5 candidates | -0.550% | +0.280% | -0.158% | +0.280% |

These small changes are kernel observations, not consumer speed gains. Both
arms produced finite outputs on three seeds and exact zero outputs with zero
route weights. Candidate outputs intentionally differ from the old quantizer;
their difference is reported without pretending it is a quality verdict.

`nvfp4_scale_projection.py` independently reads nine real rank-0 experts
(layers 3, 20, 40; experts 0, 73, 287) and evaluates their FC2 projection on
synthetic inputs on CPU. Three candidates reduce aggregate activation SSE by
9.80% and projection SSE by 9.72%; all nine projection cells improve. Keeping
the existing per-128 BF16 output rounding gives 9.72% as well. Five candidates
yield 9.77% projection improvement. This supports the three-candidate choice
but remains a tensor study, not real-prompt quality proof. Receipts are
`compile.jsonl`, `gpu.jsonl` and `projection-cpu.json`.

The full, isolated-port consumer ticket `fp4-scale-consumer17b` runs the
`extended` workload, baseline/candidate/baseline, two passes per boot. Its
quality, acceptance and C=1/C=2 results are pending. The first consumer
submission was stopped while correcting a command that expanded to five
boots; its incomplete measurement is excluded.

## Reproduction and upstream evidence

```sh
python3 -m unittest tests.test_nvfp4_scale_search_review -v
python3 probes/nvfp4_scale_search_review.py --blocks 4096 \
  --output measurements/st_nvfp4_scale_search_20260917/cpu.json
# Inside a CPU-only serving-image container with PYTHONPATH=/repo:
python3 /repo/probes/nvfp4_scale_search_review.py --blocks 4096 \
  --torch-check --output /out/runtime-cpu.json
```

Atlas snapshot `95f674951d6ab8f491f7907804a462c170f9c048`:
[five-code search](https://github.com/Atlas-Inf/atlas/blob/95f674951d6ab8f491f7907804a462c170f9c048/kernels/gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu#L269-L308),
[FFN call site](https://github.com/Atlas-Inf/atlas/blob/95f674951d6ab8f491f7907804a462c170f9c048/crates/spark-model/src/layers/dense_ffn.rs#L2595-L2614),
[same-quant CPU error reference](https://github.com/Atlas-Inf/atlas/blob/95f674951d6ab8f491f7907804a462c170f9c048/crates/spark-model/src/layers/ops/nvfp4_mmq.rs#L3-L14).
The implementation here reproduces the mathematical idea independently; no
Atlas or vendored llama kernel source was copied into the ST engine.
