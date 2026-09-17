# NVFP4 activation scale search: adoption review, 2026-09-17

**The user selected the three-candidate search (`ss1`) as the GLM production
default after reviewing the completed study and prompt ambiguity.** This branch
adds `ss1` to the production recipe. The valid 2K pair improves C=2 by 14.37%
and the multiplier from 1.331x to 1.477x, below the 1.7x target. Excluding the
ambiguous 282-unit ledger case, final-answer correctness is **32/32 for both
arms** across the full campaign and separate recheck. Strict certificate scores
remain 191/205 baseline versus 199/205 candidate, without forgiving transcription
errors. These are descriptive repeated-case totals, not a population estimate.

The same change introduces harness 47 / ko-reasoning-v3 to clarify the ledger
comparison, reservation delta and logic core's U6 scope. Generated answers and
grading are unchanged; old records retain v2. See `quality-ambiguity-audit.md`.
The source/default change is distinct from restarting a live serving process;
no v3 GPU quality or throughput result is claimed here.

The isolated consumer arm `2a427a6ffe6eb55ffe0cce9f285df0ce9205fc90` changes
only that recipe to add `ss1`; its same-build baseline is
`6908a0ed82d46536ef9429d3367a4ad605792f3d`. Their engine trees differ only in
the production recipe, and their benchmark trees are identical. Both contain
the same diagnostic-metrics update from main (`392c9d45`). The experiment is separate
from deployment/default adoption.

The subsequent 2K-only speed recheck uses baseline
`bd646ae621ef5b5979106f924a3a5ae75969eca5` and candidate
`d6a1663cbf5412ad71de8ac9d060c16c6d701988`. Their engine trees are unchanged
from the full campaign (`bc0e8c490f81` and `e099d675bfd7`); only the shared
benchmark preparation changed to cover both request-arrival orders.

The implementation was initially merged as PR #1117 (`fd99f82e`) while evidence
was being collected. At that point (`48fe5cfc`) main still used
`t,r,sf6,batch,q0`. The subsequent explicit user adoption changes the default to
`t,r,sf6,batch,q0,ss1`; `ss2` remains an optional five-candidate probe.

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

At the initial inspection the served v4 kernel imported
`quantize_block_fp4[_fast]` from FlashInfer; both its input-pack branches and
its SiLU-output branch used that max-based quantizer. V5 inherits the v4 body.
The implementation was subsequently based on main and integrated at the
static SiLU-output pack; the default still uses the original quantizer.

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
   ALU instructions/registers. Native measurements appear below. Three candidates
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

The full, isolated-port consumer tickets `fp4-scale-base17v4` and
`fp4-scale-candidate17v4` completed the identical `extended` workload, two passes per
boot. C=1 covers 2K/32K/128K twice; C=2 covers 2K/32K and the fixed 1,024-token
multiplier once. Separate tickets retain the baseline's failing quality result while
allowing the candidate to be measured; a failing gate is never waived.

| Full consumer quality | Baseline | Three-candidate search |
|---|---:|---:|
| C=1 first pass, cases | 7/9 | 6/9 |
| C=1 repeat, cases | 6/9 | 7/9 |
| C=1 final-result checks, both passes | 18/18 | 16/18 |
| C=2, 2K and 32K cases | 9/12 | 12/12 |

Equal aggregate C=1 case counts do **not** prove preserved quality. Under the
unchanged rubric, the candidate's 2K ledger answer computed the correct available
quantity, 279, but gave `전량승인` instead of the oracle's `보류` in both passes.
Both baseline passes matched the oracle. Inspection of the prompt and recorded
responses identifies an ambiguity, however: L4 says `주문 282개 이상이면 전량승인`
without explicitly making available inventory the subject of the comparison.
Both candidate responses interpreted it as an order-quantity threshold, while
correctly recognizing that 279 is below 282. This is not evidence of failed
arithmetic, and alone cannot establish quantization-induced quality degradation.
The later C=1 recheck answered this ledger case correctly. The canonical judge
still reports no valid warm candidate under the original rubric; no historical
grade was changed. Receipts: `consumer-baseline.json`,
`consumer-candidate.json`, and `consumer-qualification.json`.

At the user's request, remove the entire 2K, 282-unit ledger case from **both**
arms, including every C=1 repeat and C=2 client. Keep the other contexts' ledger
cases (orders 278 and 312) and all original scores. This post-hoc sensitivity
analysis is separate from the canonical gate:

| Excluding the 282-unit case | Baseline score | Candidate score | Fully passed cases, B -> A |
|---|---:|---:|---:|
| Full campaign C=1, two passes | 93/102 | 97/102 | 11/16 -> 13/16 |
| Full campaign C=2 | 59/64 | 64/64 | 7/10 -> 10/10 |
| Full campaign combined | 152/166 (91.57%) | 161/166 (96.99%) | 18/26 -> 23/26 |
| Separate 2K recheck C=1 | 13/13 | 12/13 | 2/2 -> 1/2 |
| Separate 2K recheck C=2 | 26/26 | 26/26 | 4/4 -> 4/4 |

Thus the full campaign favors the candidate by 9 points (+5.42 percentage
points), while the separate recheck favors the baseline by 1 point. Final-result
checks are all correct for both arms after this exclusion; the remaining score
differences concern supporting derivations/witnesses. These are few repeated
tasks, not independent draws establishing a population-level quality gain or
loss. The earlier blanket description of this experiment as a demonstrated
quality regression is too strong. `quality-excluding-order282.json` retains the
exact exclusion predicate and source hashes.

The recheck's candidate C=1 logic response writes world `010001` where the oracle
expects `010011`. Its C=2 ledger client 0 separately interprets releasing 9 of 44
reserved units as leaving 9 reserved, producing `323 - 9 = 314` instead of
`323 - (44 - 9) = 288`. Excluding the whole 282-unit case also removes this latter
error; it does not remove the logic error. `quality-diagnosis.json` retains the
prompt, visible outputs, narrowly scoped interpretation excerpts, and raw
artifact hashes. No layer activations or same-prefix logits were collected, so
the exact numerical cause of the response changes is unknown. Local group SSE
minimization does not minimize weighted projection error or answer error.

The following v3 revision explicitly compares available inventory to order
quantity and clarifies subtracting the released quantity from the existing
reservation. It has a new workload identity and needs its own matched comparison;
the old scores are not regraded or pooled with it.

Both full runs had clean C=1 preparation/profile/prefix evidence. Their raw
C=1 generation rates were 81.00 / 75.81 tok/s for baseline and 78.10 / 78.56
for the candidate; response lengths and acceptance varied between passes.
These are descriptive observations from quality-failing records, not adoption
proof. The four candidate nodes had identical helper, recipe and runtime
manifest hashes (`candidate-live-source.json`).

The full campaign's fixed multiplier was valid for baseline (64.36 -> 89.24
tok/s, 1.387x), but **invalid** for the candidate (65.85 -> 91.68 tok/s,
1.392x): four new specializations appeared on every rank. This pair cannot
support a throughput improvement claim. Its full-budget preparation happened
to admit each pair in the opposite order from measurement. The second prompt's
coexistence-prefill tails were 1087 / 1115 tokens in preparation but
1101 / 1100 in measurement. The repair prepares cyclic arrival orders, waiting
for each client's first token before starting the next. Measured requests
still start simultaneously. Ordered arrivals are rejected in a recorded
measurement phase. All 52 focused benchmark tests pass.

Tickets `fp4-scale-base17v5` and `fp4-scale-candidate17v5` completed the 2K-only
recheck, one boot/pass per arm, with the same four fixed 1,024-token requests and
repaired preparation. Both fixed measurements are valid: all four ranks remain
at 59 specializations throughout both widths, without graph-capture changes or
diagnostic instrumentation. Workload hashes match across arms and widths;
request-body hashes differ because each measurement gets a fresh cache salt.

| Fixed 2K consumer metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| C=1 output tok/s, including TTFT | 65.513 | 67.552 | +3.11% |
| C=2 aggregate output tok/s, including TTFT | 87.229 | 99.766 | +14.37% |
| C=2 / C=1 | 1.331x | 1.477x | +10.92% |
| C=1 decode-only tok/s | 72.627 | 74.416 | +2.46% |
| C=2 sum of per-request decode tok/s | 103.154 | 113.815 | +10.33% |

This pair satisfies the C=1 speed constraint but misses 1.7x. It is a single
sequential B/A observation, not a repeated B/A/B proof. The original quality
scores are C=1 3/3 vs 2/3 cases and C=2 6/6 vs 5/6. The 32K/128K campaign is not
repeated; quality limitations and the exclusion analysis above still apply.
Both tickets finished and released their reservations. The candidate ticket
correctly exits 2 for failed quality. Receipts: `consumer-speed.json` and
`phase-proof-speed.json`. The full campaign's per-rank counters and decode-width
histograms remain in `phase-proof-baseline.json` and `phase-proof-candidate.json`.

The earlier baseline run `20260917T072455-a8bbf0cf71ac`, under
`fp4-scale-consumer17b`, predates main's diagnostic-metrics update. It scored
C=1 6/9 and C=2 7/12. Its fixed-token result (65.32 -> 90.50 output tok/s,
1.386x) is **invalid**: two additional Triton specializations appeared on each
rank during C=2 timing. The bounded 64-token preparation did not cover the
full measured workload. Both new arms therefore prepare all four fixed
requests with their actual output and reasoning budgets before timing either
width. That first repair was insufficient when request arrival order changed;
the subsequent repair above covers that case. The steady-state gate, measured
requests and quality rubric remain unchanged.

The initial `fp4-scale-consumer17` submission was stopped while correcting a
command that expanded to five boots; its incomplete measurement is excluded.
Additional submissions rejected for an outdated controller worktree acquired
no GPU hold; two superseded queued v3 jobs were withdrawn before running.

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
