# Default sampled drafter selector — 2026-09-17

K remains 7. The drafter formerly launched gather, softmax, CDF, sampling,
probability publication and token selection separately for each position.
Conditional probabilities now form one batch; a single Triton kernel walks
predecessor indices using the caller's keyed uniforms and publishes token
IDs, private candidate support and the exact sparse probability it sampled.
The ordinary batched sampled path and synchronous unconstrained sampled
path use this implementation. Greedy-only and constrained-boundary proposal
paths keep their existing implementations. Rank agreement is unchanged.

## Result: the actual selector pipeline, model/head held fixed

The baseline method is extracted with AST from `Drafter.propose_rows` at
main `13ce718f`, not a hand-written approximation of its launch count.
Both calls receive identical precomputed `candidate_rows` outputs. They
execute actual codebook gather, predecessor/projection product, edge GEMM,
unary addition and sampled selection. Model, head/top-k and TP communication
are excluded. Codebooks are synthetic BF16 [4096,256], candidates=16.

| Rows | Baseline µs | Candidate µs | Reduction | Saved µs |
|---|---:|---:|---:|---:|
| 1 | 142.390 | 19.321 | 86.43% | 123.069 |
| 2 | 143.676 | 20.892 | 85.46% | 122.784 |
| 4 | 156.297 | 24.810 | 84.13% | 131.487 |
| 8 | 160.743 | 34.658 | 78.44% | 126.085 |

Medians of every baseline/candidate position in three B/A/A/B brackets,
12 graph repeats per position using `cublaslt._measure`. Raw samples include
all observed noise (`draft-sample.json`). Device: owned RTX 5050, SM120,
20 SMs, driver 595.79; Torch 2.13.0+cu132, CUDA 13.2, Triton 3.7.1.
These are **selector component timings**, not a full drafter or engine
speed claim. GB10/TP4 step/s, tokens/s and acceptance remain unmeasured.
Savings from the separate convolution seam probe are not added to these
measurements to manufacture an end-to-end result. No fleet queue or boot.

## Numerical and runtime gates

- Fourteen shape/mass-policy cases pass bit-exact token IDs, support and
  FP32 probabilities across 12 random/tied/masked/peaked input trials each.
  K=5/7/8, rows=1/2/4/8, support=4/8/16 and real keyed-input row strides.
- Two changed-input CUDA graph replays per case pass exact output and no
  Torch allocation-counter growth during replay.
- Actual selector pipeline: five changed-input comparisons per row count,
  all bit-exact against the baseline method.
- Additional uniforms directly before/at/after conditional CDF cuts pass
  for rows=1/4 and both mass policies (`draft-sample-boundaries.log`).
- CPU related suite: **50 passed / 9 skipped**, 59 total (`cpu.log`).
- **16 offline SM121 specializations pass** with no GPU device access
  (`draft-sample-compile.json`).
- Source SHA-256 hashes and baseline file hash are in the JSON receipts.

### The rejected first version matters

Simply calling Torch cumsum on the batched probability tensor passed random
inputs but failed C=1 uniforms at a CDF cut. In this runtime, Torch's single
short-row CDF uses serial accumulation whereas its multirow CDF uses a
parallel tree; several cumulative entries differ by one or two FP32 ULPs.
The final C=1 kernel retains serial order explicitly and passes the boundary
regression. Multirow CDF stays in Torch. Softmax and total-mass reduction
also stay in Torch. Support is explicitly bounded to DFlash's <=16 entries;
a larger unqualified shape fails rather than silently changing arithmetic.
The synchronous path still uses its last CDF entry for total mass; batched
proposal still uses `probs.sum`, preserving their historical distinction.

The implementation has temporary O(rows*K*16*16) probability/CDF storage in
place of per-position Torch temporaries. It adds no persistent model weights
or scratch shared between graphs; outputs remain private. Calibration alpha,
FP16 state and target precision are unchanged. The exact scan qualification
belongs to the pinned runtime; another Torch/CUDA version must rerun these
gates before claiming equivalent keyed draws.

### Additional rejected attention experiment

Changing attention query tiles from 32 to 16 retained exact outputs but was
slower on this component device: C=1 roughly 21.3 → 24.4 µs, C=2 39.2 →
42.6 µs, C=4 68.3 → 74.2 µs. No attention change was adopted. Raw samples
are `draft-attention-layout.json` and the exploratory source is
`rejected_attention_layout.py`. It fixes the GB10-shaped 48-SM partition
geometry on the 5050 and is not a GB10 tuning verdict.

## Reproduction

Populate the owned scratch checkout with this commit's engine/probes/tests.
Prepare the exact baseline file before invoking the locked `run-gpu.sh`:

```sh
git show 13ce718f:engine/profiles/glm53/drafter.py > out/draft-sample-baseline.py
```

The runner records the immutable image and owns only its named container.
`run-boundaries.sh` runs the draw-boundary regression in the same image.
CPU/no-device checks:

```sh
CUDA_VISIBLE_DEVICES= python3 probes/engine_draft_sample_compile.py --output /out/draft-sample-compile.json
python3 -m unittest tests.test_engine_draft_sample tests.test_engine_drafter tests.test_engine_draft_select tests.test_engine_draft_post_norm tests.test_engine_draft_agreement tests.test_engine_cublaslt_producer tests.test_engine_drafter_storage tests.test_engine_draft_tuning_integration tests.test_engine_early_observe -v
```
