# GB10 sparse NVFP4: calibrated and joint expert reconstruction

2026-09-12, srv4, GB10/SM121a. Follow-up to
[the native sparse feasibility probe](../st_gb10_sparse_nvfp4_20260912/README.md).

**Calibration reduces the distortion of simple pair pruning, but the tested
methods do not establish an acceptable replacement for the current model.**
All results below are local numerical errors, not language-model accuracy.
No serving kernel, original checkpoint, or production service was changed.

## Actual text inputs and separation of decisions

`engine_sparse_capture.py` runs the original checkpoint's prefix through
L0–L2 and L3 attention, then captures the L3 FFN input and router decisions.
All four TP ranks run on one GPU with LocalTP, using native KDA chunk and
explicit reference implementations for other lanes. Layer streaming keeps
the entire 45-layer model out of memory. This is actual text/model activation
capture through the engine composition, not interception of production requests.
LocalTP's FP32 rank-order sums differ from the fleet's NCCL implementation.

The supplied fixtures are small, author-written Korean/English prompts about
code, arithmetic, systems, and writing. They are not a representative public
benchmark or a production-distribution calibration set. The normal chat template
is rendered with thinking disabled, with at most 256 tokens per prompt. The
first eight positions are excluded from fitting/evaluation to reduce common
template-prefix effects. These are short prefill inputs, not generated decode
tokens or long-context examples.

| Split | Prompts | Kept tokens | Role |
|---|---:|---:|---|
| Training | 24 | 3,402 | Hessians, weight reconstruction, residual fitting, short QAT |
| Validation | 6 | 849 | Select sparse method/damping, residual rank, QAT checkpoint |
| Initial diagnostic test | 6 | 843 | Report initial experiments; never passed to the optimizers |
| First fresh holdout | 12 | 1,699 | Freeze calibrated/residual choices before capture |
| Reconstruction holdout | 8 | 1,144 | New capture after joint reconstruction choices were frozen |

The initial diagnostic results informed which method family to investigate
next. Consequently the later fresh holdouts are kept separate, rather than
calling every stage an untouched test. There is no fitting or selection in
`engine_sparse_holdout.py`; it validates prompt and artifact identities.

All replicated L3 inputs, selected expert IDs and coefficients match exactly
across the four local TP ranks. Capture peak Torch allocation was
1,961,152,512 bytes. A two-prompt compatibility run on integration base
`cda0af61` produced byte-identical tensors to the measured engine base
`7b394e88`; see `compatibility.json`.

Experts are selected solely by training route count: **10, 4, 119, 178**.
Their training counts are 1,494, 263, 227 and 225, respectively. This is limited
coverage for an input dimension of 4,096, especially for the less frequent
experts. Only rank0's slice of these four experts is reconstructed.

## Methods that were executed

1. **Magnitude baseline:** keep the two largest adjacent pairs in each K8
   group; requantize under an E4M3 scale for each logical K32 group.
2. **Activation-weighted pair selection:** Wanda-style squared weight times
   input second moment, summed within adjacent pairs.
3. **Pair-constrained SparseGPT-style reconstruction:** use the regularized
   input Hessian, select legal pairs per K32 group, and propagate pruning and
   FP4 rounding errors into later columns. Masks and E4M3 scales are fixed for
   a group during its sequential reconstruction. Damping 0.01 and 0.1 are
   candidates; validation selected 0.1 for both projections of all four experts.
   This is an adaptation, not the stock scalar-2:4 implementation.
4. **Activation-aware low-rank residual:** reduced-rank ridge regression fits
   the remaining projection error, including input K16→K32 rounding. BF16
   factors are evaluated with BF16 intermediate rounding. Candidate ranks
   are 16/32/64, selected on validation. This follows EoRA's compensation idea
   but is not an invocation or reproduction of stock EoRA.
5. **Joint expert reconstruction:** optimize both retained weight matrices and
   their log scales against the original expert's final output. The forward
   includes clamped SwiGLU, BF16 intermediate rounding, legal pair masks, E2M1
   weight/activation rounding and E4M3 scale rounding. Straight-through gradient
   estimators allow 200 Adam steps, with a per-row RMS-normalized reconstruction
   loss and a weight/scale drift penalty. Only validation chooses a checkpoint;
   the initial state remains eligible. Export needs no residual matrices.

[SparseGPT paper](https://arxiv.org/abs/2301.00774),
[Wanda implementation](https://github.com/locuslab/wanda),
[NVIDIA EoRA description](https://developer.nvidia.com/blog/a-fine-tuning-free-approach-for-rapidly-recovering-llm-compression-errors-with-eora/),
[NVIDIA QAT/QAD explanation](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/).

The reference is the existing quantized checkpoint, with the ST activation
packer's K16 reciprocal/rounding semantics. It is not the original BF16 model.
The candidate uses K32 activations and weight scales because that is what the
executed sparse instruction requires. K16→K32 activation distortion is included
in every candidate comparison. On the first fresh holdout, changing only the
activation quantization changed projection output by 6.2–8.6% relative L2.
This is a measured control, not an irreducible error bound after learning.

## First fresh holdout: calibration and residual correction

Relative L2 error of the **complete rank-local expert chain**,
`w13 → clamped SwiGLU → input quantization → w2`, versus its original values:

| Expert | Routed rows | Magnitude | Calibrated | Calibrated + residual |
|---|---:|---:|---:|---:|
| 10 | 768 | 67.40% | 48.07% | 44.83% |
| 4 | 121 | 65.04% | 28.80% | 26.59% |
| 119 | 109 | 65.52% | 41.88% | 40.00% |
| 178 | 91 | 61.47% | 33.90% | 32.42% |

These chain errors should not be compared as if they were the earlier 45%
**single-projection** Gaussian-input measurement. Each row here includes both
projections and the nonlinear activation on real prefix inputs.

Combining these four experts with the original router coefficients gives
64.03% → 31.32% → 28.45% relative L2. This is only their selected contribution:
8.01% of routed slots, across 945 tokens with at least one selected expert.
It excludes the other experts, shared expert, TP reduction, mHC and later layers.
The residual-corrected per-token median is 44.82% and p95 is 64.25%; the smaller
aggregate L2 is influenced by high-energy outputs and should not conceal tails.

The residual factors add 4,161,536 bytes for four experts. Using the prior
probe's logical E4M3 weight/metadata accounting, 13.5 MiB dense becomes 8.25 MiB
sparse, then 12.21875 MiB with residual factors: only 9.49% smaller. This excludes
runtime-specific SF6 packing, allocation padding, activations and graph buffers;
it is not a measured serving-memory or latency improvement.

## Joint reconstruction findings

The selected checkpoints after 200 steps were steps 180, 40, 20 and **0** for
experts 10, 4, 119 and 178. Validation relative L2 changed from
47.71→46.79%, 28.06→26.97%, 36.88→36.26%, and 21.00→21.00%, respectively.
Training loss decreased more than held-out error; this is consistent with
limited calibration coverage/overfitting, not proof of its sole cause.

The export check exposed an important implementation issue: recomputing a
scale from an already-quantized weight is not always idempotent for subnormal
E4M3 scales. Joint reconstruction now starts from the stored scales, learns
their logarithms, and simulates E4M3 rounding directly. Its exported weights
reproduce the selected training forward exactly. The unit suite includes a
minimum-subnormal counterexample, so an accidental second quantization fails.

`reconstruction-holdout.json` is the final comparison including this method,
on an additional eight prompts captured after the method and checkpoints were
fixed. No selection is made from those results.

| Expert | Routed rows | Magnitude | Calibrated | Calibrated + residual | Joint reconstruction |
|---|---:|---:|---:|---:|---:|
| 10 | 448 | 67.07% | 46.68% | 43.43% | 45.85% |
| 4 | 65 | 65.22% | 22.98% | 20.24% | 22.44% |
| 119 | 93 | 64.44% | 50.05% | 48.12% | 49.64% |
| 178 | 72 | 60.83% | 32.02% | 30.52% | 32.02% |

These are whole rank-local expert-chain errors, with the same definition as
the preceding table. The four experts cover 7.41% of routed slots and 592
tokens with at least one selected expert in this holdout.

| Weighted selected-expert contribution | Relative L2 | Per-token median | Per-token p95 |
|---|---:|---:|---:|
| Magnitude | 62.81% | 67.81% | 73.20% |
| Calibrated | 32.86% | 46.36% | 66.14% |
| Calibrated + residual | 30.16% | 43.37% | 64.12% |
| Joint reconstruction | 31.54% | 45.21% | 66.43% |

Joint reconstruction reduces aggregate error by 1.33 percentage points from
the calibrated starting point without adding residual GEMMs. It does not beat
the independent residual method here, and the per-token p95 does not improve.
The unchanged expert 178 is the validation-selected initial checkpoint, not a
failed export. None of these numbers measures model accuracy or the complete
MoE output. They also cannot be compared across holdouts as a progress curve:
the prompts and expert-output energies differ.

## Validation and operational decision

**13 numerical/gradient tests pass, zero skips.** They cover inverse-Hessian
reconstruction, correlated inputs, legal pair masks, export identity, subnormal
scales, validation-only selection, low-rank recovery on unseen inputs, STE
gradients, and zero gradients on removed pairs. The calibrated and jointly
reconstructed exports each passed eight native dense/sparse/reference projection
checks on GB10. This establishes arithmetic/layout correctness; it does not
establish language-model quality.

No additional latency benchmark or serving adoption is justified by these
quality results. The previous isolated-kernel timing remains the only measured
performance result. A wider real-activation calibration corpus, better coverage
per expert and a separate end-to-end quality evaluation would be needed before
judging a more extensive recovery effort. These experiments do not show that
all structured sparse training methods fail on GLM.

## Reproduction and artifacts

The remote task directory `/tmp/st-sparse-calibration-9391` contains captured
activation tensors and experimental weight/factor exports; original rank files
were mounted read-only. `provenance.json` records rank header identities,
metadata/source hashes, image ID and native library hash. JSON/log evidence is
committed; large tensors and the compiled native library remain on srv4.

With this repository mounted at `/repo`, the task directory at `/work`, original
rank files at `/ranks`, metadata at `/meta`, and the earlier library at `/native`:

```bash
export PYTHONPATH=/repo
python3 probes/engine_sparse_capture.py --ranks /ranks --metadata /meta \
  --prompts probes/fixtures/sparse_calibration_prompts.json --out /work/capture.pt
python3 probes/engine_sparse_calibrate.py --capture /work/capture.pt \
  --rank /ranks/rank0of4.safetensors --library /native/sparse.so --out /work/recovery.json
python3 probes/engine_sparse_residual.py --capture /work/capture.pt \
  --recovery /work/recovery.json --rank /ranks/rank0of4.safetensors --out /work/residual.json
python3 probes/engine_sparse_block_reconstruct.py --capture /work/capture.pt \
  --recovery /work/recovery.json --rank /ranks/rank0of4.safetensors \
  --library /native/sparse.so --out /work/block-fit.json
python3 probes/engine_sparse_capture.py --ranks /ranks --metadata /meta \
  --prompts probes/fixtures/sparse_reconstruction_holdout_prompts.json \
  --out /work/reconstruction-capture.pt
python3 probes/engine_sparse_holdout.py --capture /work/reconstruction-capture.pt \
  --training-capture /work/capture.pt --recovery /work/recovery.json \
  --residual /work/residual.json --block-fit /work/block-fit.json \
  --rank /ranks/rank0of4.safetensors --out /work/reconstruction-holdout.json
python3 -m unittest discover -s tests -p test_engine_sparse_recovery.py -v
```
