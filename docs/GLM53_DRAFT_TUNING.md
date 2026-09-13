# Optional low-cost draft controls

The three follow-ups to the [low-cost review](GLM53_DRAFT_LOW_COST_REVIEW.md)
are connected to native ST serving. They are explicit experiments: this change
does not supply checkpoint-fitted coefficients or claim higher acceptance.
No fleet or GPU run was used to develop this change.

| Control | Serving work | Application |
|---|---|---|
| Position-specific selector edge strength | Constant selection and one multiply per candidate in the existing greedy walk; no new projection, launch or collective | Greedy and sampled proposals, synchronous and batched |
| Known request boundaries | One sparse EOS-mask launch before local top-k and forced-token choices | Synchronous `min_tokens` and reasoning-budget boundaries, including sampled q |
| Draft smoothing alpha / GPTQ damping | Preparation and new packs only | Draft reader namespaces; packed formats and steady-state shapes unchanged |

The empty profile preserves alpha=1, smoothing alpha=0.5, and GPTQ damping=0.01.
Existing serving defaults remain FC FP8, automatic committed-decode calibration,
and rejection diagnostics. K comes from the serving build (currently seven
after #869); profiles never change K or the candidate count. FP32 KDA state is
unchanged. Production pins the empty tuning profile; the experimental boot can
load an explicit profile before preparation and graph capture:

```sh
STK_draft_tuning=/absolute/path/draft-profile.json <existing native boot command>
```

Every TP rank must load the same content. A CPU preparation vote rejects parse
errors, unsupported names, missing required calibration, or differing profile
digests before tuned packing starts. The digest is exposed in the boot gauges
and `st:lane_info`, alongside the active selector trace interval. Do not change
the profile underneath a running server: it is bound at boot.

## Profile schema

```json
{
  "version": 1,
  "selector_alpha": [1.0],
  "request_boundaries": true,
  "smoothing_alpha": {},
  "gptq_damping": {},
  "trace_every": 0
}
```

This example enables only request boundaries. A single selector coefficient
broadcasts to the serving K; a position-specific list must contain exactly K
finite values in [0, 2]. Missing selector coefficients mean all ones.

`smoothing_alpha` maps `layers.L.input_layernorm.weight` or
`layers.L.post_attention_layernorm.weight` to a value in [0, 1].
`gptq_damping` maps prepared draft reader names, such as
`layers.0.self_attn.qkv`, `layers.0.mlp.gate_up`, or `fc.weight`, to values in
(0, 1]. The loader validates names against the actual drafter facts.
Unknown fields and nonfinite values are rejected; `evidence` is optional
provenance written by the fitting command.

Smoothing still covers every original consumer of the norm, including the
convolution projections. The context-KV path retains its unsmoothed readers.
The store includes the effective smoothing and any nondefault damping in pack
identity. Default cache keys remain reusable; a tuned damping never imports a
legacy pack that cannot attest that setting. FC damping also reaches its
separate committed-decode pack when that calibration is active.

## Selector fitting from retained onepass records

1. Set `trace_every` to a positive interval, for example 8, with the other
   coefficients at baseline. Keep rejection diagnostics enabled. Recording
   still uses the existing onepass latency sessions; no extra target pass is
   introduced.
2. Retain the combined `latency.jsonl` files. `draft_selector` rows contain
   target IDs and candidate unary/edge scores through the accepted prefix and
   the first mismatch only. Later target logits are counterfactual and are
   excluded. Only rank zero records its walk, because that is the TP-agreed
   proposal. Sampled, policy-modified and pending-minimum rows do not train the
   greedy selector.
3. Provide an explicit request-family split in `groups.json`. Keys are the
   recorded `request_token:seq`. Repeated prompts or variants in the same
   family must share a `sample_group` and split, across all recordings.

```json
{
  "recording_a:12": {"sample_group": "proof-family-a", "split": "train"},
  "recording_a:13": {"sample_group": "code-family-b", "split": "validation"}
}
```

```sh
python bench/draft_tune.py selector latency.jsonl --groups groups.json --out selector.json
```

Rows already tagged with `sample_group` and `split` can omit `--groups`. The
fitter selects from `{0, 0.5, 0.75, 1, 1.25}` on training requests. Validation
can only veto that selection: a tie, regression, or absent held-out position
retains alpha=1. Both request groups and all observed positions are reported,
including candidate coverage. Input and split manifests receive SHA256 hashes.

Trace buffers occupy `2 * slots * K * candidates * 4` bytes. Collection forces
synchronous decode so slot reuse cannot overwrite delayed labels; JSON readback
occurs every selected step. **Such onepass runs are calibration records, and
the steady-performance gate marks them invalid even if no eligible trace row
was emitted.** The recorder persists the instrumentation flag. Reboot with
`trace_every=0` for a timing comparison.

The fitted metric is held-out position agreement, not live prefix acceptance.
Changing an earlier selected token changes later predecessor edges and target
prefixes; this trace cannot establish that counterfactual trajectory. Use a
matched live comparison before promoting fitted coefficients.

## Preparation-only W4/A8 fit

The CPU fitting command evaluates one draft norm's fused QKV or gate/up reader
shard. Its bundle is a `torch.save` dictionary, loaded on CPU with
`weights_only=True`:

| Key | Required content |
|---|---|
| `norm_key` | Draft input or post-attention norm name |
| `norm_weight` | Original unsmoothed BF16 `[K_in]` norm weights |
| `weights` | One reader name mapped to its original BF16 `[N_shard,K_in]` matrix |
| `H`, `amax` | Unsmooth-domain calibration Hessian and per-channel activation peaks |
| `weight_peaks` | `[K_in]` maxima over **all full source readers of that norm**, including conv readers and other TP shards |
| `train_ids` | Nonempty calibration request-family IDs |
| `selection`, `selection_ids` | Held-out unsmoothed BF16 norm outputs `[M,K_in]`, one family ID per row |
| `validation`, `validation_ids` | Independent final-audit inputs with the same layout |

The three request-family sets must be disjoint. Adjacent activation rows from
one request cannot be assigned to different sets. Input columns must be aligned
to 128. Existing Hessian/peak blobs alone cannot supply the independent held-out
inputs, and the tool refuses an incomplete bundle. This change does not enable
another resident activation collector or retain original BF16 weight copies.

```sh
python bench/draft_tune.py packing draft-reader.pt --out packing.json
```

The grid uses smoothing alpha `{0.25,0.5,0.75}` and damping
`{0.005,0.01,0.02}`, always including baseline. It uses the existing power-of-two
fold, GPTQ W4 codes, kernel-equivalent W4 dequantization and A8 activation
reference. Selection minimizes output reconstruction error on the selection
families; the independent validation set can veto it. Inputs and source weights
are not modified. The report includes both errors and the exact chosen values.

This fitter covers the W4/A8 reader recipe, not an FP8 FC quality experiment.
The runtime can bind explicit damping to FP8 packs too, but this W4-only output
does not establish FP8 or whole-drafter quality. It also does not validate other
readers affected by a norm fold. Retain the report as a candidate filter and
require full drafter acceptance and quality evidence before using that norm
setting as a serving default.

## Request boundary scope and verification

Minimum-token EOS exclusions happen before each rank's top-k, allowing valid
replacement candidates to enter. Reasoning-end forcing can introduce a token
outside top-k; its actual codebook row feeds the following selector position.
Spontaneous reasoning ends cancel future forcing within that draft prefix.
Minimum-token exclusions retain precedence over forcing. Grammar requests and
requests with more than 32 distinct end IDs retain their existing proposal path.
Soft repetition/frequency/presence penalties and logit bias are not fitted here.

For sampled rows, forced insertion preserves unique candidate IDs and returns
the exact normalized q drawn by the proposal. Target logit processing, rejection
sampling, stop handling and the TP agreement remain authoritative.

CPU regression tests cover these distributions, request boundaries, held-out
vetoes, signed-fold preservation, trace ownership, graph-input reset, rank
agreement and pack identity. Linux CPU CI can import the Triton definitions;
the optional CUDA numerical/capture test is skipped there. No live acceptance,
GPU graph correctness or tok/s result is implied by these checks.
