# Low-cost draft controls

The three follow-ups to the [low-cost review](GLM53_DRAFT_LOW_COST_REVIEW.md)
are connected to native ST serving. Selector FP32 output and automatic fitted
FC-bias discovery are enabled by default. Other fitted coefficients remain
explicit experiments; no checkpoint-fitted coefficients or higher acceptance
result are supplied with the code.
No fleet or GPU run was used to develop this change.

| Control | Serving work | Application |
|---|---|---|
| Position-specific selector edge strength | Constant selection and one multiply per candidate in the existing greedy walk; no new projection, launch or collective | Greedy and sampled proposals, synchronous and batched |
| Known request boundaries | One sparse EOS-mask launch before local top-k and forced-token choices | Synchronous `min_tokens` and reasoning-budget boundaries, including sampled q |
| Draft smoothing alpha / GPTQ damping | Preparation and new packs only | Draft reader namespaces; packed formats and steady-state shapes unchanged |
| FP32 selector projection output | Same BF16 operands and GEMM dimensions; removes the BF16 intermediate and widening operation | Greedy/sampled, synchronous/batched selector paths |
| Decode FC mean-error correction | FP32 vector add fused into the existing RMSNorm launch; 16 KiB per rank reserved in the compact arena | Matching fitted FP8 decode reader only; prefill has no correction |

The empty profile preserves alpha=1, smoothing alpha=0.5, and GPTQ damping=0.01.
Existing serving defaults remain FC FP8, automatic committed-decode calibration,
and rejection diagnostics. K comes from the serving build (currently seven
after #869); profiles never change K or the candidate count. FP32 KDA state is
unchanged. Production pins the empty tuning profile, which now enables FP32
selector output and looks for `/cache/draft-fc-bias.json`. Without a valid fitted
artifact, FC correction is absent and boot continues normally. The experimental
boot can load an explicit profile before preparation and graph capture:

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
  "selector_projection_fp32": true,
  "fc_bias_auto": true,
  "fc_bias": {},
  "trace_every": 0
}
```

This example adds request boundaries to the defaults. A single selector coefficient
broadcasts to the serving K; a position-specific list must contain exactly K
finite values in [0, 2]. Missing selector coefficients mean all ones.

`smoothing_alpha` maps `layers.L.input_layernorm.weight` or
`layers.L.post_attention_layernorm.weight` to a value in [0, 1].
`gptq_damping` maps prepared draft reader names, such as
`layers.0.self_attn.qkv`, `layers.0.mlp.gate_up`, or `fc.weight`, to values in
(0, 1]. The loader validates names against the actual drafter facts.
Unknown fields and nonfinite values are rejected; `evidence` is optional
provenance written by the fitting command.

`selector_projection_fp32: true` retains the BF16 matmul's output in FP32,
using `torch.mm(..., out_dtype=torch.float32)` on CUDA. The CPU reference
multiplies FP32-widened BF16 operands. It removes one output rounding boundary;
it does not change the source weight, candidate count, or recurrent state.
FP32 output is the default, including when the field is omitted. Set it to
`false` for the BF16-output path. Identical GEMM dimensions do not guarantee
identical kernel latency. Selector trace records retain this flag, and the
selector fitter carries it into its result and rejects mixtures of the two
projection modes. Older traces without this field still mean BF16 output.

`fc_bias` maps every TP rank number (string keys) to an object with
`reader_sha256` and `values`, a finite FP32 vector of exactly `hidden_size`
entries. Use the fitter below to produce the full map. Ranks share the same JSON
and profile digest, while each consumes its own correction. Missing ranks and
W4 FC policies are refused during preparation. After packing, a second vote
checks each correction against its exact source FC, executed FP8 bytes/scales,
hidden-norm weight, epsilon, local reader source and runtime version identifiers.
This binds a vector to the reader it was fitted on; it does not certify the
full target model, workload quality, or an unversioned external library build.
Keep the calibration run's full runtime/target identity alongside the bundle.

`fc_bias_auto` defaults to `true`. When `fc_bias` is empty, native boot reads
the CPU fitter's complete artifact from `/cache/draft-fc-bias.json`. It requires
selected fits and strictly improved held-out FC and normalized errors for every
rank. Every rank must read identical bytes. Only the correction is imported;
this cache file cannot alter selector or other tuning settings. File reads are
bounded to 8 MiB. Missing, malformed, mismatched or stale automatic artifacts
disable correction across all ranks and record the reason without blocking boot.
An explicit nonempty `fc_bias` takes precedence and retains strict error handling.
Set `fc_bias_auto: false` with an empty `fc_bias` to disable correction for A/B.

Correction identity checks run only after every rank has an agreed bias candidate.
They stream at most 128 weight rows to the CPU at once and never retain a second
whole FC weight. The vector is copied into its own compact arena region before
capture. No reference FC, collector, hashing, CPU readback, or extra collective
runs in steady-state decoding. Both the synchronous committed path and the
batched/precomputed context path apply the correction, including when calibration
uses the shared pack. Even a one-token prefill keeps the ordinary normalization.
Boot gauges and `st:lane_info` distinguish automatic discovery from an applied
correction: `draft_fc_bias_auto`, `draft_fc_bias` (lane) / `draft_fc_bias_applied`
(boot), and `draft_fc_bias_status` (`missing`, `disabled`, `applied-auto`,
`applied-profile`, or `skipped: ...`). The tuning digest includes the discovered
artifact identity or its absence. No artifact means no reader hashing or vector
allocation; the existing arena reservation remains 16 KiB per rank.

To disable both defaults in an explicit tuning profile:

```json
{"version": 1, "selector_projection_fp32": false, "fc_bias_auto": false, "fc_bias": {}}
```

## Fitting the FC correction

Collection is an explicit preparation/research operation, not an automatic boot
or serving observer. It needs an already prepared drafter whose BF16 sources
were retained (`prepare_fast(..., consume_weights=False, compact_into=None)`).
Use the same resolved FP8 policy/packs as the intended serving run. The helper
does not boot an engine, contact a server, reserve a GPU, or capture a graph.

```python
from bench.draft_fc_bias import collect_fc_pairs

# Each item contains actual committed-decode input rows on the reader's device:
# aux [M, 20480] BF16, 1 <= M <= 32; keep [M] bool; ids [M] request-family
# strings; split is "train" or "validation". Preserve the original M, including
# rejected/ghost rows. keep selects only the committed rows to retain.
bundle = collect_fc_pairs(drafter, batches, max_rows=4096)
torch.save(bundle, f"rank{drafter.target.comm.rank}-fc-pairs.pt")
```

The saved pairs contain the actual FP8/A8/BF16-output reader result and the
original BF16 FC result, rather than a weight-error approximation that omits
activation quantization. Collection adds a teacher GEMM and readback, so its
latencies are not steady serving measurements. At most 4,096 retained rows
produce 64 MiB of paired BF16 FC outputs per rank, plus small metadata; CPU
concatenation temporarily holds another copy of the split being joined. The
caller owns the original input storage. Exceeding the chosen bound is an error.

Fit the pairs on CPU, without Triton or CUDA:

```sh
python bench/draft_tune.py fc-bias rank0-fc-pairs.pt \
  --peer-bundle rank1-fc-pairs.pt --peer-bundle rank2-fc-pairs.pt \
  --peer-bundle rank3-fc-pairs.pt --out fc-bias.json
```

Training estimates `mean(reference - actual)` in FP64 and stores the vector in
FP32. Entire request families are held out, including across ranks. Validation
may only veto: both FC reconstruction error and error after the actual
FP32-add/BF16-normalization rounding must improve on every rank. Otherwise the
output has an empty `fc_bias` map. The command records the input file hashes,
row/family counts, baseline/candidate errors, and `live_acceptance: false`.
It never substitutes fitted residual error for measured speculative acceptance.

For automatic application on the next boot, place the identical generated
`fc-bias.json` at `/cache/draft-fc-bias.json` on every TP node (write a temporary
file and rename it into place). It must remain the fitter artifact with only
`version`, `fc_bias`, and `evidence`; use the explicit tuning profile for other
controls. Selector FP32 is already the default. Preserve explicitly chosen
tuning fields when combining results into an explicit profile. No checkpoint-fitted
bias vector ships with the code, and boot never starts collection or fitting.

For the separate offline weight-rounding study, see
[GPTQ/FP4 rounding research](GLM53_DRAFT_ROUNDING_RESEARCH.md).

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
