# NVIDIA ModelOpt NVFP4 serving adapter

This change connects the completed `st-glm53-modelopt-up-gate-v1` TP4 rank
files to ST's GLM loader, parameter specs, memory accounting and b12x W4A4
execution. It includes the first three dense MLPs. Native MTP stays excluded;
the existing DFlash2 drafter and its separate checkpoint remain in use.

The target is `/home/choiceoh/models/st-glm53-nvidia-tp4-9391` on srv3,
converted from NVIDIA revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
[Preshard provenance](../st_nvidia_preshard_20260912/README.md) records source
coverage and exact readback of all 5,224 rank tensors and 347 vision tensors.
No target weight file is rewritten by this adapter.

## Runtime contract

`facts.load` selects the quantization contract from the checkpoint metadata.
`rank_loader(expected_layout=...)` rejects a mismatched rank encoding before
arena allocation. ModelOpt ranks must carry four contiguous FP32 scales for
every dense/routed expert group. Bind rejects nonpositive/nonfinite scales
and unrepresentable FP32 products/reciprocals before any CUDA graph capture.

For each projection, let `a` be ModelOpt's `input_scale` and `w` be its
`weight_scale_2`. Its reconstruction is `x = q_x * s_x * a` and
`W = q_w * s_w * w`. The MMA kernels import the quantizer from
`flashinfer.cute_dsl.fp4_common`, which divides input by the supplied scale:

| b12x argument | Bound value | Purpose |
| --- | --- | --- |
| FC1 input global scale | `a13` | Quantize input in calibrated units |
| FC1 alpha | `a13 * w13` | Restore activation and weight magnitude |
| FC2 input scale | `a2` | Quantize the clamped SwiGLU output |
| FC2 alpha | `a2 * w2` | Restore both magnitudes after the second GEMM |

The two derived alpha tensors are created once at bind and retained with
the weight views; input scales alias the original FP32 parameters.
Across 3 dense and 42 routed layers the alphas occupy 96,792 bytes
per rank, excluding allocator alignment, within the existing workspace
budget. `_weight_views` carries the final alphas so dispatch does not apply
its automatic FC1 scale folding again. Direct micro uses a separate local
quantizer that multiplies by its scale; dispatch converts to reciprocal
scales for that backend. Tiny global scales are never folded
into E4M3 block scales, and no dense BF16 weight copies are introduced.

Dense MLPs use fixed E=1/top-k=1 routing (expert 0, weight 1), local I=3072,
followed by the existing TP all-reduce or native prefill reduction callback.
Routed layers keep E=288/top-k=8,
local I=512, their BF16 shared expert and existing TP reduction. Red Hat's
folded format keeps unit global scales and its original BF16 dense path.

NVIDIA metadata omits ST's `chat_template_mm_v2.jinja`. Direct boot uses the
bundled ST template when absent, and the launcher stages that template
explicitly. This preserves `thinking=False`, tools and multimodal request
semantics; NVIDIA's original template always opens a reasoning section.

## Validation

- All four real rank headers match all 1,306 logical parameter specs. All
  45 layers per rank have valid raw/derived scales: [contract receipt](real-contract.json).
- Original NVIDIA source projections versus the CPU reference lane:
  dense 0/1/2 and MoE 3/44, rank 0, 1 and 6 tokens; all 10 comparisons
  have zero output difference: [CPU oracle](cpu-oracle.json). MoE uses
  eight experts spread across IDs 0–287, nonuniform weights and zero routes.
- Real tokenizer, reasoning on/off, EOS, generation defaults, vision config
  and weight budget checks: [metadata smoke](metadata-smoke.log). The rank
  header holds 44.353085 GiB including 72,664 alignment bytes; logical specs
  hold 44.353017 GiB. Those two byte counts reconcile exactly.
- Unit/regression results on main `4364380d` are recorded in
  [CPU tests](main-4364380d-cpu-tests.log): 49 passed, one skipped for a
  missing DFlash2 metadata mount. With that mount restored, all three
  [budget tests](main-budget-tests.log) passed, including the skipped case.

Component GPU qualification is complete for the checked layers/shapes. The
[acceptance receipt](gpu-component-acceptance.json) selects 71 checks:
rank 0 layers 0/1/2/3/44 at 1/6/17/64/129/4096/8192 tokens, plus ranks
1/2/3 layers 0/3/44 at 1/17/129/4096 tokens. The largest normalized source
oracle error is 1.06952%. This compares execution of the same NVIDIA NVFP4
weights; it does not measure BF16 quantization loss or checkpoint quality.

The final path includes these necessary kernel corrections:

- Dense E=1 reserves at least 128 input rows. The old E=1/M=1 TMA layout
  collapsed to unsupported `UTMALDG.1D`; Compute Sanitizer located the fault.
  [CPU instruction receipt](dense-tma-cpu.json) checks the real allocator
  and verifies 2D transfers without a CUDA context. The GPU fault is gone.
- MMA receives raw ModelOpt input scales; the separate direct-micro helper
  receives reciprocals. The original inverse-scale binding caused zero output.
- The three dense layers use the static family's FP32 partial-sum buffer
  at every batch size. Short routed prefill also keeps FP32 accumulation.
- Layers whose scales cannot fit lossless SF6 use original E4M3 scale TMA
  with the same Q0/FP32 accumulation. Rank 0 layer 44 exercises this path.
  Its corrected [raw-scale receipt](gpu-raw-q0-rank0.json) supersedes layer
  44 in the earlier full-rank receipt, whose failed result is preserved.

The bounded probe reads expected projections directly from the original
checkpoint, independently of the preshard builder and scale swizzling.
CUDA uses a separate PTX activation-quantization oracle. It checks finite
outputs throughout, up to 32 evenly spaced reference rows for large inputs,
and repeated eager execution and CUDA graph replay separately. Scale views
are consumed in place exactly as in production.

Source-error acceptance remains <=2%. The original <=0.1% repeated-output
spread criterion was changed explicitly to allow either <=0.1% spread or
at most one adjacent BF16 value per element. A rank 1 dense 4096-token case
had 0.1269% spread but only one BF16 step from FP32 atomic order and final
rounding. Its original failed receipt remains unchanged. A separate
[final-criterion rerun](gpu-final-criterion-rank1.json) tests six cases with
32 eager repeats and 32 graph replays each and passes the updated criterion.
This does not accept the old multi-step BF16 atomic accumulation failures.

All 54 current CPU regressions pass in the serving image:
[final CPU tests](serving-regressions-final.log). No full-model output-quality,
performance, TP collective or production-promotion claim is made yet.

```bash
PYTHONPATH=. python3 probes/engine_modelopt_check.py \
  --source /home/choiceoh/models/glm53-nvidia-nvfp4-source-9391 \
  --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --device cuda --layers 0 1 2 3 44 --tokens 1 6 129 4096 \
  --moe-static t,r,sf6,q0 --out gpu-production.json
```

Run the stock control with `--moe-static stock` in a **new process**;
dispatch configuration is fixed before the first bind. Probe buffers and
graphs belong to the probe process, and it must only run in the coordinated
GPU window.

The requested NVIDIA default transition is gated on successful GPU and
fleet qualification. For that launch, the existing launcher accepts `CKPT` and
`RANKS_DIR` set to the NVIDIA target, with `DRAFTER` continuing to point to
the existing DFlash2 directory. Each node now holds its corresponding
rank plus `vision.safetensors`; srv3 also retains the complete target.
The fanout launcher transfers each node's assigned rank, vision and metadata
using partial files and verifies both weight files with SHA-256.
[Fanout receipt](fanout.log) records all three remote verifications; the
source rank was already checked during preshard validation. No existing
model was removed. Defaults remain unchanged until qualification passes.


## Red Hat versus NVIDIA: actual checkpoint comparison

[Checkpoint comparison](checkpoint-comparison.json) was generated by
`probes/compare_glm53_checkpoints.py` directly from both local HF sources.
Header counts include every indexed tensor; numerical comparisons are
explicit deterministic samples and use no GPU.

| Property | Red Hat | NVIDIA |
| --- | --- | --- |
| Language architecture | 45 layers, H4096, 288 experts, top-8 | Same numerical configuration |
| Vision configuration / tokenizer vocabulary and merges | GLM source | Identical |
| Dense MLP layers 0–2 | BF16, 0.843750 GiB | NVFP4, 0.237305 GiB |
| Routed experts, layers 3–44 | NVFP4, 159.469020 GiB | NVFP4, 159.469020 GiB |
| KDA A_log/dt_bias and router correction bias | 110 BF16 tensors | FP32, +585,600 bytes |
| Native MTP | FP8 experts; total 7.095107 GiB | BF16; total 13.844283 GiB |
| Actual TP4 rank 2 file | 44.504610 GiB | 44.353232 GiB |

Both ST configurations exclude native MTP and use the same DFlash2 checkpoint.
The extra NVIDIA raw download size is primarily the unused MTP, while its
served rank file is 155.010742 MiB smaller. These byte counts come from actual
headers and rank files, not the source index's estimated parameter count.
The raw numerical text config differs only by an added EP-plan annotation;
vision config and processor config are identical. NVIDIA's generation file
adds `do_sample=true` and `top_p=0.95`; comparison requests must pin sampling.
Red Hat tokenizer JSON carries a 2048-token truncation rule, NVIDIA does not;
ST's existing `boot.tokenizer` explicitly disables truncation for both.

The weight samples cover 57 projections (256 evenly spaced blocks of 16
weights each) and 92,286 values from 1,584 other compatible floating tensors.
Of those control tensors, 1,306 have identical sampled values; the largest
relative L2 difference among the remaining controls is 0.219102%.
NVIDIA dense NVFP4 versus Red Hat dense BF16 has 8.9436–9.5852% sampled
relative L2 difference, with cosine similarity 0.99542–0.99600. The 48
sampled routed projections compare two NVFP4 quantizations: median relative
L2 distance 3.9660%, range essentially zero to 9.0460%, minimum cosine
similarity 0.99590. These are weight distances, not output error or task
accuracy; routed samples cannot establish which checkpoint is closer to the
unquantized source without that reference.

The two formats encode global scales differently. For example, source L3
expert 0 up projection stores Red Hat `weight_global_scale=17280` (inverse
scale) versus NVIDIA `weight_scale_2=0.0000581287204` (dequantization scale).
ST's Red Hat preshard folds weight scales into E4M3 and uses dynamic unit
global activation scales; the NVIDIA adapter retains calibrated activation
and weight scales. The same ST build is required for an operational A/B,
while each loader still follows its checkpoint's format contract.

The initial KV12 attempt failed the rank 3 memory admission check before
arena allocation. Its rollback also used a stale source path with a newer
deployment environment, overwriting the newer release's engine directory.
The exact `abceb6a0` source and images were subsequently restored on all
four nodes; [incident and repair receipts](fleet-admission/incident.json)
preserve the failure and repair. The deployment environment was not changed
by that comparison attempt.

The KV7 comparison on srv2 used engine commit `dfd28cb03257`, image
`st-engine:nvidia-dfd28cb03257`, TP4, `t,r,sf6,q0`, DFlash2 K=5, and the
unchanged 2K/32K/128K onepass inputs with seed 7. The [Red Hat document
run](ab-kv7/redhat/result.jsonl) completed its five requests with exit 0:
the existing retrieval checks found 6 of 9 answers, and Korean corruption
was absent in all five outputs. The retrieval checker searches reasoning
and final content together; this is not a final-answer-only accuracy score.
The 32K and 128K TTFTs were 14.954 and 58.223 seconds. Raw DFlash acceptance
was 0.4787 and average tokens per step was 3.3935. These are one-run
observations, not a checkpoint speed comparison without the NVIDIA leg.

The subsequent [HTTP tool check](ab-kv7/redhat/http-full.json) failed with
503 and aborted the pair before NVIDIA started. A single synthetic tool
request reproduced the same crash on the configured Red Hat `abceb6a0`
release. After accepting an EOS draft, `Matcher.fill` attempted to fill a
mask from an already terminated xgrammar matcher. This shared engine bug
is independent of NVIDIA's quantization contract.

The small grammar repair stops the speculative walk after the EOS position
and rolls back the stop token with the other accepted drafts. The original
source fails both new real-xgrammar regressions; the fixed source passes
all 28 grammar tests. It was deployed as `prod-abceb6a0-grammar-9391`, keeping
the current Red Hat checkpoint, KV7 and qualified tile32 prefill. The
[live recovery receipt](ab-kv7/grammar-restored-http.json) records a successful
`get_weather(city="Seoul")` response followed by a normal Korean answer.
All four containers were running and the deployment environment was updated
to that repaired release only after these checks passed.

NVIDIA's full document run subsequently completed and **failed**. The
requested default checkpoint transition has therefore not been promoted.
The old pair runner's captured baseline predates the grammar repair; it must
not be rerun without refreshing and validating the restoration target.

## NVIDIA full-model qualification, 2026-09-12 07:33 UTC

The immutable candidate is `st-engine:nvidia-4f393351`, source SHA-256
`bfbfb6884846f760a825eeeedb39e854c9513b4e257ad7b54d49cd6078dbcdb7`.
It combines the ModelOpt adapter with the qualified tile32 MLA default and
grammar stop repair. All four nodes used their NVIDIA rank file, TP4, KV7,
production `t,r,sf6,q0`, and the unchanged DFlash2 K=5 checkpoint. Native MTP
remains excluded. The [recorded run](cutover-4f393351/result.jsonl) completed
all five synthetic document requests, with no unrelated completed requests
in its exclusive interval.

Short Seoul/Paris smoke requests passed. The 2K document retrieved all
three planted facts, but 32K and 128K retrieved none: **3/9** in the existing
combined reasoning/content gate. The 32K output repeated unrelated historical
text and contained one replacement character; 128K repeated punctuation and
spaces. [Outputs](cutover-4f393351/outputs.json) preserve those failures.
The subsequent quality assertion refused promotion before the final tool
check and deployment-environment mutation. Red Hat was restarted at
07:40:52 UTC, and its [HTTP status](cutover-4f393351/restored-status.json)
was verified ready. The default environment remains
`st-engine:prod-abceb6a0-grammar-9391` with Red Hat ranks. Neither an NVIDIA
default PR nor a default deployment was published by this qualification.

The earlier Red Hat result used stock MLA, so it is not a same-build A/B
against this tile32 candidate. Its observations and the failed NVIDIA
observations are retained separately; failure output cannot establish a
throughput improvement. In particular, NVIDIA's 128K 87.85 output tokens/s
comes from repetitive invalid output and must not be treated as useful
decode throughput. Prefill `prompt_tokens / TTFT` also includes first-token
work, shape JIT and prefix reuse; it is not isolated kernel throughput.

## Bounded routing diagnosis without a serving restart

The original component checks routed every token to the same eight expert
IDs. `probes/engine_modelopt_check.py --routes dispersed` now assigns distinct
top-8 routes per token across all 288 experts. Expected values are still read
from original NVIDIA projection tensors. Positional samples plus the largest
output row are checked, keeping the dequantized oracle's memory bounded.

On srv3/rank 2, the [initial four cases](cutover-4f393351/dispersed-rank2.json)
covered L3/L44 at 6912 and 129 tokens, including reuse of the large workspace
for the smaller shape. An [outlier check](cutover-4f393351/dispersed-extreme-rank2.json)
confirmed L44's largest output row against the original source. The other
[40 routed layers](cutover-4f393351/dispersed-layers4-43-rank2.json) passed at
the actual 6912-token prefill size, each routing across all 288 experts and
checking five source-oracle rows plus repeated eager and graph execution.
The largest normalized source error in that 40-layer run was 1.2122%.
The existing repeat acceptance remains <=0.1% normalized spread **or** at
most one BF16 step; some passing low-magnitude elements differ by more than
one BF16 step while satisfying the normalized bound.

These checks narrow the diagnosis; they do not reproduce real model
activations, validate all ranks, or exonerate the full runtime. The full-model
long-context failure remains unresolved and cannot yet be attributed to the
checkpoint itself. Production containers were not restarted for these probes.

## Decode, steps and prefill observations

The running Red Hat tile32 release was measured again without a restart:
[control receipt](cutover-4f393351/redhat-tile32-control/result.jsonl).
It returned 8/9 retrieval matches and no corruption. Its saved prefix tier
already contained the test documents: 32K and 128K TTFT were only 0.815 and
1.025 seconds. Those cache-hit timings cannot measure actual full prefill or
be compared to the cold NVIDIA run. A further identical-prompt control with
a new `cache_salt` was prepared, but existing production requests did not
drain within 60 seconds. It sent no benchmark requests, restored ingress,
and did not restart or interrupt any serving request.

| Observed decode metric | Running Red Hat tile32 | Failed NVIDIA tile32 |
| --- | ---: | ---: |
| 2K output tokens/s, time-weighted over three requests | 62.05 | 46.86 |
| 32K output tokens/s | 58.33 | 37.65, invalid output |
| 128K output tokens/s | 63.93 | 87.85, invalid repetition |
| All-request output tokens/s, excluding first-token time | 61.04 | 56.97, includes invalid output |
| Median interior step windows, steps/s | 17.95 | 15.95 |
| Reciprocal of the median step rate, ms/step | 55.71 | 62.70 |
| Committed tokens per speculative step | 3.3344 | 3.7178 |
| Raw DFlash acceptance | 46.6875% | 54.3562% |

Output throughput is `sum(completion_tokens - 1) / sum(decode_s)` over the
specified requests. The first token belongs to TTFT; SSE chunk gaps are not
individual token latencies. Step values are the harness's interior-window
medians, and tokens/step comes from whole-run counters. Their product is not
a replacement for measured output throughput. The failing repetitive output
also distorts draft acceptance and tokens/step, so their larger NVIDIA values
are not a quality-adjusted efficiency gain. This is one synthetic run per
configuration, with different source builds and cache histories.

For actual prefill observations the earlier stock-MLA Red Hat receipt remains
the available reference. Both runs used TP4, KV7 and DFlash2 K=5, but their
MLA implementation and compilation history differ:

| Context | Earlier Red Hat TTFT / prompt tokens per second | NVIDIA TTFT / prompt tokens per second |
| --- | ---: | ---: |
| 2K, best within-run warm request | 0.637 s / 3341 | 0.730 s / 2915 |
| 32K, single request | 14.954 s / 2176 | 21.408 s / 1520 |
| 128K, single request | 58.223 s / 2208 | 76.384 s / 1683 |

The 2K warm value uses normal prefix reuse within each run. NVIDIA also
compiled dense E=1 kernels for new input shapes during these measurements:
its first two 2K TTFTs were 6.652 and 5.694 seconds. Thus none of this table
is a clean checkpoint-only speed delta or isolated kernel benchmark.

The head-node serving-container history previously counted nine starts in
the comparison/transition interval. The corrected NVIDIA qualification added
the 07:33:13 UTC start, followed by Red Hat recovery at 07:40:52 UTC: **eleven
serving starts** in that interval. The later component and HTTP controls
added no serving-container restarts. The default NVIDIA source changes remain
unpublished because full-model qualification failed.
