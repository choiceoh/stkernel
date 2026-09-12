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

A same-build A/B is prepared on srv2 under
`/home/choiceoh/st-checkpoint-ab-9391/{redhat,nvidia}`. Both use engine commit
`dfd28cb03257`, image `st-engine:nvidia-dfd28cb03257`, TP4, KV12,
`t,r,sf6,q0`, DFlash2 K=5, identical 2K/32K/128K onepass inputs and seed 7.
Separate tier/dump directories prevent KV reuse across checkpoints. Runners
check the engine source hash, checkpoint mount and container identity and
restore the pinned Red Hat production release on exit. The fleet currently
belongs to another `st-tile32-debug` run; neither A/B leg has been launched.
