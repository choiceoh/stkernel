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
`W = q_w * s_w * w`. The b12x quantizer multiplies its input by a scale:

| b12x argument | Bound value | Purpose |
| --- | --- | --- |
| FC1 input global scale | `1 / a13` | Quantize input in calibrated units |
| FC1 alpha | `a13 * w13` | Restore activation and weight magnitude |
| FC2 input scale | `1 / a2` | Quantize the clamped SwiGLU output |
| FC2 alpha | `a2 * w2` | Restore both magnitudes after the second GEMM |

These derived FP32 tensors are created once at bind and retained with the
weight views. Across 3 dense and 42 routed layers they occupy 193,584 bytes
per rank, excluding allocator alignment, within the existing workspace
budget. `_weight_views` carries the final alphas so dispatch does not apply
its automatic FC1 scale folding again. Tiny global scales are never folded
into E4M3 block scales, and no dense BF16 weight copies are introduced.

Dense MLPs use fixed E=1/top-k=1 routing (expert 0, weight 1), local I=3072,
followed by the existing TP all-reduce. Routed layers keep E=288/top-k=8,
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
- Unit/regression results are recorded in [CPU tests](final-cpu-tests.log).

GPU numeric comparison and CUDA graph replay are **pending the shared GPU
slot**. The independent ST 128K diagnosis currently owns all four nodes;
this adapter has not interrupted it. No full-model output-quality,
performance, TP collective or production-promotion claim is made here.

The bounded probe is `probes/engine_modelopt_check.py`. It loads one layer
at a time and reads its oracle directly from the original checkpoint,
without using the preshard builder or scale-swizzle code to construct the
expected projections. CUDA mode uses an independent PTX reciprocal/FP8
rounding oracle and checks repeated eager runs and graph replay separately.
Acceptance is fixed before the run: normalized maximum output error <=2%,
eager repeat and graph replay spread <=0.1%, finite output throughout.
For a large prefill it checks up to 32 evenly spaced output rows against
the oracle while checking finiteness over the entire output.

```bash
PYTHONPATH=. python3 probes/engine_modelopt_check.py \
  --source /home/choiceoh/models/glm53-nvidia-nvfp4-source-9391 \
  --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --device cuda --layers 0 1 2 3 44 --tokens 1 6 129 4096 \
  --moe-static stock --out gpu-stock.json
```

Run the same probe with `--moe-static t,r,sf6,q0` in a **new process**;
dispatch configuration is fixed before the first bind. Probe buffers and
graphs belong to the probe process, and it must only run in the coordinated
GPU window.

For a future fleet launch, the existing launcher accepts `CKPT` and
`RANKS_DIR` set to the NVIDIA target, with `DRAFTER` continuing to point to
the existing DFlash2 directory. Each node must first have its corresponding
rank plus `vision.safetensors`; currently the complete target is on srv3.
These defaults have not been changed to promote an unqualified checkpoint.
