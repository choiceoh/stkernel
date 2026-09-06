#!/usr/bin/env python3
"""Single-GPU correctness check for the candidate NVFP4 scale/alpha helper.

Requires the imported overlay to match --source-root. Compares the actual
Triton helper to the current PyTorch FP32 scale recipe. The dispatch check
stubs FP4 quantization/GEMM, exercising only the scale branch; it does not
claim that a noncontiguous real NVFP4 GEMM is supported. No model weights,
BPROJ quality, NCCL, timing, or throughput are tested here.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch


def _stock(x, weight_scale, alpha_scale):
    amax = x.abs().amax().float().clamp_min(1.0e-12)
    scale = (2688.0 / amax).view(1)
    return scale, (alpha_scale / (scale * weight_scale)).to(torch.float32)


def _equal(actual, expected, label):
    if (actual.dtype != torch.float32 or actual.numel() != 1
            or actual.device != expected.device
            or not torch.equal(actual.view(torch.int32), expected.view(torch.int32))):
        raise AssertionError(f"{label}: got {actual}, expected {expected}")


def _source(module_name, relative, source_root):
    module = importlib.import_module(module_name)
    actual = Path(module.__file__).resolve()
    expected = source_root / relative
    actual_sha = hashlib.sha256(actual.read_bytes()).hexdigest()
    if actual_sha != hashlib.sha256(expected.read_bytes()).hexdigest():
        raise RuntimeError(f"imported source mismatch: {actual} != {expected}")
    return module, {"path": str(actual), "sha256": actual_sha}


def _dispatch_check(dense, scale_module, x, weight_scale, alpha_scale, use_fused):
    import flashinfer

    called = {"fused": 0, "quant": 0, "gemm": 0}
    actual_helper = scale_module.activation_scale_alpha

    def traced_helper(*args):
        called["fused"] += 1
        return actual_helper(*args)

    def quant_stub(value, scale):
        called["quant"] += 1
        called["scale"] = scale.clone()
        # The following GEMM is also stubbed. These tensors are shape-only
        # placeholders and must never be mistaken for a real NVFP4 pack.
        return (torch.empty((value.shape[0], value.shape[1] // 2),
                            device=value.device, dtype=torch.uint8),
                torch.empty((1, 1), device=value.device, dtype=torch.uint8))

    def gemm_stub(a, b, a_sf, b_sf, alpha, dtype, out, *args):
        called["gemm"] += 1
        called["alpha"] = alpha.clone()
        out.zero_()
        return out

    dummy_weight = torch.empty((128, x.shape[-1] // 2), device=x.device, dtype=torch.uint8)
    dummy_scale = torch.empty((1, 1), device=x.device, dtype=torch.uint8)
    before = x.clone()
    with patch.object(scale_module, "activation_scale_alpha", traced_helper), \
            patch.object(flashinfer, "nvfp4_quantize", quant_stub), \
            patch.object(flashinfer, "mm_fp4", gemm_stub):
        dense._nvfp4_dense_gemm(x, dummy_weight, dummy_scale, weight_scale, 128, alpha_scale)
    expected_scale, expected_alpha = _stock(x.reshape(-1, x.shape[-1]), weight_scale, alpha_scale)
    _equal(called["scale"], expected_scale, "dispatch scale")
    _equal(called["alpha"], expected_alpha, "dispatch alpha")
    if (called["fused"] != int(use_fused) or called["quant"] != 1
            or called["gemm"] != 1 or not torch.equal(x, before)):
        raise AssertionError(f"incorrect scale dispatch or input mutation: {called}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["VLLM_GLM53_NVFP4_SCALE_FUSED"] = "1"
    global torch
    import torch

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) != (12, 1):
        raise RuntimeError("this pinned GLM candidate probe requires SM121")
    root = args.source_root.resolve()
    scale_module, scale_source = _source(
        "vllm.model_executor.layers.glm53_nvfp4_scale",
        "overlay/modules/glm53_model/glm53_nvfp4_scale.py", root,
    )
    dense, dense_source = _source(
        "vllm.model_executor.layers.glm53_fp8_dense",
        "overlay/modules/glm53_model/glm53_fp8_dense.py", root,
    )
    if not dense._NVFP4_SCALE_FUSED:
        raise RuntimeError("candidate scale knob was not latched before import")
    generator = torch.Generator(device=device).manual_seed(20260906)
    checks = []
    for shape in ((128, 512), (512, 1536), (8192, 4096), (129, 513), (2, 31, 128)):
        for case in ("zeros", "random", "outlier", "small_finite"):
            x = torch.randn(shape, device=device, generator=generator).to(torch.bfloat16)
            if case == "zeros":
                x.zero_()
            elif case == "outlier":
                x.reshape(-1)[::8191] = -(2.0 ** 60)
                x.reshape(-1)[-1] = 2.0 ** 61
            elif case == "small_finite":
                x.mul_(2.0 ** -60)
            before = x.clone()
            for weight_value in (0.5, 123.5, 1.0e-8):
                weight_scale = torch.tensor([weight_value], device=device, dtype=torch.float32)
                weight_before = weight_scale.clone()
                for alpha_scale in (1.0, -1.0):
                    actual = scale_module.activation_scale_alpha(x, weight_scale, alpha_scale)
                    expected = _stock(x, weight_scale, alpha_scale)
                    _equal(actual[0], expected[0], f"{shape}/{case}: scale")
                    _equal(actual[1], expected[1], f"{shape}/{case}: alpha")
                    if not all(bool(torch.isfinite(item).all()) for item in actual):
                        raise AssertionError(f"nonfinite result for finite input: {shape}/{case}")
                if not torch.equal(weight_scale, weight_before):
                    raise AssertionError("weight scale mutated")
            if not torch.equal(x.view(torch.int16), before.view(torch.int16)):
                raise AssertionError("input mutated")
            checks.append({"shape": shape, "case": case, "fp32_recipe_bit_exact": True})

    weight_scale = torch.tensor([123.5], device=device, dtype=torch.float32)
    contiguous = torch.randn((128, 512), device=device).to(torch.bfloat16)
    noncontiguous = torch.randn((128, 1024), device=device).to(torch.bfloat16)[:, ::2]
    for x, use_fused in ((contiguous, True), (noncontiguous, False),
                         (contiguous.float(), False)):
        _dispatch_check(dense, scale_module, x, weight_scale, 1.0, use_fused)
    for x, ws in ((noncontiguous, weight_scale), (contiguous.float(), weight_scale),
                  (contiguous[:0], weight_scale),
                  (contiguous, weight_scale.to(torch.bfloat16)),
                  (contiguous, weight_scale.expand(2))):
        try:
            scale_module.activation_scale_alpha(x, ws, 1.0)
        except ValueError:
            continue
        raise AssertionError("helper accepted an unsupported layout/dtype/scale")
    torch.cuda.synchronize(device)
    report = {"status": "passed", "sources": {"scale": scale_source, "dense": dense_source},
              "checks": checks, "dispatch": "contiguous BF16 fused; strided/FP32 stock scale",
              "scope": "scale/alpha only; FP4 GEMM stubbed for dispatch checks"}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
