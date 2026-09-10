"""Standalone MHC component oracles; no model, vLLM or TileLang import.

The legacy fused seam and the released V4.1 seam are deliberately distinct.
V4.1 rounds post before projection, rounds pre before its RMS statistic, and
consumes a pre coefficient supplied by the preceding sublayer. Merely extending
the legacy kernel to H5120 does not establish V4.1 model equivalence.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

TOKENS = (1, 5, 6, 8, 10, 12, 16, 17, 24, 32)
HIDDENS = (4096, 5120)
NAMES = ("residual", "post_mix", "comb_mix", "layer_input")
TOL = 1e-3
V41_REFERENCE = {
    "revision": "fb2764a5cf321eaa5070ca8f9e892818f477c16d",
    "model_sha256": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "kernel_sha256": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "scope": "inference/model.py Block.hc_mixes/hc_pre/hc_post and RMSNorm; kernel.py hc_split_sinkhorn",
}


def geometry_eligible(tokens, hc, hidden):
    return (all(type(v) is int for v in (tokens, hc, hidden))
            and 1 <= tokens <= 32 and hc == 4 and hidden in HIDDENS)


def validate_metadata(shapes, dtypes):
    """CPU-only contract; no tensor library or device initialization required."""
    names = ("x", "residual", "post", "comb", "fn", "scale", "base", "norm")
    if tuple(shapes) != names or tuple(dtypes) != names:
        raise ValueError("MHC inputs must have the exact named contract")
    if len(shapes["x"]) != 2:
        raise ValueError("x must be [T,H]")
    t, h = shapes["x"]
    if not geometry_eligible(t, 4, h):
        raise ValueError("unsupported MHC geometry")
    expected = ((t, h), (t, 4, h), (t, 4), (t, 4, 4),
                (24, 4 * h), (3,), (24,), (h,))
    expected_dt = ("torch.bfloat16", "torch.bfloat16", "torch.float32",
                   "torch.float32", "torch.float32", "torch.float32",
                   "torch.float32", "torch.bfloat16")
    for name, shape, dtype in zip(names, expected, expected_dt):
        if tuple(shapes[name]) != shape or str(dtypes[name]) != dtype:
            raise ValueError(f"invalid {name} shape or dtype")
    return t, h


def _inputs(values):
    names = ("x", "residual", "post", "comb", "fn", "scale", "base", "norm")
    tensors = dict(zip(names, values))
    t, h = validate_metadata({k: tuple(v.shape) for k, v in tensors.items()},
                             {k: str(v.dtype) for k, v in tensors.items()})
    if len({v.device for v in values}) != 1:
        raise ValueError("MHC inputs must share a device")
    return t, h


def _parameters(rms_eps, norm_eps, pre_eps, sinkhorn_eps, post_mult, sinkhorn_iters):
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (rms_eps, norm_eps, pre_eps, sinkhorn_eps, post_mult))
            or type(sinkhorn_iters) is not int or sinkhorn_iters < 1):
        raise ValueError("invalid MHC arithmetic parameters")


def split_sinkhorn(mixes, scale, base, *, pre_eps=1e-6, sinkhorn_eps=1e-6,
                   post_mult=2.0, sinkhorn_iters=20):
    import torch
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + pre_eps
    post = torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8]) * post_mult
    cm = (mixes[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
    cm = torch.exp(cm - cm.amax(dim=-1, keepdim=True))
    # The first row normalization adds epsilon AFTER division.
    cm = cm / cm.sum(dim=-1, keepdim=True) + sinkhorn_eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    return pre, post, cm


def _projection(value, fn, rms_eps):
    import torch
    flat = value.flatten(1).float()
    # Avoid TF32 silently weakening the FP32 mathematical oracle on CUDA.
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        projected = torch.nn.functional.linear(flat, fn)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return projected * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)


def legacy_fused_reference(x, residual, post, comb, fn, scale, base, norm, *,
                           rms_eps=1e-6, norm_eps=1e-6, pre_eps=1e-6,
                           sinkhorn_eps=1e-6, post_mult=2.0, sinkhorn_iters=20):
    """Existing small-M fused math, including its two asymmetric BF16 seams.

    FP32 reductions are mathematical references, not a bit-exact CUDA FMA tree.
    No caller tensors are changed. Each returned tensor owns fresh storage.
    """
    import torch
    _inputs((x, residual, post, comb, fn, scale, base, norm))
    _parameters(rms_eps, norm_eps, pre_eps, sinkhorn_eps, post_mult, sinkhorn_iters)
    r32 = post.unsqueeze(-1) * x.float().unsqueeze(1)
    for k in range(4):
        r32 = r32 + comb[:, k, :].unsqueeze(-1) * residual[:, k, :].float().unsqueeze(1)
    rounded = r32.to(torch.bfloat16)
    mixes = _projection(r32, fn, rms_eps)  # deliberately NOT rounded
    pre, pm, cm = split_sinkhorn(mixes, scale, base, pre_eps=pre_eps,
                                sinkhorn_eps=sinkhorn_eps, post_mult=post_mult,
                                sinkhorn_iters=sinkhorn_iters)
    weighted = torch.zeros_like(x, dtype=torch.float32)
    for k in range(4):
        weighted = weighted + pre[:, k:k + 1] * rounded[:, k, :].float()
    denominator = torch.rsqrt(weighted.square().mean(-1, keepdim=True) + norm_eps)
    result = (weighted.to(torch.bfloat16).float() * denominator * norm.float()).to(torch.bfloat16)
    return rounded, pm, cm, result


def v41_component_reference(x, residual, post, comb, fn, scale, base, norm,
                            previous_pre, *, rms_eps=1e-20, norm_eps=1e-20,
                            pre_eps=1e-6, sinkhorn_eps=1e-6, post_mult=2.0,
                            sinkhorn_iters=20):
    """Released HF V4.1 seam. Fifth output is the pre mix carried forward.

    This is a component reference, not evidence that a serving image implements
    the model. ``previous_pre`` must come from the correct preceding sublayer.
    """
    import torch
    t, _ = _inputs((x, residual, post, comb, fn, scale, base, norm))
    _parameters(rms_eps, norm_eps, pre_eps, sinkhorn_eps, post_mult, sinkhorn_iters)
    if (tuple(previous_pre.shape) != (t, 4) or previous_pre.dtype != torch.float32
            or previous_pre.device != x.device):
        raise ValueError("previous_pre must be a matching [T,4] FP32 tensor")
    # Preserve HF's sum-then-add order and its materialized post output.
    r32 = post.unsqueeze(-1) * x.unsqueeze(1) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(2), dim=1)
    rounded = r32.to(x.dtype)
    pre, pm, cm = split_sinkhorn(_projection(rounded, fn, rms_eps), scale, base,
                                pre_eps=pre_eps, sinkhorn_eps=sinkhorn_eps,
                                post_mult=post_mult, sinkhorn_iters=sinkhorn_iters)
    weighted = torch.sum(previous_pre.unsqueeze(-1) * rounded.float(), dim=1).to(x.dtype)
    weighted32 = weighted.float()
    result = (norm.float() * (weighted32 * torch.rsqrt(
        weighted32.square().mean(-1, keepdim=True) + norm_eps))).to(x.dtype)
    return rounded, pm, cm, result, pre


def compare_outputs(got, reference, *, tolerance=TOL):
    import torch
    if len(got) != len(reference) or len(got) not in (4, 5):
        raise ValueError("MHC output count mismatch")
    if not math.isfinite(tolerance) or tolerance < 0 or tolerance > TOL:
        raise ValueError("invalid or weakened numerical tolerance")
    rows = []
    for name, actual, expected in zip(NAMES + ("next_pre",), got, reference):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"{name} output metadata mismatch")
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        diff = (actual.float() - expected.float()).norm().item() if finite else math.inf
        den = expected.float().norm().item() if finite else math.inf
        finite = finite and math.isfinite(diff) and math.isfinite(den)
        error = diff / den if den > 0 else (0. if diff == 0 else math.inf)
        worst = math.inf
        if finite:
            delta_rows = (actual.float() - expected.float()).reshape(actual.shape[0], -1).norm(dim=1)
            reference_rows = expected.float().reshape(expected.shape[0], -1).norm(dim=1)
            row_error = torch.where(reference_rows > 0, delta_rows / reference_rows,
                                    torch.where(delta_rows == 0, 0., math.inf))
            worst = row_error.amax().item()
        passed = (finite and math.isfinite(error) and error <= tolerance
                  and math.isfinite(worst) and worst <= tolerance)
        rows.append({"output": name, "relative_l2": error if math.isfinite(error) else None,
                     "worst_token_relative_l2": worst if math.isfinite(worst) else None,
                     "finite": finite, "exact": bool(torch.equal(actual, expected)),
                     "passed": passed})
    return rows


def fixture(tokens, hidden, *, device="cpu", seed=0):
    import torch
    if not geometry_eligible(tokens, 4, hidden):
        raise ValueError("unsupported fixture geometry")
    generator = torch.Generator(device=device).manual_seed(seed)
    def random(shape, dtype, scale=1.0):
        return (torch.randn(shape, device=device, dtype=torch.float32,
                            generator=generator) * scale).to(dtype)
    return (random((tokens, hidden), torch.bfloat16, .1),
            random((tokens, 4, hidden), torch.bfloat16, .1),
            random((tokens, 4), torch.float32, .5),
            random((tokens, 4, 4), torch.float32, .25),
            random((24, 4 * hidden), torch.float32, .02),
            torch.tensor([.8, 1.1, .7], dtype=torch.float32, device=device),
            random((24,), torch.float32, .2),
            random((hidden,), torch.bfloat16, .5))


def compile_extension(driver):
    """Compile/export admission without allocating or initializing CUDA."""
    import torch
    import torch.utils.cpp_extension as ce
    if (os.environ.get("MK_PROBE_NO_GPU") != "1"
            or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(glob.glob(p) for p in ("/dev/nvidia*", "/dev/kfd", "/dev/dri"))):
        raise RuntimeError("compile-only requires explicit isolation and no GPU device nodes")
    if torch.cuda.is_initialized():
        raise RuntimeError("compile-only entered with an initialized CUDA context")
    if getattr(driver, "_EXT", None) is not None:
        raise RuntimeError("compile-only requires a fresh extension load")
    original_load = ce.load
    calls = []
    def resource_load(**kwargs):
        kwargs["extra_cuda_cflags"] = list(kwargs.get("extra_cuda_cflags", ())) + [
            "-Xptxas=-v", "-Xptxas=--warn-on-spills"]
        if "arch=compute_121a,code=sm_121a" not in kwargs["extra_cuda_cflags"]:
            raise RuntimeError("compile-only requires the explicit SM121a target")
        kwargs["verbose"] = True
        calls.append({"cuda_flags": kwargs["extra_cuda_cflags"],
                      "sources": list(kwargs["sources"])})
        return original_load(**kwargs)
    try:
        ce.load = resource_load
        ext = driver._build()
    finally:
        ce.load = original_load
    if torch.cuda.is_initialized():
        raise RuntimeError("compile-only unexpectedly initialized CUDA")
    if len(calls) != 1:
        raise RuntimeError("compile-only did not observe one full translation-unit load")
    required = ("run_mhc", "run_mhc_v41")
    if not all(callable(getattr(ext, name, None)) for name in required):
        raise RuntimeError("compiled MHC extension is missing a required export")
    return {"exports": list(required), "cuda_initialized_before": False,
            "cuda_initialized_after": False, "device_nodes_absent": True,
            "load": calls[0]}


def _collapse_fixture(tokens, device, seed):
    import torch
    generator = torch.Generator(device=device).manual_seed(seed + 10101)
    return torch.rand(tokens, 4, generator=generator, device=device, dtype=torch.float32)


def _stock_comparison(values, actual, params, contract, setting):
    if setting == "off" or contract != "legacy" or values[0].shape[0] > 16:
        return {"status": "not_applicable", "scope": "old small-M fused wrapper only"}
    try:
        from vllm.model_executor.kernels.mhc import tilelang as tl
    except ModuleNotFoundError as error:
        if not (error.name == "vllm" or error.name.startswith("vllm.")):
            raise
        if setting == "require":
            raise RuntimeError("required stock wrapper is unavailable") from error
        return {"status": "unavailable", "numerical_evidence": False}
    # This process owns the wrapper; prevent its optional MK hook from turning
    # the nominal stock arm into the candidate. Do not swallow launch failures.
    resolver = getattr(tl, "_deneb_mk_hook", None)
    if resolver is not None:
        tl._deneb_mk_hook = lambda: None
    try:
        function = getattr(tl, "_mhc_fused_post_pre_tilelang_impl", None)
        function = function or tl.mhc_fused_post_pre_tilelang
        x, residual, post, comb, fn, scale, base, norm = values
        reference = function(x, residual, post, comb, fn, scale, base,
                             params["rms_eps"], 1e-6, 1e-6, 2., 20,
                             norm_weight=norm, norm_eps=params["norm_eps"])
        reference = (reference[0], reference[1].reshape(-1, 4),
                     reference[2].reshape(-1, 4, 4), reference[3])
        checks = compare_outputs(actual, reference)
        if not all(row["passed"] for row in checks):
            raise RuntimeError(f"available stock wrapper numerical failure: {checks}")
        return {"status": "compared", "outputs": checks}
    finally:
        if resolver is not None:
            tl._deneb_mk_hook = resolver


def _gpu_case(driver, hidden, tokens, contract, stock, edge="random"):
    import torch
    seed = tokens + hidden
    def make_values(chosen_seed):
        result = fixture(tokens, hidden, device="cuda", seed=chosen_seed)
        if edge in ("zero", "near_zero"):
            factor = 0. if edge == "zero" else 1e-8
            result[0].mul_(factor)
            result[1].mul_(factor)
        return result
    values = make_values(seed)
    collapse = _collapse_fixture(tokens, "cuda", seed)
    original = tuple(v.clone() for v in values) + (collapse.clone(),)
    params = dict(rms_eps=1e-20 if contract == "v41" else 1e-6,
                  norm_eps=1e-20 if contract == "v41" else 1e-6)
    def reference():
        if contract == "legacy":
            return legacy_fused_reference(*values, **params)
        return v41_component_reference(*values, collapse, **params)
    def call():
        x, residual, post, comb, fn, scale, base, norm = values
        out = driver._mhc_call(x, residual, post, comb, fn, scale, base, norm,
                              tokens, params["rms_eps"], 1e-6, 1e-6, 2.,
                              params["norm_eps"], 20, _fp32_fn=True,
                              _ar_consumer=False, _contract=contract,
                              _collapse_pre_mix=collapse if contract == "v41" else None)
        return (out[0], out[1], out[2].reshape(tokens, 4, 4), out[3], *out[4:])
    expected = reference()
    actual = call()
    torch.cuda.synchronize()
    checks = compare_outputs(actual, expected)
    unchanged = all(torch.equal(v, r) for v, r in zip((*values, collapse), original))
    if not unchanged or not all(row["passed"] for row in checks):
        raise RuntimeError(f"{contract}/H{hidden}/T{tokens} eager failure: {checks}")
    stock_receipt = _stock_comparison(values, actual, params, contract, stock)
    eager_copy = tuple(v.clone() for v in actual)

    # MHC calls remain serialized, including different geometries/contracts.
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        call()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured = call()
    replay_checks = []
    for replacement_seed in (100 + tokens, seed):
        replacement = make_values(replacement_seed)
        replacement_pre = _collapse_fixture(tokens, "cuda", replacement_seed)
        for target, new in zip((*values, collapse), (*replacement, replacement_pre)):
            target.copy_(new)
        expected = reference()
        graph.replay()
        torch.cuda.synchronize()
        detail = compare_outputs(captured, expected)
        immutable = all(torch.equal(v, r) for v, r in
                        zip((*values, collapse), (*replacement, replacement_pre)))
        if not immutable or not all(row["passed"] for row in detail):
            raise RuntimeError(f"{contract}/H{hidden}/T{tokens} changed-input replay failed")
        replay_checks.append({"seed": replacement_seed, "outputs": detail,
                              "inputs_unchanged": immutable})
    if not all(torch.equal(v, r) for v, r in zip((*values, collapse), original)):
        raise RuntimeError("A-B-A fixture did not restore original input bytes")
    if not all(torch.equal(a, b) for a, b in zip(captured, eager_copy)):
        raise RuntimeError("same-input graph output differs from eager output")

    # Concurrent independent GEMM only: no second MHC/global-counter user.
    noise_stream = torch.cuda.Stream()
    left = torch.randn(256, 256, device="cuda")
    right = torch.randn(256, 256, device="cuda")
    noise_expected = left @ right
    noise_result = torch.empty_like(noise_expected)
    noise_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(noise_stream):
        for _ in range(16):
            torch.mm(left, right, out=noise_result)
    graph.replay()
    torch.cuda.synchronize()
    concurrent_exact = all(torch.equal(a, b) for a, b in zip(captured, eager_copy))
    if not concurrent_exact or not torch.equal(noise_result, noise_expected):
        raise RuntimeError("independent-stream GEMM/MHC output interference")
    if not all(torch.equal(v, r) for v, r in zip((*values, collapse), original)):
        raise RuntimeError("independent-stream run changed an MHC input")
    return {"hidden": hidden, "tokens": tokens, "contract": contract,
            "fixture": edge,
            "eager": checks, "inputs_unchanged": unchanged, "stock": stock_receipt,
            "changed_input_graph": replay_checks, "same_input_replay_exact": True,
            "independent_gemm_stream_exact": True, "concurrent_mhc_tested": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, default=Path(__file__).resolve().parents[1]
                        / "overlay/modules/glm53_megakernel/glm53_megakernel.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--contract", choices=("legacy", "v41", "all"), default="all")
    parser.add_argument("--stock", choices=("auto", "require", "off"), default="auto")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; preserve the previous receipt")
    import torch
    spec = importlib.util.spec_from_file_location("mhc_geometry_driver", args.driver)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    contracts = ("legacy", "v41") if args.contract == "all" else (args.contract,)
    receipt = {"schema": 1, "contracts": list(contracts), "v41_model_equivalence": False,
               "compile_only": args.compile_only, "gpu_numerics": False,
               "v41_reference_source": V41_REFERENCE, "tolerance": TOL,
               "driver_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest(),
               "cuda_sha256": hashlib.sha256(args.driver.with_suffix(".cu").read_bytes()).hexdigest(),
               "rows": [], "passed": False}
    try:
        if args.compile_only:
            receipt["compile"] = compile_extension(driver)
        else:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            if (props.major, props.minor, props.multi_processor_count) != (12, 1, 48):
                raise RuntimeError("this component probe requires GB10 cc12.1/48SM")
            driver._build()
            for contract in contracts:
                for hidden in HIDDENS:
                    for tokens in TOKENS:
                        receipt["rows"].append(_gpu_case(driver, hidden, tokens, contract, args.stock))
            if "v41" in contracts:
                for edge in ("zero", "near_zero"):
                    receipt["rows"].append(_gpu_case(driver, 5120, 6, "v41", "off", edge))
            receipt["gpu_numerics"] = True
        receipt["passed"] = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
