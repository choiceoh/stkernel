"""Standalone MHC component oracles -- the seam's torch reference, no model or TileLang.

Restored from the retired fork-probe harness (the vLLM overlay decommission,
2026-09-18): the oracles are the engine's own verification asset, the bench
harness around them is not. The legacy fused seam and the released V4.1 seam
are deliberately distinct. V4.1 rounds post before projection, rounds pre
before its RMS statistic, and consumes a pre coefficient supplied by the
preceding sublayer. Merely extending the legacy kernel to H5120 does not
establish V4.1 model equivalence.
"""
from __future__ import annotations

import math

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
            and 1 <= tokens <= 128 and hc == 4 and hidden in HIDDENS)


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
