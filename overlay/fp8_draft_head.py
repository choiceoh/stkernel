# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rowwise FP8 (e4m3) quantization for the DSpark draft LM head.

Backported from vLLM PR #47584 at commit
e1321ff59585af3d09fe68a94904fc527c6f0f7a.  The caller must construct the
quantized copy after the target ``lm_head`` has been aliased onto the draft
model and before CUDA graph capture.

This module intentionally does not read an environment variable.  The DSpark
model overlay owns the opt-in gate, keeping this helper importable for CPU
tests and making the disabled path behaviorally inert.
"""

from typing import NamedTuple

import torch


_FP8_MAX = 448.0


class Fp8DraftHead(NamedTuple):
    """Rowwise-quantized copy of a vocab-sharded LM head."""

    # [num_local_vocab, hidden], one e4m3 scale per vocabulary row.
    weight_fp8: torch.Tensor
    # [1, num_local_vocab], laid out for the GEMM epilogue.
    row_scale: torch.Tensor
    # fp32 scalar 1.0 required by torch._scaled_mm.
    unit_scale: torch.Tensor


def fp8_draft_head_supported(device: torch.device | None = None) -> bool:
    """Return whether this torch/CUDA build exposes the required FP8 API."""

    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def require_fp8_draft_head_support(device: torch.device | None = None) -> None:
    """Fail closed when an explicitly enabled experiment cannot run."""

    if fp8_draft_head_supported(device):
        return
    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    raise RuntimeError(
        "VLLM_DSPARK_FP8_DRAFT_HEAD=1 requires CUDA SM89+, "
        "torch.float8_e4m3fn, and torch._scaled_mm; "
        f"found torch={torch_version}, torch_cuda={cuda_version!r}"
    )


def quantize_draft_head(weight: torch.Tensor) -> Fp8DraftHead:
    """Quantize ``[num_local_vocab, hidden]`` head weights rowwise to e4m3."""

    if weight.ndim != 2:
        raise ValueError(
            "draft lm_head weight must be two-dimensional, "
            f"got shape={tuple(weight.shape)}"
        )
    if not weight.is_cuda:
        # CPU quantization remains useful to the upstream unit tests, so only
        # reject missing dtype support here rather than requiring CUDA.
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this torch build has no float8_e4m3fn dtype")

    with torch.no_grad():
        w = weight.detach()
        row_max = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6)
        weight_fp8 = (w.float() * (_FP8_MAX / row_max)).to(
            torch.float8_e4m3fn
        )
        row_scale = (row_max / _FP8_MAX).to(w.dtype).reshape(1, -1)
        unit_scale = torch.ones(1, dtype=torch.float32, device=w.device)
    return Fp8DraftHead(weight_fp8, row_scale, unit_scale)


def fp8_draft_head_logits(
    hidden_states: torch.Tensor,
    head: Fp8DraftHead,
) -> torch.Tensor:
    """Compute local-shard draft logits with dynamic activation FP8."""

    if hidden_states.shape[-1] != head.weight_fp8.shape[-1]:
        raise ValueError(
            "draft hidden/head dimension mismatch: "
            f"hidden={hidden_states.shape[-1]}, "
            f"head={head.weight_fp8.shape[-1]}"
        )
    # This check is shape/API based and constant after model load.  The caller
    # also runs require_fp8_draft_head_support before CUDA graph capture.
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is unavailable")

    act_max = hidden_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    act_fp8 = (hidden_states * (_FP8_MAX / act_max)).to(torch.float8_e4m3fn)
    logits = torch._scaled_mm(
        act_fp8,
        head.weight_fp8.t(),
        scale_a=head.unit_scale,
        scale_b=head.unit_scale,
        out_dtype=hidden_states.dtype,
    )
    logits = logits * head.row_scale
    logits = logits * (act_max / _FP8_MAX).to(hidden_states.dtype)
    return logits
