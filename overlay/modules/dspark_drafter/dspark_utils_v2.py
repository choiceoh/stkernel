# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

logger = init_logger(__name__)


def _validate_markov_sideload(
    payload_keys,
    w1_shape,
    w2_shape,
    vocab_size: int,
    markov_rank: int,
    replicated_w1: bool,
    replicated_w2: bool,
) -> str | None:
    """Pure sideload contract check — returns an error string or None.

    Both Markov tensors must be full [vocab, rank] replicas: the sideload
    overwrites weights in place, so sharded layouts (VocabParallelEmbedding /
    ParallelLMHead) would silently mis-slice.
    """
    missing = {"markov_w1", "markov_w2"} - set(payload_keys)
    if missing:
        return f"sideload payload missing keys: {sorted(missing)}"
    expected = (vocab_size, markov_rank)
    if tuple(w1_shape) != expected:
        return f"markov_w1 shape {tuple(w1_shape)} != expected {expected}"
    if tuple(w2_shape) != expected:
        return f"markov_w2 shape {tuple(w2_shape)} != expected {expected}"
    if not replicated_w1:
        return (
            "sideload requires a replicated Markov W1 "
            "(VLLM_DSPARK_REPLICATE_MARKOV_W1=1)"
        )
    if not replicated_w2:
        return (
            "sideload requires a replicated Markov W2 "
            "(VLLM_DSPARK_REPLICATE_MARKOV_W2=1)"
        )
    return None


def _sideload_markov_head(draft_model: nn.Module, path: str) -> None:
    """Replace the checkpoint Markov W1/W2 with tensors from ``path``.

    Runs after checkpoint load and before warmup, i.e. before the lazy FP8
    quantization of W2 and before CUDA graph capture, so every consumer sees
    the sideloaded weights. Produced by tools/markov_refit.py; an explicitly
    requested sideload that cannot be applied fails closed.
    """
    import torch

    markov_head = getattr(getattr(draft_model, "model", None), "markov_head", None)
    if markov_head is None:
        raise RuntimeError(
            "VLLM_DSPARK_MARKOV_SIDELOAD set but the draft model has no "
            "markov_head"
        )
    if hasattr(markov_head, "_w2_fp8_weight"):
        raise RuntimeError(
            "Markov sideload must run before the lazy FP8 W2 quantization"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Markov sideload {path} must be a dict payload, got {type(payload)}"
        )
    w1 = payload.get("markov_w1")
    w2 = payload.get("markov_w2")
    dst1 = markov_head.markov_w1.weight
    dst2 = markov_head.markov_w2.weight
    config = draft_model.config
    err = _validate_markov_sideload(
        payload.keys(),
        tuple(w1.shape) if w1 is not None else (),
        tuple(w2.shape) if w2 is not None else (),
        int(config.vocab_size),
        int(config.dspark_markov_rank),
        bool(getattr(markov_head, "_replicate_w1", False)),
        bool(getattr(markov_head, "_replicate_w2", False)),
    )
    if err is None and tuple(dst1.shape) != tuple(w1.shape):
        err = f"model W1 shape {tuple(dst1.shape)} != payload {tuple(w1.shape)}"
    if err is None and tuple(dst2.shape) != tuple(w2.shape):
        err = f"model W2 shape {tuple(dst2.shape)} != payload {tuple(w2.shape)}"
    if err is None and not (
        bool(torch.isfinite(w1).all()) and bool(torch.isfinite(w2).all())
    ):
        err = "sideload tensors contain non-finite values"
    if err is not None:
        raise RuntimeError(f"invalid VLLM_DSPARK_MARKOV_SIDELOAD ({path}): {err}")
    with torch.no_grad():
        dst1.copy_(w1.to(device=dst1.device, dtype=dst1.dtype))
        dst2.copy_(w2.to(device=dst2.device, dtype=dst2.dtype))
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    logger.info_once(
        "DSpark(v2) Markov head sideloaded from %s (sha256=%s).", path, digest
    )


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag

    # DSpark uses non-causal attention.
    causal = False
    # Inherit the target's attention backend unless one was explicitly set for
    # the draft. Upstream unconditionally overrode with
    # speculative_config.attention_backend (None here), which resolved to the
    # platform default (flashinfer_sparse) while the target runs b12x — the
    # mismatched draft layers then hit fp32 rope-cache asserts and would use a
    # different KV layout than the sparse-SWA metadata expects.
    draft_backend = (
        speculative_config.attention_backend
        if speculative_config.attention_backend is not None
        else vllm_config.attention_config.backend
    )
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=not causal,
            backend=draft_backend,
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    # The draft's rope instances are built fresh under the draft's dtype
    # context, but every fused DSV4 op (deep_gemm_fp8_o_proj, the fused
    # rope/quant KV inserts) requires an fp32 cos_sin_cache — which is what
    # the target's layers carry. Normalize the draft to match.
    import torch

    for module in draft_model.modules():
        cache = getattr(module, "cos_sin_cache", None)
        if cache is not None and cache.dtype != torch.float32:
            module.cos_sin_cache = cache.float()

    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")

    # Opt-in Markov head sideload (domain-refit W1/W2 from tools/markov_refit.py).
    # Must precede warmup: the FP8 W2 copy quantizes lazily on first use and
    # CUDA graphs capture whatever weights exist then.
    sideload_path = os.getenv("VLLM_DSPARK_MARKOV_SIDELOAD", "").strip()
    if sideload_path:
        _sideload_markov_head(draft_model, sideload_path)

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if target_embed is not None and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    # Build exactly one draft-only FP8 copy after aliasing and before CUDA
    # graph capture.  The rowwise experiment is default-off; when enabled it
    # replaces (rather than stacks with) the image's legacy DeepGEMM copy.
    fp8_draft_head_raw = os.getenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "0")
    fp8_draft_head_value = fp8_draft_head_raw.strip().lower()
    if fp8_draft_head_value not in {
        "",
        "0",
        "1",
        "false",
        "true",
        "no",
        "yes",
        "off",
        "on",
    }:
        raise ValueError(
            "VLLM_DSPARK_FP8_DRAFT_HEAD must be a boolean value, "
            f"got {fp8_draft_head_raw!r}"
        )
    fp8_draft_head_enabled = fp8_draft_head_value in {
        "1",
        "true",
        "yes",
        "on",
    }

    if fp8_draft_head_enabled:
        maybe_init_fp8_draft_head = getattr(
            draft_model, "maybe_init_fp8_draft_head", None
        )
        if maybe_init_fp8_draft_head is None:
            raise RuntimeError(
                "VLLM_DSPARK_FP8_DRAFT_HEAD=1 but the selected DSpark model "
                "does not implement maybe_init_fp8_draft_head"
            )
        maybe_init_fp8_draft_head()
    else:
        # Preserve the exact image baseline when the new experiment is off.
        maybe_build_fp8_lm_head = getattr(
            draft_model, "maybe_build_fp8_lm_head", None
        )
        if maybe_build_fp8_lm_head is not None:
            maybe_build_fp8_lm_head()

    # Optional load-time release of the bf16 lm_head original (KV headroom).
    # Must run AFTER the draft fp8 copies above so the fail-closed consumer
    # check in the target model sees the final state. An armed knob with a
    # rolled-back nvidia_model.py must abort, not silently no-op.
    if os.getenv("VLLM_DSV4_FREE_BF16_LM_HEAD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        release_bf16_lm_head = getattr(
            target_model, "maybe_release_bf16_lm_head", None
        )
        if release_bf16_lm_head is None:
            raise RuntimeError(
                "VLLM_DSV4_FREE_BF16_LM_HEAD=1 but the target model does not "
                "implement maybe_release_bf16_lm_head (nvidia_model.py "
                "overlay rolled back?)"
            )
        release_bf16_lm_head(draft_model)

    return draft_model
