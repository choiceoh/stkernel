# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    # deneb fork: the same fp8 block copies for the drafter's own dense
    # projections. The decode trace shows 32 bf16 cutlass GEMMs/step here
    # (~0.55 GB/rank of weight reads at TP=4, ~2.5 ms) -- the same
    # read-every-forward property that armed the target in glm53_fp8_dense,
    # a fifth of the size. No-op unless VLLM_DFLASH2_FP8_DENSE=1; the
    # drafter's quality probe is ACCEPTANCE, not served output, so it rolls
    # out on its own knob.
    try:
        from vllm.model_executor.layers.glm53_fp8_dense import (
            Fp8DenseMethod,
            install_drafter_serving_check,
            maybe_build_fp8_dense,
        )

        if maybe_build_fp8_dense(dflash_model, env="VLLM_DFLASH2_FP8_DENSE"):
            # Armed is not served: the drafter's forward is torch.compiled
            # and vLLM's compile cache does not key on the swap above (the
            # 09-03 bf16 artifact was served under this knob until 27차).
            # Count the opaque op per forward and say so in the fingerprint.
            n_opaque = sum(
                1
                for m in dflash_model.modules()
                if isinstance(getattr(m, "quant_method", None), Fp8DenseMethod)
                and getattr(m.quant_method, "_opaque", False)
            )
            install_drafter_serving_check(dflash_model, n_opaque)
    except Exception:
        pass

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = dflash_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    # DFlash2 checkpoints do not carry an lm_head. get_model() still creates
    # an empty ParallelLMHead, so quantizing before the sharing block above
    # reads uninitialized storage and then discards the resulting fp8 copy.
    # Build only after the target head is attached. The helper itself logs and
    # falls back to bf16 on a quantization error; an import/integration error
    # must stay visible instead of being swallowed during boot.
    from vllm.model_executor.layers.fp8_lm_head import build_fp8_lm_head

    build_fp8_lm_head(dflash_model)

    return dflash_model
