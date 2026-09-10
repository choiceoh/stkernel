"""The V4.1 model class. Imported only when vLLM instantiates the architecture.

Derived from this image's `vllm.models.deepseek_v4.nvidia.model`, because V4.1
is the same family: 16 of the 19 config attributes that model reads are in
V4.1's config unchanged, and most of the rest of the diff is sizes it already
reads. See dsv41_vllm.py for the attribute-by-attribute account.

This file is deliberately a THIN subclass. Everything it does is either (a)
adapting the config so the V4 model's own code is correct for V4.1, or (b)
refusing to boot where V4.1 needs a structure the V4 model does not have. What
it must never do is let one of those structures be silently skipped: a missing
engram layer does not crash, it answers worse, and nothing downstream can tell.
"""

from __future__ import annotations

import logging

from dsv41_vllm import adapt_text_config, engram_layers, layer_roles

logger = logging.getLogger(__name__)


def _base():
    """The V4 classes, imported late so a failure names the real cause."""
    try:
        from vllm.models.deepseek_v4.nvidia.model import (
            DeepseekV4ForCausalLM,
        )
    except ImportError as exc:                                # noqa: BLE001
        raise ImportError(
            "DeepSeek-V4.1 is derived from this image's DeepSeek-V4 model, "
            "and that model is not importable here. V4.1 is registered by an "
            "overlay .pth, so it can be armed on an image that has no V4 -- "
            "and then it cannot run. Boot on the ds4 image "
            "(aidendle94/sparkrun-vllm-ds4-gb10)."
        ) from exc
    return DeepseekV4ForCausalLM


class _NotYetImplemented(NotImplementedError):
    """A V4.1 structure the derivation does not carry yet."""


def _check_unsupported(text, vllm_config):
    """Refuse, loudly, where V4.1 needs something the V4 model does not have.

    Each of these degrades rather than crashes if it is skipped, which is why
    they are checked here instead of being discovered in the output.
    """
    problems = []

    tables = engram_layers(text)
    if tables:
        problems.append(
            f"engram: layers {sorted(tables)} carry conditional-memory tables "
            f"({sum(tables.values()):,} rows total, 188.8 GiB). The V4 model "
            f"has no such layer, and a decoder that skips them runs at full "
            f"speed and answers worse. dsv41_engram holds the SSD-backed "
            f"lookup; wiring it into the layer is not done.")

    stages = int(getattr(text, "num_nextn_predict_layers", 0) or 0)
    if stages > 1:
        problems.append(
            f"DSpark: {stages} stages under mtp.*, against the V4 model's 1. "
            f"Booting with one stage loads the first and leaves "
            f"{stages - 1} sets of weights unread.")

    if getattr(vllm_config.model_config, "hf_config", None) is not None:
        vision = getattr(vllm_config.model_config.hf_config, "vision_config",
                         None)
        limit = getattr(vllm_config.model_config, "limit_mm_per_prompt", None)
        wants_images = bool(limit) and any(
            v for k, v in dict(limit).items() if k == "image")
        if vision is not None and wants_images:
            problems.append(
                "vision: the checkpoint ships a ViT tower and this launch "
                "allows images, but the derivation carries no encoder. Set "
                "--limit-mm-per-prompt '{\"image\":0,\"video\":0}'.")

    if problems:
        raise _NotYetImplemented(
            "DeepSeek-V4.1 needs structures this derivation does not carry "
            "yet:\n  - " + "\n  - ".join(problems)
            + "\n\nSet DSV41_ALLOW_PARTIAL=1 to boot anyway. That is a "
              "DEGRADED model, correct only as a plumbing test -- never as a "
              "quality measurement, and never in front of traffic.")


def _build():
    base = _base()

    class DeepseekV41ForCausalLM(base):                       # type: ignore[misc,valid-type]
        """V4.1 = V4's decoder, plus the structures listed in dsv41_vllm."""

        def __init__(self, *, vllm_config, prefix: str = ""):
            import os

            hf = vllm_config.model_config.hf_config
            text = adapt_text_config(hf)
            kv, idx = layer_roles(text)
            logger.info(
                "DeepSeek-V4.1: %d layers, hidden %d, %d experts; KV sources "
                "%s, index sources %s", text.num_hidden_layers,
                text.hidden_size, text.n_routed_experts, sorted(kv),
                sorted(idx))

            if os.environ.get("DSV41_ALLOW_PARTIAL", "0").strip() not in (
                    "1", "true", "yes"):
                _check_unsupported(text, vllm_config)
            else:
                logger.warning(
                    "DSV41_ALLOW_PARTIAL=1: booting without engram, extra "
                    "DSpark stages or vision. This is a plumbing test, not a "
                    "model.")

            super().__init__(vllm_config=vllm_config, prefix=prefix)

    return DeepseekV41ForCausalLM


# vLLM resolves "dsv41_impl:DeepseekV41ForCausalLM" by attribute lookup, so the
# class has to exist at module scope -- built here, at import time, which is
# already lazy: this module is imported only when the architecture is used.
DeepseekV41ForCausalLM = _build()
