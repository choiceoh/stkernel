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
import re

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
            self._install_o_proj()

            import dsv41_scales

            dsv41_scales.install(self)

        # Names this derivation has no module for. The V4 model already skips
        # `mtp.`; V4.1 adds a vision tower, an aligner and the image marker
        # embeddings, none of which exist here, and each of which otherwise
        # stops the load with "no module or parameter named 'aligner'".
        #
        # `.engram.embed.` is listed for the same reason but is NOT the real
        # defence: vLLM materializes every tensor the index names before any
        # skip runs, and one engram table is 94 GiB. Removing them from the
        # INDEX is what keeps them out of host memory --
        # tools/dsv41_preshard.py does that for both the staged view and the
        # per-rank files. The entry here only makes a hand-assembled directory
        # fail with a message instead of a name error.
        # Derived from the full name-pattern diff between the two
        # checkpoints, not one boot failure at a time. V4.1 has 30 name
        # patterns V4 does not; these are the ones this derivation has no
        # destination for.
        SKIP_SUBSTRS = (
            "mtp.",                     # the DSpark block, 3 stages
            "vision.", "aligner.",      # the ViT tower and its aligner
            "image_start", "image_end", "image_newline",
            ".engram.",                 # tables AND the small projections
            "ffn.gate.bias_vl",         # the vision-token gate bias
            "attn.indexer.k_norm", "attn.indexer.wk",
        )

        def load_weights(self, weights):
            from vllm.model_executor.models.utils import AutoWeightsLoader

            loader = AutoWeightsLoader(self,
                                       skip_substrs=list(self.SKIP_SUBSTRS))
            loaded = loader.load_weights(weights,
                                         mapper=self.hf_to_vllm_mapper)
            self._report_unloaded(loaded)
            self.model.finalize_mega_moe_weights()
            self.model.setup_b12x_wo_projection()
            return loaded

        def _report_unloaded(self, loaded) -> None:
            """Name every parameter no checkpoint tensor reached.

            An unloaded parameter keeps whatever `initialize_dummy_weights` or
            `torch.empty` left in it. The model then runs at full speed and is
            wrong in a way no output inspection finds -- which is exactly what
            V4.1 sets up: it does NOT ship `hc_head_base/fn/scale`, and the V4
            model allocates them. Silence here would be a model built on three
            tensors of uninitialized memory.
            """
            import os
            import sys

            missing = sorted(name for name, _ in self.named_parameters()
                             if name not in loaded)
            if not missing:
                sys.stderr.write("[dsv41] every parameter was loaded\n")
                sys.stderr.flush()
                return
            groups = {}
            for name in missing:
                key = re.sub(r"\.\d+\.", ".N.", name)
                groups.setdefault(key, 0)
                groups[key] += 1
            lines = "\n".join(f"      {n:5d}  {k}"
                               for k, n in sorted(groups.items()))
            sys.stderr.write(
                f"[dsv41] {len(missing)} PARAMETER(S) NOT LOADED -- they hold "
                f"uninitialized memory:\n{lines}\n")
            sys.stderr.flush()
            if os.environ.get("DSV41_ALLOW_PARTIAL", "0").strip() not in (
                    "1", "true", "yes"):
                raise RuntimeError(
                    f"{len(missing)} parameter(s) had no checkpoint tensor. "
                    f"Set DSV41_ALLOW_PARTIAL=1 to run anyway; the result is "
                    f"a model computing with uninitialized memory and must "
                    f"never be measured for quality.")

        def _install_o_proj(self) -> None:
            """Swap every attention's `_o_proj` for the bf16 grouped path.

            Per instance rather than by subclassing the attention class: which
            class is used is `_select_dsv4_attn_cls(vllm_config)`'s decision,
            and there are several. Patching the instances leaves that choice
            where it belongs and still reaches every one of them.
            """
            import os

            if os.environ.get("DSV41_FUSED_O_PROJ", "0").strip() in (
                    "1", "true", "yes"):
                logger.warning(
                    "DSV41_FUSED_O_PROJ=1: keeping the image's fused fp8 "
                    "o-projection. It reads weight scales at 128 granularity "
                    "and this checkpoint ships 32, so expect either a rank "
                    "assertion or silently wrong numbers.")
                return

            import dsv41_o_proj

            patched = 0
            for module in self.modules():
                if hasattr(module, "_o_proj") and hasattr(module, "wo_a"):
                    dsv41_o_proj.install(module)
                    patched += 1
            if not patched:
                raise RuntimeError(
                    "no attention module exposed `_o_proj` and `wo_a`, so the "
                    "32-granular o-projection was never installed. Booting on "
                    "would use the image's 128-granular kernel against 32-wide "
                    "scales.")
            logger.info("DeepSeek-V4.1: bf16 grouped o-projection on %d "
                        "attention layers", patched)

    return DeepseekV41ForCausalLM


# vLLM resolves "dsv41_impl:DeepseekV41ForCausalLM" by attribute lookup, so the
# class has to exist at module scope -- built here, at import time, which is
# already lazy: this module is imported only when the architecture is used.
DeepseekV41ForCausalLM = _build()
