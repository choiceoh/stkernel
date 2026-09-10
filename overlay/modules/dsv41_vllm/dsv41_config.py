"""`deepseek_v41` as a transformers config type, registered at runtime.

vLLM refuses the checkpoint before it ever looks at architectures: the config
says `model_type: deepseek_v41` and transformers has never heard of it, so
`ModelConfig` fails with "Transformers does not recognize this architecture",
which reads like a broken checkpoint rather than a missing 23-line class.

Registering with `AutoConfig` rather than vLLM's private `_CONFIG_REGISTRY`:
the latter maps model_type to a class NAME resolved through a lazy import map
inside the wheel, so extending it means matching an internal shape that the
next image can change. AutoConfig is the documented extension point, and vLLM
falls through to it.

The config is a NESTED one -- `text_config` (`deepseek_v41_text`) plus
`vision_config` -- so both types are registered. The values are not restated
here: every field comes from the checkpoint's own config.json, and this class
only exists to give transformers a type to instantiate. Restating defaults
would create a second source of truth for things like `rms_norm_eps 1e-20`,
where being quietly wrong is indistinguishable from being right.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class DeepseekV41TextConfig(PretrainedConfig):
    """The 40-layer CED backbone. Fields arrive as kwargs from config.json."""

    model_type = "deepseek_v41_text"

    # V4 hashed its first `num_hash_layers` MoE layers. V4.1 has no hash MoE --
    # engram replaced it -- and the V4 model reads the attribute bare
    # (`extract_layer_index(prefix) < config.num_hash_layers`), so its absence
    # is an AttributeError at layer 0 rather than a fallback. 0 is the value,
    # and it belongs on the config class where it cannot depend on an adapter
    # having run first.
    num_hash_layers = 0

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: "dict[str, Any] | None" = None,
        rope_parameters: "dict[str, Any] | None" = None,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        # transformers moved to `rope_parameters`; the checkpoint still writes
        # `rope_scaling`, and V4's class in this image resolves it the same
        # way. Keeping both pointed at one object means a consumer that reads
        # either gets the YaRN factor 16 the checkpoint actually ships.
        self.rope_parameters = rope_scaling or rope_parameters
        super().__init__(**kwargs)


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DeepseekV41Config(PretrainedConfig):
    """The outer config: text backbone, vision tower, and the quant scheme."""

    model_type = "deepseek_v41"
    num_hash_layers = 0                    # see DeepseekV41TextConfig
    sub_configs = {"text_config": DeepseekV41TextConfig,
                   "vision_config": DeepseekV41VisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = DeepseekV41VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(**kwargs)
        if text_config is not None:
            self._flatten(text_config)

    # The outer config is a FLAT VIEW of the text backbone.
    #
    # The image's DeepSeek-V4 model reads `vllm_config.model_config.hf_config`
    # directly -- `config.hidden_size`, `config.n_routed_experts`,
    # `config.index_topk`. V4's config is flat, so that works there. V4.1
    # nests everything under `text_config`, and an outer object that does not
    # forward answers `hidden_size` with PretrainedConfig's default (768) and
    # builds a 768-wide model without complaining once.
    #
    # Forwarding rather than swapping the config object: vLLM hands the same
    # hf_config to the tokenizer, the scheduler and the quant config, and
    # replacing it under them is a much larger blast radius than adding
    # attributes that were missing.
    OUTER_OWNED = frozenset((
        "architectures", "model_type", "quantization_config", "text_config",
        "vision_config", "dtype", "torch_dtype", "transformers_version",
        "bos_token_id", "eos_token_id", "pad_token_id", "image_token_id",
        "tie_word_embeddings",
    ))

    def _flatten(self, text_config):
        for name, value in vars(text_config).items():
            if name.startswith("_") or name in self.OUTER_OWNED:
                continue
            if getattr(self, name, None) is None:
                setattr(self, name, value)


REGISTERED = (("deepseek_v41", DeepseekV41Config),
              ("deepseek_v41_text", DeepseekV41TextConfig),
              ("deepseek_v41_vision", DeepseekV41VisionConfig))


def register():
    """Idempotent. A second call must not raise -- .pth files run per site dir."""
    from transformers import AutoConfig, CONFIG_MAPPING

    done = []
    for model_type, cls in REGISTERED:
        if model_type in CONFIG_MAPPING:
            continue
        AutoConfig.register(model_type, cls)
        done.append(model_type)
    return done
