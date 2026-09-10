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
        # vLLM reads sizes off the top-level config for a text model. The
        # checkpoint puts every one of them under text_config, so without
        # this the outer object answers `hidden_size` with a transformers
        # default and the model is built at the wrong width.
        if text_config is not None:
            for name in ("hidden_size", "num_hidden_layers", "vocab_size",
                         "num_attention_heads", "max_position_embeddings",
                         "rope_scaling", "rope_theta", "rms_norm_eps",
                         "num_key_value_heads", "head_dim",
                         "num_nextn_predict_layers", "sliding_window"):
                value = getattr(text_config, name, None)
                if value is not None and getattr(self, name, None) is None:
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
