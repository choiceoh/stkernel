# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from torch import nn

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
# deneb fork: fused small-M gate for sm_121 (overlay/modules/moe_gate_sm121).
from vllm.model_executor.layers.fp8_lm_head import (
    Fp8HeadLogitsProcessor,
    decodable_vocab_size,
)
from vllm.model_executor.layers.fused_moe.router.moe_gate_sm121 import (
    DenebGateLinear,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
    hc_contract,
    hc_expand,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from .attention import Glm5NextMLAAttention
from .glm53_prefill_fastpath import warm_glm53_prefill_metadata_runtime
from .kda import Glm5NextLinearAttention
from .multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    Glm5NextVisionTransformer,
)


logger = init_logger(__name__)

_PREFILL_SP_ENABLED = os.environ.get("VLLM_GLM53_PREFILL_SP") == "1"
if _PREFILL_SP_ENABLED:
    from vllm.distributed.device_communicators.glm53_prefill_collectives import (
        partial_tp_output,
        prefill_all_gather,
        prefill_reduce_scatter,
        prefill_shard,
    )


def _prefill_sp_metadata_ok(metadata, layer_names, num_tokens):
    """Inspect host metadata only: every target layer must be pure prefill."""
    if not isinstance(metadata, dict) or not layer_names:
        return False
    for name, kind in layer_names:
        item = metadata.get(name)
        counts = (
            getattr(item, "num_actual_tokens", None),
            getattr(item, "num_prefills", None),
            getattr(item, "num_decodes", None),
            getattr(item, "num_decode_tokens", None),
        )
        if not all(type(value) is int for value in counts):
            return False
        actual, prefills, decodes, decode_tokens = counts
        if actual != num_tokens or prefills <= 0 or decodes != 0 or decode_tokens != 0:
            return False
        if kind == "kda":
            # GDN separates ordinary decode from speculative verification.
            if (
                getattr(item, "num_spec_decodes", None) != 0
                or getattr(item, "num_spec_decode_tokens", None) != 0
                or getattr(item, "num_prefill_tokens", None) != num_tokens
            ):
                return False
        elif kind != "mla":
            return False
    return True


def _prefill_sp_layer_reduction_ok(layer):
    """Require exactly one late TP sum in each attention/MLP call.

    The request-local communicator scope counts that sum again at execution.
    These guards reject early-reduced or transformed MoE paths before the
    first SP collective; no layer attributes are changed for this request.
    """
    if (
        not layer.mhc
        or layer.is_mtp_layer
        or layer.is_sequence_parallel
        or getattr(layer.self_attn.o_proj, "reduce_results", None) is not True
    ):
        return False
    if not layer._mlp_is_moe:
        return getattr(layer.mlp.down_proj, "reduce_results", None) is True
    runner = layer.mlp.experts
    config = getattr(runner, "moe_config", None)
    if (
        config is None
        or getattr(config, "tp_size", None) != 4
        or getattr(config, "ep_size", None) != 1
        or getattr(config, "dp_size", None) != 1
        or getattr(config, "is_sequence_parallel", None) is not False
        or getattr(config, "skip_final_all_reduce", None) is not False
        or getattr(config, "moe_backend", None) not in ("flashinfer_b12x", "b12x")
        or getattr(runner, "routed_input_transform", None) is not None
        or getattr(runner, "routed_output_transform", None) is not None
        or getattr(runner, "_fused_output_is_reduced", None) is not False
    ):
        return False
    router = getattr(runner, "router", None)
    if router is None or any(
        cls.__name__ == "ZeroExpertRouter" for cls in type(router).__mro__
    ):
        return False
    shared = layer.mlp.shared_experts
    return shared is None or getattr(shared.down_proj, "reduce_results", None) is False

# deneb fork (glm53_prep_fused): fused decode input preparation. Import only
# -- a boot without the module mounted is stock -- and the installer is inert
# unless VLLM_GLM53_PREP_FUSED is set; it pins the runner preimages and any
# failure is loud in the boot log, never fatal.
try:
    from .glm53_prep_fused import install_glm53_prep_fused
except ImportError as _e:
    install_glm53_prep_fused = None
    # only "module not mounted" is silent; an ImportError raised INSIDE the
    # module (a renamed vllm symbol on an image bump) must show in the log
    if _e.name != f"{__package__}.glm53_prep_fused":
        logger.exception("[prep-fused] module import failed -> stock path")
if install_glm53_prep_fused is not None:
    try:
        install_glm53_prep_fused()
    except Exception:
        logger.exception("[prep-fused] install failed -> stock path")

# deneb fork (glm53_draft_dump, 37차): the target's per-token features for
# drafter training, from prefill steps; inert without VLLM_GLM53_DRAFT_DUMP.
try:
    from .glm53_draft_dump import install_glm53_draft_dump
except ImportError as _e:
    install_glm53_draft_dump = None
    if _e.name != f"{__package__}.glm53_draft_dump":
        logger.exception("[draft-dump] module import failed -> not installed")
if install_glm53_draft_dump is not None:
    try:
        install_glm53_draft_dump()
    except Exception:
        logger.exception("[draft-dump] install failed -> not installed")

# deneb fork (glm53_dflash_early_fc): the drafter's fc under the target head
# + sampler. Same shape of install as prep_fused: import only, inert unless
# VLLM_GLM53_DFLASH_EARLY_FC=1, loud on failure, never fatal.
try:
    from .glm53_dflash_early_fc import install_glm53_dflash_early_fc
except ImportError as _e:
    install_glm53_dflash_early_fc = None
    if _e.name != f"{__package__}.glm53_dflash_early_fc":
        logger.exception("[dflash-early-fc] module import failed -> stock path")
if install_glm53_dflash_early_fc is not None:
    try:
        install_glm53_dflash_early_fc()
    except Exception:
        logger.exception("[dflash-early-fc] install failed -> stock path")

# deneb fork (glm53_drafter_prep): skip the per-step host build of the draft
# attention metadata on FULL drafter replays (34차: 420-640 us of GPU idle
# per step behind a DtoH sync). Same install shape: import only, inert
# unless VLLM_GLM53_DRAFTER_PREP names a mode, loud on failure, never fatal.
try:
    from .glm53_drafter_prep import install_glm53_drafter_prep
except ImportError as _e:
    install_glm53_drafter_prep = None
    if _e.name != f"{__package__}.glm53_drafter_prep":
        logger.exception("[drafter-prep] module import failed -> stock path")
if install_glm53_drafter_prep is not None:
    try:
        install_glm53_drafter_prep()
    except Exception:
        logger.exception("[drafter-prep] install failed -> stock path")


def _validate_decodable_vocab_bound(
    decodable_vocab: int,
    model_vocab: int,
) -> int:
    """Reject tokenizer bounds that would address past the target head."""
    if decodable_vocab > model_vocab:
        raise ValueError(
            "decodable vocabulary cannot exceed target model vocabulary "
            f"({decodable_vocab} > {model_vocab}); check "
            "VLLM_GLM53_DECODABLE_VOCAB and tokenizer.json"
        )
    return decodable_vocab


def _mk_smlp(mlp, x):
    """MK_SEG_SMLP hook: import only, a boot without the megakernel module
    (or with the segment off) is stock. An armed launch stays loud."""
    try:
        from vllm.model_executor.layers.glm53_megakernel import smlp_forward
    except Exception:
        return None
    return smlp_forward(mlp, x)


class Glm5NextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

        self.swiglu_limit = swiglu_limit
        if self.swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit=self.swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        # deneb fork (glm53_megakernel MK_SEG_SMLP): the whole MLP as one
        # launch when the segment is armed and the row batch is a decode
        # batch; None means stock. The fused path returns this rank's
        # partial like down_proj would before its reduction, so the
        # linear's reduce_results contract is honoured here.
        out = _mk_smlp(self, x)
        if out is not None:
            if getattr(self.down_proj, "reduce_results", False):
                from vllm.distributed import tensor_model_parallel_all_reduce

                out = tensor_model_parallel_all_reduce(out)
            return out
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    def __init__(
        self,
        config: Glm5NextConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = DenebGateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        swiglu_limit = getattr(config, "swiglu_limit", None)
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=None,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # The MoE runner HOLDS this layer's gate (the factory was given it)
        # and applies it itself, after the shared-expert aux-stream sync, so
        # the gate GEMM overlaps the aux stream -- and it overwrites whatever
        # router_logits it is handed. Computing the gate here as well made
        # every MoE layer launch the router TWICE per forward (trace: gate
        # kernels exactly 2x the topk kernels, 84 vs 42/step, both eager,
        # 20us apart, identical grids). Hand it None and let the runner's
        # overlapped call be the only one.
        router_logits = None
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        self.layer_kind = "kda" if config.is_kda_layer(layer_idx) else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self._prefill_sp_metadata_name = (
            f"{prefix}.self_attn"
            if self.layer_kind == "kda"
            else f"{prefix}.self_attn.attn"
        )

        if config.is_kda_layer(layer_idx):
            self.self_attn = Glm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # MLA layers require the latent head dims, which are guaranteed set
            # on MLA configs; narrow away the `int | None`.
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=getattr(config, "mla_nope", False),
            )

        # MTP layers sit past the base model's hidden layers (layer_idx >=
        # num_hidden_layers), so they're outside mlp_layer_types; default them
        # to the last base layer's MLP type (sparse/MoE for these checkpoints).
        mlp_layer_types = config.mlp_layer_types
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
        )
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Cached for the hot forward path (isinstance per layer per step).
        self._mlp_is_moe = isinstance(self.mlp, Glm5NextMoE)
        # In SP, the attention output projection leaves a partial sum; the
        # decoder-layer reduce_scatter after attention completes it (DSv4 pattern).
        # MTP layers use the non-mHC path which has no sp_reduce_scatter, so
        # their o_proj must still reduce normally.
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_no_norm_weight = config.mhc_no_norm_weight
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
        prefill_sequence_parallel: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # 70B or MTP layers: KDA + MoE without HC.
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(
                attn_output, residual=residual
            )
            hidden_states = self.mlp(hidden_states)
            if self.is_mtp_layer:
                # Return the unsummed pair: the MTP caller feeds it straight
                # into shared_head's fused_add_rms_norm (one kernel instead of
                # a separate residual-add + norm). The sum itself is unchanged
                # (fp32-accumulated inside the fused kernel).
                return hidden_states, residual, None, None
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        # mHC start. `post`/`comb` carry the previous layer's deferred
        # hc_post inputs (its ffn-pre outputs); when present, fuse that
        # hc_post with this layer's attn hc_pre into one kernel (inter-layer
        # fusion). Layer 0 has no incoming state -> standalone hc_pre.
        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                x = hc_expand(x, self.n)
            residual = x
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )

        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        if prefill_sequence_parallel:
            # MHC owns only this rank's contiguous token shard. Attention
            # continues to see the original complete metadata/positions.
            x = prefill_all_gather(x, num_tokens=positions.shape[0])
            with partial_tp_output(num_tokens=positions.shape[0]):
                x = self.self_attn(hidden_states=x, positions=positions)
            x = prefill_reduce_scatter(x)
        elif self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]
            x = self.self_attn(hidden_states=x, positions=positions)
            x = sp_reduce_scatter(x)
        else:
            x = self.self_attn(hidden_states=x, positions=positions)

        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        # Fully Connected
        if prefill_sequence_parallel:
            # Keep the existing TP expert/dense weights and full-token MoE
            # routing. Suppress only their terminal sum, then scatter it.
            x = prefill_all_gather(x, num_tokens=positions.shape[0])
            with partial_tp_output(num_tokens=positions.shape[0]):
                x = self.mlp(x)
            x = prefill_reduce_scatter(x)
        elif self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)

        # mHC end. The last mHC layer materializes its final hc_post (nothing
        # to fuse with) then contracts; every other layer defers its hc_post to
        # the next layer's fused pre, returning the state.
        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None

        return x, residual, post, comb

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre_op(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return post_mix, res_mix, layer_input

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post_op(x, residual, post, comb)

    def hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        return self.mhc_fused_post_pre_op(
            x=x,
            residual=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            n_splits=1,
            tile_n=1,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )


class Glm5NextModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        """
        if config.index_topk is not None:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
        """
        # `index_topk` is declared on Glm5NextTextConfig with a default of None,
        # so hasattr() is True even for full-MLA configs (no kpool indexer).
        # Gate on the value being set instead.
        self.is_v32 = getattr(config, "index_topk", None) is not None
        if self.is_v32:
            topk_tokens = config.index_topk
            # kpool widens the topk buffer: selecting topk_tokens//kpool pools and
            # expanding them yields topk_tokens token indices, plus an always-
            # selected tail of up to kpool-1 incomplete-pool tokens. The attention
            # backend reads the width dynamically via topk_indices.shape[1].
            kpool = getattr(config, "index_kpool", 1) or 1
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # The sparse MLA attention kernel
            # (triton_convert_req_index_to_global_index) tiles the topk
            # dimension in BLOCK_N=128 columns and requires the buffer width
            # to be a multiple of it; otherwise it raises
            # "NUM_TOPK_TOKENS must be divisible by BLOCK_N". Round up: the
            # extra slots stay -1 (the indexer op initializes the buffer to
            # -1) and are masked out by the attention kernel, so they do not
            # affect the softmax over the selected tokens.
            sparse_topk_block_n = 128
            buffer_width = (
                (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
            ) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Full-MLA config (no kpool sparse indexer): no topk buffer.
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # The active slice is fixed after construction; cache it so forward
        # doesn't rebuild the slice (a fresh list) every step.
        self._active_layers = self.layers[self.start_layer : self.end_layer]

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        parallel = vllm_config.parallel_config
        self._prefill_sp_config_ok = (
            _PREFILL_SP_ENABLED
            and config.mhc
            and config.hidden_size == 4096
            and config.mhc_num_residual_streams == 4
            and not self.is_sequence_parallel
            and parallel.tensor_parallel_size == 4
            and parallel.pipeline_parallel_size == 1
            and getattr(parallel, "data_parallel_size", 1) == 1
            and not getattr(parallel, "enable_expert_parallel", False)
            and not getattr(parallel, "enable_eplb", False)
            and getattr(parallel, "decode_context_parallel_size", 1) == 1
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
            and self.start_layer == 0
            and self.end_layer == config.num_hidden_layers
        )
        self._prefill_sp_metadata_layers = tuple(
            (layer._prefill_sp_metadata_name, layer.layer_kind)
            for layer in self._active_layers
        )
        self._prefill_sp_announced = False
        self._prefill_sp_declined = False

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _prefill_sp_decline(self, reason: str) -> bool:
        # 39차: an armed knob whose gate declines every forward reads as
        # "no effect" in a bracket; say WHY, once, so P1's silence (0 %
        # at 32K with eight knobs armed) cannot repeat unexplained.
        if _PREFILL_SP_ENABLED and not self._prefill_sp_declined:
            logger.warning("[prefill-sp] NOT selected: %s", reason)
            self._prefill_sp_declined = True
        return False

    def _can_prefill_sequence_parallel(self, hidden_states, positions) -> bool:
        if not self._prefill_sp_config_ok:
            return self._prefill_sp_decline(
                "config gate (needs mhc, hidden 4096, 4 streams, TP4/PP1/DP1, no EP/EPLB/DCP/PCP, "
                "no sequence-parallel MoE, all layers on this rank)")
        if (
            torch.compiler.is_compiling()
            or hidden_states.device.type != "cuda"
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 2
            or hidden_states.shape[1] != 4096
            or not hidden_states.is_contiguous()
            or positions.ndim != 1
            or positions.device != hidden_states.device
            or positions.shape[0] != hidden_states.shape[0]
            or torch.cuda.is_current_stream_capturing()
            or not is_forward_context_available()
        ):
            return self._prefill_sp_decline(
                f"tensor/context gate (compiling={torch.compiler.is_compiling()} "
                f"capturing={torch.cuda.is_current_stream_capturing()} T={positions.shape[0]})")
        if positions.shape[0] < 128:
            return False   # decode-sized forward: silently stock, not a decline worth a line
        if not _prefill_sp_metadata_ok(
            get_forward_context().attn_metadata,
            self._prefill_sp_metadata_layers,
            positions.shape[0],
        ):
            return self._prefill_sp_decline(
                f"metadata gate (T={positions.shape[0]}: not a pure-prefill batch on every layer, "
                "or metadata names/kinds unknown)")
        # Quant kernels and wrappers are installed after construction, so
        # their reduction contract is checked on the actual serving objects.
        for layer in self._active_layers:
            if not _prefill_sp_layer_reduction_ok(layer):
                o = getattr(getattr(layer.self_attn, "o_proj", None), "reduce_results", None)
                return self._prefill_sp_decline(
                    f"layer reduction contract (layer {layer.layer_idx}: o_proj.reduce_results={o}, "
                    f"moe={layer._mlp_is_moe}, mhc={layer.mhc}, seq_parallel={layer.is_sequence_parallel})")
        if not self._prefill_sp_announced:
            logger.info(
                "[prefill-sp] MHC token shards selected (T=%d, TP=4, full-token attention/MoE)",
                positions.shape[0],
            )
            self._prefill_sp_announced = True
        return True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # post/comb (deferred mHC hc_post state) are not propagated across
            # PP ranks; the receiving rank's first mHC layer uses standalone pre.
            post = None
            comb = None

        full_num_tokens = positions.shape[0]
        prefill_sequence_parallel = self._can_prefill_sequence_parallel(hidden_states, positions)
        if prefill_sequence_parallel:
            hidden_states = prefill_shard(hidden_states)
        elif self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        # DFLASH2-AUX-CAPTURE (EAGLE-3 aux hidden states; mirrors
        # DeepseekV4Model.forward in vllm/models/deepseek_v4/nvidia/model.py).
        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(self._active_layers, start=self.start_layer):
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb,
                prefill_sequence_parallel=prefill_sequence_parallel,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # `idx + 1` matches deepseek_v4: the runner already converted
                # DFlash target_layer_ids to id+1 semantics
                # (gpu_model_runner._get_eagle3_aux_layers_from_config), so
                # this captures the OUTPUT of 0-based decoder layer `idx`.
                if post is not None:
                    # Mid-stack mHC layer: its final hc_post is deferred to
                    # the next layer's fused pre. Materialize the multi-stream
                    # reconstruction here (pure op -- the deferred
                    # residual/post/comb state is not mutated), then contract
                    # hc streams exactly like the last layer does.
                    # hc_contract == mean over streams, the same contraction
                    # deepseek_v4 uses (aux_recon.mean(dim=1)).
                    aux_recon = layer.hc_post(hidden_states, residual, post, comb)
                    aux_hidden_state = hc_contract(aux_recon, layer.n)
                else:
                    # Last mHC layer (already hc_post + hc_contract'ed inside
                    # the layer) or a non-mHC layer: the output is already
                    # plain [num_tokens, hidden_size].
                    aux_hidden_state = hidden_states
                if prefill_sequence_parallel:
                    aux_hidden_state = prefill_all_gather(
                        aux_hidden_state, num_tokens=full_num_tokens
                    )
                elif self.is_sequence_parallel:
                    # Aux states are consumed at full-sequence granularity;
                    # gather the SP shard (deepseek_v4 pattern).
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[
                        :full_num_tokens
                    ]
                aux_hidden_states.append(aux_hidden_state)

        if not get_pp_group().is_last_rank:
            # PP is gated off for GLM5Next (no make_empty_intermediate_tensors),
            # so this branch is not exercised. post/comb are the deferred
            # hc_post state of this rank's last mHC layer; a future PP path
            # would need to propagate them, but for now they are dropped (the
            # receiving rank's first layer would fall back to standalone pre).
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if prefill_sequence_parallel:
            hidden_states = prefill_all_gather(hidden_states, num_tokens=full_num_tokens)
        elif self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            # (final_hidden_states, list-of-aux) -- gpu_model_runner unpacks
            # this tuple when use_aux_hidden_state_outputs is set; identical
            # to DeepseekV4Model.forward's aux return.
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            # Indexer: fuse wk and weights_proj
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
            # KDA: merge q, k, v, b, f_a, g_a projections into one GEMM
            (".in_proj_qkvbfg_a", ".q_proj", 0),
            (".in_proj_qkvbfg_a", ".k_proj", 1),
            (".in_proj_qkvbfg_a", ".v_proj", 2),
            (".in_proj_qkvbfg_a", ".b_proj", 3),
            (".in_proj_qkvbfg_a", ".f_a_proj", 4),
            (".in_proj_qkvbfg_a", ".g_a_proj", 5),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # GLM5-Next NoPE: checkpoint's kv_a_proj_with_mqa has only kv_lora_rank
        # rows, but the model expects kv_lora_rank + qk_rope_head_dim rows.
        # Pad the missing rope portion with zeros.
        kv_a_pad_size = 0
        if self.config.mla_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim

        _pending_wk_fp8: dict = {}

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs: dict = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
            # weights_proj into wk_weights_proj.
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # FP8 checkpoint: dequantize BF16-kept MLA projections
            # (q_a_proj / kv_a_proj_with_mqa / o_proj) to BF16.
            if _try_load_fp8_attn_proj(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            ):
                continue

            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in enumerate(expert_params_mapping):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        return loaded_params


class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, SupportsEagle3, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        # deneb fork: fp8 target head (VLLM_TARGET_LM_HEAD_FP8, default
        # off). 154,880 x 4096 bf16 is 0.32 GB per rank read every step.
        self.logits_processor = Fp8HeadLogitsProcessor(
            self.config.vocab_size,
            scale=logit_scale,
            fp8_env="VLLM_TARGET_LM_HEAD_FP8",
            # W4 head on the megakernel's v2 lane (30차 §13): the served
            # logits, so its own knob, off until the operator arms it
            mk_env="VLLM_GLM53_MK_HEAD_TARGET",
        )
        # LM head rows the tokenizer cannot decode are live argmax candidates
        # (see _decodable_vocab_size). Mask them in compute_logits.
        self._decodable_vocab = _validate_decodable_vocab_bound(
            decodable_vocab_size(vllm_config.model_config.tokenizer),
            self.config.vocab_size,
        )
        self._orphan_hits: torch.Tensor | None = None
        self._orphan_calls = 0
        if self._decodable_vocab is None:
            logger.warning(
                "[vocab-mask] OFF -- decodable vocab unknown; %d LM head rows "
                "stay reachable",
                self.config.vocab_size,
            )
        elif self._decodable_vocab >= self.config.vocab_size:
            logger.info(
                "[vocab-mask] not needed: tokenizer covers all %d rows",
                self.config.vocab_size,
            )
        else:
            logger.info(
                "[vocab-mask] masking ids %d..%d (%d rows the tokenizer has no "
                "token for) out of %d",
                self._decodable_vocab,
                self.config.vocab_size - 1,
                self.config.vocab_size - self._decodable_vocab,
                self.config.vocab_size,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if (
            logits is not None
            and self._decodable_vocab is not None
            and logits.shape[-1] > self._decodable_vocab
        ):
            n = self._decodable_vocab
            # Count first: how often one of these rows was about to win. Two
            # reductions over the vocab, accumulated on device -- the read is
            # the only sync and it happens once every 256 calls. compute_logits
            # runs outside the captured graph (the graph wraps the model
            # forward), so a sync here is safe.
            if os.environ.get("VLLM_GLM53_VOCAB_MASK_AUDIT") == "1":
                if self._orphan_hits is None:
                    self._orphan_hits = torch.zeros(
                        (), dtype=torch.long, device=logits.device
                    )
                self._orphan_hits += (
                    logits[..., n:].amax(dim=-1) > logits[..., :n].amax(dim=-1)
                ).sum()
                self._orphan_calls += 1
                if self._orphan_calls % 256 == 0:
                    logger.info(
                        "[vocab-mask] orphan row would have won %d times in "
                        "%d calls",
                        int(self._orphan_hits.item()),
                        self._orphan_calls,
                    )
            logits[..., n:] = float("-inf")
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        loaded = loader.load_weights(weights)
        # deneb fork: the post-load work (fp8 copies, prefill fastpath, kpool)
        # runs from whoever owns the WHOLE checkpoint walk. Served under the
        # multimodal wrapper, that is Glm5NextForConditionalGeneration, which
        # sets _defer_post_load and calls run_post_load() once its own loader
        # returns; this method is then entered once per contiguous run of
        # `language_model.*` names in the checkpoint stream (AutoWeightsLoader
        # groups a streaming iterator with itertools.groupby), NOT once at the
        # end -- see run_post_load for what running the hooks here cost.
        if not getattr(self, "_defer_post_load", False):
            self.run_post_load()
        return loaded

    def run_post_load(self) -> None:
        """Everything that must see the fully loaded checkpoint, exactly once.

        This used to run inline in load_weights, which the comment there
        believed was "after the checkpoint is fully walked". It was not: the
        wrapper's AutoWeightsLoader calls this module's load_weights once per
        contiguous run of its prefix in the stream, so the fp8-dense pass ran
        ~25 s into the load on unloaded weights, and again after every run.
        Each early pass quantised 180 linears, built 180 MK W4 packs and ran
        360 verification GEMMs on garbage, and its transients left the
        caching allocator holding tens of GiB on every node -- the memory
        cliff behind the 09-03 srv3 OOM and the 09-04 wedges (MEASUREMENTS,
        instrumented boot 2026-09-04). The design that tolerated the early
        call ("a later call rebuilds the copy") was correct about numerics
        and blind to memory.
        """
        # fp8 block copies of the bf16 dense projections. No-op unless
        # VLLM_GLM53_FP8_DENSE=1; failures disarm per-layer.
        try:
            from vllm.model_executor.layers.glm53_fp8_dense import (
                maybe_build_fp8_dense,
            )

            maybe_build_fp8_dense(self)
        except Exception:
            pass

        # Compile the exact pooled-prefill metadata launch signatures before
        # the first request.
        try:
            warm_glm53_prefill_metadata_runtime(self)
        except Exception:
            pass


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3
):
    # The text model (KDA + dense-MLA + MoE) is a hybrid mamba model. The
    # multimodal wrapper must declare the same interfaces so vLLM treats it as
    # hybrid (auto-aligns mamba/attention block sizes, sizes the mamba state
    # cache); the mamba-state classmethods delegate to the text model.
    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    # NOTE: weight-prefix mapping is inherited from Glm4vForConditionalGeneration
    # (``model.visual.`` -> ``visual.``, ``model.language_model.`` ->
    # ``language_model.model.``, ``lm_head.`` -> ``language_model.lm_head.``),
    # matching the GLM-OCR / GLM-4V serialization convention. If the real
    # checkpoint's safetensors keys differ (e.g. ``language_model.model.`` with
    # no outer ``model.``), override ``hf_to_vllm_mapper`` accordingly.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )
        # deneb fork: this wrapper owns the checkpoint walk, so the language
        # model's post-load hooks run from load_weights below, once -- not
        # from its own load_weights, which the loader enters per prefix run.
        self.language_model._defer_post_load = True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        run = getattr(self.language_model, "run_post_load", None)
        if callable(run):
            run()
        return loaded

        # Glm5NextForCausalLM does not implement make_empty_intermediate_tensors,
        # so pipeline parallelism is gated off (consistent with the text-only
        # model) and we intentionally do not alias it here.

    def get_encoder_cudagraph_config(self):
        # The forked vision tower (multimodal.py) has no abs-pos embeddings, so its
        # prepare_encoder_metadata does not produce "pos_embeds". Drop it from the
        # buffer_keys inherited from Glm4vForConditionalGeneration so encoder
        # CUDA-graph capture/replay does not expect a buffer that is never filled.
        config = super().get_encoder_cudagraph_config()
        config.buffer_keys = [k for k in config.buffer_keys if k != "pos_embeds"]
        return config

    def forward(self, *args, **kwargs):
        """Release the dead bf16 sources on the first forward, then delegate.

        The release cannot live in maybe_build_fp8_dense: AutoWeightsLoader
        calls that before the checkpoint is walked, and a freed source breaks
        the loader's own shape read (parameter.py:221, IndexError on a 1-D
        empty tensor). It needs a point that provably runs after loading, and
        a forward is proof the weights landed.

        It has to be THIS forward. Glm5NextForCausalLM.forward looks like the
        obvious place and is never called -- Glm4vForConditionalGeneration.
        forward reaches past it, straight into `self.language_model.model(...)`
        -- so a trigger there is dead code, which cost two boots that came up
        looking healthy with the release silently skipped. This class sits
        above the compiled region, so the one-time free also stays out of any
        traced graph.

        Runs inside the profile forward, before KV sizing, so the freed bytes
        become KV. No-op unless VLLM_GLM53_FP8_DENSE_FREE_BF16=1.
        """
        if not getattr(self, "_bf16_released", False):
            self._bf16_released = True
            try:
                from vllm.model_executor.layers.glm53_fp8_dense import (
                    maybe_free_fp8_dense_bf16,
                )

                maybe_free_fp8_dense_bf16(self)
            except Exception:
                logger.exception("[fp8-dense] bf16 release skipped")
        # AR prefetch hints (tp_oneshot_ar, VLLM_GLM53_AR_PREFETCH): the shim
        # keys its learned hints by "which collective of the target forward",
        # so the forward boundary has to come from here -- the same reason
        # the release above lives here: this class is above the compiled
        # region, and the drafter is a different class, so its collectives
        # never see the target's table. No-op unless the knob is set.
        osar = _osar_shim()
        if osar is not None:
            osar.begin_forward("target")
        try:
            return super().forward(*args, **kwargs)
        finally:
            if osar is not None:
                osar.end_forward()


_OSAR = None


def _osar_shim():
    """The one-shot AR shim, resolved once; None when it is not mounted or
    predates the prefetch hints."""
    global _OSAR
    if _OSAR is None:
        try:
            from vllm.distributed.device_communicators import (
                dsv4_oneshot_shim as shim,
            )

            _OSAR = shim if hasattr(shim, "begin_forward") else False
        except Exception:
            _OSAR = False
    return _OSAR or None


def get_spec_layer_idx_from_weight_name(
    config: Glm5NextConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(
                f"model.layers.{layer_idx + i}."
            ) or weight_name.startswith(f"layers.{layer_idx + i}."):
                return layer_idx + i
    return None


def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
    if "indexer.wk." not in name or "wk_weights" in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False
    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=GroupShape(block_size, block_size),
        out_dtype=torch.bfloat16,
    )

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


def _dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a block-FP8 (e4m3) weight with per-block scale to BF16.

    Unlike ``scaled_dequantize`` this tolerates a non-divisible (partial last
    block) shape by zero-padding to a multiple of ``block_size`` before the
    scale broadcast and trimming back afterwards (e.g. kv_a_proj_with_mqa is
    576 rows = 4*128 + 64).
    """
    out_dim, in_dim = weight_fp8.shape
    pad_out = (-out_dim) % block_size
    pad_in = (-in_dim) % block_size
    w = weight_fp8
    if pad_out or pad_in:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    # scale_inv is (ceil(out/block), ceil(in/block)); broadcast to (out, in).
    s = scale_inv.to(torch.float32)
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    out = (w.to(torch.float32) * s_full).to(torch.bfloat16)
    return out[:out_dim, :in_dim].contiguous()


# FP8 checkpoint projections that the MODEL keeps in BF16, so the block-FP8
# (weight + weight_scale_inv) must be dequantized to BF16 on load.
# Maps checkpoint proj-suffix -> (buffer key, model target base, fused shard id
# or None for a direct projection, whether NoPE rope-padding applies).
_FP8_ATTN_PROJS = {
    ".q_a_proj.": ("q_a", "fused_qkv_a_proj", 0, False),
    ".kv_a_proj_with_mqa.": ("kv_a", "fused_qkv_a_proj", 1, True),
    ".q_b_proj.": ("q_b", "q_b_proj", None, False),
    ".o_proj.": ("o_proj", "o_proj", None, False),
}


def _try_load_fp8_attn_proj(
    name,
    tensor,
    buf,
    params_dict,
    loaded_params,
    kv_a_pad_size: int,
) -> bool:
    """Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on load.

    The FP8 checkpoint stores these as block-FP8 (weight + weight_scale_inv),
    but the model holds them in BF16 (``fused_qkv_a_proj`` is always BF16 via
    DeepSeekV2FusedQkvAProjLinear; ``o_proj`` is excluded by
    modules_to_not_convert). When the model target is BF16 (no
    ``weight_scale_inv`` param) we dequantize; otherwise we return False so the
    normal stacked/direct path loads the FP8 tensor as-is.
    """
    matched = None
    for suffix, info in _FP8_ATTN_PROJS.items():
        if suffix in name:
            matched = (suffix, info)
            break
    if matched is None:
        return False
    suffix, (key, target_base, shard_id, is_kva) = matched
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False

    layer_prefix = name.rsplit(suffix, 1)[0]
    target_w = f"{layer_prefix}.{target_base}.weight"
    target_s = f"{layer_prefix}.{target_base}.weight_scale_inv"
    # If the model actually kept this projection in FP8, let the normal path
    # handle it (it has a weight_scale_inv param).
    if target_s in params_dict:
        return False

    entry = buf.setdefault(layer_prefix, {}).setdefault(key, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    buf[layer_prefix].pop(key, None)
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    # NoPE: pad kv_a rope portion (kv_lora_rank -> kv_lora_rank + qk_rope_head_dim).
    if is_kva and kv_a_pad_size > 0:
        pad = torch.zeros(
            kv_a_pad_size,
            weight_bf16.shape[1],
            dtype=weight_bf16.dtype,
            device=weight_bf16.device,
        )
        weight_bf16 = torch.cat([weight_bf16, pad], dim=0)

    param = params_dict[target_w]
    if shard_id is None:
        param.weight_loader(param, weight_bf16)
    else:
        param.weight_loader(param, weight_bf16, shard_id)
    loaded_params.add(target_w)
    return True
