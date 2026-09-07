# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM5-Next KDA (linear-attention) layer.

Model-specific, self-contained KDA: separate q/k/v short-conv + the GLM5-Next
spec-decode verify path + the bounded ``safe_gate`` variant, and ``_forward`` is
an eager break point under Breakable CUDA Graph
(``@eager_break_during_capture``).

Moved out of the shared ``kimi_gdn_linear_attn.py`` (which reverts to Kimi
Linear's fused-conv version): the separate-conv layout + spec-verify are
GLM5-Next-only. ``forward`` calls ``self._forward`` directly (no
``torch.ops.vllm.kda_attention`` indirection) so the only un-capturable work is
the decorated ``_forward``.
"""

import os

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import (
    maybe_disable_graph_partition,
    set_weight_attrs,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import (
    FusedRMSNormGated,
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

logger = init_logger(__name__)


_KDA_PREFILL_DIRECT_OUT = (
    os.environ.get("VLLM_GLM53_KDA_PREFILL_DIRECT_OUT") == "1"
)
if _KDA_PREFILL_DIRECT_OUT:
    logger.warning("[kda-prefill] direct output armed (pure prefill only)")


def _kda_prefill_output_buffer(
    core_attn_out, num_actual_tokens, local_num_heads, head_dim,
    *, use_spec, num_prefills, num_decodes,
):
    """Select only a dense pure-prefill destination; retain the merge otherwise."""
    if not _KDA_PREFILL_DIRECT_OUT:
        return None
    if (
        use_spec
        or num_prefills <= 0
        or num_decodes != 0
        or num_actual_tokens <= 0
        or core_attn_out.device.type != "cuda"
        or core_attn_out.dtype != torch.bfloat16
        or core_attn_out.ndim != 4
        or core_attn_out.shape[0] != 1
        or core_attn_out.shape[1] < num_actual_tokens
        or core_attn_out.shape[2] != local_num_heads
        or core_attn_out.shape[3] != head_dim
        or not core_attn_out.is_contiguous()
    ):
        # 39차: say once why an armed lane is not taken (P1: use_spec is
        # True on every batch of a speculative-decoding runner). Self-
        # contained (function attribute + globals().get) because the logic
        # suite execs this def alone in a stub namespace.
        _fn = _kda_prefill_output_buffer
        if num_prefills > 0 and num_decodes == 0 and not getattr(_fn, "_announced", False):
            _fn._announced = True
            _lg = globals().get("logger")
            if _lg is not None:
                _lg.warning("[kda-prefill] direct output NOT taken on a pure-prefill batch: use_spec=%s "
                            "tokens=%d shape=%s", use_spec, num_actual_tokens, tuple(core_attn_out.shape))
        return None
    _fn = _kda_prefill_output_buffer
    if not getattr(_fn, "_announced", False):
        _fn._announced = True
        _lg = globals().get("logger")
        if _lg is not None:
            _lg.warning("[kda-prefill] direct output engaged (tokens=%d)", num_actual_tokens)
    # A prefix of the only batch remains dense, even with capture padding.
    return core_attn_out[:, :num_actual_tokens]


class _Glm5NextMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with multiple replicated output shards.

    Extends K3's ``_KimiGDNMergedColumnParallelLinear`` to support two
    replicated shards (f_a, g_a) instead of one. Pre-multiplies each
    replicated entry's output_size by tp_size so the per-rank shard
    divides back to the full size, and forces tp_rank=0 during weight
    loading for replicated shards.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_ids: tuple[int, ...],
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = set(replicated_shard_ids)
        output_sizes = output_sizes.copy()
        for sid in self.replicated_shard_ids:
            output_sizes[sid] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def _cast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Fuse the fp32 cast + sigmoid into one Inductor kernel."""
    return x.float().sigmoid()


# deneb fork (glm53_kda_onepass, launch-count bundle 2): two opt-in Triton
# paths for the stock KDA chain. Everything -- knobs, guards, the boot
# self-test, announce lines, arrival counters -- lives in that module; this
# file only imports it once a knob is armed (the exact string 1; the
# profile's default 0 costs no import) and makes two calls.
_KDA_ONEPASS_MODULE = "vllm.model_executor.layers.glm53_kda_onepass"
_KDA_ONEPASS_ENVS = ("VLLM_GLM53_KDA_DUAL_GEMM", "VLLM_GLM53_KDA_ONEPASS")
_kda_fusion: dict = {"resolved": False, "mod": None, "dual": False, "onepass": False}


def _kda_fusion_state() -> dict:
    """Resolve the module once, on the first eager forward (never under
    capture: resolve() allocates its counters and runs the self-test)."""
    st = _kda_fusion
    if st["resolved"]:
        return st
    st["resolved"] = True
    import os as _os

    if not any(_os.environ.get(k, "").strip() == "1" for k in _KDA_ONEPASS_ENVS):
        return st
    try:
        import importlib

        mod = importlib.import_module(_KDA_ONEPASS_MODULE)
    except ImportError as e:
        if isinstance(e, ModuleNotFoundError) and e.name == _KDA_ONEPASS_MODULE:
            logger.warning(
                "[kda-onepass] a knob is set but %s is not mounted -> stock KDA chain",
                _KDA_ONEPASS_MODULE)
        else:
            logger.exception("[kda-onepass] import failed -> stock KDA chain")
        return st
    resolved = mod.resolve()
    st["mod"] = mod
    st["dual"] = bool(resolved["dual"])
    st["onepass"] = bool(resolved["onepass"])
    return st


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    # Declared int (set in __init__ from config) so mypy doesn't see the
    # getattr-derived `Any | None` at the kernel call sites.
    head_dim: int
    num_heads: int
    conv_size: int

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # conv_state width must include num_spec so the spec-decode conv update
        # (causal_conv1d_update with num_accepted_tokens + max_query_len) can
        # slide the window across the draft-verify tokens without reading past
        # the allocated width. Matches qwen_gdn_linear_attn.get_state_shape.
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # GLM5-Next keeps the KDA projections BF16 even in fp8 checkpoints (no
        # weight_scale_inv is stored for them), so strip the quant config for
        # this layer's construction -- mirrors the MLA path.
        saved_quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config

        # Linear-attention head config: read the flattened top-level fields when
        # present (new schema); fall back to the legacy linear_attn_config dict
        # otherwise (shared base is also used by KimiLinearConfig). Narrow via
        # locals so the int-typed attrs are assigned a non-None value.
        head_dim = getattr(config, "linear_head_dim", None)
        num_heads = getattr(config, "linear_num_heads", None)
        conv_size = getattr(config, "linear_conv_kernel_dim", None)
        if head_dim is None or num_heads is None or conv_size is None:
            kda_config = config.linear_attn_config  # type: ignore[attr-defined]
            assert kda_config is not None, "linear_attn_config must be set"
            head_dim = kda_config["head_dim"]
            num_heads = kda_config["num_heads"]
            conv_size = kda_config["short_conv_kernel_size"]
        assert head_dim is not None
        assert num_heads is not None
        assert conv_size is not None
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.conv_size = conv_size
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.tp_size)

        # Merge q, k, v, b, f_a, g_a projections into one GEMM (6→1 launches).
        # Order matches checkpoint's fused_qkvbfg_a_proj convention.
        # Shards 4 (f_a) and 5 (g_a) are replicated across TP ranks.
        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                projection_size,  # q (shard 0)
                projection_size,  # k (shard 1)
                projection_size,  # v (shard 2)
                self.num_heads,  # b (shard 3)
                self.head_dim,  # f_a (shard 4, replicated)
                self.head_dim,  # g_a (shard 5, replicated)
            ],
            replicated_shard_ids=(4, 5),
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvbfg_a",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)
        # Lazily-built merged q|k|v conv weight (built on first forward, after
        # weights are loaded). See _forward.
        self._merged_conv_weight: torch.Tensor | None = None

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # GLM5-Next checkpoints A_log as 1-D (num_heads,); the param is 4-D, so
        # reshape on load before the sharded loader runs.
        def _a_log_weight_loader(param, loaded_weight):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        self.A_log.weight_loader = _a_log_weight_loader

        # Bounded KDA gate variant: GLM5-Next uses
        # y = lower_bound * sigmoid(exp(A)*(g+g_bias)) instead of the default
        # unbounded y = -exp(A)*softplus(g+g_bias). Read by _forward.
        linear_lower_bound = getattr(config, "linear_lower_bound", None)
        if linear_lower_bound is not None:
            self.kda_safe_gate = True
            self.kda_lower_bound = linear_lower_bound
        else:
            legacy = getattr(config, "linear_attn_config", None) or {}
            if legacy.get("safe_gate", True):
                self.kda_safe_gate = True
                self.kda_lower_bound = legacy.get("lower_bound", -5.0)
            else:
                self.kda_safe_gate = False
                self.kda_lower_bound = -5.0
        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        self._conv_state_dim_first = is_conv_state_dim_first()

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # (deneb fork: the megakernel's MK_SEG_KDA takeover of this block --
        # one persistent launch for spec-verify decode steps -- was sunset in
        # 34차 §8; the stock chain below, with KDA_ONEPASS / KDA_DUAL_GEMM
        # behind their own knobs, is the block's only path.)
        num_tokens = hidden_states.size(0)
        # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 separate GEMMs).
        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )

        # Beta stays raw (bf16) here: the recurrent kernel sigmoids it in fp32
        # at load (SIGMOID_BETA), and only the chunked prefill path needs the
        # pre-computed fp32 sigmoid — computed lazily in _forward. Pure decode
        # / spec-verify steps then skip the _cast_sigmoid kernel and its fp32
        # intermediate entirely.
        beta = beta_raw.unsqueeze(0)
        # deneb fork (glm53_kda_onepass): f_b and g_b read adjacent 128-column
        # slices of the same merged row -- one launch for both when armed;
        # None = the stock two GEMMs (prefill by design, or a declined shape).
        _fus = _kda_fusion_state()
        _pair = None
        if _fus["dual"]:
            _pair = _fus["mod"].gate_gemms(
                projected, 3 * self.local_projection_size + self.local_num_heads,
                self.head_dim, self.f_b_proj.weight, self.g_b_proj.weight)
        if _pair is None:
            _pair = (self.f_b_proj(f_a)[0], self.g_b_proj(g_a)[0])
        _g1, _gp = _pair
        g1 = _g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        g_proj_states = _gp
        # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
        g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Call _forward directly (not via the registered op) so the KDA core
        # is an eager break point under Breakable CG, mirroring KimiK3's KDA
        # (vllm/models/kimi_k3/nvidia/kda.py). torch.ops.vllm.kda_attention is
        # neither a splitting op nor @eager_break_during_capture-decorated, so
        # routing through it lets the host-branching prefill body be
        # Inductor-compiled + stream-captured under PIECEWISE -> stale garbage.
        # qkv stays merged through the short-conv (one conv call, not three).
        # deneb fork (glm53_kda_onepass): _forward owns the gated norm for
        # every path (stock: o_norm in place after the merge; one-pass: in the
        # kernel), so no flag has to travel back here.
        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
            g2=g2,
            projected=projected,
            g2_flat=g_proj_states,
        )
        core_attn_out = core_attn_out.reshape(core_attn_out.size(1), -1)
        out = self.o_proj(core_attn_out)[0]
        return out

    @eager_break_during_capture
    def _forward(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        g2: torch.Tensor,
        projected: torch.Tensor,
        g2_flat: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            #     # V1 profile run
            self.o_norm(core_attn_out, g2)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        # Spec-decode metadata (all None when speculative decoding is disabled).
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = spec_sequence_masks is not None and num_spec_decodes > 0
        # KDA gate variant: GLM5-Next checkpoints with
        # linear_attn_config["safe_gate"]=True use the bounded gate
        # y=lower_bound*sigmoid(exp(A)*(g+g_bias)) instead of the default
        # unbounded y=-exp(A)*softplus(g+g_bias). Both attrs are always set
        # in __init__ (this class is GLM5Next-only).
        safe_gate = self.kda_safe_gate
        lower_bound = self.kda_lower_bound
        constant_caches = self.kv_cache

        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        # Layout is process-global and resolved once at init (see __init__).
        if not self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)

        # One merged short-conv over q|k|v instead of three separate calls. The
        # 1D conv is independent per channel, so concatenating q/k/v along the
        # channel dim and running a single causal_conv1d is bit-identical to
        # three calls. The merged weight is q|k|v conv weights concatenated;
        # built once and cached (params are fixed after load). conv_state is
        # already stored as the merged q|k|v state, so it is used directly.
        if self._merged_conv_weight is None:

            def _w(m):
                return m.weight.view(m.weight.size(0), m.weight.size(2))

            self._merged_conv_weight = torch.cat(
                [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                dim=0,
            ).contiguous()
        conv_weights = self._merged_conv_weight
        conv_bias = self.q_conv1d.bias

        # deneb fork (glm53_kda_onepass): a pure spec-verify step -- every
        # token belongs to a draft-verify block, no prefill and no plain
        # decode rows -- runs conv + recurrence + gated norm as ONE launch
        # straight from the merged projection rows (no q/k/v/beta copies).
        # The module decides on the real tensors and announces its verdict
        # once; False means the stock chain below serves this step.
        _fus = _kda_fusion_state()
        if (
            _fus["onepass"]
            and use_spec
            and attn_metadata_narrowed.num_prefills == 0
            and attn_metadata_narrowed.num_decodes == 0
            and (non_spec_token_indx is None or non_spec_token_indx.numel() == 0)
            and self.kda_safe_gate
            and self.o_norm.activation == "sigmoid"
            and self.o_norm.weight is not None
            and _fus["mod"].spec_onepass(
                self, projected, g1, g2_flat, conv_weights, conv_state,
                recurrent_state, spec_query_start_loc, spec_state_indices_tensor,
                num_accepted_tokens, core_attn_out[0, :num_actual_tokens],
                num_actual_tokens=num_actual_tokens,
                num_spec_decodes=num_spec_decodes, lower_bound=float(lower_bound),
            )
        ):
            return

        # Split projections / gating into spec (draft-verify) and non-spec token
        # groups when speculative decoding is active. Spec tokens carry
        # num_spec+1 recurrent-state columns each and are advanced with
        # num_accepted_tokens for rejection-sampling rollback; non-spec tokens
        # are one-per-request. Mirrors olmo_gdn_linear_attn.py. Projections are
        # [n, *] (token dim 0); g1/beta are [1, n, h, d] (token dim 1).
        if use_spec:
            # In a pure spec-verify step (no non-spec tokens) the metadata
            # builder sets spec_token_indx = arange(num_actual_tokens), making
            # the index_select calls below identity copies. Skip them on this
            # steady-state decode hot path. The outputs alias the inputs here;
            # the downstream conv/recurrent kernels read them without mutating
            # in place, so the aliasing is safe.
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0:
                qkv_spec = qkv_proj_states
                g1_spec = g1
                beta_spec = beta
            else:
                qkv_spec = qkv_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
            if non_spec_token_indx is not None and non_spec_token_indx.numel() > 0:
                qkv_ns = qkv_proj_states.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
            else:
                qkv_ns = g1_ns = beta_ns = None
        else:
            qkv_spec = g1_spec = beta_spec = None
            qkv_ns, g1_ns, beta_ns = qkv_proj_states, g1, beta

        # --- causal conv1d: spec (draft-verify) path ---
        if use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            conv_idx = spec_state_indices_tensor[:, 0][:num_spec_decodes]
            conv_mql = spec_state_indices_tensor.size(-1)
            qkv_spec = causal_conv1d_update(
                qkv_spec,
                conv_state,
                conv_weights,
                conv_bias,
                activation="silu",
                conv_state_indices=conv_idx,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=conv_mql,
            )
            q_spec, k_spec, v_spec = qkv_spec.split(self.local_projection_size, dim=-1)

        # --- causal conv1d: non-spec path (prefill or plain decode) ---
        q_ns = k_ns = v_ns = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert qkv_ns is not None
            qkv_ns = causal_conv1d_fn(
                qkv_ns.transpose(0, 1),
                conv_weights,
                conv_bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_decodes
            ]
            qkv_ns = causal_conv1d_update(
                qkv_ns,
                conv_state,
                conv_weights,
                conv_bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)

        def _rearr(x):
            return x.reshape(1, -1, self.local_num_heads, self.head_dim)

        # --- core attention: spec (draft-verify) path ---
        core_attn_out_spec = None
        # In a pure spec-verify step (no non-spec tokens) the recurrent kernel
        # can write straight into the layer output buffer, skipping the
        # fresh allocation + copy below. Mixed steps must scatter via
        # spec_token_indx, so they keep the kernel-managed output.
        spec_out = (
            core_attn_out[0, :num_actual_tokens].unsqueeze(0)
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0
            else None
        )
        if use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            assert spec_query_start_loc is not None
            # Gate computed inside the recurrent kernel (COMPUTE_GATE) from
            # raw g1 — replicates fused_kda_gate's arithmetic bit-for-bit and
            # skips its launch + fp32 [n, H, D] intermediate per layer.
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=_rearr(q_spec),
                k=_rearr(k_spec),
                v=_rearr(v_spec),
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )

        # --- core attention: non-spec path (prefill or plain decode) ---
        core_attn_out_non_spec = None
        # Pure prefill can opt into the same direct destination as plain
        # decode. Mixed steps retain their gather/scatter merge below.
        ns_out = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert q_ns is not None
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            initial_state = gather_initial_states(
                recurrent_state, non_spec_state_indices_tensor, has_initial_state
            )
            ns_out = _kda_prefill_output_buffer(
                core_attn_out, num_actual_tokens, self.local_num_heads, self.head_dim,
                use_spec=use_spec,
                num_prefills=attn_metadata_narrowed.num_prefills,
                num_decodes=attn_metadata_narrowed.num_decodes,
            )
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                raw_g=g1_ns,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_ns is raw bf16 from forward.
                beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                out=ns_out,
            )
            # Init cache
            scatter_states(
                recurrent_state,
                last_recurrent_state,
                non_spec_state_indices_tensor,
            )
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            # Plain decode step (no spec tokens): token order is dense, so the
            # kernel can write straight into the layer output buffer. A mixed
            # step scatters non-spec output via non_spec_token_indx instead.
            # Gate computed in-kernel (COMPUTE_GATE), beta sigmoided in-kernel.
            if not use_spec:
                ns_out = spec_out
            core_attn_out_non_spec, _ = fused_recurrent_kda(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                g=g1_ns,
                beta=beta_ns,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata_narrowed.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                out=ns_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )

        # --- merge spec / non-spec outputs back into token order ---
        if use_spec and core_attn_out_non_spec is not None:
            assert core_attn_out_spec is not None
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged.squeeze(0)
        elif use_spec:
            assert core_attn_out_spec is not None
            if spec_out is None:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            assert core_attn_out_non_spec is not None
            if ns_out is None:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                    0, :num_actual_tokens
                ]
        # gated RMSNorm in place (rms_norm_gated writes y = x); the one-pass
        # path above applied it inside its kernel and returned before this
        self.o_norm(core_attn_out, g2)
