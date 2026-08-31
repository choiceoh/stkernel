# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import WeakValueDictionary

from vllm.logger import init_logger
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)


logger = init_logger(__name__)


# Attrs vLLM stashes on expert Parameters. Replacing a Parameter for the EP
# dummy row has to copy these or a later load_weights / EPLB walk dies.
_VLLM_WEIGHT_ATTRS = (
    "weight_loader",
    "input_dim",
    "output_dim",
    "packed_dim",
    "pack_factor",
    "load_hint",
)


def b12x_ep_kernel_expert_count(num_local_experts: int, use_ep: bool) -> int:
    """Experts the b12x wrapper is constructed with.

    EP adds one dummy so remote top-k slots have an isolated expert to land
    on. The fused kernel is never told ``num_local != num_experts`` — that
    path is flashinfer #3383 (weight_E vs state_E, then illegal address).
    """
    return num_local_experts + 1 if use_ep else num_local_experts


def remap_b12x_ep_slot(
    expert,
    *,
    num_local_experts: int,
    local_expert_offset: int = 0,
    expert_map=None,
) -> int:
    """One top-k slot → local expert id, or -1 if remote / padding.

    ``topk_ids`` may carry -1 as a padding sentinel. That must not index
    ``expert_map``: PyTorch advanced indexing wraps -1 onto the last row,
    which is a real expert on some ranks, at the original scale. That is
    the FC2-quant pollution this remap exists to prevent.
    """
    expert = int(expert)
    if expert < 0:
        return -1
    if expert_map is not None:
        if expert >= len(expert_map):
            return -1
        return int(expert_map[expert])
    local = expert - int(local_expert_offset)
    if local < 0 or local >= num_local_experts:
        return -1
    return local


def remap_b12x_ep_routing(
    topk_ids,
    topk_weights,
    *,
    num_local_experts: int,
    local_expert_offset: int = 0,
    expert_map=None,
):
    """Map global top-k ids onto a local-only b12x kernel.

    The SM12x fused kernel indexes weights and ``virt_route_scratch`` with
    the ids it is given. Under EP those ids are global and the weights are
    local — that is flashinfer #3383. We never show the kernel an EP
    geometry: it sees ``num_local_experts + 1`` experts, the extra one a
    dummy that receives every remote slot at scale 0.

    ``expert_map``, when given, is the vLLM table (global → local or -1).
    Without it the linear shard ``[offset, offset+local)`` is assumed.

    Dumping remote slots onto a *real* local expert is forbidden. b12x
    dynamically quantizes FC2 input per expert batch, so a ghost token in
    expert 0 would change that expert's real tokens.
    """
    dummy = num_local_experts
    out_ids = []
    out_w = []
    for ids, weights in zip(topk_ids, topk_weights):
        row_ids = []
        row_w = []
        for expert, weight in zip(ids, weights):
            local = remap_b12x_ep_slot(
                expert,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
                expert_map=expert_map,
            )
            if local < 0:
                row_ids.append(dummy)
                row_w.append(0.0)
            else:
                row_ids.append(int(local))
                row_w.append(float(weight))
        out_ids.append(row_ids)
        out_w.append(row_w)
    return out_ids, out_w


# b12x static→dynamic cutover is routed_rows = tokens * top_k against 640.
# Decode graphs at GRAPH_CAP=32 * top_k=8 = 256 stay on the dummy remap
# (fixed shape, alloc-free). Prefill crosses 640 and drops dummy slots.
B12X_EP_COMPACT_MIN_ROUTED = 640


def b12x_ep_compact_enabled(env_get=os.environ.get) -> bool:
    return env_get("VLLM_B12X_EP_COMPACT", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def b12x_ep_should_compact(
    routed_pairs,
    *,
    enabled=True,
    min_routed=B12X_EP_COMPACT_MIN_ROUTED,
) -> bool:
    return bool(enabled) and int(routed_pairs) > int(min_routed)


def compact_b12x_ep_pairs(ids, weights, *, dummy):
    """Keep only local slots. Returns (token_index, local_ids, scales)."""
    tok, loc, sc = [], [], []
    for t, (row_i, row_w) in enumerate(zip(ids, weights)):
        for expert, weight in zip(row_i, row_w):
            if int(expert) != dummy:
                tok.append(t)
                loc.append(int(expert))
                sc.append(float(weight))
    return tok, loc, sc


def _ep_buf(existing, shape, dtype, device):
    if existing is not None:
        return existing
    return torch.empty(shape, dtype=dtype, device=device)


def remap_b12x_ep_tensors(
    topk_ids,
    topk_weights,
    *,
    num_local_experts: int,
    local_expert_offset: int = 0,
    expert_map=None,
    out_ids=None,
    out_scales=None,
    long_idx=None,
    mapped=None,
    remote=None,
    tmp_a=None,
    tmp_b=None,
):
    """Runtime remap. Same slot rule as ``remap_b12x_ep_slot``.

    All intermediates take ``out=`` / in-place ops when scratch buffers
    are passed. ``apply`` preallocates those once; a call that omits them
    (tests) allocates for that call only.
    """
    shape = topk_ids.shape
    device = topk_ids.device
    dummy = num_local_experts
    out_ids = _ep_buf(out_ids, shape, torch.int32, device)
    out_scales = _ep_buf(out_scales, shape, topk_weights.dtype, device)
    remote = _ep_buf(remote, shape, torch.bool, device)
    tmp_a = _ep_buf(tmp_a, shape, torch.bool, device)
    tmp_b = _ep_buf(tmp_b, shape, torch.bool, device)

    if expert_map is not None:
        map_len = int(expert_map.size(0))
        if map_len <= 0:
            out_ids.fill_(dummy)
            out_scales.copy_(topk_weights)
            out_scales.zero_()
            return out_ids, out_scales
        long_idx = _ep_buf(long_idx, shape, torch.int64, device)
        mapped = _ep_buf(mapped, shape, expert_map.dtype, device)
        long_idx.copy_(topk_ids)
        torch.ge(long_idx, 0, out=tmp_a)
        torch.lt(long_idx, map_len, out=tmp_b)
        torch.logical_and(tmp_a, tmp_b, out=tmp_a)
        # in_range lives in tmp_a. Clamp is only a safe gather index.
        long_idx.clamp_(0, map_len - 1)
        torch.gather(expert_map, 0, long_idx.reshape(-1), out=mapped.reshape(-1))
        torch.lt(mapped, 0, out=tmp_b)
        torch.logical_not(tmp_a, out=remote)
        torch.logical_or(remote, tmp_b, out=remote)
        out_ids.copy_(mapped)
    else:
        out_ids.copy_(topk_ids)
        out_ids.sub_(int(local_expert_offset))
        torch.lt(topk_ids, 0, out=tmp_a)
        torch.lt(out_ids, 0, out=tmp_b)
        torch.logical_or(tmp_a, tmp_b, out=remote)
        torch.ge(out_ids, num_local_experts, out=tmp_a)
        torch.logical_or(remote, tmp_a, out=remote)

    out_ids.masked_fill_(remote, dummy)
    out_scales.copy_(topk_weights)
    out_scales.masked_fill_(remote, 0)
    return out_ids, out_scales


# Weight rows the dummy pad must extend. Optional scale_2 tensors skip
# when absent; a present tensor with the wrong E is a hard error.
_B12X_EP_PAD_REQUIRED = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
)


def b12x_ep_pad_dim0(dim0, num_local_experts, *, required, name):
    """What to do with one expert-major tensor at dummy-pad time.

    Returns ``pad``, ``already``, or ``skip``. Raises on a required hole
    or a first-dim that is neither local nor local+1.
    """
    if dim0 is None:
        if required:
            raise RuntimeError(f"b12x EP dummy pad: {name} missing")
        return "skip"
    if dim0 == num_local_experts:
        return "pad"
    if dim0 == num_local_experts + 1:
        return "already"
    raise RuntimeError(
        f"b12x EP dummy pad: {name} E={dim0} want {num_local_experts}"
    )


# FusedMoEExperts.w1_scale (and w2 / g1_alphas / g2_alphas / a2_gscale) are
# read-only properties over FusedMoEQuantConfig. Assigning self.w1_scale
# raises AttributeError on the image (glm53:v13-b12x). Write the QuantDesc
# fields the properties actually return.
_B12X_EP_SCALE_ALIASES = {
    "w1_scale": ("quant_config._w1.scale",),
    "w2_scale": ("quant_config._w2.scale",),
    "g1_alphas": ("quant_config._w1.alpha_or_gscale", "_g1_alphas"),
    "g2_alphas": ("quant_config._w2.alpha_or_gscale", "_g2_alphas"),
    "a2_gscale": ("quant_config._a2.alpha_or_gscale",),
}


def _b12x_ep_set_dotted(obj, path, value) -> bool:
    cur = obj
    parts = path.split(".")
    for part in parts[:-1]:
        cur = getattr(cur, part, None)
        if cur is None:
            return False
    try:
        setattr(cur, parts[-1], value)
    except AttributeError:
        return False
    return True


def b12x_ep_set_scale(obj, name, value):
    """Bind a dummy-padded scale so ``obj.name`` reads ``value``.

    Tries a direct setattr first (plain attributes), then the QuantDesc
    aliases. Raises if the readable value is still not ``value``.
    """
    try:
        setattr(obj, name, value)
        return name
    except AttributeError:
        pass
    for path in _B12X_EP_SCALE_ALIASES.get(name, ()):
        if _b12x_ep_set_dotted(obj, path, value):
            current = getattr(obj, name, None)
            if current is value:
                return path
    current = getattr(obj, name, None)
    if current is value:
        return "already"
    raise RuntimeError(
        f"b12x EP dummy pad: cannot bind {name} "
        f"(read-only property; aliases {_B12X_EP_SCALE_ALIASES.get(name, ())} "
        "did not take the write)"
    )


def _cat_dummy_row(tensor: "torch.Tensor", fill: float) -> "torch.Tensor":
    dummy = tensor.new_empty((1, *tensor.shape[1:]))
    dummy.fill_(fill)
    return torch.cat([tensor.detach(), dummy], dim=0)


def _replace_dim0(module: "torch.nn.Module", name: str, new_tensor: "torch.Tensor"):
    old = getattr(module, name)
    saved = {key: getattr(old, key) for key in _VLLM_WEIGHT_ATTRS if hasattr(old, key)}
    if isinstance(old, torch.nn.Parameter):
        new_param = torch.nn.Parameter(new_tensor, requires_grad=False)
        for key, value in saved.items():
            setattr(new_param, key, value)
        setattr(module, name, new_param)
        return new_param
    for key, value in saved.items():
        setattr(new_tensor, key, value)
    setattr(module, name, new_tensor)
    return new_tensor


@dataclass(frozen=True)
class _B12xWrapperKey:
    """Everything a B12xMoEWrapper's buffers are sized by."""

    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    max_num_tokens: int
    num_local_experts: int
    activation: str
    swiglu_alpha: float | None
    swiglu_beta: float | None
    swiglu_limit: float | None


# One wrapper per geometry, not per layer. Each carries ~541 MiB of graph-stable
# scratch on this deployment (measured: 288 experts, top_k 8, 4096/2048,
# max_num_tokens 2048), and GLM-5.3-Flash has 43 MoE layers of identical
# geometry -- 22.7 GiB per rank of duplicate buffers, a quarter of the whole GMU
# budget. It surfaced as KV that would not grow past ~16 GiB and as a fleet that
# ran out of memory at every turn.
#
# Weak values so a wrapper dies with the last layer holding it; layers keep the
# strong reference in self._wrapper. Sharing is safe because layers run one
# after another on the same stream -- the wrapper writes into its buffers on
# every call and nothing outlives the call.
#
# Same idea as vllm-project/vllm#48698, against the file this image ships.
_B12X_WRAPPERS: "WeakValueDictionary[_B12xWrapperKey, Any]" = WeakValueDictionary()
_B12X_WRAPPERS_LOCK = Lock()


def _shared_wrapper(key: _B12xWrapperKey):
    with _B12X_WRAPPERS_LOCK:
        w = _B12X_WRAPPERS.get(key)
        if w is None:
            from flashinfer.fused_moe import B12xMoEWrapper

            w = B12xMoEWrapper(
                num_experts=key.num_experts,
                top_k=key.top_k,
                hidden_size=key.hidden_size,
                intermediate_size=key.intermediate_size,
                use_cuda_graph=True,
                max_num_tokens=key.max_num_tokens,
                num_local_experts=key.num_local_experts,
                activation=key.activation,
                swiglu_alpha=key.swiglu_alpha,
                swiglu_beta=key.swiglu_beta,
                swiglu_limit=key.swiglu_limit,
            )
            _B12X_WRAPPERS[key] = w
            logger.info_once(
                "b12x MoE wrapper shared across layers with matching geometry "
                "(%d experts, top_k %d, %d/%d)",
                key.num_experts, key.top_k, key.hidden_size,
                key.intermediate_size,
            )
        return w


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark).

    Uses ``b12x_fused_moe`` from FlashInfer PR #3080 which fuses token
    dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
    single kernel call.  Input quantization (BF16→FP4) is performed inside the
    kernel so BF16 hidden states are passed directly.

    Weight scale factors are converted to the MMA layout produced by
    ``convert_sf_to_mma_layout`` once during ``process_weights_after_loading``
    and cached as ``w1_sf_mma`` / ``w2_sf_mma``.

    Only NVFP4 (kNvfp4Static/kNvfp4Dynamic) quantization is supported.

    Expert parallelism: the fused kernel rejects ``num_local != num_experts``
    and indexes weights by the ids it is given (flashinfer #3383). When EP
    is on we construct the wrapper as a local-only MoE (``E = local + 1``),
    remap global top-k ids onto that space, and park remote slots on a
    dummy expert at scale 0. Decode/graph batches stay on that dummy
    remap (alloc-free scratch). Prefill (routed pairs > 640) drops dummy
    slots and runs top_k=1 pairs so remote GEMM is not paid. vLLM's EP
    all-reduce (DP=1) combines ranks.
    """

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.GELU_TANH: "gelu_tanh",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == "nvfp4", (
            "FlashInferB12xExperts only supports nvfp4 quantization."
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        # FC2 input scale tensor bound in process_weights_after_loading: the
        # calibrated (now-zeroed) a2_gscale for static-quant checkpoints, or
        # a synthesized uniform-1.0 tensor for W4A16 checkpoints that lack
        # one. Holding it on the instance keeps apply() alloc-free.
        self._fc2_input_scale: torch.Tensor | None = None

        # Shape params for B12xMoEWrapper construction.
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts
        self._use_ep = bool(moe_config.moe_parallel_config.use_ep)
        self._ep_ids: torch.Tensor | None = None
        self._ep_scales: torch.Tensor | None = None
        self._ep_long: torch.Tensor | None = None
        self._ep_mapped: torch.Tensor | None = None
        self._ep_remote: torch.Tensor | None = None
        self._ep_tmp_a: torch.Tensor | None = None
        self._ep_tmp_b: torch.Tensor | None = None
        self._ep_dummy_padded = False
        self._ep_capacity_probed = False

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"FlashInferB12xExperts does not support "
                f"activation {activation!r}. "
                f"Supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        self._activation_str = self._ACTIVATION_MAP[activation]

        # SwiGLU clamp support. The kernel expresses a clamped gated
        # activation only under "swigluoai_uninterleave", whose math reduces
        # to plain clamped SwiGLU at alpha=1.0 / beta=0.0 — see this patch's
        # module docstring for the equivalence.
        limit = getattr(quant_config, "gemm1_clamp_limit", None)
        if limit is None:
            limit = getattr(moe_config, "swiglu_limit", None)
        self._swiglu_limit = limit
        self._swiglu_alpha = 1.0
        self._swiglu_beta = 0.0
        if limit is not None:
            if self._activation_str != "silu":
                raise ValueError(
                    "FlashInferB12xExperts can only clamp SiLU-gated MoE; "
                    f"got activation {self._activation_str!r} with "
                    f"swiglu_limit={limit}."
                )
            self._activation_str = "swigluoai_uninterleave"
            alpha = getattr(quant_config, "gemm1_alpha", None)
            beta = getattr(quant_config, "gemm1_beta", None)
            if alpha is not None:
                self._swiglu_alpha = float(alpha)
            if beta is not None:
                self._swiglu_beta = float(beta)

        # Lazily created on first apply() call.
        self._wrapper: Any | None = None
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Normalise block scales to absorb the per-expert weight global scale
        # (w_gs).  vLLM's NVFP4 convention stores:
        #   block_scale = max_abs * w_gs / fp4_max,  g1_alphas = 1/w_gs
        # The SM12x kernel treats w1_alpha (= g1_alphas) as a per-expert weight
        # dequant multiplier separate from input_gs (activation scale).  We bake
        # w_gs into the block scales so that w1_alpha = 1.0 and the kernel sees
        # the simpler form:
        #   block_scale = max_abs / fp4_max,  w1_alpha = 1.0
        # The FP4-packed values and dequantised results are identical in both
        # representations.  We set scale_2 = 1.0 to signal that the bake-in is
        # already done.
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.float() * layer.w13_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w13_weight_scale.dtype)
        layer.w13_weight_scale_2.data.fill_(1.0)

        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.float() * layer.w2_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w2_weight_scale.dtype)
        layer.w2_weight_scale_2.data.fill_(1.0)

        # The SM12x kernel uses dynamic per-block quantization for FC2 input
        # activations (the SwiGLU output before the down projection).  The
        # calibrated a2_gscale from the modelopt checkpoint (~tens to hundreds)
        # is intended for static-quantisation backends (TRTLLM/CUTLASS) and
        # causes every intermediate activation to saturate at max FP4 when
        # multiplied by values that large.  Force to 1.0 so the kernel uses
        # its own per-block dynamic scale.
        if self.a2_gscale is not None:
            self.a2_gscale.fill_(1.0)
            self._fc2_input_scale = self.a2_gscale
        else:
            # W4A16 NVFP4 checkpoints have no calibrated a2_gscale; b12x
            # performs dynamic per-block FC2-input quantization, so a uniform
            # 1.0 scale per expert is equivalent to the bake-in above for
            # static-quant checkpoints. Allocate once here so apply() stays
            # alloc-free.
            self._fc2_input_scale = torch.ones(
                self.num_local_experts,
                device=layer.w13_weight.device,
                dtype=torch.float32,
            )

        if self._use_ep:
            self._pad_dummy_expert(layer)

        # Precompute MMA-layout views of the weight scale factors once here
        # rather than recomputing on every forward pass.
        assert self.w1_scale is not None
        num_experts_w1, m1, k1_sf = self.w1_scale.shape
        k1 = k1_sf * 16
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1, k1_sf),
            m=m1,
            k=k1,
            num_groups=num_experts_w1,
        )

        assert self.w2_scale is not None
        num_experts_w2, m2, k2_sf = self.w2_scale.shape
        k2 = k2_sf * 16
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2, k2_sf),
            m=m2,
            k=k2,
            num_groups=num_experts_w2,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # b12x performs in-kernel BF16->FP4 activation quant, so W4A16
        # NVFP4 checkpoints (activation_key=None, e.g. mixed-precision
        # compressed-tensors layouts) are runtime-compatible.
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.GELU_TANH,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # EP is remapped onto a local-only wrapper in apply(). EPLB would
        # move experts after the dummy row is padded and is not wired.
        return not getattr(moe_parallel_config, "enable_eplb", False)

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x_fused_moe applies topk weights internally.
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # b12x_fused_moe manages its own internal workspace.
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # B12xMoEWrapper expects BF16 hidden states and performs its own FP4
        # quantization internally.  Returning True prevents the modular kernel
        # from pre-quantizing activations.
        return True

    @property
    def _kernel_num_experts(self) -> int:
        return b12x_ep_kernel_expert_count(self.num_local_experts, self._use_ep)

    def _pad_dummy_expert(self, layer: torch.nn.Module) -> None:
        """Append one zero-weight expert so remote top-k slots stay isolated.

        Done once after load. Peak memory is a brief cat; afterwards the
        extra expert is ~one NVFP4 expert (~8 MiB at 4096/2048).
        """
        if self._ep_dummy_padded:
            return
        n = self.num_local_experts
        for name, fill in (
            ("w13_weight", 0.0),
            ("w2_weight", 0.0),
            ("w13_weight_scale", 0.0),
            ("w2_weight_scale", 0.0),
            ("w13_weight_scale_2", 1.0),
            ("w2_weight_scale_2", 1.0),
        ):
            tensor = getattr(layer, name, None)
            action = b12x_ep_pad_dim0(
                None if tensor is None else int(tensor.shape[0]),
                n,
                required=name in _B12X_EP_PAD_REQUIRED,
                name=name,
            )
            if action != "pad":
                continue
            _replace_dim0(layer, name, _cat_dummy_row(tensor, fill))

        # Properties: w1_scale / w2_scale / g1_alphas have no setter.
        # They read FusedMoEQuantConfig QuantDesc fields — write those.
        b12x_ep_set_scale(self, "w1_scale", layer.w13_weight_scale)
        b12x_ep_set_scale(self, "w2_scale", layer.w2_weight_scale)
        ones = torch.ones(
            n + 1, device=layer.w13_weight.device, dtype=torch.float32
        )
        self._fc2_input_scale = ones
        for name in ("g1_alphas", "g2_alphas", "a2_gscale"):
            if getattr(self, name, None) is None:
                continue
            b12x_ep_set_scale(self, name, ones.clone())

        self._ep_dummy_padded = True
        logger.info_once(
            "b12x EP: wrapper sees %d local experts + 1 dummy "
            "(global %d, rank %d); remote top-k slots map to the dummy",
            self.num_local_experts, self.global_num_experts, self.ep_rank,
        )

    def _ensure_ep_scratch(
        self,
        device: torch.device,
        scale_dtype: torch.dtype,
        map_dtype: torch.dtype,
    ) -> None:
        need = (
            self._ep_ids is None
            or self._ep_ids.device != device
            or self._ep_scales is None
            or self._ep_scales.dtype != scale_dtype
            or self._ep_mapped is None
            or self._ep_mapped.dtype != map_dtype
        )
        if not need:
            return
        rows = max(int(self.max_num_tokens or 0), 1)
        shape = (rows, self.topk)
        self._ep_ids = torch.empty(shape, dtype=torch.int32, device=device)
        self._ep_scales = torch.empty(shape, dtype=scale_dtype, device=device)
        self._ep_long = torch.empty(shape, dtype=torch.int64, device=device)
        self._ep_mapped = torch.empty(shape, dtype=map_dtype, device=device)
        self._ep_remote = torch.empty(shape, dtype=torch.bool, device=device)
        self._ep_tmp_a = torch.empty(shape, dtype=torch.bool, device=device)
        self._ep_tmp_b = torch.empty(shape, dtype=torch.bool, device=device)

    def _remap_ep_tensors(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = topk_ids.size(0)
        if tokens > self._ep_ids.size(0):
            raise ValueError(
                f"b12x EP remap: {tokens} tokens exceeds "
                f"max_num_tokens={self._ep_ids.size(0)}"
            )
        return remap_b12x_ep_tensors(
            topk_ids,
            topk_weights,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            expert_map=expert_map,
            out_ids=self._ep_ids[:tokens],
            out_scales=self._ep_scales[:tokens],
            long_idx=self._ep_long[:tokens],
            mapped=self._ep_mapped[:tokens],
            remote=self._ep_remote[:tokens],
            tmp_a=self._ep_tmp_a[:tokens],
            tmp_b=self._ep_tmp_b[:tokens],
        )

    def _apply_ep_compact(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Run only local slots as top_k=1 pairs. Eager / prefill only.

        After remap, dummy id = num_local_experts. Dropping those slots
        is how EP avoids paying GEMM for ~3/4 of routed pairs. Shape is
        data-dependent, so this path must not run under a CUDA graph.
        """
        dummy = self.num_local_experts
        local = topk_ids != dummy
        sel = local.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        n = int(sel.numel())
        if n == 0:
            output.zero_()
            return output

        tokens = topk_ids.size(0)
        topk = topk_ids.size(1)
        token_index = (
            torch.arange(tokens, device=topk_ids.device, dtype=torch.int64)
            .unsqueeze(1)
            .expand(tokens, topk)
            .reshape(-1)
            .index_select(0, sel)
        )
        pair_ids = topk_ids.reshape(-1).index_select(0, sel).view(n, 1)
        pair_scales = topk_weights.reshape(-1).index_select(0, sel).view(n, 1)
        pair_x = hidden_states.index_select(0, token_index)
        pair_out = torch.empty(
            (n, hidden_states.size(1)),
            dtype=output.dtype,
            device=output.device,
        )

        from flashinfer.fused_moe import b12x_fused_moe

        kernel_e = self._kernel_num_experts
        # The wrapper's scratch is sized for max_num_tokens ROWS, but a
        # compacted pair list is up to tokens*top_k long -- 8x over on this
        # model. Overrunning it is an illegal write whose fault surfaces at
        # whatever kernel syncs next (we found it as an ILLEGAL_ADDRESS inside
        # the dense fp8 GEMM two layers later). Walk the pairs in slices the
        # scratch can hold. Growing max_num_tokens instead would key a second
        # shared wrapper and double the workspace for all 43 MoE layers.
        limit = max(int(self.max_num_tokens or 0), 1)
        for lo in range(0, n, limit):
            hi = min(lo + limit, n)
            b12x_fused_moe(
                x=pair_x[lo:hi],
                w1_weight=w1,
                w1_weight_sf=self.w1_sf_mma,
                w2_weight=w2,
                w2_weight_sf=self.w2_sf_mma,
                token_selected_experts=pair_ids[lo:hi],
                token_final_scales=pair_scales[lo:hi],
                num_experts=kernel_e,
                top_k=1,
                num_local_experts=kernel_e,
                w1_alpha=self.g1_alphas,
                w2_alpha=self.g2_alphas,
                fc2_input_scale=self._fc2_input_scale,
                output=pair_out[lo:hi],
                activation=self._activation_str,
                swiglu_alpha=self._swiglu_alpha,
                swiglu_beta=self._swiglu_beta,
                swiglu_limit=self._swiglu_limit,
            )
        output.zero_()
        output.index_add_(0, token_index, pair_out)
        return output

    def _ensure_wrapper(self) -> None:
        """Lazily create B12xMoEWrapper on first use."""
        if self._wrapper is not None:
            return

        kernel_e = self._kernel_num_experts
        self._wrapper = _shared_wrapper(
            _B12xWrapperKey(
                num_experts=kernel_e,
                top_k=self.topk,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                max_num_tokens=self.max_num_tokens,
                num_local_experts=kernel_e,
                activation=self._activation_str,
                swiglu_alpha=self._swiglu_alpha,
                swiglu_beta=self._swiglu_beta,
                swiglu_limit=self._swiglu_limit,
            )
        )

    def _ep_capacity_probe(self, wrapper, topk_ids) -> None:
        """Name the bound the b12x static kernel would overrun. Runs once.

        The static kernel compacts the distinct experts of a batch into
        arrival-order slots and writes ``token_map[slot, row]`` where ``row``
        is an unguarded ``atomicAdd`` on a per-expert counter. One expert
        receiving more than ``token_map.shape[1]`` rows is an out-of-bounds
        write, reported later as an ILLEGAL_ADDRESS in whatever kernel syncs
        next -- we first met it two layers away inside the dense fp8 GEMM.

        Under EP the dummy expert absorbs every remote slot, which on this
        model is ~3/4 of all routing, so it is the one that can overrun.
        One device sync, once per process.
        """
        if self._ep_capacity_probed:
            return
        self._ep_capacity_probed = True
        try:
            flat = topk_ids.reshape(-1).to(torch.int64)
            counts = torch.bincount(flat, minlength=self._kernel_num_experts)
            worst = int(counts.max())
            worst_id = int(counts.argmax())
            pairs = int(flat.numel())
            caps = []
            for name in ("_static_workspace", "_dynamic_workspace"):
                ws = getattr(wrapper, name, None)
                tm = getattr(ws, "token_map", None) if ws is not None else None
                caps.append(
                    f"{name.strip('_').split('_')[0]}={tuple(tm.shape)}"
                    if tm is not None else f"{name.strip('_').split('_')[0]}=none"
                )
            logger.warning(
                "[b12x EP capacity] pairs=%d experts=%d worst expert=%d rows=%d "
                "(dummy=%d) token_map %s",
                pairs, self._kernel_num_experts, worst_id, worst,
                self.num_local_experts, " ".join(caps),
            )
            # Raise only when NO workspace can hold this routing. The two
            # layouts differ in kind: static's token_map is
            # [state_E, max_rows] -- per-expert capacity, so EP's dummy pile-up
            # is what overruns it -- while dynamic's is a flat compacted plane
            # sized by total pairs, which the skew cannot reach. A disabled
            # static workspace (cutover pinned to 0 leaves max_rows=1) must not
            # be read as a failure while dynamic is the one being selected.
            fits = []
            for name in ("_static_workspace", "_dynamic_workspace"):
                ws = getattr(wrapper, name, None)
                tm = getattr(ws, "token_map", None) if ws is not None else None
                if tm is None:
                    continue
                if tm.dim() >= 2:
                    ok = (
                        worst <= int(tm.shape[1])
                        and self._kernel_num_experts <= int(tm.shape[0])
                    )
                else:
                    ok = pairs <= int(tm.shape[0])
                fits.append((name, ok, tuple(tm.shape)))
            if fits and not any(ok for _, ok, _ in fits):
                detail = " ".join(f"{n}{shape}" for n, _, shape in fits)
                raise RuntimeError(
                    f"b12x EP overruns every workspace: {pairs} pairs, expert "
                    f"{worst_id} takes {worst} rows, {self._kernel_num_experts} "
                    f"experts -- token_map {detail}. Force the dynamic backend "
                    f"(FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0) or lower "
                    f"MAX_BATCHED."
                )
        except RuntimeError:
            raise
        except Exception as exc:  # probe must never be the thing that breaks EP
            logger.warning("[b12x EP capacity] probe unavailable (%s: %s)",
                           type(exc).__name__, exc)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for FlashInferB12xExperts"
        )
        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for FlashInferB12xExperts"
        )
        assert self._fc2_input_scale is not None, (
            "_fc2_input_scale must be set by process_weights_after_loading"
        )
        assert self.w1_sf_mma is not None and self.w2_sf_mma is not None, (
            "process_weights_after_loading must run before FlashInferB12xExperts.apply"
        )

        self._ensure_wrapper()
        wrapper = self._wrapper
        assert wrapper is not None

        if self._use_ep:
            expect_e = self._kernel_num_experts
            if w1.size(0) != expect_e:
                raise RuntimeError(
                    f"b12x EP dummy pad missing: w1 E={w1.size(0)} "
                    f"want {expect_e} (local {self.num_local_experts} + dummy)"
                )
            if self.g1_alphas.numel() < expect_e or self.g2_alphas.numel() < expect_e:
                raise RuntimeError(
                    "b12x EP dummy pad missing on g1/g2 alphas "
                    f"(g1={tuple(self.g1_alphas.shape)} g2={tuple(self.g2_alphas.shape)} "
                    f"want {expect_e})"
                )
            map_dtype = (
                expert_map.dtype if expert_map is not None else torch.int32
            )
            self._ensure_ep_scratch(
                topk_ids.device, topk_weights.dtype, map_dtype
            )
            topk_ids, topk_weights = self._remap_ep_tensors(
                topk_ids, topk_weights, expert_map
            )
            self._ep_capacity_probe(wrapper, topk_ids)
            if b12x_ep_should_compact(
                topk_ids.size(0) * topk_ids.size(1),
                enabled=b12x_ep_compact_enabled(),
            ):
                return self._apply_ep_compact(
                    output, hidden_states, w1, w2, topk_ids, topk_weights
                )

        # deneb fork: when the wrapper supports out= (overlay module
        # glm53_b12x_out takes over flashinfer's b12x_moe.py to add it), make
        # the caller's buffer the scatter target so the MoE result is written
        # once. The copy_ this replaces was the second write of the same bytes
        # per layer, ~42 copy kernels/step on this lane. Rollback:
        # VLLM_B12X_DIRECT_OUT=0. Capture-safe for the same reason the copy
        # was: the buffer address replays either way.
        run_kwargs = dict(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
        )
        direct = os.environ.get(
            "VLLM_B12X_DIRECT_OUT", "1").strip().lower() in (
            "1", "true", "yes", "on")
        if direct:
            wrapper.run(**run_kwargs, out=output)
            return output
        output.copy_(wrapper.run(**run_kwargs))
