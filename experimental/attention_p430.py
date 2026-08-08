# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v4.common.ops import (
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.utils.math_utils import cdiv
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.utils.torch_utils import current_stream, direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

logger = init_logger(__name__)


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    ``fp8_ds_mla`` format is UE8M0 block-scaled fp8 packed as ``uint8`` (the
    canonical ``fp8_ds_mla`` string is written back onto ``cache_config`` so the
    page-size specs pick the 576B per-token slot). Plain-row backends store each
    token's KV row in its element dtype: bf16 or per-tensor FP8 E4M3.
    """
    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


def _resolve_skip_topk(
    config: Any,
    layer_id: int,
    local_start_layer: int = 0,
    local_end_layer: int | None = None,
) -> bool:
    """Whether ``layer_id`` reuses the previous C4A layer's top-k (IndexCache).

    deneb fork (2026-08-08): port of upstream vllm-project/vllm PR #51209
    (IndexCache for DeepSeek-V4, validated on V4-Flash-0731 incl. DSpark) onto
    this eldritch/b12x fork. Refer: https://arxiv.org/abs/2603.12201.

    Only C4A layers (``compress_ratio == 4``) run an indexer, so
    ``index_topk_freq`` / ``index_topk_pattern`` (``F`` = compute, ``S`` = reuse)
    are indexed over those layers, not over all layers; trailing
    ``compress_ratios`` entries (MTP / draft slots) are ignored.
    ``topk_indices_buffer`` is rank-local, so each pipeline stage's first C4A
    layer must compute its own top-k.
    """
    if not getattr(config, "use_index_cache", False):
        return False
    compress_ratios = getattr(config, "compress_ratios", None)
    if compress_ratios is None:
        return False
    num_hidden_layers = getattr(config, "num_hidden_layers", len(compress_ratios))
    c4a_layers = [
        i for i, ratio in enumerate(compress_ratios[:num_hidden_layers]) if ratio == 4
    ]
    if layer_id not in c4a_layers:
        return False
    c4a_idx = c4a_layers.index(layer_id)

    if local_end_layer is None:
        local_end_layer = num_hidden_layers
    local_c4a_layers = [
        i for i in c4a_layers if local_start_layer <= i < local_end_layer
    ]
    is_first_on_rank = bool(local_c4a_layers) and layer_id == local_c4a_layers[0]

    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is None:
        skip = c4a_idx % getattr(config, "index_topk_freq", 1) != 0
        return skip and not is_first_on_rank

    invalid = sorted(set(pattern) - {"F", "S"})
    if invalid:
        raise ValueError(
            f"index_topk_pattern only accepts 'F' (full) and 'S' (shared), "
            f"got {invalid}."
        )
    if len(pattern) != len(c4a_layers):
        raise ValueError(
            f"index_topk_pattern has {len(pattern)} entries but this model has "
            f"{len(c4a_layers)} C4A layers; one F/S character per C4A layer is "
            "required (V4 patterns are shorter than V3.2 ones)."
        )
    if pattern[c4a_idx] == "S" and is_first_on_rank:
        raise ValueError(
            f"index_topk_pattern marks C4A layer {layer_id} as shared, but it is "
            "the first C4A layer on its pipeline rank and has no previous "
            "top-k to reuse."
        )
    return pattern[c4a_idx] == "S"


class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass — ``DeepseekV4FlashMLAAttention`` /
    ``DeepseekV4FlashInferSM120Attention`` /
    ``DeepseekV4FlashInferMLAAttention`` (CUDA) or
    ``DeepseekV4ROCMAiterMLAAttention`` (ROCm) — selected by the platform-specific
    deepseek_v4 model module. The base is never instantiated directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8); False = plain
    # bf16 / per-tensor fp8 KV row. Backends can override the instance hook when
    # a single attention class dispatches across arch-specific layouts.
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the dummy-run
    # path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q/output buffers are allocated at.

        The layer allocates the q/output buffers at
        ``[N, get_padded_num_q_heads(n_local_heads), head_dim]``. Must satisfy
        ``result >= num_heads``. Backends with no padding constraint return
        ``num_heads``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Platform-specific sparse MLA forward; writes attention into ``output``."""
        raise NotImplementedError

    @abstractmethod
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + wo_a + wo_b output projection (platform-specific)."""
        raise NotImplementedError

    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        return self.use_fp8_ds_mla_layout

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        self._use_b12x_wo = bool(envs.VLLM_USE_B12X_WO_PROJECTION)
        self._b12x_wo_projection_weights: Any | None = None
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # NOTE(zyongye) Compress ratio can't be 0
        # we do this for because MTP layer is not included
        # in the compress ratio list
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        elif layer_id < len(config.compress_ratios):
            self.compress_ratio = config.compress_ratios[layer_id]
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        if self._use_b12x_wo:
            if not hasattr(self.wo_a, "weight_scale_inv"):
                raise RuntimeError(
                    "VLLM_USE_B12X_WO_PROJECTION requires FP8 wo_a.weight_scale_inv"
                )
            # Preserve checkpoint UE8M0 scales for the fused b12x WO kernel.
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        if self._use_b12x_wo:
            if not hasattr(self.wo_b, "weight_scale_inv"):
                raise RuntimeError(
                    "VLLM_USE_B12X_WO_PROJECTION requires FP8 wo_b.weight_scale_inv"
                )
            self.wo_a.b12x_skip_generic_block_fp8_linear = True
            self.wo_b.b12x_skip_generic_block_fp8_linear = True

        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer

        # deneb fork (PR #51209 port): IndexCache — skipped C4A layers keep their
        # indexer allocated but never run it, reusing the top-k the previous C4A
        # layer left in the shared rank-local topk_indices_buffer.
        pp_group = get_pp_group()
        _local_start, _local_end = get_pp_indices(
            config.num_hidden_layers, pp_group.rank_in_group, pp_group.world_size
        )
        self.skip_topk = _resolve_skip_topk(config, layer_id, _local_start, _local_end)
        if self.skip_topk:
            logger.info_once("IndexCache: some C4A layers reuse the previous top-k.")

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs compressor. None on
            # ROCm, where aux_stream_list is None.
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
            )

        # Will be None on ROCm for now.
        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format. The same
        # resolution drives the SWA cache tensor dtype below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Create the compressor for layers with compress_ratio > 1; after the
        # attention setup above so its KV-cache prefix (self.prefix) is set.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )

    def _validate_wo_projection_tensors(self) -> tuple[int, int, int, int]:
        if not hasattr(self.wo_a, "weight_scale_inv"):
            raise RuntimeError(
                "DeepSeek V4 b12x WO path requires wo_a.weight_scale_inv"
            )
        if not hasattr(self.wo_b, "weight_scale_inv"):
            raise RuntimeError(
                "DeepSeek V4 b12x WO path requires wo_b.weight_scale_inv"
            )
        if getattr(self.wo_a, "weight_scale_inv_is_cutlass_interleaved", False):
            raise RuntimeError(
                "DeepSeek V4 b12x WO path requires canonical wo_a scales"
            )
        if getattr(self.wo_b, "weight_scale_inv_is_cutlass_interleaved", False):
            raise RuntimeError(
                "DeepSeek V4 b12x WO path requires canonical wo_b scales"
            )

        groups = self.n_local_groups
        heads_per_group = self.n_local_heads // groups
        group_width = heads_per_group * self.head_dim
        rank = self.o_lora_rank
        hidden = self.hidden_size

        wo_a_shape = (groups * rank, group_width)
        wo_b_shape = (hidden, groups * rank)
        wo_a_scale_shape = (groups * ((rank + 127) // 128), (group_width + 127) // 128)
        wo_b_scale_shape = ((hidden + 127) // 128, (groups * rank + 127) // 128)

        if tuple(self.wo_a.weight.shape) != wo_a_shape:
            raise RuntimeError(
                "DeepSeek V4 b12x WO-A weight shape mismatch: "
                f"expected {wo_a_shape}, got {tuple(self.wo_a.weight.shape)}"
            )
        if tuple(self.wo_b.weight.shape) != wo_b_shape:
            raise RuntimeError(
                "DeepSeek V4 b12x WO-B weight shape mismatch: "
                f"expected {wo_b_shape}, got {tuple(self.wo_b.weight.shape)}"
            )
        if tuple(self.wo_a.weight_scale_inv.shape) != wo_a_scale_shape:
            raise RuntimeError(
                "DeepSeek V4 b12x WO-A scale shape mismatch: "
                f"expected {wo_a_scale_shape}, "
                f"got {tuple(self.wo_a.weight_scale_inv.shape)}"
            )
        if tuple(self.wo_b.weight_scale_inv.shape) != wo_b_scale_shape:
            raise RuntimeError(
                "DeepSeek V4 b12x WO-B scale shape mismatch: "
                f"expected {wo_b_scale_shape}, "
                f"got {tuple(self.wo_b.weight_scale_inv.shape)}"
            )
        if self.wo_a.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "DeepSeek V4 b12x WO-A weight must be torch.float8_e4m3fn, "
                f"got {self.wo_a.weight.dtype}"
            )
        if self.wo_b.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "DeepSeek V4 b12x WO-B weight must be torch.float8_e4m3fn, "
                f"got {self.wo_b.weight.dtype}"
            )
        return groups, group_width, rank, hidden

    def setup_b12x_wo_projection(self) -> None:
        if not self._use_b12x_wo or self._b12x_wo_projection_weights is not None:
            return

        groups, group_width, rank, hidden = self._validate_wo_projection_tensors()

        from b12x.gemm.wo_projection import (
            pack_wo_projection_fp8_block_scaled_weights_mxfp8,
        )

        self._b12x_wo_projection_weights = (
            pack_wo_projection_fp8_block_scaled_weights_mxfp8(
                self.wo_a.weight.detach(),
                self.wo_a.weight_scale_inv.detach(),
                self.wo_b.weight.detach(),
                self.wo_b.weight_scale_inv.detach(),
                groups=groups,
                group_width=group_width,
                rank=rank,
                hidden=hidden,
            )
        )

    def _apply_b12x_wo_projection(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        *,
        o_storage: torch.Tensor | None = None,
        o_storage_offset: int = 0,
        o_stride_0: int = 0,
        o_stride_1: int = 0,
        o_stride_2: int = 0,
    ) -> torch.Tensor:
        del o_storage, o_storage_offset, o_stride_0, o_stride_1, o_stride_2
        num_tokens = int(o.shape[0])
        if num_tokens == 0:
            return torch.empty((0, self.hidden_size), dtype=o.dtype, device=o.device)
        if o.dtype != torch.bfloat16:
            raise RuntimeError(
                "DeepSeek V4 b12x WO projection requires bf16 attention output, "
                f"got {o.dtype}"
            )
        if self._b12x_wo_projection_weights is None:
            self.setup_b12x_wo_projection()
        weights = self._b12x_wo_projection_weights
        if weights is None:
            raise RuntimeError("DeepSeek V4 b12x WO weights were not packed")

        from b12x.gemm.wo_projection import wo_projection_inv_rope_mxfp8

        # Functional chain: each step allocates + returns its own output, so no
        # caller-owned scratch / bind is needed (and none can be mutated in the
        # traced graph). Pass the runtime tensors directly.
        out = wo_projection_inv_rope_mxfp8(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            weights,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            stream=current_stream().cuda_stream,
        )
        if self.wo_b.reduce_results and self.wo_b.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if envs.VLLM_USE_BREAKABLE_CUDAGRAPH and not self._use_b12x_wo:
            # DS4 breakable-cudagraph serving is faster when metadata-free
            # input GEMMs/RMSNorm stay in the captured graph and only the
            # metadata-dependent sparse attention body enters the eager break.
            # deneb fork (PR #51430): the attention input preparation is graph-safe
            # too, so it stays captured as well and only the sparse indexer op and
            # MLA attention run in the eager break below.
            # Keep the opaque custom op for AOT/non-breakable execution and for
            # B12X WO, where it prevents o_padded aliasing across graph pieces.
            qr_kv, kv_score, indexer_kv_score, indexer_weights = (
                self.attn_gemm_parallel_execute(hidden_states)
            )
            qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
            qr, kv = fused_q_kv_rmsnorm(
                qr,
                kv,
                self.q_norm.weight.data,
                self.kv_norm.weight.data,
                self.eps,
            )
            q, indexer_inputs = self._attn_prep(
                hidden_states,
                qr,
                kv,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                positions,
            )
            index_q, index_q_scale, index_k, index_weights_out = indexer_inputs
            self._sparse_indexer_and_attn(
                hidden_states,
                index_q,
                index_q_scale,
                index_k,
                index_weights_out,
                q,
                kv,
                positions,
                o_padded,
            )
        else:
            # Attention runs inside a single `out`-mutating custom op (the
            # torch.compile / breakable-cudagraph boundary). This is load-bearing:
            # without it, `attention_impl` is traced inline and the o_padded buffer
            # is threaded around the internal graph breaks (indexer / kv-cache
            # update) as TWO aliasing outputs of its producer piece. The AOT
            # piecewise runtime then merges that aliased pair into a single flat
            # 1-D synthetic base (torch.empty((0,)).set_(storage)), so the WO
            # consumer's assert_size_stride(o_padded, (tokens, heads, dim)) fails
            # with "wrong number of dimensions". Wrapping the whole attention as one
            # op makes o_padded the op's single mutated output, threaded cleanly
            # across the boundary, and keeps the attention's internal CUDA streams /
            # events out of the compiled artifact (so it stays serializable).
            torch.ops.vllm.deepseek_v4_attention(
                hidden_states,
                positions,
                o_padded,
                self.prefix,
            )
        o = o_padded[:, : self.n_local_heads, :]

        if self._use_b12x_wo:
            return torch.ops.vllm.deepseek_v4_b12x_wo_projection(
                o,
                positions,
                self.prefix,
                self.hidden_size,
                o_padded,
                0,
                self.padded_heads * self.head_dim,
                self.head_dim,
                1,
            )

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        # fused_wqa_wkv (heaviest) on default; the three lighter input GEMMs
        # on aux streams 0..2 when their owning module exists. ln_events[0]
        # is the fan-out start event; ln_events[1..3] are per-aux done events.
        # On ROCm, aux_streams is None and execute_in_parallel runs serially.
        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None and not self.skip_topk:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # ReplicatedLinear returns (output, bias); bias is None.
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            # MergedColumnParallelLinear returns (output, bias); bias is None.
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    # deneb fork: port of upstream PR #51430. The prep (wq_b + kv_insert +
    # compressor + indexer q/rope/quant) is metadata-dependent but graph-safe,
    # so it can live inside the captured cudagraph; only indexer_op and the MLA
    # attention need the eager break. Upstream simply renamed attention_impl,
    # but our fork calls it from a second site -- the opaque `deepseek_v4_attention`
    # custom op (AOT / non-breakable / B12X WO), which upstream does not have and
    # which must keep running prep and attention inside ONE decorated call. So the
    # body is split into two undecorated halves with two decorated entry points:
    # attention_impl (unchanged contract, custom-op path) and
    # _sparse_indexer_and_attn (narrowed, breakable-cudagraph path).
    def _attn_prep(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
        """Returns (q, indexer_inputs); indexer_inputs is all-None when the
        indexer does not run (SWA-only layer, or an IndexCache-skipped C4A)."""
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        indexer_inputs: tuple[torch.Tensor | None, ...] = (None, None, None, None)

        # wq_b + kv_insert (+ MLA compressor when an indexer is present) ride
        # on the default stream so q stays on its consumer stream (forward_mqa
        # downstream reads q on default). Indexer/compressor go on aux for
        # overlap with default's GEMM + cache write.
        if self.indexer is not None and not self.skip_topk:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                return q

            # 3-way overlap (matches TRT-LLM PR #14142 Level 1): default runs
            # wq_b+kv_insert; slot [0] prepares the indexer inputs; slot [1] runs
            # the MLA compressor. Slot [2] is reserved for the indexer's inner
            # overlap. ROCm (aux_streams is None) falls back to sequential.
            q, (indexer_inputs, _) = execute_in_parallel(
                wq_b_kv_insert,
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif self.compressor is not None:
            # wq_b + kv_insert on default, compressor on aux.
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                return q

            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        return q, indexer_inputs

    def _sparse_attn_body(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_q_scale: torch.Tensor | None,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        if self.indexer is not None and index_q is not None:
            assert index_weights is not None
            q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
            self.indexer.indexer_op(
                hidden_states,
                q_quant,
                index_k,
                index_weights,
            )

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)

    @eager_break_during_capture
    def _sparse_indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_q_scale: torch.Tensor | None,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Narrowed eager break: prep already ran inside the captured graph."""
        self._sparse_attn_body(
            hidden_states,
            index_q,
            index_q_scale,
            index_k,
            index_weights,
            q,
            kv,
            positions,
            out,
        )

    @eager_break_during_capture
    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        """Wide eager break: prep + sparse attention in one decorated call. Used
        only by the opaque `deepseek_v4_attention` custom op, whose whole point is
        that the entire attention path is a single mutating op."""
        q, indexer_inputs = self._attn_prep(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
        )
        self._sparse_attn_body(hidden_states, *indexer_inputs, q, kv, positions, out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # The fused insert ops require int64 position_ids; the runner's positions
        # buffer is already int64, so no cast is needed.
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cache_dtype = swa_kv_cache.dtype

        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if cache_dtype == torch.uint8:
            # fp8_ds_mla UE8M0 paged path. Horizontally fused:
            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling
            #            the padding head slots; the kernel allocates and returns
            #            the padded q tensor.
            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )

        # Plain-row path: the [num_blocks, block_size, 512] cache stores the KV
        # row in its element dtype (no Q padding). bf16 rewrites q in place;
        # per-tensor fp8 writes a separately-allocated fp8 q and quantizes the
        # KV row.
        block_size = swa_metadata.block_size
        swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
        if cache_dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
            return q

        # per-tensor fp8 (torch.float8_e4m3fn)
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            swa_kv_cache_3d,
            swa_metadata.slot_mapping,
            positions,
            cos_sin_cache,
            self._flashinfer_fp8_kv_scale,
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else None,
            model_version="deepseek_v4",
        )


def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    # Opaque wrapper around the whole MLA attention path. The layer is recovered
    # from the forward context by name so the op stays a plain custom op, while
    # `out` is threaded across piecewise compile as a single mutated output.
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    qr_kv, kv_score, indexer_kv_score, indexer_weights = (
        self.attn_gemm_parallel_execute(hidden_states)
    )
    qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
    qr, kv = fused_q_kv_rmsnorm(
        qr,
        kv,
        self.q_norm.weight.data,
        self.kv_norm.weight.data,
        self.eps,
    )
    self.attention_impl(
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        out,
    )


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)


def deepseek_v4_b12x_wo_projection(
    o: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    hidden_size: int,
    o_storage: torch.Tensor,
    o_storage_offset: int,
    o_stride_0: int,
    o_stride_1: int,
    o_stride_2: int,
) -> torch.Tensor:
    del hidden_size
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self._apply_b12x_wo_projection(
        o,
        positions,
        o_storage=o_storage,
        o_storage_offset=o_storage_offset,
        o_stride_0=o_stride_0,
        o_stride_1=o_stride_1,
        o_stride_2=o_stride_2,
    )


def deepseek_v4_b12x_wo_projection_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    hidden_size: int,
    o_storage: torch.Tensor,
    o_storage_offset: int,
    o_stride_0: int,
    o_stride_1: int,
    o_stride_2: int,
) -> torch.Tensor:
    del (
        positions,
        layer_name,
        o_storage,
        o_storage_offset,
        o_stride_0,
        o_stride_1,
        o_stride_2,
    )
    return torch.empty((o.shape[0], hidden_size), dtype=o.dtype, device=o.device)


direct_register_custom_op(
    op_name="deepseek_v4_b12x_wo_projection",
    op_func=deepseek_v4_b12x_wo_projection,
    fake_impl=deepseek_v4_b12x_wo_projection_fake,
)


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4. Both use the same
        # per-row indexer cache layout; compressed variants store fewer rows.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


def _indexer_sp_owned_ranges(k_cache_prefix: str):
    """Batch row ranges this rank must compute indexer-q for, under
    VLLM_DSV4_INDEXER_SP. None => compute all rows (feature off, warmup,
    decode-only batch, or metadata unavailable). Partition math MUST match
    the sparse_attn_indexer prefill-loop slicing exactly.
    """
    import os as _os

    if _os.environ.get("VLLM_DSV4_INDEXER_SP") != "1":
        return None
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        if tp <= 1:
            return None
        rank = get_tensor_model_parallel_rank()
    except Exception:
        return None
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    try:
        from vllm.model_executor.layers.sparse_attn_indexer import (
            _resolve_layer_name,
        )

        key = _resolve_layer_name(k_cache_prefix)
    except Exception:
        key = k_cache_prefix
    md = attn_metadata.get(key)
    if md is None:
        return None
    if getattr(md, "dcp_world_size", 1) != 1:
        return None
    if getattr(md, "num_prefills", 0) <= 0:
        return None
    prefill_md = getattr(md, "prefill", None)
    if prefill_md is None:
        return None
    ranges = []
    num_decode_tokens = int(getattr(md, "num_decode_tokens", 0))
    if num_decode_tokens > 0:
        # decode rows stay replicated on every rank
        ranges.append((0, num_decode_tokens))
    for chunk in prefill_md.chunks:
        n = int(chunk.token_end - chunk.token_start)
        if n < 256:
            # small chunks stay replicated (matches indexer gate)
            ranges.append((int(chunk.token_start), int(chunk.token_end)))
            continue
        shard = -(-n // tp)
        off = min(rank * shard, n)
        end = min(off + shard, n)
        if end > off:
            ranges.append(
                (int(chunk.token_start) + off, int(chunk.token_start) + end)
            )
    return ranges


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        cp_size = (
            vllm_config.parallel_config.prefill_context_parallel_size
            * vllm_config.parallel_config.decode_context_parallel_size
        )
        self.max_model_len = cdiv(
            vllm_config.model_config.max_model_len,
            self.compress_ratio * cp_size,
        )
        self.prefix = prefix

        self.max_total_seq_len = cdiv(
            get_max_prefill_buffer_size(vllm_config),
            self.compress_ratio * cp_size,
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indexer cache uses the same per-row layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

        # None on ROCm — maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self.ln_events: list[torch.cuda.Event] = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # deneb fork (PR #51430): prepare only. indexer_op is issued by the
        # caller's eager break so this whole body can stay in the captured graph.
        compressor = self.compressor

        def wq_b_and_q_quant():
            _ranges = _indexer_sp_owned_ranges(self.k_cache.prefix)
            if _ranges is not None:
                # Sequence-parallel: compute q/rope/quant only for the rows
                # this rank's sharded indexer will read; scatter into
                # full-size buffers (unowned rows stay uninitialized and are
                # never read by the sliced indexer paths).
                _idx = torch.cat(
                    [
                        torch.arange(a, b, device=qr.device, dtype=torch.long)
                        for a, b in _ranges
                    ]
                )
                _q_c, _ = self.wq_b(qr.index_select(0, _idx))
                _q_c = _q_c.view(-1, self.n_head, self.head_dim)
                _qq, _ww = fused_indexer_q_rope_quant(
                    positions.index_select(0, _idx),
                    _q_c,
                    rotary_emb.cos_sin_cache,
                    indexer_weights.index_select(0, _idx),
                    self.softmax_scale,
                    self.n_head**-0.5,
                    use_fp4=self.use_fp4_kv,
                )
                _n = qr.shape[0]

                def _scatter(part):
                    if isinstance(part, (tuple, list)):
                        return type(part)(_scatter(x) for x in part)
                    full = part.new_empty((_n,) + tuple(part.shape[1:]))
                    if part.element_size() == 1:
                        # index_copy_ lacks fp8 support; bit-copy via uint8
                        full.view(torch.uint8).index_copy_(
                            0, _idx, part.contiguous().view(torch.uint8)
                        )
                    else:
                        full.index_copy_(0, _idx, part)
                    return full

                return _scatter(_qq), _scatter(_ww)
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            q = q.view(-1, self.n_head, self.head_dim)
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
            )

        # compressor returns None and writes K to the indexer KV cache; the
        # join orders that write before indexer_op (skip_k_cache_insert=True).
        (q_quant, weights), k = maybe_execute_in_parallel(
            wq_b_and_q_quant,
            lambda: compressor(compressed_kv_score, positions, rotary_emb),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        if isinstance(q_quant, tuple):
            q, q_scale = q_quant
        else:
            q, q_scale = q_quant, None
        return q, q_scale, k, weights
