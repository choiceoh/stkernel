# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer

deneb fork (2026-08-08, slimmed for the TP=4 4x GB10 stack): paths that are
structurally unreachable on this deployment are removed —
- breakable-cudagraph forward split (VLLM_USE_BREAKABLE_CUDAGRAPH is pinned 0;
  enabling it kills the engine on this stack),
- b12x WO projection (VLLM_USE_B12X_WO_PROJECTION is not exposed by the
  launcher; setup_b12x_wo_projection stays as a no-op for external callers),
- plain-row bf16 / per-tensor-fp8 KV layouts (only the SM120 fp8_ds_mla
  subclass is ever instantiated on GB10),
- pipeline-parallel IndexCache rank math (PP=1 on this stack),
- index_topk_pattern (only index_topk_freq is exposed via hf-overrides),
- FP4 indexer cache (requires SM10x; GB10 is SM121).
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
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
from vllm.models.deepseek_v4.eager_scratch import get_eager_scratch_pool

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
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
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
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

try:
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _resolve_layer_name,
    )
except ImportError:  # pragma: no cover - fork variants without the helper
    _resolve_layer_name = None  # type: ignore[assignment]

logger = init_logger(__name__)

# VLLM_DSV4_COMPRESSOR_FP8 (default off): run the ATTENTION compressor input
# GEMM via fp8 deepgemm instead of bf16 cuBLAS. The bf16 kernel already sits AT
# the LPDDR bandwidth floor (measured 273 GB/s effective), so the only remaining
# lever is byte reduction: fp8 halves the ~0.7 GB/step these replicated weights
# stream, on a chip where decode is bandwidth-bound. Same lazy-quant pattern as
# DSparkMarkovHeadOptimized (VLLM_DSPARK_MARKOV_W2_FP8): first call happens in
# the eager warmup profile run, so CUDA-graph capture sees the steady-state
# branch. Scope is the VALUE path only: the indexer compressor (selection path
# — top-k flips instead of averaging; ~11% of the bytes) and weights_proj
# (N=64 < deepgemm 128 block) stay bf16.
_COMPRESSOR_FP8 = os.getenv(
    "VLLM_DSV4_COMPRESSOR_FP8", "0"
).strip().lower() in {"1", "true", "yes", "on"}


def _compressor_fp8_kv_score(
    owner: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """fp8-deepgemm replacement for torch.mm(x, fused_wkv_wgate.weight.T,
    out_dtype=fp32). Keeps the bf16 weight in place (loader/reload safety);
    the fp8 copies cost +50% memory on these weights (~0.35 GB/rank)."""
    if not hasattr(owner, "_dsv4_comp_fp8_w"):
        from vllm.models.deepseek_v4.nvidia.dspark_v2 import (
            _quantize_fp8_deepgemm,
        )
        dg_w, dg_ws = _quantize_fp8_deepgemm(owner.fused_wkv_wgate.weight)
        owner._dsv4_comp_fp8_w = dg_w
        owner._dsv4_comp_fp8_ws = dg_ws
        logger.info_once("DSV4 compressor kv_score GEMMs on fp8 deepgemm.")
    from vllm.models.deepseek_v4.nvidia.dspark_v2 import _fp8_gemm

    return _fp8_gemm(
        hidden_states, owner._dsv4_comp_fp8_w, owner._dsv4_comp_fp8_ws
    ).to(torch.float32)


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    """Map ``--kv-cache-dtype`` to ``(cache_dtype_str, torch_dtype)``.

    This stack only runs the fp8_ds_mla block format: UE8M0 block-scaled fp8
    packed as ``uint8`` (the canonical ``fp8_ds_mla`` string is written back
    onto ``cache_config`` so the page-size specs pick the 576B per-token slot).
    The plain-row bf16 / per-tensor FP8 layouts belonged to the SM100
    FlashInfer subclass, which is never instantiated on GB10.
    """
    assert use_fp8_ds_mla_layout, (
        "TP4 GB10 overlay: plain-row KV layouts were removed; only the "
        "fp8_ds_mla layout (SM120 subclass) is supported."
    )
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


def _resolve_skip_topk(config: Any, layer_id: int) -> bool:
    """Whether ``layer_id`` reuses the previous C4A layer's top-k (IndexCache).

    deneb fork (2026-08-08): port of upstream vllm-project/vllm PR #51209
    (IndexCache for DeepSeek-V4, validated on V4-Flash-0731 incl. DSpark) onto
    this eldritch/b12x fork. Refer: https://arxiv.org/abs/2603.12201.

    Only C4A layers (``compress_ratio == 4``) run an indexer, so
    ``index_topk_freq`` is indexed over those layers, not over all layers;
    trailing ``compress_ratios`` entries (MTP / draft slots) are ignored.
    ``topk_indices_buffer`` is rank-local, but with PP=1 (this stack) the
    first C4A layer overall (``c4a_idx == 0``) always computes its own top-k
    because ``0 % freq == 0``.
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
    return c4a_idx % getattr(config, "index_topk_freq", 1) != 0


class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by
    a subclass. On this stack that is always
    ``DeepseekV4FlashInferSM120Attention`` (GB10 / SM121); the base is never
    instantiated directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # KV-cache per-token block format. This stack only supports the default:
    # fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8).
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

        # Multi-stream GEMM gate; env is fixed per container, so read it once
        # instead of per forward call.
        self._multi_stream_gemm_threshold = (
            envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
        )

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
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

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

        # deneb fork (PR #51209 port): IndexCache — skipped C4A layers keep
        # their indexer modules (checkpoint weights must still load) but never
        # run them, reusing the top-k the previous C4A layer left in the
        # shared rank-local topk_indices_buffer. Their indexer k-cache spec is
        # dropped (spec_enabled=False below) so the never-touched cache costs
        # no KV pages.
        self.skip_topk = _resolve_skip_topk(config, layer_id)
        if self.skip_topk:
            logger.info_once("IndexCache: some C4A layers reuse the previous top-k.")

        # deneb fork: port of upstream PR #49236 (eager-scratch pool), slimmed.
        # Upstream plumbs the pool through nvidia/model.py; we keep the image
        # model.py untouched via a per-device singleton. See eager_scratch.py
        # for what was dropped (padded-q bucket needs a CUDA op this image
        # lacks; FP4 bucket replaced by the FP8 one this stack uses).
        self.eager_scratch_pool = get_eager_scratch_pool(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.head_dim,
            config.index_n_heads,
            config.index_head_dim,
            config.index_topk,
            torch.device("cuda"),
        )

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs compressor.
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
                skip_topk=self.skip_topk,
                eager_scratch_pool=self.eager_scratch_pool,
            )

        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]
        # Pre-sliced stream/event views for the per-call fan-outs:
        # attn_gemm_parallel_execute / attention_impl run per layer per eager
        # step, so their per-call python stays attribute reads only.
        if aux_stream_list is not None:
            assert len(aux_stream_list) >= 3
            self._aux_streams3 = aux_stream_list[:3]
            self._aux_streams2 = aux_stream_list[:2]
            self._aux_stream0 = aux_stream_list[0]
        else:
            self._aux_streams3 = None
            self._aux_streams2 = None
            self._aux_stream0 = None
        self._ln_done_events = self.ln_events[1:4]
        self._ln_events12 = self.ln_events[1:3]

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
        self._swa_prefix = self.swa_cache_layer.prefix
        # Identity-guarded caches for per-call tensor aliases: `.data` and
        # `.view()` construct a fresh python Tensor on every access, and both
        # ran per layer per eager step. The `is` guards keep a hypothetical
        # weight/cache rebind correct.
        self._split_sizes = [self.q_lora_rank, self.head_dim]
        self._q_norm_w_src: torch.Tensor | None = None
        self._q_norm_w: torch.Tensor | None = None
        self._kv_norm_w_src: torch.Tensor | None = None
        self._kv_norm_w: torch.Tensor | None = None
        self._swa_kv_2d_src: torch.Tensor | None = None
        self._swa_kv_2d: torch.Tensor | None = None

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix:
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
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

    def setup_b12x_wo_projection(self) -> None:
        """No-op: the b12x WO projection path is not used on this stack.

        Kept because the model runner may invoke it unconditionally after
        weight loading.
        """
        return

    def _qkv_norm_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        qp, kp = self.q_norm.weight, self.kv_norm.weight
        if self._q_norm_w_src is not qp:
            self._q_norm_w = qp.data
            self._q_norm_w_src = qp
        if self._kv_norm_w_src is not kp:
            self._kv_norm_w = kp.data
            self._kv_norm_w_src = kp
        return self._q_norm_w, self._kv_norm_w

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-allocate attention output with backend-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Attention runs inside a single `out`-mutating custom op (the
        # torch.compile boundary). This is load-bearing: without it,
        # `attention_impl` is traced inline and the o_padded buffer is threaded
        # around the internal graph breaks (indexer / kv-cache update) as TWO
        # aliasing outputs of its producer piece. The AOT piecewise runtime
        # then merges that aliased pair into a single flat 1-D synthetic base
        # (torch.empty((0,)).set_(storage)), so the WO consumer's
        # assert_size_stride(o_padded, (tokens, heads, dim)) fails with "wrong
        # number of dimensions". Wrapping the whole attention as one op makes
        # o_padded the op's single mutated output, threaded cleanly across the
        # boundary, and keeps the attention's internal CUDA streams / events
        # out of the compiled artifact (so it stays serializable).
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            positions,
            o_padded,
            self.prefix,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self._aux_streams3

        # fused_wqa_wkv (heaviest) on default; the three lighter input GEMMs
        # on aux streams 0..2 when their owning module exists. ln_events[0]
        # is the fan-out start event; ln_events[1..3] are per-aux done events.
        # With aux_streams None, execute_in_parallel runs serially.
        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                if _COMPRESSOR_FP8:
                    return _compressor_fp8_kv_score(compressor, hidden_states)
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
                # Stays bf16 by operator decision (2026-08-10): this is the
                # SELECTION path — it feeds index top-k, where quantization
                # error flips discrete choices instead of averaging out — and
                # it carries only ~11% of the compressor byte savings anyway.
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
            self._ln_done_events,
            aux_streams,
            enable=hidden_states.shape[0] <= self._multi_stream_gemm_threshold,
        )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

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
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # wq_b + kv_insert (+ MLA compressor when an indexer is present) ride
        # on the default stream so q stays on its consumer stream (forward_mqa
        # downstream reads q on default). Indexer/compressor go on aux for
        # overlap with default's GEMM + cache write.
        def wq_b_kv_insert() -> torch.Tensor:
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            return self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        if self.indexer is not None and not self.skip_topk:
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            # 3-way overlap (matches TRT-LLM PR #14142 Level 1): default runs
            # wq_b+kv_insert; slot [0] runs the full indexer; slot [1] runs the
            # MLA compressor. Slot [2] is reserved for the indexer's inner
            # overlap. aux_streams None falls back to sequential.
            q, _ = execute_in_parallel(
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
                self._ln_events12,
                self._aux_streams2,
                enable=self._aux_streams2 is not None,
            )
        elif self.compressor is not None:
            # wq_b + kv_insert on default, compressor on aux.
            compressor = self.compressor

            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                self._aux_stream0,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            q = wq_b_kv_insert()

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)

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
            # downstream gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self._swa_prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # The fused insert ops require int64 position_ids; the runner's positions
        # buffer is already int64, so no cast is needed.
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache

        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        # fp8_ds_mla UE8M0 paged path (the only layout on this stack).
        # Horizontally fused:
        #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling
        #            the padding head slots; the kernel allocates and returns
        #            the padded q tensor.
        #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.
        if self._swa_kv_2d_src is not swa_kv_cache:
            assert swa_kv_cache.dtype == torch.uint8, (
                "TP4 GB10 overlay: plain-row KV insert paths were removed; the "
                f"SWA cache must be fp8_ds_mla/uint8, got {swa_kv_cache.dtype}"
            )
            self._swa_kv_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            self._swa_kv_2d_src = swa_kv_cache
        swa_kv_cache_2d = self._swa_kv_2d
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

    def _global_topk_output_buffers(
        self, topk_indices: torch.Tensor
    ) -> "tuple[torch.Tensor, torch.Tensor] | None":
        # deneb fork: PR #49236. C4A-only: the pool's global bucket is sized
        # for index_topk; C128A keeps its own per-layer buffers.
        if self.compress_ratio != 4 or self.eager_scratch_pool is None:
            return None
        return self.eager_scratch_pool.global_topk_outputs(topk_indices)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment.
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,
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
    qr, kv = qr_kv.split(self._split_sizes, dim=-1)
    q_norm_w, kv_norm_w = self._qkv_norm_weights()
    qr, kv = fused_q_kv_rmsnorm(
        qr,
        kv,
        q_norm_w,
        kv_norm_w,
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


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
        spec_enabled: bool = True,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        self.spec_enabled = spec_enabled
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # IndexCache (#51209): a skip-topk layer never runs its indexer, so
        # its k-cache (and the compressor state packed into the same pages)
        # is never written or read. Returning None allocates no pages for it
        # — the same mechanism SWA-only attention layers use above.
        if not self.spec_enabled:
            return None
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


# VLLM_DSV4_INDEXER_SP is fixed per container (launcher knob IDXSP, default 1);
# read it once at import instead of per layer call. TP world size / rank are
# cached on first use (distributed groups exist by the first forward).
_INDEXER_SP_ENABLED = os.environ.get("VLLM_DSV4_INDEXER_SP") == "1"
_indexer_sp_tp_rank: tuple[int, int] | None = None

# Incident kill-switches (2026-08-09 warmup failure: C128A layers saw
# c128a_prefill_topk_indices=None). The behavior-bearing optimizations land
# default-OFF until each is individually cleared in prod; enable via the
# launcher knobs TRIMIDX / SPFAST (and C4AREUSE in flashinfer_sparse.py).
_TRIM_SKIP_INDEXER_KV = os.environ.get("DENEB_TRIM_SKIP_INDEXER_KV") == "1"
_SP_SINGLE_SPAN = os.environ.get("DENEB_SP_SINGLE_SPAN") == "1"


def _indexer_sp_owned_ranges(k_cache_prefix: str):
    """Batch row ranges this rank must compute indexer-q for, under
    VLLM_DSV4_INDEXER_SP. None => compute all rows (feature off, warmup,
    decode-only batch, or metadata unavailable). Partition math MUST match
    the sparse_attn_indexer prefill-loop slicing exactly.
    """
    if not _INDEXER_SP_ENABLED:
        return None
    global _indexer_sp_tp_rank
    if _indexer_sp_tp_rank is None:
        try:
            _indexer_sp_tp_rank = (
                get_tensor_model_parallel_world_size(),
                get_tensor_model_parallel_rank(),
            )
        except Exception:
            return None
    tp, rank = _indexer_sp_tp_rank
    if tp <= 1:
        return None

    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    if _resolve_layer_name is not None:
        try:
            key = _resolve_layer_name(k_cache_prefix)
        except Exception:
            key = k_cache_prefix
    else:
        key = k_cache_prefix
    md = attn_metadata.get(key)
    if md is None:
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
    if not ranges:
        return ranges
    # Ranges ascend (decode rows first, then chunks in batch order); merging
    # adjacent spans preserves the owned-row SET exactly and lets the caller
    # take the zero-copy single-span path (decode rows adjoin rank 0's first
    # shard, and a pure-prefill single chunk is one span on every rank).
    merged = [ranges[0]]
    for a, b in ranges[1:]:
        if a == merged[-1][1]:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


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
        skip_topk: bool = False,
        eager_scratch_pool=None,
    ):
        super().__init__()
        self.eager_scratch_pool = eager_scratch_pool  # deneb fork: PR #49236
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        # FP4 indexer cache requires the SM10x paged-MQA kernels (B200/GB200);
        # GB10 is SM121, so this stack always runs the FP8 indexer cache.
        assert not vllm_config.attention_config.use_fp4_indexer_cache, (
            "use_fp4_indexer_cache is not supported on GB10 (SM121)."
        )
        self.use_fp4_kv = False
        logger.info_once("Using FP8 indexer cache for Lightning Indexer.")

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

        # No context parallelism on this stack (CP world size 1).
        self.max_model_len = cdiv(
            vllm_config.model_config.max_model_len,
            self.compress_ratio,
        )
        self.prefix = prefix

        self.max_total_seq_len = cdiv(
            get_max_prefill_buffer_size(vllm_config),
            self.compress_ratio,
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indexer cache uses the same per-row layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
            # skip-topk layers never run this indexer; don't spend KV pages
            # on a cache that is never written or read. Gated default-OFF:
            # trimming specs changes KV-group composition, the prime suspect
            # for the image FlashMLA builder leaving c128a_* metadata unset.
            spec_enabled=not (skip_topk and _TRIM_SKIP_INDEXER_KV),
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

        # None falls back to sequential in maybe_execute_in_parallel.
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
    ) -> torch.Tensor:
        compressor = self.compressor

        def wq_b_and_q_quant():
            _ranges = _indexer_sp_owned_ranges(self.k_cache.prefix)
            if _ranges is not None:
                # Sequence-parallel: compute q/rope/quant only for the rows
                # this rank's sharded indexer will read; place them into
                # full-size buffers (unowned rows stay uninitialized and are
                # never read by the sliced indexer paths).
                #
                # Single contiguous span — the dominant shape (pure-prefill
                # single chunk on every rank; decode rows fused with rank 0's
                # first shard) — uses zero-copy dim-0 slices in and one
                # contiguous copy out (kill-switch gated); multi-span uses
                # index_select / index_copy_ over a concatenated index.
                _n = qr.shape[0]
                if _SP_SINGLE_SPAN and len(_ranges) == 1:
                    _a, _b = _ranges[0]
                    _idx = None

                    def _take(t: torch.Tensor) -> torch.Tensor:
                        return t[_a:_b]
                else:
                    _idx = torch.cat(
                        [
                            torch.arange(a, b, device=qr.device, dtype=torch.long)
                            for a, b in _ranges
                        ]
                    )

                    def _take(t: torch.Tensor) -> torch.Tensor:
                        return t.index_select(0, _idx)

                _q_c, _ = self.wq_b(_take(qr))
                _q_c = _q_c.view(-1, self.n_head, self.head_dim)
                _qq, _ww = fused_indexer_q_rope_quant(
                    _take(positions),
                    _q_c,
                    rotary_emb.cos_sin_cache,
                    _take(indexer_weights),
                    self.softmax_scale,
                    self.n_head**-0.5,
                    use_fp4=self.use_fp4_kv,
                )

                # deneb fork: PR #49236 — reuse the pool's full-size buffers
                # as the scatter targets (index_q_fp8 first, weights second).
                _pool_targets = (
                    list(self.eager_scratch_pool.indexer_q_outputs(_n))
                    if self.eager_scratch_pool is not None and not self.use_fp4_kv
                    else None
                )

                def _emplace(part):
                    if isinstance(part, (tuple, list)):
                        return type(part)(_emplace(x) for x in part)
                    if _pool_targets:
                        full = _pool_targets.pop(0)
                        assert full.shape == (_n,) + tuple(part.shape[1:])
                    else:
                        full = part.new_empty((_n,) + tuple(part.shape[1:]))
                    if part.element_size() == 1:
                        # index_copy_ lacks fp8 support; bit-copy via uint8
                        # (kept for the slice path too — proven pattern).
                        part8 = part.contiguous().view(torch.uint8)
                        if _idx is None:
                            full[_a:_b].view(torch.uint8).copy_(part8)
                        else:
                            full.view(torch.uint8).index_copy_(0, _idx, part8)
                    elif _idx is None:
                        full[_a:_b].copy_(part)
                    else:
                        full.index_copy_(0, _idx, part)
                    return full

                return _emplace(_qq), _emplace(_ww)
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            q = q.view(-1, self.n_head, self.head_dim)
            # deneb fork: PR #49236 — write into the shared eager-scratch
            # buffers instead of per-call allocations (None keeps old path).
            _bufs = (
                self.eager_scratch_pool.indexer_q_outputs(qr.shape[0])
                if self.eager_scratch_pool is not None and not self.use_fp4_kv
                else None
            )
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
                output_buffers=_bufs,
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
        return self.indexer_op(hidden_states, q_quant, k, weights)
