# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV4 sparse SWA cache + metadata builder.

deneb fork (2026-08-08, slimmed for the TP=4 4x GB10 stack): paths that are
structurally unreachable on this deployment are removed —
- ROCm / XPU builder dispatch (all four nodes are CUDA GB10),
- DCP (decode context parallel) index kernels and branches (the launcher
  exposes no CP flags),
- the FlashMLA per-layer-type tile-scheduler plan: SM120 drives DSV4 MLA
  through b12x, which does not use the FlashMLA tile scheduler (and
  _flashmla_C is not built for sm_121a), so the skip branch was always taken
  on this stack. The tile_sched_* metadata fields stay (always None) for
  consumers,
- the upstream-DSpark parallel-drafting threshold doubling (fork impl only).
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

if TYPE_CHECKING:
    from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta

# DeepseekV4 decode layer types, keyed by compress_ratio. Kept (with
# _layer_type_for) for import compatibility even though the FlashMLA
# tile-scheduler that consumed them is not used on SM120.
_LAYER_TYPE_SWAONLY = "swaonly"
_LAYER_TYPE_C4A = "c4a"
_LAYER_TYPE_C128A = "c128a"


def _layer_type_for(compress_ratio: int) -> str:
    if compress_ratio <= 1:
        return _LAYER_TYPE_SWAONLY
    if compress_ratio == 4:
        return _LAYER_TYPE_C4A
    if compress_ratio == 128:
        return _LAYER_TYPE_C128A
    raise ValueError(
        f"Unsupported DeepseekV4 compress_ratio={compress_ratio}; "
        "expected 1, 4, or 128."
    )


class DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Block size is constrained by tensor sharing between SWA and C4A KV blocks.
        # Since both block types share the same physical tensor, they must use the
        # same page size. The C4A KV block shape [256//4, head_dim] = [64, head_dim]
        # determines the SWA block size of 64 tokens per block.
        # TODO(yifan): make SWA block size automatically determined and configurable.
        self.block_size = 64
        # uint8: fp8_ds_mla UE8M0 paged layout (the only layout on this stack).
        assert self.dtype == torch.uint8, (
            "TP4 GB10 overlay: the SWA cache runs the fp8_ds_mla (uint8) "
            f"layout only, got {self.dtype}"
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla's UE8M0 paged layout needs 576B alignment.
        assert self.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=576,
            model_version="deepseek_v4",
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekSparseSWABackend


class DeepseekSparseSWABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 256

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @staticmethod
    def get_builder_cls() -> type["DeepseekSparseSWAMetadataBuilder"]:
        return DeepseekSparseSWAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class DeepseekSparseSWAMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    seq_lens: torch.Tensor | None = None  # [num_seqs]
    query_start_loc: torch.Tensor | None = None  # [num_seqs + 1]
    query_start_loc_cpu: torch.Tensor | None = None  # [num_seqs + 1]

    is_valid_token: torch.Tensor | None = None  # [num_tokens]
    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    decode_swa_indices: torch.Tensor | None = None  # [num_decode_tokens, width]
    decode_swa_lens: torch.Tensor | None = None  # [num_decode_tokens]
    # deneb fork: port of upstream PR #51042. The non-causal (DSpark) buffer is
    # allocated noncausal_index_width wide, not window_size wide, so the row
    # stride has to travel with the metadata -- the consumer cannot assume
    # window_size or it reshapes every decode row past 0 at the wrong offset.
    decode_swa_width: int = 0
    # Paged-coordinate per-token SWA slot ids / lengths for prefill rows.
    prefill_swa_indices: torch.Tensor | None = None
    prefill_swa_lens: torch.Tensor | None = None  # [num_prefill_tokens]

    # Number of decode/prefill requests/tokens (batch is reordered: decodes first)
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # Pre-computed prefill metadata shared across all DeepseekV4 attention layers.
    prefill_seq_lens: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_window_size: int = 0
    prefill_max_model_len: int = 0
    prefill_max_num_batched_tokens: int = 0

    # Per-layer-type FlashMLA tile-scheduler metadata. Always None on this
    # stack: SM120 drives DSV4 MLA through b12x, which does not use the
    # FlashMLA tile scheduler (_flashmla_C is not built for sm_121a). The
    # fields are kept so consumers that probe them keep seeing the same None
    # they always saw here.
    tile_sched_swaonly: "FlashMLASchedMeta | None" = None
    tile_sched_c4a: "FlashMLASchedMeta | None" = None
    tile_sched_c128a: "FlashMLASchedMeta | None" = None
    # Cross-layer per-step cache. One metadata instance is shared by every
    # layer's SWA prefix, so the SM120 attention uses this to reuse the C4A
    # top-k globalization across skip-topk (IndexCache) layers
    # ("c4a_decode_global" / "c4a_prefill_global" keys); the image's SM100
    # path stores its mixed sparse indices here under other keys.
    flashinfer_sparse_index_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    def get_prefill_chunk_plan(
        self, compress_ratio: int, prefill_chunk_size: int
    ) -> list[tuple[int, int, int, int]]:
        if self.num_prefills == 0:
            return []

        assert self.prefill_seq_lens_cpu is not None
        assert self.prefill_query_lens_cpu is not None

        # query_len <= max_num_batched_tokens and
        # gather_len = query_len + min(prefix_len, window_size - 1), so the
        # worst-case gathered width is bounded by
        # max_num_batched_tokens + window_size - 1. The compressed prefix pool
        # is bounded by ceil(max_model_len / compress_ratio).
        max_workspace_area = prefill_chunk_size * (
            (
                0
                if compress_ratio <= 1
                else cdiv(self.prefill_max_model_len, compress_ratio)
            )
            + self.prefill_window_size
            + self.prefill_max_num_batched_tokens
        )
        prefix_lens_cpu = self.prefill_seq_lens_cpu - self.prefill_query_lens_cpu
        gather_lens_cpu = self.prefill_query_lens_cpu + torch.clamp(
            prefix_lens_cpu, min=0, max=self.prefill_window_size - 1
        )
        compressed_lens_cpu = (
            torch.zeros_like(self.prefill_seq_lens_cpu)
            if compress_ratio <= 1
            else torch.div(
                self.prefill_seq_lens_cpu,
                compress_ratio,
                rounding_mode="floor",
            )
        )

        chunk_plan: list[tuple[int, int, int, int]] = []
        chunk_start = 0
        while chunk_start < self.num_prefills:
            chunk_max_compressed = int(compressed_lens_cpu[chunk_start].item())
            chunk_max_gather = int(gather_lens_cpu[chunk_start].item())
            chunk_end = chunk_start + 1

            while chunk_end < self.num_prefills:
                candidate_max_compressed = max(
                    chunk_max_compressed,
                    int(compressed_lens_cpu[chunk_end].item()),
                )
                candidate_max_gather = max(
                    chunk_max_gather,
                    int(gather_lens_cpu[chunk_end].item()),
                )
                candidate_width = candidate_max_compressed + candidate_max_gather
                candidate_area = (chunk_end - chunk_start + 1) * candidate_width
                if candidate_area > max_workspace_area:
                    break
                chunk_max_compressed = candidate_max_compressed
                chunk_max_gather = candidate_max_gather
                chunk_end += 1

            chunk_plan.append(
                (
                    chunk_start,
                    chunk_end,
                    chunk_max_compressed,
                    chunk_max_compressed + chunk_max_gather,
                )
            )
            chunk_start = chunk_end

        return chunk_plan


class DeepseekSparseSWAMetadataBuilder(AttentionMetadataBuilder):
    """Builds metadata for DeepseekV4 SWA cache.

    Similar to the indexer, this handles mixed batches by:
    1. Using split_decodes_and_prefills() to determine the boundary
    2. Building separate metadata for decode and prefill portions

    Supports:
    - Mixed decode/prefill batches
    - MTP (Multi-Token Prediction) where decode has query_len > 1
    - Chunked prefill (aligns with the indexer's chunking)
    """

    # Base threshold: query_len <= 1 is decode
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_exact_metadata_reuse: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size
        self.max_model_len = self.vllm_config.model_config.max_model_len
        self.max_num_batched_tokens = (
            self.vllm_config.scheduler_config.max_num_batched_tokens
        )

        # TP4 GB10 overlay: context parallelism support removed. Fail fast if
        # the deployment config ever drifts.
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.decode_context_parallel_size == 1
            and parallel_config.prefill_context_parallel_size == 1
        ), "TP4 GB10 overlay: DCP/PCP support was removed from this builder."

        # Handle MTP: adjust decode_threshold like the indexer does
        spec_config = self.vllm_config.speculative_config
        self.num_speculative_tokens = (
            spec_config.num_speculative_tokens if spec_config else 0
        )
        # Decode can have query_len up to 1 + num_speculative_tokens.
        # This MUST match the flashmla_sparse / indexer threshold so that
        # all backends agree on the decode/prefill split. (The upstream-DSpark
        # parallel-drafting 2x does not apply: this stack runs the fork impl.)
        self.decode_threshold = (
            self.reorder_batch_threshold + self.num_speculative_tokens
        )

        hf_config = self.vllm_config.model_config.hf_config
        assert hasattr(hf_config, "sliding_window")
        self.window_size = hf_config.sliding_window

        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_indices = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.req_ids_arange = torch.arange(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_indices = torch.zeros(
            max_tokens,
            1,
            self.window_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_lens = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # Allocated unconditionally — consumer picks paged-direct vs dequant
        # at call time.
        self.prefill_swa_indices = torch.zeros(
            max_tokens,
            1,
            self.window_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.prefill_swa_lens = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.is_valid_token = torch.zeros(
            max_tokens,
            dtype=torch.bool,
            device=self.device,
        )

        # DSpark draft: the block is non-causal (every query attends to the
        # trailing window of context PLUS all query tokens, including future ones),
        # so its per-token index list is wider than `window_size`. Upstream pads
        # to a multiple of 128, but our flashinfer pin only instantiates the
        # sparse_mla_sm120 decode-dsv4 kernel for topk in {128, 512, 1024}
        # (_DECODE_DSV4_DISPATCH); any other width (e.g. 256) falls through to
        # the >64-token paged kernel and ICHECK-fails on a decode-sized batch.
        # Round up to the next instantiated width — the kernel masks by the
        # per-token topk_length (decode_swa_lens), so the padding is cheap.
        self.is_dspark = spec_config is not None and spec_config.use_dspark()
        if self.is_dspark:
            needed = self.window_size + self.num_speculative_tokens
            for cand in (128, 512, 1024):
                if needed <= cand:
                    self.noncausal_index_width = cand
                    break
            else:
                self.noncausal_index_width = cdiv(needed, 128) * 128
        else:
            self.noncausal_index_width = 0
        self.decode_swa_indices_noncausal: torch.Tensor | None = None
        self._max_tokens = max_tokens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekSparseSWAMetadata:
        """Build SWA metadata for mixed decode/prefill batches.

        The batch is assumed to be reordered with decodes first (by vLLM scheduler).
        We use split_decodes_and_prefills() to find the boundary, then build
        separate window_topk_idxs for each portion.

        For prefill, we use chunked prefill to align with the indexer's chunking.
        """
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # Split into decode and prefill portions using configurable threshold
        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            common_attn_metadata.split_decodes_and_prefills(
                decode_threshold=self.decode_threshold
            )
        )

        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility.
        if num_prefill_tokens == 0 and num_decode_tokens == num_reqs:
            token_to_req_indices = self.req_ids_arange[:num_decode_tokens]
        elif common_attn_metadata.batch_topology is not None:
            x = torch.from_numpy(
                common_attn_metadata.batch_topology.req_id_per_token_np
            ).pin_memory()
            token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
            token_to_req_indices.copy_(x, non_blocking=True)
        else:
            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
            token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
            token_to_req_indices.copy_(x, non_blocking=True)

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        # In-place mask write: torch.ge(out=) skips the per-step bool
        # temporary (slot_mapping >= 0) plus its copy kernel.
        torch.ge(slot_mapping, 0, out=is_valid_token)

        non_causal = not common_attn_metadata.causal
        decode_swa_width = self.noncausal_index_width if non_causal else self.window_size
        decode_swa_indices = self.decode_swa_indices
        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            if non_causal:
                assert self.is_dspark, (
                    "Non-causal DeepseekV4 SWA is only supported for the DSpark "
                    "speculation mode, but causal=False was set without DSpark."
                )
                if self.decode_swa_indices_noncausal is None:
                    self.decode_swa_indices_noncausal = torch.zeros(
                        self._max_tokens,
                        1,
                        self.noncausal_index_width,
                        dtype=torch.int32,
                        device=self.device,
                    )
                decode_swa_indices = self.decode_swa_indices_noncausal
                _compute_dspark_noncausal_swa_indices_kernel[(num_decode_tokens,)](
                    decode_swa_indices,
                    decode_swa_indices.stride(0),
                    self.decode_swa_lens,
                    self.window_size,
                    self.noncausal_index_width,
                    query_start_loc,
                    seq_lens,
                    token_to_req_indices,
                    is_valid_token,
                    block_table,
                    block_table.stride(0),
                    self.block_size,
                    token_offset=0,
                    TRITON_BLOCK_SIZE=1024,
                )
            else:
                _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
                    self.decode_swa_indices,
                    self.decode_swa_indices.stride(0),
                    self.decode_swa_lens,
                    self.window_size,
                    query_start_loc,
                    seq_lens,
                    token_to_req_indices,
                    is_valid_token,
                    block_table,
                    block_table.stride(0),
                    self.block_size,
                    token_offset=0,
                    TRITON_BLOCK_SIZE=1024,
                )

        # Prefill SWA indices live in paged coordinates. `token_offset` lets
        # the kernel read is_valid_token / token_to_req_indices at absolute
        # prefill positions while writing output starting at index 0.
        if num_prefill_tokens > 0:
            prefill_swa_indices = self.prefill_swa_indices[:num_prefill_tokens]
            prefill_swa_lens = self.prefill_swa_lens[:num_prefill_tokens]
            _compute_swa_indices_and_lens_kernel[(num_prefill_tokens,)](
                prefill_swa_indices,
                prefill_swa_indices.stride(0),
                prefill_swa_lens,
                self.window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table.stride(0),
                self.block_size,
                token_offset=num_decode_tokens,
                TRITON_BLOCK_SIZE=1024,
            )

        # Pre-compute DeepseekV4 prefill metadata shared across all attention layers.
        deepseek_v4_fields = self._build_deepseek_v4_metadata(
            num_decodes,
            num_prefills,
            seq_lens,
            seq_lens_cpu,
            query_start_loc,
            query_start_loc_cpu,
        )

        # tile_sched_* stay at their None defaults: the FlashMLA tile scheduler
        # is not used on SM120 (b12x path).
        return DeepseekSparseSWAMetadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            decode_swa_indices=decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            decode_swa_width=decode_swa_width,
            prefill_swa_indices=(
                self.prefill_swa_indices[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            prefill_swa_lens=(
                self.prefill_swa_lens[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            **deepseek_v4_fields,  # type: ignore[arg-type]
        )

    def _build_deepseek_v4_metadata(
        self,
        num_decodes: int,
        num_prefills: int,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
    ) -> dict[str, torch.Tensor | int | None]:
        """Pre-compute DeepseekV4 prefill metadata during the metadata build phase.

        Returns a dict of keyword arguments to pass to the
        DeepseekSparseSWAMetadata constructor.

        Note: C128A sparse metadata is computed by the FlashMLASparse builder
        (which owns the C128A block_table), not here.
        """
        result: dict[str, torch.Tensor | int | None] = {}

        # --- Prefill query metadata (single Triton kernel + CPU slicing) ---
        if num_prefills > 0:
            assert seq_lens_cpu is not None
            pfx_gather_lens = torch.empty(
                num_prefills, dtype=torch.int32, device=seq_lens.device
            )
            _compute_prefill_metadata_kernel[(1,)](
                pfx_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                self.window_size,
                BLOCK_SIZE=triton.next_power_of_2(num_prefills),
            )

            result["prefill_seq_lens"] = seq_lens[num_decodes:]
            result["prefill_seq_lens_cpu"] = seq_lens_cpu[num_decodes:]
            result["prefill_gather_lens"] = pfx_gather_lens
            result["prefill_query_lens_cpu"] = (
                query_start_loc_cpu[num_decodes + 1 : num_decodes + num_prefills + 1]
                - query_start_loc_cpu[num_decodes : num_decodes + num_prefills]
            ).to(dtype=torch.int32)
            result["prefill_window_size"] = self.window_size
            result["prefill_max_model_len"] = self.max_model_len
            result["prefill_max_num_batched_tokens"] = self.max_num_batched_tokens

        return result


@triton.jit
def _compute_prefill_metadata_kernel(
    # Outputs
    prefill_gather_lens_ptr,
    # Inputs
    seq_lens_ptr,
    query_start_loc_ptr,
    num_prefills,
    num_decodes,
    window_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute prefill gather_lens in a single pass."""
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < num_prefills
    # SM12x + Triton 3.6 raises IMA on out-of-bounds address arithmetic for
    # masked-off lanes even though the load mask gates the actual read, so
    # clamp the offset. Caller guarantees num_prefills > 0.
    safe_offset = tl.minimum(offset, num_prefills - 1)

    seq_len = tl.load(seq_lens_ptr + num_decodes + safe_offset, mask=mask)
    qsl_start = tl.load(query_start_loc_ptr + num_decodes + safe_offset, mask=mask)
    qsl_end = tl.load(query_start_loc_ptr + num_decodes + safe_offset + 1, mask=mask)

    query_len = qsl_end - qsl_start
    prefix_len = seq_len - query_len
    gather_len = query_len + tl.minimum(prefix_len, window_size - 1)

    tl.store(prefill_gather_lens_ptr + offset, gather_len, mask=mask)


def warmup_deepseek_v4_prefill_metadata_kernel(
    device: torch.device,
    *,
    max_num_prefills: int,
    window_size: int,
) -> int:
    """Precompile the Triton prefill metadata kernel used by DSv4 SWA.

    Mixed sparse-MLA warmup can miss the pure-prefill first-request shape on a
    fresh Triton cache. Warm both pure prefill and mixed decode+prefill cases
    across the small power-of-two BLOCK_SIZE specializations used by the
    metadata builder.
    """
    if not torch.cuda.is_available():
        return 0

    max_num_prefills = max(1, int(max_num_prefills))
    cases: set[int] = {1, max_num_prefills}
    block_size = 1
    while block_size < max_num_prefills:
        block_size *= 2
        cases.add(min(block_size, max_num_prefills))

    launches = 0
    for num_prefills in sorted(cases):
        for num_decodes in (0, 1):
            query_lens = torch.tensor(
                [1] * num_decodes + [16] * num_prefills,
                dtype=torch.int32,
                device=device,
            )
            seq_lens = query_lens.clone()
            query_start_loc = torch.empty(
                query_lens.numel() + 1,
                dtype=torch.int32,
                device=device,
            )
            query_start_loc[0].zero_()
            torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])
            prefill_gather_lens = torch.empty(
                num_prefills,
                dtype=torch.int32,
                device=device,
            )
            _compute_prefill_metadata_kernel[(1,)](
                prefill_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                window_size,
                BLOCK_SIZE=triton.next_power_of_2(num_prefills),
            )
            launches += 1
    torch.cuda.synchronize(device)
    return launches


@triton.jit(do_not_specialize=["token_offset"])
def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    token_offset,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, window_size, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < window_size,
        )


# TODO(ben): unify this kernel to reduce duplication
@triton.jit(do_not_specialize=["token_offset"])
def _compute_dspark_noncausal_swa_indices_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    index_width,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    token_offset,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Non-causal per-token indices for the DSpark draft block.

    Here, we populate the topk indices with the trailing window of context tokens,
    plus all query tokens (including future ones).
    """
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    # Block-anchored window (shared by every token in the block) + full block.
    start_pos = tl.maximum(prefix_len - window_size, 0)
    end_pos = seq_len

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < index_width,
        )
