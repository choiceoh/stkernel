# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 FlashInfer sparse MLA backend.

deneb fork (2026-08-08, slimmed for the TP=4 4x GB10 stack): only the SM120
(consumer Blackwell, GB10 = SM121) path is kept. The SM100 (B200-class)
attention class is reduced to an import-safe stub, and with it the plain-row
bf16 / per-tensor-fp8 KV handling; GB10 always runs the fp8_ds_mla layout.
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
from vllm.v1.attention.backend import MultipleOf

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}


def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
    workspace = _flashinfer_dsv4_workspace_by_device.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _flashinfer_dsv4_workspace_by_device[device] = workspace
    return workspace


class DeepseekV4FlashInferMLASparseBackend(DeepseekV4FlashMLABackend):
    """FlashInfer backend using the DSv4 sparse metadata/cache layout.

    Inheriting from the FlashMLA V4 backend reuses its
    ``DeepseekV4FlashMLAMetadata`` builder. SM12x only on this stack.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV4"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # TP4 GB10 overlay: SM10x support removed with the SM100 class.
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if device_capability.major != 12:
            return (
                "FLASHINFER_MLA_SPARSE_DSV4 requires SM12x on this overlay "
                "(SM10x support removed)"
            )
        if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla"):
            return "kv_cache_dtype not supported"
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            return (
                "FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer's "
                "sparse MLA decode API"
            )
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SM12x always uses the FlashMLA fp8_ds_mla paged layout.
        return DeepseekV4FlashMLABackend.get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str,
        )


class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):
    """Import-safe stub of the SM100 (B200-class) attention layer.

    The full implementation (TRTLLM-gen decode/prefill on the plain-row
    per-tensor-FP8/bf16 KV layout) was removed from this overlay: the TP=4
    stack is 4x GB10 (SM121), where the platform selector always picks
    ``DeepseekV4FlashInferSM120Attention``. The name is kept so any
    import-time reference in the base image keeps resolving. Restore from git
    history (commit e95f82f) if SM10x support is ever needed again.
    """

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "DeepseekV4FlashInferMLAAttention (SM100) was removed from the "
            "TP4 GB10 overlay; only DeepseekV4FlashInferSM120Attention is "
            "supported on this stack."
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        raise RuntimeError("SM100 path removed from the TP4 GB10 overlay.")

    def forward_mqa(self, q, kv, positions, output) -> None:
        raise RuntimeError("SM100 path removed from the TP4 GB10 overlay.")

    def _o_proj(self, o, positions) -> torch.Tensor:
        raise RuntimeError("SM100 path removed from the TP4 GB10 overlay.")


class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):
    """DeepSeek V4 sparse MLA attention through FlashInfer's SM120 kernels."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = True

    @staticmethod
    def _get_workspace(device: torch.device) -> torch.Tensor:
        return _get_flashinfer_dsv4_workspace(device)

    @staticmethod
    def _as_sparse_cache(kv_cache: torch.Tensor) -> torch.Tensor:
        # fp8_ds_mla caches are uint8 by construction on this stack; only the
        # layout dim may differ (the per-tensor-fp8 view belonged to the
        # removed SM100 path).
        if kv_cache.dim() == 4:
            return kv_cache
        return kv_cache.unsqueeze(-2)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 FlashInfer MLA Sparse does not support {num_heads} heads "
            "(SM120 kernel requires h_q in {16, 32, 64, 128})."
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
        # fp8_ds_mla layout stores uint8 blocks; the per-tensor FP8 scale
        # buffers of the removed SM100 path do not apply here.
        assert self.kv_cache_torch_dtype == torch.uint8, (
            "TP4 GB10 overlay: SM120 attention requires the fp8_ds_mla "
            f"(uint8) KV layout, got {self.kv_cache_torch_dtype}"
        )
        # Identity-guarded caches for the _as_sparse_cache dim-fix views. The
        # KV tensors are bound once at startup, so rebuilding the view on
        # every per-layer forward call is wasted python; the `is` guard keeps
        # this correct even if a cache tensor were ever rebound.
        self._swa_view_src: torch.Tensor | None = None
        self._swa_view: torch.Tensor | None = None
        self._ckv_view_src: torch.Tensor | None = None
        self._ckv_view: torch.Tensor | None = None

    def _reserve_empty_forward_workspace(self) -> None:
        self._get_workspace(
            torch.device("cuda", torch.accelerator.current_device_index())
        )

    def _swa_sparse_cache(self) -> torch.Tensor:
        src = self.swa_cache_layer.kv_cache
        if self._swa_view_src is not src:
            self._swa_view = self._as_sparse_cache(src)
            self._swa_view_src = src
        return self._swa_view

    def _compressed_sparse_cache(self, kv_cache: torch.Tensor) -> torch.Tensor:
        if self._ckv_view_src is not kv_cache:
            self._ckv_view = self._as_sparse_cache(kv_cache)
            self._ckv_view_src = kv_cache
        return self._ckv_view

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        # deneb fork: port of upstream PR #51202. Gate on token counts, not
        # request counts -- a request can be classified prefill/decode yet
        # contribute zero tokens to this split, and the empty call then trips
        # the same zero-element reshape that PR #49059 guards one level down.
        if swa_metadata.num_prefill_tokens > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decode_tokens > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Output may be padded to backend-supported head counts.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        assert output.dtype == q.dtype, (
            f"output dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_empty_forward_workspace()
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers don't allocate their own compressed KV cache.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _prepare_query(self, q: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # fp8_ds_mla layout: q arrives bf16 (and already head-padded by the
        # fused qnorm/rope/insert kernel on the normal path).
        assert q.dtype == torch.bfloat16
        padded_heads = output.shape[1]
        if q.shape[1] < padded_heads:
            padded_query = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_query[:, : q.shape[1], :] = q
            q = padded_query
        return q.contiguous()

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                # IndexCache (#51209) extension: a skip-topk (S) layer reads
                # the exact buffer contents the previous F layer's indexer
                # left, so its globalization (local top-k -> paged slot ids)
                # is identical as well — skip relaunching the kernels. F
                # layers (skip_topk=False) always recompute and refresh the
                # entry, and the first C4A layer is always F, so S layers can
                # never observe a stale step. Under full-cudagraph capture
                # the reuse simply bakes the F layer's output tensor into the
                # S layers' kernel arguments.
                cache = swa_metadata.flashinfer_sparse_index_cache
                cached = cache.get("c4a_decode_global") if self.skip_topk else None
                if cached is None:
                    block_size = attn_metadata.block_size // self.compress_ratio
                    global_indices, extra_sparse_lengths = (
                        compute_global_topk_indices_and_lens(
                            self.topk_indices_buffer[:num_decode_tokens],
                            swa_metadata.token_to_req_indices,
                            attn_metadata.block_table[:num_decodes],
                            block_size,
                            is_valid,
                        )
                    )
                    cache["c4a_decode_global"] = (
                        global_indices,
                        extra_sparse_lengths,
                    )
                else:
                    global_indices, extra_sparse_lengths = cached
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)
        swa_cache = self._swa_sparse_cache()
        extra_cache = (
            self._compressed_sparse_cache(kv_cache) if kv_cache is not None else None
        )
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                f"[dspark-debug] layer={getattr(self, 'prefix', '?')} "
                f"compress_ratio={self.compress_ratio} swa_only={swa_only} "
                f"num_decodes={num_decodes} num_decode_tokens={num_decode_tokens} "
                f"attn_metadata={type(attn_metadata).__name__ if attn_metadata is not None else None} "
                f"c128a_g={getattr(attn_metadata, 'c128a_global_decode_topk_indices', 'MISSING') is not None} "
                f"c128a_l={getattr(attn_metadata, 'c128a_decode_topk_lens', 'MISSING') is not None} "
                f"topk_buf={self.topk_indices_buffer is not None} "
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache,
            workspace_buffer=self._get_workspace(q.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif self.compress_ratio == 4:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "C4A prefill requires top-k indices from the indexer."
                )
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            if attn_metadata is None:
                raise RuntimeError("C128A prefill metadata is missing.")
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if local_topk_indices is not None:
            if attn_metadata is None:
                raise RuntimeError("C4A prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError("C4A prefill request mapping is missing.")
            if swa_metadata.is_valid_token is None:
                raise RuntimeError("C4A prefill validity metadata is missing.")
            # Same IndexCache-driven reuse as decode. Only C4A has the F/S
            # structure that guarantees a fresh recompute each step, so the
            # C128A globalization stays per-layer.
            is_c4a = self.compress_ratio == 4
            cache = swa_metadata.flashinfer_sparse_index_cache
            cached = (
                cache.get("c4a_prefill_global")
                if is_c4a and self.skip_topk
                else None
            )
            if cached is None:
                prefill_token_slice = slice(
                    num_decode_tokens, num_decode_tokens + num_prefill_tokens
                )
                block_size = attn_metadata.block_size // self.compress_ratio
                extra_sparse_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        local_topk_indices,
                        swa_metadata.token_to_req_indices[prefill_token_slice],
                        attn_metadata.block_table,
                        block_size,
                        swa_metadata.is_valid_token[prefill_token_slice],
                    )
                )
                if is_c4a:
                    cache["c4a_prefill_global"] = (
                        extra_sparse_indices,
                        extra_sparse_lengths,
                    )
            else:
                extra_sparse_indices, extra_sparse_lengths = cached

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        q = self._prepare_query(q, output)
        swa_kv_paged = self._swa_sparse_cache()
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = self._compressed_sparse_cache(compressed_k_cache)

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            # deneb fork: port of upstream PR #49059. FULL_AND_PIECEWISE cudagraph
            # padding can produce a trailing SM120 sparse-MLA prefill chunk with
            # query_start == query_end; FlashInfer then raises "cannot reshape
            # tensor of 0 elements into shape [0, -1]" and kills EngineCore.
            if query_start == query_end:
                continue

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )

            q_chunk = q[query_start:query_end]
            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:
                raise RuntimeError(
                    "Compressed sparse MLA prefill requires compressed sparse indices."
                )
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
                workspace_buffer=self._get_workspace(q.device),
                sparse_indices=swa_indices_chunk,
                compressed_kv_cache=extra_kv_paged,
                out=output[query_start:query_end],
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens_chunk,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_topk_lens=extra_sparse_lengths_chunk,
            )
