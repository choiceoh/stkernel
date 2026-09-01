# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 dense-prefill cache-only indexer path.

Fresh short prefills admitted by the SM121 dense-MLA backend do not consume
the sparse indexer's query scores.  The kpool op already stops before top-k,
but the model used to reach that stop only after computing q_b, FWHT+FP8
quantization, and fp32 head weights.  This module installs an
exact-gated ``Indexer.forward`` path that builds only the K and gate data
needed by the index/tail caches, then calls the same kpool op.  K and gate
share one load-time fused weight so the path needs only one projection GEMM.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer_kpool import (
    _glm53_dense_mha_scoring_unused,
)
from vllm.platforms import current_platform

from .attention import Indexer, _fused_indexer_k_norm

_GLM53_SM121_MLA_PREFILL_ENV = "VLLM_GLM53_SM121_MLA_PREFILL"
_GLM53_PREFILL_KG_WEIGHT = "_glm53_prefill_kg_weight"


def _glm53_cache_only_indexer_contract(
    *,
    rope_dim: int,
    head_dim: int,
    quant_block_size: int,
    scale_fmt: str | None,
    index_kpool: int,
    topk_tokens: int,
    n_head: int,
    op_type: str,
    use_fp4_cache: bool,
    hidden_dtype: torch.dtype,
    k_weight_dtype: torch.dtype,
    k_weight_shape: tuple[int, ...],
    hidden_size: int,
    gate_dtype: torch.dtype,
    gate_shape: tuple[int, ...],
    ape_shape: tuple[int, ...],
    ape_dtype: torch.dtype,
    fused_weight_dtype: torch.dtype,
    fused_weight_shape: tuple[int, ...],
    fused_weight_contiguous: bool,
) -> bool:
    """Fail closed to the original indexer on any production-contract drift."""
    return (
        rope_dim == 0
        and head_dim == 128
        and quant_block_size == 128
        and scale_fmt == "ue8m0"
        and index_kpool == 4
        and topk_tokens == 2048
        and n_head == 32
        and op_type == "SparseAttnIndexerKpool"
        and use_fp4_cache is False
        and hidden_dtype == torch.bfloat16
        and k_weight_dtype == torch.bfloat16
        and k_weight_shape == (head_dim + n_head, hidden_size)
        and gate_dtype == torch.bfloat16
        and gate_shape == (head_dim, hidden_size)
        and ape_shape == (index_kpool, head_dim)
        and ape_dtype == torch.float32
        and fused_weight_dtype == torch.bfloat16
        and fused_weight_shape == (2 * head_dim, hidden_size)
        and fused_weight_contiguous
    )


def prepare_glm53_prefill_fastpath(model: torch.nn.Module) -> int:
    """Build persistent K+gate weights after checkpoint loading has finished.

    The source parameters must remain intact for decode/MQA fallback.  The
    additional non-persistent buffer is only 2 * head_dim rows per sparse
    layer and moves with the owning module without entering its state dict.
    Any contract or allocation failure leaves that layer on the original
    indexer path.
    """
    if (
        os.environ.get(_GLM53_SM121_MLA_PREFILL_ENV) != "1"
        or not getattr(Indexer, "_glm53_prefill_fastpath_installed", False)
    ):
        return 0

    built = 0
    with torch.no_grad():
        for indexer in model.modules():
            if not isinstance(indexer, Indexer):
                continue
            op = indexer.indexer_op
            k_weight = indexer.wk_weights_proj.weight
            gate = indexer.index_kpool_compress_gate
            ape = indexer.index_kpool_compress_ape
            if k_weight.ndim != 2:
                continue
            hidden_size = k_weight.shape[1]
            if not _glm53_cache_only_indexer_contract(
                rope_dim=indexer.rope_dim,
                head_dim=indexer.head_dim,
                quant_block_size=indexer.quant_block_size,
                scale_fmt=indexer.scale_fmt,
                index_kpool=indexer.index_kpool,
                topk_tokens=indexer.topk_tokens,
                n_head=indexer.n_head,
                op_type=type(op).__name__,
                use_fp4_cache=bool(getattr(op, "use_fp4_cache", True)),
                hidden_dtype=torch.bfloat16,
                k_weight_dtype=k_weight.dtype,
                k_weight_shape=tuple(k_weight.shape),
                hidden_size=hidden_size,
                gate_dtype=gate.dtype,
                gate_shape=tuple(gate.shape),
                ape_shape=tuple(ape.shape),
                ape_dtype=ape.dtype,
                fused_weight_dtype=torch.bfloat16,
                fused_weight_shape=(2 * indexer.head_dim, hidden_size),
                fused_weight_contiguous=True,
            ):
                continue
            try:
                fused_weight = torch.cat(
                    (k_weight[: indexer.head_dim].detach(), gate.detach()),
                    dim=0,
                )
            except RuntimeError:
                continue
            if _GLM53_PREFILL_KG_WEIGHT in indexer._buffers:
                setattr(indexer, _GLM53_PREFILL_KG_WEIGHT, fused_weight)
            else:
                indexer.register_buffer(
                    _GLM53_PREFILL_KG_WEIGHT,
                    fused_weight,
                    persistent=False,
                )
            built += 1
    return built


def _glm53_indexer_scoring_unused(indexer: Indexer) -> bool:
    """Read the current request and prove the cache-only path is sufficient."""
    is_cuda = current_platform.is_cuda()
    if not is_cuda:
        return False
    forward_context = get_forward_context()
    if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
        return False
    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        return False
    k_cache_prefix = indexer.indexer_op.k_cache.prefix
    return _glm53_dense_mha_scoring_unused(
        k_cache_prefix=k_cache_prefix,
        attn_metadata=attn_metadata,
        is_cuda=True,
        cudagraph_full=False,
        stream_capturing=torch.cuda.is_current_stream_capturing(),
    )


def _glm53_cache_only_indexer_forward(
    self: Indexer,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    """Build persistent kpool state without computing unused query scores."""
    original = Indexer._glm53_prefill_original_forward
    op = self.indexer_op
    k_weight = self.wk_weights_proj.weight
    gate = self.index_kpool_compress_gate
    ape = self.index_kpool_compress_ape
    fused_weight = getattr(self, _GLM53_PREFILL_KG_WEIGHT, None)
    if not isinstance(fused_weight, torch.Tensor):
        return original(self, hidden_states, qr, positions, rotary_emb)
    if not (
        _glm53_cache_only_indexer_contract(
            rope_dim=self.rope_dim,
            head_dim=self.head_dim,
            quant_block_size=self.quant_block_size,
            scale_fmt=self.scale_fmt,
            index_kpool=self.index_kpool,
            topk_tokens=self.topk_tokens,
            n_head=self.n_head,
            op_type=type(op).__name__,
            use_fp4_cache=bool(getattr(op, "use_fp4_cache", True)),
            hidden_dtype=hidden_states.dtype,
            k_weight_dtype=k_weight.dtype,
            k_weight_shape=tuple(k_weight.shape),
            hidden_size=hidden_states.shape[-1],
            gate_dtype=gate.dtype,
            gate_shape=tuple(gate.shape),
            ape_shape=tuple(ape.shape),
            ape_dtype=ape.dtype,
            fused_weight_dtype=fused_weight.dtype,
            fused_weight_shape=tuple(fused_weight.shape),
            fused_weight_contiguous=fused_weight.is_contiguous(),
        )
        and positions is not None
        and _glm53_indexer_scoring_unused(self)
    ):
        return original(self, hidden_states, qr, positions, rotary_emb)

    # Both cache-state projections consume the same hidden rows. Their source
    # parameters stay separate for the fallback, while the load-time buffer
    # lets dense prefill produce K+gate in one GEMM. The split outputs are
    # views; both downstream kpool writers accept their explicit row strides.
    kg = F.linear(hidden_states, fused_weight)
    k, gate_score = kg.split(self.head_dim, dim=-1)
    k = _fused_indexer_k_norm(
        k,
        self.k_norm.weight,
        self.k_norm.bias,
        self.head_dim,
        self.k_norm.eps,
    )
    # q_quant and weights are not read: the shared predicate above is exactly
    # the predicate used by sparse_attn_indexer_kpool after its cache writes.
    # Reuse K as a zero-allocation placeholder instead of manufacturing dead
    # query/weight tensors solely to satisfy the custom-op schema.
    return op(
        hidden_states,
        k,
        k,
        k,
        gate_score=gate_score,
        compress_ape=ape,
        index_kpool=self.index_kpool,
        positions=positions,
    )


def install_glm53_prefill_fastpath() -> None:
    """Install once after ``.attention`` has finished defining Indexer."""
    # GLM's dense-MHA admission is exact-value opt-in. When it is disabled,
    # leave Indexer.forward byte-for-byte untouched so decode and MQA prefill
    # do not pay even the request-predicate overhead of this optimization.
    if os.environ.get(_GLM53_SM121_MLA_PREFILL_ENV) != "1":
        return
    if getattr(Indexer, "_glm53_prefill_fastpath_installed", False):
        return
    Indexer._glm53_prefill_original_forward = Indexer.forward
    Indexer.forward = _glm53_cache_only_indexer_forward
    Indexer._glm53_prefill_fastpath_installed = True
