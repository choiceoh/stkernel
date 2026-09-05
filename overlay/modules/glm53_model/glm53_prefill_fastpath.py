# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 fused K+gate indexer and dense-prefill cache-only path.

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
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .attention import (
    Indexer,
    _fused_indexer_k_norm,
    _fused_indexer_weight_scale,
    _pad_indexer_heads,
)
from .ops.kpool_compress import fwht128_quant_fp8

# deneb fork (glm53_indexer_gate_splitk): the split-K helper when that module
# is mounted, else the stock fp32 torch.mm. Resolved once, on first call, so
# the fused indexer forward stays loadable without the sibling file. Only
# "module not mounted" is tolerated silently; an ImportError raised INSIDE the
# helper file is logged, and a knob that asks for split-K while the helper is
# missing is announced rather than quietly served stock.
_HEAD_GATE = None
_HEAD_GATE_MODULE = "vllm.models.glm5next.nvidia.glm53_indexer_gate"


def _glm53_head_gate(x, w):
    global _HEAD_GATE
    if _HEAD_GATE is None:
        try:
            from vllm.models.glm5next.nvidia.glm53_indexer_gate import head_gate as fn
        except ImportError as e:
            if not (isinstance(e, ModuleNotFoundError) and e.name == _HEAD_GATE_MODULE):
                logger.exception("[indexer-gate] helper import failed -> stock torch.mm")
            elif os.environ.get("VLLM_GLM53_INDEXER_GATE_SPLITK", "").strip() == "1":
                logger.warning("[indexer-gate] VLLM_GLM53_INDEXER_GATE_SPLITK=1 but "
                               "glm53_indexer_gate_splitk is not mounted -> stock torch.mm")

            def fn(x, w):
                return torch.mm(x.float(), w)
        _HEAD_GATE = fn
    return _HEAD_GATE(x, w)

logger = init_logger(__name__)

_GLM53_SM121_MLA_PREFILL_ENV = "VLLM_GLM53_SM121_MLA_PREFILL"
_GLM53_FUSED_K_GATE_ENV = "VLLM_GLM53_FUSED_K_GATE"
_GLM53_PREFILL_METADATA_WARMUP_ENV = "VLLM_GLM53_PREFILL_METADATA_WARMUP"
_GLM53_PREFILL_KG_WEIGHT = "_glm53_prefill_kg_weight"


def _glm53_fused_k_gate_enabled() -> bool:
    """The dense-prefill arm implies fusion; fusion can also stand alone."""
    return (
        os.environ.get(_GLM53_FUSED_K_GATE_ENV) == "1"
        or os.environ.get(_GLM53_SM121_MLA_PREFILL_ENV) == "1"
    )


def warm_glm53_prefill_metadata_runtime(model: torch.nn.Module) -> int:
    """Populate Triton's real runtime cache for GLM's pooled metadata key.

    ``VllmJitKernel.warmup`` compiles synthetic pointer variants, but Triton's
    launch cache still distinguishes aligned and sliced int32 pointers.  Run
    one valid row for both variants of GLM's pooled compression ratio so
    the first user prefill cannot become the compilation request.  This is
    best-effort and independent of the dense-prefill experiment.
    """
    if os.environ.get(_GLM53_PREFILL_METADATA_WARMUP_ENV, "1") != "1":
        return 0
    if not torch.cuda.is_available():
        return 0

    vllm_config = getattr(model, "vllm_config", None)
    hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if (
        getattr(hf_config, "model_type", None) != "glm5_next_text"
        or getattr(hf_config, "index_kpool", None) != 4
        or getattr(parallel_config, "tensor_parallel_size", None) != 4
    ):
        return 0

    parameter = next(model.parameters(), None)
    if parameter is None or parameter.device.type != "cuda":
        return 0

    # Resolve dynamically because another image profile owns a different
    # overlay for the same vLLM module; only the GLM profile guarantees this
    # private kernel symbol.
    import importlib

    indexer_module = importlib.import_module(
        "vllm.v1.attention.backends.mla.indexer"
    )
    metadata_kernel = getattr(
        indexer_module, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL"
    )

    device = parameter.device
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_compressed_seq_lens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    token_to_seq = torch.empty(1, dtype=torch.int32, device=device)
    cu_seqlen_ks = torch.empty(1, dtype=torch.int32, device=device)
    cu_seqlen_ke = torch.empty(1, dtype=torch.int32, device=device)
    aligned = torch.tensor([4], dtype=torch.int32, device=device)
    unaligned_storage = torch.tensor([-1, 4], dtype=torch.int32, device=device)

    launches = 0
    for uncompressed_seq_lens in (aligned, unaligned_storage[1:]):
        metadata_kernel(
            query_start_loc,
            uncompressed_seq_lens,
            cu_compressed_seq_lens,
            cu_compressed_seq_lens,
            token_to_seq,
            cu_seqlen_ks,
            cu_seqlen_ke,
            0,
            1,
            0,
            1,
            1,
            num_reqs=1,
            COMPRESS_RATIO=4,
        )
        launches += 1
    torch.cuda.synchronize(device)
    return launches


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
        not _glm53_fused_k_gate_enabled()
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
            # The fused forward keeps an fp32 copy of the head-weight rows in
            # `_wp_fp32`, filled on first use. Nothing created the slot, so
            # the very first call raised
            #   AttributeError: 'Indexer' object has no attribute '_wp_fp32'
            # and the engine died at load. Create it here, where the rest of
            # this layer's fast-path state is set up.
            indexer._wp_fp32 = None
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


def _glm53_fused_indexer_forward(
    self: Indexer,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    """Fuse K+gate for exact GLM prefill, mixed, and decode paths."""
    original = Indexer._glm53_prefill_original_forward
    op = self.indexer_op
    k_weight = self.wk_weights_proj.weight
    gate = self.index_kpool_compress_gate
    ape = self.index_kpool_compress_ape
    fused_weight = getattr(self, _GLM53_PREFILL_KG_WEIGHT, None)
    if not isinstance(fused_weight, torch.Tensor):
        return original(self, hidden_states, qr, positions, rotary_emb)
    if not _glm53_cache_only_indexer_contract(
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
    ):
        return original(self, hidden_states, qr, positions, rotary_emb)

    if positions is not None and _glm53_indexer_scoring_unused(self):
        return _glm53_cache_only_indexer_forward(
            self, hidden_states, qr, positions, rotary_emb
        )
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)

    # The head-weight rows from wk_weights_proj are intentionally not emitted
    # in BF16: ranking keeps the stock FP32 projection below. The fused buffer
    # contains only the K rows and the trained per-token pool gate.
    kg = F.linear(hidden_states, fused_weight)
    k, gate_score = kg.split(self.head_dim, dim=-1)
    # getattr, not attribute access: install() and prepare() are gated
    # separately, so the forward can be live on a layer prepare() skipped
    # (contract drift, allocation failure). Falling back to computing it
    # here is correct and cheap; crashing at load is not.
    if getattr(self, "_wp_fp32", None) is None:
        self._wp_fp32 = (
            k_weight.data[self.head_dim :, :].t().contiguous().float()
        )
    weights = _glm53_head_gate(hidden_states, self._wp_fp32)  # deneb fork (glm53_indexer_gate_splitk)

    k = _fused_indexer_k_norm(
        k,
        self.k_norm.weight,
        self.k_norm.bias,
        self.head_dim,
        self.k_norm.eps,
    )
    # The exact production contract is NoPE. RoPE or a different geometry
    # failed closed above, so the stock query transform can be preserved
    # without duplicating its shape-sensitive rotary branch.
    q = q.view(-1, self.head_dim)
    q_fp8, q_scale = fwht128_quant_fp8(q)
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
    q_scale = q_scale.view(-1, self.n_head, 1)
    weights = _fused_indexer_weight_scale(
        weights,
        q_scale,
        self.softmax_scale * self.n_head**-0.5,
    )
    if self.n_head < 32:
        pad = 32 - self.n_head
        q_fp8 = _pad_indexer_heads(q_fp8, pad)
        weights = _pad_indexer_heads(weights, pad)

    return op(
        hidden_states,
        q_fp8,
        k,
        weights,
        gate_score=gate_score,
        compress_ape=ape,
        index_kpool=self.index_kpool,
        positions=positions,
    )


def install_glm53_prefill_fastpath() -> None:
    """Install once after ``.attention`` has finished defining Indexer."""
    # Fusion can serve decode independently of the dense-prefill experiment.
    # With both exact-value knobs disabled, leave Indexer.forward untouched.
    if not _glm53_fused_k_gate_enabled():
        return
    if getattr(Indexer, "_glm53_prefill_fastpath_installed", False):
        return
    Indexer._glm53_prefill_original_forward = Indexer.forward
    Indexer.forward = _glm53_fused_indexer_forward
    Indexer._glm53_prefill_fastpath_installed = True
    logger.info("glm53 fused K+gate: ARMED (Indexer.forward replaced)")
