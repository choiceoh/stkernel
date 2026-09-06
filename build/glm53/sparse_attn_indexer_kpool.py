# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    _kpool_softmax_rotate_write_cache_kernel,
    _expand_pools_and_append_tail_kernel,
    expand_pools_and_append_tail,
    expand_pools_to_tokens,
    kpool_decode_update_and_maybe_write_cache_batched,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# Rollback switches are process configuration. Latch them at import instead of
# reading os.environ in every sparse-indexer layer call.
_KPOOL_TAIL_CACHE_ENABLED = (
    os.environ.get("VLLM_KPOOL_SKIP_TAIL_CACHE") != "1"
)
_KPOOL_DECODE_WRITE_ENABLED = (
    os.environ.get("VLLM_KPOOL_SKIP_DECODE_WRITE") != "1"
)
# deneb fork (launch-count bundle 2): hand the batched kpool update kernel
# the runner's int64 positions directly instead of a per-layer int32 cast
# copy (11 `direct_copy` launches per decode step, one before each
# `_kpool_decode_update_batched_kernel`). The kernel's position arithmetic
# is value-identical in int64 (positions are far below 2^31), so the cache
# writes are bit-identical; only the launch goes. Exact "1" arms; the
# uniform decode path only (the padded non-uniform path keeps the scatter).
_KPOOL_UPDATE_DIRECT_POS = (
    os.environ.get("VLLM_GLM53_KPOOL_UPDATE_DIRECT_POS", "").strip() == "1"
)
_GLM53_PREFILL_WRITE_PLAN_CACHE = "glm53_kpool_prefill_write_plans"

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _glm53_dense_mha_layer_name(k_cache_prefix: str) -> str:
    """Resolve the sibling MLA metadata key for GLM's kpool indexer.

    ``MultiHeadLatentAttentionWrapper`` can normally bind this name through a
    field on the stock sparse-indexer op.  The GLM kpool fork predates that
    field, so its short dense-prefill path kept computing a top-k buffer that
    attention never reads.  Derive the same name from the exact GLM module
    layout and fail closed if that layout ever changes.
    """
    suffix = ".indexer.k_cache"
    if not k_cache_prefix.endswith(suffix):
        return ""
    return f"{k_cache_prefix[: -len(suffix)]}.attn"


def _glm53_dense_mha_scoring_unused(
    *,
    k_cache_prefix: str,
    attn_metadata: dict,
    is_cuda: bool,
    cudagraph_full: bool,
    stream_capturing: bool,
) -> bool:
    """Whether the sibling MLA will ignore this indexer's scoring output."""
    if not is_cuda or cudagraph_full or stream_capturing:
        return False
    dense_mha_layer = _glm53_dense_mha_layer_name(k_cache_prefix)
    if not dense_mha_layer:
        return False
    mla_metadata = attn_metadata.get(dense_mha_layer)
    prefill_metadata = getattr(mla_metadata, "prefill", None)
    return bool(
        getattr(prefill_metadata, "use_dense_mha", False)
        and getattr(mla_metadata, "num_decode_tokens", -1) == 0
    )


# kpool write helper: form pools from the current token batch and compress them
# into the index K cache via the fused Triton kernel.


def _kpool_prefill_windows(x: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Zero-copy ``[window, pool slot, dim]`` views over token-major input."""
    # Tensor.unfold appends the window dimension, producing [W, D, P]. Move
    # that dimension between W and D; both operations are views. The resulting
    # strides are [D, D, 1], exactly what the compression kernel accepts.
    return x.unfold(0, pool_size, 1).movedim(-1, 1)


def _kpool_compress_strided_write_cache(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Launch the stock compressor without materializing strided windows.

    The image's public wrapper calls ``contiguous()`` on K and score even
    though its Triton kernel already receives both row and pool-slot strides.
    Prefill windows overlap by ``pool_size - 1`` rows, so materializing them
    copies almost ``pool_size`` times the source. Preserve the zero-copy view
    and invoke the pinned kernel with the same output/cache contract.
    """
    assert slot_k.ndim == 3 and slot_k.stride(-1) == 1
    assert slot_score.shape == slot_k.shape and slot_score.stride(-1) == 1
    assert ape.shape == slot_k.shape[1:] and ape.stride(-1) == 1
    assert slot_k.shape[2] == head_dim
    assert slot_k.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8
    assert loc.shape == write_mask.shape == (slot_k.shape[0],)
    assert loc.dtype == torch.int64 and write_mask.dtype == torch.bool
    if slot_k.shape[0] == 0:
        return

    page_size = kv_cache.shape[1]
    buf_fp8 = kv_cache.view(torch.float8_e4m3fn)
    buf_fp32 = kv_cache.view(torch.float32)
    buf_numel_per_page = kv_cache.stride(0)
    s_offset_nbytes_in_page = page_size * head_dim

    # RETURN_COMPRESSED=False makes both output-only pointers inert. Reuse the
    # cache views, matching the public wrapper without allocating dummies.
    _kpool_softmax_rotate_write_cache_kernel[(slot_k.shape[0],)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        buf_fp8,
        buf_fp32,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=s_offset_nbytes_in_page,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=True,
        RETURN_COMPRESSED=False,
        WRITE_CACHE=True,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


@triton.jit
def _fill_short_prefill_topk_kernel(
    out_ptr,
    positions_ptr,
    out_stride_0,
    out_stride_1,
    positions_stride_0,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    """Write causal full-attention indices directly into the top-k buffer."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    position = tl.load(positions_ptr + row * positions_stride_0)
    values = tl.where(cols <= position, cols, -1)
    tl.store(
        out_ptr + row * out_stride_0 + cols * out_stride_1,
        values,
        mask=mask,
    )


def _fill_short_prefill_topk(
    out: torch.Tensor, positions: torch.Tensor
) -> None:
    """One-pass replacement for fill + broadcast arange + boolean mask."""
    assert out.ndim == 2 and out.dtype == torch.int32
    assert positions.ndim == 1 and positions.shape[0] == out.shape[0]
    if out.shape[0] == 0 or out.shape[1] == 0:
        return
    block_n = 256
    grid = (out.shape[0], triton.cdiv(out.shape[1], block_n))
    _fill_short_prefill_topk_kernel[grid](
        out,
        positions,
        out.stride(0),
        out.stride(1),
        positions.stride(0),
        out.shape[1],
        BLOCK_N=block_n,
        num_warps=4,
    )


def _kpool_prefill_write_plan(
    slot_mapping: torch.Tensor,
    kpool: int,
    cache: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pool destinations/mask, optionally shared by this forward.

    Every sparse layer receives the same immutable metadata tensor during a
    dense prefill. Key on its exact view identity and layout; a different
    allocation, offset, stride, dtype, device, length, or pool width gets a
    separate plan. The caller supplies a forward-context-owned cache only for
    the non-captured dense path, so no tensor address can escape that forward.
    """
    key = None
    if cache is not None:
        key = (
            slot_mapping.data_ptr(),
            slot_mapping.shape[0],
            slot_mapping.stride(0),
            slot_mapping.dtype,
            slot_mapping.device,
            kpool,
        )
        cached = cache.get(key)
        if cached is not None:
            return cached

    loc = slot_mapping[kpool - 1 :].to(torch.int64)
    plan = (loc, loc >= 0)
    if cache is not None:
        cache[key] = plan
    return plan


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
    write_plan_cache: dict | None = None,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. Every position is treated as a pool-completion
    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first (``torch.nonzero`` + boolean-mask gather + an
    ``ok.all()`` check) costs two device syncs on the eager prefill path and
    buys nothing numerically. Assumes pool-aligned chunk starts (same
    invariant as sglang).
    """
    n = slot_mapping.shape[0]
    # No complete sliding window exists in a batch smaller than one pool.
    if n < kpool:
        return
    # Row j is the window ending at token j + kpool - 1. Align the destination
    # and mask to that completion token. Invalid/trailing pool slots remain
    # no-ops inside the kernel, exactly as before.
    k_windows = _kpool_prefill_windows(k, kpool)
    score_windows = _kpool_prefill_windows(gate_score, kpool)
    loc, write_mask = _kpool_prefill_write_plan(
        slot_mapping,
        kpool,
        write_plan_cache,
    )
    _kpool_compress_strided_write_cache(
        kv_cache,
        k_windows,
        score_windows,
        ape,
        loc,
        write_mask,
        head_dim,
        round_scale,
    )


@triton.jit
def _kpool_seed_tail_cache_strided_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    key_stride_0,
    score_stride_0,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Seed one request tail while preserving split-projection row strides."""
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // KPOOL
    ahead = tl.load(
        tslot_ptr + i + KPOOL,
        mask=i + KPOOL < n_tokens,
        other=-1,
    ).to(tl.int64)
    if ahead >= 0 and ahead // KPOOL == blk:
        return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM
    key = tl.load(key_ptr + i * key_stride_0 + offs, mask=mask)
    score = tl.load(score_ptr + i * score_stride_0 + offs, mask=mask)
    tl.store(tail_ptr + base + offs, key, mask=mask)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, score, mask=mask)


def _kpool_seed_tail_cache_strided(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
) -> None:
    """Stride-aware equivalent of the image's contiguous-only tail seeder."""
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == gate_score.dtype == torch.bfloat16
    assert key.shape == gate_score.shape
    assert key.ndim == 2 and key.shape[1] == head_dim
    assert key.stride(1) == gate_score.stride(1) == 1
    assert tail_slot_mapping.ndim == 1
    assert tail_slot_mapping.shape[0] == key.shape[0]
    n_tokens = tail_slot_mapping.shape[0]
    if n_tokens == 0:
        return
    _kpool_seed_tail_cache_strided_kernel[(n_tokens,)](
        key,
        gate_score,
        tail_slot_mapping,
        tail_kv_cache,
        key.stride(0),
        gate_score.stride(0),
        n_tokens,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


def _build_decode_scatter_indices(
    decode_lens: torch.Tensor,
    num_requests: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (request id, intra-request index) for a non-uniform decode
    batch, with ``n == decode_lens.sum()`` as a host int (avoids a
    device sync and keeps both repeat_interleaves sync-free).

    Shared by every ``_scatter_decode_tokens_by_request`` call in a step:
    building it per call would repeat the same repeat_interleave/cumsum chain
    up to 5x per layer on the eager decode break.
    """
    device = decode_lens.device
    dl = decode_lens.to(torch.int64)
    req_id = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int64),
        dl,
        output_size=n,
    )
    req_starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.int64), dl[:-1]]),
        dim=0,
    )
    # Broadcast the per-request start offsets to per-token (length n ==
    # dl.sum()) so each token's intra-request index subtracts its own
    # request's start.
    starts = torch.repeat_interleave(req_starts, dl, output_size=n)
    intra = torch.arange(n, device=device, dtype=torch.int64) - starts
    return req_id, intra


def _scatter_decode_tokens_by_request(
    tokens: torch.Tensor,
    pad_value,
    num_requests: int,
    lmax: int,
    scatter_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Group ``[N, ...]`` decode tokens into a padded ``[num_requests, lmax, ...]``
    layout: request ``r``'s tokens at row ``r`` in order; short requests padded.

    Unlike ``pack_seq_triton`` this is dtype-agnostic (needed for the int32
    slot/pos tensors) — it scatters with the shared per-step indices from
    ``_build_decode_scatter_indices``. Used only for the non-uniform
    (``requires_padding``) decode batch; uniform batches use a zero-copy
    reshape.
    """
    req_id, intra = scatter_indices
    out = torch.full(
        (num_requests, lmax, *tokens.shape[1:]),
        pad_value,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    out[req_id, intra] = tokens
    return out


def _decode_topk_seq_lens(
    positions: torch.Tensor,
    decode_lens: torch.Tensor,
    num_decode_tokens: int,
    batch_size: int,
    next_n: int,
    requires_padding: bool,
) -> torch.Tensor:
    """Token-granular seq_len (pos + 1) per pool-topk row, layout-aware.

    ``pool_topk`` (and the logits it comes from) follow the padded
    ``[batch_size, next_n]`` grid whenever ``requires_padding`` is set, so row
    ``(b, t)`` corresponds to flat decode token ``offset_b + t`` -- NOT
    ``b * next_n + t``. Slicing flat ``positions[: batch_size * next_n]``
    (the uniform-layout shortcut) misaligns every row after the first
    non-uniform request and, past the decode region, reads prefill tokens'
    positions; ``expand_pools_and_append_tail`` then anchors the tail at
    another request's length, dropping the row's real tail tokens or emitting
    indices past its sequence (out-of-bounds block-table reads). Padded rows
    get 0 (empty tail); they are dropped by ``unpack_seq_triton`` anyway.
    """
    n = batch_size * next_n
    if not requires_padding:
        return positions[:n].to(torch.int32) + 1
    scatter_idx = _build_decode_scatter_indices(
        decode_lens, batch_size, num_decode_tokens
    )
    padded = _scatter_decode_tokens_by_request(
        positions[:num_decode_tokens].to(torch.int32),
        -1,
        batch_size,
        next_n,
        scatter_idx,
    )
    return padded.reshape(n) + 1  # pad rows: -1 + 1 = 0 -> empty tail


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def _pool_topk_scratch_fits(
    total_rows: int,
    active_rows: int,
    output_width: int,
    num_rows: int,
    select_k: int,
) -> bool:
    """Whether inactive output storage can hold packed pool-level top-k."""
    return (
        0 <= active_rows <= total_rows
        and num_rows >= 0
        and select_k > 0
        and num_rows * select_k
        <= (total_rows - active_rows) * output_width
    )


def _inactive_pool_topk_scratch(
    topk_indices_buffer: torch.Tensor,
    active_rows: int,
    num_rows: int,
    select_k: int,
) -> torch.Tensor | None:
    """Return packed CUDA top-k scratch wholly outside active output rows.

    The CUDA top-k ops index output as ``row * select_k`` and do not accept an
    output stride, so ``topk_indices_buffer[:, :select_k]`` is not a valid
    destination when the persistent output is wider. Both CUDA top-k variants
    write every destination element (including trailing ``-1`` sentinels), so
    inactive, uninitialised rows are safe scratch. Keeping scratch after the
    active prefix also prevents padded decode rows from overwriting a mixed
    batch's already-computed prefill top-k.
    """
    if (
        topk_indices_buffer.ndim != 2
        or topk_indices_buffer.dtype != torch.int32
        or not topk_indices_buffer.is_contiguous()
        or not _pool_topk_scratch_fits(
            topk_indices_buffer.shape[0],
            active_rows,
            topk_indices_buffer.shape[1],
            num_rows,
            select_k,
        )
    ):
        return None
    scratch_elems = num_rows * select_k
    inactive = topk_indices_buffer[active_rows:].view(-1)
    return inactive[:scratch_elems].view(num_rows, select_k)


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


_ALWAYS_SELECT_TAIL: bool | None = None


def _always_select_tail() -> bool:
    """Read index_kpool_always_select_tail off the model config, once.

    Nothing else in vLLM reads this flag, so its default (True) has never taken
    effect. Resolved lazily and cached: the decode path runs per step.
    """
    global _ALWAYS_SELECT_TAIL
    if _ALWAYS_SELECT_TAIL is None:
        val = True
        cfg = get_current_vllm_config_or_none()
        hf = getattr(getattr(cfg, "model_config", None), "hf_config", None)
        hf = getattr(hf, "text_config", hf)
        if hf is not None:
            val = bool(getattr(hf, "index_kpool_always_select_tail", True))
        _ALWAYS_SELECT_TAIL = val
    return _ALWAYS_SELECT_TAIL


def _force_tail_pool_into_logits(
    logits: torch.Tensor, token_seq_lens: torch.Tensor, pool_size: int
) -> None:
    """Make the most recent completed pool unbeatable in the pool-level top-k.

    ``token_seq_lens`` is token-granular and follows the same row order as
    ``logits`` (padded rows carry 0, which yields no pool and is skipped).
    Mutates ``logits`` in place -- it comes straight from the paged-MQA kernel
    and the top-k below is its only consumer.
    """
    last_pool = token_seq_lens.to(torch.int64) // pool_size - 1
    col = last_pool.clamp(0, logits.shape[1] - 1).unsqueeze(1)
    keep = logits.gather(1, col)
    biggest = torch.full_like(keep, torch.finfo(logits.dtype).max)
    logits.scatter_(1, col, torch.where(last_pool.unsqueeze(1) >= 0, biggest, keep))


# deneb fork (glm53_kpool_tail_select, 34차 "module and node fusion"): the
# decode top-k glue of every full-attention layer -- `_decode_topk_seq_lens`
# (positions -> token seq_len, 2 launches), `_force_tail_pool_into_logits`
# (8 launches: int64 cast, floor-div, sub, clamp, gather, full_like, compare,
# where + scatter) and the final copy of the expanded indices into the
# persistent buffer (1 launch) -- as ONE Triton launch plus a direct-write
# expand. Eleven aten kernels x 11 layers = ~120 graph nodes and ~0.25 ms of
# a decode step (34차 trace, profiler-inflated). Integer index arithmetic and
# the same finfo.max bias, so the outputs are bit-identical to the stock
# chain by construction (probes/indexer_decode_fused_check.py is the gate).
# Uniform spec-verify decode only (the padded layout keeps the stock chain).
_INDEXER_DECODE_FUSED = (
    os.environ.get("VLLM_GLM53_INDEXER_DECODE_FUSED", "0").strip() == "1"
)
_FUSED_ANNOUNCED: set = set()


@triton.jit
def _glm53_indexer_tail_select_kernel(
    pos_ptr,       # [n_rows] int64/int32 token positions (row r = decode token r)
    logits_ptr,    # [n_rows, n_cols] fp32 pool logits, mutated in place
    seq_out_ptr,   # [n_rows] int32 token-granular seq_len = pos + 1
    n_rows,
    n_cols,
    logits_s0,
    big,           # finfo(logits.dtype).max
    POOL: tl.constexpr,
    FORCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    m = rows < n_rows
    pos = tl.load(pos_ptr + rows, mask=m, other=-1)
    # stock: positions[:n].to(int32) + 1 -- int32 add after the narrowing
    seq = pos.to(tl.int32) + 1
    tl.store(seq_out_ptr + rows, seq, mask=m)
    if FORCE:
        # stock: last_pool = seq.to(int64) // POOL - 1; col = clamp(last_pool,
        # 0, n_cols - 1); logits[r, col] = biggest where last_pool >= 0 (else
        # the gathered value is written back unchanged)
        last_pool = seq.to(tl.int64) // POOL - 1
        col = tl.minimum(tl.maximum(last_pool, 0), n_cols - 1)
        wm = m & (last_pool >= 0)
        tl.store(logits_ptr + rows.to(tl.int64) * logits_s0 + col, big, mask=wm)


def indexer_tail_select_fused(
    positions: torch.Tensor,
    logits: torch.Tensor,
    n_rows: int,
    pool_size: int,
    force_tail: bool,
) -> torch.Tensor:
    """One launch for `_decode_topk_seq_lens` (uniform layout) and
    `_force_tail_pool_into_logits`. Returns the int32 token seq_len per row;
    `logits` is mutated in place exactly as the stock scatter does."""
    seq = torch.empty(n_rows, dtype=torch.int32, device=positions.device)
    if n_rows == 0:
        return seq
    BLOCK = 128
    _glm53_indexer_tail_select_kernel[(triton.cdiv(n_rows, BLOCK),)](
        positions,
        logits,
        seq,
        n_rows,
        logits.shape[1],
        logits.stride(0),
        float(torch.finfo(logits.dtype).max),
        POOL=pool_size,
        FORCE=bool(force_tail),
        BLOCK=BLOCK,
    )
    return seq


def expand_pools_and_append_tail_into(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """`expand_pools_and_append_tail` written straight into `out` (a row-major
    view of the persistent top-k buffer) instead of a fresh tensor plus a
    copy. Same kernel, same values; returns the written view."""
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    dst = out[:rows, :out_cols]
    if dst.stride(1) != 1 or dst.dtype != torch.int32 or rows == 0:
        return None
    BLOCK_COLS = 128
    _expand_pools_and_append_tail_kernel[(rows, triton.cdiv(out_cols, BLOCK_COLS))](
        pool_ids,
        seq_lens,
        dst,
        topk,
        out_cols,
        POOL_SIZE=pool_size,
        BLOCK_COLS=BLOCK_COLS,
        pid_s0=pool_ids.stride(0),
        out_s0=dst.stride(0),
    )
    return dst


@eager_break_during_capture
def sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Sentinel allocation so the profiler's peak-memory measurement covers
        # the runtime logits tensor. The decode-path fp8_fp4_paged_mqa_logits
        # output is [B*next_n, max_model_len] float32 -- sized by max_model_len,
        # NOT bounded by the prefill chunk cap. This profiling branch returns
        # the fake before ever calling that kernel, so its output tensor is
        # invisible unless we size this sentinel to the real worst-case decode
        # batch; otherwise large max_model_len / max_num_batched_tokens OOMs at
        # warmup (the old fixed VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512MiB was
        # ~10x too small at max_model_len=1M / b8192).
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # float32 logits -> 4 bytes/element; uint8 sentinel so elems == bytes.
        decode_logits_elems = worst_decode_tokens * max_model_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_kpool_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # Prove the no-consumer case before cache insertion so all sparse layers in
    # this forward can share the identical completion slice/write mask.
    # The return remains after both persistent cache writes below.
    is_cuda = current_platform.is_cuda()
    dense_mha_scoring_unused = _glm53_dense_mha_scoring_unused(
        k_cache_prefix=k_cache_prefix,
        attn_metadata=attn_metadata,
        is_cuda=is_cuda,
        cudagraph_full=(
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        ),
        stream_capturing=(
            torch.cuda.is_current_stream_capturing() if is_cuda else True
        ),
    )
    write_plan_cache = None
    if dense_mha_scoring_unused:
        write_plan_cache = forward_context.additional_kwargs.setdefault(
            _GLM53_PREFILL_WRITE_PLAN_CACHE,
            {},
        )

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # kpool prefill write: pool kpool consecutive prefill tokens via
            # softmax(gate+ape)-weighted sum -> Hadamard -> fp8 -> pool slots.
            # Decode tokens (the first num_decode_tokens in the batch) cannot be
            # pooled here — their pool's earlier tokens are not in this batch —
            # so they are deferred to the tail-buffer kernel in has_decode.
            # compress_ratio == index_kpool makes slot_mapping pool-granular.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                    write_plan_cache=write_plan_cache,
                )
                # Persist the prefill tail (trailing incomplete pool's raw K +
                # gate score) into the paged tail cache, so the decode side can
                # compress the boundary pool correctly -- including across PD
                # transfer, where the connector ships this block. Their tail
                # slots land at offsets pos % kpool of the request's tail block,
                # exactly where the decode reconstruction reads them.
                #
                # This must run PER REQUEST (sglang writes the tail inside its
                # per-request extend loop, `set_compress_tail_for_request`).
                # Taking the batch's trailing `n_prefill % kpool` tokens only
                # covers the LAST request: every other request in a
                # multi-request prefill batch then compresses its boundary pool
                # against a stale tail block (the ring is reused across
                # requests), corrupting one pool at each request's
                # prompt->decode boundary. Invisible on single-request probes;
                # hit by every concurrent-serving batch.
                if (
                    tail_kv_cache is not None
                    and tail_prefix is not None
                    and _KPOOL_TAIL_CACHE_ENABLED
                ):
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                        # Seed each request's trailing <= kpool raw K + gate
                        # into the paged tail ring with one kernel. The old
                        # scatter chain (block-id compare + nonzero/boolean
                        # gathers + 2 indexed writes) cost ~12 elementwise ops
                        # and 4 device syncs per layer; the kernel derives the
                        # same per-request tail membership in-kernel: token i
                        # is in its request's tail iff the token kpool ahead
                        # maps to a different tail block (1 block/req) or is
                        # past the batch. Writes are one-per-token to distinct
                        # pos % kpool offsets, so the result is identical.
                        _kpool_seed_tail_cache_strided(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            index_kpool,
                            head_dim,
                        )
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    # Fresh short GLM prefills can use dense MLA when the SM121 prefill arm is
    # admitted.  Dense MLA does not consume top-k indices, but this kpool fork
    # used to continue below and build/mask a [tokens, 2048] buffer in every
    # sparse layer.  Keep the index-K and tail-cache writes above -- decode and
    # future cached turns need them -- then skip only the dead scoring output.
    #
    # Match the stock sparse indexer's capture/mixed-batch guards.  A long or
    # cached-context prefill has use_dense_mha=False; a mixed batch has decode
    # tokens; and an unknown module layout resolves to an empty name.  All of
    # those retain the existing kpool top-k path.
    if dense_mha_scoring_unused:
        return topk_indices_buffer

    short_prefill = False
    prefill_metadata = None
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short-sequence full-attention fast path (mirrors sglang
        # IndexerKPool._full_topk_for_short_sequence). When every prefill
        # request's full context is <= topk_tokens, sparse selection would
        # pick ALL pools anyway (topk_pool = topk_tokens // index_kpool >=
        # num_pools, plus the always-selected tail == every token), so running
        # the MQA-logits is pointless and (in this port) triggers OOBs. Skip
        # it and attend to every token causally instead. The index-K cache was
        # already written above; this only fills the topk buffer. Real
        # sparsity only kicks in for contexts > topk_tokens.
        n_prefill_sf = num_tokens - num_decode_tokens
        # Host-side short-prefill predicate: max_prefill_seq_len is computed
        # in the metadata builder (exact for prefill rows) and equals
        # positions[prefill_slice].max() + 1, so this replaces a
        # positions.max().item() device sync per layer. -1 (unknown metadata)
        # falls back to the device-side check.
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )

    # Every non-short path scatters only a subset of columns and therefore
    # needs the full sentinel initialization. A pure short prefill writes every
    # active cell below in one pass, so the old fill would be overwritten in
    # its entirety. Mixed decode+prefill keeps the conservative full fill for
    # padded/decoded rows that the short-prefill writer does not cover.
    short_prefill_covers_active = (
        short_prefill
        and not has_decode
        and hidden_states.shape[0] == num_tokens
    )
    if not short_prefill_covers_active:
        topk_indices_buffer[: hidden_states.shape[0]] = -1

    if has_prefill:
        assert prefill_metadata is not None
        if short_prefill:
            # short_prefill is only True when positions is not None (above),
            # but narrow explicitly for the indexer below.
            assert positions is not None
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _fill_short_prefill_topk(
                _buf, positions[num_decode_tokens:num_tokens]
            )
            prefill_chunks = ()
        else:
            # Get the full shared workspace buffers only when sparse scoring
            # consumes them. The short full-attention path used to allocate or
            # look these up and then iterate an empty chunk tuple.
            # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale)
            # and MXFP4 (head_dim/2 bytes packed + ue8m0 scales).
            workspace_manager = current_workspace_manager()
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                values_spec,
                scales_spec,
            )
            prefill_chunks = prefill_metadata.chunks

        for chunk in prefill_chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            logits = fp8_fp4_mqa_logits(
                (q_slice_cast, q_scale_slice),
                (k_quant_cast, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            num_rows = logits.shape[0]

            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            if index_kpool > 1:
                pool_topk = (
                    _inactive_pool_topk_scratch(
                        topk_indices_buffer,
                        hidden_states.shape[0],
                        num_rows,
                        select_k,
                    )
                    if current_platform.is_cuda()
                    else None
                )
                if pool_topk is None:
                    pool_topk = torch.full(
                        (num_rows, select_k),
                        -1,
                        dtype=torch.int32,
                        device=logits.device,
                    )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )

            if index_kpool > 1:
                pool_ids = (
                    topk_dst
                    if current_platform.is_cuda()
                    else topk_dst.to(torch.int64)
                )
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). seq_len is token-granular
                    # (pos+1); the kernel derives pool_len internally.
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    valid = pool_ids >= 0
                    expanded = expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)

        # kpool decode write (must precede the logits read). Append each decode
        # token's k/gate to its REQUEST's tail ring; when a pool fills
        # (pos % kpool == kpool-1) compress + write at the pool slot that
        # compress_ratio hands us via slot_mapping.
        #
        # Spec verify batches next_n (>1) tokens per request. The per-request
        # tail ring must accumulate a request's tokens IN POSITION ORDER, so we
        # group tokens by request ([num_requests, next_n, ...]) and run the
        # per-request kernel once per token-slot — sequential launches keep each
        # request's tokens ordered (token t stashes before token t+1 reads it
        # for pool completion). Mirrors sglang's _forward_cuda_target_verify
        # (per-request kpool write plan, seqlen_per_q = write_start + k + 1).
        # Plain decode (next_n == 1) collapses to a single launch.
        #
        # NOTE: positions must be TOKEN-granular (per-token position, not the
        # pool-granular decode_metadata.seq_lens which is divided by
        # compress_ratio). The kernel derives the pool phase and tail-ring index
        # from pos % kpool, so a pool-granular pos misaligns every pool; a
        # per-request pos under spec is also too short (B entries for B*next_n
        # tokens) and reads out of bounds.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
            and _KPOOL_DECODE_WRITE_ENABLED
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            # The indexer's flatten decode path rewrites decode_lens to all-1s
            # and reports requires_padding=False even for a variable MTP-verify
            # batch (e.g. one request verifies 3 tokens while the rest verify
            # 4). The logits read is fine with that, but the kpool WRITE must
            # group tokens by their original request. Uniformity and the scatter
            # lmax are precomputed on the host in build()
            # (decode_is_uniform / write_max_decode_len), so this branch needs
            # no runtime .item() -- a .item() under cudagraph capture forces a
            # host sync and invalidates the stream.
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                # Legacy metadata without per-request lens: fall back to the
                # host-side requires_padding flag. Unreached now (per-request
                # lens is always populated for decode), kept defensive.
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
            if not use_uniform:
                # Non-uniform decode_lens (mixed plain-decode + spec-verify, or
                # a variable MTP-verify batch): scatter actual tokens into a
                # padded [B, lmax] layout. int32 tensors can't go through
                # pack_seq_triton (float/uint8 only). The scatter indices are
                # shared by all five scatters below (and the tail slot one).
                scatter_idx = _build_decode_scatter_indices(
                    group_lens, num_requests, num_decode_tokens
                )
                dec_k = _scatter_decode_tokens_by_request(
                    k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:num_decode_tokens],
                    0,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                next_n = num_decode_tokens // num_requests
                shape2 = (num_requests, next_n)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                if _KPOOL_UPDATE_DIRECT_POS and positions.dtype in (
                    torch.int64, torch.int32
                ):
                    # int64 straight into the kernel: no cast launch
                    dec_pos = positions[:num_decode_tokens].view(shape2)
                else:
                    dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            # Paged tail cache replaces the transient _DECODE_TAIL ring. Group
            # the tail group's token-granular slot_mapping per-request, mirroring
            # dec_slot / dec_pos, so the kernel gets each request's current-token
            # tail slot (block * kpool + pos % kpool).
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if tail_meta is None or tail_kv_cache is None:
                dec_tail_slot = None
            elif not use_uniform:
                dec_tail_slot = _scatter_decode_tokens_by_request(
                    tail_meta.slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
            # The compress kernel writes the raw fp8 cache (not the quant view);
            # pass the underlying kv_cache, not kv_cache_quant_view.
            if dec_tail_slot is not None:
                # Single batched launch over [num_requests, next_n] replaces the
                # per-token sequential loop. The kernel iterates each request's
                # tokens in position order internally, preserving the
                # pool-completion read-after-stash dependency that the loop
                # provided. Inputs are already grouped per request (uniform:
                # view; non-uniform: _scatter_decode_tokens_by_request padded to
                # [B, lmax]) — no per-token .contiguous() copies needed.
                kpool_decode_update_and_maybe_write_cache_batched(
                    kv_cache_raw,
                    tail_kv_cache,
                    dec_tail_slot,
                    dec_k,
                    dec_gate,
                    compress_ape,
                    dec_slot,
                    dec_pos,
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
            padded_weights = weights[:num_decode_tokens]
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        logits = fp8_fp4_paged_mqa_logits(
            (padded_q_quant_cast, padded_q_scale),
            kv_cache,
            padded_weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        dec_seq = None
        fused_tail = False
        if index_kpool > 1:
            # Token-granular seq_len per row. Computed here rather than after the
            # top-k because the tail bias needs it too; expand_pools_and_append_tail
            # reuses this exact tensor below.
            if (
                positions is not None
                and _INDEXER_DECODE_FUSED
                and not decode_metadata.requires_padding
                and positions.dtype in (torch.int64, torch.int32)
                and logits.dim() == 2
                and logits.stride(1) == 1
            ):
                # deneb fork (34차): seq_len + the tail-pool bias in ONE launch
                # (the stock pair below is 10 aten kernels), bit-identical.
                dec_seq = indexer_tail_select_fused(
                    positions[:num_padded_tokens], logits, num_rows, index_kpool,
                    _always_select_tail(),
                )
                fused_tail = True
                if "capture" not in _FUSED_ANNOUNCED:
                    _FUSED_ANNOUNCED.add("capture")
                    logger.warning(
                        "[indexer-fused] tail-select fused: rows=%d pools=%d "
                        "(one launch for seq_len + tail bias; expanded indices "
                        "written straight into the top-k buffer)%s",
                        num_rows, logits.shape[1],
                        " [capture]" if torch.cuda.is_current_stream_capturing() else "")
            elif positions is not None:
                dec_seq = _decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[:num_rows]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            # Only on the token-granular path: the fallback below reads
            # decode_metadata.seq_lens, which is already pool-granular, and
            # dividing it again would aim the bias at the wrong pool.
            if positions is not None and not fused_tail and _always_select_tail():
                # The appended tail covers only [pool_len*kpool, seq_len), which
                # is empty when seq_len % kpool == 0. Pin the last completed pool
                # so recency never depends on winning the top-k.
                _force_tail_pool_into_logits(logits, dec_seq, index_kpool)
            pool_topk = (
                _inactive_pool_topk_scratch(
                    topk_indices_buffer,
                    hidden_states.shape[0],
                    num_rows,
                    select_k,
                )
                if current_platform.is_cuda()
                else None
            )
            if pool_topk is None:
                pool_topk = torch.full(
                    (num_rows, select_k),
                    -1,
                    dtype=torch.int32,
                    device=logits.device,
                )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # SM121/GB10 (48 SMs, 99KB smem): persistent_topk oversubscribes past
        # ~24K ctx and its FilteredTopK fallback needs 128KB smem -> hard raise.
        # Route small-SM parts to top_k_per_row_decode instead.
        if (
            current_platform.is_cuda()
            and select_k in (512, 1024, 2048)
            and torch.cuda.get_device_properties(0).multi_processor_count >= 78
        ):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_dst,
                topk_workspace,
                select_k,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )

        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            pool_ids = (
                topk_dst
                if current_platform.is_cuda()
                else topk_dst.to(torch.int64)
            )
            # NOTE: decode_metadata.seq_lens is POOL-granular (divided by
            # compress_ratio in the indexer metadata builder) because it feeds
            # the paged-MQA logits. The fused kernel needs TOKEN-granular seq_len,
            # so recover it from the decode tokens' positions (pos == seq_len-1).
            # Using the compressed seq_lens yields dec_seq=0 for seq_len<kpool
            # -> empty topk -> the sparse MLA attends to nothing -> decode
            # degradation. The row->token mapping must follow the PADDED
            # [B, next_n] layout on non-uniform batches (see
            # _decode_topk_seq_lens).
            assert dec_seq is not None  # hoisted above the top-k
            if fused_tail:
                # deneb fork (34차): expand straight into the persistent buffer
                # (no fresh tensor, no copy); None = a layout the direct write
                # cannot take, stock below
                written = expand_pools_and_append_tail_into(
                    pool_ids, dec_seq, index_kpool, topk_indices_buffer
                )
                if written is not None:
                    return topk_indices_buffer
            out = expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


def sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool",
    op_func=sparse_attn_indexer_kpool,
    # The indexer writes the index-K cache in place (prefill k-cache insert +
    # kpool decode write), so kv_cache must be declared as mutated — otherwise
    # under full-graph compile dynamo assumes it is unchanged across the
    # indexer→MLA boundary and the MLA reads stale/misaligned KV. The paged tail
    # cache is likewise written in place (prefill tail scatter + decode stash).
    mutates_args=["topk_indices_buffer", "kv_cache", "tail_kv_cache"],
    fake_impl=sparse_attn_indexer_kpool_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(
                hidden_states,
                q_quant,
                k,
                weights,
                gate_score=gate_score,
                compress_ape=compress_ape,
                index_kpool=index_kpool,
                positions=positions,
            )
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer_kpool(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
