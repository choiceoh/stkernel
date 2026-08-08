# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.import_utils import has_deep_gemm
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

logger = init_logger(__name__)

# deneb probe (temporary, measurement only): per-chunk query-row stats for the
# indexer sequence-parallel gate.
_DENEB_SP_STATS = {"n": 0, "ge256": 0, "rows": 0, "min": 1 << 30, "max": 0}

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_B12X_PAGED_INDEX_PAGE_SIZE = 64
_B12X_PAGED_INDEX_HEAD_DIM = 128
_B12X_PAGED_INDEX_SCALE_BYTES = 4
_B12X_PAGED_INDEX_PAGE_WIDTH = _B12X_PAGED_INDEX_PAGE_SIZE * (
    _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES
)
_B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT = 32768
_B12X_PAGED_INDEX_TILE_BLOCK_Q = 32
_B12X_PAGED_INDEX_TILE_BLOCK_K = 512
_B12X_CONTIGUOUS_PREFILL_BLOCK_K = 256
_B12X_CONTIGUOUS_PREFILL512_BLOCK_K = 512
_B12X_CONTIGUOUS_PREFILL512_MIN_Q_ROWS = 1024
_B12X_CONTIGUOUS_PREFILL512_MIN_K_ROWS = 4096
_B12X_CONTIGUOUS_PREFILL512_SUPPORTED_HEADS = (32, 64)
# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32
_B12X_PREFILL_PAGED_ROUTE = "packed_contiguous"


_DCP_QUOTA_TOPK_SCRATCH: dict = {}


def _dcp_quota_topk_scratch(
    rows_cap: int, qk: int, device: "torch.device"
) -> "torch.Tensor":
    """Persistent contiguous int32 [rows_cap, qk] out-buffer for quota topk.

    Allocated on first (eager warmup) use so cudagraph replays see a stable
    address; rows_cap tracks the full topk buffer so one entry serves every
    batch size.
    """
    key = (device, rows_cap, qk)
    buf = _DCP_QUOTA_TOPK_SCRATCH.get(key)
    if buf is None:
        buf = torch.empty((rows_cap, qk), dtype=torch.int32, device=device)
        _DCP_QUOTA_TOPK_SCRATCH[key] = buf
    return buf


def _dcp_decode_gather_v2_requested() -> bool:
    """Doc-1 V2 exact-global-top-k decode gather (successor to Tier-3
    quota). The indexer needs NO layout change for V2: attention consumes
    the ordinary merged global top-k (_merge_b12x_dcp_topk) and gathers
    the owning ranks' native CKV records itself. This helper exists only
    to keep quota force-disabled when both envs are set (V2 preferred; the
    b12x_mla_sparse builder logs the preference and clears the quota
    metadata bit — this env check is belt-and-braces on top of that).
    Same truthiness contract as b12x_mla_sparse._env_flag.
    """
    value = os.environ.get("VLLM_B12X_MLA_DECODE_GATHER_V2")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _dcp_decode_quota_requested() -> bool:
    """Tier-3 sparse decode gather: per-rank LOCAL top-(topk/dcp) at decode,
    no candidate merge; attention gathers the records (fixed shapes)."""
    if _dcp_decode_gather_v2_requested():
        # Mutually exclusive with the V2 decode gather; V2 wins.
        return False
    return os.environ.get("VLLM_B12X_MLA_DECODE_SPARSE_GATHER", "0") == "1"


def _dcp_global_topk_requested() -> bool:
    raw = os.environ.get("VLLM_DCP_GLOBAL_TOPK", "1")
    return raw.lower() in ("1", "true", "yes", "on")


def _use_persistent_topk_decode(topk_tokens: int) -> bool:
    return current_platform.is_cuda() and topk_tokens in (512, 1024, 2048)


def _local_to_global_position(
    local_idx: torch.Tensor, rank: int, world_size: int, interleave: int
) -> torch.Tensor:
    return (
        (local_idx // interleave) * (world_size * interleave)
        + rank * interleave
        + (local_idx % interleave)
    )


def _global_to_local_position(
    global_idx: torch.Tensor, interleave: int, world_size: int
) -> torch.Tensor:
    big = interleave * world_size
    return (global_idx // big) * interleave + (global_idx % interleave)


@triton.jit
def _dcp_pack_topk_candidates_kernel(
    topk_indices_ptr,
    logits_ptr,
    row_starts_ptr,
    packed_ptr,
    topk_stride_0: tl.constexpr,
    topk_stride_1: tl.constexpr,
    logits_stride_0: tl.constexpr,
    logits_stride_1: tl.constexpr,
    packed_stride_0: tl.constexpr,
    packed_stride_1: tl.constexpr,
    packed_stride_2: tl.constexpr,
    logits_width,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    TOPK_TOKENS: tl.constexpr,
    HAS_LOGITS: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOPK_TOKENS

    idx = tl.load(
        topk_indices_ptr + row * topk_stride_0 + offsets * topk_stride_1,
        mask=mask,
        other=-1,
    )
    invalid = idx < 0
    idx_safe = tl.maximum(idx, 0)

    scores = tl.full((BLOCK_K,), -float("inf"), tl.float32)
    if HAS_LOGITS:
        score_idx = idx_safe
        if HAS_ROW_STARTS:
            row_start = tl.load(row_starts_ptr + row)
            score_idx += row_start
        score_idx = tl.minimum(score_idx, logits_width - 1)
        scores = tl.load(
            logits_ptr + row * logits_stride_0 + score_idx * logits_stride_1,
            mask=mask,
            other=-float("inf"),
        )
        scores = scores.to(tl.float32)
        scores = tl.where(invalid, -float("inf"), scores)

    global_pos = (
        (idx_safe // INTERLEAVE) * (WORLD_SIZE * INTERLEAVE)
        + RANK * INTERLEAVE
        + (idx_safe % INTERLEAVE)
    )
    global_pos = tl.where(invalid, -1, global_pos)

    packed_base = packed_ptr + row * packed_stride_0 + offsets * packed_stride_2
    tl.store(packed_base, global_pos, mask=mask)
    tl.store(
        packed_base + packed_stride_1,
        scores.to(tl.int32, bitcast=True),
        mask=mask,
    )


@triton.jit
def _dcp_finalize_topk_remap_kernel(
    all_candidates_ptr,
    selected_ptr,
    topk_indices_ptr,
    candidates_stride_0: tl.constexpr,
    candidates_stride_1: tl.constexpr,
    candidates_stride_2: tl.constexpr,
    selected_stride_0: tl.constexpr,
    selected_stride_1: tl.constexpr,
    topk_stride_0: tl.constexpr,
    topk_stride_1: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    TOPK_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOPK_TOKENS

    selected = tl.load(
        selected_ptr + row * selected_stride_0 + offsets * selected_stride_1,
        mask=mask,
        other=0,
    )
    global_pos = tl.load(
        all_candidates_ptr + row * candidates_stride_0 + selected * candidates_stride_2,
        mask=mask,
        other=-1,
    )

    owner = (global_pos // INTERLEAVE) % WORLD_SIZE
    big = INTERLEAVE * WORLD_SIZE
    local_pos = (global_pos // big) * INTERLEAVE + (global_pos % INTERLEAVE)
    mine = (owner == RANK) & (global_pos >= 0)
    final = tl.where(mine, local_pos, -1)

    tl.store(
        topk_indices_ptr + row * topk_stride_0 + offsets * topk_stride_1,
        final,
        mask=mask,
    )


@triton.jit
def _b12x_dcp_unpack_gathered_candidates_kernel(
    gathered_ptr,
    candidate_indices_ptr,
    candidate_score_bits_ptr,
    ROWS,
    TOPK_TOKENS: tl.constexpr,
    CANDIDATE_WIDTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offsets < CANDIDATE_WIDTH

    rank = offsets // TOPK_TOKENS
    local_offset = offsets - rank * TOPK_TOKENS
    raw_row = rank * ROWS + row
    raw_base = (raw_row * 2) * TOPK_TOKENS + local_offset

    candidate_ids = tl.load(gathered_ptr + raw_base, mask=mask, other=-1)
    candidate_scores = tl.load(
        gathered_ptr + raw_base + TOPK_TOKENS,
        mask=mask,
        other=0,
    )
    out_base = row * CANDIDATE_WIDTH + offsets
    tl.store(candidate_indices_ptr + out_base, candidate_ids, mask=mask)
    tl.store(candidate_score_bits_ptr + out_base, candidate_scores, mask=mask)


def _indexer_sp_group() -> tuple[int, int] | None:
    """(tp_size, tp_rank) when prefill indexer sequence-sharding is enabled.

    VLLM_DSV4_INDEXER_SP=1 shards the prefill indexer scoring+topk by QUERY
    TOKENS across TP ranks (each rank: all heads, 1/tp of the chunk tokens vs
    the full context) and all-gathers the per-token topk indices. Bit-exact:
    identical math, partitioned rows. Decode path untouched.
    """
    if os.environ.get("VLLM_DSV4_INDEXER_SP") != "1":
        return None
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        if tp <= 1:
            return None
        return tp, get_tensor_model_parallel_rank()
    except Exception:
        return None


def _indexer_sp_gather(
    topk_indices_full: torch.Tensor,
    n_chunk: int,
    shard: int,
    off: int,
    end: int,
    tp: int,
    num_shards: int | None = None,
    stride: int = 1,
) -> None:
    """All-gather per-rank contiguous token shards of topk indices.

    Under DCP (stride == dcp_world_size) row shards are owned per DCP
    *group*: every member of a group sends the identical merged slice, so
    reconstruction reads one copy per group, at TP slot ``g * stride``.
    """
    from vllm.distributed import tensor_model_parallel_all_gather

    if num_shards is None:
        num_shards = tp
    local = torch.empty(
        (shard, topk_indices_full.shape[1]),
        dtype=topk_indices_full.dtype,
        device=topk_indices_full.device,
    )
    valid = end - off
    if valid > 0:
        local[:valid] = topk_indices_full[off:end]
    gathered = tensor_model_parallel_all_gather(local, dim=0)
    if stride == 1 and shard * tp == n_chunk:
        topk_indices_full.copy_(gathered)
        return
    for g in range(num_shards):
        o = g * shard
        v = min(shard, n_chunk - o)
        if v > 0:
            src = g * stride * shard
            topk_indices_full[o : o + v].copy_(gathered[src : src + v])


def _use_triton_dcp_remap(topk_indices: torch.Tensor) -> bool:
    return HAS_TRITON and current_platform.is_cuda() and topk_indices.is_cuda


def _dcp_pack_topk_candidates(
    topk_indices: torch.Tensor,
    logits: torch.Tensor | None,
    topk_tokens: int,
    rank: int,
    world_size: int,
    interleave: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack local candidate global positions and score bits for one all-gather."""
    packed = torch.empty(
        (topk_indices.shape[0], 2, topk_tokens),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if topk_indices.numel() == 0:
        return packed

    if _use_triton_dcp_remap(topk_indices):
        block_k = triton.next_power_of_2(topk_tokens)
        num_warps = 4 if block_k <= 256 else 8
        logits_arg = logits if logits is not None else topk_indices
        row_starts_arg = row_starts if row_starts is not None else topk_indices
        _dcp_pack_topk_candidates_kernel[(topk_indices.shape[0],)](
            topk_indices,
            logits_arg,
            row_starts_arg,
            packed,
            topk_indices.stride(0),
            topk_indices.stride(1),
            logits.stride(0) if logits is not None else 0,
            logits.stride(1) if logits is not None else 0,
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            logits.shape[1] if logits is not None else 1,
            RANK=rank,
            WORLD_SIZE=world_size,
            INTERLEAVE=interleave,
            TOPK_TOKENS=topk_tokens,
            HAS_LOGITS=logits is not None,
            HAS_ROW_STARTS=row_starts is not None,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
        return packed

    invalid = topk_indices < 0
    idx_safe = torch.clamp(topk_indices, min=0)
    if logits is None:
        local_scores = torch.full(
            topk_indices.shape,
            float("-inf"),
            dtype=torch.float32,
            device=topk_indices.device,
        )
    else:
        score_idx = idx_safe.to(torch.int64)
        if row_starts is not None:
            score_idx = score_idx + row_starts.to(
                device=score_idx.device, dtype=score_idx.dtype
            ).view(-1, 1)
        score_idx = torch.clamp(score_idx, min=0, max=logits.shape[1] - 1)
        local_scores = torch.gather(logits, 1, score_idx).to(torch.float32)
    local_scores = local_scores.masked_fill(invalid, float("-inf"))

    global_pos = _local_to_global_position(idx_safe, rank, world_size, interleave)
    global_pos = torch.where(invalid, global_pos.new_full((), -1), global_pos)

    packed[:, 0, :].copy_(global_pos.to(torch.int32))
    packed[:, 1, :].copy_(local_scores.contiguous().view(torch.int32))
    return packed


def _dcp_finalize_topk_remap(
    all_candidates: torch.Tensor,
    selected: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    rank: int,
    world_size: int,
    interleave: int,
) -> None:
    """Finalize selected global candidates into rank-local top-k indices."""
    if topk_indices.numel() == 0:
        return

    if _use_triton_dcp_remap(topk_indices):
        block_k = triton.next_power_of_2(topk_tokens)
        num_warps = 4 if block_k <= 256 else 8
        _dcp_finalize_topk_remap_kernel[(topk_indices.shape[0],)](
            all_candidates,
            selected,
            topk_indices,
            all_candidates.stride(0),
            all_candidates.stride(1),
            all_candidates.stride(2),
            selected.stride(0),
            selected.stride(1),
            topk_indices.stride(0),
            topk_indices.stride(1),
            RANK=rank,
            WORLD_SIZE=world_size,
            INTERLEAVE=interleave,
            TOPK_TOKENS=topk_tokens,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
        return

    sel_global = torch.gather(all_candidates[:, 0, :], 1, selected.to(torch.int64))
    owner = (sel_global // interleave) % world_size
    local_of_g = _global_to_local_position(sel_global, interleave, world_size)
    mine = (owner == rank) & (sel_global >= 0)
    final = torch.where(mine, local_of_g, local_of_g.new_full((), -1))
    topk_indices.copy_(final.to(topk_indices.dtype))


def _dcp_all_gather_first_dim_into(
    group,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    """All-gather into caller-owned output, concatenating along dim 0."""
    assert input_tensor.is_contiguous()
    assert output_tensor.is_contiguous()
    assert output_tensor.shape[0] == input_tensor.shape[0] * group.world_size
    assert output_tensor.shape[1:] == input_tensor.shape[1:]

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_gather(output_tensor, input_tensor)
        return

    device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        import torch.distributed as dist

        dist.all_gather_into_tensor(output_tensor, input_tensor, group=device_group)
        return

    gathered = group.all_gather(input_tensor, dim=0)
    output_tensor.copy_(gathered)


def _unpack_b12x_dcp_gathered_candidates(
    gathered_candidates: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_score_bits: torch.Tensor,
    *,
    dcp_world_size: int,
    topk_tokens: int,
) -> None:
    rows = int(candidate_indices.shape[0])
    candidate_width = int(candidate_indices.shape[1])
    if _use_triton_dcp_remap(candidate_indices):
        block_k = 1024
        grid = (rows, triton.cdiv(candidate_width, block_k))
        _b12x_dcp_unpack_gathered_candidates_kernel[grid](
            gathered_candidates,
            candidate_indices,
            candidate_score_bits,
            rows,
            topk_tokens,
            candidate_width,
            BLOCK_K=block_k,
        )
        return

    gathered_view = gathered_candidates.view(dcp_world_size, rows, 2, topk_tokens)
    for rank in range(dcp_world_size):
        start = rank * topk_tokens
        end = start + topk_tokens
        candidate_indices[:, start:end].copy_(gathered_view[rank, :, 0, :])
        candidate_score_bits[:, start:end].copy_(gathered_view[rank, :, 1, :])


def _get_dcp_warmup_params() -> tuple[int, int, int]:
    dcp_world_size = 1
    dcp_rank = 0
    cp_kv_cache_interleave_size = 1
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            parallel_config = vllm_config.parallel_config
            dcp_world_size = int(parallel_config.decode_context_parallel_size)
            cp_kv_cache_interleave_size = int(
                parallel_config.cp_kv_cache_interleave_size
            )
    except Exception:
        pass

    try:
        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        if int(dcp_group.world_size) > 1:
            dcp_world_size = int(dcp_group.world_size)
            dcp_rank = int(dcp_group.rank_in_group)
    except Exception:
        pass

    return dcp_world_size, dcp_rank, cp_kv_cache_interleave_size


def _sync_dcp_warmup() -> None:
    """Finish one-time DCP warmup on every rank before graph warmup proceeds."""
    if current_platform.is_cuda():
        torch.cuda.synchronize()

    try:
        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        if int(dcp_group.world_size) <= 1:
            return
        dcp_group.barrier()
    except Exception:
        return
    finally:
        if current_platform.is_cuda():
            torch.cuda.synchronize()


def _prewarm_b12x_dcp_topk_merge(
    *,
    q_rows: int,
    topk_tokens: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    device: torch.device,
) -> None:
    if dcp_world_size <= 1 or topk_tokens <= 0:
        return

    topk = int(topk_tokens)
    # Graph warmup later exercises both small decode rows and prefill chunks.
    # Use the real runtime merge path here, including DCP all-gather, so Triton
    # and b12x row-topk kernels are loaded before rank-sensitive attention
    # collectives begin.
    warmup_rows = (1, 2, 4, max(1, int(q_rows)))
    seen_rows: set[int] = set()
    base_ids = torch.arange(topk, dtype=torch.int32, device=device)
    for rows in warmup_rows:
        rows = int(rows)
        if rows in seen_rows:
            continue
        seen_rows.add(rows)
        topk_indices = base_ids.expand(rows, topk).clone()
        topk_scores = torch.zeros((rows, topk), dtype=torch.float32, device=device)
        _merge_b12x_dcp_topk(
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            topk_tokens=topk,
            dcp_world_size=int(dcp_world_size),
            dcp_rank=int(dcp_rank),
            cp_kv_cache_interleave_size=int(cp_kv_cache_interleave_size),
        )
        _sync_dcp_warmup()


def _dcp_global_topk_remap(
    topk_indices: torch.Tensor,
    logits: torch.Tensor | None,
    topk_tokens: int,
    rank: int,
    world_size: int,
    interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Convert rank-local DCP top-k into one shared global top-k in place."""
    if world_size <= 1 or topk_indices.numel() == 0:
        return

    from vllm.distributed.parallel_state import get_dcp_group

    candidates = _dcp_pack_topk_candidates(
        topk_indices,
        logits,
        topk_tokens,
        rank,
        world_size,
        interleave,
        row_starts=row_starts,
    )
    all_candidates = get_dcp_group().all_gather(candidates.contiguous(), dim=2)
    all_scores = all_candidates[:, 1, :].view(torch.float32)
    _, selected = torch.topk(all_scores, topk_tokens, dim=1)
    _dcp_finalize_topk_remap(
        all_candidates,
        selected,
        topk_indices,
        topk_tokens,
        rank,
        world_size,
        interleave,
    )


def _assert_b12x_prefill_paged_route(obj: object, *, owner: str) -> None:
    route = getattr(obj, "route", None)
    if route is None:
        route = getattr(getattr(obj, "layout", None), "route", None)
    if route != _B12X_PREFILL_PAGED_ROUTE:
        raise RuntimeError(
            "B12X sparse prefill expected the b12x planner to resolve the paged "
            f"source to {_B12X_PREFILL_PAGED_ROUTE!r}, got {route!r} from {owner}."
        )


def _get_b12x_indexer_paged_supertile_k() -> int:
    raw = os.environ.get("B12X_PAGED_INDEX_SUPERTILE_K")
    tokens = _B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT if raw is None else int(raw)
    tokens = max(tokens, _B12X_PAGED_INDEX_TILE_BLOCK_K)
    return (
        (tokens + _B12X_PAGED_INDEX_TILE_BLOCK_K - 1)
        // _B12X_PAGED_INDEX_TILE_BLOCK_K
        * _B12X_PAGED_INDEX_TILE_BLOCK_K
    )


def _get_b12x_paged_indexer_profile_q_rows(q_rows: int) -> int:
    """Return the largest q chunk the real prefill chunker can hand to b12x."""
    max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
    tile_k = _get_b12x_indexer_paged_supertile_k()
    max_q_rows = max(1, max_logits_elems // max(1, tile_k))
    return min(max(1, int(q_rows)), max_q_rows)


def _get_b12x_paged_indexer_profile_k_rows(
    max_model_len: int,
    total_seq_lens: int,
) -> int:
    known_k_rows = max(int(max_model_len), int(total_seq_lens), 0)
    if known_k_rows > 0:
        return known_k_rows
    return _get_b12x_indexer_paged_supertile_k()


def _get_b12x_paged_indexer_profile_warmup_k_rows(profile_k_rows: int) -> int:
    raw = os.environ.get("B12X_PAGED_INDEX_PROFILE_WARMUP_K")
    cap = _get_b12x_indexer_paged_supertile_k() if raw is None else int(raw)
    cap = max(cap, _B12X_PAGED_INDEX_TILE_BLOCK_K)
    cap = (
        (cap + _B12X_PAGED_INDEX_TILE_BLOCK_K - 1)
        // _B12X_PAGED_INDEX_TILE_BLOCK_K
        * _B12X_PAGED_INDEX_TILE_BLOCK_K
    )
    return min(max(1, int(profile_k_rows)), cap)


def _get_dcp_local_k_rows(
    k_rows: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> int:
    if dcp_world_size <= 1:
        return int(k_rows)
    interleave = max(1, int(cp_kv_cache_interleave_size))
    world_size = int(dcp_world_size)
    rank = int(dcp_rank)
    base = int(k_rows) // interleave // world_size * interleave
    remainder = int(k_rows) - base * world_size
    rank_remainder = min(max(remainder - rank * interleave, 0), interleave)
    return base + rank_remainder


def _b12x_profile_rows_or_empty(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    rows = max(1, int(rows))
    if int(tensor.shape[0]) >= rows:
        return tensor[:rows].contiguous()
    return torch.empty(
        (rows, *tuple(tensor.shape[1:])),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _b12x_profile_weights_2d(
    weights: torch.Tensor,
    *,
    q_rows: int,
    num_q_heads: int,
    device: torch.device,
) -> torch.Tensor:
    q_rows = max(1, int(q_rows))
    num_q_heads = int(num_q_heads)
    if (
        weights.ndim == 2
        and int(weights.shape[0]) >= q_rows
        and int(weights.shape[1]) == num_q_heads
        and weights.dtype == torch.float32
        and weights.device == device
    ):
        return weights[:q_rows].contiguous()
    return torch.empty(
        (q_rows, num_q_heads),
        dtype=torch.float32,
        device=device,
    )


def _b12x_sparse_indexer_requested(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)

    if envs.VLLM_USE_B12X_SPARSE_INDEXER:
        return True

    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False

    backend = vllm_config.attention_config.backend
    if isinstance(backend, str):
        return backend == "B12X_MLA_SPARSE"
    return getattr(backend, "name", None) == "B12X_MLA_SPARSE"


def _ensure_b12x_sparse_indexer_supported() -> None:
    if not current_platform.is_cuda():
        raise RuntimeError("B12X sparse indexer/top-k requires CUDA.")
    if not current_platform.is_device_capability_family(120):
        raise RuntimeError(
            "B12X sparse indexer/top-k currently requires an SM120 GPU."
        )


def _use_b12x_sparse_indexer(enabled: bool | None = None) -> bool:
    if not _b12x_sparse_indexer_requested(enabled):
        return False
    _ensure_b12x_sparse_indexer_supported()
    return True


def use_b12x_sparse_indexer(enabled: bool | None = None) -> bool:
    return _use_b12x_sparse_indexer(enabled)


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


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


def _flatten_b12x_paged_index_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    expected_shape_tail = (
        _B12X_PAGED_INDEX_PAGE_SIZE,
        _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES,
    )

    if kv_cache.ndim != 3 or kv_cache.dtype != torch.uint8:
        raise RuntimeError(
            "b12x paged indexer cache must be rank-3 uint8 with "
            f"shape [num_blocks, {expected_shape_tail[0]}, "
            f"{expected_shape_tail[1]}], got shape={tuple(kv_cache.shape)} "
            f"dtype={kv_cache.dtype}."
        )
    if tuple(kv_cache.shape[1:]) != expected_shape_tail:
        raise RuntimeError(
            "b12x paged indexer cache has an unsupported shape, "
            f"got {tuple(kv_cache.shape)}; expected tail {expected_shape_tail}."
        )
    if kv_cache.stride(1) != expected_shape_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "b12x paged indexer cache has an unsupported layout, "
            f"shape={tuple(kv_cache.shape)} stride={tuple(kv_cache.stride())}; "
            f"expected inner strides ({expected_shape_tail[1]}, 1)."
        )

    return kv_cache.as_strided(
        (int(kv_cache.shape[0]), _B12X_PAGED_INDEX_PAGE_WIDTH),
        (int(kv_cache.stride(0)), 1),
    )


def _run_b12x_paged_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    topk_scores: torch.Tensor | None = None,
    active_width: torch.Tensor | None = None,
    shared_page_table: bool = False,
) -> torch.Tensor:
    """Run b12x paged indexer top-k with caller-owned scratch.

    b12x sizes scratch from indexer K-cache rows/pages. For compressed sources
    such as C4, the metadata builder has already converted model context tokens
    to indexer K rows before this call. ``active_width`` is the builder-computed
    live K-row window (a metadata tensor, not an in-kernel reduction); when
    None, b12x falls back to the capacity cap.
    """
    from b12x.attention.indexer import (
        INDEXER_SOURCE_LAYOUT_PAGED,
        PAGED_INDEX_PAGE_SIZE,
        B12XIndexerScratchCaps,
        index_topk_fp8,
        plan_indexer_scratch,
    )

    if int(PAGED_INDEX_PAGE_SIZE) != _B12X_PAGED_INDEX_PAGE_SIZE:
        raise RuntimeError(
            "b12x paged indexer page-size contract changed, got "
            f"{PAGED_INDEX_PAGE_SIZE}; expected "
            f"{_B12X_PAGED_INDEX_PAGE_SIZE}."
        )

    index_k_cache = _flatten_b12x_paged_index_cache(kv_cache)
    expected_num_q_heads = int(q_fp8.shape[1])
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=q_fp8.device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=expected_num_q_heads,
            max_q_rows=int(q_fp8.shape[0]),
            max_page_table_width=int(block_table.shape[1]),
            topk=int(topk_tokens),
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=bool(shared_page_table),
        )
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(plan, owner="scratch plan")
    scratch = current_workspace_manager().get_simultaneous(
        *plan.shapes_and_dtypes()
    )
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
        shared_page_table=shared_page_table,
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(binding, owner="binding")
    return index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
        page_size=PAGED_INDEX_PAGE_SIZE,
        expected_num_q_heads=expected_num_q_heads,
        out_indices=topk_indices,
        out_scores=topk_scores,
    )


def _merge_b12x_dcp_topk(
    *,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor | None,
    topk_tokens: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> None:
    if dcp_world_size <= 1 or topk_indices.numel() == 0:
        return
    if topk_scores is None:
        raise RuntimeError(
            "B12X sparse indexer DCP requires a topk_scores_buffer for "
            "cross-rank candidate merge."
        )

    from b12x.attention.indexer.tiled_topk import run_row_topk

    from vllm.distributed.parallel_state import get_dcp_group
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_dcp_local_topk_to_global,
        triton_gather_topk_ids_by_position,
    )

    triton_convert_dcp_local_topk_to_global(
        topk_indices,
        topk_scores,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )

    if not topk_scores.is_contiguous():
        raise RuntimeError("B12X sparse indexer DCP requires contiguous topk scores.")

    rows = int(topk_indices.shape[0])
    candidate_width = int(dcp_world_size * topk_tokens)
    (
        candidates,
        gathered_candidates,
        candidate_indices,
        candidate_score_bits,
        candidate_lengths,
    ) = current_workspace_manager().get_simultaneous(
        ((rows, 2, int(topk_tokens)), torch.int32),
        ((rows * int(dcp_world_size), 2, int(topk_tokens)), torch.int32),
        ((rows, candidate_width), torch.int32),
        ((rows, candidate_width), torch.int32),
        ((rows,), torch.int32),
    )
    candidates[:, 0, :].copy_(topk_indices)
    candidates[:, 1, :].copy_(topk_scores.view(torch.int32))

    dcp_group = get_dcp_group()
    _dcp_all_gather_first_dim_into(
        dcp_group,
        candidates,
        gathered_candidates,
    )
    _unpack_b12x_dcp_gathered_candidates(
        gathered_candidates,
        candidate_indices,
        candidate_score_bits,
        dcp_world_size=dcp_world_size,
        topk_tokens=int(topk_tokens),
    )
    candidate_scores = candidate_score_bits.view(torch.float32)
    candidate_lengths.fill_(candidate_width)
    run_row_topk(
        row_logits=candidate_scores,
        lengths=candidate_lengths,
        topk=int(topk_tokens),
        output_values=topk_scores,
        output_indices=topk_indices,
    )
    triton_gather_topk_ids_by_position(
        candidate_indices,
        topk_indices,
        topk_indices,
    )


def _convert_b12x_dcp_local_topk_to_global(
    *,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor | None,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> None:
    if dcp_world_size <= 1 or topk_indices.numel() == 0:
        return
    if topk_scores is None:
        raise RuntimeError(
            "B12X sparse indexer DCP requires a topk_scores_buffer for "
            "local-to-global candidate conversion."
        )

    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_dcp_local_topk_to_global,
    )

    triton_convert_dcp_local_topk_to_global(
        topk_indices,
        topk_scores,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )


def _prewarm_b12x_paged_indexer_prefill(
    *,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_tokens: int,
    profile_q_rows: int,
    profile_k_rows: int,
    page_table_k_rows: int | None = None,
) -> None:
    q_rows = max(1, int(profile_q_rows))
    k_rows = max(1, int(profile_k_rows))
    page_table_rows = (
        k_rows if page_table_k_rows is None else max(1, int(page_table_k_rows))
    )
    num_q_heads = int(q_quant.shape[1])
    page_table_width = max(
        1,
        (page_table_rows + _B12X_PAGED_INDEX_PAGE_SIZE - 1)
        // _B12X_PAGED_INDEX_PAGE_SIZE,
    )
    q_warm = _b12x_profile_rows_or_empty(q_quant, q_rows)
    weights_warm = _b12x_profile_weights_2d(
        weights,
        q_rows=q_rows,
        num_q_heads=num_q_heads,
        device=q_quant.device,
    )
    seq_lens = torch.full(
        (q_rows,),
        k_rows,
        dtype=torch.int32,
        device=q_quant.device,
    )
    block_table = torch.zeros(
        (1, page_table_width),
        dtype=torch.int32,
        device=q_quant.device,
    ).expand(q_rows, page_table_width)
    kv_cache_warm = kv_cache
    if int(kv_cache_warm.shape[0]) <= 0:
        kv_cache_warm = torch.empty(
            (
                1,
                _B12X_PAGED_INDEX_PAGE_SIZE,
                _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES,
            ),
            dtype=torch.uint8,
            device=q_quant.device,
        )
    topk_indices = torch.empty(
        (q_rows, int(topk_tokens)),
        dtype=torch.int32,
        device=q_quant.device,
    )
    _run_b12x_paged_topk(
        q_fp8=q_warm,
        weights=weights_warm,
        kv_cache=kv_cache_warm,
        seq_lens=seq_lens,
        block_table=block_table,
        schedule_metadata=None,
        topk_indices=topk_indices,
        topk_tokens=int(topk_tokens),
        shared_page_table=True,
    )


def _prewarm_b12x_contiguous_prefill_variants(
    *,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    profile_q_rows: int,
) -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None or q_quant.dtype != fp8_dtype:
        return
    if q_quant.device.type != "cuda":
        return
    if q_quant.ndim != 3 or int(q_quant.shape[2]) != _B12X_PAGED_INDEX_HEAD_DIM:
        return

    try:
        from b12x.attention.indexer.contiguous_kernel import (
            run_contiguous_logits_kernel,
        )
        from b12x.attention.indexer.tiled_topk import run_tiled_topk
    except (AttributeError, ImportError, ModuleNotFoundError):
        return

    q_rows = max(1, int(profile_q_rows))
    num_q_heads = int(q_quant.shape[1])
    topk = int(topk_tokens)
    q_warm = _b12x_profile_rows_or_empty(q_quant, q_rows)
    weights_warm = _b12x_profile_weights_2d(
        weights,
        q_rows=q_rows,
        num_q_heads=num_q_heads,
        device=q_quant.device,
    )

    for block_k in (
        _B12X_CONTIGUOUS_PREFILL_BLOCK_K,
        _B12X_CONTIGUOUS_PREFILL512_BLOCK_K,
    ):
        if (
            block_k == _B12X_CONTIGUOUS_PREFILL512_BLOCK_K
            and (
                q_rows < _B12X_CONTIGUOUS_PREFILL512_MIN_Q_ROWS
                or num_q_heads not in _B12X_CONTIGUOUS_PREFILL512_SUPPORTED_HEADS
            )
        ):
            continue
        k_rows = max(block_k, topk, 1)
        if block_k == _B12X_CONTIGUOUS_PREFILL512_BLOCK_K:
            k_rows = max(k_rows, _B12X_CONTIGUOUS_PREFILL512_MIN_K_ROWS)
        k_rows = (k_rows + block_k - 1) // block_k * block_k
        k_quant = torch.empty(
            (k_rows, _B12X_PAGED_INDEX_HEAD_DIM),
            dtype=fp8_dtype,
            device=q_quant.device,
        )
        k_scale = torch.empty((k_rows,), dtype=torch.float32, device=q_quant.device)
        k_start = torch.zeros((q_rows,), dtype=torch.int32, device=q_quant.device)
        k_end = torch.full(
            (q_rows,),
            k_rows,
            dtype=torch.int32,
            device=q_quant.device,
        )
        num_q_tiles = (
            q_rows + _B12X_PAGED_INDEX_TILE_BLOCK_Q - 1
        ) // _B12X_PAGED_INDEX_TILE_BLOCK_Q
        num_k_tiles = (k_rows + block_k - 1) // block_k
        tile_logits = torch.empty(
            (
                num_q_tiles
                * num_k_tiles
                * _B12X_PAGED_INDEX_TILE_BLOCK_Q
                * block_k,
            ),
            dtype=torch.float32,
            device=q_quant.device,
        )
        output_values = torch.empty(
            (q_rows, topk),
            dtype=torch.float32,
            device=q_quant.device,
        )
        output_indices = torch.empty(
            (q_rows, topk),
            dtype=torch.int32,
            device=q_quant.device,
        )
        run_contiguous_logits_kernel(
            q_fp8=q_warm,
            weights=weights_warm,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
            preinitialize_invalid_logits=True,
            tile_logits=tile_logits,
            tile_k_offset=0,
            tile_num_k_tiles=num_k_tiles,
            prefill_block_k=block_k,
        )
        run_tiled_topk(
            tile_logits=tile_logits,
            k_start=None,
            lengths=k_end,
            topk=topk,
            block_q=_B12X_PAGED_INDEX_TILE_BLOCK_Q,
            block_k=block_k,
            output_values=output_values,
            output_indices=output_indices,
            num_k_tiles=num_k_tiles,
            input_extent=k_rows,
            zero_row_start=True,
        )


# --- Boot-time decode-shape prewarm sweep -----------------------------------
# Decode-step token count T (total decode tokens in a batch, T in
# [1, max_num_seqs * (1 + spec depth)]) is the single variable that selects
# b12x decode-path kernel specializations. A cold CuTe compile of any of them
# during serving stalls every DCP rank behind the compiling one (07-20: a T=4
# fused-indexer compile on a degraded node wedged production for 65 min), so
# every reachable specialization is compiled at engine start instead.
_B12X_DECODE_PREWARM_MAX_T = 15
_B12X_DECODE_PREWARM_DONE: set = set()
# Mirrors B12xMLASparseImpl._prewarm_extend_kernels_once: GLM cache records
# are 656 B/token (fp8_ds_mla) unless a packed-latent cache dtype is used.
_B12X_MLA_KV_RECORD_BYTES = {
    "nvfp4_ds_mla": 368,
    "nf3_ds_mla": 304,
    "nf3bf16_ds_mla": 368,
}


def _b12x_decode_prewarm_requested() -> bool:
    value = os.environ.get("VLLM_B12X_DECODE_PREWARM", "1")
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _b12x_freeze_kernels_requested() -> bool:
    """VLLM_B12X_FREEZE_KERNELS (default OFF): freeze b12x kernel resolution
    after the decode prewarm sweep. The freeze check sits on the cute.compile
    cache-MISS path only (memory/disk cache hits still load), so with a cold
    disk cache the prefill singletons that only compile on the first real
    request would raise KernelResolutionFrozenError and fail that request.
    Keep OFF unless the kernel disk cache is known-complete for this config.
    """
    value = os.environ.get("VLLM_B12X_FREEZE_KERNELS", "0")
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _prewarm_b12x_decode_indexer_shapes(
    *,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
    dcp_world_size: int,
) -> tuple[int, int]:
    """Compile every decode-route paged-indexer specialization for T in
    [1, _B12X_DECODE_PREWARM_MAX_T] through the real _run_b12x_paged_topk
    entry (decode mode, per-row page table).

    T <= FUSED_MAX_ROWS(6) resolves to the fused score+select kernel whose
    build key carries ctas_per_group = min(page_table_width, num_sms // T)
    and the derived merge_threshold -- one kernel per T. T >= 7 resolves to
    the tiled supertile scorer + tiled_topk selector; the production decode
    block table has constant width max_model_len / (page * total CP) (> one
    32768-token supertile under the production config), so the chunk loop
    runs both the is_first=True and is_first=False tiled_topk variants. The
    fabricated page table must match that constant width -- a narrower one
    would compile a different (unreachable) ctas_per_group specialization.
    """
    device = q_quant.device
    num_q_heads = int(q_quant.shape[1])
    topk = int(topk_tokens)
    try:
        from vllm.v1.worker.cp_utils import get_total_cp_world_size

        total_cp = max(1, int(get_total_cp_world_size()))
    except Exception:
        total_cp = max(1, int(dcp_world_size))
    page_width = max(
        1,
        (int(max_model_len) + _B12X_PAGED_INDEX_PAGE_SIZE * total_cp - 1)
        // (_B12X_PAGED_INDEX_PAGE_SIZE * total_cp),
    )
    kv_cache_warm = kv_cache
    if int(kv_cache_warm.shape[0]) <= 0:
        kv_cache_warm = torch.empty(
            (
                1,
                _B12X_PAGED_INDEX_PAGE_SIZE,
                _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES,
            ),
            dtype=torch.uint8,
            device=device,
        )
    warmed = 0
    failed = 0
    for rows in range(1, _B12X_DECODE_PREWARM_MAX_T + 1):
        try:
            q_warm = _b12x_profile_rows_or_empty(q_quant, rows)
            weights_warm = _b12x_profile_weights_2d(
                weights,
                q_rows=rows,
                num_q_heads=num_q_heads,
                device=device,
            )
            seq_lens = torch.full(
                (rows,),
                _B12X_PAGED_INDEX_PAGE_SIZE,
                dtype=torch.int32,
                device=device,
            )
            block_table = torch.zeros(
                (rows, page_width), dtype=torch.int32, device=device
            )
            topk_indices = torch.empty(
                (rows, topk), dtype=torch.int32, device=device
            )
            topk_scores = None
            if dcp_world_size > 1:
                # Production decode under DCP always requests scores for the
                # cross-rank merge; mirror it (does not change build keys).
                topk_scores = torch.empty(
                    (rows, topk), dtype=torch.float32, device=device
                )
            _run_b12x_paged_topk(
                q_fp8=q_warm,
                weights=weights_warm,
                kv_cache=kv_cache_warm,
                seq_lens=seq_lens,
                block_table=block_table,
                schedule_metadata=None,
                topk_indices=topk_indices,
                topk_tokens=topk,
                topk_scores=topk_scores,
            )
            warmed += 1
            logger.info(
                "B12X decode prewarm: indexer %s route warmed for T=%d "
                "(page_table_width=%d).",
                "fused" if rows <= 6 else "tiled",
                rows,
                page_width,
            )
        except Exception:
            failed += 1
            logger.warning(
                "B12X decode prewarm: indexer warm failed for T=%d.",
                rows,
                exc_info=True,
            )
    return warmed, failed


def _find_b12x_sparse_mla_impl() -> object | None:
    """Any B12xMLASparseImpl from the registered attention layers (all GLM
    layers share dims/plans, so one impl warms the process-wide caches)."""
    try:
        layers = get_forward_context().no_compile_layers
    except Exception:
        return None
    for layer in layers.values():
        impl = getattr(layer, "impl", None)
        if impl is None:
            continue
        if hasattr(impl, "_sparse_mla_decode_forward") and hasattr(
            impl, "_decode_plan"
        ):
            return impl
    return None


def _prewarm_b12x_mla_decode_shapes() -> tuple[int, int]:
    """Compile every attention.mla.sm120.decode split-plan specialization.

    The decode grid kernel and its split merge key on (num_splits,
    chunks_per_split); rows is a dynamic (unkeyed) dim, so one launch per
    plan suffices. Production pins forced_num_splits=_num_splits_cap on
    every decode call, so that plan is warmed first. The wave-balanced
    heuristic plans (reachable only if the pin is removed or overridden) are
    enumerated with plan_unified_decode_splits over T in [1, MAX_T] for both
    the actual head-block geometry (per_token_head = T * h_blocks) and
    h_blocks=1, then deduped and each launched once through the impl's real
    decode forward (which also compiles that plan's merge specialization).
    """
    impl = _find_b12x_sparse_mla_impl()
    if impl is None:
        logger.info(
            "B12X decode prewarm: no B12X sparse-MLA attention impl "
            "registered; skipping MLA decode sweep."
        )
        return 0, 0
    try:
        from b12x.attention.mla.kernel import plan_unified_decode_splits

        device = impl.device
        if device.type != "cuda":
            return 0, 0
        kernel_heads = int(impl._kernel_num_heads)
        q_head_dim = int(impl.q_head_dim)
        topk = int(impl.topk_tokens)
        block_size = int(impl.block_size)
        num_splits_cap = int(impl._num_splits_cap)
        rows_cap = max(
            1, min(_B12X_DECODE_PREWARM_MAX_T, int(impl._decode_max_rows))
        )
        record_bytes = _B12X_MLA_KV_RECORD_BYTES.get(
            str(impl.kv_cache_dtype), 656
        )
        sm_count = int(
            torch.cuda.get_device_properties(device).multi_processor_count
        )
        # hpb=16 head blocks (GLM heads never take the heads==8 native-H8 arm
        # at TP4xDCP4: the DCP query gather yields the full head set).
        h_blocks = kernel_heads // 16 + (1 if kernel_heads % 16 else 0)
    except Exception:
        logger.warning(
            "B12X decode prewarm: unable to derive MLA decode geometry; "
            "skipping MLA decode sweep.",
            exc_info=True,
        )
        return 0, 0

    plans: list[tuple[int, int, int]] = []
    seen_plans: set[tuple[int, int]] = set()

    def _note_plan(num_splits: int, chunks_per_split: int, rows: int) -> None:
        key = (int(num_splits), int(chunks_per_split))
        if key in seen_plans:
            return
        seen_plans.add(key)
        plans.append((key[0], key[1], max(1, min(int(rows), rows_cap))))

    _, num_splits, chunks_per_split = plan_unified_decode_splits(
        topk=topk,
        max_chunks=num_splits_cap,
        forced_num_splits=num_splits_cap,
        num_tokens=1,
        h_blocks=h_blocks,
        sm_count=sm_count,
    )
    _note_plan(num_splits, chunks_per_split, 1)
    for rows in range(1, _B12X_DECODE_PREWARM_MAX_T + 1):
        for hb in (h_blocks, 1):
            _, num_splits, chunks_per_split = plan_unified_decode_splits(
                topk=topk,
                max_chunks=num_splits_cap,
                forced_num_splits=None,
                num_tokens=rows,
                h_blocks=hb,
                sm_count=sm_count,
            )
            _note_plan(num_splits, chunks_per_split, rows)

    try:
        # Same spec the impl pre-touched at init, so no workspace growth; the
        # kernel only writes inside this arena while q/indices are plain
        # allocations (no aliasing).
        (scratch_storage,) = current_workspace_manager().get_simultaneous(
            ((int(impl._scratch_nbytes),), torch.uint8)
        )
    except Exception:
        logger.warning(
            "B12X decode prewarm: MLA decode scratch borrow failed; "
            "skipping MLA decode sweep.",
            exc_info=True,
        )
        return 0, len(plans)
    # One zero page is enough: prewarm top-k indices all point at slot zero.
    kv_cache_warm = torch.zeros(
        (1, block_size, record_bytes), dtype=torch.uint8, device=device
    )
    warmed = 0
    failed = 0
    for num_splits, chunks_per_split, rows in plans:
        try:
            q = torch.zeros(
                (rows, kernel_heads, q_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            selected_indices = torch.zeros(
                (rows, topk), dtype=torch.int32, device=device
            )
            cache_seqlens = torch.full(
                (1,), block_size, dtype=torch.int32, device=device
            )
            nsa_cache_seqlens = torch.ones(
                (rows,), dtype=torch.int32, device=device
            )
            binding = impl._decode_plan.bind(
                scratch=scratch_storage,
                q=q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            forward_kwargs = dict(
                binding=binding,
                kv_cache=kv_cache_warm,
                sm_scale=impl.scale,
                v_head_dim=impl.kv_lora_rank,
                forced_num_splits=int(num_splits),
                **impl._b12x_nvfp4_kwargs,
            )
            if impl.need_to_return_lse_for_decode:
                impl._sparse_mla_decode_forward(
                    return_lse=True, lse_scale="natural", **forward_kwargs
                )
            else:
                impl._sparse_mla_decode_forward(**forward_kwargs)
            warmed += 1
            logger.info(
                "B12X decode prewarm: MLA decode plan warmed "
                "(num_splits=%d, chunks_per_split=%d, rows=%d).",
                num_splits,
                chunks_per_split,
                rows,
            )
        except Exception:
            failed += 1
            logger.warning(
                "B12X decode prewarm: MLA decode plan failed "
                "(num_splits=%d, chunks_per_split=%d, rows=%d).",
                num_splits,
                chunks_per_split,
                rows,
                exc_info=True,
            )
    return warmed, failed


def _prewarm_b12x_moe_decode_shapes() -> tuple[int, int]:
    """Warm the b12x dynamic-MoE decode launches for tokens 1..8 via the
    stock warmup helper (the w4a16 small-m direct kernel caches on the exact
    token count m and has no disk cache, so it recompiles every boot without
    this). The helper resolves each request to the smallest dynamic-backend
    token count >= the request, so some requests coalesce; a repeat is a
    cheap relaunch of an already-compiled signature. Pure per-rank compute,
    no collectives.
    """
    try:
        from vllm.model_executor.layers.fused_moe.b12x_moe import (
            warmup_b12x_moe_dynamic,
        )

        layers = get_forward_context().no_compile_layers
    except Exception:
        logger.warning(
            "B12X decode prewarm: MoE warmup helper unavailable; skipping "
            "MoE sweep.",
            exc_info=True,
        )
        return 0, 0
    # warmup_b12x_moe_dynamic walks Module.modules() looking for MoE runners
    # (.routed_experts); the model itself is not reachable from here, so hang
    # the registered runner layers off a throwaway container module.
    shim = torch.nn.Module()
    count = 0
    for layer in layers.values():
        if getattr(layer, "routed_experts", None) is not None and isinstance(
            layer, torch.nn.Module
        ):
            shim.add_module(f"moe_{count}", layer)
            count += 1
    if count == 0:
        logger.info(
            "B12X decode prewarm: no MoE runner layers registered; skipping "
            "MoE sweep."
        )
        return 0, 0
    warmed = 0
    failed = 0
    for tokens in range(1, 9):
        try:
            warmup_b12x_moe_dynamic(shim, tokens=tokens)
            warmed += 1
            logger.info(
                "B12X decode prewarm: MoE dynamic warm pass done for "
                "tokens=%d.",
                tokens,
            )
        except Exception:
            failed += 1
            logger.warning(
                "B12X decode prewarm: MoE dynamic warm failed for tokens=%d.",
                tokens,
                exc_info=True,
            )
    return warmed, failed


def _prewarm_b12x_decode_shapes(
    *,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
    dcp_world_size: int,
) -> None:
    """Boot-time sweep compiling every reachable decode-path b12x kernel
    specialization (VLLM_B12X_DECODE_PREWARM, default on) so no cold CuTe
    compile can happen mid-serving. Runs once per process from the profile
    branch, after the existing prefill prewarms. Every rank executes the same
    sequence with _sync_dcp_warmup() between phases; the sweep itself issues
    no collectives (the decode DCP top-k merge is already covered by
    _prewarm_b12x_dcp_topk_merge), so a per-item failure on one rank cannot
    strand another rank in a collective. Each item is try/except'd, logged,
    and skipped on failure -- the sweep must never break a boot.
    """
    if not _b12x_decode_prewarm_requested():
        return
    if q_quant.device.type != "cuda":
        return
    key = (q_quant.device.index, int(topk_tokens))
    if key in _B12X_DECODE_PREWARM_DONE:
        return
    _B12X_DECODE_PREWARM_DONE.add(key)

    warmed = 0
    failed = 0
    try:
        warmed, failed = _prewarm_b12x_decode_indexer_shapes(
            q_quant=q_quant,
            weights=weights,
            kv_cache=kv_cache,
            topk_tokens=topk_tokens,
            max_model_len=max_model_len,
            dcp_world_size=dcp_world_size,
        )
    except Exception:
        failed += 1
        logger.warning(
            "B12X decode prewarm: indexer phase failed.", exc_info=True
        )
    _sync_dcp_warmup()

    try:
        w, f = _prewarm_b12x_mla_decode_shapes()
        warmed += w
        failed += f
    except Exception:
        failed += 1
        logger.warning(
            "B12X decode prewarm: MLA decode phase failed.", exc_info=True
        )
    _sync_dcp_warmup()

    try:
        w, f = _prewarm_b12x_moe_decode_shapes()
        warmed += w
        failed += f
    except Exception:
        failed += 1
        logger.warning(
            "B12X decode prewarm: MoE phase failed.", exc_info=True
        )
    _sync_dcp_warmup()

    logger.info(
        "B12X decode prewarm sweep complete: %d item(s) warmed, %d failed.",
        warmed,
        failed,
    )

    if _b12x_freeze_kernels_requested():
        try:
            from b12x.cute.runtime_control import freeze_kernel_resolution

            freeze_kernel_resolution(
                reason="VLLM_B12X_FREEZE_KERNELS after decode prewarm sweep"
            )
            logger.warning(
                "B12X kernel resolution FROZEN after the decode prewarm "
                "sweep: any kernel shape missing from the memory/disk cache "
                "now raises KernelResolutionFrozenError instead of compiling "
                "(first-real-request prefill singletons included)."
            )
        except Exception:
            logger.warning(
                "B12X decode prewarm: freeze_kernel_resolution failed.",
                exc_info=True,
            )


def _reserve_b12x_paged_indexer_scratch(
    *,
    q_rows: int,
    num_q_heads: int,
    topk_tokens: int,
    total_k_rows: int,
    device: torch.device,
    shared_page_table: bool = False,
) -> None:
    from b12x.attention.indexer import (
        INDEXER_SOURCE_LAYOUT_PAGED,
        PAGED_INDEX_PAGE_SIZE,
        B12XIndexerScratchCaps,
        plan_indexer_scratch,
    )

    page_table_width = max(
        1,
        (max(1, int(total_k_rows)) + int(PAGED_INDEX_PAGE_SIZE) - 1)
        // int(PAGED_INDEX_PAGE_SIZE),
    )
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=int(num_q_heads),
            max_q_rows=max(1, int(q_rows)),
            max_page_table_width=page_table_width,
            topk=int(topk_tokens),
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=bool(shared_page_table),
        )
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(plan, owner="scratch reservation plan")
    current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())


def _sparse_attn_indexer_impl(
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
    use_b12x_sparse_indexer: bool = False,
    topk_scores_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        run_profile_work = (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and forward_context.batch_descriptor is not None
        )
        if not run_profile_work:
            return sparse_attn_indexer_fake(
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
                use_b12x_sparse_indexer,
            )

        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        if _b12x_sparse_indexer_requested(use_b12x_sparse_indexer):
            _ensure_b12x_sparse_indexer_supported()
            profile_q_rows = _get_b12x_paged_indexer_profile_q_rows(
                int(q_quant.shape[0])
            )
            profile_k_rows = _get_b12x_paged_indexer_profile_k_rows(
                max_model_len=max_model_len,
                total_seq_lens=total_seq_lens,
            )
            warmup_k_rows = _get_b12x_paged_indexer_profile_warmup_k_rows(
                profile_k_rows
            )
            _reserve_b12x_paged_indexer_scratch(
                q_rows=profile_q_rows,
                num_q_heads=int(q_quant.shape[1]),
                topk_tokens=int(topk_tokens),
                total_k_rows=profile_k_rows,
                device=q_quant.device,
                shared_page_table=False,
            )
            _reserve_b12x_paged_indexer_scratch(
                q_rows=profile_q_rows,
                num_q_heads=int(q_quant.shape[1]),
                topk_tokens=int(topk_tokens),
                total_k_rows=profile_k_rows,
                device=q_quant.device,
                shared_page_table=True,
            )
            (
                dcp_world_size,
                dcp_rank,
                cp_kv_cache_interleave_size,
            ) = _get_dcp_warmup_params()
            _prewarm_b12x_paged_indexer_prefill(
                q_quant=q_quant,
                weights=weights,
                kv_cache=kv_cache,
                topk_tokens=int(topk_tokens),
                profile_q_rows=profile_q_rows,
                profile_k_rows=warmup_k_rows,
                page_table_k_rows=profile_k_rows,
            )
            if dcp_world_size > 1:
                dcp_profile_k_rows = _get_dcp_local_k_rows(
                    profile_k_rows,
                    dcp_world_size,
                    dcp_rank,
                    cp_kv_cache_interleave_size,
                )
                dcp_warmup_k_rows = _get_b12x_paged_indexer_profile_warmup_k_rows(
                    dcp_profile_k_rows
                )
                if dcp_warmup_k_rows != warmup_k_rows:
                    _prewarm_b12x_paged_indexer_prefill(
                        q_quant=q_quant,
                        weights=weights,
                        kv_cache=kv_cache,
                        topk_tokens=int(topk_tokens),
                        profile_q_rows=profile_q_rows,
                        profile_k_rows=dcp_warmup_k_rows,
                        page_table_k_rows=dcp_profile_k_rows,
                    )
            _prewarm_b12x_dcp_topk_merge(
                q_rows=profile_q_rows,
                topk_tokens=int(topk_tokens),
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                device=q_quant.device,
            )
            _prewarm_b12x_contiguous_prefill_variants(
                q_quant=q_quant,
                weights=weights,
                topk_tokens=int(topk_tokens),
                profile_q_rows=profile_q_rows,
            )
            _prewarm_b12x_decode_shapes(
                q_quant=q_quant,
                weights=weights,
                kv_cache=kv_cache,
                topk_tokens=int(topk_tokens),
                max_model_len=int(max_model_len),
                dcp_world_size=dcp_world_size,
            )
        else:
            # Reserve workspace for indexer during profiling run.
            current_workspace_manager().get_simultaneous(
                values_spec, scales_spec, ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8)
            )

            # Dummy allocation to simulate peak logits tensor memory during
            # inference. The B12X path above streams one supertile at a time and
            # has already reserved its fixed scratch via the workspace manager.
            # FP8 elements so elements == bytes.
            max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            _ = torch.empty(
                max_logits_elems, dtype=torch.uint8, device=hidden_states.device
            )

        return sparse_attn_indexer_fake(
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
            use_b12x_sparse_indexer,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
    dcp_world_size = attn_metadata_narrowed.dcp_world_size
    dcp_rank = attn_metadata_narrowed.dcp_rank
    cp_kv_cache_interleave_size = attn_metadata_narrowed.cp_interleave_size
    dcp_global_topk = _dcp_global_topk_requested() and dcp_world_size > 1

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        use_b12x_indexer = _use_b12x_sparse_indexer(use_b12x_sparse_indexer)
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable b12x sparse indexer."
            )
        if not use_b12x_indexer:
            workspace_manager = current_workspace_manager()
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                values_spec,
                scales_spec,
            )
        for chunk in prefill_metadata.chunks:
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            weights_slice = weights[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if chunk.total_seq_lens <= 0:
                topk_indices.fill_(-1)
                if dcp_global_topk:
                    if use_b12x_indexer:
                        if topk_scores_buffer is None:
                            raise RuntimeError(
                                "B12X sparse indexer DCP requires topk_scores_buffer."
                            )
                        topk_scores = topk_scores_buffer[
                            chunk.token_start : chunk.token_end, :topk_tokens
                        ]
                        topk_scores.fill_(-float("inf"))
                        _merge_b12x_dcp_topk(
                            topk_indices=topk_indices,
                            topk_scores=topk_scores,
                            topk_tokens=topk_tokens,
                            dcp_world_size=dcp_world_size,
                            dcp_rank=dcp_rank,
                            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                        )
                        topk_indices.masked_fill_(topk_scores == -float("inf"), -1)
                    else:
                        _dcp_global_topk_remap(
                            topk_indices,
                            None,
                            topk_tokens,
                            dcp_rank,
                            dcp_world_size,
                            cp_kv_cache_interleave_size,
                        )
                continue

            # Sequence-parallel prefill indexer: this rank scores only its
            # contiguous token sub-range of the chunk (against the full
            # context); results are all-gathered below.
            _sp = _indexer_sp_group()
            _n_chunk = int(q_slice.shape[0])
            _cu_ks = chunk.cu_seqlen_ks
            _cu_ke = chunk.cu_seqlen_ke
            _sp_shards = 0
            _sp_stride = 1
            # deneb probe: is the >=256 gate actually letting SP engage? Histogram
            # the per-chunk query-row count so we can see how often prefill chunks
            # fall below the threshold (long contexts fill the KV budget with few
            # query rows, which would silently disable indexer sequence-parallel).
            _DENEB_SP_STATS["n"] += 1
            _DENEB_SP_STATS["rows"] += _n_chunk
            _DENEB_SP_STATS["min"] = min(_DENEB_SP_STATS["min"], _n_chunk)
            _DENEB_SP_STATS["max"] = max(_DENEB_SP_STATS["max"], _n_chunk)
            if _n_chunk >= 256:
                _DENEB_SP_STATS["ge256"] += 1
            if _DENEB_SP_STATS["n"] % 200 == 0:
                s = _DENEB_SP_STATS
                logger.info(
                    "[deneb-sp-probe] chunks=%d ge256=%d (%.1f%%) rows avg=%.1f "
                    "min=%d max=%d sp_env=%s",
                    s["n"], s["ge256"], 100.0 * s["ge256"] / s["n"],
                    s["rows"] / s["n"], s["min"], s["max"], _sp is not None,
                )
            if _sp is not None and _n_chunk >= 256:
                _tp, _tp_rank = _sp
                if dcp_world_size == 1:
                    _sp_shards, _sp_idx = _tp, _tp_rank
                elif _tp % dcp_world_size == 0 and _tp > dcp_world_size:
                    # DCP-aware SP: shard rows by DCP *group* (consecutive
                    # grouping per parallel_state). Members of one group score
                    # the same rows against complementary KV shards; the
                    # existing DCP merge below then yields the global top-k
                    # for that row range, and the TP-wide gather reads one
                    # copy per group. Removes the tp/dcp-fold indexer
                    # scoring redundancy under DCP prefill.
                    _sp_shards = _tp // dcp_world_size
                    _sp_idx = _tp_rank // dcp_world_size
                    _sp_stride = dcp_world_size
            _sp_active = _sp_shards > 1
            if _sp_active:
                _shard = -(-_n_chunk // _sp_shards)
                _off = min(_sp_idx * _shard, _n_chunk)
                _sp_end = min(_off + _shard, _n_chunk)
                _topk_indices_full = topk_indices
                q_slice = q_slice[_off:_sp_end]
                if q_scale_slice is not None:
                    q_scale_slice = q_scale_slice[_off:_sp_end]
                weights_slice = weights_slice[_off:_sp_end]
                topk_indices = _topk_indices_full[_off:_sp_end]
                _cu_ks = _cu_ks[_off:_sp_end]
                _cu_ke = _cu_ke[_off:_sp_end]

            if use_b12x_indexer:
                if chunk.num_reqs != 1:
                    raise RuntimeError(
                        "B12X sparse prefill requires single-request chunks so "
                        "the page table can be row-shared without packing."
                    )
                row_has_no_kv = _cu_ke <= _cu_ks
                seq_lens = torch.where(
                    row_has_no_kv,
                    torch.zeros_like(_cu_ks),
                    _cu_ke - _cu_ks,
                )
                # Our fork's chunk.total_seq_lens is already the dcp-LOCAL
                # total (indexer.py builds compressed_seq_lens via
                # get_dcp_local_seq_lens under DCP).
                local_k_rows = int(chunk.total_seq_lens)
                # Do not expose stale capacity-only page-table columns to the
                # packed B12X gather. One page is retained for zero-K ranks.
                active_page_width = max(
                    1,
                    (local_k_rows + _B12X_PAGED_INDEX_PAGE_SIZE - 1)
                    // _B12X_PAGED_INDEX_PAGE_SIZE,
                )
                active_page_width = min(
                    active_page_width, int(chunk.block_table.shape[1])
                )
                block_table = chunk.block_table[:1, :active_page_width].expand(
                    int(q_slice.shape[0]),
                    active_page_width,
                )
                topk_scores = None
                if dcp_world_size > 1:
                    if topk_scores_buffer is None:
                        raise RuntimeError(
                            "B12X sparse indexer DCP requires topk_scores_buffer."
                        )
                    topk_scores = topk_scores_buffer[
                        chunk.token_start : chunk.token_end, :topk_tokens
                    ]
                    if _sp_active:
                        topk_scores = topk_scores[_off:_sp_end]
                _run_b12x_paged_topk(
                    q_fp8=q_slice.contiguous(),
                    weights=weights_slice.contiguous(),
                    kv_cache=kv_cache,
                    seq_lens=seq_lens,
                    block_table=block_table,
                    schedule_metadata=None,
                    topk_indices=topk_indices,
                    topk_tokens=topk_tokens,
                    topk_scores=topk_scores,
                    shared_page_table=True,
                )
                if dcp_global_topk:
                    _merge_b12x_dcp_topk(
                        topk_indices=topk_indices,
                        topk_scores=topk_scores,
                        topk_tokens=topk_tokens,
                        dcp_world_size=dcp_world_size,
                        dcp_rank=dcp_rank,
                        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                    )
                else:
                    _convert_b12x_dcp_local_topk_to_global(
                        topk_indices=topk_indices,
                        topk_scores=topk_scores,
                        dcp_world_size=dcp_world_size,
                        dcp_rank=dcp_rank,
                        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                    )
                topk_indices.masked_fill_(row_has_no_kv[:, None], -1)
                if _sp_active:
                    _indexer_sp_gather(
                        _topk_indices_full, _n_chunk, _shard, _off, _sp_end, _tp,
                        num_shards=_sp_shards, stride=_sp_stride,
                    )
                continue

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
            if current_platform.is_xpu():
                if q_scale_slice is not None:
                    raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                    q_slice_cast,
                    k_quant_cast,
                    k_scale_cast,
                    weights_slice,
                    _cu_ks,
                    _cu_ke,
                )
            else:
                from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights_slice,
                    _cu_ks,
                    _cu_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            ops.top_k_per_row_prefill(
                logits,
                _cu_ks,
                _cu_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
            if dcp_global_topk:
                _dcp_global_topk_remap(
                    topk_indices,
                    logits,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=_cu_ks,
                )
            if _sp_active:
                _indexer_sp_gather(
                    _topk_indices_full, _n_chunk, _shard, _off, _sp_end, _tp,
                    num_shards=_sp_shards, stride=_sp_stride,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        use_b12x_indexer = _use_b12x_sparse_indexer(use_b12x_sparse_indexer)
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable b12x sparse indexer."
            )

        b12x_seq_lens = decode_metadata.seq_lens
        b12x_block_table = decode_metadata.block_table
        if b12x_seq_lens.dim() == 2:
            b12x_batch_size, b12x_next_n = b12x_seq_lens.shape
            if num_decode_tokens == b12x_batch_size * b12x_next_n:
                b12x_seq_lens = b12x_seq_lens.reshape(-1).contiguous()
                b12x_block_table = b12x_block_table.repeat_interleave(
                    b12x_next_n, dim=0
                ).contiguous()
        b12x_decode_supported = (
            use_b12x_indexer
            and not decode_metadata.requires_padding
            and b12x_seq_lens.dim() == 1
        )
        if use_b12x_indexer and (
            decode_metadata.requires_padding or b12x_seq_lens.dim() != 1
        ):
            raise RuntimeError(
                "b12x sparse indexer decode requires an unpadded rank-1 "
                "seq_lens contract after native-spec normalization; refusing "
                "to fall back to DeepGEMM. "
                f"requires_padding={decode_metadata.requires_padding}, "
                f"seq_lens_shape={tuple(decode_metadata.seq_lens.shape)}, "
                f"normalized_seq_lens_shape={tuple(b12x_seq_lens.shape)}, "
                f"num_decode_tokens={num_decode_tokens}."
            )

        if b12x_decode_supported:
            # Prefix slice of an already-contiguous buffer stays contiguous
            # (b12x_seq_lens/b12x_block_table are normalized contiguous upstream),
            # so .contiguous() here was a guaranteed no-op per decoded token.
            seq_lens = b12x_seq_lens[:num_decode_tokens]
            block_table = b12x_block_table[:num_decode_tokens]
            # Quota engages iff the b12x attention metadata bit says so — the
            # bit is False for the MTP draft group (single-layer) and for
            # non-verify batches, so both consumers always agree on layout.
            _quota = False
            if _dcp_decode_quota_requested() and dcp_world_size > 1:
                for _md in attn_metadata.values():
                    if hasattr(_md, "dcp_quota_eligible"):
                        _quota = bool(_md.dcp_quota_eligible)
                        break
            _qk = topk_tokens // dcp_world_size if _quota else topk_tokens
            topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
            _quota_out = None
            if _quota:
                # Per-shard quota: keep LOCAL top-(topk/dcp) physical slots in
                # the leading columns; no merge, no global conversion. The
                # attention side gathers the records (fixed-shape collective).
                # index_topk_fp8 demands a contiguous out buffer, and a column
                # slice of the wide topk buffer is strided — select into a
                # persistent scratch and copy into the leading columns.
                topk_indices[:, _qk:].fill_(-1)
                _quota_out = _dcp_quota_topk_scratch(
                    topk_indices_buffer.shape[0], _qk, topk_indices.device
                )[:num_decode_tokens]
            topk_scores = None
            if dcp_world_size > 1 and not _quota:
                if topk_scores_buffer is None:
                    raise RuntimeError(
                        "B12X sparse indexer DCP requires topk_scores_buffer."
                    )
                topk_scores = topk_scores_buffer[:num_decode_tokens, :topk_tokens]
            # b12x consumes indexer K-row metadata. DSV4/C4 seq_lens and
            # active_width have already been compressed by the metadata builder;
            # GLM passes one K row per context token.
            _run_b12x_paged_topk(
                q_fp8=q_quant[:num_decode_tokens].contiguous(),
                weights=weights[:num_decode_tokens].contiguous(),
                kv_cache=kv_cache,
                seq_lens=seq_lens,
                block_table=block_table,
                schedule_metadata=decode_metadata.schedule_metadata,
                active_width=decode_metadata.active_width,
                topk_indices=_quota_out if _quota else topk_indices,
                topk_tokens=_qk if _quota else topk_tokens,
                topk_scores=topk_scores,
            )
            if _quota:
                topk_indices[:, :_qk].copy_(_quota_out)
                return topk_indices_buffer
            if dcp_global_topk:
                _merge_b12x_dcp_topk(
                    topk_indices=topk_indices,
                    topk_scores=topk_scores,
                    topk_tokens=topk_tokens,
                    dcp_world_size=dcp_world_size,
                    dcp_rank=dcp_rank,
                    cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                )
            else:
                _convert_b12x_dcp_local_topk_to_global(
                    topk_indices=topk_indices,
                    topk_scores=topk_scores,
                    dcp_world_size=dcp_world_size,
                    dcp_rank=dcp_rank,
                    cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                )
            return topk_indices_buffer

        schedule_metadata = decode_metadata.schedule_metadata
        if schedule_metadata is None:
            raise RuntimeError(
                "DeepGEMM/XPU sparse indexer decode requires schedule metadata; "
                "enable VLLM_USE_B12X_SPARSE_INDEXER for the b12x path or check "
                "the indexer metadata builder."
            )

        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
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
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                schedule_metadata,
                max_model_len,
            )
        else:
            from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            # The cluster-cooperative kernel is an SM90 path. On SM120 it can
            # launch-fail during DS4 sparse-indexer warmup; use persistent_topk.
            and current_platform.is_device_capability_family(90)
        )
        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif _use_persistent_topk_decode(topk_tokens):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                decode_metadata.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if dcp_global_topk:
            _dcp_global_topk_remap(
                topk_indices,
                logits,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


# ---------------------------------------------------------------------------
# VLLM_B12X_INDEXER_STREAM: run the eager prefill indexer body on ONE
# process-global dedicated CUDA stream so its score GEMMs and its DCP top-k
# merge all-gather (still on get_dcp_group()) overlap the main-stream
# q_b_proj GEMM + rope (mla.py issues the indexer op before q_b_proj when
# the env is set). Consumers join via the per-layer event registry below:
#   * b12x_mla_sparse.forward_mqa waits the layer's event at entry (before
#     the offset-0 workspace borrow, the first topk_indices consumer, and
#     the sync CKV gather's get_dcp_group() collective), and
#   * b12x_mla_sparse.dcp_prefill_ckv_gather_eligible drains ALL pending
#     events whenever it denies the gather, because the non-gather DCP
#     route runs a get_dcp_group() all-gather on the main stream BEFORE
#     forward_mqa (two collectives on one communicator must never overlap
#     across streams).
# Default off -> byte-identical inline execution. Stream launch only for
# eager, prefill-only invocations: capture, decode, mixed prefill+decode,
# profile, and dummy runs all stay inline on the main stream.
# ---------------------------------------------------------------------------

_INDEXER_STREAM_ENV = "VLLM_B12X_INDEXER_STREAM"
# ONE global side stream, created lazily on first use (NOT per layer).
_indexer_stream: "torch.cuda.Stream | None" = None
# extract_layer_index(layer name) -> event recorded on the side stream after
# the full indexer body (including its DCP merge all-gather). The consumer
# resolves the same integer via B12xMLASparseImpl._resolve_layer_index.
_indexer_stream_events: dict[int, "torch.cuda.Event"] = {}

# ---------------------------------------------------------------------------
# DCP COMM TOKEN (VLLM_B12X_KV_STREAM, step 2): single-communicator
# discipline for get_dcp_group() collectives issued from SIDE streams. The
# indexer's top-k merge (indexer stream, this module) and the CKV sync
# gather (KV stream, b12x_mla_sparse) share ONE communicator — a dedicated
# gather communicator is banned (>1.3GB NCCL buffers at boot peak) — and two
# collectives on one communicator must never be in flight concurrently on
# different streams. Every side-stream launcher of a dcp_group collective:
#   1. ACQUIRES: its stream waits the previous holder's event, then
#   2. RELEASES: hands the token to an event recorded on its own stream
#      after its collective, making itself the holder.
# All launches happen on the single python model thread, so the chain is
# linear and needs no host lock: each holder waited its predecessor, so
# waiting the newest token transitively covers every earlier side-stream
# collective. Main-stream dcp_group collectives are covered separately: the
# b12x gather-eligibility deny path drains both event registries AND
# acquires this token on the main stream before any such collective runs.
# ---------------------------------------------------------------------------
_dcp_comm_token: "torch.cuda.Event | None" = None


def dcp_comm_token_acquire() -> None:
    """Order the CURRENT stream after the last side-stream dcp collective.

    No-op until a side stream has released a token, so env-off callers pay
    only a None check.
    """
    if _dcp_comm_token is not None:
        torch.cuda.current_stream().wait_event(_dcp_comm_token)


def dcp_comm_token_release(event: "torch.cuda.Event") -> None:
    """Hand the token to ``event`` (already recorded on its issuing stream)."""
    global _dcp_comm_token
    _dcp_comm_token = event


def _indexer_stream_requested() -> bool:
    # Same truthiness contract as b12x_mla_sparse._env_flag.
    value = os.getenv(_INDEXER_STREAM_ENV)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _get_indexer_stream(device: "torch.device") -> "torch.cuda.Stream":
    global _indexer_stream
    if _indexer_stream is None:
        _indexer_stream = torch.cuda.Stream(device=device)
    return _indexer_stream


def wait_indexer_stream_events(layer_idx: "int | None" = None) -> None:
    """Make the CURRENT stream wait on pending indexer-stream events.

    ``layer_idx=None`` drains every pending event (the conservative safety
    join used before any main-stream collective on the indexer merge's
    communicator). Entries are cleared once waited. No-op when nothing is
    pending, so decode / env-off callers pay only a dict truthiness check.
    """
    if not _indexer_stream_events:
        return
    if layer_idx is None:
        events = list(_indexer_stream_events.values())
        _indexer_stream_events.clear()
    else:
        event = _indexer_stream_events.pop(layer_idx, None)
        if event is None:
            return
        events = [event]
    current = torch.cuda.current_stream()
    for event in events:
        current.wait_event(event)


def _indexer_stream_layer_idx(k_cache_prefix: LayerNameType) -> "int | None":
    """Registry key for a side-stream launch, or None to stay inline."""
    if not _indexer_stream_requested():
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        # Dummy/profile run: keep warmup and scratch reservation on main.
        return None
    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if (
        not isinstance(metadata, DeepseekV32IndexerMetadata)
        or metadata.num_prefills <= 0
        or metadata.num_decodes > 0
    ):
        # Prefill-only batches only: decode consumers never consult the
        # event registry, so decode (and mixed) invocations stay inline.
        return None
    from vllm.model_executor.models.utils import extract_layer_index

    try:
        return extract_layer_index(_resolve_layer_name(k_cache_prefix))
    except (ValueError, AssertionError, IndexError):
        return None


def sparse_attn_indexer(
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
    use_b12x_sparse_indexer: bool = False,
    topk_scores_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    layer_idx = _indexer_stream_layer_idx(k_cache_prefix)
    if layer_idx is None:
        return _sparse_attn_indexer_impl(
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
            use_b12x_sparse_indexer,
            topk_scores_buffer,
        )
    main_stream = torch.cuda.current_stream()
    stream = _get_indexer_stream(q_quant.device)
    # The side stream must observe the q/kv projections already enqueued on
    # the main stream before scoring against them.
    stream.wait_stream(main_stream)
    with torch.cuda.stream(stream):
        # SINGLE-COMM DISCIPLINE (dcp comm token): the body's DCP top-k
        # merge all-gather runs on get_dcp_group(); with VLLM_B12X_KV_STREAM
        # a CKV gather collective on the SAME communicator may still be in
        # flight on the KV stream. Acquire before the body; the recorded
        # per-layer event below doubles as the released token (it covers
        # the merge, which the body issues before its final top-k writes).
        dcp_comm_token_acquire()
        out = _sparse_attn_indexer_impl(
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
            use_b12x_sparse_indexer,
            topk_scores_buffer,
        )
        event = torch.cuda.Event()
        event.record(stream)
    # Allocator guard: these main-stream-allocated inputs may be freed (and
    # their blocks reused by main-stream kernels, e.g. inductor buffer reuse
    # after the op's call site) before the side stream finishes reading them.
    # record_stream defers block reuse until the side stream catches up. The
    # persistent tensors (kv cache, topk buffers) need no guard.
    for tensor in (hidden_states, q_quant, q_scale, k, weights):
        if isinstance(tensor, torch.Tensor):
            tensor.record_stream(stream)
    # Overwrite is safe: an unclaimed older event for this layer was recorded
    # earlier on the SAME stream, so waiting on the new event covers it.
    _indexer_stream_events[layer_idx] = event
    dcp_comm_token_release(event)
    return out


def sparse_attn_indexer_fake(
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
    use_b12x_sparse_indexer: bool = False,
    topk_scores_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    del topk_scores_buffer
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer", "topk_scores_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
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
        topk_scores_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_scores_buffer = topk_scores_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.use_b12x_sparse_indexer = use_b12x_sparse_indexer()
        if self.use_b12x_sparse_indexer:
            if self.use_fp4_cache:
                raise RuntimeError(
                    "B12X sparse indexer/top-k requires the FP8 paged index "
                    "cache; disable use_fp4_indexer_cache."
                )
        elif current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
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
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
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
            self.use_b12x_sparse_indexer,
            self.topk_scores_buffer,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

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
