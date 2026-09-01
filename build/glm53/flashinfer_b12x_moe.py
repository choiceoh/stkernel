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


def b12x_ep_kernel_expert_count(
    num_local_experts: int, use_ep: bool, no_dummy: bool = False
) -> int:
    """Experts visible to the selected b12x kernel path.

    The default EP path removes remote slots before the direct top-k=1
    call, so its weights stay at exactly ``num_local_experts``. The rollback
    path keeps the historical extra dummy row for ``wrapper.run``. The fused
    kernel is never told ``num_local != num_experts`` — that path is
    flashinfer #3383 (weight_E vs state_E, then illegal address).
    """
    return num_local_experts + int(use_ep and not no_dummy)


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



def read_b12x_ep_bool(name, default, env_get=os.environ.get) -> bool:
    """Read one EP boolean once, rejecting ambiguous spellings.

    These flags select kernels and must not silently change meaning because of
    a typo. Raising during expert construction also keeps the environment read
    out of CUDA graph capture and replay.
    """
    raw = env_get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{name} must be one of 0/1, false/true, no/yes, off/on; "
        f"got {raw!r}"
    )


def read_b12x_ep_exact_bool(name, default=False, env_get=os.environ.get) -> bool:
    """Read an experimental 0/1 latch; aliases and typos fail at setup."""
    raw = env_get(name)
    if raw is None:
        return bool(default)
    value = str(raw)
    if value in ("0", "1"):
        return value == "1"
    raise ValueError(f"{name} must be exactly 0 or 1; got {raw!r}")


def b12x_ep_mode_from_env(
    env_get=os.environ.get,
) -> tuple[bool, bool, bool, bool]:
    """Return latched EP mode and the two mutually-exclusive experiments."""
    no_dummy = read_b12x_ep_bool(
        "VLLM_B12X_EP_NO_DUMMY", True, env_get=env_get
    )
    disable_micro = read_b12x_ep_bool(
        "VLLM_B12X_EP_DISABLE_MICRO", False, env_get=env_get
    )
    zero_micro = read_b12x_ep_exact_bool(
        "VLLM_B12X_EP_ZERO_WEIGHT_MICRO", False, env_get=env_get
    )
    stock_topk_micro = read_b12x_ep_exact_bool(
        "VLLM_B12X_EP_STOCK_TOPK_MICRO", False, env_get=env_get
    )
    if zero_micro and stock_topk_micro:
        raise RuntimeError(
            "VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1 and "
            "VLLM_B12X_EP_STOCK_TOPK_MICRO=1 are mutually exclusive."
        )
    if (zero_micro or stock_topk_micro) and (not no_dummy or disable_micro):
        experiment = (
            "VLLM_B12X_EP_ZERO_WEIGHT_MICRO"
            if zero_micro
            else "VLLM_B12X_EP_STOCK_TOPK_MICRO"
        )
        raise RuntimeError(
            f"{experiment}=1 requires the local-only "
            "VLLM_B12X_EP_NO_DUMMY=1 path and "
            "VLLM_B12X_EP_DISABLE_MICRO=0."
        )
    if no_dummy and disable_micro:
        raise RuntimeError(
            "VLLM_B12X_EP_DISABLE_MICRO=1 is incompatible with the fixed "
            "no-dummy path (VLLM_B12X_EP_NO_DUMMY=1, the default). "
            "For the plain-static diagnostic, also set "
            "VLLM_B12X_EP_NO_DUMMY=0."
        )
    return no_dummy, disable_micro, zero_micro, stock_topk_micro


def require_b12x_ep_micro_limit(micro_max_tokens) -> int:
    """Verify that an eight-row fixed slice still selects the micro kernel."""
    try:
        limit = int(micro_max_tokens)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "the fixed B12x EP no-dummy path cannot verify FlashInfer's "
            f"_MICRO_MAX_TOKENS={micro_max_tokens!r}"
        ) from exc
    if limit < B12X_EP_FIXED_MICRO_MAX_PAIRS:
        raise RuntimeError(
            "the fixed B12x EP no-dummy path requires FlashInfer "
            f"_MICRO_MAX_TOKENS >= {B12X_EP_FIXED_MICRO_MAX_PAIRS}; "
            f"got {limit}. Refusing a silent static-kernel fallback."
        )
    return limit


# The opt-in native-top-k lane must remain under both of FlashInfer's stock
# micro cutovers. GLM top-k=8 makes the 40-routed-row limit the tighter one.
B12X_EP_STOCK_TOPK_MICRO_MAX_TOKENS = 8
B12X_EP_STOCK_TOPK_MICRO_MAX_ROUTED_ROWS = 40
B12X_EP_STOCK_TOPK_MICRO_TOKEN_COUNTS = (8, 16, 32)
B12X_EP_STOCK_TOPK_MICRO_TOPK = 8
B12X_EP_STOCK_TOPK_MICRO_EXPERTS = 72


def require_b12x_ep_micro_limits(
    micro_max_tokens, micro_max_routed_rows
) -> tuple[int, int]:
    """Verify the two private FlashInfer limits this fixed path relies on."""
    values = (
        (
            "_MICRO_MAX_TOKENS",
            micro_max_tokens,
            B12X_EP_STOCK_TOPK_MICRO_MAX_TOKENS,
        ),
        (
            "_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK",
            micro_max_routed_rows,
            B12X_EP_STOCK_TOPK_MICRO_MAX_ROUTED_ROWS,
        ),
    )
    parsed = []
    for name, raw, required in values:
        try:
            limit = int(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "the fixed B12x EP no-dummy path cannot verify FlashInfer's "
                f"{name}={raw!r}"
            ) from exc
        if limit < required:
            raise RuntimeError(
                "the fixed B12x EP no-dummy path requires FlashInfer "
                f"{name} >= {required}; got {limit}. Refusing a silent "
                "static-kernel fallback."
            )
        parsed.append(limit)
    return parsed[0], parsed[1]


def b12x_ep_stock_topk_token_limit(
    max_num_tokens,
    top_k,
    micro_max_tokens=B12X_EP_STOCK_TOPK_MICRO_MAX_TOKENS,
    micro_max_routed_rows=B12X_EP_STOCK_TOPK_MICRO_MAX_ROUTED_ROWS,
) -> int:
    """Token rows per stock top-k call while both cutovers remain true."""
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    routed_limit = int(micro_max_routed_rows) // top_k
    if routed_limit <= 0:
        raise RuntimeError(
            f"top_k={top_k} exceeds the micro routed-row capacity "
            f"{micro_max_routed_rows}"
        )
    workspace_limit = max(int(max_num_tokens or 0), 1)
    token_limit = min(workspace_limit, int(micro_max_tokens), routed_limit)
    if token_limit <= 0:
        raise RuntimeError("the fixed B12x EP token limit is not positive")
    return token_limit


def b12x_ep_stock_topk_token_spans(num_tokens, token_limit):
    """Balanced, shape-only token spans with no one-row tail when avoidable."""
    num_tokens = int(num_tokens)
    token_limit = int(token_limit)
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if token_limit <= 0:
        raise ValueError(f"token_limit must be positive, got {token_limit}")
    if num_tokens == 0:
        return ()
    num_calls = (num_tokens + token_limit - 1) // token_limit
    base, extra = divmod(num_tokens, num_calls)
    spans = []
    lo = 0
    for call_index in range(num_calls):
        size = base + int(call_index < extra)
        hi = lo + size
        spans.append((lo, hi))
        lo = hi
    return tuple(spans)


def b12x_ep_stock_topk_micro_chunks(
    num_tokens,
    top_k,
    num_local_experts,
    *,
    enabled,
):
    """Return the exact opt-in stock-micro plan, else fail closed."""
    tokens = int(num_tokens)
    top_k = int(top_k)
    if not (
        enabled
        and tokens in B12X_EP_STOCK_TOPK_MICRO_TOKEN_COUNTS
        and top_k == B12X_EP_STOCK_TOPK_MICRO_TOPK
        and int(num_local_experts) == B12X_EP_STOCK_TOPK_MICRO_EXPERTS
    ):
        return ()
    limit = b12x_ep_stock_topk_token_limit(tokens, top_k)
    return b12x_ep_stock_topk_token_spans(tokens, limit)


def require_b12x_ep_stock_topk_micro_dispatch(moe_dispatch):
    """Verify the private stock-dispatch contract before opting in."""
    if not hasattr(moe_dispatch, "_FORCED_BACKEND"):
        raise RuntimeError(
            "VLLM_B12X_EP_STOCK_TOPK_MICRO=1 cannot verify FlashInfer's "
            "_FORCED_BACKEND contract"
        )
    if getattr(moe_dispatch, "_FORCED_BACKEND", None) is not None:
        raise RuntimeError(
            "VLLM_B12X_EP_STOCK_TOPK_MICRO=1 requires FlashInfer's "
            "automatic backend selection (_FORCED_BACKEND must be unset)"
        )
    return require_b12x_ep_micro_limits(
        getattr(moe_dispatch, "_MICRO_MAX_TOKENS", None),
        getattr(
            moe_dispatch,
            "_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK",
            None,
        ),
    )


def disable_b12x_micro_for_ep(no_dummy: bool, disable_micro: bool) -> str:
    """Keep EP off the micro MoE kernels. Returns what it did, for the log.

    `use_direct_micro` already excludes EP by shape -- it requires
    `n <= _DIRECT_MICRO_MAX_N` (512) and EP leaves the intermediate unsharded
    at 2048. Plain `micro` has **no n bound**: it only checks
    `num_tokens <= _MICRO_MAX_TOKENS` (8), and a C=1 decode is exactly 8
    tokens (one sequence x 1+k positions), so EP decode lands there.

    moe_micro_kernel's phase 0 does not initialise `global_to_local_expert` or
    `active_expert_count` (its own module docstring says so), and our EP
    disguise hands the kernel remapped local ids that depend on that table.
    Pinning the token bound to 0 turns both micro variants off, leaving plain
    static -- whose per-expert `token_map` holds a decode batch comfortably
    (256 pairs against 640 rows).

    Default is to LEAVE MICRO ON: it is the only decode-tuned path EP
    qualifies for, and its own docstring says it takes LOCAL expert ids
    from a pre-pass -- the same contract our EP remap already honours.
    The flag is parsed and latched during expert construction. The static
    diagnostic requires both VLLM_B12X_EP_DISABLE_MICRO=1 and
    VLLM_B12X_EP_NO_DUMMY=0; the no-dummy path must remain micro-sliced.
    """
    if not no_dummy and not disable_micro:
        return "micro left on -- EP qualifies for it (no n bound)"
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    except Exception as exc:
        if no_dummy:
            raise RuntimeError(
                "the fixed B12x EP no-dummy path cannot import FlashInfer "
                "moe_dispatch to verify its micro boundary"
            ) from exc
        return f"micro bound untouched ({type(exc).__name__}: {exc})"
    prior = getattr(moe_dispatch, "_MICRO_MAX_TOKENS", None)
    if no_dummy:
        limit = require_b12x_ep_micro_limit(prior)
        return f"micro left on -- verified token bound {limit}"
    if prior in (None, 0):
        return f"micro bound already {prior}"
    moe_dispatch._MICRO_MAX_TOKENS = 0
    return f"micro bound {prior} -> 0 (EP uses plain static)"



# The functional b12x dispatcher keys micro/static on x.shape[0]. Stable
# DFlash verification shapes have 8/16/32 tokens at C=1/2/4; after top-k=8 is
# flattened into top_k=1 pair rows, those become 64/128/256 rows and therefore
# 8/16/32 calls. Eight is the pinned FlashInfer build's _MICRO_MAX_TOKENS
# boundary; an unsliced fixed call selects the static kernel observed to hang.
B12X_EP_FIXED_MICRO_MAX_PAIRS = 8

# The kernel-overlay experiment widens the exact E=72/m=8 shape to 64 routed
# rows by dropping zero-weight sentinels before row materialisation. The stock
# experiment above remains inside the dispatcher's ordinary 40-row cutover.
# The chunker slices at CHUNK_TOKENS exactly, so any positive multiple of it
# is already a valid plan -- the old (8, 16, 32) tuple left C=3 (24 tokens with
# MAX_SEQS=4, 1+k=8) falling back to the pair path for no reason. Bound the
# lane at the compact cutover instead: above it `b12x_ep_should_compact` takes
# over and dropping the remote slots outright is the cheaper shape.
B12X_EP_ZERO_WEIGHT_MICRO_MAX_TOKENS = 80
B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS = 8
B12X_EP_ZERO_WEIGHT_MICRO_TOPK = 8
B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS = 72
B12X_EP_ZERO_WEIGHT_MICRO_MAX_ROWS = 64



def b12x_ep_micro_chunk_tokens(env_get=os.environ.get) -> int:
    """Tokens per micro call. Frozen at import; capture-safe.

    Every call carries a fixed cost the token count does not change -- phase-0
    zeroing, the expert compaction, the resident-grid barrier -- so at a given
    batch, fewer and wider calls is strictly less overhead. The stock ceiling
    is FlashInfer's `_MICRO_MAX_TOKENS` (8); raising it only makes sense
    because this lane already ships its own `moe_dispatch` overlay, and the
    caller must still prove the live dispatcher admits the wider shape
    (require_b12x_ep_zero_weight_micro_dispatch does that at setup).

    Invalid or unset leaves the stock 8, so a typo cannot silently widen the
    call and land on a kernel that was never validated for it.
    """
    raw = (env_get("VLLM_B12X_EP_MICRO_CHUNK_TOKENS") or "").strip()
    if not raw:
        return B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS
    if value < B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS or value % 8 or value > 64:
        return B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS
    return value


def b12x_ep_zero_weight_micro_chunks(
    num_tokens,
    top_k,
    num_local_experts,
    *,
    enabled,
):
    """Return exact stable token slices, or an empty fail-closed plan."""
    tokens = int(num_tokens)
    chunk = b12x_ep_micro_chunk_tokens()
    if not (
        enabled
        and tokens >= chunk
        and tokens <= B12X_EP_ZERO_WEIGHT_MICRO_MAX_TOKENS
        and int(top_k) == B12X_EP_ZERO_WEIGHT_MICRO_TOPK
        and int(num_local_experts) == B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS
    ):
        return ()
    return tuple(
        (lo, lo + chunk) for lo in range(0, (tokens // chunk) * chunk, chunk)
    )


def b12x_ep_micro_tail(num_tokens, top_k, num_local_experts, *, enabled):
    """Token range the chunk plan could not cover, or None.

    Real decode batches are almost never a multiple of the chunk. With chunked
    prefill in the mix this lane saw 9, 36, 49, 52, 60 tokens and only 8/16/32
    were admitted -- everything else fell to the pair path at up to 60 calls
    per layer, which is what collapsed C>=2. Cover the aligned prefix with
    micro calls and leave only the short tail to that fallback: 49 tokens
    becomes 6 micro calls plus one tail, not 49 pair calls.

    The split is a function of the token count alone, so a captured graph
    replays the same launches.
    """
    chunks = b12x_ep_zero_weight_micro_chunks(
        num_tokens, top_k, num_local_experts, enabled=enabled
    )
    if not chunks:
        return None
    covered = chunks[-1][1]
    tokens = int(num_tokens)
    return (covered, tokens) if covered < tokens else None


def require_b12x_ep_zero_weight_micro_dispatch(moe_dispatch) -> int:
    """Prove the live FlashInfer overlay admits only the exact E=72 lane."""
    if getattr(moe_dispatch, "_B12X_EP_ZERO_WEIGHT_MICRO", False) is not True:
        raise RuntimeError(
            "VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1 requires the matching "
            "FlashInfer moe_dispatch overlay (its process latch is not armed)"
        )
    gate = getattr(
        moe_dispatch, "_b12x_ep_zero_weight_micro_expert_id", None
    )
    if not callable(gate):
        raise RuntimeError(
            "VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1 requires the matching "
            "FlashInfer micro dispatch gate"
        )
    sentinel = gate(
        enabled=True,
        state_E=B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS,
        weight_E=B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS,
        num_tokens=B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS,
        k=4096,
        n=2048,
        num_topk=B12X_EP_ZERO_WEIGHT_MICRO_TOPK,
        activation_precision="fp4",
        quant_mode="nvfp4",
        activation="swigluoai_uninterleave",
        swiglu_limit=10.0,
        forced_backend=getattr(moe_dispatch, "_FORCED_BACKEND", None),
    )
    if sentinel != B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS:
        raise RuntimeError(
            "FlashInfer zero-weight micro gate rejected the pinned GLM EP "
            f"contract (sentinel={sentinel!r})"
        )
    return sentinel


def b12x_ep_fixed_slice_limit(
    max_num_tokens, micro_max_pairs=B12X_EP_FIXED_MICRO_MAX_PAIRS
):
    """Rows per fixed EP call, capped so the dispatcher stays on micro."""
    workspace_limit = max(int(max_num_tokens or 0), 1)
    micro_limit = max(int(micro_max_pairs or 0), 1)
    return min(workspace_limit, micro_limit)


def b12x_ep_fixed_pair_plan(
    local_flags,
    num_pairs,
    slice_size=B12X_EP_FIXED_MICRO_MAX_PAIRS,
):
    """Fixed-shape (index, keep) plan that drops the dummy expert entirely.

    Pure index arithmetic over a boolean mask, expressed so the caller can run
    it with tensors: no ``nonzero``, no data-dependent shape, no host sync --
    which is what kept the compact path out of CUDA graphs.

    The plan is ``num_pairs`` long, so it never drops a real local pair: the
    first ``n_local`` slots are the local pairs in order, and every slot after
    them REPEATS one of those same pairs, cycling, at router weight 0.

    Why repeats and not a dummy expert. The sentinel cannot reach an E-local
    kernel, while repeating an existing pair guarantees a valid expert id and
    introduces no new weight plane. The pinned micro kernel quantizes FC2 per
    routed row, but keeping a repeat inside the same call is also conservative
    across dispatcher revisions: the call's real-row value set is unchanged.
    The dummy expert -- and the 12 MiB/layer of zero weights it adds --
    disappears.

    The runtime submits this plan in ``slice_size``-row micro calls. If a
    slice mixes real pairs with padding, its padding repeats only the real
    pairs in that SAME slice. Slices containing padding only may cycle over the
    whole local set because they have no real output to perturb.

    Returns (src_index, keep) as plain lists for the pure-python path.
    """
    flags = [bool(f) for f in local_flags][:num_pairs]
    local_positions = [i for i, f in enumerate(flags) if f]
    n_local = len(local_positions)
    slice_size = max(int(slice_size), 1)
    if n_local == 0:
        # Nothing routes here this step: keep the shape, zero every weight.
        return [0] * num_pairs, [False] * num_pairs
    src, keep = [], []
    for slot in range(num_pairs):
        if slot < n_local:
            src.append(local_positions[slot])
            keep.append(True)
        else:
            slice_start = (slot // slice_size) * slice_size
            local_in_slice = max(
                0, min(n_local - slice_start, slice_size)
            )
            if local_in_slice:
                src_pos = slice_start + (
                    (slot - slice_start) % local_in_slice
                )
            else:
                src_pos = slot % n_local
            src.append(local_positions[src_pos])
            keep.append(False)
    return src, keep


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
    local — that is flashinfer #3383. Remote routes become the sentinel
    ``num_local_experts`` at scale 0. The default direct paths remove or
    replace that sentinel before an ``E = num_local_experts`` kernel call;
    the rollback path materializes it as an extra zero-weight expert.

    ``expert_map``, when given, is the vLLM table (global → local or -1).
    Without it the linear shard ``[offset, offset+local)`` is assumed.

    This helper only remaps; the selected runtime path owns the sentinel's
    final removal or padded storage.
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
# Decode graphs at GRAPH_CAP=32 * top_k=8 = 256 stay fixed-shape and replace
# dummy slots with zero-weight repeats. Prefill crosses 640 and drops them.
B12X_EP_COMPACT_MIN_ROUTED = 640

# The measured fingerprint is worth one device sync per process, not one per
# MoE layer -- this model has 42 of them and they share a wrapper geometry.
_EP_CAPACITY_LOGGED = False
_EP_MICRO_DISABLED = False


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
            remote.fill_(True)
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


@dataclass(frozen=True)
class _B12xEpFixedWorkspaceKey:
    """Geometry that sizes the pinned direct-EP micro workspace."""

    num_experts: int
    top_k: int
    max_rows: int
    hidden_size: int
    intermediate_size: int
    device: str
    activation: str


# The functional API keeps one replaceable workspace per geometry. A compact
# prefill can grow that cache after decode CUDA graphs were captured, dropping
# the last Python reference to the small workspace whose addresses the graphs
# recorded. Pin shape-specific workspaces and pass them explicitly for direct
# decode. The default top-k=1 lane uses 8 rows (~3.4 MiB at E=72, 4096/2048);
# the stock top-k=8 experiment uses 40 rows and the kernel-overlay experiment
# has its own 64-row object. Compact prefill keeps using the
# independent functional cache, so it cannot replace graph addresses.
_B12X_EP_FIXED_WORKSPACES: "WeakValueDictionary[_B12xEpFixedWorkspaceKey, Any]" = (
    WeakValueDictionary()
)
_B12X_EP_FIXED_WORKSPACES_LOCK = Lock()


def _shared_ep_fixed_workspace(
    key: _B12xEpFixedWorkspaceKey, device: torch.device
):
    with _B12X_EP_FIXED_WORKSPACES_LOCK:
        workspace = _B12X_EP_FIXED_WORKSPACES.get(key)
        if workspace is None:
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
                allocate_sm120_moe_workspace,
            )

            workspace = allocate_sm120_moe_workspace(
                state_E=key.num_experts,
                weight_E=key.num_experts,
                k=key.hidden_size,
                n=key.intermediate_size,
                num_topk=key.top_k,
                device=device,
                max_rows=key.max_rows,
                quant_mode="nvfp4",
                backend="static",
                activation=key.activation,
            )
            _B12X_EP_FIXED_WORKSPACES[key] = workspace
            logger.info_once(
                "b12x EP fixed workspace pinned across layers "
                "(%d experts, top_k %d, max_rows %d, %d/%d)",
                key.num_experts,
                key.top_k,
                key.max_rows,
                key.hidden_size,
                key.intermediate_size,
            )
        return workspace


B12X_EP_COMPACT_PAIR_ALIGN = 64


def b12x_ep_compact_pair_count(n_local, align=B12X_EP_COMPACT_PAIR_ALIGN):
    """Round a compacted pair count up to a launch-shape bucket.

    b12x JIT-compiles per launch shape -- its cache key carries the row count
    -- and compact's row count is the number of LOCAL pairs, which is data
    dependent. One bench saw 33 distinct values (48, 83, 102, 103, 104, 119,
    122, 130, ... 336) and each minted a kernel. Bucketing to 64 collapses
    those to six.

    The padding is bounded by `align - 1` rows, which at prefill scale
    (thousands of pairs) is under a percent, and for a small call is a few
    rows of a call that was already small.
    """
    n = int(n_local)
    if n <= 0:
        return 0
    return ((n + align - 1) // align) * align


def b12x_ep_compact_warmup_buckets(
    max_num_tokens, top_k, num_local_experts, global_num_experts,
    align=B12X_EP_COMPACT_PAIR_ALIGN, floors=6,
):
    """Bucket sizes worth compiling at load instead of mid-request.

    b12x JITs per launch shape and prefill's shape is the local-pair count, so
    the first request at a new size stalls the engine for seconds -- the JIT
    monitor says as much ("consider warmup to cover this shape/config"). #163
    collapsed those to `align` buckets; this walks the ones a real prefill can
    reach and pays for them once, at load.

    The ladder halves from the largest chunk this engine can schedule down to
    one bucket, so a handful of compiles covers every chunked-prefill size the
    scheduler produces. Ordered largest-first: the big one dominates the cost
    and is the one a first long prompt hits.
    """
    tokens = int(max_num_tokens or 0)
    k = int(top_k or 0)
    local = int(num_local_experts or 0)
    total = int(global_num_experts or 0)
    if tokens <= 0 or k <= 0 or local <= 0 or total < local:
        return ()
    # A rank only materialises the slots its own experts own.
    peak = b12x_ep_compact_pair_count(max(1, tokens * k * local // total), align)
    out, size = [], peak
    while size >= align and len(out) < max(1, int(floors)):
        out.append(size)
        size = b12x_ep_compact_pair_count(size // 2, align)
        if out and size == out[-1]:
            break
    return tuple(out)



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
    and indexes weights by the ids it is given (flashinfer #3383). The default
    direct path keeps exactly the local weight rows, remaps global top-k ids,
    then removes every remote sentinel before a direct top-k=1 call.
    Decode/graph batches replace remote slots with zero-weight repeats and
    submit at most eight pairs per call, so the dispatcher stays on its proven
    micro kernel instead of static. Prefill (routed pairs > 640) drops remote
    slots. The default-off ``VLLM_B12X_EP_STOCK_TOPK_MICRO=1`` experiment
    instead keeps token-major top-k=8 under the stock 40-row cutover, reducing
    stable 8/16/32-token shapes to 2/4/7 calls. The mutually-exclusive
    default-off ``VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1`` experiment keeps E=72 and
    admits those shapes as disjoint eight-token top-k=8 calls; the kernel
    removes exact-zero sentinel pairs before row materialization.
    ``VLLM_B12X_EP_NO_DUMMY=0`` restores the padded ``E = local + 1`` wrapper
    path. vLLM's EP all-reduce (DP=1) combines ranks.
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
        self._ep_no_dummy = False
        self._ep_disable_micro = False
        self._ep_zero_weight_micro = False
        self._ep_stock_topk_micro = False
        if self._use_ep:
            (
                self._ep_no_dummy,
                self._ep_disable_micro,
                self._ep_zero_weight_micro,
                self._ep_stock_topk_micro,
            ) = b12x_ep_mode_from_env()
        self._ep_ids: torch.Tensor | None = None
        self._ep_scales: torch.Tensor | None = None
        self._ep_long: torch.Tensor | None = None
        self._ep_mapped: torch.Tensor | None = None
        self._ep_remote: torch.Tensor | None = None
        self._ep_fill_ids: torch.Tensor | None = None
        self._ep_tmp_a: torch.Tensor | None = None
        self._ep_tmp_b: torch.Tensor | None = None
        self._ep_dummy_padded = False
        # Strong reference for every captured layer. The global map shares the
        # object by geometry but intentionally holds it weakly.
        self._ep_fixed_workspace: Any | None = None
        self._ep_stock_topk_workspace: Any | None = None
        self._ep_zero_weight_workspace: Any | None = None

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
            if not self._ep_no_dummy:
                self._pad_dummy_expert(layer)
            global _EP_MICRO_DISABLED
            if not _EP_MICRO_DISABLED:
                micro_status = disable_b12x_micro_for_ep(
                    self._ep_no_dummy, self._ep_disable_micro
                )
                _EP_MICRO_DISABLED = True
                logger.warning(
                    "[b12x EP] %s",
                    micro_status,
                )

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
        if self._use_ep:
            self._warm_compact_shapes(layer)


        if self._ep_no_dummy:
            device = layer.w13_weight.device
            self._ep_fixed_workspace = _shared_ep_fixed_workspace(
                _B12xEpFixedWorkspaceKey(
                    num_experts=self._kernel_num_experts,
                    top_k=1,
                    max_rows=B12X_EP_FIXED_MICRO_MAX_PAIRS,
                    hidden_size=self.hidden_dim,
                    intermediate_size=self.intermediate_size_per_partition,
                    device=str(device),
                    activation=self._activation_str,
                ),
                device,
            )
            if self._ep_stock_topk_micro:
                geometry = (
                    self.global_num_experts,
                    self._kernel_num_experts,
                    self.topk,
                    self.hidden_dim,
                    self.intermediate_size_per_partition,
                    self._activation_str,
                    self._swiglu_limit,
                )
                expected = (
                    288,
                    B12X_EP_STOCK_TOPK_MICRO_EXPERTS,
                    B12X_EP_STOCK_TOPK_MICRO_TOPK,
                    4096,
                    2048,
                    "swigluoai_uninterleave",
                    10.0,
                )
                if geometry != expected:
                    raise RuntimeError(
                        "VLLM_B12X_EP_STOCK_TOPK_MICRO=1 only supports the "
                        f"pinned GLM EP geometry {expected}; got {geometry}"
                    )
                from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
                    moe_dispatch,
                )

                require_b12x_ep_stock_topk_micro_dispatch(moe_dispatch)
                self._ep_stock_topk_workspace = _shared_ep_fixed_workspace(
                    _B12xEpFixedWorkspaceKey(
                        num_experts=self._kernel_num_experts,
                        top_k=B12X_EP_STOCK_TOPK_MICRO_TOPK,
                        max_rows=B12X_EP_STOCK_TOPK_MICRO_MAX_ROUTED_ROWS,
                        hidden_size=self.hidden_dim,
                        intermediate_size=self.intermediate_size_per_partition,
                        device=str(device),
                        activation=self._activation_str,
                    ),
                    device,
                )
            if self._ep_zero_weight_micro:
                geometry = (
                    self._kernel_num_experts,
                    self.topk,
                    self.hidden_dim,
                    self.intermediate_size_per_partition,
                    self._activation_str,
                    self._swiglu_limit,
                )
                expected = (
                    B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS,
                    B12X_EP_ZERO_WEIGHT_MICRO_TOPK,
                    4096,
                    2048,
                    "swigluoai_uninterleave",
                    10.0,
                )
                if geometry != expected:
                    raise RuntimeError(
                        "VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1 only supports the "
                        f"pinned GLM EP geometry {expected}; got {geometry}"
                    )
                from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
                    moe_dispatch,
                )

                require_b12x_ep_zero_weight_micro_dispatch(moe_dispatch)
                self._ep_zero_weight_workspace = _shared_ep_fixed_workspace(
                    _B12xEpFixedWorkspaceKey(
                        num_experts=self._kernel_num_experts,
                        top_k=B12X_EP_ZERO_WEIGHT_MICRO_TOPK,
                        max_rows=B12X_EP_ZERO_WEIGHT_MICRO_MAX_ROWS,
                        hidden_size=self.hidden_dim,
                        intermediate_size=self.intermediate_size_per_partition,
                        device=str(device),
                        activation=self._activation_str,
                    ),
                    device,
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
        return b12x_ep_kernel_expert_count(
            self.num_local_experts, self._use_ep, self._ep_no_dummy
        )

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
            or (
                self._ep_stock_topk_micro and self._ep_fill_ids is None
            )
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
        if self._ep_stock_topk_micro:
            self._ep_fill_ids = torch.empty(
                (rows, 1), dtype=torch.int32, device=device
            )
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

    def _apply_ep_zero_weight_micro(
        self, output, hidden_states, w1, w2, topk_ids, topk_weights
    ):
        """Opt-in local-only top-k=8 micro lane for stable decode shapes.

        Remote routes remain sentinel E at exact weight zero. The matching
        micro variant removes those pairs before row_counts/token_map append,
        so the sentinel can never reach alpha or weight-plane addressing. Each
        launch owns disjoint token/output rows; there is no pair gather or
        index_add, and all launch shapes are fixed by the captured token shape.
        """
        chunks = b12x_ep_zero_weight_micro_chunks(
            topk_ids.size(0),
            topk_ids.size(1),
            self._kernel_num_experts,
            enabled=self._ep_zero_weight_micro,
        )
        if not chunks:
            raise RuntimeError("zero-weight micro called outside its exact shape gate")
        if self._ep_zero_weight_workspace is None:
            raise RuntimeError(
                "zero-weight micro workspace was not allocated before apply"
            )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
        )

        logger.info_once(
            "b12x EP zero-weight micro: %d tokens -> %d top-k=8 calls "
            "(8 tokens / 64 routed pairs each)",
            topk_ids.size(0), len(chunks),
        )
        for lo, hi in chunks:
            launch_sm120_moe(
                a=hidden_states[lo:hi],
                topk_ids=topk_ids[lo:hi],
                topk_weights=topk_weights[lo:hi],
                w1_weight=w1,
                w1_weight_sf=self.w1_sf_mma,
                w1_alpha=self.g1_alphas,
                fc2_input_scale=self._fc2_input_scale,
                input_global_scale=None,
                w2_weight=w2,
                w2_weight_sf=self.w2_sf_mma,
                w2_alpha=self.g2_alphas,
                num_experts=self._kernel_num_experts,
                top_k=B12X_EP_ZERO_WEIGHT_MICRO_TOPK,
                num_local_experts=self._kernel_num_experts,
                scatter_output=output[lo:hi],
                activation=self._activation_str,
                swiglu_alpha=self._swiglu_alpha,
                swiglu_beta=self._swiglu_beta,
                swiglu_limit=self._swiglu_limit,
                activation_precision="fp4",
                quant_mode="nvfp4",
                source_format="modelopt",
                _workspace=self._ep_zero_weight_workspace,
            )
        tail = b12x_ep_micro_tail(
            topk_ids.size(0),
            topk_ids.size(1),
            self._kernel_num_experts,
            enabled=self._ep_zero_weight_micro,
        )
        if tail is not None and self._ep_tail_padded_micro(
            output, hidden_states, w1, w2, topk_ids, topk_weights, tail
        ):
            return output
        if tail is not None:
            lo, hi = tail
            # The micro calls above wrote output[0:lo] and touched no other
            # row, so the fallback runs on a disjoint view: everything it does
            # stays inside [lo, hi) and cannot erase them. Fewer than one chunk
            # of tokens reach it, so the gather it pays is bounded.
            self._apply_ep_fixed(
                output[lo:hi],
                hidden_states[lo:hi],
                w1,
                w2,
                topk_ids[lo:hi],
                topk_weights[lo:hi],
            )
        return output

    def _ep_tail_padded_micro(
        self, output, hidden_states, w1, w2, topk_ids, topk_weights, tail
    ):
        """Run the short tail as a FULL chunk, padded. Returns whether it ran.

        The b12x kernel is JIT-compiled per launch shape -- its cache key is
        `static_m{rows}_k..._n..._r{routed}` -- so a tail whose length is
        `tokens % chunk` mints a new kernel for every remainder it sees. On a
        C=4 run that cost 114 compilations mid-bench: the engine reported
        71-75 tok/s in the gaps and 13.4 tok/s on average, because every new
        shape stalled it for seconds.

        Padding the tail up to one chunk collapses every launch in this lane
        onto a single shape. The pad rows carry router weight 0 and repeat the
        tail's own first row, so they add nothing and cannot move an expert's
        FC2 amax (the max of a set is unchanged by repeating a member).
        """
        lo, hi = tail
        chunk = b12x_ep_micro_chunk_tokens()
        rem = hi - lo
        if rem <= 0 or rem >= chunk:
            return False
        buf = self._ep_tail_buffers(chunk, hidden_states, topk_ids, topk_weights)
        if buf is None:
            return False
        pad_x, pad_ids, pad_w, pad_out = buf
        pad_x[:rem].copy_(hidden_states[lo:hi])
        pad_ids[:rem].copy_(topk_ids[lo:hi])
        pad_w[:rem].copy_(topk_weights[lo:hi])
        # Pad rows repeat the tail's first row at weight 0.
        pad_x[rem:].copy_(hidden_states[lo:lo + 1].expand(chunk - rem, -1))
        pad_ids[rem:].copy_(topk_ids[lo:lo + 1].expand(chunk - rem, -1))
        pad_w[rem:].zero_()
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
        )

        launch_sm120_moe(
            a=pad_x,
            topk_ids=pad_ids,
            topk_weights=pad_w,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            input_global_scale=None,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            num_experts=self._kernel_num_experts,
            top_k=B12X_EP_ZERO_WEIGHT_MICRO_TOPK,
            num_local_experts=self._kernel_num_experts,
            scatter_output=pad_out,
            activation=self._activation_str,
            swiglu_alpha=self._swiglu_alpha,
            swiglu_beta=self._swiglu_beta,
            swiglu_limit=self._swiglu_limit,
            activation_precision="fp4",
            quant_mode="nvfp4",
            source_format="modelopt",
            _workspace=self._ep_zero_weight_workspace,
        )
        output[lo:hi].copy_(pad_out[:rem])
        return True

    def _ep_tail_buffers(self, chunk, hidden_states, topk_ids, topk_weights):
        """Lazily pin the one-chunk staging tensors. None if shapes drift."""
        key = (
            chunk,
            hidden_states.size(1),
            topk_ids.size(1),
            hidden_states.dtype,
            topk_ids.dtype,
            topk_weights.dtype,
            hidden_states.device,
        )
        if getattr(self, "_ep_tail_key", None) != key:
            try:
                self._ep_tail_x = torch.zeros(
                    (chunk, hidden_states.size(1)),
                    dtype=hidden_states.dtype, device=hidden_states.device)
                self._ep_tail_ids = torch.zeros(
                    (chunk, topk_ids.size(1)),
                    dtype=topk_ids.dtype, device=hidden_states.device)
                self._ep_tail_w = torch.zeros(
                    (chunk, topk_weights.size(1)),
                    dtype=topk_weights.dtype, device=hidden_states.device)
                self._ep_tail_out = torch.zeros(
                    (chunk, hidden_states.size(1)),
                    dtype=hidden_states.dtype, device=hidden_states.device)
            except Exception as exc:
                logger.warning_once(
                    "b12x EP tail padding unavailable (%s: %s); "
                    "falling back to the variable-length tail",
                    type(exc).__name__, exc)
                self._ep_tail_key = None
                return None
            self._ep_tail_key = key
        return (
            self._ep_tail_x, self._ep_tail_ids,
            self._ep_tail_w, self._ep_tail_out,
        )

    def _apply_ep_stock_topk_micro(
        self, output, hidden_states, w1, w2, topk_ids, topk_weights
    ):
        """Opt-in stock top-k micro decode for pinned GLM graph shapes.

        Keep the router's native token-major top_k=8 layout. Remote routes
        already carry weight zero after remap; replace only their sentinel ids
        with the smallest local id in that token, then let the stock micro
        kernel do its weighted scatter directly into the token output. This
        does not introduce a weight plane that token did not already use
        (except an all-remote token, whose output is sealed back to zero).

        The stock multi-top-k micro cutover is 40 routed rows, so a top_k=8
        call may contain at most five tokens. Balanced shape-only chunks turn
        the stable 8/16/32-token graph shapes into 2/4/7 calls, versus the old
        8/16/32 top_k=1 pair calls. No flatten, sort, gather, pair buffer, mask,
        or index_add remains in this captured experiment. The separate opt-in
        kernel-overlay lane still runs first and can use 8-token chunks.
        """
        dummy = self.num_local_experts
        tokens, topk = topk_ids.size(0), topk_ids.size(1)
        spans = b12x_ep_stock_topk_micro_chunks(
            tokens,
            topk,
            self._kernel_num_experts,
            enabled=self._ep_stock_topk_micro,
        )
        if not spans:
            raise RuntimeError(
                "stock top-k micro called outside its exact shape gate"
            )
        limit = max(hi - lo for lo, hi in spans)
        if dummy <= 0:
            raise RuntimeError("b12x EP fixed decode requires a local expert")
        if (
            self._ep_remote is None
            or self._ep_fill_ids is None
            or self._ep_tmp_a is None
        ):
            raise RuntimeError(
                "b12x EP fixed routing scratch was not allocated before apply"
            )
        if self._ep_stock_topk_workspace is None:
            raise RuntimeError(
                "stock top-k micro workspace was not allocated before apply"
            )

        # amin returns a real local id whenever this token has one because the
        # remote sentinel is E (larger than every valid id). All-remote rows
        # clamp E to E-1; their router weights are all zero. Reuse preallocated
        # scratch and the remap mask so CUDA capture records no tensor allocs.
        fill_ids = self._ep_fill_ids[:tokens]
        torch.amin(topk_ids, dim=1, keepdim=True, out=fill_ids)
        all_remote = self._ep_tmp_a[:tokens, :1]
        torch.eq(fill_ids, dummy, out=all_remote)
        fill_ids.clamp_max_(dummy - 1)
        torch.where(
            self._ep_remote[:tokens], fill_ids, topk_ids, out=topk_ids
        )

        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
        )

        logger.info_once(
            "b12x EP stock top-k micro: %d tokens x top_k %d -> %d balanced "
            "stock micro calls (at most %d tokens / %d routed rows each)",
            tokens, topk, len(spans), limit, limit * topk,
        )
        output.zero_()
        for lo, hi in spans:
            launch_sm120_moe(
                a=hidden_states[lo:hi],
                topk_ids=topk_ids[lo:hi],
                topk_weights=topk_weights[lo:hi],
                w1_weight=w1,
                w1_weight_sf=self.w1_sf_mma,
                w1_alpha=self.g1_alphas,
                fc2_input_scale=self._fc2_input_scale,
                input_global_scale=None,
                w2_weight=w2,
                w2_weight_sf=self.w2_sf_mma,
                w2_alpha=self.g2_alphas,
                num_experts=self._kernel_num_experts,
                top_k=topk,
                num_local_experts=self._kernel_num_experts,
                scatter_output=output[lo:hi],
                activation=self._activation_str,
                swiglu_alpha=self._swiglu_alpha,
                swiglu_beta=self._swiglu_beta,
                swiglu_limit=self._swiglu_limit,
                activation_precision="fp4",
                quant_mode="nvfp4",
                source_format="modelopt",
                _workspace=self._ep_stock_topk_workspace,
            )
        output.masked_fill_(all_remote, 0)
        return output

    def _apply_ep_fixed(
        self, output, hidden_states, w1, w2, topk_ids, topk_weights
    ):
        """Decode path with no dummy expert. Fixed shape, capture-safe.

        Keeps every routed slot -- length stays tokens*top_k -- but rewrites
        the remote ones into REPEATS of this rank's own local pairs at weight
        0, cycling so the spares spread over the experts already present.

        What that buys: the kernel never sees expert `num_local_experts`, so
        it never reads that expert's plane of zero weights. On this model that
        is 12 MiB per layer per step (2*2048*4096 + 4096*2048 at NVFP4), about
        2 ms/step of pure waste, and it is a WEIGHT read -- paid once per layer
        no matter how few rows land on it.

        Safe by construction: no real local pair is dropped (the plan is as
        long as the input), every padding id names a real local pair, and every
        padding weight is zero. Mixed slices repeat only their own real rows,
        preserving the call's real-row value set as a conservative contract.

        Calls are capped at eight pair rows. Stable DFlash verification shapes
        have 8/16/32 tokens at C=1/2/4; top-k=8 flattening produces
        64/128/256 pair rows and therefore 8/16/32 micro calls per MoE layer.
        One unsliced fixed call would select the static kernel observed to
        hang. The number and shapes of calls are fixed by the captured graph
        shape. This establishes the safe dispatch shape, not an end-to-end
        throughput win.

        No nonzero, no host sync -- every step is index arithmetic on shapes
        that are fixed once tokens is fixed. Padding in a mixed call repeats
        only real pairs from that same call, keeping its per-expert FC2 amax
        unchanged across dispatcher revisions.
        """
        dummy = self.num_local_experts
        tokens, topk = topk_ids.size(0), topk_ids.size(1)
        device = topk_ids.device
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weights.reshape(-1)
        pairs = flat_ids.numel()
        limit = b12x_ep_fixed_slice_limit(self.max_num_tokens)

        is_local = flat_ids != dummy
        # Local pairs first, original order kept; remote ones fall to the tail.
        order = torch.argsort(is_local.to(torch.int8), descending=True, stable=True)
        n_local = is_local.sum()
        pos = torch.arange(pairs, device=device)
        keep = pos < n_local
        slice_start = (pos // limit) * limit
        local_in_slice = torch.clamp(
            n_local - slice_start, min=0, max=limit
        )
        slice_src = slice_start + (
            (pos - slice_start) % torch.clamp(local_in_slice, min=1)
        )
        global_src = pos % torch.clamp(n_local, min=1)
        padding_src = torch.where(local_in_slice > 0, slice_src, global_src)
        src = torch.where(keep, pos, padding_src)
        idx = order.index_select(0, src)

        pair_ids = flat_ids.index_select(0, idx)
        # n_local == 0 leaves the tail pointing at a remote slot; send those to
        # expert 0 instead. Every one of them carries weight 0.
        pair_ids = torch.where(
            pair_ids == dummy, torch.zeros_like(pair_ids), pair_ids
        )
        pair_scales = torch.where(
            keep, flat_w.index_select(0, idx), torch.zeros_like(flat_w)
        )
        token_index = (
            torch.arange(tokens, device=device, dtype=torch.int64)
            .unsqueeze(1)
            .expand(tokens, topk)
            .reshape(-1)
            .index_select(0, idx)
        )

        pair_x = hidden_states.index_select(0, token_index)
        # zeros, not empty. Three quarters of these rows carry router weight 0
        # (the repeats that replace remote slots), and we cannot assume the
        # kernel writes a row it has no work for. index_add_ below sums EVERY
        # row into the token-major output, so an unwritten row contributes
        # uninitialised memory -- and an uninitialised bit pattern can be NaN,
        # which no later multiply can clear.
        #
        # _apply_ep_compact gets away with empty because it drops the remote
        # slots outright, leaving only rows with real weights. This path is the
        # first to hand the buffer rows the kernel may skip.
        pair_out = torch.zeros(
            (pairs, hidden_states.size(1)),
            dtype=output.dtype,
            device=output.device,
        )
        if self._ep_fixed_workspace is None:
            raise RuntimeError(
                "b12x EP fixed workspace was not allocated before apply"
            )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
        )

        logger.info_once(
            "b12x EP fixed decode: %d pairs -> %d micro-sized calls "
            "(at most %d pairs each; micro-eligible by shape)",
            pairs, (pairs + limit - 1) // limit, limit,
        )
        for lo in range(0, pairs, limit):
            hi = min(lo + limit, pairs)
            launch_sm120_moe(
                a=pair_x[lo:hi],
                topk_ids=pair_ids[lo:hi].view(-1, 1).to(torch.int32),
                topk_weights=pair_scales[lo:hi].view(-1, 1),
                w1_weight=w1,
                w1_weight_sf=self.w1_sf_mma,
                w1_alpha=self.g1_alphas,
                fc2_input_scale=self._fc2_input_scale,
                input_global_scale=None,
                w2_weight=w2,
                w2_weight_sf=self.w2_sf_mma,
                w2_alpha=self.g2_alphas,
                num_experts=self._kernel_num_experts,
                top_k=1,
                num_local_experts=self._kernel_num_experts,
                scatter_output=pair_out[lo:hi],
                activation=self._activation_str,
                swiglu_alpha=self._swiglu_alpha,
                swiglu_beta=self._swiglu_beta,
                swiglu_limit=self._swiglu_limit,
                activation_precision="fp4",
                quant_mode="nvfp4",
                source_format="modelopt",
                _workspace=self._ep_fixed_workspace,
            )
        # Second guard, for the other way this breaks: if the kernel writes a
        # zero-weight row WITHOUT applying token_final_scales, that row holds a
        # full unscaled expert output and index_add_ would add it to a real
        # token. Masking makes the sum correct under either kernel behaviour.
        pair_out.mul_(keep.unsqueeze(1).to(pair_out.dtype))
        output.zero_()
        output.index_add_(0, token_index, pair_out)
        return output

    def _warm_compact_shapes(self, layer) -> None:
        """Compile the compact prefill shapes now, not mid-request. Opt-in.

        b12x JITs per launch shape, so the first prefill at a new size stalls
        the engine for seconds -- vLLM's own JIT monitor says "consider warmup
        to cover this shape/config". #163 collapsed prefill to `align` buckets;
        this pays for the ladder of them once, here.

        Off by default: it costs load time, and the gain is only real if
        compiles actually dominate a first prefill. Arm with
        VLLM_B12X_EP_WARM_COMPACT=1 and read it off TTFT for a cold long
        prompt, not off steady-state throughput.
        """
        if os.environ.get("VLLM_B12X_EP_WARM_COMPACT", "0").strip() != "1":
            return
        buckets = b12x_ep_compact_warmup_buckets(
            self.max_num_tokens, self.topk,
            self.num_local_experts, self.global_num_experts,
        )
        if not buckets:
            return
        try:
            from flashinfer.fused_moe import b12x_fused_moe

            w1 = layer.w13_weight
            w2 = layer.w2_weight
            device = w1.device
            hidden = self.hidden_dim
            dtype = self._warm_activation_dtype()
            kernel_e = self._kernel_num_experts
            done = []
            for rows in buckets:
                x = torch.zeros((rows, hidden), dtype=dtype, device=device)
                ids = torch.zeros((rows, 1), dtype=torch.int32, device=device)
                sc = torch.zeros((rows, 1), dtype=torch.float32, device=device)
                out = torch.zeros((rows, hidden), dtype=dtype, device=device)
                b12x_fused_moe(
                    x=x, w1_weight=w1, w1_weight_sf=self.w1_sf_mma,
                    w2_weight=w2, w2_weight_sf=self.w2_sf_mma,
                    token_selected_experts=ids, token_final_scales=sc,
                    num_experts=kernel_e, top_k=1, num_local_experts=kernel_e,
                    w1_alpha=self.g1_alphas, w2_alpha=self.g2_alphas,
                    fc2_input_scale=self._fc2_input_scale, output=out,
                    activation=self._activation_str,
                    swiglu_alpha=self._swiglu_alpha,
                    swiglu_beta=self._swiglu_beta,
                    swiglu_limit=self._swiglu_limit,
                )
                done.append(rows)
            # info_once hashes its args to dedupe, so every one must be
            # hashable -- a list here raised TypeError and the whole warmup
            # was skipped. It said so out loud, which is the only reason this
            # was caught instead of read as "warmed but no effect".
            logger.info_once(
                "b12x EP: warmed %d compact prefill shapes at load (%s) -- "
                "a first long prompt no longer compiles mid-request",
                len(done), ",".join(str(r) for r in done),
            )
        except Exception as exc:
            # A warmup that fails must cost nothing but a line in the log.
            logger.warning_once(
                "b12x EP compact warmup skipped (%s: %s); shapes will compile "
                "on first use as before",
                type(exc).__name__, exc,
            )

    def _warm_activation_dtype(self):
        """Activation dtype for the warmup tensors, bf16 unless told otherwise."""
        for name in ("_activation_dtype", "activation_dtype", "in_dtype"):
            got = getattr(self, name, None)
            if isinstance(got, torch.dtype):
                return got
        return torch.bfloat16

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
        # Pad the pair list up to a launch-shape bucket. Without this the row
        # count is the local-pair count -- data dependent -- and every value
        # mints a fresh JIT kernel mid-inference.
        n_pad = b12x_ep_compact_pair_count(n)
        if n_pad > n:
            fill = torch.arange(n_pad, device=sel.device) % n
            sel = sel.index_select(0, fill)
            token_index = token_index.index_select(0, fill)
        keep = torch.arange(n_pad, device=sel.device) < n
        pair_ids = topk_ids.reshape(-1).index_select(0, sel).view(n_pad, 1)
        pair_scales = topk_weights.reshape(-1).index_select(0, sel).view(n_pad, 1)
        # Pad rows repeat pairs already in this list at weight 0: a duplicate
        # cannot move an expert's FC2 amax, and a zero scale contributes
        # nothing to the token it points at.
        pair_scales = torch.where(
            keep.view(n_pad, 1), pair_scales, torch.zeros_like(pair_scales)
        )
        pair_x = hidden_states.index_select(0, token_index)
        # zeros, not empty: the pad rows carry weight 0 and the kernel may skip
        # them, and an uninitialised bit pattern can be NaN (see #146).
        pair_out = torch.zeros(
            (n_pad, hidden_states.size(1)),
            dtype=output.dtype,
            device=output.device,
        )
        n = n_pad

        from flashinfer.fused_moe import b12x_fused_moe

        kernel_e = self._kernel_num_experts
        # A compacted pair list is up to tokens*top_k long -- 8x the configured
        # token capacity on this model. Walk it in bounded slices so the
        # functional cache never needs a single oversized workspace. That
        # cache is deliberately separate from fixed decode's pinned workspace.
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
        pair_out.mul_(keep.view(-1, 1).to(pair_out.dtype))
        output.zero_()
        output.index_add_(0, token_index, pair_out)
        return output

    def _ensure_wrapper(self) -> None:
        """Lazily create the non-EP or padded EP fallback wrapper."""
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
        """Guard and fingerprint the padded EP fallback workspace once.

        The static kernel writes ``token_map[slot, row]`` where ``row`` is an
        unguarded ``atomicAdd`` on a per-expert counter, so one expert taking
        more than ``token_map.shape[1]`` rows is an out-of-bounds write --
        reported later as an ILLEGAL_ADDRESS in whatever kernel syncs next.
        Under EP the dummy expert absorbs every remote slot, ~3/4 of all
        routing on this model, so it is the one that overruns.

        The guard uses the analytic worst case (every pair on one expert),
        which needs only shapes -- no device sync, so it can run on every
        call. The measured distribution costs a sync and runs once per
        process. Capacity only: which workspace the runtime *selects* is a
        separate question (see FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS).
        """
        global _EP_CAPACITY_LOGGED
        pairs = int(topk_ids.numel())
        fits = []
        for name in ("_static_workspace", "_dynamic_workspace"):
            ws = getattr(wrapper, name, None)
            tm = getattr(ws, "token_map", None) if ws is not None else None
            if tm is None:
                continue
            if tm.dim() >= 2:
                ok = (
                    pairs <= int(tm.shape[1])
                    and self._kernel_num_experts <= int(tm.shape[0])
                )
            else:
                ok = pairs <= int(tm.shape[0])
            fits.append((name, ok, tuple(tm.shape)))
        if fits and not any(ok for _, ok, _ in fits):
            detail = " ".join(f"{n}{shape}" for n, _, shape in fits)
            raise RuntimeError(
                f"b12x EP overruns every workspace: {pairs} pairs over "
                f"{self._kernel_num_experts} experts -- token_map {detail}. "
                f"Force the dynamic backend "
                f"(FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0) or lower "
                f"MAX_BATCHED."
            )
        if _EP_CAPACITY_LOGGED:
            return
        _EP_CAPACITY_LOGGED = True
        try:
            counts = torch.bincount(
                topk_ids.reshape(-1).to(torch.int64),
                minlength=self._kernel_num_experts,
            )
            worst = int(counts.max())
            worst_id = int(counts.argmax())
            logger.warning(
                "[b12x EP capacity] pairs=%d experts=%d worst expert=%d "
                "rows=%d (%.0f%%, dummy=%d) token_map %s",
                pairs, self._kernel_num_experts, worst_id, worst,
                100.0 * worst / max(pairs, 1), self.num_local_experts,
                " ".join(f"{n}{shape}" for n, _, shape in fits),
            )
        except Exception as exc:  # the fingerprint must never break EP
            logger.warning("[b12x EP capacity] measurement unavailable (%s: %s)",
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

        if self._use_ep:
            expect_e = self._kernel_num_experts
            if w1.size(0) != expect_e:
                raise RuntimeError(
                    f"b12x EP weight shape mismatch: w1 E={w1.size(0)} "
                    f"want {expect_e} (local {self.num_local_experts}, "
                    f"no_dummy={int(self._ep_no_dummy)})"
                )
            if self.g1_alphas.numel() < expect_e or self.g2_alphas.numel() < expect_e:
                raise RuntimeError(
                    "b12x EP g1/g2 alpha shape mismatch "
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
            zero_micro_chunks = b12x_ep_zero_weight_micro_chunks(
                topk_ids.size(0),
                topk_ids.size(1),
                self._kernel_num_experts,
                enabled=self._ep_zero_weight_micro,
            )
            if zero_micro_chunks:
                return self._apply_ep_zero_weight_micro(
                    output, hidden_states, w1, w2, topk_ids, topk_weights
                )
            stock_topk_chunks = b12x_ep_stock_topk_micro_chunks(
                topk_ids.size(0),
                topk_ids.size(1),
                self._kernel_num_experts,
                enabled=self._ep_stock_topk_micro,
            )
            if stock_topk_chunks:
                return self._apply_ep_stock_topk_micro(
                    output, hidden_states, w1, w2, topk_ids, topk_weights
                )
            if self._ep_no_dummy and b12x_ep_should_compact(
                topk_ids.size(0) * topk_ids.size(1),
                enabled=b12x_ep_compact_enabled(),
            ):
                # Big/eager batches: dropping the remote slots outright is the
                # cheaper shape, and prefill is not captured.
                return self._apply_ep_compact(
                    output, hidden_states, w1, w2, topk_ids, topk_weights
                )
            if self._ep_no_dummy:
                # Decode: keep the shape fixed for capture, but pay no dummy.
                return self._apply_ep_fixed(
                    output, hidden_states, w1, w2, topk_ids, topk_weights
                )

        # Both direct EP paths return above. Construct the large graph wrapper
        # only for non-EP or the explicit padded rollback path; neither direct
        # path consumes its workspace.
        self._ensure_wrapper()
        wrapper = self._wrapper
        assert wrapper is not None
        if self._use_ep:
            self._ep_capacity_probe(wrapper, topk_ids)

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
