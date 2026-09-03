# SPDX-License-Identifier: Apache-2.0
"""Exact adjacent-query union sparse MLA prefill for GLM-5.3 NoPE.

The Triton tiling and epoch-marked union workspace are adapted from SGLang's
Apache-2.0 ``triton_sparse_mla_prefill`` implementation.  This integration is
an exact-value opt-in monkey patch over vLLM's SM90/SM12x FlashInfer backend;
the original backend remains the fallback for every contract mismatch.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger("vllm.glm53.union_prefill")

_UNION_ENV = "VLLM_GLM53_UNION_PREFILL"
_DENSE_PREFIX_ENV = "VLLM_GLM53_DENSE_PREFIX_PREFILL"
_UNION_SPAN_BUDGET = 512 << 20
_UNION_WS: dict[tuple, tuple] = {}
_UNION_DECLINED: set = set()
# The whole value of this arm is the overlap between adjacent tokens' top-k
# sets, and that number has never been measured -- the arm never ran. The
# union kernel already computes it: union_len is |union| per group, against
# G x per-token length if the tokens shared nothing. Log the ratio for the
# first few groups so the arm's ceiling stops being an assumption.
#
#   saving = 1 - |union| / (G x len)      0 = no overlap, 1 - 1/G = identical
_UNION_STATS_SEEN: list = [0]
_UNION_STATS_MAX = 8


def _measure_overlap(physical, group_size, tag):
    """The one number that decides whether this arm is worth any kernel work.

    Saving is `1 - |union| / (G x per-token top-k)`: 0 when adjacent tokens
    pick disjoint slots, `1 - 1/G` when they pick the same ones. It has never
    been measured, because the arm has never run -- and the arm cannot run
    until its union builder stops being O(span), which is a real kernel. So
    measure it here, in plain torch, on the indices the caller already has,
    BEFORE deciding to write that kernel.

    Sampled and bounded: this is instrumentation, not a serving path.
    """
    if _UNION_STATS_SEEN[0] >= _UNION_STATS_MAX:
        return
    _UNION_STATS_SEEN[0] += 1
    with torch.no_grad():
        rows = physical.shape[0] // group_size * group_size
        if rows == 0:
            return
        sample = min(rows, 256 * group_size)
        grouped = physical[:sample].reshape(-1, group_size * physical.shape[1])
        valid = grouped >= 0
        per_token = valid.sum(dim=1).float() / group_size
        # |union| per group, without a span-sized buffer: sort and count the
        # value changes. Only the count is needed, so this is cheap.
        keys = torch.where(valid, grouped, torch.full_like(grouped, 2**30))
        keys, _ = keys.sort(dim=1)
        changed = torch.ones_like(keys, dtype=torch.bool)
        changed[:, 1:] = keys[:, 1:] != keys[:, :-1]
        changed &= keys < 2**30
        union = changed.sum(dim=1).float()
        ideal = per_token * group_size
        saved = (1.0 - union / ideal.clamp_min(1.0)).mean().item()
        logger.warning(
            "[union-prefill] overlap #%d (%s): groups=%d G=%d "
            "per-token-topk=%.0f |union|=%.0f of %.0f -> gather saved "
            "%.1f pct (ceiling %.1f pct)",
            _UNION_STATS_SEEN[0], tag, grouped.shape[0], group_size,
            per_token.mean().item(), union.mean().item(),
            ideal.mean().item(), 100.0 * saved,
            100.0 * (1.0 - 1.0 / group_size),
        )


@triton.jit
def _glm53_sparse_prefill_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    out_ptr,
    sm_scale,
    kv_scale,
    topk,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    heads = tl.arange(0, BLOCK_H)
    hmask = heads < H
    dims = tl.arange(0, D)
    q = tl.load(
        q_ptr + token * H * D + heads[:, None] * D + dims[None, :],
        mask=hmask[:, None],
        other=0.0,
    )

    max_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    sum_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    cols = tl.arange(0, BLOCK_N)
    length = tl.load(len_ptr + token)
    for start in tl.range(0, length, BLOCK_N):
        valid_col = (start + cols) < length
        slots = tl.load(
            idx_ptr + token * topk + start + cols,
            mask=valid_col,
            other=-1,
        )
        valid = valid_col & (slots >= 0)
        rows = tl.where(valid, slots, 0).to(tl.int64)
        kv = tl.load(
            kv_ptr + rows[:, None] * D + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        logits = tl.dot(q, tl.trans(kv)) * (sm_scale * kv_scale)
        logits = tl.where(valid[None, :], logits, -float("inf"))

        max_new = tl.maximum(max_i, tl.max(logits, axis=1))
        max_safe = tl.where(max_new == -float("inf"), 0.0, max_new)
        alpha = tl.exp(max_i - max_safe)
        probs = tl.exp(logits - max_safe[:, None])
        sum_i = sum_i * alpha + tl.sum(probs, axis=1)
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.bfloat16), kv)
        max_i = max_new

    denom = tl.where(sum_i == 0.0, 1.0, sum_i)
    acc = acc * (kv_scale / denom[:, None])
    tl.store(
        out_ptr + token * H * D + heads[:, None] * D + dims[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


@triton.jit
def _union_dense_prefix_prepare_kernel(
    logical_ptr,
    physical_ptr,
    len_ptr,
    req_ptr,
    dense_ptr,
    union_idx_ptr,
    union_bits_ptr,
    union_len_ptr,
    K,
    U_CAP,
    ENABLE: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    token_lanes = tl.arange(0, G)
    lens = tl.load(len_ptr + group * G + token_lanes)
    reqs = tl.load(req_ptr + group * G + token_lanes)
    same_req = tl.min(reqs, axis=0) == tl.max(reqs, axis=0)
    max_len = tl.max(lens, axis=0)
    max_token = tl.argmax(lens, axis=0)
    bad = tl.zeros([], tl.int32)
    cols = tl.arange(0, BLOCK)
    for start in tl.range(0, K, BLOCK):
        col = start + cols
        inb = col < K
        for tok in range(0, G):
            value = tl.load(
                logical_ptr + (group * G + tok) * K + col,
                mask=inb,
                other=-1,
            )
            expected = tl.where(col < lens[tok], col, -1)
            bad += tl.sum((inb & (value != expected)).to(tl.int32))
    dense = ENABLE & same_req & (bad == 0)
    tl.store(dense_ptr + group, dense.to(tl.int8))
    tl.store(union_len_ptr + group, tl.where(dense, max_len, 0))

    for start in tl.range(0, K, BLOCK):
        col = start + cols
        inb = col < K
        physical = tl.load(
            physical_ptr + (group * G + max_token) * K + col,
            mask=inb & (col < max_len),
            other=-1,
        )
        bits = tl.zeros([BLOCK], tl.int32)
        for tok in range(0, G):
            bits |= (col < lens[tok]).to(tl.int32) << tok
        dst = group.to(tl.int64) * U_CAP + col
        tl.store(union_idx_ptr + dst, physical, mask=dense & inb & (col < max_len))
        tl.store(union_bits_ptr + dst, bits, mask=dense & inb & (col < max_len))


@triton.jit
def _union_mark_kernel(
    idx_ptr,
    map_ptr,
    dense_ptr,
    K,
    span,
    base,
    epoch,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid // G
    token = pid % G
    sparse = tl.load(dense_ptr + group) == 0
    cols = tl.arange(0, BLOCK)
    ep8 = tl.full([BLOCK], 0, tl.int8) + epoch
    for start in tl.range(0, K, BLOCK):
        value = tl.load(
            idx_ptr + pid.to(tl.int64) * K + start + cols,
            mask=(start + cols) < K,
            other=-1,
        )
        value -= base
        valid = sparse & (value >= 0) & (value < span)
        addr = (
            group.to(tl.int64) * span
            + tl.where(valid, value, 0).to(tl.int64)
        ) * G + token
        tl.store(map_ptr + addr, ep8, mask=valid)


@triton.jit
def _union_compact_kernel(
    map_ptr,
    dense_ptr,
    union_idx_ptr,
    union_bits_ptr,
    union_len_ptr,
    span,
    U_CAP,
    epoch,
    BLOCK: tl.constexpr,
    LANES: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    sparse = tl.load(dense_ptr + group) == 0
    cols = tl.arange(0, BLOCK)
    cursor = tl.zeros([], tl.int32)
    for start in tl.range(0, span, BLOCK):
        inb = (start + cols) < span
        word = tl.load(
            map_ptr + group * span + start + cols,
            mask=sparse & inb,
            other=0,
        ).to(tl.int32)
        bits = ((word & 255) == epoch).to(tl.int32)
        bits |= (((word >> 8) & 255) == epoch).to(tl.int32) * 2
        if LANES == 4:
            bits |= (((word >> 16) & 255) == epoch).to(tl.int32) * 4
            bits |= (((word >> 24) & 255) == epoch).to(tl.int32) * 8
        present = sparse & inb & (bits != 0)
        position = (
            cursor
            + tl.cumsum(present.to(tl.int32), axis=0)
            - present.to(tl.int32)
        )
        dst = group * U_CAP + position
        tl.store(union_idx_ptr + dst, (start + cols).to(tl.int32), mask=present)
        tl.store(union_bits_ptr + dst, bits, mask=present)
        cursor += tl.sum(present.to(tl.int32))
    tl.store(union_len_ptr + group, cursor, mask=sparse)


@triton.jit
def _glm53_union_prefill_kernel(
    q_ptr,
    kv_ptr,
    union_idx_ptr,
    union_bits_ptr,
    union_len_ptr,
    dense_ptr,
    out_ptr,
    sm_scale,
    kv_scale,
    U_CAP,
    base,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    group = tl.program_id(0)
    GH: tl.constexpr = G * H
    rows = tl.arange(0, GH)
    token_of_row = rows // H
    dims = tl.arange(0, D)
    q = tl.load(
        q_ptr + group.to(tl.int64) * GH * D + rows[:, None] * D + dims[None, :]
    )
    max_i = tl.full([GH], -float("inf"), tl.float32)
    sum_i = tl.zeros([GH], tl.float32)
    acc = tl.zeros([GH, D], tl.float32)
    cols = tl.arange(0, BLOCK_N)
    length = tl.load(union_len_ptr + group)
    dense = tl.load(dense_ptr + group) != 0
    union_base = group.to(tl.int64) * U_CAP
    for start in tl.range(0, length, BLOCK_N):
        inb = (start + cols) < length
        slot = tl.load(
            union_idx_ptr + union_base + start + cols,
            mask=inb,
            other=-1,
        )
        bits = tl.load(
            union_bits_ptr + union_base + start + cols,
            mask=inb,
            other=0,
        )
        valid = inb & (slot >= 0)
        physical = tl.where(dense, slot, slot + base)
        physical = tl.where(valid, physical, 0).to(tl.int64)
        kv = tl.load(
            kv_ptr + physical[:, None] * D + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        logits = tl.dot(q, tl.trans(kv)) * (sm_scale * kv_scale)
        owned = ((bits[None, :] >> token_of_row[:, None]) & 1) != 0
        logits = tl.where(owned & valid[None, :], logits, -float("inf"))

        max_new = tl.maximum(max_i, tl.max(logits, axis=1))
        max_safe = tl.where(max_new == -float("inf"), 0.0, max_new)
        alpha = tl.exp(max_i - max_safe)
        probs = tl.exp(logits - max_safe[:, None])
        sum_i = sum_i * alpha + tl.sum(probs, axis=1)
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.bfloat16), kv)
        max_i = max_new

    denom = tl.where(sum_i == 0.0, 1.0, sum_i)
    acc = acc * (kv_scale / denom[:, None])
    tl.store(
        out_ptr
        + group.to(tl.int64) * GH * D
        + rows[:, None] * D
        + dims[None, :],
        acc.to(out_ptr.dtype.element_ty),
    )


def _topk_length(indices: torch.Tensor) -> torch.Tensor:
    width = indices.shape[-1]
    valid = indices >= 0
    any_valid = valid.any(dim=-1)
    last = width - torch.flip(valid, [-1]).int().argmax(dim=-1)
    return torch.where(any_valid, last, torch.zeros_like(last)).to(torch.int32)


def _base_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    kv_scale: float,
) -> None:
    if not q.shape[0]:
        return
    _glm53_sparse_prefill_kernel[(q.shape[0],)](
        q,
        kv,
        indices,
        lengths,
        out,
        sm_scale,
        kv_scale,
        indices.shape[1],
        H=q.shape[1],
        BLOCK_H=16,
        D=512,
        BLOCK_N=32,
        num_warps=4,
        num_stages=2,
    )


def _decline_kernel(reason: str) -> None:
    """Say why the kernel refused, once, then let the caller serve stock.

    Four `return None`s used to sit here unannotated. Two of them are ordinary
    (a short tail, an empty batch); two are the arm quietly not applying to
    the shapes this fleet actually serves. Nothing distinguished them in a
    log, which is how "ARMED" and "ran" stayed different things.
    """
    if reason not in _UNION_DECLINED:
        _UNION_DECLINED.add(reason)
        logger.warning(
            "union prefill DECLINED in the kernel and the stock path is "
            "serving: %s", reason)
    return None


def glm53_union_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    logical_indices: torch.Tensor,
    physical_indices: torch.Tensor,
    req_ids: torch.Tensor,
    sm_scale: float,
    kv_scale: float,
    group_size: int,
    dense_prefix: bool,
) -> torch.Tensor | None:
    """Return exact sparse MLA output, or ``None`` when the budget rejects it."""
    tokens, heads, dim = q.shape
    # GH = group_size * heads is the kernel's row tile, and every row carries a
    # [D] fp32 accumulator, so it is capped at 32. The hook above admits only
    # (16, 512) q, which makes the cap a statement about the WIDTH:
    #
    #     group_size 2 -> 32 rows, runs
    #     group_size 4 -> 64 rows, never runs
    #
    # width=4 was the value this lane booted with. It returned None here on
    # every prefill and the caller quietly used the stock path -- not an
    # exception, so it did not even show up in the fallback count. Say it
    # once, so "ARMED width=4" and "ran" stop being different things in
    # silence.
    if group_size not in (2, 4) or group_size * heads > 32 or dim != 512:
        if group_size * heads > 32:
            return _decline_kernel(
                f"group_size={group_size} x heads={heads} = "
                f"{group_size * heads} exceeds the kernel's 32-row tile; this "
                f"model gives 16 heads per rank at TP4, so only "
                f"VLLM_GLM53_UNION_PREFILL=2 can run")
        return _decline_kernel(
            f"group_size={group_size} dim={dim} heads={heads}")
    main_tokens = tokens // group_size * group_size
    if main_tokens == 0:
        return _decline_kernel(f"only {tokens} tokens, fewer than one group")
    q = q.contiguous()
    kv = kv.contiguous()
    logical_indices = logical_indices.contiguous()
    physical_indices = physical_indices.contiguous()
    req_ids = req_ids.contiguous()
    lengths = _topk_length(physical_indices)

    valid = physical_indices[:main_tokens] >= 0
    sentinel = torch.full_like(physical_indices[:main_tokens], 2**31 - 1)
    minimum = torch.where(valid, physical_indices[:main_tokens], sentinel).amin()
    maximum = physical_indices[:main_tokens].amax()
    min_slot, max_slot = torch.stack((minimum, maximum)).tolist()
    if max_slot < 0:
        return _decline_kernel("no valid slot in this batch")
    span = max_slot - min_slot + 1
    groups = main_tokens // group_size
    if groups * span * group_size > _UNION_SPAN_BUDGET:
        # The mark buffer is one byte per (group, slot, token), so it scales
        # with the SPAN of physical slots the batch touches -- not with the
        # top-k count. A long context whose selected slots are spread across
        # the cache blows this even though each token still picks only 2048.
        return _decline_kernel(
            f"mark buffer would be {groups * span * group_size / 1e6:.0f} MB "
            f"(groups={groups} x span={span} x G={group_size}), over the "
            f"{_UNION_SPAN_BUDGET / 1e6:.0f} MB budget")

    span_alloc = (span + 4095) // 4096 * 4096
    width = physical_indices.shape[1]
    capacity = group_size * width
    key = (group_size, span_alloc, capacity, q.device)
    buffers = _UNION_WS.get(key)
    if buffers is None or buffers[0].shape[0] < groups:
        buffers = (
            torch.zeros(
                groups,
                span_alloc * group_size,
                dtype=torch.int8,
                device=q.device,
            ),
            torch.empty(groups, capacity, dtype=torch.int32, device=q.device),
            torch.empty(groups, capacity, dtype=torch.int32, device=q.device),
            torch.empty(groups, dtype=torch.int32, device=q.device),
            torch.empty(groups, dtype=torch.int8, device=q.device),
            [0],
        )
        _UNION_WS[key] = buffers
    mark_all, union_idx, union_bits, union_len, dense, epoch_box = buffers
    epoch = epoch_box[0] + 1
    if epoch > 127:
        mark_all.zero_()
        epoch = 1
    epoch_box[0] = epoch
    mark = mark_all[:groups]
    union_idx = union_idx[:groups]
    union_bits = union_bits[:groups]
    union_len = union_len[:groups]
    dense = dense[:groups]

    _union_dense_prefix_prepare_kernel[(groups,)](
        logical_indices[:main_tokens],
        physical_indices[:main_tokens],
        lengths[:main_tokens],
        req_ids[:main_tokens],
        dense,
        union_idx,
        union_bits,
        union_len,
        width,
        capacity,
        ENABLE=dense_prefix,
        G=group_size,
        BLOCK=1024,
        num_warps=4,
    )
    _union_mark_kernel[(groups * group_size,)](
        physical_indices[:main_tokens],
        mark,
        dense,
        width,
        span_alloc,
        min_slot,
        epoch,
        G=group_size,
        BLOCK=1024,
        num_warps=4,
    )
    mark_words = mark.view(torch.int16) if group_size == 2 else mark.view(torch.int32)
    _union_compact_kernel[(groups,)](
        mark_words,
        dense,
        union_idx,
        union_bits,
        union_len,
        span_alloc,
        capacity,
        epoch,
        BLOCK=1024,
        LANES=group_size,
        num_warps=4,
        num_stages=2,
    )

    out = torch.empty(tokens, heads, dim, dtype=torch.bfloat16, device=q.device)
    _glm53_union_prefill_kernel[(groups,)](
        q[:main_tokens],
        kv,
        union_idx,
        union_bits,
        union_len,
        dense,
        out[:main_tokens],
        sm_scale,
        kv_scale,
        capacity,
        min_slot,
        H=heads,
        G=group_size,
        D=dim,
        BLOCK_N=32,
        num_warps=4,
        num_stages=2,
    )
    if _UNION_STATS_SEEN[0] < _UNION_STATS_MAX:
        _UNION_STATS_SEEN[0] += 1
        per_token = lengths[:main_tokens].float().mean().item()
        union_mean = union_len.float().mean().item()
        ideal = per_token * group_size
        logger.warning(
            "[union-prefill] #%d groups=%d G=%d per-token-topk=%.0f "
            "|union|=%.0f (of %.0f if disjoint) -> gather saved %.1f pct",
            _UNION_STATS_SEEN[0], groups, group_size, per_token, union_mean,
            ideal, 100.0 * (1.0 - union_mean / ideal) if ideal else 0.0,
        )
    _base_sparse_prefill(
        q[main_tokens:],
        kv,
        physical_indices[main_tokens:],
        lengths[main_tokens:],
        out[main_tokens:],
        sm_scale,
        kv_scale,
    )
    return out


_UNION_REPORTED: set = set()

# Shadow: run the union path AND the FlashInfer path, compare, log, and serve
# FlashInfer's answer. The module has always claimed its output is exact, and
# that claim had never once been tested on the fleet -- the width bug meant
# every prefill took the fallback, so "ARMED" and "ran" were different things
# for the whole life of this arm. The first boot where it actually ran was
# also the first with a U+FFFD in Korean output, which is either a real
# coincidence or this arm, and statistics on one event cannot say which.
#
# Same idiom as the megakernel's KDA/MLA shadows: cost a step, keep the stock
# answer, and turn "is it exact?" into a number in the boot log.
_UNION_SHADOW_ENV = "VLLM_GLM53_UNION_PREFILL_SHADOW"
_UNION_SHADOW_SEEN: list = [0]
_UNION_SHADOW_MAX = 32


def _union_shadow_enabled() -> bool:
    return os.environ.get(_UNION_SHADOW_ENV, "").strip() == "1"


# The column tile triton_convert_req_index_to_global_index defaults to and
# asserts against. Named here because the width handed to it must be a
# multiple of it, and the assertion that catches a mismatch is inside vLLM.
_CONVERT_BLOCK_N = 128


def _read_group_size() -> int:
    raw = os.environ.get(_UNION_ENV, "0")
    try:
        value = int(raw)
    except ValueError:
        value = -1
    group = value if value in (2, 4) else 0
    # Only 2 and 4 are real widths. Anything else -- notably 1, the value a
    # "turn everything on" sweep reaches for -- reads as off, and used to do
    # so without a word. A knob that silently means its opposite is how this
    # lane has repeatedly measured a baseline and called it a result.
    if raw not in ("0", "") and not group and raw not in _UNION_REPORTED:
        _UNION_REPORTED.add(raw)
        logger.warning(
            "%s=%s is not a union width (only 2 or 4); union prefill is OFF",
            _UNION_ENV, raw,
        )
    return group


def _glm53_union_forward_mqa(self, q, kv_cache, attn_metadata, layer):
    original = type(self)._glm53_union_original_forward_mqa
    group_size = _read_group_size()

    def decline(reason: str):
        """Say why once, then serve stock.

        Every condition here is a way this arm can silently not run, and
        silence is how it spent its whole life logging "ARMED" without ever
        executing -- first a width the converter rejected, then a row cap that
        made the configured width impossible, and neither left a mark."""
        if reason not in _UNION_DECLINED:
            _UNION_DECLINED.add(reason)
            logger.warning(
                "union prefill DECLINED at the entry gate and the stock path "
                "is serving: %s", reason)
        return original(self, q, kv_cache, attn_metadata, layer)

    if not group_size:
        return original(self, q, kv_cache, attn_metadata, layer)
    if torch.cuda.is_current_stream_capturing():
        return original(self, q, kv_cache, attn_metadata, layer)
    if not (isinstance(q, tuple) and len(q) == 2):
        return decline(f"q is {type(q).__name__}, not the (nope, rope) pair")
    nope, rope = q
    for reason, ok in (
        ("q is not a cuda bf16 3-d tensor",
         nope.is_cuda and nope.dtype == torch.bfloat16 and nope.ndim == 3),
        (f"q heads/dim {tuple(nope.shape[1:])} != (16, 512)",
         nope.ndim == 3 and tuple(nope.shape[1:]) == (16, 512)),
        (f"rope part width {rope.shape[-1]} != 0", rope.shape[-1] == 0),
        ("kv cache is not fp8",
         bool(getattr(self, "use_fp8_kv_cache", False))),
        (f"qk_rope_head_dim={getattr(self, 'qk_rope_head_dim', -1)} != 0",
         getattr(self, "qk_rope_head_dim", -1) == 0),
        (f"kv_lora_rank={getattr(self, 'kv_lora_rank', -1)} != 512",
         getattr(self, "kv_lora_rank", -1) == 512),
        (f"num_decodes={getattr(attn_metadata, 'num_decodes', -1)} != 0 "
         "(mixed batch)",
         getattr(attn_metadata, "num_decodes", -1) == 0),
        ("no prefill in this batch",
         getattr(attn_metadata, "num_prefills", 0) > 0),
        (f"num_actual_tokens="
         f"{getattr(attn_metadata, 'num_actual_tokens', -1)} != {nope.shape[0]}"
         " q rows",
         getattr(attn_metadata, "num_actual_tokens", -1) == nope.shape[0]),
        (f"topk_tokens={getattr(attn_metadata, 'topk_tokens', -1)} != 2048",
         getattr(attn_metadata, "topk_tokens", -1) == 2048),
        (f"cp_interleave="
         f"{getattr(attn_metadata, 'cp_kv_cache_interleave_size', -1)} != 1",
         getattr(attn_metadata, "cp_kv_cache_interleave_size", -1) == 1),
    ):
        if not ok:
            return decline(reason)

    try:
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )

        tokens = q[0].shape[0]
        # KPool emits a fixed 2048-token selection plus at most three live
        # tail tokens, so 2051 columns carry data. The cells beyond that are
        # rounded scratch capacity: outside FlashInfer's planned valid length,
        # and they may retain data from an older, wider batch.
        #
        # 2051 cannot be handed to the converter, which tiles columns and
        # asserts NUM_TOPK_TOKENS % BLOCK_N == 0. 2051 = 7 x 293, so no usable
        # BLOCK_N divides it, and passing it raised on EVERY prefill:
        #
        #     AssertionError: NUM_TOPK_TOKENS (2051) must be divisible by
        #     BLOCK_N (128)
        #
        # caught by the except below, logged, and fallen back to FlashInfer --
        # 11 prefills, 11 fallbacks, in a boot whose log said "union prefill:
        # ARMED width=4". This arm has never once run.
        #
        # So round the width up to the tile and mark the rounded tail -1,
        # which the converter documents as "invalid": "Only when
        # token_indices[...] == -1 do we output -1". The copy is the price of
        # not mutating a buffer the fallback path also reads -- tokens x 2176
        # int32, about 0.9 pct of a 32K prefill against the ~12 pct this arm
        # is supposed to be worth.
        want = 2051
        width = -(-want // _CONVERT_BLOCK_N) * _CONVERT_BLOCK_N
        raw = self.topk_indices_buffer[:tokens]
        carried = min(want, raw.shape[1])
        if raw.shape[1] >= width:
            logical = raw[:, :width].clone()
        else:
            logical = raw.new_full((tokens, width), -1)
            logical[:, :carried] = raw[:, :carried]
        logical[:, carried:] = -1
        assert logical.shape[1] % _CONVERT_BLOCK_N == 0
        physical, _ = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:tokens],
            attn_metadata.block_table,
            logical,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=logical.shape[1],
            BLOCK_N=_CONVERT_BLOCK_N,
            return_valid_counts=True,
        )
        if os.environ.get("VLLM_GLM53_UNION_PREFILL_STATS", "") == "1":
            _measure_overlap(physical, group_size, f"tokens={tokens}")
        flat_kv = kv_cache.view(torch.float8_e4m3fn).reshape(-1, 512)
        output = glm53_union_sparse_prefill(
            q[0],
            flat_kv,
            logical,
            physical,
            attn_metadata.req_id_per_token[:tokens],
            float(self.scale),
            float(layer._k_scale_float or 1.0),
            group_size,
            os.environ.get(_DENSE_PREFIX_ENV, "0") == "1",
        )
        if output is not None:
            if _union_shadow_enabled():
                # The stock answer is the one served; the union answer is only
                # measured. Bounded so a long run does not pay for it forever.
                if _UNION_SHADOW_SEEN[0] < _UNION_SHADOW_MAX:
                    _UNION_SHADOW_SEEN[0] += 1
                    ref = original(self, q, kv_cache, attn_metadata, layer)
                    ref_t = ref[0] if isinstance(ref, tuple) else ref
                    diff = (output.float() - ref_t.float())
                    denom = ref_t.float().norm().item() or 1.0
                    logger.warning(
                        "[union-prefill] shadow #%d rel=%.3e max=%.3e "
                        "tokens=%d group=%d",
                        _UNION_SHADOW_SEEN[0],
                        diff.norm().item() / denom,
                        diff.abs().amax().item(),
                        tokens, group_size,
                    )
                    return ref
                return original(self, q, kv_cache, attn_metadata, layer)
            return output, None
    except Exception:
        logger.warning(
            "GLM union sparse prefill failed; using FlashInfer.",
            exc_info=True,
        )
    return original(self, q, kv_cache, attn_metadata, layer)


def install_glm53_union_prefill() -> bool:
    """Install the exact-gated backend wrapper once when the feature is armed."""
    if not _read_group_size():
        return False
    try:
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
            FlashInferMLASparseSM90Impl,
        )
    except Exception:
        return False
    if getattr(FlashInferMLASparseSM90Impl, "_glm53_union_installed", False):
        return True
    FlashInferMLASparseSM90Impl._glm53_union_original_forward_mqa = (
        FlashInferMLASparseSM90Impl.forward_mqa
    )
    FlashInferMLASparseSM90Impl.forward_mqa = _glm53_union_forward_mqa
    FlashInferMLASparseSM90Impl._glm53_union_installed = True
    logger.info(
        "glm53 union prefill: ARMED width=%d dense_prefix=%s",
        _read_group_size(),
        os.environ.get(_DENSE_PREFIX_ENV, "0") == "1",
    )
    return True
