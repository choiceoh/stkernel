# SPDX-License-Identifier: Apache-2.0
"""Profile-selected eager GLM TP4 prefill collectives for sharded residual/MHC work.

The model gathers normalized local rows before each unchanged TP attention or
MLP. A lexical scope defers its single terminal all-reduce; reduce-scatter
then sums the rank partials directly into the local residual shard. Decode
never enters this scope. The profile selects lossy FP8 v3 transport for
large chunks and native BF16 for short chunks; mode 0 uses BF16 throughout.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os

import torch
from torch.distributed import ReduceOp
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)
_ENABLED = os.environ.get("VLLM_GLM53_PREFILL_SP") == "1"
_FP8_MODE = os.environ.get("VLLM_GLM53_PREFILL_SP_FP8", "0").strip()
_FP8 = _FP8_MODE in ("1", "2", "3")
# 39차 v2 (operator "두개 도전해"): the v1 FP8 transport lost to BF16 (+7.1 % vs
# +12.6 %) because its reduce-scatter first agreed on block maxima with an extra
# all-reduce, then summed FP8 partials natively (headroom 448/4, rounding), then
# decoded -- three kernels and two collectives per reduction. v2 keeps the FP8
# all-gather (pack / all-gather / unpack) and turns the reduce-scatter into an
# all-to-all of per-block-scaled FP8 partials that the receiver sums in FP32:
# one pack kernel (absmax + quantize fused, no scale agreement), two NCCL
# all-to-alls (values and scales), one unpack-sum kernel.
_FP8_V2 = _FP8_MODE == "2"
_FP8_V3 = _FP8_MODE == "3"
# A serving 2K request lost 52--86 ms to unconditional v3, while 6912-row
# chunks dominate the long requests that improved. Start between those
# measured shapes; this is a candidate boundary, not a measured crossover.
# Latch the same profile setting on all ranks. Zero preserves the ungated
# v3 control for paired experiments; historical v1/v2 behavior is unchanged.
_FP8_V3_MIN_TOKENS = int(os.environ.get("VLLM_GLM53_PREFILL_SP_FP8_MIN_TOKENS", "4096"))
if _FP8_V3_MIN_TOKENS < 0:
    raise ValueError("prefill FP8 minimum tokens must be nonnegative")
_FP8_AG_MIN_TOKENS = int(os.environ.get("VLLM_GLM53_PREFILL_SP_FP8_AG_MIN_TOKENS", "-1"))
_FP8_RS_MIN_TOKENS = int(os.environ.get("VLLM_GLM53_PREFILL_SP_FP8_RS_MIN_TOKENS", "-1"))
if min(_FP8_AG_MIN_TOKENS, _FP8_RS_MIN_TOKENS) < -1:
    raise ValueError("per-collective FP8 minimum must be -1 (inherit) or nonnegative")
_FUSE_MHC = os.environ.get("VLLM_GLM53_PREFILL_SP_FUSE_MHC") == "1"
_DIRECT_NCCL = os.environ.get("VLLM_GLM53_PREFILL_SP_DIRECT_NCCL") == "1"
# v2 actually launches TWO all-to-alls: values and scales. v3 packs each
# destination's scales after its values, so both travel in one byte exchange.
# Quantization and FP32 accumulation stay the same; this targets call cost.
if _ENABLED:
    # 39차: the boot-log anchor the bracket greps -- an armed knob is not
    # evidence of invocation, but a missing anchor IS evidence of no arming.
    logger.warning("[prefill-sp] sequence-parallel prefill armed (fp8 transport=%s%s)", _FP8,
                   " v3: packed values+scales, one all-to-all + fp32 sum" if _FP8_V3 else
                   " v2: per-block scales, two all-to-alls + fp32 sum" if _FP8_V2 else "")
_BLOCK = 2048
_TP = 4
_HIDDEN = 4096
_PARTIAL = ContextVar("glm53_prefill_partial", default=None)


@dataclass
class _PartialOutput:
    communicator: object
    num_tokens: int
    reductions: int = 0


@dataclass(frozen=True)
class PackedPrefillRows:
    """Invocation-owned receive storage, consumed only by eager MHC wrappers.

    Keep the receive tensor alive through both auxiliary reconstruction and
    the following layer. Neither consumer mutates it or a shared workspace.
    This object never enters a torch custom op or a captured decode graph.
    """
    payload: object
    rows: int
    payload_bytes: int


def is_packed_prefill(value):
    return isinstance(value, PackedPrefillRows)


def _tp_comm():
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    comm = group.device_communicator
    if not _ENABLED or group.world_size != _TP or comm is None:
        raise RuntimeError("prefill SP requires its explicit knob and a TP4 communicator")
    pynccl = comm.pynccl_comm
    if pynccl is None or pynccl.disabled:
        raise RuntimeError("prefill SP requires an active PyNCCL communicator")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prefill SP is an eager prefill experiment")
    return comm, pynccl


@contextmanager
def partial_tp_output(*, num_tokens):
    """Defer exactly one full-token TP output reduction until reduce-scatter.

    The terminal operator still returns full-token rank partials, so existing
    attention/MoE output reshapes and shared-expert combination remain valid.
    This must never swallow errors or use a rank-local fallback after entering
    a collective sequence. The model only admits the pinned pure-prefill path.
    """
    comm, _ = _tp_comm()
    if _PARTIAL.get() is not None or num_tokens < 128:
        raise RuntimeError("invalid or nested prefill partial-output scope")
    scope = _PartialOutput(comm, num_tokens)
    token = _PARTIAL.set(scope)
    try:
        yield
        if scope.reductions != 1:
            raise RuntimeError(
                f"prefill SP expected one terminal TP reduction, observed {scope.reductions}"
            )
    finally:
        _PARTIAL.reset(token)


def maybe_partial_all_reduce(comm, tensor):
    """Called before ordinary dispatch; None leaves all existing arms intact."""
    scope = _PARTIAL.get()
    if scope is None or comm is not scope.communicator:
        return None
    if (tensor.ndim != 2 or tuple(tensor.shape) != (scope.num_tokens, _HIDDEN)
            or tensor.dtype != torch.bfloat16 or not tensor.is_cuda
            or not tensor.is_contiguous() or scope.reductions != 0):
        raise RuntimeError("prefill SP encountered an unexpected TP reduction contract")
    scope.reductions += 1
    return tensor


def _check(tensor):
    # These are PyNcclCommunicator's in-place (out, input) collectives.
    # CudaCommunicator has a different API and is only used for scope identity.
    _, pynccl = _tp_comm()
    if (tensor.ndim != 2 or tensor.shape[0] < 32 or tensor.shape[1] != _HIDDEN
            or tensor.dtype != torch.bfloat16 or not tensor.is_cuda
            or tensor.device != pynccl.device or not tensor.is_contiguous()):
        raise ValueError("prefill collective requires contiguous CUDA BF16 [rows,4096]")
    return pynccl


def _payload_bytes(local_n):
    # Appending 4-byte scales skews peer-packet alignment for odd shard row
    # counts. Keep every sender/receiver slice aligned, not just allocation 0.
    used = local_n + 4 * (local_n // _BLOCK)
    return ((used + 127) // 128) * 128


def _use_fp8(num_tokens, *, reduce_scatter=False):
    # Shape metadata is host-side and identical across TP ranks. Never use a
    # local shard length, request's total context, or tensor values here.
    if not _FP8 or not _FP8_V3:
        return _FP8
    threshold = _FP8_RS_MIN_TOKENS if reduce_scatter else _FP8_AG_MIN_TOKENS
    if threshold < 0:
        threshold = _FP8_V3_MIN_TOKENS
    elif reduce_scatter:
        logger.info_once("[prefill-sp] reduce-scatter FP8 length gate engaged")
    else:
        logger.info_once("[prefill-sp] all-gather FP8 length gate engaged")
    return num_tokens >= threshold


# 39차 P2A2: the element counts were tl.constexpr, so every distinct prefill
# length (every prompt) recompiled six kernels -- the 2K request after the
# 8192-token warm-up paid 0.5 s (cold 1.48 s vs warm 0.87 s). They are
# runtime integers now (do_not_specialize keeps Triton from re-specializing
# on divisibility); the masks were already runtime.
@triton.jit(do_not_specialize=["N"])
def _block_absmax(X, Max, N, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(Max + block, tl.max(tl.abs(x), 0))


@triton.jit(do_not_specialize=["N", "OUT_N"])
def _encode(X, Max, Packed, Scale, N, OUT_N,
            LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    amax = tl.load(Max + block)
    # Shared pow2 scaling avoids an extra arbitrary-multiply rounding axis.
    # RS reserves room for all four inputs to have the same sign (448/4).
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 1.0e-30) / LIMIT)))
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    q = (x / scale).to(tl.float8e4nv)
    # Full padding rows are zero under the actual-input mask. H=4096 keeps
    # them in separate scale blocks from every real row.
    tl.store(Packed + offsets, q, offsets < OUT_N)
    tl.store(Scale + block, scale)


@triton.jit(do_not_specialize=["N", "OUT_N"])
def _copy_pad(X, Out, N, OUT_N,
              BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Out + offsets, x, offsets < OUT_N)


@triton.jit(do_not_specialize=["N"])
def _decode(Packed, Scale, Out, N, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    # 39차 P2A: `other=0` on an FP8 pointer does not compile on the image's
    # Triton ("cannot cast int32[constexpr] to fp8e4nv") -- the first SP-
    # admitted prefill killed the boot. Masked lanes are never stored, so no
    # fill value is needed.
    q = tl.load(Packed + offsets, mask=offsets < N).to(tl.float32)
    scale = tl.load(Scale + block)
    tl.store(Out + offsets, q * scale, offsets < N)


@triton.jit(do_not_specialize=["N"])
def _pack_gather(X, Packed, Scale, N, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 1.0e-30) / 448.0)))
    tl.store(Packed + offsets, (x / scale).to(tl.float8e4nv), offsets < N)
    tl.store(Scale + block, scale)


@triton.jit(do_not_specialize=["LOCAL_N", "PAYLOAD_BYTES"])
def _unpack_gather(Packed, Scales, Out, LOCAL_N,
                   PAYLOAD_BYTES, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    local_blocks = LOCAL_N // BLOCK
    rank = block // local_blocks
    local_block = block % local_blocks
    offsets = local_block * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Packed + rank * PAYLOAD_BYTES + offsets).to(tl.float32)
    scale = tl.load(Scales + rank * (PAYLOAD_BYTES // 4)
                    + LOCAL_N // 4 + local_block)
    tl.store(Out + rank * LOCAL_N + offsets, q * scale)


@triton.jit(do_not_specialize=["N", "OUT_N"])
def _pack_rs(X, Packed, Scale, N, OUT_N, BLOCK: tl.constexpr):
    """v2 reduce-scatter pack: one block (2048 elements, half a row) -> its own
    absmax scale + FP8 payload. Rows past N (the padding rows) pack as zero.
    Blocks are laid out in row order, so destination rank r's shard is the
    contiguous byte range [r*OUT_N/4, (r+1)*OUT_N/4) -- all_to_all_single
    with equal splits sends it without any reordering."""
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, mask=offsets < N, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 1.0e-30) / 448.0)))
    tl.store(Packed + offsets, (x / scale).to(tl.float8e4nv), mask=offsets < OUT_N)
    tl.store(Scale + block, scale)


@triton.jit(do_not_specialize=["LOCAL_N", "NUM_BLOCKS"])
def _unpack_sum(Packed, Scales, Out, LOCAL_N, NUM_BLOCKS, TP: tl.constexpr,
                BLOCK: tl.constexpr):
    """v2 reduce-scatter unpack: sum the TP partials of one local block in
    FP32 -- partial r sits at byte offset r*LOCAL_N of the received payload
    and its scales at r*NUM_BLOCKS."""
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for r in tl.static_range(TP):
        q = tl.load(Packed + r * LOCAL_N + offsets).to(tl.float32)
        scale = tl.load(Scales + r * NUM_BLOCKS + block)
        acc += q * scale
    tl.store(Out + offsets, acc)


@triton.jit(do_not_specialize=["N", "LOCAL_N", "PAYLOAD_BYTES"])
def _pack_rs_payload(X, Packed, Scales, N, LOCAL_N, PAYLOAD_BYTES,
                     BLOCK: tl.constexpr):
    block = tl.program_id(0)
    local_blocks = LOCAL_N // BLOCK
    rank = block // local_blocks
    local_block = block % local_blocks
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, mask=offsets < N, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 1.0e-30) / 448.0)))
    # Each equal-sized destination packet is [FP8 values | FP32 scales].
    # LOCAL_N is whole BF16 rows, so both views and every packet are aligned.
    local_offsets = local_block * BLOCK + tl.arange(0, BLOCK)
    tl.store(Packed + rank * PAYLOAD_BYTES + local_offsets,
             (x / scale).to(tl.float8e4nv))
    tl.store(Scales + rank * (PAYLOAD_BYTES // 4) + LOCAL_N // 4 + local_block,
             scale)
    if local_block == local_blocks - 1:
        # Initialize the alignment gap in this packet. One CTA owns it and
        # its at-most-31 FP32 words never overlap actual scales or values.
        tail = tl.arange(0, 32)
        scale_end = LOCAL_N // 4 + local_blocks
        tl.store(Scales + rank * (PAYLOAD_BYTES // 4) + scale_end + tail, 0.0,
                 mask=scale_end + tail < PAYLOAD_BYTES // 4)


@triton.jit(do_not_specialize=["LOCAL_N", "PAYLOAD_BYTES"])
def _unpack_sum_payload(Packed, Scales, Out, LOCAL_N, PAYLOAD_BYTES,
                        TP: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    # all_to_all_single returns packets in source-rank order, matching v2.
    for rank in tl.static_range(TP):
        q = tl.load(Packed + rank * PAYLOAD_BYTES + offsets).to(tl.float32)
        scale = tl.load(Scales + rank * (PAYLOAD_BYTES // 4)
                        + LOCAL_N // 4 + block)
        acc += q * scale
    tl.store(Out + offsets, acc)


@triton.jit(do_not_specialize=["LOCAL_N", "PAYLOAD_BYTES"])
def _unpack_sum_mhc_post(Packed, Scales, Residual, Post, Comb, Out,
                         LOCAL_N, PAYLOAD_BYTES, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offsets = row * 4096 + h
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for rank in tl.static_range(4):
        q = tl.load(Packed + rank * PAYLOAD_BYTES + offsets).to(tl.float32)
        scale = tl.load(Scales + rank * (PAYLOAD_BYTES // 4)
                        + LOCAL_N // 4 + offsets // 2048)
        acc += q * scale
    # Preserve the old unpack store/load's BF16 rounding BEFORE hc_post.
    x = acc.to(tl.bfloat16).to(tl.float32)
    base = row * (4 * 4096) + h
    r0 = tl.load(Residual + base).to(tl.float32)
    r1 = tl.load(Residual + base + 4096).to(tl.float32)
    r2 = tl.load(Residual + base + 8192).to(tl.float32)
    r3 = tl.load(Residual + base + 12288).to(tl.float32)
    for hc in tl.static_range(4):
        p = tl.load(Post + row * 4 + hc)
        updated = p * x
        updated = tl.fma(tl.load(Comb + row * 16 + hc), r0, updated)
        updated = tl.fma(tl.load(Comb + row * 16 + 4 + hc), r1, updated)
        updated = tl.fma(tl.load(Comb + row * 16 + 8 + hc), r2, updated)
        updated = tl.fma(tl.load(Comb + row * 16 + 12 + hc), r3, updated)
        tl.store(Out + base + hc * 4096, updated)


def prefill_mhc_post(packed, residual, post, comb):
    """Decode the ordered FP8 sum directly inside the stock MHC post map."""
    if not isinstance(packed, PackedPrefillRows):
        raise ValueError("MHC fused input must be packed prefill rows")
    rows = packed.rows
    payload = packed.payload
    if (rows < 32
            or packed.payload_bytes != _payload_bytes(rows * _HIDDEN)
            or tuple(payload.shape) != (packed.payload_bytes * _TP,)
            or payload.dtype != torch.uint8 or not payload.is_cuda
            or not payload.is_contiguous()
            or torch.cuda.is_current_stream_capturing()):
        raise ValueError("invalid eager packed prefill MHC input")
    if tuple(post.shape) == (rows, 4, 1):
        post = post.view(rows, 4)
    for tensor, shape, dtype in (
        (residual, (rows, 4, _HIDDEN), torch.bfloat16),
        (post, (rows, 4), torch.float32),
        (comb, (rows, 4, 4), torch.float32),
    ):
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                or tensor.device != payload.device or not tensor.is_contiguous()):
            raise ValueError("packed prefill MHC requires contiguous BF16 residual and FP32 mixes")
    out = torch.empty_like(residual)
    _unpack_sum_mhc_post[(rows, 4)](
        payload.view(torch.float8_e4m3fn), payload.view(torch.float32),
        residual, post, comb, out, LOCAL_N=rows * _HIDDEN,
        PAYLOAD_BYTES=packed.payload_bytes, BLOCK=1024, num_warps=4,
    )
    logger.info_once("[prefill-sp] FP8 unpack and MHC post fused")
    return out


def _quantize(tensor, maxima, limit, *, num_rows=None):
    shape = tensor.shape if num_rows is None else (num_rows, _HIDDEN)
    packed = torch.empty(shape, device=tensor.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty_like(maxima)
    _encode[(maxima.numel(),)](
        tensor, maxima, packed, scales, N=tensor.numel(), OUT_N=packed.numel(),
        LIMIT=limit, BLOCK=_BLOCK
    )
    return packed, scales


def _maxima(tensor, *, padded_numel=None):
    n = tensor.numel() if padded_numel is None else padded_numel
    maxima = torch.empty(triton.cdiv(n, _BLOCK),
                         device=tensor.device, dtype=torch.float32)
    _block_absmax[(maxima.numel(),)](tensor, maxima, N=tensor.numel(), BLOCK=_BLOCK)
    return maxima


def prefill_shard(tensor):
    """Shard real rows with at most three trailing padding rows globally.

    Each rank already owns the full embedding output. Return a contiguous
    view when its shard is full; only the final partial shard needs a local
    copy. Padding remains in row-local MHC and is trimmed before consumers.
    """
    _check(tensor)
    if tensor.shape[0] < 128:
        raise ValueError("prefill sharding requires at least 128 real rows")
    from vllm.distributed import get_tensor_model_parallel_rank

    rows = triton.cdiv(tensor.shape[0], _TP)
    start = get_tensor_model_parallel_rank() * rows
    shard = tensor[start:min(start + rows, tensor.shape[0])]
    if shard.shape[0] == rows:
        return shard
    out = torch.empty((rows, _HIDDEN), device=tensor.device, dtype=tensor.dtype)
    _copy_pad[(triton.cdiv(out.numel(), _BLOCK),)](
        shard, out, N=shard.numel(), OUT_N=out.numel(), BLOCK=_BLOCK
    )
    return out


def prefill_all_gather(tensor, *, num_tokens=None):
    """Gather normalized shards and optionally trim their trailing padding."""
    comm = _check(tensor)
    padded_rows = tensor.shape[0] * _TP
    if (num_tokens is not None
            and (type(num_tokens) is not int
                 or not padded_rows - _TP < num_tokens <= padded_rows)):
        raise ValueError("prefill gather expects the real row count of its TP4 shards")
    out = torch.empty((padded_rows, _HIDDEN),
                      device=tensor.device, dtype=tensor.dtype)
    real_rows = padded_rows if num_tokens is None else num_tokens
    if not _use_fp8(real_rows):
        if _FP8_V3:
            logger.info_once("[prefill-sp] short chunk BF16 all-gather engaged")
        comm.all_gather(out, tensor)
        return out if num_tokens is None else out[:num_tokens]
    # One payload contains FP8 bytes followed by FP32 scales. All-gather has
    # no arithmetic, so UINT8 transport is sufficient. This avoids a second
    # metadata collective and fuses the local absmax with quantization.
    n = tensor.numel()
    blocks = n // _BLOCK  # H=4096 guarantees complete, aligned blocks.
    payload_bytes = _payload_bytes(n) if _FP8_V3 else n + 4 * blocks
    payload = torch.empty(payload_bytes, device=tensor.device, dtype=torch.uint8)
    if _FP8_V3:
        # The RS encoder also packs a single local AG packet, including its
        # alignment gap. Quantization is identical to _pack_gather.
        _pack_rs_payload[(blocks,)](
            tensor, payload.view(torch.float8_e4m3fn), payload.view(torch.float32),
            N=n, LOCAL_N=n, PAYLOAD_BYTES=payload_bytes, BLOCK=_BLOCK,
        )
    else:
        _pack_gather[(blocks,)](
            tensor, payload[:n].view(torch.float8_e4m3fn),
            payload[n:].view(torch.float32), N=n, BLOCK=_BLOCK
        )
    gathered = torch.empty(payload_bytes * _TP, device=tensor.device,
                           dtype=torch.uint8)
    comm.all_gather(gathered, payload)
    _unpack_gather[(blocks * _TP,)](
        gathered.view(torch.float8_e4m3fn), gathered.view(torch.float32), out,
        LOCAL_N=n, PAYLOAD_BYTES=payload_bytes, BLOCK=_BLOCK
    )
    return out if num_tokens is None else out[:num_tokens]


def prefill_reduce_scatter(tensor, *, defer_mhc_post=False):
    """Reduce full-token rank partials directly to the local MHC shard.

    FP8 v1 agrees on block maxima before native FP8 SUM. v2/v3 instead send
    each rank's local scales with its values and sum the decoded terms in
    FP32 on the receiver. All FP8 modes change precision and need quality
    validation. Metadata collectives and quantization costs count in timing.
    """
    comm = _check(tensor)
    rows = triton.cdiv(tensor.shape[0], _TP)
    padded_rows = rows * _TP
    use_fp8 = _use_fp8(tensor.shape[0], reduce_scatter=True)
    defer = _FUSE_MHC and defer_mhc_post and _FP8_V3 and use_fp8
    out = None if defer else torch.empty((rows, _HIDDEN),
                                        device=tensor.device, dtype=tensor.dtype)
    if not use_fp8:
        if _FP8_V3:
            logger.info_once("[prefill-sp] short chunk BF16 reduce-scatter engaged")
        if padded_rows != tensor.shape[0]:
            padded = torch.empty((padded_rows, _HIDDEN),
                                 device=tensor.device, dtype=tensor.dtype)
            _copy_pad[(triton.cdiv(padded.numel(), _BLOCK),)](
                tensor, padded, N=tensor.numel(), OUT_N=padded.numel(), BLOCK=_BLOCK
            )
            tensor = padded
        comm.reduce_scatter(out, tensor)
        return out
    if _FP8_V3:
        return _reduce_scatter_v3(tensor, out, padded_rows)
    if _FP8_V2:
        return _reduce_scatter_v2(tensor, out, padded_rows)
    # Encode padding directly into the FP8 payload: a full BF16 pad/copy
    # before each collective would erase a material part of the wire saving.
    maxima = _maxima(tensor, padded_numel=padded_rows * _HIDDEN)
    maxima = comm.all_reduce(maxima, op=ReduceOp.MAX)
    if maxima is None:
        raise RuntimeError("PyNCCL disabled during prefill FP8 scale agreement")
    packed, scales = _quantize(tensor, maxima, 448.0 / _TP, num_rows=padded_rows)
    reduced = torch.empty(out.shape, device=tensor.device, dtype=packed.dtype)
    comm.reduce_scatter(reduced, packed)
    # H=4096 is a multiple of BLOCK; each rank boundary is block aligned.
    from vllm.distributed import get_tensor_model_parallel_rank

    scale_count = out.numel() // _BLOCK
    start = get_tensor_model_parallel_rank() * scale_count
    _decode[(scale_count,)](
        reduced, scales[start:start + scale_count], out,
        N=out.numel(), BLOCK=_BLOCK
    )
    return out



def _reduce_scatter_v2(tensor, out, padded_rows):
    """FP8 v2 reduce-scatter: per-block scales travel with the payload, the
    receiver sums the TP partials in FP32. No scale-agreement collective, no
    native FP8 SUM (so no headroom loss); NCCL all-to-all moves the same bytes
    the FP8 reduce-scatter did (3/4 of the payload leaves each rank)."""
    from vllm.distributed import get_tp_group

    n_padded = padded_rows * _HIDDEN
    blocks = n_padded // _BLOCK                 # H=4096 -> 2 blocks per row
    local_n = n_padded // _TP
    local_blocks = blocks // _TP
    # payload: [fp8 bytes (n_padded)] [fp32 scales (blocks)] with every rank's
    # shard a contiguous slice of each part, so one all_to_all per part
    packed = torch.empty(n_padded, device=tensor.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(blocks, device=tensor.device, dtype=torch.float32)
    _pack_rs[(blocks,)](tensor, packed, scales, N=tensor.numel(), OUT_N=n_padded, BLOCK=_BLOCK)
    group = get_tp_group().device_group
    recv = torch.empty(n_padded, device=tensor.device, dtype=torch.float8_e4m3fn)
    recv_scales = torch.empty(blocks, device=tensor.device, dtype=torch.float32)
    torch.distributed.all_to_all_single(recv.view(torch.uint8), packed.view(torch.uint8), group=group)
    torch.distributed.all_to_all_single(recv_scales, scales, group=group)
    _unpack_sum[(local_blocks,)](
        recv, recv_scales, out, LOCAL_N=local_n, NUM_BLOCKS=local_blocks, TP=_TP, BLOCK=_BLOCK
    )
    return out


def _reduce_scatter_v3(tensor, out, padded_rows):
    """Same v2 quantization/reduction, with one values+scales exchange.

    Every rank sends TP equal packets; no extra interleave/copy kernel is
    needed. Both temporary allocations belong to this invocation so calls
    on different streams cannot share or overwrite a scratch buffer.
    """
    local_n = padded_rows * _HIDDEN // _TP
    local_blocks = local_n // _BLOCK
    payload_bytes = _payload_bytes(local_n)
    packed = torch.empty(payload_bytes * _TP, device=tensor.device,
                         dtype=torch.uint8)
    recv = torch.empty_like(packed)
    _pack_rs_payload[(local_blocks * _TP,)](
        tensor, packed.view(torch.float8_e4m3fn), packed.view(torch.float32),
        N=tensor.numel(), LOCAL_N=local_n, PAYLOAD_BYTES=payload_bytes, BLOCK=_BLOCK,
    )
    _exchange_packets(recv, packed, payload_bytes)
    if out is None:
        logger.info_once("[prefill-sp] packed FP8 reduce-scatter engaged (one values+scales all-to-all)")
        return PackedPrefillRows(recv, padded_rows // _TP, payload_bytes)
    _unpack_sum_payload[(local_blocks,)](
        recv.view(torch.float8_e4m3fn), recv.view(torch.float32), out,
        LOCAL_N=local_n, PAYLOAD_BYTES=payload_bytes, TP=_TP, BLOCK=_BLOCK,
    )
    logger.info_once("[prefill-sp] packed FP8 reduce-scatter engaged (one values+scales all-to-all)")
    return out


def _exchange_packets(recv, packed, payload_bytes):
    if not _DIRECT_NCCL:
        from vllm.distributed import get_tp_group
        torch.distributed.all_to_all_single(recv, packed, group=get_tp_group().device_group)
        return
    _, comm = _tp_comm()
    if (comm.world_size != _TP or payload_bytes <= 0
            or any(tuple(t.shape) != (payload_bytes * _TP,) or t.dtype != torch.uint8
                   or t.device != comm.device or not t.is_contiguous() for t in (recv, packed))
            or recv.data_ptr() == packed.data_ptr()):
        raise ValueError("direct NCCL exchange requires distinct contiguous TP4 byte packets")
    # Construct views before opening the group. All ranks issue the same
    # peer order, including self, on the producer/consumer CUDA stream.
    sends = tuple(packed[r * payload_bytes:(r + 1) * payload_bytes] for r in range(_TP))
    recvs = tuple(recv[r * payload_bytes:(r + 1) * payload_bytes] for r in range(_TP))
    stream = torch.cuda.current_stream()
    comm.group_start()
    try:
        for peer in range(_TP):
            comm.send(sends[peer], peer, stream)
            comm.recv(recvs[peer], peer, stream)
    finally:
        comm.group_end()
    logger.info_once("[prefill-sp] direct PyNCCL packet exchange engaged")
