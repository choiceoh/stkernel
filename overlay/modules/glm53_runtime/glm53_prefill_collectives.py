# SPDX-License-Identifier: Apache-2.0
"""Opt-in eager GLM TP4 prefill collectives for sharded residual/MHC work.

The model gathers normalized local rows before each unchanged TP attention or
MLP. A lexical scope defers its single terminal all-reduce; reduce-scatter
then sums the rank partials directly into the local residual shard. Decode
never enters this scope. Optional FP8 transport is a separate precision
experiment, not a lossless encoding or a serving default.
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
_FP8 = os.environ.get("VLLM_GLM53_PREFILL_SP_FP8") == "1"
if _ENABLED:
    # 39차: the boot-log anchor the bracket greps -- an armed knob is not
    # evidence of invocation, but a missing anchor IS evidence of no arming.
    logger.warning("[prefill-sp] sequence-parallel prefill armed (fp8 transport=%s)", _FP8)
_BLOCK = 2048
_TP = 4
_HIDDEN = 4096
_PARTIAL = ContextVar("glm53_prefill_partial", default=None)


@dataclass
class _PartialOutput:
    communicator: object
    num_tokens: int
    reductions: int = 0


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
    _, comm = _tp_comm()
    if (tensor.ndim != 2 or tensor.shape[0] < 32 or tensor.shape[1] != _HIDDEN
            or tensor.dtype != torch.bfloat16 or not tensor.is_cuda
            or tensor.device != comm.device or not tensor.is_contiguous()):
        raise ValueError("prefill collective requires contiguous CUDA BF16 [rows,4096]")
    return comm


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
    if not _FP8:
        comm.all_gather(out, tensor)
        return out if num_tokens is None else out[:num_tokens]
    # One payload contains FP8 bytes followed by FP32 scales. All-gather has
    # no arithmetic, so UINT8 transport is sufficient. This avoids a second
    # metadata collective and fuses the local absmax with quantization.
    n = tensor.numel()
    blocks = n // _BLOCK  # H=4096 guarantees complete, aligned blocks.
    payload_bytes = n + 4 * blocks
    payload = torch.empty(payload_bytes, device=tensor.device, dtype=torch.uint8)
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


def prefill_reduce_scatter(tensor):
    """Reduce full-token rank partials directly to the local MHC shard.

    FP8 mode first agrees on block maxima across ranks. Using unrelated local
    scales with NCCL SUM is incorrect. Common scale and headroom avoid FP8
    overflow; native FP8 reduction still changes rounding and needs quality
    validation. Metadata collectives and quantization costs count in timing.
    """
    comm = _check(tensor)
    rows = triton.cdiv(tensor.shape[0], _TP)
    padded_rows = rows * _TP
    out = torch.empty((rows, _HIDDEN),
                      device=tensor.device, dtype=tensor.dtype)
    if not _FP8:
        if padded_rows != tensor.shape[0]:
            padded = torch.empty((padded_rows, _HIDDEN),
                                 device=tensor.device, dtype=tensor.dtype)
            _copy_pad[(triton.cdiv(padded.numel(), _BLOCK),)](
                tensor, padded, N=tensor.numel(), OUT_N=padded.numel(), BLOCK=_BLOCK
            )
            tensor = padded
        comm.reduce_scatter(out, tensor)
        return out
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
