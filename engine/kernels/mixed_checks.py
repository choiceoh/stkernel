"""One bounded reduction/readback for the fixed mixed FFN value contract."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['D', 'P'])
def _check(DX, PX, DR, PR, S0, S1, S2, S3, Status, D, P, B: tl.constexpr):
    block = tl.program_id(0)
    start = block * B
    offsets = start + tl.arange(0, B)
    x = tl.load(PX + offsets, offsets < P, other=0).to(tl.float32)
    bad = ~(tl.abs(x) < float('inf'))
    if start < D:
        x = tl.load(DX + offsets, offsets < D, other=0).to(tl.float32)
        bad |= ~(tl.abs(x) < float('inf'))
    if start < P // 512:
        x = tl.load(PR + offsets, offsets < P // 512, other=0)
        bad |= ~(tl.abs(x) < float('inf'))
    if start < D // 512:
        x = tl.load(DR + offsets, offsets < D // 512, other=0)
        bad |= ~(tl.abs(x) < float('inf'))
    if tl.sum(bad.to(tl.int32), 0) != 0:
        tl.atomic_or(Status, 1, sem='relaxed')
    if block == 0:
        s0 = tl.load(S0 + offsets, offsets < 288, other=1)
        s1 = tl.load(S1 + offsets, offsets < 288, other=1)
        s2 = tl.load(S2 + offsets, offsets < 288, other=1)
        s3 = tl.load(S3 + offsets, offsets < 288, other=1)
        valid = ((s0 > 0) & (s0 < float('inf')) & (s1 > 0) & (s1 < float('inf'))
                 & (s2 > 0) & (s2 < float('inf')) & (s3 > 0) & (s3 < float('inf')))
        if tl.sum((~valid).to(tl.int32), 0) != 0:
            tl.atomic_or(Status, 2, sem='relaxed')


def check_values(inputs, scales):
    """Metadata/GB10/eager checks belong to the caller, before any launch.

    Read all BF16 source values, FP32 route weights and positive FP32 scales.
    Keep dimensions runtime-valued so different arrivals share one kernel.
    """
    decode, prefill, decode_routes, prefill_routes = inputs
    status = torch.zeros(1, dtype=torch.int32, device=decode.device)
    block = 16384
    _check[(triton.cdiv(max(decode.numel(), prefill.numel()), block),)](decode, prefill, decode_routes, prefill_routes,
        *scales, status, decode.numel(), prefill.numel(), block, num_warps=8)
    result = int(status.item())
    if result & 1:
        raise ValueError('mixed source values must be finite')
    if result & 2:
        raise ValueError('mixed experts require positive finite per-expert scales')
