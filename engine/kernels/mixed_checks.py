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
        scale_offsets = tl.arange(0, 512)
        s0 = tl.load(S0 + scale_offsets, scale_offsets < 288, other=1)
        s1 = tl.load(S1 + scale_offsets, scale_offsets < 288, other=1)
        s2 = tl.load(S2 + scale_offsets, scale_offsets < 288, other=1)
        s3 = tl.load(S3 + scale_offsets, scale_offsets < 288, other=1)
        valid = ((s0 > 0) & (s0 < float('inf')) & (s1 > 0) & (s1 < float('inf'))
                 & (s2 > 0) & (s2 < float('inf')) & (s3 > 0) & (s3 < float('inf')))
        if tl.sum((~valid).to(tl.int32), 0) != 0:
            tl.atomic_or(Status, 2, sem='relaxed')


class PendingValueCheck:
    """Owned readback whose event can overlap independent CPU route planning."""
    def __init__(self, status):
        self.status = status
        self.host = torch.empty(1, dtype=torch.int32, pin_memory=True)
        self.host.copy_(status, non_blocking=True)
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(status.device))

    def wait(self):
        self.ready.synchronize()
        result = int(self.host.item())
        if result & 1:
            raise ValueError('mixed source values must be finite')
        if result & 2:
            raise ValueError('mixed experts require positive finite per-expert scales')


def begin_check_values(inputs, scales):
    """Caller checks metadata/eager/device before launch and waits before use."""
    decode, prefill, decode_routes, prefill_routes = inputs
    status = torch.zeros(1, dtype=torch.int32, device=decode.device)
    block = 16384
    _check[(triton.cdiv(max(decode.numel(), prefill.numel()), block),)](decode, prefill, decode_routes, prefill_routes,
        *scales, status, decode.numel(), prefill.numel(), block, num_warps=8)
    return PendingValueCheck(status)


def check_values(inputs, scales):
    """Synchronous entry for differential validation and standalone callers."""
    begin_check_values(inputs, scales).wait()
