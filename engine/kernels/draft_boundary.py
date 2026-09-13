"""Sparse proposal EOS masking: one tiny launch, no vocabulary-sized policy copy."""
import triton
import triton.language as tl


@triton.jit
def _mask(X, PACKET, START: tl.constexpr, V: tl.constexpr, STRIDE: tl.constexpr):
    step = tl.program_id(0)
    end = tl.load(PACKET + 3 + tl.arange(0, 32)) - START
    live = (step < tl.load(PACKET)) & (end >= 0) & (end < V)
    tl.store(X + step * STRIDE + end, -float('inf'), live)


def mask_ends(logits, start, packet):
    _mask[(logits.shape[0],)](logits, packet, start, logits.shape[1], logits.stride(0), num_warps=1)
