"""Drafter residual/RMS producer for the vocabulary head's MXFP8 input."""
import torch
import triton
import triton.language as tl
from . import mxfp8


@triton.jit
def _add_norm_mx(A, B, W, H, Q, S, M: tl.constexpr, T: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 4096)
    total = (tl.load(A + row*4096 + col).to(tl.float32)
             + tl.load(B + row*4096 + col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    scale = tl.rsqrt(tl.sum(total*total)/4096 + EPS)
    h = ((total*scale).to(tl.bfloat16)*tl.load(W + col)).to(tl.bfloat16)
    tl.store(H + row*4096 + col, h)
    # The anchor contributes to the drafter block but never to the vocabulary head.
    if row % T != 0:
        out_row = row//T*(T-1) + row%T-1
        blocks = tl.reshape(h.to(tl.float32), (32, 128))
        sf, inv = mxfp8._power2_scale(tl.maximum(tl.max(tl.abs(blocks), 1), 1e-4))
        quant = (blocks*inv[:, None]).to(tl.float8e4nv)
        tl.store(Q + out_row*4096 + col, tl.reshape(quant, (4096,)))
        group = tl.arange(0, 32)
        tl.store(S + mxfp8._word_offset(out_row, group, 32), mxfp8._scale_word(sf))
    ROWS: tl.constexpr = M//T*(T-1)
    if row == M-1 and ROWS % 128 != 0:
        pad = ROWS//128*128 + tl.arange(0, 128)
        group = tl.arange(0, 32)
        tl.store(S + mxfp8._word_offset(pad[:, None], group[None, :], 32),
                 0x7f7f7f7f, pad[:, None] >= ROWS)


def add_norm_head(a, b, weight, eps, block_rows):
    """Return full normalized BF16 rows and compact non-anchor FP8/MX rows.

    One block contains an anchor followed by its draft positions. All buffers
    belong to this invocation/captured graph; no cross-graph mutable cache.
    """
    if (a.ndim != 2 or a.shape[1] != 4096 or a.shape != b.shape
            or a.dtype != torch.bfloat16 or b.dtype != a.dtype or not a.is_cuda
            or b.device != a.device or not a.is_contiguous() or not b.is_contiguous()
            or weight.shape != (4096,) or weight.dtype != a.dtype or weight.device != a.device
            or not weight.is_contiguous()):
        raise ValueError('head producer requires contiguous CUDA BF16 residuals [M,4096] and weight [4096]')
    if (type(block_rows) is not int or block_rows < 2 or a.shape[0] == 0
            or a.shape[0] % block_rows or not 0 < eps < float('inf')):
        raise ValueError('head producer requires complete anchor/draft blocks and positive finite epsilon')
    rows = a.shape[0]//block_rows*(block_rows-1)
    h = torch.empty_like(a)
    q, s = mxfp8.buffers(rows, 4096, a.device)
    _add_norm_mx[(a.shape[0],)](a, b, weight, h, q, s.view(torch.int32),
                               a.shape[0], block_rows, eps, num_warps=8)
    return h, (q, s)
