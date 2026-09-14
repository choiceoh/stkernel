"""Captured K=7 MLA contractions with token-major outputs and original weights.

Reuse the existing BF16/FP32 absorb kernel with small-M tiles. Both consumers
can then use views, removing the two head-major to token-major copies per DSA
layer. The owner binds shape, weights and widths before graph capture.
"""
import torch

from engine.kernels.mla.prefill_absorb import _absorb

ROWS = (8, 16, 24, 32)


class DecodeAbsorb:
    def __init__(self, query_weight, output_weight, *, rows):
        from engine.base.kernel_shape import bound
        cell = bound().attention
        if (cell.kind, cell.heads, cell.head_dim) != ('mla', 16, 512):
            raise ValueError('decode absorb is compiled for the 16-head 512-latent MLA cell')
        rows = tuple(rows)
        if not rows or rows != ROWS[:len(rows)] or any(type(m) is not int for m in rows):
            raise ValueError('decode absorb needs a K=7 capture prefix')
        weights = (query_weight, output_weight)
        if (any(w.shape != (16, 256, 512) or w.dtype != torch.bfloat16 or not w.is_cuda
                or w.stride(2) != 1 or w.stride(1) != 512 or w.stride(0) < 256*512 for w in weights)
                or query_weight.device != output_weight.device):
            raise ValueError('decode absorb requires original CUDA BF16 kv_b slices [16,256,512]')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('prepare decode absorb weights before capture')
        self.weights, self.rows, self.executed = weights, rows, set()

    def __call__(self, x, *, transpose=False):
        if type(transpose) is not bool:
            raise ValueError('decode absorb transpose must be boolean')
        inner, outer = (512, 256) if transpose else (256, 512)
        w = self.weights[int(transpose)]
        if (x.ndim != 3 or x.shape[0] not in self.rows or x.shape[1:] != (16, inner)
                or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != w.device):
            raise ValueError('decode absorb requires declared contiguous BF16 MLA rows')
        m = x.shape[0]
        bm = 16 if m <= 16 else 32
        out = torch.empty((m, 16, outer), dtype=x.dtype, device=x.device)
        _absorb[((m+bm-1)//bm, outer//64, 16)](
            x, w, out, m, 16, inner, outer, w.stride(0), w.stride(1),
            transpose, bm, 64, 64, num_warps=4, num_stages=2)
        self.executed.add((m, 'output' if transpose else 'query'))
        return out
