"""K=7 FP32 head-gate partials, reduced inside the indexer boundary.

Read the existing [32,4096] FP32 weight, without a resident transpose or an
activation cast allocation. Each CTA reuses four heads over eight rows.
Products and both reduction levels stay FP32. The fixed tree is deterministic,
but differs from cuBLAS: this is a separately qualified numerical change.
"""
import torch
import triton as tr
import triton.language as tl

ROWS = (8, 16, 24, 32)
SPLITS = 16


@tr.jit
def _gate_partials(X, W, P, XS: tl.constexpr):
    split, heads, rows = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = heads * 4 + tl.arange(0, 4)
    k = split * 256 + tl.arange(0, 256)
    w = tl.load(W + h[:, None] * 4096 + k[None, :])
    for m in tl.static_range(8):
        row = rows * 8 + m
        x = tl.load(X + row * XS + k).to(tl.float32)
        partial = tl.sum(w * x[None, :], 1)
        tl.store(P + (row * 16 + split) * 32 + h, partial)


class IndexerHeadGate:
    """Own a fixed capture declaration and retain the original FP32 weight."""
    def __init__(self, weight, *, rows):
        from engine.base.kernel_shape import bound
        shape = bound()
        if (shape.hidden, shape.indexer.heads, shape.indexer.head_dim, shape.indexer.compress) != (4096, 32, 128, 'kpool'):
            raise ValueError('head gate is compiled for the 4096/32/128 kpool cell')
        rows = tuple(rows)
        if (not rows or rows != ROWS[:len(rows)] or any(type(m) is not int for m in rows)
                or weight.shape != (32, 4096) or weight.dtype != torch.float32
                or not weight.is_cuda or not weight.is_contiguous()):
            raise ValueError('head gate requires FP32 CUDA [32,4096] and a K=7 capture prefix')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('prepare head gates before graph capture')
        self.weight, self.rows, self.executed = weight, rows, set()

    def __call__(self, x):
        if (x.ndim != 2 or x.shape[0] not in self.rows or x.shape[1] != 4096
                or x.dtype != torch.bfloat16 or x.device != self.weight.device
                or x.stride(1) != 1 or x.stride(0) < 4096):
            raise ValueError('head gate input must match the bound BF16 decode rows, width and device')
        partials = torch.empty((x.shape[0], SPLITS, 32), dtype=torch.float32, device=x.device)
        _gate_partials[(SPLITS, 8, x.shape[0] // 8)](
            x, self.weight, partials, x.stride(0), num_warps=4, enable_fp_fusion=False)
        self.executed.add(x.shape[0])
        return partials
