"""Explicit, currently unselected small-projection decode candidates.

KDA's two inputs stay separate; each BF16 dot accumulates in FP32 and rounds
once to BF16. The indexer joins weights once after input smoothing, never
inside a graph. Neither candidate changes recurrent state arithmetic.

The widths come from the bound kernel shape (engine/base/kernel_shape): the
paired KDA projection maps the low-rank `k_dim` input to `heads x k_dim`
outputs, the indexer pair maps `hidden` to the indexer's head width.
"""
import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl


@tr.jit
def _kda_pair(X0, X1, W0, W1, Y, M: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    which = tl.program_id(2)
    x = tl.where(which == 0, X0, X1)
    w = tl.where(which == 0, W0, W1)
    stride = tl.where(which == 0, XS0, XS1)
    row = tl.program_id(1) * BM + tl.arange(0, BM)
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, K)
    a = tl.load(x + row[:, None] * stride + k[None, :], row[:, None] < M, 0)
    b = tl.load(w + col[None, :] * K + k[:, None])
    out = tl.dot(a, b).to(tl.bfloat16)
    tl.store(Y + (which * M + row[:, None]) * N + col[None, :], out, row[:, None] < M)


def _weights(a, b, shape):
    if any(w.shape != shape or w.dtype != torch.bfloat16 or not w.is_cuda
           or not w.is_contiguous() for w in (a, b)) or a.device != b.device:
        raise ValueError(f'paired projection requires matching contiguous CUDA BF16 weights {shape}')


def _input(x, width, device):
    if (x.ndim != 2 or x.shape[0] not in (1, 6, 7, 14, 21, 28)
            or x.shape[1] != width or x.stride(-1) != 1
            or x.dtype != torch.bfloat16 or x.device != device):
        raise ValueError(f'paired projection requires declared BF16 decode rows of width {width}')


def _kda_widths():
    """(output width, input width) of the paired KDA low-rank projection: heads x k_dim <- k_dim."""
    from engine.base.kernel_shape import bound
    linear = bound().linear
    if linear is None:
        raise ValueError('the bound kernel shape declares no linear attention; the paired KDA projection does not apply')
    return linear.heads * linear.k_dim, linear.k_dim


def _indexer_widths():
    """(output width, input width) of the paired indexer projection: the index head <- hidden."""
    from engine.base.kernel_shape import bound
    shape = bound()
    if shape.indexer is None:
        raise ValueError('the bound kernel shape declares no sparse indexer; the paired indexer projection does not apply')
    return shape.indexer.head_dim, shape.hidden


class KdaPair:
    BM, BN = 16, 64

    def __init__(self, f_b, g_b):
        self.n, self.k = _kda_widths()
        if self.n % self.BN or self.k & (self.k - 1):
            raise ValueError(f'paired KDA projection needs an output width in {self.BN}-column tiles and a power-of-two input width')
        _weights(f_b, g_b, (self.n, self.k))
        self.f_b, self.g_b = f_b, g_b

    def __call__(self, f_a, g_a):
        _input(f_a, self.k, self.f_b.device)
        _input(g_a, self.k, self.g_b.device)
        if f_a.shape != g_a.shape:
            raise ValueError('paired KDA inputs must have equal rows')
        rows = f_a.shape[0]
        output = torch.empty((2, rows, self.n), device=f_a.device, dtype=f_a.dtype)
        _kda_pair[(self.n // self.BN, tr.cdiv(rows, self.BM), 2)](
            f_a, g_a, self.f_b, self.g_b, output, rows, f_a.stride(0), g_a.stride(0),
            self.BM, self.BN, self.k, self.n, num_warps=4)
        return output[0], output[1]


class IndexerPair:
    def __init__(self, wk, gate):
        self.head, self.hidden = _indexer_widths()
        _weights(wk, gate, (self.head, self.hidden))
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('prepare indexer weights after smoothing and before capture')
        self.weight = torch.cat((wk, gate), dim=0)

    def __call__(self, x):
        _input(x, self.hidden, self.weight.device)
        return F.linear(x, self.weight).chunk(2, dim=-1)
