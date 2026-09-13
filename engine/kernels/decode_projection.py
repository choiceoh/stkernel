"""Explicit GLM TP4 decode projections and indexer boundary fusion.

KDA's two inputs stay separate; each BF16 dot accumulates in FP32 and rounds
once to BF16. The indexer joins weights once after input smoothing, never
inside a graph. Neither candidate changes recurrent state arithmetic.
"""
import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl
from engine.kernels.kpool import _fwht_stage

DECODE_ROWS = (1, 6, 7, 14, 21, 28)


@tr.jit
def _kda_pair(X0, X1, W0, W1, Y, M: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    which = tl.program_id(2)
    x = tl.where(which == 0, X0, X1)
    w = tl.where(which == 0, W0, W1)
    stride = tl.where(which == 0, XS0, XS1)
    row = tl.program_id(1) * BM + tl.arange(0, BM)
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    a = tl.load(x + row[:, None] * stride + k[None, :], row[:, None] < M, 0)
    b = tl.load(w + col[None, :] * 128 + k[:, None])
    out = tl.dot(a, b).to(tl.bfloat16)
    tl.store(Y + (which * M + row[:, None]) * 2048 + col[None, :], out, row[:, None] < M)


def _weights(a, b, shape):
    if any(w.shape != shape or w.dtype != torch.bfloat16 or not w.is_cuda
           or not w.is_contiguous() for w in (a, b)) or a.device != b.device:
        raise ValueError(f'paired projection requires matching contiguous CUDA BF16 weights {shape}')


def _input(x, width, device):
    if (x.ndim != 2 or x.shape[0] not in DECODE_ROWS
            or x.shape[1] != width or x.stride(-1) != 1
            or x.dtype != torch.bfloat16 or x.device != device):
        raise ValueError(f'paired projection requires declared BF16 decode rows of width {width}')


class KdaPair:
    def __init__(self, f_b, g_b):
        _weights(f_b, g_b, (2048, 128))
        self.f_b, self.g_b = f_b, g_b

    def __call__(self, f_a, g_a):
        _input(f_a, 128, self.f_b.device)
        _input(g_a, 128, self.g_b.device)
        if f_a.shape != g_a.shape:
            raise ValueError('paired KDA inputs must have equal rows')
        rows = f_a.shape[0]
        output = torch.empty((2, rows, 2048), device=f_a.device, dtype=f_a.dtype)
        _kda_pair[(32, tr.cdiv(rows, 16), 2)](
            f_a, g_a, self.f_b, self.g_b, output, rows, f_a.stride(0), g_a.stride(0),
            16, 64, num_warps=4)
        return output[0], output[1]


class IndexerPair:
    def __init__(self, wk, gate, *, storage=None):
        _weights(wk, gate, (128, 4096))
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('prepare indexer weights after smoothing and before capture')
        if storage is None:  # standalone qualification; the serving owner supplies arena storage
            storage = torch.empty((256, 4096), dtype=wk.dtype, device=wk.device)
        if (storage.shape != (256, 4096) or storage.dtype != wk.dtype
                or storage.device != wk.device or not storage.is_contiguous()):
            raise ValueError('indexer pair storage must be contiguous CUDA BF16 [256,4096]')
        storage[:128].copy_(wk)
        storage[128:].copy_(gate)
        self.weight = storage

    def __call__(self, x):
        _input(x, 4096, self.weight.device)
        return F.linear(x, self.weight).chunk(2, dim=-1)


@tr.jit
def _indexer_boundary(Q, K, W, NW, NB, Q8, KO, WE,
                      NH: tl.constexpr, KS: tl.constexpr, SCALE: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 128)
    if head == NH:
        # Keep the original BF16 projection boundary and FP32 LayerNorm order.
        x = tl.load(K + row * KS + col).to(tl.float32)
        mean = tl.sum(x, 0) / 128
        centered = x - mean
        var = tl.sum(centered * centered, 0) / 128
        nw = tl.load(NW + col).to(tl.float32)
        nb = tl.load(NB + col).to(tl.float32)
        tl.store(KO + row * 128 + col, centered * tl.rsqrt(var + 1e-6) * nw + nb)
    else:
        index = row * NH + head
        x = tl.load(Q + index * 128 + col).to(tl.float32)
        x = _fwht_stage(x, 128, 64, 1)
        x = _fwht_stage(x, 128, 32, 2)
        x = _fwht_stage(x, 128, 16, 4)
        x = _fwht_stage(x, 128, 8, 8)
        x = _fwht_stage(x, 128, 4, 16)
        x = _fwht_stage(x, 128, 2, 32)
        x = _fwht_stage(x, 128, 1, 64)
        x = (x * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), 0), 1e-4)
        scale = tl.exp2(tl.ceil(tl.log2(amax * (1.0 / 448.0))))
        quant = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
        tl.store(Q8 + index * 128 + col, quant)
        w = tl.load(W + index)
        tl.store(WE + index, (w * scale) * SCALE)


def indexer_boundary(q, k, weights, norm_weight, norm_bias, scale):
    """Fold three post-projection launches and the temporary scale matrix.

    Head weights retain the original FP32 GEMM. K and the FWHT result retain
    their BF16 rounding points; this changes neither KDA state nor its cast.
    """
    if (q.ndim != 3 or q.shape[0] not in DECODE_ROWS or q.shape[2] != 128
            or q.shape[1] <= 0 or not q.is_contiguous() or q.dtype != torch.bfloat16 or not q.is_cuda):
        raise ValueError('indexer boundary needs declared contiguous BF16 decode queries [M,H,128]')
    rows, heads, _ = q.shape
    if (k.shape != (rows, 128) or k.dtype != q.dtype or k.stride(-1) != 1
            or weights.shape != (rows, heads) or weights.dtype != torch.float32 or not weights.is_contiguous()
            or norm_weight.shape != (128,) or norm_bias.shape != (128,)
            or not norm_weight.is_contiguous() or not norm_bias.is_contiguous()
            or norm_weight.dtype != torch.float32 or norm_bias.dtype != torch.float32
            or any(t.device != q.device for t in (k, weights, norm_weight, norm_bias))):
        raise ValueError('indexer boundary key/head/norm contract mismatch')
    q8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    key = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    effective = torch.empty_like(weights)
    _indexer_boundary[(rows, heads + 1)](q, k, weights, norm_weight, norm_bias, q8, key, effective,
                                        heads, k.stride(0), float(scale), num_warps=1, enable_fp_fusion=False)
    return q8, key, effective
