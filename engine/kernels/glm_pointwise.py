"""GLM pointwise lanes, preserving the composition's rounding boundaries.

The router retains torch.topk, including its tie order. Its decode projection
can consume the original BF16 operands with FP32 accumulation and output.
The clamped SwiGLU rounds once, after the FP32 product (unlike the drafter's
unclamped SwiGLU, which also rounds its sigmoid product to BF16).
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


def router_logits(x, gate):
    """BF16 checkpoint values, tensor-core accumulation, FP32 logits.

    Keep logits in FP32 through sigmoid, bias and selection. A BF16 output
    matmul would discard logit resolution before top-k. No weight repacking
    or TF32 conversion is needed: both original operands are already BF16.
    """
    if (x.ndim != 2 or gate.ndim != 2 or x.shape[1] != gate.shape[1]
            or not x.is_cuda or x.device != gate.device
            or x.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16):
        raise ValueError('tensor-core router requires matching CUDA BF16 operands')
    return torch.mm(x, gate.T, out_dtype=torch.float32)


@tr.jit
def _activation(G, U, O, SG: tl.constexpr, SU: tl.constexpr,
                D: tl.constexpr, LIMIT: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    g = tl.minimum(tl.load(G + r * SG + d, d < D, 0).to(tl.float32), LIMIT)
    u = tl.minimum(tl.maximum(tl.load(U + r * SU + d, d < D, 0).to(tl.float32), -LIMIT), LIMIT)
    sig = tl.div_rn(1., 1. + libdevice.exp(-g))
    tl.store(O + r * D + d, (g * sig) * u, d < D)


def swiglu_clamped(g, u, limit):
    if g.ndim != 2 or g.shape != u.shape or g.stride(-1) != 1 or u.stride(-1) != 1:
        raise ValueError("clamped SwiGLU needs matching rows with contiguous columns")
    out = torch.empty(g.shape, device=g.device, dtype=torch.bfloat16)
    if g.numel():
        _activation[(g.shape[0], tr.cdiv(g.shape[1], 512))](
            g, u, out, g.stride(0), u.stride(0), g.shape[1], float(limit), 512,
            enable_fp_fusion=False)
    return out


@tr.jit
def _scores(X, Bias, O, E: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    e = tl.arange(0, B)
    x = tl.load(X + r * E + e, e < E, 0)
    s = tl.div_rn(1., 1. + libdevice.exp(-x))
    bias = tl.load(Bias + e, e < E, 0)
    tl.store(O + r * E + e, s + bias, e < E)


@tr.jit
def _weights(X, Sel, Ids, W, E: tl.constexpr, K: tl.constexpr,
             SCALE: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, B)
    ids = tl.load(Sel + r * K + i, i < K, 0).to(tl.int32)
    x = tl.load(X + r * E + ids, i < K, 0)
    s = tl.where(i < K, tl.div_rn(1., 1. + libdevice.exp(-x)), 0.)
    w = tl.div_rn(s, tl.sum(s, 0)) * SCALE
    tl.store(Ids + r * K + i, ids, i < K)
    tl.store(W + r * K + i, w, i < K)


def route_weights(logits, bias, topk, scale):
    if logits.ndim != 2 or not logits.is_contiguous() or logits.dtype != torch.float32:
        raise ValueError("router requires the contiguous FP32 projection")
    rows, experts = logits.shape
    if bias.shape != (experts,) or not 0 < topk <= experts:
        raise ValueError("router bias and top-k must match the expert count")
    scores = torch.empty_like(logits)
    ids = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    weights = torch.empty((rows, topk), dtype=torch.float32, device=logits.device)
    if rows:
        _scores[(rows,)](logits, bias, scores, experts, tr.next_power_of_2(experts), enable_fp_fusion=False)
        selected = scores.topk(topk, dim=-1).indices
        _weights[(rows,)](logits, selected, ids, weights, experts, topk,
                          float(scale), tr.next_power_of_2(topk), enable_fp_fusion=False)
    return ids, weights


@tr.jit
def _layernorm(X, W, Bias, O, SX: tl.constexpr, D: tl.constexpr,
               EPS: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + r * SX + d, d < D, 0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    centered = tl.where(d < D, x - mean, 0.)
    var = tl.sum(centered * centered, 0) / D
    w = tl.load(W + d, d < D, 0).to(tl.float32)
    bias = tl.load(Bias + d, d < D, 0).to(tl.float32)
    tl.store(O + r * D + d, centered * tl.rsqrt(var + EPS) * w + bias, d < D)


def layernorm(x, w, bias, eps):
    if x.ndim != 2 or x.stride(-1) != 1 or w.shape != (x.shape[1],) or bias.shape != w.shape:
        raise ValueError("indexer LayerNorm requires rows and one weight/bias per column")
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel():
        _layernorm[(x.shape[0],)](x, w, bias, out, x.stride(0), x.shape[1], float(eps),
                                 tr.next_power_of_2(x.shape[1]), enable_fp_fusion=False)
    return out
