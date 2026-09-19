"""IEEE FP32 router projection, with resident FP32 weights at every row count."""
from pathlib import Path

import torch
import triton
import triton.language as tl

_EXT = None


def build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        from engine.kernels.common.native_cache import prepare_sources
        from engine.kernels.native_root import build_root
        src = Path(__file__).with_suffix('.cpp')
        flags = ['-O3']
        key, directory, sources = prepare_sources(
            build_root('router-fp32'), [src], (flags, torch.__version__, torch.version.cuda))
        _EXT = load(name='st_router_fp32_' + key, sources=list(sources),
                    extra_cflags=flags, build_directory=str(directory), verbose=False)
    return _EXT


def router_logits(x, weight):
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or not x.is_cuda or x.device != weight.device
            or x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != torch.float32):
        raise ValueError('FP32 router requires matching CUDA BF16/FP32 inputs and FP32 weights')
    if _EXT is None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('FP32 router must be built and warmed before graph capture')
    return build().run(x, weight)


# The router projection on the tensor cores (`router_logits_mma`): every product of a BF16 activation and a BF16 weight
# is exact in FP32 (eight bits of mantissa times eight), so a BF16 MMA that accumulates and returns FP32 is the IEEE
# FP32 projection of a router whose weights are BF16 values -- Qwen3.8's, the checkpoint's BF16 gates -- to the
# accumulation's rounding, which an FP32 GEMM has too. The MMA's own FP32 sums truncate, so a K tile's products are
# summed from zero by the MMA and the tiles added in IEEE round-to-nearest (an `add.rn.f32` the compiler cannot fold
# back into the MMA's accumulator): the running sum rounds as an FP32 GEMM's does. A weight with more than eight bits
# splits exactly into three BF16 terms (high, middle, low: 24 bits), three MMAs a K tile. K walks MMA_K at a time at
# every row count, so an element's sum is the same sequence of operations in a prefill chunk and a decode step: the
# two score alike to the bit.
MMA_K = 32
# rows from this take the wide tile, (BLOCK_M, BLOCK_N, warps, stages); fewer, a decode step's, the narrow one
MMA_TILES = ((64, (128, 128, 8, 3)), (1, (16, 32, 4, 3)))



@triton.jit
def _ieee_add(a, b, FP32_DOT: tl.constexpr):
    if FP32_DOT:                                                  # the interpreter has no PTX; its add is IEEE's
        return a + b
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)

@triton.jit
def _router_mma(X, W, OUT, M, N, K, sX, sW, sO, SPLIT: tl.constexpr, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, FP32_DOT: tl.constexpr):
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)            # a row block's column blocks side by side
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        km = ks < K
        x = tl.load(X + rows[:, None] * sX + ks[None, :], mask=(rows[:, None] < M) & km[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * sW + ks[None, :], mask=(cols[:, None] < N) & km[None, :], other=0.0)
        x = x.to(tl.bfloat16)
        high = w.to(tl.bfloat16)
        if FP32_DOT:                                              # the interpreter reads a BF16 dot's bits as ints
            part = tl.dot(x.to(tl.float32), tl.trans(high.to(tl.float32)))
        else:
            part = tl.dot(x, tl.trans(high))
        if SPLIT == 3:
            rest = w.to(tl.float32) - high.to(tl.float32)        # exact: two floats within one BF16 step
            middle = rest.to(tl.bfloat16)
            low = (rest - middle.to(tl.float32)).to(tl.bfloat16)
            if FP32_DOT:
                part = tl.dot(x.to(tl.float32), tl.trans(middle.to(tl.float32)), part)
                part = tl.dot(x.to(tl.float32), tl.trans(low.to(tl.float32)), part)
            else:
                part = tl.dot(x, tl.trans(middle), part)
                part = tl.dot(x, tl.trans(low), part)
        acc = _ieee_add(acc, part, FP32_DOT)
    tl.store(OUT + rows[:, None] * sO + cols[None, :], acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


def router_logits_mma(x, weight):
    """FP32 logits [N, E] of x [N, H] (BF16 values: BF16, or FP32 holding BF16) against the router [E, H] -- BF16, one
    MMA a K tile, or FP32, three (its high, middle and low BF16 terms) -- on the tensor cores, FP32 accumulated and
    returned (the module's note above). CUDA, or the Triton interpreter on the CPU."""
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1] or x.device != weight.device
            or x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype not in (torch.bfloat16, torch.float32)
            or x.stride(1) != 1 or weight.stride(1) != 1):
        raise ValueError('the MMA router takes x [N, H] and a router [E, H], BF16 or FP32, packed along H')
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty(m, n, device=x.device, dtype=torch.float32)
    if m:
        bm, bn, warps, stages = next(tile for least, tile in MMA_TILES if m >= least)
        _router_mma[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
            x, weight, out, m, n, k, x.stride(0), weight.stride(0), out.stride(0),
            SPLIT=1 if weight.dtype == torch.bfloat16 else 3, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=MMA_K,
            FP32_DOT=not x.is_cuda, num_warps=warps, num_stages=stages)
    return out
