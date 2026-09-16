"""ST group-128 FP8 values in the cuBLASLt MX32 scale layout.

The four MX32 groups share the original scale. No FP8 value is requantized.
Full row tiles group four 32-row quarters so their scale words are contiguous.
Short decode batches keep adjacent rows and launch no empty row tiles. The
last producer owns scale padding without another launch or padded input rows.
"""
import torch
import triton
import triton.language as tl


def scale_bytes(rows, cols):
    if type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0 or cols % 128:
        raise ValueError('MX scales require positive rows and 128-aligned columns')
    return ((rows + 127) // 128) * (cols // 128) * 512


@triton.jit
def _word_offset(row, group, G: tl.constexpr):
    return (row // 128 * G + group) * 128 + row % 32 * 4 + row % 128 // 32


@triton.jit
def _scale_word(scale):
    return ((scale.to(tl.uint32, bitcast=True) >> 23) & 255) * 0x01010101


def row_programs(rows):
    return triton.cdiv(rows, 4)


@triton.jit
def _rows(M, TILED: tl.constexpr):
    pid = tl.program_id(0)
    if TILED:
        quarter = pid // 32 * 128 + pid % 32 + tl.arange(0, 4)*32
        # Keep only ceil(tail/4) producers for an incomplete row tile: M=129
        # launches 33, not 64 CTAs per K128 group. Full tiles stay coalesced.
        tail = M // 128 * 128 + pid % 32 * 4 + tl.arange(0, 4)
        return tl.where(pid < M // 128 * 32, quarter, tail)
    else:
        return pid*4 + tl.arange(0, 4)


@triton.jit
def _publish(S, scale, row, group, M, G: tl.constexpr):
    tl.store(S + _word_offset(row, group, G), _scale_word(scale), row < M)
    # The last four-row producer initializes all missing rows of the last
    # 128-row scale tile. Padding is metadata only: Q and D keep real M.
    if tl.program_id(0) == tl.num_programs(0) - 1:
        padding = (M // 128) * 128 + tl.arange(0, 128)
        tl.store(S + _word_offset(padding, group, G), 0x7F7F7F7F,
                 (padding >= M) & (padding < tl.cdiv(M, 128) * 128))


@triton.jit(do_not_specialize=['M'])
def _quantize(X, Q, S, M, K: tl.constexpr, G: tl.constexpr, TILED: tl.constexpr):
    row = _rows(M, TILED)
    group = tl.program_id(1)
    col = group * 128 + tl.arange(0, 128)
    x = tl.load(X + row[:, None] * K + col[None, :], row[:, None] < M,
                other=0.).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(amax / 448.)))
    tl.store(Q + row[:, None] * K + col[None, :],
             (x / scale[:, None]).to(tl.float8e4nv), row[:, None] < M)
    _publish(S, scale, row, group, M, G)


@triton.jit
def _pack_weights(S, Out):
    block = tl.program_id(0)
    scale = tl.load(S + block)
    tl.store(Out + block * 128 + tl.arange(0, 128), _scale_word(scale))



def buffers(rows, cols, device):
    return (torch.empty((rows, cols), device=device, dtype=torch.float8_e4m3fn),
            torch.empty(scale_bytes(rows, cols), device=device, dtype=torch.uint8))


def quantize(x, *, out=None):
    if (x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16
            or not x.is_contiguous() or x.shape[0] <= 0 or x.shape[1] <= 0 or x.shape[1] % 128):
        raise ValueError('MX quantization requires contiguous CUDA BF16 [M,K], K aligned to 128')
    m, k = x.shape
    q, scales = buffers(m, k, x.device) if out is None else out
    if (q.shape != x.shape or q.dtype != torch.float8_e4m3fn or not q.is_contiguous()
            or scales.dtype != torch.uint8 or scales.ndim != 1 or scales.numel() != scale_bytes(m, k)
            or not scales.is_contiguous() or q.device != x.device or scales.device != x.device):
        raise ValueError('MX quantization outputs do not match the input')
    if out is not None:
        from .fp8 import require_disjoint
        require_disjoint(x, q, scales)
    _quantize[(row_programs(m), k // 128)](x, q, scales.view(torch.int32), m, k, k // 128, m >= 128, num_warps=4)
    return q, scales


def pack_weight_scales(scales, rows, cols):
    if (rows % 128 or cols % 128 or tuple(scales.shape) != (rows // 128, cols // 128)
            or not scales.is_cuda or scales.dtype != torch.float32 or not scales.is_contiguous()):
        raise ValueError('MX weight packing requires contiguous FP32 block-128 CUDA scales')
    # Pack-time check, before capture. The mapping is exact only for the
    # positive normal powers of two produced by ST's FP8 packer.
    bits = scales.view(torch.int32)
    valid = ((bits & 0x7FFFFF) == 0) & (bits > 0) & (bits < 0x7F800000)
    if not bool(valid.all()):
        raise ValueError('MX weight scales must be positive normal powers of two')
    result = torch.empty(scale_bytes(rows, cols), device=scales.device, dtype=torch.uint8)
    _pack_weights[(scales.numel(),)](scales, result.view(torch.int32), num_warps=4)
    return result
