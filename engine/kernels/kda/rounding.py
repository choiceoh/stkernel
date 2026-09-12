"""FP32 -> FP16 stochastic stores, with no mutable or sampler RNG state.

Philox10 is addressed by logical state element and absolute token position.
Never key it by a ring row, slot, pointer, launch count, or CTA layout: those
change on rollback, prefix restore and graph replay. The caller supplies a
layer/rank domain. Arithmetic continues from the unrounded FP32 register.
"""
import triton
import triton.language as tl


@triton.jit
def fp16_sr(value, position, element, WIDTH: tl.constexpr, SEED):
    # uint64 matters: GLM's 262144 cells wrap a uint32 counter at 16K tokens.
    counter = position.to(tl.uint64) * WIDTH + element.to(tl.uint64)
    bits = tl.randint(SEED, counter, n_rounds=10)
    return round_with_bits(value, bits)


@triton.jit
def round_with_bits(value, bits):
    # SM121 cannot assemble cvt.rs.f16x2.f32. Select adjacent values explicitly.
    # For finite in-range values, the gap is a power of two, so the probability
    # calculation is exact FP32. No FP64 arithmetic or extra kernel is needed.
    x = value.to(tl.float32)
    magnitude = tl.abs(x)
    lo = magnitude.to(tl.float16, fp_downcast_rounding="rtz")
    lo_bits = lo.to(tl.uint16, bitcast=True)
    hi = (lo_bits + 1).to(tl.uint16).to(tl.float16, bitcast=True)
    probability = (magnitude - lo.to(tl.float32)) / (hi.to(tl.float32) - lo.to(tl.float32))
    valid = (magnitude <= 65504.) & (magnitude != lo.to(tl.float32))
    threshold = (tl.where(valid, probability, 0.) * 4294967296.).to(tl.uint32)
    rounded = tl.where(bits.to(tl.uint32) < threshold, lo_bits + 1, lo_bits).to(tl.uint16)
    sign = ((x.to(tl.uint32, bitcast=True) >> 16) & 0x8000).to(tl.uint16)
    result = (rounded | sign).to(tl.float16, bitcast=True)
    # Preserve RTNE's infinities/NaNs/overflow behavior, and signed exact zeros.
    return tl.where(magnitude <= 65504., result, x.to(tl.float16))


@triton.jit(do_not_specialize=["POSITION", "SEED"])
def _copy(SRC, DST, POSITION, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
          SEED, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(SRC + i // (K * V) * S0 + (i // V % K) * S1 + i % V * S2,
                    i < H * K * V, other=0)
    rounded = fp16_sr(value, POSITION, i, H * K * V, SEED)
    tl.store(DST + i, rounded, i < H * K * V)


def store(dst, src, position, seed):
    """Store one [H,K,V] prefill/functional state; allow strided FP32 input."""
    if src.ndim != 3 or src.shape != dst.shape or not dst.is_contiguous():
        raise ValueError("KDA state store requires [H,K,V] and a dense destination")
    _copy[(triton.cdiv(src.numel(), 256),)](
        src, dst, position, *src.shape, *src.stride(), seed, 256)
