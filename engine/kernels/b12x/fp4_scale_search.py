"""Experimental NVFP4 scale selection, preserving FlashInfer's scale convention.

Static FC2 uses ss1/ss2; as1/as2 also cover FC1, dynamic prefill and dense MLPs.
Candidate zero is the original packed result, including exceptional-input behavior. Score
the actual packed E2M1 bytes, not an approximate quantizer or proxy threshold.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from flashinfer.cute_dsl.fp4_common import (
    fp8_e4m3_to_f32, fp8_e4m3_to_f32_and_rcp, rcp_approx_ftz,
    quantize_and_pack_16, quantize_and_pack_16_fast,
    quantize_block_fp4, quantize_block_fp4_fast,
)


@dsl_user_op
def _pair_error(bits: Uint32, x: Float32, y: Float32,
                scaled: Float32, norm: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [bits.ir_value(loc=loc, ip=ip)] +
        [v.ir_value(loc=loc, ip=ip) for v in (x, y, scaled, norm)],
        """{
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 h0, h1;
            .reg .b32 halves;
            .reg .f32 q0, q1, d0, d1, sq;
            mov.b32 {b0, b1, b2, b3}, $1;
            cvt.rn.f16x2.e2m1x2 halves, b0;
            mov.b32 {h0, h1}, halves;
            cvt.f32.f16 q0, h0;
            cvt.f32.f16 q1, h1;
            mul.rn.f32 q0, q0, $4;
            mul.rn.f32 q1, q1, $4;
            mul.rn.f32 d0, $2, $5;
            mul.rn.f32 d1, $3, $5;
            sub.rn.f32 d0, d0, q0;
            sub.rn.f32 d1, d1, q1;
            mul.rn.f32 sq, d0, d0;
            fma.rn.f32 $0, d1, d1, sq;
        }""", "=f,r,f,f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@cute.jit
def _error(values: cute.Tensor, packed: Uint64, scale: Float32, norm: Float32):
    error = Float32(0.0)
    for pair in cutlass.range_constexpr(8):
        bits = Uint32((packed >> Uint64(8 * pair)) & Uint64(255))
        error += _pair_error(bits, values[2 * pair], values[2 * pair + 1], scale, norm)
    return error


@cute.jit
def quantize_block_fp4_search(values: cute.Tensor, max_abs: Float32,
                               global_scale: Float32, radius: cutlass.Constexpr,
                               fast_math: cutlass.Constexpr = True):
    if cutlass.const_expr(fast_math):
        best_packed, best_scale = quantize_block_fp4_fast(values, max_abs, global_scale)
    else:
        best_packed, best_scale = quantize_block_fp4(values, max_abs, global_scale)
    if cutlass.const_expr(radius > 0):
        # Nonfinite/zero/negative scales and nonfinite maxima retain baseline.
        # A NaN in any value makes the SSE unordered and likewise keeps baseline.
        if (global_scale > Float32(0.0)) & (global_scale < Float32(float('inf'))) & \
                (max_abs > Float32(0.0)) & (max_abs < Float32(float('inf'))):
            norm = rcp_approx_ftz(max_abs)
            gs_recip = rcp_approx_ftz(global_scale)
            initial_code = Int32(best_scale)
            scale = fp8_e4m3_to_f32(Uint32(best_scale)) * global_scale * norm
            best_error = _error(values, best_packed, scale, norm)
            for i in cutlass.range_constexpr(2 * radius):
                # Baseline, -1, +1, -2, +2; strict comparison preserves ties.
                offset = (i // 2 + 1) * (-1 if i % 2 == 0 else 1)
                code = initial_code + Int32(offset)
                if (code > Int32(0)) & (code <= Int32(126)):
                    sf = fp8_e4m3_to_f32(Uint32(code))
                    if cutlass.const_expr(fast_math):
                        inv = fp8_e4m3_to_f32_and_rcp(Uint32(code)) * gs_recip
                        packed = quantize_and_pack_16_fast(values, inv)
                    else:
                        packed = quantize_and_pack_16(values, Float32(1.0) / (sf * global_scale))
                    error = _error(values, packed, sf * global_scale * norm, norm)
                    if error < best_error:
                        best_packed, best_scale, best_error = packed, Uint8(code), error
    return best_packed, best_scale
