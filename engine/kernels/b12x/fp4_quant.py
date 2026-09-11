"""SM12x instructions for the native NVFP4 MoE input packer.

Three-input absolute max shortens the scale reduction ahead of the
native E2M1 block-scaled MMA. Scale rounding and FP4 encoding stay in
the existing quantizer. Requires SM100+; this package targets SM12x.
"""
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def max_abs3(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
        "max.abs.f32 $0, $1, $2, $3;", "=f,f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@cute.jit
def max_abs_16(values: cute.Tensor) -> Float32:
    # 17 leaves (including +0) -> eight PTX max3 operations, depth 3.
    # GB10 lowers these to 16 FMNMX instructions, with at most six dependent
    # max instructions instead of sixteen. This exposes independent work;
    # it does not halve the SASS instruction count.
    # The final +0 preserves the old zero-initialized reduction for all-NaN
    # and signed-zero blocks. No .NaN or .ftz modifier is added.
    a = max_abs3(values[0], values[1], values[2])
    b = max_abs3(values[3], values[4], values[5])
    c = max_abs3(values[6], values[7], values[8])
    d = max_abs3(values[9], values[10], values[11])
    e = max_abs3(values[12], values[13], values[14])
    return max_abs3(max_abs3(a, b, c), max_abs3(d, e, values[15]), Float32(0.0))
