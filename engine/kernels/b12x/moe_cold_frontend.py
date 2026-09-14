"""Pack only cold routes into disjoint M128 expert tiles, once per invocation."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Uint8, Uint64
from flashinfer.cute_dsl.fp4_common import (
    fabs_f32, fmax_f32, quantize_block_fp4_fast, get_ptr_as_int64, st_global_u64,
)


class ColdRouteProducer:
    @cute.jit
    def __call__(self, prefill: cute.Tensor, route_weights: cute.Tensor,
                 sources: cute.Tensor, input_scale: cute.Tensor,
                 packed: cute.Tensor, scales: cute.Tensor,
                 token_map: cute.Tensor, token_weights: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(prefill, route_weights, sources, input_scale, packed, scales,
                    token_map, token_weights).launch(
                        grid=(sources.shape[0], 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, prefill: cute.Tensor, route_weights: cute.Tensor,
               sources: cute.Tensor, input_scale: cute.Tensor,
               packed: cute.Tensor, scales: cute.Tensor,
               token_map: cute.Tensor, token_weights: cute.Tensor):
        route, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        expert, row = sources[route, 0], sources[route, 1]
        token, slot = sources[route, 2], sources[route, 3]
        values = cute.make_rmem_tensor((16,), cutlass.Float32)
        maximum = cutlass.Float32(0.)
        for i in cutlass.range_constexpr(16):
            value = cutlass.Float32(prefill[token, tid * Int32(16) + Int32(i)])
            values[i] = value
            maximum = fmax_f32(maximum, fabs_f32(value))
        bits, scale = Uint64(0), Uint8(0)
        bits, scale = quantize_block_fp4_fast(values, maximum, input_scale[expert])
        st_global_u64(get_ptr_as_int64(packed, row * Int32(2048) + tid * Int32(8)), bits)
        offset = ((row // Int32(128)) * Int32(32768) + (tid // Int32(4)) * Int32(512)
                  + (row % Int32(32)) * Int32(16) + ((row // Int32(32)) % Int32(4)) * Int32(4)
                  + tid % Int32(4))
        scales[offset] = scale
        if tid == Int32(0):
            token_map[row] = token
            token_weights[row] = route_weights[token, slot]


def compile_cold_producer():
    from pathlib import Path
    from . import moe_dispatch as md
    def tensor(dtype, shape, align=16):
        return cute.runtime.make_fake_compact_tensor(dtype, shape,
            stride_order=tuple(reversed(range(len(shape)))), assumed_align=align)
    p, r, a, s, m = (cute.sym_int32() for _ in range(5))
    args = (tensor(cutlass.BFloat16, (p, 4096)), tensor(cutlass.Float32, (p, 8)),
            tensor(cutlass.Int32, (r, 4)), tensor(cutlass.Float32, (288,)),
            tensor(cutlass.Uint8, (a,)), tensor(cutlass.Uint8, (s,)),
            tensor(cutlass.Int32, (m,), 4), tensor(cutlass.Float32, (m,)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True))
    return md.build_and_load_cute_dsl_kernel(md._CUTE_DSL_MODULE, 'cold_route_producer_v1',
        lambda: cute.compile(ColdRouteProducer(), *args, options='--opt-level 2 --enable-tvm-ffi'),
        extra_key_files=(*md._kernel_source_files(), Path(__file__)))
