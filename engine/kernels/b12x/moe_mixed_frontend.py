"""Explicit two-source route producer for the private prepared V5 body.

Every descriptor names a source, original token and top-k slot. No concat,
atomic row allocation, or implicit token-count expansion. The quantizer and
SFA addressing are the ordinary static frontend's, including expert scales.
"""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Uint8, Uint64
from flashinfer.cute_dsl.fp4_common import (
    fabs_f32, fmax_f32, quantize_block_fp4_fast, get_ptr_as_int64, st_global_u64,
)


class MixedRouteProducer:
    @cute.jit
    def __call__(self, decode: cute.Tensor, prefill: cute.Tensor,
                 decode_weights: cute.Tensor, prefill_weights: cute.Tensor,
                 sources: cute.Tensor, experts: cute.Tensor, input_scale: cute.Tensor,
                 packed: cute.Tensor, scales: cute.Tensor,
                 token_map: cute.Tensor, token_weights: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(decode, prefill, decode_weights, prefill_weights, sources, experts,
                    input_scale, packed, scales, token_map, token_weights).launch(
                        grid=(sources.shape[0], 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, decode: cute.Tensor, prefill: cute.Tensor,
               decode_weights: cute.Tensor, prefill_weights: cute.Tensor,
               sources: cute.Tensor, experts: cute.Tensor, input_scale: cute.Tensor,
               packed: cute.Tensor, scales: cute.Tensor,
               token_map: cute.Tensor, token_weights: cute.Tensor):
        route, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        local, row = sources[route, 0], sources[route, 1]
        kind, token, slot = sources[route, 2], sources[route, 3], sources[route, 4]
        expert = experts[local]
        gs = input_scale[expert]
        values = cute.make_rmem_tensor((16,), cutlass.Float32)
        maximum = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(16):
            value = cutlass.Float32(0.0)
            if kind == Int32(0):
                value = cutlass.Float32(decode[token, tid * Int32(16) + Int32(i)])
            else:
                value = cutlass.Float32(prefill[token, tid * Int32(16) + Int32(i)])
            values[i] = value
            maximum = fmax_f32(maximum, fabs_f32(value))
        bits, scale = Uint64(0), Uint8(0)
        bits, scale = quantize_block_fp4_fast(values, maximum, gs)
        offset = (local * Int32(32) + row) * Int32(2048) + tid * Int32(8)
        st_global_u64(get_ptr_as_int64(packed, offset), bits)
        # 128 padded SFA rows per expert, 64 K tiles of 512 scale bytes.
        sf_offset = (local * Int32(32768) + (tid // Int32(4)) * Int32(512)
                     + (row % Int32(32)) * Int32(16) + (row // Int32(32)) * Int32(4)
                     + tid % Int32(4))
        scales[sf_offset] = scale
        if tid == Int32(0):
            token_map[local, row] = route
            if kind == Int32(0):
                token_weights[local, row] = decode_weights[token, slot]
            else:
                token_weights[local, row] = prefill_weights[token, slot]


def compile_producer():
    """Runtime row extents keep source/route counts out of the build key."""
    from . import moe_dispatch as md
    from pathlib import Path
    def tensor(dtype, shape, align=16):
        return cute.runtime.make_fake_compact_tensor(dtype, shape,
            stride_order=tuple(reversed(range(len(shape)))), assumed_align=align)
    d, p, r = cute.sym_int32(), cute.sym_int32(), cute.sym_int32()
    args = (tensor(cutlass.BFloat16, (d, 4096)), tensor(cutlass.BFloat16, (p, 4096)),
            tensor(cutlass.Float32, (d, 8)), tensor(cutlass.Float32, (p, 8)),
            tensor(cutlass.Int32, (r, 5), 4), tensor(cutlass.Int32, (288,), 4),
            tensor(cutlass.Float32, (288,)), tensor(cutlass.Uint8, (288*32*2048,)),
            tensor(cutlass.Uint8, (288*128*256,)), tensor(cutlass.Int32, (288, 32), 4),
            tensor(cutlass.Float32, (288, 32)), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True))
    return md.build_and_load_cute_dsl_kernel(md._CUTE_DSL_MODULE, 'mixed_route_producer_v1',
        lambda: cute.compile(MixedRouteProducer(), *args, options='--opt-level 2 --enable-tvm-ffi'),
        extra_key_files=(*md._kernel_source_files(), Path(__file__)))
