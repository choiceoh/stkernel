"""Private eight-row Q0 preparation; quantization and FP32 scatter stay intact.

Only the explicitly selected M128 prefill lanes use this producer. Source pins
protect the shared struct order sB -> sC -> sA. Its route metadata occupies
startup-idle sB, and its 64 KiB input span uses sC plus the first part of sA.
No extra shared storage is allocated. Compute starts after the unchanged
resident-grid publication barriers; the startup aliases are then dead.
"""
import cutlass
import cutlass.cute as cute

from ._prefill_q0_batch8 import Q0Batch8Body
from .moe_dynamic_gated_sf6_q0 import stock_contract_matches
from .moe_dynamic_gated_sf6_q0_words import MoEGatedDynamicKernelSF6Q0Words
from .moe_dynamic_gated_raw_q0 import MoEGatedDynamicKernelRawQ0
from .moe_dynamic_prefill_n128_tiled import MoEGatedPrefillN128TiledQ0


def check_layout(kernel, hidden_size):
    if (hidden_size != 4096 or kernel.tile_shape_mnk != (128,128,128)
            or kernel.num_mma_warps != 8 or kernel.threads_per_cta != 288
            or kernel.share_input_across_experts or not stock_contract_matches()):
        raise ValueError('eight-row Q0 requires the pinned unshared M128 GLM producer')
    b = cute.size_in_bytes(kernel.b_dtype,kernel.b_smem_layout_staged)
    c = cute.size_in_bytes(cutlass.BFloat16,kernel.epi_smem_layout_staged)
    a = cute.size_in_bytes(kernel.a_dtype,kernel.a_smem_layout_staged)
    # A rounded-up field boundary would invalidate subtraction from sC.
    if (b < 3*288*4 or b % kernel.buffer_align_bytes or c % kernel.buffer_align_bytes
            or c != 4*4096*2 or c+a < 8*4096*2):
        raise ValueError('eight-row Q0 startup storage does not fit its disjoint aliases')
    return dict(route_bytes=3*288*4,b_bytes=b,c_bytes=c,a_bytes=a,
                input_bytes=8*4096*2,producer_warps=8)


class _Batch8(Q0Batch8Body):
    def _setup_attributes(self, hidden_size):
        super()._setup_attributes(hidden_size)
        check_layout(self, hidden_size)


class PrefillQ0Batch8Packed(_Batch8, MoEGatedDynamicKernelSF6Q0Words):
    pass


class PrefillQ0Batch8Raw(_Batch8, MoEGatedDynamicKernelRawQ0):
    pass


class PrefillQ0Batch8N128(_Batch8, MoEGatedPrefillN128TiledQ0):
    pass
