"""GLM TP4 Q0/FP32 accumulation when a layer's scales cannot fit SF6.

Keep the original tile-major weights and raw E4M3 scale TMA descriptors.
Routing, input quantization and FP32 scatter are shared with the SF6 Q0
lane, so lossless scale compression is independent of accumulation dtype.
"""
import cutlass
import cutlass.cute as cute

from .moe_dynamic_gated_tiled import MoEGatedDynamicKernelTiled
from .moe_dynamic_gated_sf6_q0 import MoEGatedDynamicKernelSF6Q0, stock_contract_matches


class MoEGatedDynamicKernelRawQ0(MoEGatedDynamicKernelTiled):
    initialize_route_q0_and_publish = MoEGatedDynamicKernelSF6Q0.initialize_route_q0_and_publish
    scatter_sC_to_gmem = MoEGatedDynamicKernelSF6Q0.scatter_sC_to_gmem

    def _setup_attributes(self, hidden_size):
        if (hidden_size != 4096 or self.tile_shape_mnk != (128,128,128)
                or self.share_input_across_experts
                or (self.activation,self.swiglu_alpha,self.swiglu_beta,self.swiglu_limit)
                   != ("swigluoai_uninterleave",1.,0.,10.)
                or self.num_mma_warps != 8 or self.threads_per_cta != 288):
            raise ValueError("raw TP Q0 requires exact unshared M128 GLM geometry")
        if not stock_contract_matches():
            raise RuntimeError("raw TP Q0 inherited source changed")
        super()._setup_attributes(hidden_size)
        if (cute.size_in_bytes(self.a_dtype,self.a_smem_layout_staged) < 3*288*4
                or cute.size_in_bytes(cutlass.BFloat16,self.epi_smem_layout_staged) < 4*4096*2):
            raise ValueError("raw TP Q0 startup aliases exceed shared backing")
