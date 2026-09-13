# SPDX-License-Identifier: Apache-2.0
"""Private M64 TP prefill candidate over the original packed SF6 weights.

The pinned M128 implementation stays intact. This subclass owns the smaller
M tile, its two-row Q0 staging contract and its 16-row-per-warp scatter strips.
FC1/Q1/FC2 arithmetic and the BF16-contribution/FP32-accumulation helper remain
inherited. Route allocation and atomic order can differ: GPU numerical and
consumer qualification are required. No serving selector enables this lane.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32
from flashinfer.cute_dsl.fp4_common import get_smem_ptr_as_int32, get_ptr_as_int64

from ._moe_dynamic.gated import MoEGatedDynamicKernel, load_shared_i32_f32_pair
from .moe_dynamic_ep_local import scatter_add_weighted_bf16x8_to_f32
from .moe_dynamic_gated_sf6_q0_words import MoEGatedDynamicKernelSF6Q0Words
from .moe_dynamic_gated_sf6_q0 import stock_contract_matches
from ._prefill_m64_bodies import M64Bodies


class MoEGatedDynamicKernelPrefillM64(M64Bodies, MoEGatedDynamicKernelSF6Q0Words):
    def __init__(self, *args, mma_tiler_mn=(64, 128), **kwargs):
        if args or mma_tiler_mn != (64, 128):
            raise ValueError('private TP prefill requires explicit M64/N128')
        # Initialize the inherited barriers and N/K contracts normally, then
        # replace only the geometry owned by this subclass before lowering.
        super().__init__(mma_tiler_mn=(128, 128), **kwargs)
        self.tile_shape_mnk = (64, 128, 128)
        self.fc1_tile_shape_mnk = (64, 64, 128)
        self.epi_tile = (64, 128)

    def _setup_attributes(self, hidden_size):
        if (hidden_size != 4096 or self.tile_shape_mnk != (64, 128, 128)
                or not self.reform_sf_pack or self.share_input_across_experts
                or (self.activation, self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit)
                   != ('swigluoai_uninterleave', 1., 0., 10.)
                or self.num_mma_warps != 8 or self.threads_per_cta != 288
                or not stock_contract_matches()):
            raise ValueError('private M64 TP prefill geometry/source contract differs')
        MoEGatedDynamicKernel._setup_attributes(self, hidden_size)
        # The inherited producer derives its batch from M*N/H: two BF16
        # input rows now fit in the startup-idle epilogue backing, not four.
        if (cute.size_in_bytes(self.a_dtype, self.a_smem_layout_staged) < 3 * 288 * 4
                or cute.size_in_bytes(cutlass.BFloat16, self.epi_smem_layout_staged) < 2 * 4096 * 2):
            raise ValueError('M64 Q0 startup aliases exceed shared backing')

    def _check_sf6_shapes(self, w13, down, packed1, packed2):
        if (self.tile_shape_mnk != (64, 128, 128)
                or len(w13.shape) != 3 or len(down.shape) != 3
                or w13.shape[0] != 1024 or cute.size(w13.shape[1]) != 4096
                or w13.shape[2] != 288 or down.shape[0] != 4096
                or cute.size(down.shape[1]) != 512 or down.shape[2] != 288):
            raise ValueError('M64 requires exact E288/H4096/I512 tiled weights')
        for tensor, shape in ((packed1, (288, 128, 1552)),
                              (packed2, (288, 64, 1552))):
            if tuple(tensor.shape) != shape or tensor.element_type != cutlass.Uint8:
                raise ValueError('M64 requires the original exact SF6 scale planes')

    @cute.jit
    def scatter_sC_to_gmem(self, tidx, output_tile_idx, valid_rows: Int32,
                          sC: cute.Tensor, tRS_sD: cute.Tensor,
                          scatter_output: cute.Tensor, scatter_tok_base_addr: Int32,
                          scatter_weight_base_addr: Int32, down_alpha_value):
        lane = Int32(tidx) & Int32(31)
        warp = Int32(tidx) >> Int32(5)
        row_base = (warp >> Int32(1)) * Int32(16)
        col_base = (warp & Int32(1)) * Int32(64)
        rows = valid_rows - row_base
        if rows > Int32(16):
            rows = Int32(16)
        if rows < Int32(0):
            rows = Int32(0)
        vector = lane
        while vector < rows * Int32(8):
            row = row_base + vector // Int32(8)
            col = col_base + (vector % Int32(8)) * Int32(8)
            token, weight = load_shared_i32_f32_pair(scatter_tok_base_addr + row * Int32(8))
            # N128 retains K_SW128. The raw address needs the same explicit
            # S<3,4,3> transform as the pinned M128 epilogue.
            offset = Int32(sC.layout((row, col, Int32(0))))
            offset = offset ^ ((offset & Int32(0x1C0)) >> Int32(3))
            scatter_add_weighted_bf16x8_to_f32(
                get_ptr_as_int64(scatter_output, token * Int32(scatter_output.shape[1])
                                 + output_tile_idx * Int32(128) + col),
                get_smem_ptr_as_int32(sC, offset), weight, down_alpha_value)
            vector += Int32(32)
