"""Opt-in E72 tile-major EP decode, with the TP v5 TMA/compute pipeline.

One immutable tiled FP4 weight allocation is shared with EP tiled prefill.
SF6 uses the inherited lossless scale restoration for both native geometries;
raw MMA scales remain a separate reference specialization. Inputs are
already mapped to local IDs: [0,72) routes execute; sentinel72 and every other
invalid ID are ignored before indexing any expert state, scales or weights.
Signed-zero route weights are skipped; NaNs on valid routes remain selected.

The kernel body is a bounded fork of v4: only route admission, output zero
width, and BF16-rounded contributions widened to FP32 RED differ. FC1/FC2,
activation, both BF16 rounding sites and every publication/pipeline barrier
remain in the same order. FP32 global RED follows its hardware FTZ semantics;
this is a new accumulation ABI and still requires the normal numerical gate.
SF6 native M1..8 shares the existing FC1 A/SFA ring between gate and up;
their independent weight/SF6 stages and the I128 rounding boundary stay intact.
"""

from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint64, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync

from flashinfer.cute_dsl.utils import (
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)
from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32,
    fabs_f32,
    fmax_f32,
    rcp_approx_ftz,
    quantize_block_fp4,
    quantize_block_fp4_fast,
    get_ptr_as_int64,
    ld_shared_i32_relaxed,
    st_global_f32,
    st_global_i32,
    shared_ptr_to_u32,
    st_shared_u8,
    st_global_u64,
    scatter_add_v4_bf16x2,
)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    Sm120B12xBlockScaledDenseGemmKernel as DenseGemmKernel,
)
from .moe_activation import gated_activation_f32, is_gated_activation
from .moe_static_common import (
    _bulk_g2s,
    STAMP_BARRIER1,
    STAMP_DMA_BASE,
    STAMP_ITEMS,
    STAMP_MMA_END,
    STAMP_SLOTS,
    _atomic_cas_global_i32,
    _compact_static_get_work_tile,
    _ld_global_acquire_i32,
    _ld_shared_f32,
    _ld_shared_i32,
    _ld_shared_i32_volatile,
    _spin_wait_global_eq_i32,
    _st_global_i64,
    _st_global_release_i32,
    _st_shared_f32,
    _st_shared_i32,
    _threadfence,
)


_SF_VEC_SIZE = 16
_COMPACT_STATIC_TILE_M = 128
_TILE_M = 32
_FC1_TILE_N = 64
# packed FC1 scales (cell q, moe_sf_pack): one 4096 B scale block becomes two
# byte-aligned planes (2048 + 1024) plus a 16 B tail holding the block's base,
# so a stage moves 3088 B and the expansion writes the 4096 B back in place.
_SF_BLOCK_BYTES = 4096
_SF_PLANE_A = 2048
_SF_PLANE_B = 1024
_SF_BASE_OFF = 3072
_SF_STAGE_BYTES = 3088
_FC1_TILE_K = 512
_FC2_TILE_N = 128
_FC2_TILE_K = 128



import hashlib
from dataclasses import dataclass
from pathlib import Path
from .moe_static_kernel_v5 import MoEStaticKernelV5
from .moe_reform_sf_pack import REFORM_SF_STAGE

STOCK_V4_SHA256 = "eeb31ed9e3c0c285ea4aac48e95a2a9652ae390ba12a0f9d37de5cff4c00e42c"
STOCK_V5_SHA256 = "4c3e8f66fb678d14fe2d97dd352c6e7ddb7b95b2b5ce710ebdb26398a232226e"
EP_TILED_CACHE_TAG = "glm53_ep_static_tiled_fp32_v1"
EP_TILED_A_RING_CACHE_TAG = "glm53_ep_static_sf6_a_ring_v1"


def ep_tiled_scale_mode(reform_sf_pack):
    if type(reform_sf_pack) is not bool:
        raise TypeError("EP tiled reform_sf_pack must be bool")
    return "sf6_v1" if reform_sf_pack else "raw_mma_scales"


def ep_tiled_geometry(num_tokens, max_rows, max_active_clusters):
    """Pure admission: bound every compact row and physical 128-row TMA tile."""
    for name, value in (("num_tokens", num_tokens), ("max_rows", max_rows),
                        ("max_active_clusters", max_active_clusters)):
        if type(value) is not int:
            raise TypeError(name + " must be an integer")
    if not 1 <= num_tokens <= 32:
        raise ValueError("EP tiled decode admits only native M1..32")
    if max_rows % 128 or not num_tokens * 8 <= max_rows <= 256:
        raise ValueError("EP tiled decode requires 128-aligned row capacity >= M*8")
    if not 1 <= max_active_clusters <= 48:
        raise ValueError("EP tiled decode requires at most 48 resident SM121 CTAs")
    reform = num_tokens <= 8
    return dict(m=num_tokens, max_rows=max_rows, mac=max_active_clusters,
                reform=reform, fc1=(16, 128, 256) if reform else (32, 64, 512),
                fc2=(16, 256, 128) if reform else (32, 128, 128))


def ep_tiled_source_contract():
    for name, expected in (("moe_static_kernel_v4.py", STOCK_V4_SHA256),
                           ("moe_static_kernel_v5.py", STOCK_V5_SHA256)):
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("EP tiled inherited source changed: " + name)


@dsl_user_op
def scatter_add_v4_bf16x2_to_f32(addr, v0, v1, v2, v3, v4, v5, v6, v7,
                                *, loc=None, ip=None):
    """Keep stock satfinite BF16 contributions; change only their sum storage."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)] + [v.ir_value(loc=loc, ip=ip)
            for v in (v0, v1, v2, v3, v4, v5, v6, v7)],
        "{ .reg .b32 p0,p1,p2,p3; .reg .b16 h0,h1,h2,h3,h4,h5,h6,h7;"
        " .reg .f32 f0,f1,f2,f3,f4,f5,f6,f7; .reg .b64 pnext;"
        " cvt.rn.satfinite.bf16x2.f32 p0, $2, $1;"
        " cvt.rn.satfinite.bf16x2.f32 p1, $4, $3;"
        " cvt.rn.satfinite.bf16x2.f32 p2, $6, $5;"
        " cvt.rn.satfinite.bf16x2.f32 p3, $8, $7;"
        " mov.b32 {h0,h1}, p0; mov.b32 {h2,h3}, p1;"
        " mov.b32 {h4,h5}, p2; mov.b32 {h6,h7}, p3;"
        " cvt.f32.bf16 f0,h0; cvt.f32.bf16 f1,h1;"
        " cvt.f32.bf16 f2,h2; cvt.f32.bf16 f3,h3;"
        " cvt.f32.bf16 f4,h4; cvt.f32.bf16 f5,h5;"
        " cvt.f32.bf16 f6,h6; cvt.f32.bf16 f7,h7;"
        " red.global.add.v4.f32 [$0], {f0,f1,f2,f3};"
        " add.u64 pnext,$0,16; red.global.add.v4.f32 [pnext], {f4,f5,f6,f7}; }",
        "l,f,f,f,f,f,f,f,f", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


class MoEStaticEPTiledKernel(MoEStaticKernelV5):
    """Same 160-thread TP producer/consumer geometry, EP-local routing only."""
    def __init__(self, *, num_tokens, max_rows, max_active_clusters,
                 input_scales_are_reciprocal=False, fast_math=True,
                 reform_sf_pack=False):
        ep_tiled_source_contract()
        ep_tiled_scale_mode(reform_sf_pack)
        geometry = ep_tiled_geometry(num_tokens, max_rows, max_active_clusters)
        self.ep_num_tokens = num_tokens
        self.ep_max_rows = max_rows
        super().__init__(sf_vec_size=16, output_tile_count_n=16,
            fc1_stages=2, fc2_stages=2, decode_reform=geometry["reform"],
            reform_sf_pack=reform_sf_pack,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math, activation="swigluoai_uninterleave",
            swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.0)
        # V4 intentionally rejects arbitrary SF6 + experimental combinations.
        # Its initialization above establishes every SF6 layout/expansion
        # attribute first. This EP-only M16/K256 geometry reuses the existing
        # independent A ring: one A/SFA transfer, then both gate/up consumers,
        # then release. B/SFB keep their original two stages and barriers.
        # No storage/layout is conditional on a_ring in the inherited init.
        self.a_ring = bool(reform_sf_pack and geometry["reform"])

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        b_w13: cute.Tensor,        # (N, K_in, K_tiles, E) fp4, tile-major
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,       # (H, K_in, K_tiles, E) fp4, tile-major
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        active_expert_count: cute.Tensor,
        weight_expert_ids: cute.Tensor,
        global_to_local_expert: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        stamps: cute.Tensor,
        next_item: cute.Tensor,
        sfb1_packed: cute.Tensor,   # packed FC1 scales; dummy off lane
        sfb2_packed: cute.Tensor,   # sf6 FC2 scales; dummy off lane
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._check_ep_call(a_input, topk_ids, topk_weights, b_w13, b_down,
                            row_counts, token_map, scatter_output)
        return MoEStaticKernelV5.__call__(self,
            a_input,
            topk_ids,
            topk_weights,
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            b_w13,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            row_counts,
            active_expert_count,
            weight_expert_ids,
            global_to_local_expert,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights,
            stamps,
            next_item,
            sfb1_packed,
            sfb2_packed,
            max_active_clusters,
            stream,
        )

    def _check_ep_call(self, a, ids, weights, w13, down, rows, token_map, output):
        if (tuple(a.shape) != (self.ep_num_tokens, 4096)
                or a.element_type != cutlass.BFloat16
                or tuple(ids.shape) != (self.ep_num_tokens * 8,)
                or ids.element_type not in (cutlass.Int32, cutlass.Int64)
                or tuple(weights.shape) != (self.ep_num_tokens * 8,)
                or weights.element_type != cutlass.Float32
                or tuple(w13.shape) != (4096, 512, 8, 72)
                or tuple(down.shape) != (4096, 128, 16, 72)
                or w13.element_type != cutlass.Float4E2M1FN
                or down.element_type != cutlass.Float4E2M1FN
                or tuple(rows.shape) != (72,)
                or tuple(token_map.shape) != (72, self.ep_max_rows)
                or tuple(output.shape) != (self.ep_num_tokens, 4096)
                or output.element_type != cutlass.Float32):
            raise ValueError("EP tiled static kernel ABI/geometry mismatch")

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_b_w13: cute.CopyAtom,
        mB_w13: cute.Tensor,
        tma_sfb_w13: cute.CopyAtom,
        mSFB_w13: cute.Tensor,
        tma_b_down: cute.CopyAtom,
        mB_down: cute.Tensor,
        tma_sfb_down: cute.CopyAtom,
        mSFB_down: cute.Tensor,
        tiled_mma1: cute.TiledMma,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a1_smem_staged: cute.ComposedLayout,
        b1_smem_staged: cute.ComposedLayout,
        sfa1_smem_staged: cute.Layout,
        sfb1_smem_staged: cute.Layout,
        epi1_smem_staged: cute.ComposedLayout,
        b2_smem_staged: cute.ComposedLayout,
        sfb2_smem_staged: cute.Layout,
        a2_smem_layout: cute.ComposedLayout,
        sfa2_smem_layout: cute.Layout,
        epi_smem_staged: cute.ComposedLayout,
        row_counts: cute.Tensor,
        active_expert_count: cute.Tensor,
        weight_expert_ids: cute.Tensor,
        global_to_local_expert: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        stamps: cute.Tensor,
        next_item: cute.Tensor,
        sfb1_packed: cute.Tensor,   # (E, blocks/expert, stage bytes) u8
        sfb2_packed: cute.Tensor,
    ):
        """Kernel entry point."""
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        _, _, gdim_z = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        is_cta_leader = Int32(Int32(tidx) == Int32(0))
        stamp_row = Int32(bidz) * Int32(STAMP_SLOTS)

        if cutlass.const_expr(self.stamps):
            if Int32(tidx) == Int32(0):
                _st_global_i64(
                    get_ptr_as_int64(stamps, stamp_row), cute.arch.globaltimer()
                )

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_b_w13)
            cpasync.prefetch_descriptor(tma_b_down)
            if cutlass.const_expr(not self.reform_sf_pack):
                cpasync.prefetch_descriptor(tma_sfb_w13)
                cpasync.prefetch_descriptor(tma_sfb_down)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        a1_smem_one = cute.slice_(a1_smem_staged, (None, None, 0))
        b1_smem_one = cute.slice_(b1_smem_staged, (None, None, 0))
        sfa1_smem_one = cute.slice_(sfa1_smem_staged, (None, None, 0))
        sfb1_smem_one = cute.slice_(sfb1_smem_staged, (None, None, 0))
        fc1_tma_bytes = cute.size_in_bytes(self.b_dtype, b1_smem_one)
        a_tma_bytes = cute.size_in_bytes(self.a_dtype, a1_smem_one) + cute.size_in_bytes(
            self.sf_dtype, sfa1_smem_one
        )
        if cutlass.const_expr(not self.skip_a and not self.a_ring):
            fc1_tma_bytes += a_tma_bytes
        if cutlass.const_expr(not self.skip_sf):
            if cutlass.const_expr(self.reform_sf_pack):
                fc1_tma_bytes += 1552 * self.sf1_packed_blocks
            elif cutlass.const_expr(self.sf_pack):
                fc1_tma_bytes += _SF_STAGE_BYTES
            else:
                fc1_tma_bytes += cute.size_in_bytes(self.sf_dtype, sfb1_smem_one)
        b2_smem_one = cute.slice_(b2_smem_staged, (None, None, 0))
        sfb2_smem_one = cute.slice_(sfb2_smem_staged, (None, None, 0))
        fc2_tma_bytes = cute.size_in_bytes(self.b_dtype, b2_smem_one)
        if cutlass.const_expr(self.reform_sf_pack):
            fc2_tma_bytes += self.sf2_stage_bytes
        else:
            fc2_tma_bytes += cute.size_in_bytes(self.sf_dtype, sfb2_smem_one)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            ctrl: cute.struct.MemRange[cutlass.Int32, 2]
            fc1_bars: cute.struct.MemRange[cutlass.Int64, self.fc1_stages * 2]
            fc2_bars: cute.struct.MemRange[cutlass.Int64, self.fc2_stages * 2]
            a_bars: cute.struct.MemRange[cutlass.Int64, self.fc1_stages * 2]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, _COMPACT_STATIC_TILE_M
            ]
            scatter_weight_cache: cute.struct.MemRange[
                cutlass.Float32, _COMPACT_STATIC_TILE_M
            ]
            sA1: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a1_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB1: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b1_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFA1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa1_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb1_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b2_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb2_smem_staged)],
                self.buffer_align_bytes,
            ]
            sA2: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a2_smem_layout)],
                self.buffer_align_bytes,
            ]
            sSFA2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa2_smem_layout)],
                self.buffer_align_bytes,
            ]
            sC1: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi1_smem_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]

        storage = smem.allocate(Storage)

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        fc1_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.fc1_stages,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=fc1_tma_bytes,
            barrier_storage=storage.fc1_bars.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        fc2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.fc2_stages,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=fc2_tma_bytes,
            barrier_storage=storage.fc2_bars.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        # A ring (a_ring only; the init is harmless otherwise)
        a_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.fc1_stages,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=a_tma_bytes,
            barrier_storage=storage.a_bars.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.sync_threads()

        sA1 = storage.sA1.get_tensor(a1_smem_staged.outer, swizzle=a1_smem_staged.inner)
        sB1 = storage.sB1.get_tensor(b1_smem_staged.outer, swizzle=b1_smem_staged.inner)
        sB2 = storage.sB2.get_tensor(b2_smem_staged.outer, swizzle=b2_smem_staged.inner)
        sA2 = storage.sA2.get_tensor(a2_smem_layout.outer, swizzle=a2_smem_layout.inner)
        cute.recast_tensor(sA1, cutlass.Uint8)
        cute.recast_tensor(sB1, cutlass.Uint8)
        cute.recast_tensor(sB2, cutlass.Uint8)
        cute.recast_tensor(sA2, cutlass.Uint8)
        sSFA1 = storage.sSFA1.get_tensor(sfa1_smem_staged)
        sSFB1 = storage.sSFB1.get_tensor(sfb1_smem_staged)
        sSFB2 = storage.sSFB2.get_tensor(sfb2_smem_staged)
        sSFA2 = storage.sSFA2.get_tensor(sfa2_smem_layout)
        cute.recast_tensor(sSFA1, cutlass.Uint8)
        cute.recast_tensor(sSFB1, cutlass.Uint8)
        cute.recast_tensor(sSFB2, cutlass.Uint8)
        cute.recast_tensor(sSFA2, cutlass.Uint8)
        sC1 = storage.sC1.get_tensor(
            epi1_smem_staged.outer, swizzle=epi1_smem_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_staged.outer, swizzle=epi_smem_staged.inner
        )
        sfa2_base_addr = shared_ptr_to_u32(storage.sSFA2.data_ptr())
        a2_base_addr = shared_ptr_to_u32(storage.sA2.data_ptr())
        sfb1_base_addr = shared_ptr_to_u32(storage.sSFB1.data_ptr())
        sfb2_base_addr = shared_ptr_to_u32(storage.sSFB2.data_ptr())
        ctrl_base_addr = shared_ptr_to_u32(storage.ctrl.data_ptr())
        scatter_tok_base_addr = shared_ptr_to_u32(storage.scatter_tok_cache.data_ptr())
        scatter_weight_base_addr = shared_ptr_to_u32(
            storage.scatter_weight_cache.data_ptr()
        )

        num_tokens = Int32(a_input.shape[0])
        cols = Int32(a_input.shape[1])
        num_experts = Int32(row_counts.shape[0])
        sf_blocks_per_row = cols // Int32(self.sf_vec_size)
        output_bytes_per_row = cols // Int32(2)
        max_rows = Int32(token_map.shape[1])
        total_pairs = Int32(topk_ids.shape[0])
        num_topk = total_pairs // num_tokens
        expert_scale_stride = Int32(scale_storage.shape[0]) // num_experts
        num_global_experts = Int32(global_to_local_expert.shape[0])
        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)
        sf_k_tile = Int32(self.sf_vec_size * 4)
        num_k_tiles = (cols + sf_k_tile - Int32(1)) // sf_k_tile

        # ------------------------------------------------------------------
        # Phase 0 / Phase 1 (stock frontend)
        # ------------------------------------------------------------------
        i = flat_tid
        while i < num_experts:
            row_counts[i] = Int32(0)
            i += flat_stride
        i = flat_tid
        while i < num_global_experts:
            global_to_local_expert[i] = Int32(-1)
            i += flat_stride
        if flat_tid == Int32(0):
            active_expert_count[Int32(0)] = Int32(0)
            if cutlass.const_expr(self.even or self.split):
                next_item[Int32(0)] = Int32(0)
        scatter_total = num_tokens * cols
        j = flat_tid
        while j < scatter_total:
            scatter_output[j // cols, j % cols] = cutlass.Float32(0.0)
            j += flat_stride
        cute.arch.sync_threads()
        self._resident_grid_barrier(
            barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader
        )
        if cutlass.const_expr(self.stamps):
            if Int32(tidx) == Int32(0):
                _st_global_i64(
                    get_ptr_as_int64(stamps, stamp_row + Int32(STAMP_BARRIER1)),
                    cute.arch.globaltimer(),
                )

        pair_idx = Int32(bidz)
        while pair_idx < total_pairs:
            expert_id = topk_ids[pair_idx].to(Int32)
            # Never derive map/scale/weight addresses for a remote sentinel.
            if expert_id >= Int32(0) and expert_id < num_experts:
                weight = topk_weights[pair_idx].to(cutlass.Float32)
                if weight != cutlass.Float32(0.0):
                    token_idx = pair_idx // num_topk
                    local_expert_id = Int32(0)
                    row = Int32(0)
                    if is_cta_leader > Int32(0):
                        prior_local_expert_id = _atomic_cas_global_i32(
                            get_ptr_as_int64(global_to_local_expert, expert_id),
                            Int32(-1),
                            Int32(-2),
                        )
                        if prior_local_expert_id == Int32(-1):
                            local_expert_id = atomic_add_global_i32(
                                get_ptr_as_int64(active_expert_count, Int32(0)),
                                Int32(1),
                            )
                            weight_expert_ids[local_expert_id] = expert_id
                            _st_global_release_i32(
                                get_ptr_as_int64(global_to_local_expert, expert_id),
                                local_expert_id,
                            )
                        else:
                            if prior_local_expert_id == Int32(-2):
                                _spin_wait_global_eq_i32(
                                    get_ptr_as_int64(global_to_local_expert, expert_id),
                                    Int32(-2),
                                )
                                prior_local_expert_id = _ld_global_acquire_i32(
                                    get_ptr_as_int64(global_to_local_expert, expert_id),
                                )
                            local_expert_id = prior_local_expert_id
                        row = atomic_add_global_i32(
                            get_ptr_as_int64(row_counts, local_expert_id),
                            Int32(1),
                        )
                        if cutlass.const_expr(self.even or self.split):
                            if row % Int32(self.tile_m) == Int32(0):
                                atomic_add_global_i32(
                                    get_ptr_as_int64(next_item, Int32(0)),
                                    Int32(self.output_tile_count_n),
                                )
                        map_idx = local_expert_id * max_rows + row
                        st_global_i32(get_ptr_as_int64(token_map, map_idx), token_idx)
                        st_global_f32(get_ptr_as_int64(token_weights, map_idx), weight)
                        _st_shared_i32(ctrl_base_addr + Int32(0), local_expert_id)
                        _st_shared_i32(ctrl_base_addr + Int32(4), row)
                    cute.arch.sync_threads()
                    local_expert_id = _ld_shared_i32(ctrl_base_addr + Int32(0))
                    row = _ld_shared_i32(ctrl_base_addr + Int32(4))

                    gs_value = input_global_scale[expert_id].to(cutlass.Float32)
                    if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(0.0):
                        if self.fast_math:
                            gs_value = rcp_approx_ftz(gs_value)
                        else:
                            gs_value = cutlass.Float32(1.0) / gs_value
                    sf_idx = Int32(tidx)
                    while sf_idx < sf_blocks_per_row:
                        block_start = sf_idx * Int32(self.sf_vec_size)
                        values = cute.make_rmem_tensor((self.sf_vec_size,), cutlass.Float32)
                        block_max = cutlass.Float32(0.0)
                        for elem_idx in cutlass.range_constexpr(self.sf_vec_size):
                            value = cutlass.Float32(
                                a_input[token_idx, block_start + Int32(elem_idx)]
                            )
                            values[elem_idx] = value
                            block_max = fmax_f32(block_max, fabs_f32(value))
                        scale_byte = Uint8(0)
                        packed_lo = Uint64(0)
                        if self.fast_math:
                            packed_lo, scale_byte = quantize_block_fp4_fast(
                                values, block_max, gs_value
                            )
                        else:
                            packed_lo, scale_byte = quantize_block_fp4(
                                values, block_max, gs_value
                            )
                        output_offset = (
                            local_expert_id * max_rows * output_bytes_per_row
                            + row * output_bytes_per_row
                            + sf_idx * Int32(self.sf_vec_size // 2)
                        )
                        st_global_u64(
                            get_ptr_as_int64(packed_a_storage, output_offset), packed_lo
                        )
                        m_tile_idx = row // Int32(32 * 4)
                        k_tile_idx = sf_idx // Int32(4)
                        outer_m_idx = row % Int32(32)
                        inner_m_idx = (row % Int32(32 * 4)) // Int32(32)
                        inner_k_idx = sf_idx % Int32(4)
                        scale_offset = (
                            local_expert_id * expert_scale_stride
                            + m_tile_idx * num_k_tiles * Int32(32 * 4 * 4)
                            + k_tile_idx * Int32(32 * 4 * 4)
                            + outer_m_idx * Int32(4 * 4)
                            + inner_m_idx * Int32(4)
                            + inner_k_idx
                        )
                        scale_storage[scale_offset] = scale_byte
                        sf_idx += Int32(self.threads_per_cta)

            cute.arch.sync_threads()
            pair_idx += Int32(gdim_z)

        self._resident_grid_barrier(
            barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader
        )
        if cutlass.const_expr(self.stamps):
            if Int32(tidx) == Int32(0):
                _st_global_i64(
                    get_ptr_as_int64(stamps, stamp_row + Int32(1)),
                    cute.arch.globaltimer(),
                )
        # Item striding: n_active CTAs, the rest exit after the frontend. With
        # `even`, the candidate leaving the fewest empty slots in its last
        # wave wins (ties: the largest); every candidate still saturates DRAM
        # (32 CTAs need 7.5 GB/s each; a lone CTA streams ~10).
        n_active = Int32(gdim_z)
        start_work_idx = Int32(bidz)
        total_items = Int32(0x3FFFFFFF)
        if cutlass.const_expr(self.even or self.split):
            total_items = next_item[Int32(0)]
        if cutlass.const_expr(self.even):
            best_waste = Int32(0x7FFFFFFF)
            for cand in (48, 44, 40, 36, 32):
                n_c = Int32(cand)
                if n_c <= Int32(gdim_z):
                    waves = (total_items + n_c - Int32(1)) // n_c
                    waste = waves * n_c - total_items
                    if waste < best_waste:
                        best_waste = waste
                        n_active = n_c
            if Int32(bidz) >= n_active:
                start_work_idx = Int32(0x3FFFFFFF)   # decodes as no work
        # split plan: items >= split_base are last-wave items (role 0 for
        # their striding owner); helper_idx is this CTA's role-1 item or -1
        split_base = Int32(0x3FFFFFFF)
        helper_idx = Int32(-1)
        if cutlass.const_expr(self.split):
            full_waves = total_items // Int32(gdim_z)
            p_last = total_items - full_waves * Int32(gdim_z)
            if p_last > Int32(0):
                if p_last * Int32(2) <= Int32(gdim_z):
                    split_base = full_waves * Int32(gdim_z)
                    if Int32(bidz) >= p_last:
                        if Int32(bidz) < p_last * Int32(2):
                            helper_idx = split_base + Int32(bidz) - p_last

        # ------------------------------------------------------------------
        # Tiled views and TMA partitions
        # ------------------------------------------------------------------
        gA = cute.local_tile(mA, self.sa1_tile_shape_mk, (None, None, None))
        gB_w13_tiled = cute.local_tile(
            mB_w13,
            cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA = cute.local_tile(mSFA, self.sfa1_tile_shape_mk, (None, None, None))
        if cutlass.const_expr(not self.reform_sf_pack):
            gSFB_w13_tiled = cute.local_tile(
                mSFB_w13, self.sfb1_tile_shape_nk, (None, None, None)
            )
        gB_down = cute.local_tile(
            mB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        if cutlass.const_expr(not self.reform_sf_pack):
            gSFB_down = cute.local_tile(
                mSFB_down, self.sfb_tile_shape_nk, (None, None, None)
            )
        thr_mma1 = tiled_mma1.get_slice(tidx)
        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord[1]
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord[0]

        tAsA, tAgA = cpasync.tma_partition(
            tma_a, a_cta_crd, a_cta_layout,
            cute.group_modes(sA1, 0, 2), cute.group_modes(gA, 0, 2),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_sfa, a_cta_crd, a_cta_layout,
            cute.group_modes(sSFA1, 0, 2), cute.group_modes(gSFA, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsB1, tBgB_w13 = cpasync.tma_partition(
            tma_b_w13, b_cta_crd, b_cta_layout,
            cute.group_modes(sB1, 0, 2), cute.group_modes(gB_w13_tiled, 0, 2),
        )
        # the FC1 SFB smem block's two 64-row halves (the MMA side reads one
        # per half; the DMA side always lands the whole 128-row block)
        sfb1_tile = cute.slice_(self.fc1_tile_shape_mnk, (0, None, None))
        sSFB1_0 = cute.local_tile(sSFB1, sfb1_tile, (0, 0, None))
        if cutlass.const_expr(self.decode_reform):
            sSFB1_1 = sSFB1_0
        else:
            sSFB1_1 = cute.local_tile(sSFB1, sfb1_tile, (1, 0, None))
        if cutlass.const_expr(not self.reform_sf_pack):
            tBsSFB1, tBgSFB_w13 = cpasync.tma_partition(
                tma_sfb_w13, b_cta_crd, b_cta_layout,
                cute.group_modes(sSFB1, 0, 2), cute.group_modes(gSFB_w13_tiled, 0, 2),
            )
            tBsSFB1 = cute.filter_zeros(tBsSFB1)
            tBgSFB_w13 = cute.filter_zeros(tBgSFB_w13)
        tBsB2, tBgB_down = cpasync.tma_partition(
            tma_b_down, b_cta_crd, b_cta_layout,
            cute.group_modes(sB2, 0, 2), cute.group_modes(gB_down, 0, 2),
        )
        if cutlass.const_expr(not self.reform_sf_pack):
            tBsSFB2, tBgSFB_down = cpasync.tma_partition(
                tma_sfb_down, b_cta_crd, b_cta_layout,
                cute.group_modes(sSFB2, 0, 2), cute.group_modes(gSFB_down, 0, 2),
            )
            tBsSFB2 = cute.filter_zeros(tBsSFB2)
            tBgSFB_down = cute.filter_zeros(tBgSFB_down)

        # FC1 MMA fragments (tiled_mma1). The B scale block holds both 64-row
        # halves: sub-tile per half (static), as the stock kernel does through
        # sfb_tile_offset.
        tCsA1 = thr_mma1.partition_A(sA1)
        tCrA1 = tiled_mma1.make_fragment_A(tCsA1[None, None, None, 0])
        tCsB1 = thr_mma1.partition_B(sB1)
        tCrB1 = tiled_mma1.make_fragment_B(tCsB1[None, None, None, 0])
        tCrSFB1_0 = self._partition_fragment_SFB(
            sSFB1_0[None, None, 0], thr_mma1, tidx)  # type: ignore[arg-type]
        tCrSFB1_1 = self._partition_fragment_SFB(
            sSFB1_1[None, None, 0], thr_mma1, tidx)  # type: ignore[arg-type]

        # FC2 fragments (tiled_mma), A from the quantized intermediate
        tCsA2 = thr_mma.partition_A(sA2)
        tCrA2 = tiled_mma.make_fragment_A(tCsA2[None, None, None, 0])
        sSFA2_tile = cute.local_tile(
            sSFA2,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (0, 0, None),
        )
        tCrSFA2 = self._dense_cls._partition_fragment_SFA(
            self, sSFA2_tile[None, None, 0], thr_mma, tidx  # type: ignore[arg-type]
        )
        tCsB2 = thr_mma.partition_B(sB2)
        tCrB2 = tiled_mma.make_fragment_B(tCsB2[None, None, None, 0])
        tCrSFB2 = self._partition_fragment_SFB(
            sSFB2[None, None, 0], thr_mma, tidx  # type: ignore[arg-type]
        )

        tCsC1_for_shape = thr_mma1.partition_C(sC1[None, None, 0])
        acc1_shape = tCsC1_for_shape.shape[:3]
        gate_acc = cute.make_rmem_tensor(acc1_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(acc1_shape, self.acc_dtype)
        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        acc_shape = tCsC_for_shape.shape[:3]

        k_tile_cnt1 = cute.size(gA, mode=[3])           # K / 512 = 8
        intermediate_tile_cnt = cute.size(gB_w13_tiled, mode=[2])   # 2*I_tp / 64
        gate_tile_cnt = intermediate_tile_cnt // Int32(2)
        output_tile_cnt = cute.size(gB_down, mode=[2])   # K / 128 = 32

        fc1_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.fc1_stages
        )
        fc1_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.fc1_stages
        )
        a_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.fc1_stages
        )
        a_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.fc1_stages
        )
        fc2_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.fc2_stages
        )
        fc2_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.fc2_stages
        )

        # ===================================================================
        # MMA WARP GROUP (warps 0-3)
        # ===================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            num_k_blocks1 = cute.size(tCrA1, mode=[2])   # 8
            num_k_blocks = cute.size(tCrA2, mode=[2])    # 2

            atom_ld_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_ld_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            atom_ld_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.sf_dtype
            )
            # FC1 (tiled_mma1) copies
            smem_copy_A1 = cute.make_tiled_copy_A(atom_ld_A, tiled_mma1)
            smem_copy_B1 = cute.make_tiled_copy_B(atom_ld_B, tiled_mma1)
            smem_copy_SFA1 = cute.make_tiled_copy(
                atom_ld_SF,
                self._dense_cls._get_layoutSFA_TV(self, tiled_mma1),  # type: ignore[arg-type]
                (
                    cute.size(tiled_mma1.permutation_mnk[0]),
                    cute.size(tiled_mma1.permutation_mnk[2]),
                ),
            )
            smem_copy_SFB1 = cute.make_tiled_copy(
                atom_ld_SF,
                self._dense_cls._get_layoutSFB_TV(self, tiled_mma1),  # type: ignore[arg-type]
                (
                    cute.size(tiled_mma1.permutation_mnk[1]),
                    cute.size(tiled_mma1.permutation_mnk[2]),
                ),
            )
            # FC2 (tiled_mma) copies
            smem_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
            smem_copy_B = cute.make_tiled_copy_B(atom_ld_B, tiled_mma)
            smem_copy_SFA = cute.make_tiled_copy(
                atom_ld_SF,
                self._dense_cls._get_layoutSFA_TV(self, tiled_mma),  # type: ignore[arg-type]
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_copy_SFB = cute.make_tiled_copy(
                atom_ld_SF,
                self._dense_cls._get_layoutSFB_TV(self, tiled_mma),  # type: ignore[arg-type]
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_ld_A1 = smem_copy_A1.get_slice(tidx)
            thr_ld_B1 = smem_copy_B1.get_slice(tidx)
            thr_ld_SFA1 = smem_copy_SFA1.get_slice(tidx)
            thr_ld_SFB1 = smem_copy_SFB1.get_slice(tidx)
            thr_ld_A = smem_copy_A.get_slice(tidx)
            thr_ld_B = smem_copy_B.get_slice(tidx)
            thr_ld_SFA = smem_copy_SFA.get_slice(tidx)
            thr_ld_SFB = smem_copy_SFB.get_slice(tidx)

            csA1 = thr_ld_A1.partition_S(sA1)
            crA1 = thr_ld_A1.retile(tCrA1)
            csB1 = thr_ld_B1.partition_S(sB1)
            crB1 = thr_ld_B1.retile(tCrB1)
            csSFB1_0 = thr_ld_SFB1.partition_S(sSFB1_0)
            csSFB1_1 = thr_ld_SFB1.partition_S(sSFB1_1)
            fz_crSFB1_0 = cute.filter_zeros(thr_ld_SFB1.retile(tCrSFB1_0))
            fz_crSFB1_1 = cute.filter_zeros(thr_ld_SFB1.retile(tCrSFB1_1))
            csA2 = thr_ld_A.partition_S(sA2)
            crA2 = thr_ld_A.retile(tCrA2)
            csSFA2 = thr_ld_SFA.partition_S(sSFA2_tile)
            fz_crSFA2 = cute.filter_zeros(thr_ld_SFA.retile(tCrSFA2))
            csB2 = thr_ld_B.partition_S(sB2)
            crB2 = thr_ld_B.retile(tCrB2)
            csSFB2_full = thr_ld_SFB.partition_S(sSFB2)
            fz_crSFB2 = cute.filter_zeros(thr_ld_SFB.retile(tCrSFB2))

            # FC1 epilogue (32 x 64 bf16 staging in sC1)
            _is_m_major = self.c_layout.is_m_major_c()
            copy_atom_r2s = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2), cutlass.BFloat16
            )
            tiled_copy_C_Atom1 = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma1)
            tiled_copy_r2s1 = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom1)
            thr_copy_r2s1 = tiled_copy_r2s1.get_slice(tidx)
            tRS_sD1 = thr_copy_r2s1.partition_D(sC1)
            tRS_rGate = tiled_copy_r2s1.retile(gate_acc)
            tRS_rUp = tiled_copy_r2s1.retile(up_acc)
            rD1_shape = cute.shape(thr_copy_r2s1.partition_S(sC1))
            tRS_rD1_layout = cute.make_layout(rD1_shape[:3])
            tRS_rD1 = cute.make_rmem_tensor(tRS_rD1_layout.shape, self.acc_dtype)
            tRS_rD1_out = cute.make_rmem_tensor(tRS_rD1_layout.shape, cutlass.BFloat16)
            mma_tile_m1 = self.tile_m // cute.size(tRS_rGate, mode=[1])
            mma_tile_n1 = self.fc1_tile_n // cute.size(tRS_rGate, mode=[2])
            MmaMPerEpiM1 = self.epi1_tile[0] // mma_tile_m1
            MmaNPerEpiN1 = self.epi1_tile[1] // mma_tile_n1

            # FC2 epilogue (32 x 128 bf16 staging in sC)
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            tRS_rDown = tiled_copy_r2s.retile(down_acc)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)
            mma_tile_m = self.tile_m // cute.size(tRS_rDown, mode=[1])
            mma_tile_n = self.fc2_tile_n // cute.size(tRS_rDown, mode=[2])
            MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
            MmaNPerEpiN = self.epi_tile[1] // mma_tile_n

            scatter_N = Int32(scatter_output.shape[1])
            lane_id = Int32(tidx) & Int32(31)
            warp_in_tile = Int32(tidx) >> Int32(5)
            if cutlass.const_expr(self.decode_reform):
                warp_m_base = Int32(0)
                warp_n_base = warp_in_tile * Int32(64)
            else:
                warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)
                warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)
            a2_rows = Int32(self.tile_m)
            sA2_u8 = cute.recast_tensor(sA2[None, None, 0], cutlass.Uint8)
            sf_blocks_per_half = Int32(self.fc1_tile_n // self.sf_vec_size)   # 4

            num_persistent_clusters = n_active
            cluster_shape_mn = (
                Int32(self.cluster_shape_mn[0]),
                Int32(self.cluster_shape_mn[1]),
            )
            cta_id_in_cluster = (
                Int32(bidx % cluster_shape_mn[0]),
                Int32(bidy % cluster_shape_mn[1]),
                Int32(0),
            )
            current_work_linear_idx = start_work_idx
            current_local_expert_idx = Int32(0)
            accum_tile_m = Int32(0)
            item_no = Int32(0)
            role = Int32(2)
            if current_work_linear_idx >= split_base:
                role = Int32(0)
            if helper_idx >= Int32(0):
                if current_work_linear_idx >= total_items:
                    current_work_linear_idx = helper_idx
                    role = Int32(1)
                    helper_idx = Int32(-1)
            tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                _compact_static_get_work_tile(
                    row_counts,
                    active_expert_count,
                    tile_m=Int32(self.tile_m),
                    num_tiles_n=Int32(self.output_tile_count_n),
                    cluster_shape_mn=cluster_shape_mn,
                    current_work_linear_idx=current_work_linear_idx,
                    current_local_expert_idx=current_local_expert_idx,
                    accum_tile_m=accum_tile_m,
                    cta_id_in_cluster=cta_id_in_cluster,
                )
            )
            peek = fc1_pipeline.consumer_try_wait(fc1_cons_state)
            if is_valid_tile:
                fc1_pipeline.consumer_wait(fc1_cons_state, peek)

            while is_valid_tile:
                local_expert_idx = tile_coord[2]
                weight_expert_idx = weight_expert_ids[local_expert_idx]
                alpha_value = alpha[weight_expert_idx].to(cutlass.Float32)
                valid_rows = row_counts[local_expert_idx]
                tile_m_base = tile_coord[0] * Int32(self.tile_m)
                stamp_item = stamp_row + Int32(2) + item_no * Int32(5)
                if cutlass.const_expr(self.stamps):
                    if Int32(tidx) == Int32(0):
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_item),
                                cute.arch.globaltimer(),
                            )
                # the A scale block holds 4 m-tiles: this tile's sub-tile
                sfa_tile_offset = tile_coord[0] % self.sfa_tiles_per_block
                sSFA1_tile = cute.local_tile(
                    sSFA1,
                    cute.slice_(self.fc1_tile_shape_mnk, (None, 0, None)),
                    (sfa_tile_offset, 0, None),
                )
                csSFA1_tile = thr_ld_SFA1.partition_S(sSFA1_tile)
                tCrSFA1_tile = self._dense_cls._partition_fragment_SFA(
                    self, sSFA1_tile[None, None, 0], thr_mma1, tidx  # type: ignore[arg-type]
                )
                fz_crSFA1_tile = cute.filter_zeros(thr_ld_SFA1.retile(tCrSFA1_tile))
                valid_tile_rows = valid_rows - tile_m_base
                if valid_tile_rows > Int32(self.tile_m):
                    valid_tile_rows = Int32(self.tile_m)
                if valid_tile_rows < Int32(0):
                    valid_tile_rows = Int32(0)

                cache_row = Int32(tidx)
                if cache_row < Int32(_COMPACT_STATIC_TILE_M):
                    tok = Int32(0)
                    wv = cutlass.Float32(0.0)
                    if cache_row < valid_tile_rows:
                        tok = token_map[local_expert_idx, tile_m_base + cache_row].to(
                            Int32
                        )
                        wv = token_weights[
                            local_expert_idx, tile_m_base + cache_row
                        ].to(cutlass.Float32)
                    _st_shared_i32(scatter_tok_base_addr + cache_row * Int32(4), tok)
                    _st_shared_f32(scatter_weight_base_addr + cache_row * Int32(4), wv)

                down_alpha_value = down_alpha[weight_expert_idx].to(cutlass.Float32)
                gs_value = global_scale[weight_expert_idx].to(cutlass.Float32)
                if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                    0.0
                ):
                    if self.fast_math:
                        gs_value = rcp_approx_ftz(gs_value)
                    else:
                        gs_value = cutlass.Float32(1.0) / gs_value

                if cutlass.const_expr(self.split):
                    if role != Int32(2):
                        # split item: the other half of the intermediate (and
                        # its scales) must read as zero in FC2
                        zb = Int32(tidx) * Int32(16)
                        for zi in cutlass.range_constexpr(16):
                            sA2_u8[zb + Int32(zi)] = Uint8(0)
                        zs = Int32(tidx) * Int32(8)
                        for zi in cutlass.range_constexpr(8):
                            st_shared_u8(sfa2_base_addr + zs + Int32(zi), Uint8(0))
                # ============================================================
                # PHASE A: FC1 as two 64-wide halves (gate + up per stage)
                # ============================================================
                for h in cutlass.range_constexpr(self.fc1_halves):
                    if (role - Int32(2)) * (role - Int32(h)) == Int32(0):
                        if cutlass.const_expr(h == 0):
                            fz_crSFB1 = fz_crSFB1_0
                            tCrSFB1 = tCrSFB1_0
                            csSFB1 = csSFB1_0
                        else:
                            fz_crSFB1 = fz_crSFB1_1
                            tCrSFB1 = tCrSFB1_1
                            csSFB1 = csSFB1_1
                        gate_acc.fill(0.0)
                        up_acc.fill(0.0)
                        # per k tile: a gate stage, then an up stage (each
                        # A + one B + SFA + SFB over K 512 = 8 k blocks)
                        for _k_tile in range(0, k_tile_cnt1, 1, unroll=1):  # type: ignore[call-overload]
                            if cutlass.const_expr(self.a_ring):
                                apeek = a_pipeline.consumer_try_wait(a_cons_state)
                                a_pipeline.consumer_wait(a_cons_state, apeek)
                            for gu in cutlass.range_constexpr(2):
                                peek = fc1_pipeline.consumer_try_wait(fc1_cons_state)
                                fc1_pipeline.consumer_wait(fc1_cons_state, peek)
                                if cutlass.const_expr(self.reform_sf_pack):
                                    for sf_block in cutlass.range_constexpr(self.sf1_packed_blocks):
                                        self._sf_expand_stage(
                                            sfb1_base_addr
                                            + fc1_cons_state.index * Int32(self.sf1_block_bytes)
                                            + Int32(sf_block * 2048), Int32(tidx), 2048,
                                        )
                                elif cutlass.const_expr(self.sf_pack):
                                    self._sf_expand_stage(
                                        sfb1_base_addr
                                        + fc1_cons_state.index * Int32(self.sf1_block_bytes),
                                        Int32(tidx), self.sf1_block_bytes,
                                    )
                                if cutlass.const_expr(self.a_ring):
                                    a_slot = a_cons_state.index
                                else:
                                    a_slot = fc1_cons_state.index
                                csA_p = csA1[None, None, None, a_slot]
                                csB_p = csB1[None, None, None, fc1_cons_state.index]
                                fz_csSFA_p = cute.filter_zeros(
                                    csSFA1_tile[None, None, None, a_slot]
                                )
                                fz_csSFB_p = cute.filter_zeros(
                                    csSFB1[None, None, None, fc1_cons_state.index]
                                )
                                cute.copy(smem_copy_A1, csA_p[None, None, 0], crA1[None, None, 0])
                                cute.copy(smem_copy_B1, csB_p[None, None, 0], crB1[None, None, 0])
                                cute.copy(
                                    smem_copy_SFA1, fz_csSFA_p[None, None, 0],
                                    fz_crSFA1_tile[None, None, 0],
                                )
                                cute.copy(
                                    smem_copy_SFB1, fz_csSFB_p[None, None, 0],
                                    fz_crSFB1[None, None, 0],
                                )
                                for k_block_idx in cutlass.range_constexpr(num_k_blocks1):
                                    k_next = (
                                        0 if k_block_idx + 1 == num_k_blocks1
                                        else k_block_idx + 1
                                    )
                                    if k_next > 0:
                                        cute.copy(
                                            smem_copy_A1, csA_p[None, None, k_next],
                                            crA1[None, None, k_next],
                                        )
                                        cute.copy(
                                            smem_copy_B1, csB_p[None, None, k_next],
                                            crB1[None, None, k_next],
                                        )
                                        cute.copy(
                                            smem_copy_SFA1, fz_csSFA_p[None, None, k_next],
                                            fz_crSFA1_tile[None, None, k_next],
                                        )
                                        cute.copy(
                                            smem_copy_SFB1, fz_csSFB_p[None, None, k_next],
                                            fz_crSFB1[None, None, k_next],
                                        )
                                    for _mt in range(self.num_m_tiles):
                                        for _nt in range(self.num_n_tiles1):
                                            mma_atom.set(
                                                WarpField.SFA,
                                                tCrSFA1_tile[None, _mt, k_block_idx].iterator,
                                            )
                                            mma_atom.set(
                                                WarpField.SFB,
                                                tCrSFB1[None, _nt, k_block_idx].iterator,
                                            )
                                            if cutlass.const_expr(gu == 0):
                                                cute.gemm(
                                                    mma_atom,
                                                    gate_acc[None, _mt, _nt],
                                                    tCrA1[None, _mt, k_block_idx],
                                                    tCrB1[None, _nt, k_block_idx],
                                                    gate_acc[None, _mt, _nt],
                                                )
                                            else:
                                                cute.gemm(
                                                    mma_atom,
                                                    up_acc[None, _mt, _nt],
                                                    tCrA1[None, _mt, k_block_idx],
                                                    tCrB1[None, _nt, k_block_idx],
                                                    up_acc[None, _mt, _nt],
                                                )
                                fc1_pipeline.consumer_release(fc1_cons_state)
                                fc1_cons_state.advance()
                            if cutlass.const_expr(self.a_ring):
                                a_pipeline.consumer_release(a_cons_state)
                                a_cons_state.advance()

                        # ---- activation of this half -> sC1 -> quant into sA2 ----
                        epi_m_valid = valid_rows - tile_m_base
                        if epi_m_valid > Int32(0):
                            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN1):
                                for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM1):
                                    tRS_rD_slice = tRS_rD1[(None, mma_m_in_epi, mma_n_in_epi)]
                                    gate_slice = tRS_rGate[(None, mma_m_in_epi, mma_n_in_epi)]
                                    up_slice = tRS_rUp[(None, mma_m_in_epi, mma_n_in_epi)]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(tRS_rD_slice)
                                    ):
                                        g = alpha_value * gate_slice[elem_idx]
                                        u = alpha_value * up_slice[elem_idx]
                                        tRS_rD_slice[elem_idx] = gated_activation_f32(
                                            g,
                                            u,
                                            activation=self.activation,
                                            limit=self.swiglu_limit,
                                            alpha=self.swiglu_alpha,
                                            beta=self.swiglu_beta,
                                            fast_math=self.fast_math,
                                        )
                            acc_vec = tRS_rD1.load()
                            acc_vec = acc_vec.to(cutlass.BFloat16)
                            tRS_rD1_out.store(acc_vec)
                            cute.copy(
                                tiled_copy_r2s1, tRS_rD1_out, tRS_sD1[(None, None, None, 0)]
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        epi_rows = epi_m_valid
                        if epi_rows > Int32(self.tile_m):
                            epi_rows = Int32(self.tile_m)
                        if epi_rows < Int32(0):
                            epi_rows = Int32(0)
                        quant_idx = Int32(tidx)
                        while quant_idx < epi_rows * sf_blocks_per_half:
                            local_row = quant_idx // sf_blocks_per_half
                            row = local_row
                            sfb_local = quant_idx - local_row * sf_blocks_per_half
                            sf_block = Int32(h) * sf_blocks_per_half + sfb_local
                            block_start = sfb_local * Int32(self.sf_vec_size)

                            values = cute.make_rmem_tensor(
                                (self.sf_vec_size,), cutlass.Float32
                            )
                            block_max = cutlass.Float32(0.0)
                            for elem_idx in cutlass.range_constexpr(self.sf_vec_size):
                                value = cutlass.Float32(
                                    sC1[local_row, block_start + elem_idx, 0]
                                )
                                values[elem_idx] = value
                                block_max = fmax_f32(block_max, fabs_f32(value))
                            scale_byte = Uint8(0)
                            packed_lo = Uint64(0)
                            if self.fast_math:
                                packed_lo, scale_byte = quantize_block_fp4_fast(
                                    values, block_max, gs_value
                                )
                            else:
                                packed_lo, scale_byte = quantize_block_fp4(
                                    values, block_max, gs_value
                                )
                            packed_base = sf_block * Int32(self.sf_vec_size // 2)
                            xor_bits = ((row >> Int32(1)) & Int32(0x3)) << Int32(4)
                            for byte_idx in cutlass.range_constexpr(self.sf_vec_size // 2):
                                src_pcol = packed_base + Int32(byte_idx)
                                dst_flat = (src_pcol ^ xor_bits) * a2_rows + row
                                byte_val = Uint8(
                                    (packed_lo >> Uint64(byte_idx * 8)) & Uint64(0xFF)
                                )
                                if cutlass.const_expr(self.decode_reform):
                                    # Convert the outer FP4 nibble offset to
                                    # bytes BEFORE applying the pointer swizzle.
                                    fp4_offset = cute.crd2idx(
                                        (row, src_pcol * Int32(2), 0), a2_smem_layout.outer
                                    )
                                    byte_offset = fp4_offset // Int32(2)
                                    byte_offset = byte_offset ^ ((byte_offset >> Int32(3)) & Int32(0x30))
                                    st_shared_u8(a2_base_addr + byte_offset, byte_val)
                                else:
                                    sA2_u8[dst_flat] = byte_val
                            outer_m_idx = row % Int32(32)
                            inner_m_idx = row // Int32(32)
                            inner_k_idx = sf_block % Int32(4)
                            k_tile_idx = sf_block // Int32(4)
                            sf_raw_idx = (
                                k_tile_idx * Int32(32 * 4 * 4)
                                + outer_m_idx * Int32(4 * 4)
                                + inner_m_idx * Int32(4)
                                + inner_k_idx
                            )
                            st_shared_u8(sfa2_base_addr + sf_raw_idx, scale_byte)
                            quant_idx += Int32(
                                self.num_mma_warps * self.num_threads_per_warp
                            )
                        # sC1 is reused by the next half / next item after this
                        self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.stamps):
                    if Int32(tidx) == Int32(0):
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_item + Int32(1)),
                                cute.arch.globaltimer(),
                            )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()
                if cutlass.const_expr(self.stamps):
                    if Int32(tidx) == Int32(0):
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_item + Int32(2)),
                                cute.arch.globaltimer(),
                            )

                # ============================================================
                # PHASE B: FC2 sweep (v2 verbatim)
                # ============================================================
                csA2_p = csA2[None, None, None, 0]
                fz_csSFA2_p = cute.filter_zeros(csSFA2[None, None, None, 0])
                for _kb in cutlass.range_constexpr(num_k_blocks):
                    cute.copy(smem_copy_A, csA2_p[None, None, _kb], crA2[None, None, _kb])
                    cute.copy(
                        smem_copy_SFA, fz_csSFA2_p[None, None, _kb],
                        fz_crSFA2[None, None, _kb],
                    )

                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    fc2_peek = fc2_pipeline.consumer_try_wait(fc2_cons_state)
                    fc2_pipeline.consumer_wait(fc2_cons_state, fc2_peek)
                    if cutlass.const_expr(self.reform_sf_pack):
                        if cutlass.const_expr(self.decode_reform):
                            self._sf_expand_stage(
                                sfb2_base_addr + fc2_cons_state.index * Int32(2048),
                                Int32(tidx), 2048,
                            )
                        else:
                            self._sf_expand_stage(
                                sfb2_base_addr + fc2_cons_state.index * Int32(1024),
                                Int32(tidx), 1024,
                            )
                    csB2_p = csB2[None, None, None, fc2_cons_state.index]
                    fz_csSFB2_p = cute.filter_zeros(
                        csSFB2_full[None, None, None, fc2_cons_state.index]
                    )
                    cute.copy(smem_copy_B, csB2_p[None, None, 0], crB2[None, None, 0])
                    cute.copy(
                        smem_copy_SFB, fz_csSFB2_p[None, None, 0], fz_crSFB2[None, None, 0]
                    )
                    down_acc.fill(0.0)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )
                        if k_block_idx == num_k_blocks - 1:
                            fc2_pipeline.consumer_release(fc2_cons_state)
                            fc2_cons_state.advance()
                        if k_next > 0:
                            cute.copy(
                                smem_copy_B, csB2_p[None, None, k_next],
                                crB2[None, None, k_next],
                            )
                            cute.copy(
                                smem_copy_SFB, fz_csSFB2_p[None, None, k_next],
                                fz_crSFB2[None, None, k_next],
                            )
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA2[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB2[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    down_acc[None, _mt, _nt],
                                    tCrA2[None, _mt, k_block_idx],
                                    tCrB2[None, _nt, k_block_idx],
                                    down_acc[None, _mt, _nt],
                                )

                    tile_n_base_cur = output_tile_idx * Int32(self.fc2_tile_n)
                    for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                        for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                            tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                            down_epi_acc_slice = tRS_rDown[(None, mma_m_in_epi, mma_n_in_epi)]
                            for elem_idx in cutlass.range_constexpr(
                                cute.size(tRS_rD_slice)
                            ):
                                tRS_rD_slice[elem_idx] = (
                                    down_alpha_value * down_epi_acc_slice[elem_idx]
                                )
                    acc_vec = tRS_rD.load()
                    acc_vec = acc_vec.to(cutlass.BFloat16)
                    tRS_rD_out.store(acc_vec)
                    cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, 0)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    warp_epi_rows = valid_tile_rows - warp_m_base
                    if warp_epi_rows > Int32(64):
                        warp_epi_rows = Int32(64)
                    if warp_epi_rows < Int32(0):
                        warp_epi_rows = Int32(0)
                    tile_vec_cols = Int32(64) // Int32(8)
                    vec_idx = lane_id
                    while vec_idx < warp_epi_rows * tile_vec_cols:
                        local_row = vec_idx // tile_vec_cols
                        local_vec_col = vec_idx - local_row * tile_vec_cols
                        local_col = warp_n_base + local_vec_col * Int32(8)
                        global_col = tile_n_base_cur + local_col
                        cached_row = warp_m_base + local_row
                        tok = ld_shared_i32_relaxed(
                            scatter_tok_base_addr + cached_row * Int32(4)
                        )
                        wv = _ld_shared_f32(
                            scatter_weight_base_addr + cached_row * Int32(4)
                        )
                        sc_v0 = cutlass.Float32(sC[warp_m_base + local_row, local_col, 0])
                        sc_v1 = cutlass.Float32(
                            sC[warp_m_base + local_row, local_col + Int32(1), 0]
                        )
                        sc_v2 = cutlass.Float32(
                            sC[warp_m_base + local_row, local_col + Int32(2), 0]
                        )
                        sc_v3 = cutlass.Float32(
                            sC[warp_m_base + local_row, local_col + Int32(3), 0]
                        )
                        sc_v4 = cutlass.Float32(
                            sC[warp_m_base + local_row, local_col + Int32(4), 0]
                        )
                        sc_v5 = cutlass.Float32(
                            sC[warp_m_base + local_row, local_col + Int32(5), 0]
                        )
                        sc_v6 = cutlass.Float32(
                            sC[warp_m_base + local_row, local_col + Int32(6), 0]
                        )
                        sc_v7 = cutlass.Float32(
                            sC[warp_m_base + local_row, local_col + Int32(7), 0]
                        )
                        scatter_add_v4_bf16x2_to_f32(
                            get_ptr_as_int64(
                                scatter_output, tok * scatter_N + global_col
                            ),
                            wv * sc_v0, wv * sc_v1, wv * sc_v2, wv * sc_v3,
                            wv * sc_v4, wv * sc_v5, wv * sc_v6, wv * sc_v7,
                        )
                        vec_idx += Int32(self.num_threads_per_warp)
                    self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.stamps):
                    if Int32(tidx) == Int32(0):
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_item + Int32(3)),
                                cute.arch.globaltimer(),
                            )

                item_no += Int32(1)
                current_work_linear_idx += num_persistent_clusters
                role = Int32(2)
                if current_work_linear_idx >= split_base:
                    role = Int32(0)
                if helper_idx >= Int32(0):
                    if current_work_linear_idx >= total_items:
                        current_work_linear_idx = helper_idx
                        role = Int32(1)
                        helper_idx = Int32(-1)
                tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                    _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_m),
                        num_tiles_n=Int32(self.output_tile_count_n),
                        cluster_shape_mn=cluster_shape_mn,
                        current_work_linear_idx=current_work_linear_idx,
                        current_local_expert_idx=current_local_expert_idx,
                        accum_tile_m=accum_tile_m,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )
                )
                peek = fc1_pipeline.consumer_try_wait(fc1_cons_state)
                if is_valid_tile:
                    fc1_pipeline.consumer_wait(fc1_cons_state, peek)
            if cutlass.const_expr(self.stamps):
                if Int32(tidx) == Int32(0):
                    _st_global_i64(
                        get_ptr_as_int64(stamps, stamp_row + Int32(STAMP_MMA_END)),
                        cute.arch.globaltimer(),
                    )
                    _st_global_i64(
                        get_ptr_as_int64(stamps, stamp_row + Int32(STAMP_MMA_END + 1)),
                        Int64(item_no),
                    )

        # ===================================================================
        # DMA WARP (warp 4): FC1 half 0, half 1, then FC2, item after item
        # ===================================================================
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            num_persistent_clusters = n_active
            cluster_shape_mn = (
                Int32(self.cluster_shape_mn[0]),
                Int32(self.cluster_shape_mn[1]),
            )
            cta_id_in_cluster = (
                Int32(bidx % cluster_shape_mn[0]),
                Int32(bidy % cluster_shape_mn[1]),
                Int32(0),
            )
            current_work_linear_idx = start_work_idx
            current_local_expert_idx = Int32(0)
            accum_tile_m = Int32(0)
            item_no = Int32(0)
            is_dma_lane0 = Int32(tidx) == Int32(self.tma_load_warp_id * 32)
            # packed FC1 scales: [E][row block][k tile] of _SF_STAGE_BYTES each,
            # the order the host packer writes (moe_sf_pack.pack_sf_inline)
            sfb1_packed_base = get_ptr_as_int64(sfb1_packed, Int32(0))
            sf_blocks_per_expert = Int64(cute.size(sfb1_packed.shape[1]))
            sfb2_packed_base = get_ptr_as_int64(sfb2_packed, Int32(0))
            sf2_blocks_per_expert = Int64(cute.size(sfb2_packed.shape[1]))
            n_slices = Int32(self.output_tile_count_n)
            role = Int32(2)
            if current_work_linear_idx >= split_base:
                role = Int32(0)
            if helper_idx >= Int32(0):
                if current_work_linear_idx >= total_items:
                    current_work_linear_idx = helper_idx
                    role = Int32(1)
                    helper_idx = Int32(-1)
            tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                _compact_static_get_work_tile(
                    row_counts,
                    active_expert_count,
                    tile_m=Int32(self.tile_m),
                    num_tiles_n=Int32(self.output_tile_count_n),
                    cluster_shape_mn=cluster_shape_mn,
                    current_work_linear_idx=current_work_linear_idx,
                    current_local_expert_idx=current_local_expert_idx,
                    accum_tile_m=accum_tile_m,
                    cta_id_in_cluster=cta_id_in_cluster,
                )
            )

            while is_valid_tile:
                tc = tile_coord
                intermediate_slice = tc[1]
                local_expert_idx = tc[2]
                weight_expert_idx = weight_expert_ids[local_expert_idx]
                stamp_dma = stamp_row + Int32(STAMP_DMA_BASE) + item_no * Int32(3)
                if cutlass.const_expr(self.stamps):
                    if is_dma_lane0:
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_dma),
                                cute.arch.globaltimer(),
                            )
                tAgA_mk = tAgA[(None, tc[0], None, local_expert_idx)]
                sfa_tile_coord_m = tc[0] // self.sfa_tiles_per_block
                tAgSFA_mk = tAgSFA[(None, sfa_tile_coord_m, None, local_expert_idx)]

                # ---- FC1: two 64-wide halves of the 128-wide slice ----
                for h in cutlass.range_constexpr(self.fc1_halves):
                    if (role - Int32(2)) * (role - Int32(h)) == Int32(0):
                        up_tile = intermediate_slice * Int32(self.fc1_halves) + Int32(h)
                        gate_tile = gate_tile_cnt + up_tile
                        tBgB_up_nk = tBgB_w13[(None, up_tile, None, weight_expert_idx)]
                        tBgB_gate_nk = tBgB_w13[(None, gate_tile, None, weight_expert_idx)]
                        # the SFB gmem tile is the 128-row block both halves
                        # share (39차 §3b: a 64-row box is not expressible)
                        sfb_up_idx = up_tile // Int32(self.sfb1_tiles_per_block)
                        sfb_gate_idx = gate_tile // Int32(self.sfb1_tiles_per_block)
                        if cutlass.const_expr(not self.reform_sf_pack):
                            tBgSFB_up_nk = tBgSFB_w13[(None, sfb_up_idx, None, weight_expert_idx)]
                            tBgSFB_gate_nk = tBgSFB_w13[(None, sfb_gate_idx, None, weight_expert_idx)]
                        for k_tile in range(0, k_tile_cnt1, 1, unroll=1):  # type: ignore[call-overload]
                            if cutlass.const_expr(self.a_ring):
                                a_pipeline.producer_acquire(a_prod_state)
                                abar = a_pipeline.producer_get_barrier(a_prod_state)
                                cute.copy(
                                    tma_a, tAgA_mk[(None, k_tile)],
                                    tAsA[(None, a_prod_state.index)], tma_bar_ptr=abar,
                                )
                                cute.copy(
                                    tma_sfa, tAgSFA_mk[(None, k_tile)],
                                    tAsSFA[(None, a_prod_state.index)], tma_bar_ptr=abar,
                                )
                                a_pipeline.producer_commit(a_prod_state)
                                a_prod_state.advance()
                            for gu in cutlass.range_constexpr(2):
                                fc1_pipeline.producer_acquire(fc1_prod_state)
                                bar = fc1_pipeline.producer_get_barrier(fc1_prod_state)
                                if cutlass.const_expr(not self.skip_a and not self.a_ring):
                                    cute.copy(
                                        tma_a, tAgA_mk[(None, k_tile)],
                                        tAsA[(None, fc1_prod_state.index)], tma_bar_ptr=bar,
                                    )
                                if cutlass.const_expr(gu == 0):
                                    cute.copy(
                                        tma_b_w13, tBgB_gate_nk[(None, k_tile)],
                                        tBsB1[(None, fc1_prod_state.index)], tma_bar_ptr=bar,
                                    )
                                else:
                                    cute.copy(
                                        tma_b_w13, tBgB_up_nk[(None, k_tile)],
                                        tBsB1[(None, fc1_prod_state.index)], tma_bar_ptr=bar,
                                    )
                                if cutlass.const_expr(not self.skip_a and not self.a_ring):
                                    cute.copy(
                                        tma_sfa, tAgSFA_mk[(None, k_tile)],
                                        tAsSFA[(None, fc1_prod_state.index)], tma_bar_ptr=bar,
                                    )
                                if cutlass.const_expr(not self.skip_sf):
                                    if cutlass.const_expr(self.reform_sf_pack):
                                        if is_dma_lane0:
                                            if cutlass.const_expr(gu == 0):
                                                sfb_blk = sfb_gate_idx
                                            else:
                                                sfb_blk = sfb_up_idx
                                            # K512 uses consecutive SF6 K256
                                            # blocks with independent bases.
                                            # Their shared destinations are
                                            # disjoint even after expansion.
                                            for sf_block in cutlass.range_constexpr(self.sf1_packed_blocks):
                                                _bulk_g2s(
                                                    sfb1_base_addr
                                                    + fc1_prod_state.index * Int32(self.sf1_block_bytes)
                                                    + Int32(sf_block * 2048),
                                                    sfb1_packed_base
                                                    + (Int64(weight_expert_idx) * sf_blocks_per_expert
                                                       + (Int64(sfb_blk) * Int64(k_tile_cnt1)
                                                          + Int64(k_tile)) * Int64(self.sf1_packed_blocks)
                                                       + Int64(sf_block)) * Int64(1552),
                                                    Int32(1552), shared_ptr_to_u32(bar),
                                                )
                                    elif cutlass.const_expr(self.sf_pack):
                                        # Packed scales, one request,
                                        # into the stage's own 4 KB buffer; the
                                        # MMA warps expand it there (39차 §4c)
                                        if is_dma_lane0:
                                            if cutlass.const_expr(gu == 0):
                                                sfb_blk = sfb_gate_idx
                                            else:
                                                sfb_blk = sfb_up_idx
                                            _bulk_g2s(
                                                sfb1_base_addr
                                                + fc1_prod_state.index * Int32(self.sf1_block_bytes),
                                                sfb1_packed_base
                                                + (Int64(weight_expert_idx) * sf_blocks_per_expert
                                                   + Int64(sfb_blk) * Int64(k_tile_cnt1)
                                                   + Int64(k_tile)) * Int64(self.sf1_stage_bytes),
                                                Int32(self.sf1_stage_bytes),
                                                shared_ptr_to_u32(bar),
                                            )
                                    elif cutlass.const_expr(gu == 0):
                                        cute.copy(
                                            tma_sfb_w13, tBgSFB_gate_nk[(None, k_tile)],
                                            tBsSFB1[(None, fc1_prod_state.index)], tma_bar_ptr=bar,
                                        )
                                    else:
                                        cute.copy(
                                            tma_sfb_w13, tBgSFB_up_nk[(None, k_tile)],
                                            tBsSFB1[(None, fc1_prod_state.index)], tma_bar_ptr=bar,
                                        )
                                fc1_pipeline.producer_commit(fc1_prod_state)
                                fc1_prod_state.advance()
                if cutlass.const_expr(self.stamps):
                    if is_dma_lane0:
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_dma + Int32(1)),
                                cute.arch.globaltimer(),
                            )

                # ---- FC2: the item's 32 down tiles ----
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    fc2_pipeline.producer_acquire(fc2_prod_state)
                    bar2 = fc2_pipeline.producer_get_barrier(fc2_prod_state)
                    cute.copy(
                        tma_b_down,
                        tBgB_down[(None, output_tile_idx, intermediate_slice,
                                   weight_expert_idx)],
                        tBsB2[(None, fc2_prod_state.index)],
                        tma_bar_ptr=bar2,
                    )
                    if cutlass.const_expr(self.reform_sf_pack):
                        if is_dma_lane0:
                            if cutlass.const_expr(self.decode_reform):
                                sf2_dest = sfb2_base_addr + fc2_prod_state.index * Int32(2048)
                                sf2_tile = output_tile_idx
                            else:
                                sf2_dest = sfb2_base_addr + fc2_prod_state.index * Int32(1024)
                                sf2_tile = output_tile_idx // Int32(2)
                            sf2_source = (sfb2_packed_base
                                + (Int64(weight_expert_idx) * sf2_blocks_per_expert
                                   + Int64(sf2_tile) * Int64(n_slices)
                                   + Int64(intermediate_slice)) * Int64(1552))
                            if cutlass.const_expr(self.decode_reform):
                                _bulk_g2s(sf2_dest, sf2_source, Int32(1552), shared_ptr_to_u32(bar2))
                            else:
                                self._sf6_copy_fc2_half(sf2_dest, sf2_source,
                                    shared_ptr_to_u32(bar2), output_tile_idx % Int32(2))
                    else:
                        cute.copy(
                            tma_sfb_down,
                            tBgSFB_down[(None, output_tile_idx, intermediate_slice,
                                         weight_expert_idx)],
                            tBsSFB2[(None, fc2_prod_state.index)],
                            tma_bar_ptr=bar2,
                        )
                    fc2_pipeline.producer_commit(fc2_prod_state)
                    fc2_prod_state.advance()
                if cutlass.const_expr(self.stamps):
                    if is_dma_lane0:
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_dma + Int32(2)),
                                cute.arch.globaltimer(),
                            )

                item_no += Int32(1)
                current_work_linear_idx += num_persistent_clusters
                role = Int32(2)
                if current_work_linear_idx >= split_base:
                    role = Int32(0)
                if helper_idx >= Int32(0):
                    if current_work_linear_idx >= total_items:
                        current_work_linear_idx = helper_idx
                        role = Int32(1)
                        helper_idx = Int32(-1)
                tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                    _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_m),
                        num_tiles_n=Int32(self.output_tile_count_n),
                        cluster_shape_mn=cluster_shape_mn,
                        current_work_linear_idx=current_work_linear_idx,
                        current_local_expert_idx=current_local_expert_idx,
                        accum_tile_m=accum_tile_m,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )
                )

            fc1_pipeline.producer_tail(fc1_prod_state)
            fc2_pipeline.producer_tail(fc2_prod_state)
            if cutlass.const_expr(self.a_ring):
                a_pipeline.producer_tail(a_prod_state)
        return


def ep_tiled_compile_spec(*, num_tokens, max_rows=256, max_active_clusters=48,
                          topk_ids_dtype=None, input_scales_are_reciprocal=False,
                          fast_math=True, reform_sf_pack=False):
    """Build real CuTe fake operands without querying/initializing CUDA.

    Return (kernel, compile_args, cache_key). compile_args include the constexpr
    resident grid count and TVM-FFI fake stream; launch omits both as stock does.
    The normal CPU fleet probe can call cute.compile(kernel, *compile_args).
    """
    import torch
    from flashinfer.cute_dsl.utils import make_ptr
    geometry = ep_tiled_geometry(num_tokens, max_rows, max_active_clusters)
    scale_mode = ep_tiled_scale_mode(reform_sf_pack)
    if topk_ids_dtype is None:
        topk_ids_dtype = torch.int32
    if topk_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("EP tiled route IDs must be int32 or int64")
    m, mac = num_tokens, max_active_clusters
    k, n, state_E, weight_E, num_topk = 4096, 2048, 72, 72, 8
    sf_vec_size, sf_dtype = 16, cutlass.Float8E4M3FN
    a_dtype, ab_dtype = cutlass.BFloat16, cutlass.Float4E2M1FN
    weight_dtype, alpha_dtype = cutlass.Float4E2M1FN, cutlass.Float32
    tiled = True
    config = {"sf_pack": False, "reform_sf_pack": reform_sf_pack}
    TILED_W13_K_IN, TILED_W2_K_IN = 512, 128
    _STATIC_V2_STAMP_SLOTS = STAMP_SLOTS
    _align_up = lambda v, a: (v + a - 1) // a * a
    kernel = MoEStaticEPTiledKernel(
        num_tokens=m, max_rows=max_rows, max_active_clusters=mac,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math, reform_sf_pack=reform_sf_pack)
    w1_rows = 2 * n
    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)

    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype, (m * num_topk,), assumed_align=topk_ids_align
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m * num_topk,), assumed_align=4
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (max_rows, k, state_E), stride_order=(1, 0, 2), assumed_align=16
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * max_rows * (k // 2),), assumed_align=16
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * rows_pad_k * cols_pad_k,), assumed_align=16
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    if tiled:
        # tile-major storage (moe_static_kernel_v5): (rows, K_in, K/K_in, E),
        # K_in the stride-1 mode, then the rows -- one contiguous chunk per
        # (k tile, row), rows adjacent; the runtime view is the same 4-D
        # permutation of the re-laid-out bytes (_get_weight_views(tiled=True))
        if k % TILED_W13_K_IN != 0 or n % TILED_W2_K_IN != 0:
            raise ValueError(
                f"tiled expert weights need K % {TILED_W13_K_IN} == 0 and "
                f"I_tp % {TILED_W2_K_IN} == 0 (got K={k}, I_tp={n})"
            )
        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (w1_rows, TILED_W13_K_IN, k // TILED_W13_K_IN, weight_E),
            stride_order=(1, 0, 2, 3), assumed_align=16,
        )
        b_down_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (k, TILED_W2_K_IN, n // TILED_W2_K_IN, weight_E),
            stride_order=(1, 0, 2, 3), assumed_align=16,
        )
    else:
        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (w1_rows, k, weight_E), stride_order=(1, 0, 2), assumed_align=16
        )
        b_down_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (k, n, weight_E), stride_order=(1, 0, 2), assumed_align=16
        )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (weight_E,), assumed_align=4
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m, k), stride_order=(1, 0), assumed_align=16
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E, max_rows), stride_order=(1, 0), assumed_align=4
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (state_E, max_rows), stride_order=(1, 0), assumed_align=16
    )
    stamps_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (mac, _STATIC_V2_STAMP_SLOTS), stride_order=(1, 0), assumed_align=8
    )
    next_item_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    # cell q: the packed FC1 scales, (E, blocks per expert, stage bytes) u8; off
    # the lane a 16 B dummy the kernel never reads.
    if config.get("reform_sf_pack"):
        sfb1_packed_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (weight_E, (n * 2 // 128) * (k // 256), REFORM_SF_STAGE),
            stride_order=(2, 1, 0), assumed_align=16,
        )
    elif config.get("sf_pack"):
        _sf_blocks = (n * 2 // 128) * (k // TILED_W13_K_IN)
        sfb1_packed_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (weight_E, _sf_blocks, SF_STAGE_BYTES),
            # row-major, the torch tensor's own order: without this the fake is
            # compact in the OTHER direction (strides (1, E, ...)) and the call
            # fails with "Mismatched sfb1_packed.strides[0]"
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
    else:
        sfb1_packed_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (1, 1, 16), stride_order=(2, 1, 0), assumed_align=16
        )
    sfb2_packed_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (weight_E, (k // 256) * (n // 128), REFORM_SF_STAGE)
        if config.get("reform_sf_pack") else (1, 1, 16),
        stride_order=(2, 1, 0), assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    args = (
        a_input_fake, topk_ids_fake, topk_weights_fake, packed_a_fake,
        sfa_fake, packed_a_storage_fake, scale_storage_fake,
        barrier_count_fake, barrier_epoch_fake, b_w13_fake, sfb_w13_fake,
        b_down_fake, sfb_down_fake, row_counts_fake, active_expert_count_fake,
        weight_expert_ids_fake, global_to_local_expert_fake, input_gs_fake,
        alpha_fake, down_alpha_fake, global_scale_fake, scatter_fake,
        token_map_fake, token_weights_fake, stamps_fake, next_item_fake,
        sfb1_packed_fake, sfb2_packed_fake, mac, stream_fake,
    )
    key = (EP_TILED_CACHE_TAG, m, max_rows, mac, str(topk_ids_dtype),
           bool(input_scales_are_reciprocal), bool(fast_math),
           geometry["fc1"], geometry["fc2"], "nvfp4", scale_mode,
           "swigluoai_uninterleave", 1.0, 0.0, 10.0, "fp32_scatter")
    if reform_sf_pack and geometry["reform"]:
        key += (EP_TILED_A_RING_CACHE_TAG,)
    return kernel, args, key


_EP_TILED_KERNEL_CACHE = {}


def get_ep_tiled_decode_kernel(**kwargs):
    """Private cache namespace; cannot resolve to TP/static/micro artifacts."""
    import torch
    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import moe_dispatch
    # Shape-only admission/key stays cheap once the graph has been warmed.
    m = kwargs["num_tokens"]
    max_rows = kwargs.get("max_rows", 256)
    mac = kwargs.get("max_active_clusters", 48)
    geometry = ep_tiled_geometry(m, max_rows, mac)
    scale_mode = ep_tiled_scale_mode(kwargs.get("reform_sf_pack", False))
    dtype = kwargs.get("topk_ids_dtype") or torch.int32
    if dtype not in (torch.int32, torch.int64):
        raise TypeError("EP tiled route IDs must be int32 or int64")
    key = (EP_TILED_CACHE_TAG, m, max_rows, mac, str(dtype),
           bool(kwargs.get("input_scales_are_reciprocal", False)),
           bool(kwargs.get("fast_math", True)), geometry["fc1"], geometry["fc2"],
           "nvfp4", scale_mode, "swigluoai_uninterleave", 1.0, 0.0, 10.0,
           "fp32_scatter")
    if kwargs.get("reform_sf_pack", False) and geometry["reform"]:
        key += (EP_TILED_A_RING_CACHE_TAG,)
    if key in _EP_TILED_KERNEL_CACHE:
        return _EP_TILED_KERNEL_CACHE[key], mac
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("EP tiled decode specialization was not warmed before capture")
    kernel, args, actual_key = ep_tiled_compile_spec(**kwargs)
    if actual_key != key:
        raise RuntimeError("EP tiled compile/cache key mismatch")
    compiled = build_and_load_cute_dsl_kernel(
        "b12x_ep_tiled_decode", moe_dispatch._disk_kernel_name("ep_tiled_decode", key),
        lambda: cute.compile(kernel, *args, options="--opt-level 2 --enable-tvm-ffi"),
        extra_key_files=moe_dispatch._kernel_source_files() + (__file__,),
    )
    _EP_TILED_KERNEL_CACHE[key] = compiled
    return compiled, mac


@dataclass
class EPTiledDecodeScratch:
    scatter_fp32: object
    stamps: object
    counter: object
    dummy_scales: object
    max_active_clusters: int
    max_tokens: int


def allocate_ep_tiled_decode_scratch(*, device, max_active_clusters=48, max_tokens=32):
    """Allocate only before capture; callers keep the owner through graph life."""
    import torch
    ep_tiled_geometry(max_tokens, 256, max_active_clusters)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("EP tiled decode scratch must be allocated before capture")
    return EPTiledDecodeScratch(
        torch.empty((max_tokens, 4096), device=device, dtype=torch.float32),
        torch.zeros((max_active_clusters, STAMP_SLOTS), device=device, dtype=torch.int64),
        torch.zeros((1,), device=device, dtype=torch.int32),
        torch.empty((1, 1, 16), device=device, dtype=torch.uint8),
        max_active_clusters, max_tokens,
    )


def warm_ep_tiled_decode(*, max_rows=256, max_active_clusters=48,
                         token_counts=range(1, 33), reform_sf_pack=False):
    """Prepare every requested native M; no success marker or CUDA execution."""
    rows = tuple(token_counts)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("EP tiled warmup needs distinct token counts")
    for m in rows:
        get_ep_tiled_decode_kernel(num_tokens=m, max_rows=max_rows,
                                  max_active_clusters=max_active_clusters,
                                  reform_sf_pack=reform_sf_pack)
    return rows


def launch_ep_tiled_decode(*, workspace, weights, a, topk_ids, topk_weights,
                           input_gs, down_input_scale, output, scratch,
                           input_scales_are_reciprocal=False, fast_math=True):
    """Launch only the tiled EP path over pre-remapped routes, then BF16 copy.

    No dummy expert exists in weights. Root's EP remapper applies expert_map
    first; this entry accepts only its local IDs or sentinels. It never calls
    the row-major micro fallback or creates/reorders expert weight storage.
    Workspace use is serialized as in the existing model-owned MoE workspace.
    """
    import torch
    m = a.shape[0]
    ep_tiled_geometry(m, workspace.max_rows, scratch.max_active_clusters)
    if ((workspace.state_E, workspace.weight_E, workspace.k, workspace.n,
         workspace.num_topk, workspace.activation_precision, workspace.quant_mode)
            != (72, 72, 4096, 2048, 8, "fp4", "nvfp4")):
        raise ValueError("EP tiled decode workspace geometry mismatch")
    device = a.device
    if device.type != "cuda" or workspace.device != device:
        raise ValueError("EP tiled decode requires the workspace CUDA device")
    def require(t, shape, dtype, name, contiguous=True):
        if (tuple(t.shape) != shape or t.dtype != dtype or t.device != device
                or (contiguous and not t.is_contiguous())):
            raise ValueError("EP tiled decode invalid " + name)
    require(a, (m, 4096), torch.bfloat16, "input")
    require(output, (m, 4096), torch.bfloat16, "output")
    require(topk_ids, (m, 8), torch.int32, "local route IDs")
    require(topk_weights, (m, 8), torch.float32, "route weights")
    if not weights.tiled:
        raise ValueError("EP tiled decode requires tile-major weights")
    reform_sf_pack = weights.reform_scales is not None
    if reform_sf_pack:
        owner = weights.reform_scales
        if (not weights.packed_only or not owner.enabled
                or weights._w13_sf_storage is not None or weights._down_sf_storage is not None):
            raise ValueError("EP tiled SF6 requires a complete packed-only scale owner")
        require(owner.fc1, (72, 512, REFORM_SF_STAGE), torch.uint8, "SF6 FC1 plane")
        require(owner.fc2, (72, 256, REFORM_SF_STAGE), torch.uint8, "SF6 FC2 plane")
        if weights.sfb1_packed is not owner.fc1 or weights.sfb2_packed is not owner.fc2:
            raise ValueError("EP tiled SF6 scale arguments do not alias their owner")
        sfb1_packed, sfb2_packed = owner.fc1, owner.fc2
        # V5's SF6 branch never creates raw-scale descriptors. Keep the dead
        # pointer arguments backed by the same live packed planes, not raw memory.
        sfb1_address, sfb2_address = sfb1_packed.data_ptr(), sfb2_packed.data_ptr()
    else:
        if weights.packed_only:
            raise ValueError("EP tiled packed-only weights have no SF6 owner")
        if weights._w13_sf_storage is None or weights._down_sf_storage is None:
            raise ValueError("EP tiled decode raw weight scale owner is missing")
        for name, value, count in (("FC1 scales", weights._w13_sf_storage, 72*4096*256),
                                   ("FC2 scales", weights._down_sf_storage, 72*4096*128)):
            if (value.numel() != count or value.element_size() != 1
                    or value.device != device or not value.is_contiguous()):
                raise ValueError("EP tiled decode invalid raw " + name)
        sfb1_packed = sfb2_packed = scratch.dummy_scales
        sfb1_address = weights._w13_sf_storage.data_ptr()
        sfb2_address = weights._down_sf_storage.data_ptr()
    for name, shape, dtype in (
            ("row_counts", (72,), torch.int32),
            ("token_map", (72, workspace.max_rows), torch.int32),
            ("token_weights", (72, workspace.max_rows), torch.float32),
            ("packed_input", (72, workspace.max_rows, 2048), torch.uint8),
            ("packed_input_scale", (72, workspace.max_rows, 256), torch.uint8),
            ("barrier_count", (1,), torch.int32), ("barrier_epoch", (1,), torch.int32),
            ("active_expert_count", (1,), torch.int32),
            ("weight_expert_ids", (72,), torch.int32),
            ("global_to_local_expert", (72,), torch.int32)):
        require(getattr(workspace, name), shape, dtype, "workspace " + name)
    require(weights.w13_fp4, (4096, 256, 8, 72), torch.float4_e2m1fn_x2,
            "FC1 tile-major view", False)
    require(weights.down_fp4, (4096, 64, 16, 72), torch.float4_e2m1fn_x2,
            "FC2 tile-major view", False)
    if tuple(weights.w13_fp4.stride()) != (256, 1, 1048576, 8388608):
        raise ValueError("EP tiled decode FC1 tile-major strides mismatch")
    if tuple(weights.down_fp4.stride()) != (64, 1, 262144, 4194304):
        raise ValueError("EP tiled decode FC2 tile-major strides mismatch")
    for name, value in (("input scale", input_gs), ("down input scale", down_input_scale),
                        ("FC1 alpha", weights.w1_alpha), ("FC2 alpha", weights.w2_alpha)):
        require(value, (72,), torch.float32, name)
    require(scratch.scatter_fp32, (scratch.max_tokens, 4096), torch.float32, "FP32 scratch")
    if m > scratch.max_tokens:
        raise ValueError("EP tiled decode output scratch is too small")
    require(scratch.stamps, (scratch.max_active_clusters, STAMP_SLOTS), torch.int64, "stamps")
    require(scratch.counter, (1,), torch.int32, "counter")
    require(scratch.dummy_scales, (1, 1, 16), torch.uint8, "dead packed-scale argument")
    compiled, _ = get_ep_tiled_decode_kernel(
        num_tokens=m, max_rows=workspace.max_rows,
        max_active_clusters=scratch.max_active_clusters,
        topk_ids_dtype=topk_ids.dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal, fast_math=fast_math,
        reform_sf_pack=reform_sf_pack)
    accum = scratch.scatter_fp32[:m]
    accum.record_stream(torch.cuda.current_stream(device))
    compiled(
        a, topk_ids.view(-1), topk_weights.view(-1), workspace.packed_a_view,
        workspace.packed_input_scale.data_ptr(), workspace.packed_a_flat,
        workspace.scale_flat, workspace.barrier_count, workspace.barrier_epoch,
        weights.w13_fp4, sfb1_address, weights.down_fp4,
        sfb2_address, workspace.row_counts,
        workspace.active_expert_count, workspace.weight_expert_ids,
        workspace.global_to_local_expert, input_gs, weights.w1_alpha,
        weights.w2_alpha, down_input_scale, accum, workspace.token_map,
        workspace.token_weights, scratch.stamps, scratch.counter,
        sfb1_packed, sfb2_packed,
    )
    output.copy_(accum)
    return output
