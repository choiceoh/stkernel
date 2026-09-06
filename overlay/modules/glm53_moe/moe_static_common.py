# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the b12x static (decode) MoE kernels.

The stamp-slot layout and the PTX helpers (shared/global loads and stores
with the orderings the grid barriers need, the atomic CAS of the dynamic
scheduler, the compact-static work-tile map) were born in the first static
kernel (moe_static_kernel_v2, 35차) and were imported by v3 and v4 from it.
34차 §8 sunset v2 and v3 (superseded by v4 `u`, the profile default), so the
shared pieces live here and moe_static_kernel_v4 imports them from here.
"""
from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from cutlass.cutlass_dsl import (
    Int32,
    Int64,
    Uint8,
    Uint64,
    T,
    Integer,
    dsl_user_op,
)
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


_SF_VEC_SIZE = 16
_COMPACT_STATIC_TILE_M = 128
# dynamic scheduling: the DMA warp claims items from a global counter and
# hands (m_tile, slice, expert) to the MMA warps through this smem ring; the
# first FC1 stage's full barrier publishes each entry (release/acquire)
_RING = 8
# stamps tensor: [grid, STAMP_SLOTS] int64 (ns, %globaltimer)
#   0 kernel start (MMA warp 0 lane 0)   1 frontend done (compute start)
#   2 + 5*i + {0 item start, 1 FC1 done, 2 quant done, 3 FC2+scatter done,
#              4 unused}                 for items i < STAMP_ITEMS
#   2 + 5*STAMP_ITEMS      compute end   +1 item count
#   DMA_BASE + 3*i + {0 FC1 issue start, 1 FC1 issued, 2 FC2 issued}
STAMP_ITEMS = 8
STAMP_MMA_END = 2 + 5 * STAMP_ITEMS
STAMP_DMA_BASE = STAMP_MMA_END + 2
STAMP_BARRIER1 = STAMP_DMA_BASE + 3 * STAMP_ITEMS   # after grid barrier 1
STAMP_SLOTS = STAMP_BARRIER1 + 1


@cute.jit
def _compact_static_get_work_tile(
    row_counts: cute.Tensor,
    active_expert_count: cute.Tensor,
    *,
    tile_m: Int32,
    num_tiles_n: Int32,
    cluster_shape_mn: Tuple[Int32, Int32],
    current_work_linear_idx: Int32,
    current_local_expert_idx: Int32,
    accum_tile_m: Int32,
    cta_id_in_cluster: cute.Coord,
) -> Tuple[Tuple[Int32, Int32, Int32], Integer, Int32, Int32]:
    num_active_experts = active_expert_count[Int32(0)]
    scan_local_expert_idx = current_local_expert_idx
    tile_m_minus_one = tile_m - Int32(1)

    while scan_local_expert_idx < num_active_experts:
        batch_rows = row_counts[scan_local_expert_idx]
        batch_m_tiles = (batch_rows + tile_m_minus_one) // tile_m
        if (accum_tile_m + batch_m_tiles) * num_tiles_n > current_work_linear_idx:
            current_local_expert_idx = scan_local_expert_idx
            scan_local_expert_idx = num_active_experts
        else:
            accum_tile_m += batch_m_tiles
            scan_local_expert_idx += Int32(1)
            current_local_expert_idx = scan_local_expert_idx

    is_valid = current_local_expert_idx < num_active_experts
    if is_valid:
        batch_rows = row_counts[current_local_expert_idx]
        is_valid = (
            accum_tile_m + (batch_rows + tile_m_minus_one) // tile_m
        ) * num_tiles_n > current_work_linear_idx

    cur_cluster_coord = (
        current_work_linear_idx // num_tiles_n - accum_tile_m,
        current_work_linear_idx % num_tiles_n,
        current_local_expert_idx,
    )
    cur_tile_coord = (
        Int32(cur_cluster_coord[0]) * cluster_shape_mn[0] + cta_id_in_cluster[0],
        Int32(cur_cluster_coord[1]) * cluster_shape_mn[1] + cta_id_in_cluster[1],
        Int32(cur_cluster_coord[2]),
    )
    return cur_tile_coord, is_valid, current_local_expert_idx, accum_tile_m


@dsl_user_op
def _st_shared_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
@dsl_user_op
def _ld_shared_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_shared_i32_volatile(addr, *, loc=None, ip=None):
    """A shared load the compiler must re-issue every time (the claim slot and
    the ring are rewritten by another warp between reads; the plain load has
    no side effects and was hoisted out of the item loop -- 35차 §8)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_shared_f32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_global_acquire_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.s32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_global_release_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_global_i64(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _spin_wait_global_eq_i32(addr, expected, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(expected).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred %p0;\n"
        ".reg .s32 %val;\n"
        "spin_loop:\n"
        "  ld.global.acquire.gpu.s32 %val, [$0];\n"
        "  setp.eq.s32 %p0, %val, $1;\n"
        "  @%p0 bra spin_loop;\n"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _threadfence(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _atomic_cas_global_i32(addr, compare, value, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(compare).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.cas.b32 $0, [$1], $2, $3;",
            "=r,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
