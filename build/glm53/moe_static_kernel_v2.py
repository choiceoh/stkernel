"""
MoEStaticKernelV2 -- the decode-streaming rework of flashinfer's static
routed W4A4 MoE kernel for SM121 (GB10, 48 SMs).

Same contract as ``moe_static_kernel.MoEStaticKernel`` (same workspace, same
weight views, same routing frontend, same numerics: the FC1 accumulation
order per output element, the fp4 quantization of the intermediate and the
FC2 accumulation are the stock code paths), different streaming structure.

What the stock kernel does per work item (one (m_tile, intermediate_slice,
expert) triple) is three producer/consumer passes with a drained pipeline
and a CTA-wide barrier between them:

    gate pass  (A + B_gate, 2 stages)  -> pass_sync barrier
    up pass    (A + B_up,   2 stages)  -> activation + quant
    FC2 sweep  (B_down, 2 stages in the gate buffers) -> pass_sync barrier
    next item

At the served decode shape (8 tokens x top-8 -> ~40 experts x 4 slices = 160
items over 48 resident CTAs, ~0.9 MB of weights per item) that is DRAM-bound
with only 18 KB of weight bytes in flight per SM, restarted three times per
item, and the whole item's 128-row A tile (8 KB per k-tile, of which <= 8
rows are real tokens) is re-read from L2 twice.

This kernel keeps ONE continuously advancing pipeline per operand class:

    FC1: one stage = A + B_gate + B_up + SFA + SFB_gate + SFB_up on one
         mbarrier (fc1_stages deep); the MMA warps accumulate gate and up in
         the same k loop
    FC2: its own stage buffers (fc2_stages deep), so the DMA warp issues
         the item's 32 down tiles as soon as the FC1 loads are out, and the
         next item's FC1 loads right after -- there is no barrier at the
         item boundary; the quantized intermediate lives in its own buffer
         (sA2/sSFA2) instead of FC1's stage 0

Pipeline states are never reset between items (index/phase advance
monotonically), so any stage count works; producer_tail drains at exit.

The A TMA box is ``a_rows`` rows (default = tile_m) instead of the stock
128, and tile_m = 32 is admitted (2 warps x m16 along M, 2 along N), which
is what keeps gate_acc + up_acc + down_acc (3 x 32 regs) plus two B
fragment sets inside the 232-register MMA-warp budget.

Optional ``stamps``: ``%globaltimer`` at kernel start, after the routing
frontend, per item (start / FC1 done / quant done / FC2+scatter done) from
MMA warp 0, and per item (FC1 issued / FC2 issued) from the DMA warp, into
an int64 [grid, STAMP_SLOTS] tensor -- the per-CTA timeline the probe reads.

Selected by ``moe_dispatch`` (``VLLM_GLM53_B12X_STATIC_V2``) for the exact
GLM-5.3 TP geometry only; default off.
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
# stamps tensor: [grid, STAMP_SLOTS] int64 (ns, %globaltimer)
#   0 kernel start (MMA warp 0 lane 0)   1 frontend done (compute start)
#   2 + 5*i + {0 item start, 1 FC1 done, 2 quant done, 3 FC2+scatter done,
#              4 unused}                 for items i < STAMP_ITEMS
#   2 + 5*STAMP_ITEMS      compute end   +1 item count
#   DMA_BASE + 3*i + {0 FC1 issue start, 1 FC1 issued, 2 FC2 issued}
STAMP_ITEMS = 8
STAMP_MMA_END = 2 + 5 * STAMP_ITEMS
STAMP_DMA_BASE = STAMP_MMA_END + 2
STAMP_SLOTS = STAMP_DMA_BASE + 3 * STAMP_ITEMS


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


class MoEStaticKernelV2:
    """Decode-streaming static MoE kernel (gated NVFP4 only)."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        fc1_stages: int = 2,
        fc2_stages: int = 4,
        a_rows: int | None = None,
        stamps: bool = False,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
        swiglu_alpha: float = 1.702,
        swiglu_beta: float = 1.0,
        swiglu_limit: float | None = None,
    ):
        if activation not in {"silu", "gelu_tanh", "swigluoai_uninterleave"}:
            raise ValueError(f"unsupported activation {activation!r} (v2 is gated only)")
        if sf_vec_size != _SF_VEC_SIZE:
            raise ValueError("v2 is the NVFP4 (sf_vec_size=16) lane")
        tile_m, tile_n = int(mma_tiler_mn[0]), int(mma_tiler_mn[1])
        if tile_n != 128 or tile_m not in (32, 64, 128):
            raise ValueError(f"v2 tile must be (32|64|128, 128), got {mma_tiler_mn}")
        a_rows = tile_m if a_rows is None else int(a_rows)
        if a_rows % tile_m != 0 or a_rows > 128 or a_rows < tile_m:
            raise ValueError(f"a_rows {a_rows} must be a multiple of tile_m {tile_m}, <= 128")
        if int(fc1_stages) < 1 or int(fc2_stages) < 1:
            raise ValueError("pipeline stages must be >= 1")
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.activation = activation
        self.is_gated = is_gated_activation(activation)
        assert self.is_gated
        self.fast_math = bool(fast_math)
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.swiglu_limit = float(swiglu_limit) if swiglu_limit is not None else None
        self.fc1_stages = int(fc1_stages)
        self.fc2_stages = int(fc2_stages)
        self.stamps = bool(stamps)
        tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (tile_m, tile_n, tile_k)
        self.sa_tile_shape_mk = (a_rows, tile_k)
        self.sa_tiles_per_block = a_rows // tile_m
        # Scale-factor tiles are laid out in 128-row blocks; the SF boxes
        # stay 128 rows (1 KB per k-tile) and the kernel consumes the live
        # subset, exactly like the stock kernel.
        self.sfa_tile_shape_mk = (128, tile_k)
        self.sfa_tiles_per_block = 128 // tile_m
        self.sfb_tile_shape_nk = (max(128, tile_n), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // tile_n
        self.output_tile_count_n = output_tile_count_n
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi_tile = (tile_m, tile_n)
        self.occupancy = 1
        self.num_mma_warps = 4
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.buffer_align_bytes = 1024
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 32
        self.mma_register_requirement = 232
        self.smem_bytes = 0

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFA(self, sfa_tensor, tiled_mma)

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFB(self, sfb_tensor, tiled_mma)

    def _get_layoutSFA_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFA_TV(self, tiled_mma)  # type: ignore[arg-type]

    def _get_layoutSFB_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFB_TV(self, tiled_mma)  # type: ignore[arg-type]

    def _make_a_smem_layout(self, rows: int, stages: int):
        import cutlass.utils.hopper_helpers as sm90_utils

        a_is_k_major = self.a_layout.is_k_major_a()
        tile = (rows, self.tile_shape_mnk[2])
        a_major_mode_size = tile[1 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.a_layout,
                self.a_dtype,
                a_major_mode_size,
            ),
            self.a_dtype,
        )
        return cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(tile, stages),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

    def _staged_b_layouts(self, stages: int):
        (
            _,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
            epi_smem_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            stages,
            cutlass.BFloat16,
            self.c_layout,
            1,
            self.sf_vec_size,
            self.tiled_mma,
        )
        return b_smem_staged, sfa_smem_staged, sfb_smem_staged, epi_smem_staged

    def _smem_bytes_estimate(self) -> int:
        def _align_up(value: int, align: int) -> int:
            return ((value + align - 1) // align) * align

        offset = (
            2 * 4
            + (self.fc1_stages + self.fc2_stages) * 2 * 8
            + _COMPACT_STATIC_TILE_M * 4
            + _COMPACT_STATIC_TILE_M * 4
        )
        buffers = [
            cute.size_in_bytes(self.a_dtype, self.a_smem_layout_staged),
            cute.size_in_bytes(self.b_dtype, self.b_smem_layout_staged),
            cute.size_in_bytes(self.b_dtype, self.b_smem_layout_staged),
            cute.size_in_bytes(self.sf_dtype, self.sfa_smem_layout_staged),
            cute.size_in_bytes(self.sf_dtype, self.sfb_smem_layout_staged),
            cute.size_in_bytes(self.sf_dtype, self.sfb_smem_layout_staged),
            cute.size_in_bytes(self.b_dtype, self.b2_smem_layout_staged),
            cute.size_in_bytes(self.sf_dtype, self.sfb2_smem_layout_staged),
            cute.size_in_bytes(self.a_dtype, self.a2_smem_layout),
            cute.size_in_bytes(self.sf_dtype, self.sfa2_smem_layout),
            cute.size_in_bytes(cutlass.BFloat16, self.epi_smem_layout_staged),
        ]
        for size in buffers:
            offset = _align_up(offset, self.buffer_align_bytes) + size
        return offset

    def _setup_attributes(self, hidden_size: int):
        import cutlass.utils.blackwell_helpers as sm120_utils

        self._hidden_size = hidden_size
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_shape = (2, 2, 1)
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk,
            self.sf_vec_size,
            False,
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_m_tiles = self.tile_shape_mnk[0] // (16 * atom_shape[0])
        self.num_n_tiles = self.tile_shape_mnk[1] // (8 * atom_shape[1])
        self.num_k_blocks = self.tile_shape_mnk[2] // 64

        self.epi_stage = 1
        self.a_smem_layout_staged = self._make_a_smem_layout(
            self.sa_tile_shape_mk[0], self.fc1_stages
        )
        (
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._staged_b_layouts(self.fc1_stages)
        (
            self.b2_smem_layout_staged,
            _,
            self.sfb2_smem_layout_staged,
            _,
        ) = self._staged_b_layouts(self.fc2_stages)
        self.a2_smem_layout = self._make_a_smem_layout(self.sa_tile_shape_mk[0], 1)
        self.sfa2_smem_layout = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        self.smem_bytes = self._smem_bytes_estimate()
        if self.smem_bytes > self.smem_capacity:
            raise ValueError(
                f"v2 smem {self.smem_bytes} B exceeds {self.smem_capacity} B "
                f"(tile_m {self.tile_shape_mnk[0]}, a_rows {self.sa_tile_shape_mk[0]}, "
                f"fc1 {self.fc1_stages} x fc2 {self.fc2_stages} stages)"
            )

    @cute.jit
    def _resident_grid_barrier(
        self,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        grid_x: Int32,
        is_cta_leader: Int32,
    ):
        cute.arch.sync_threads()
        _threadfence()
        if is_cta_leader > Int32(0):
            barrier_count_addr = get_ptr_as_int64(barrier_count, Int32(0))
            barrier_epoch_addr = get_ptr_as_int64(barrier_epoch, Int32(0))
            old_epoch = _ld_global_acquire_i32(barrier_epoch_addr)
            arrived = atomic_add_global_i32(barrier_count_addr, Int32(1))
            if arrived == grid_x - Int32(1):
                st_global_i32(barrier_count_addr, Int32(0))
                _st_global_release_i32(barrier_epoch_addr, old_epoch + Int32(1))
            else:
                _spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)
        cute.arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,  # [num_tokens, K] bf16
        topk_ids: cute.Tensor,  # [num_tokens * topk] int32
        topk_weights: cute.Tensor,  # [num_tokens * topk] float32
        packed_a: cute.Tensor,  # [max_rows, K, E] fp4x2 view for compute
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,  # flat uint8 backing packed_a
        scale_storage: cute.Tensor,  # flat uint8 backing sfa_ptr
        barrier_count: cute.Tensor,  # [1] int32 (host-zeroed)
        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)
        b_w13: cute.Tensor,  # [2*I_tp, K, E] -- concatenated up+gate
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,  # [K, I_tp, E]
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,  # [state_E]
        active_expert_count: cute.Tensor,  # [1]
        weight_expert_ids: cute.Tensor,  # [E]
        global_to_local_expert: cute.Tensor,  # [weight_E]
        input_global_scale: cute.Tensor,  # [E]
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,  # [num_tokens, K]
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        stamps: cute.Tensor,  # [grid, STAMP_SLOTS] int64
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = packed_a.element_type
        self.b_dtype = b_w13.element_type
        self.sf_dtype = sfa_ptr.dtype
        self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
        self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        hidden_size = a_input.shape[1]
        self._setup_attributes(hidden_size=hidden_size)

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            packed_a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_w13_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_w13.shape, self.sf_vec_size
        )
        sfb_w13_tensor = cute.make_tensor(sfb_w13_ptr, sfb_w13_layout)

        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a,
            self.a_smem_layout_staged,
            self.sa_tile_shape_mk,
            1,
        )
        tma_sfa, gSFA = self._dense_cls._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            self.sfa_tile_shape_mk,
            1,
            internal_type=cutlass.Int16,
        )
        tma_b_w13, gB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            b_w13,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_w13_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )
        sfb_down_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_down.shape, self.sf_vec_size
        )
        sfb_down_tensor = cute.make_tensor(sfb_down_ptr, sfb_down_layout)
        tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
            b_down,
            self.b2_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_sfb_down, gSFB_down = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_down_tensor,
            self.sfb2_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )

        grid = (*self.cluster_shape_mn, max_active_clusters)
        self.kernel(
            a_input,
            topk_ids,
            topk_weights,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            tma_a,
            gA,
            tma_sfa,
            gSFA,
            tma_b_w13,
            gB_w13,
            tma_sfb_w13,
            gSFB_w13,
            tma_b_down,
            gB_down,
            tma_sfb_down,
            gSFB_down,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.b2_smem_layout_staged,
            self.sfb2_smem_layout_staged,
            self.a2_smem_layout,
            self.sfa2_smem_layout,
            self.epi_smem_layout_staged,
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
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            cooperative=True,
            stream=stream,
        )

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
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
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
            cpasync.prefetch_descriptor(tma_sfb_w13)
            cpasync.prefetch_descriptor(tma_b_down)
            cpasync.prefetch_descriptor(tma_sfb_down)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        a_smem_one = cute.slice_(a_smem_staged, (None, None, 0))
        b_smem_one = cute.slice_(b_smem_staged, (None, None, 0))
        sfa_smem_one = cute.slice_(sfa_smem_staged, (None, None, 0))
        sfb_smem_one = cute.slice_(sfb_smem_staged, (None, None, 0))
        fc1_tma_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_one)
            + 2 * cute.size_in_bytes(self.b_dtype, b_smem_one)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_one)
            + 2 * cute.size_in_bytes(self.sf_dtype, sfb_smem_one)
        )
        b2_smem_one = cute.slice_(b2_smem_staged, (None, None, 0))
        sfb2_smem_one = cute.slice_(sfb2_smem_staged, (None, None, 0))
        fc2_tma_bytes = cute.size_in_bytes(
            self.b_dtype, b2_smem_one
        ) + cute.size_in_bytes(self.sf_dtype, sfb2_smem_one)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            ctrl: cute.struct.MemRange[cutlass.Int32, 2]
            fc1_bars: cute.struct.MemRange[cutlass.Int64, self.fc1_stages * 2]
            fc2_bars: cute.struct.MemRange[cutlass.Int64, self.fc2_stages * 2]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, _COMPACT_STATIC_TILE_M
            ]
            scatter_weight_cache: cute.struct.MemRange[
                cutlass.Float32, _COMPACT_STATIC_TILE_M
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sBg: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sBu: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFBg: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFBu: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
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

        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sBg = storage.sBg.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sBu = storage.sBu.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sB2 = storage.sB2.get_tensor(b2_smem_staged.outer, swizzle=b2_smem_staged.inner)
        sA2 = storage.sA2.get_tensor(a2_smem_layout.outer, swizzle=a2_smem_layout.inner)
        cute.recast_tensor(sA, cutlass.Uint8)
        cute.recast_tensor(sBg, cutlass.Uint8)
        cute.recast_tensor(sBu, cutlass.Uint8)
        cute.recast_tensor(sB2, cutlass.Uint8)
        cute.recast_tensor(sA2, cutlass.Uint8)
        sSFA = storage.sSFA.get_tensor(sfa_smem_staged)
        sSFBg = storage.sSFBg.get_tensor(sfb_smem_staged)
        sSFBu = storage.sSFBu.get_tensor(sfb_smem_staged)
        sSFB2 = storage.sSFB2.get_tensor(sfb2_smem_staged)
        sSFA2 = storage.sSFA2.get_tensor(sfa2_smem_layout)
        cute.recast_tensor(sSFA, cutlass.Uint8)
        cute.recast_tensor(sSFBg, cutlass.Uint8)
        cute.recast_tensor(sSFBu, cutlass.Uint8)
        cute.recast_tensor(sSFB2, cutlass.Uint8)
        cute.recast_tensor(sSFA2, cutlass.Uint8)
        sC = storage.sC.get_tensor(
            epi_smem_staged.outer,
            swizzle=epi_smem_staged.inner,
        )
        sfa2_base_addr = shared_ptr_to_u32(storage.sSFA2.data_ptr())
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
        # Phase 0: cooperative init (stock)
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
        scatter_total = num_tokens * cols
        j = flat_tid
        while j < scatter_total:
            scatter_output[j // cols, j % cols] = cutlass.BFloat16(0.0)
            j += flat_stride
        cute.arch.sync_threads()
        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # ------------------------------------------------------------------
        # Phase 1: route + pack (stock)
        # ------------------------------------------------------------------
        pair_idx = Int32(bidz)
        while pair_idx < total_pairs:
            expert_id = topk_ids[pair_idx].to(Int32)
            token_idx = pair_idx // num_topk
            weight = topk_weights[pair_idx].to(cutlass.Float32)
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
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )
        if cutlass.const_expr(self.stamps):
            if Int32(tidx) == Int32(0):
                _st_global_i64(
                    get_ptr_as_int64(stamps, stamp_row + Int32(1)),
                    cute.arch.globaltimer(),
                )

        # ------------------------------------------------------------------
        # Tiled views and TMA partitions
        # ------------------------------------------------------------------
        gA = cute.local_tile(mA, self.sa_tile_shape_mk, (None, None, None))
        gB_w13_tiled = cute.local_tile(
            mB_w13,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA = cute.local_tile(mSFA, self.sfa_tile_shape_mk, (None, None, None))
        gSFB_w13_tiled = cute.local_tile(
            mSFB_w13, self.sfb_tile_shape_nk, (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord[1]
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord[0]

        tAsA, tAgA = cpasync.tma_partition(
            tma_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        tBsBg, tBgB_w13 = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sBg, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsBu, _ = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sBu, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsSFBg, tBgSFB_w13 = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFBg, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFBu, _ = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFBu, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsBu = cute.filter_zeros(tBsBu)
        tBsSFBg = cute.filter_zeros(tBsSFBg)
        tBgSFB_w13 = cute.filter_zeros(tBgSFB_w13)
        tBsSFBu = cute.filter_zeros(tBsSFBu)

        gB_down = cute.local_tile(
            mB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFB_down = cute.local_tile(
            mSFB_down, self.sfb_tile_shape_nk, (None, None, None)
        )
        tBsB2, tBgB_down = cpasync.tma_partition(
            tma_b_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB2, 0, 2),
            cute.group_modes(gB_down, 0, 2),
        )
        tBsSFB2, tBgSFB_down = cpasync.tma_partition(
            tma_sfb_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB2, 0, 2),
            cute.group_modes(gSFB_down, 0, 2),
        )
        tBsSFB2 = cute.filter_zeros(tBsSFB2)
        tBgSFB_down = cute.filter_zeros(tBgSFB_down)

        # MMA fragment partitions (FC1 operands from the stage buffers, FC2
        # A operand from the dedicated quantized-intermediate buffers)
        tCsA_full = thr_mma.partition_A(sA)
        tCrA_full = tiled_mma.make_fragment_A(tCsA_full[None, None, None, 0])
        tCrSFA_full = self._dense_cls._partition_fragment_SFA(
            self,  # type: ignore[arg-type]
            sSFA[None, None, 0],
            thr_mma,
            tidx,
        )
        tCsBg = thr_mma.partition_B(sBg)
        tCrBg = tiled_mma.make_fragment_B(tCsBg[None, None, None, 0])
        tCsBu = thr_mma.partition_B(sBu)
        tCrBu = tiled_mma.make_fragment_B(tCsBu[None, None, None, 0])
        tCrSFBg = self._dense_cls._partition_fragment_SFB(
            self,  # type: ignore[arg-type]
            sSFBg[None, None, 0],
            thr_mma,
            tidx,
        )
        tCrSFBu = self._dense_cls._partition_fragment_SFB(
            self,  # type: ignore[arg-type]
            sSFBu[None, None, 0],
            thr_mma,
            tidx,
        )
        tCsA2 = thr_mma.partition_A(sA2)
        tCrA2 = tiled_mma.make_fragment_A(tCsA2[None, None, None, 0])
        tCrSFA2 = self._dense_cls._partition_fragment_SFA(
            self,  # type: ignore[arg-type]
            sSFA2[None, None, 0],
            thr_mma,
            tidx,
        )
        tCsB2 = thr_mma.partition_B(sB2)
        tCrB2 = tiled_mma.make_fragment_B(tCsB2[None, None, None, 0])
        tCrSFB2 = self._dense_cls._partition_fragment_SFB(
            self,  # type: ignore[arg-type]
            sSFB2[None, None, 0],
            thr_mma,
            tidx,
        )

        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        sub_shape = tCsC_for_shape.shape[:3]
        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        gate_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA, mode=[3])
        fc1_k_tile_cnt = k_tile_cnt
        intermediate_tile_cnt = cute.size(gB_w13_tiled, mode=[2])
        gate_tile_cnt = intermediate_tile_cnt // Int32(2)
        output_tile_cnt = cute.size(gB_down, mode=[2])

        fc1_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.fc1_stages
        )
        fc1_cons_state = pipeline.make_pipeline_state(
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
            num_k_blocks = cute.size(tCrA_full, mode=[2])

            atom_ld_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_ld_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
            smem_copy_B = cute.make_tiled_copy_B(atom_ld_B, tiled_mma)
            atom_ld_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.sf_dtype
            )
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

            thr_ld_A = smem_copy_A.get_slice(tidx)
            thr_ld_B = smem_copy_B.get_slice(tidx)
            thr_ld_SFA = smem_copy_SFA.get_slice(tidx)
            thr_ld_SFB = smem_copy_SFB.get_slice(tidx)

            csA_full = thr_ld_A.partition_S(sA)
            crA_full = thr_ld_A.retile(tCrA_full)
            csBg = thr_ld_B.partition_S(sBg)
            crBg = thr_ld_B.retile(tCrBg)
            csBu = thr_ld_B.partition_S(sBu)
            crBu = thr_ld_B.retile(tCrBu)
            csSFA_full = thr_ld_SFA.partition_S(sSFA)
            crSFA_full = thr_ld_SFA.retile(tCrSFA_full)
            csSFBg_full = thr_ld_SFB.partition_S(sSFBg)
            crSFBg_full = thr_ld_SFB.retile(tCrSFBg)
            csSFBu_full = thr_ld_SFB.partition_S(sSFBu)
            crSFBu_full = thr_ld_SFB.retile(tCrSFBu)
            csA2 = thr_ld_A.partition_S(sA2)
            crA2 = thr_ld_A.retile(tCrA2)
            csSFA2 = thr_ld_SFA.partition_S(sSFA2)
            crSFA2 = thr_ld_SFA.retile(tCrSFA2)
            csB2 = thr_ld_B.partition_S(sB2)
            crB2 = thr_ld_B.retile(tCrB2)
            csSFB2_full = thr_ld_SFB.partition_S(sSFB2)
            crSFB2_full = thr_ld_SFB.retile(tCrSFB2)

            num_persistent_clusters = Int32(gdim_z)
            cluster_shape_mn = (
                Int32(self.cluster_shape_mn[0]),
                Int32(self.cluster_shape_mn[1]),
            )
            cta_id_in_cluster = (
                Int32(bidx % cluster_shape_mn[0]),
                Int32(bidy % cluster_shape_mn[1]),
                Int32(0),
            )
            current_work_linear_idx = Int32(bidz)
            current_local_expert_idx = Int32(0)
            accum_tile_m = Int32(0)
            item_no = Int32(0)
            tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                _compact_static_get_work_tile(
                    row_counts,
                    active_expert_count,
                    tile_m=Int32(self.tile_shape_mnk[0]),
                    num_tiles_n=Int32(self.output_tile_count_n),
                    cluster_shape_mn=cluster_shape_mn,
                    current_work_linear_idx=current_work_linear_idx,
                    current_local_expert_idx=current_local_expert_idx,
                    accum_tile_m=accum_tile_m,
                    cta_id_in_cluster=cta_id_in_cluster,
                )
            )

            _is_m_major = self.c_layout.is_m_major_c()
            copy_atom_r2s = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                cutlass.BFloat16,
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rGate = tiled_copy_r2s.retile(gate_acc)
            tRS_rUp = tiled_copy_r2s.retile(up_acc)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)
            mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
            mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
            down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
            MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
            MmaNPerEpiN = self.epi_tile[1] // mma_tile_n

            fz_crSFA = cute.filter_zeros(crSFA_full)
            fz_crSFBg = cute.filter_zeros(crSFBg_full)
            fz_crSFBu = cute.filter_zeros(crSFBu_full)
            fz_crSFA2 = cute.filter_zeros(crSFA2)
            fz_crSFB2 = cute.filter_zeros(crSFB2_full)

            scatter_N = Int32(scatter_output.shape[1])
            lane_id = Int32(tidx) & Int32(31)
            warp_in_tile = Int32(tidx) >> Int32(5)
            warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)
            warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)

            packed_cols = Int32(self.tile_shape_mnk[2] // 2)
            sf_blocks_per_row_tile = Int32(self.tile_shape_mnk[2] // self.sf_vec_size)
            a2_rows = Int32(self.sa_tile_shape_mk[0])
            sA2_u8 = cute.recast_tensor(sA2[None, None, 0], cutlass.Uint8)

            while is_valid_tile:
                # tile_coord = (m_tile, intermediate_slice, local_expert_idx)
                local_expert_idx = tile_coord[2]
                weight_expert_idx = weight_expert_ids[local_expert_idx]
                alpha_value = alpha[weight_expert_idx].to(cutlass.Float32)
                valid_rows = row_counts[local_expert_idx]
                tile_m_base = tile_coord[0] * Int32(self.tile_shape_mnk[0])
                stamp_item = stamp_row + Int32(2) + item_no * Int32(5)
                if cutlass.const_expr(self.stamps):
                    if Int32(tidx) == Int32(0):
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_item),
                                cute.arch.globaltimer(),
                            )

                # a_rows may hold sa_tiles_per_block m-tiles; pick this
                # m-tile's sub-tile of the A stage (stock logic)
                sa_tile_offset = tile_coord[0] % self.sa_tiles_per_block
                sa_row_base = sa_tile_offset * Int32(self.tile_shape_mnk[0])
                if cutlass.const_expr(self.sa_tiles_per_block > 1):
                    sA_tile = cute.local_tile(
                        sA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (sa_tile_offset, 0, None),
                    )
                    csA_tile = thr_ld_A.partition_S(sA_tile)
                    tCsA_tile = thr_mma.partition_A(sA_tile)
                    tCrA_tile = tiled_mma.make_fragment_A(
                        tCsA_tile[None, None, None, 0]
                    )
                    crA_tile = thr_ld_A.retile(tCrA_tile)
                else:
                    csA_tile = csA_full
                    tCrA_tile = tCrA_full
                    crA_tile = crA_full
                sfa_tile_offset = tile_coord[0] % self.sfa_tiles_per_block
                if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                    sSFA_tile = cute.local_tile(
                        sSFA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (sfa_tile_offset, 0, None),
                    )
                    csSFA_tile = thr_ld_SFA.partition_S(sSFA_tile)
                    tCrSFA_tile = self._dense_cls._partition_fragment_SFA(
                        self,  # type: ignore[arg-type]
                        sSFA_tile[None, None, 0],
                        thr_mma,
                        tidx,
                    )
                    crSFA_tile = thr_ld_SFA.retile(tCrSFA_tile)
                    fz_crSFA_tile = cute.filter_zeros(crSFA_tile)
                else:
                    csSFA_tile = csSFA_full
                    tCrSFA_tile = tCrSFA_full
                    fz_crSFA_tile = fz_crSFA
                valid_tile_rows = valid_rows - tile_m_base
                if valid_tile_rows > Int32(self.tile_shape_mnk[0]):
                    valid_tile_rows = Int32(self.tile_shape_mnk[0])
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
                # (the scatter caches are read only after the FC1 epilog
                # barriers below; no barrier needed here)

                down_alpha_value = down_alpha[weight_expert_idx].to(cutlass.Float32)

                # ============================================================
                # PHASE A: FC1 gate + up in one k loop over the fc1 pipeline
                # ============================================================
                gate_acc.fill(0.0)
                up_acc.fill(0.0)
                peek = fc1_pipeline.consumer_try_wait(fc1_cons_state)
                fc1_pipeline.consumer_wait(fc1_cons_state, peek)
                csA_p = csA_tile[None, None, None, fc1_cons_state.index]
                csBg_p = csBg[None, None, None, fc1_cons_state.index]
                csBu_p = csBu[None, None, None, fc1_cons_state.index]
                fz_csSFA_p = cute.filter_zeros(
                    csSFA_tile[None, None, None, fc1_cons_state.index]
                )
                fz_csSFBg_p = cute.filter_zeros(
                    csSFBg_full[None, None, None, fc1_cons_state.index]
                )
                fz_csSFBu_p = cute.filter_zeros(
                    csSFBu_full[None, None, None, fc1_cons_state.index]
                )
                cute.copy(smem_copy_A, csA_p[None, None, 0], crA_tile[None, None, 0])
                cute.copy(smem_copy_B, csBg_p[None, None, 0], crBg[None, None, 0])
                cute.copy(smem_copy_B, csBu_p[None, None, 0], crBu[None, None, 0])
                cute.copy(
                    smem_copy_SFA, fz_csSFA_p[None, None, 0], fz_crSFA_tile[None, None, 0]
                )
                cute.copy(
                    smem_copy_SFB, fz_csSFBg_p[None, None, 0], fz_crSFBg[None, None, 0]
                )
                cute.copy(
                    smem_copy_SFB, fz_csSFBu_p[None, None, 0], fz_crSFBu[None, None, 0]
                )
                for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):  # type: ignore[call-overload]
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )
                        if k_block_idx == num_k_blocks - 1:
                            fc1_pipeline.consumer_release(fc1_cons_state)
                            fc1_cons_state.advance()
                            peek = fc1_pipeline.consumer_try_wait(fc1_cons_state)
                            csA_p = csA_tile[None, None, None, fc1_cons_state.index]
                            csBg_p = csBg[None, None, None, fc1_cons_state.index]
                            csBu_p = csBu[None, None, None, fc1_cons_state.index]
                            fz_csSFA_p = cute.filter_zeros(
                                csSFA_tile[None, None, None, fc1_cons_state.index]
                            )
                            fz_csSFBg_p = cute.filter_zeros(
                                csSFBg_full[None, None, None, fc1_cons_state.index]
                            )
                            fz_csSFBu_p = cute.filter_zeros(
                                csSFBu_full[None, None, None, fc1_cons_state.index]
                            )
                            fc1_pipeline.consumer_wait(fc1_cons_state, peek)
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFBg[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    gate_acc[None, _mt, _nt],
                                    tCrA_tile[None, _mt, k_block_idx],
                                    tCrBg[None, _nt, k_block_idx],
                                    gate_acc[None, _mt, _nt],
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFBu[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    up_acc[None, _mt, _nt],
                                    tCrA_tile[None, _mt, k_block_idx],
                                    tCrBu[None, _nt, k_block_idx],
                                    up_acc[None, _mt, _nt],
                                )
                        cute.copy(
                            smem_copy_A,
                            csA_p[None, None, k_next],
                            crA_tile[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B,
                            csBg_p[None, None, k_next],
                            crBg[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B,
                            csBu_p[None, None, k_next],
                            crBu[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFA,
                            fz_csSFA_p[None, None, k_next],
                            fz_crSFA_tile[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB,
                            fz_csSFBg_p[None, None, k_next],
                            fz_crSFBg[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB,
                            fz_csSFBu_p[None, None, k_next],
                            fz_crSFBu[None, None, k_next],
                        )
                # last k-tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    if k_block_idx == num_k_blocks - 1:
                        fc1_pipeline.consumer_release(fc1_cons_state)
                        fc1_cons_state.advance()
                    if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                        cute.copy(
                            smem_copy_A,
                            csA_p[None, None, k_next],
                            crA_tile[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B,
                            csBg_p[None, None, k_next],
                            crBg[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B,
                            csBu_p[None, None, k_next],
                            crBu[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFA,
                            fz_csSFA_p[None, None, k_next],
                            fz_crSFA_tile[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB,
                            fz_csSFBg_p[None, None, k_next],
                            fz_crSFBg[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_SFB,
                            fz_csSFBu_p[None, None, k_next],
                            fz_crSFBu[None, None, k_next],
                        )
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_tile[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFBg[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                gate_acc[None, _mt, _nt],
                                tCrA_tile[None, _mt, k_block_idx],
                                tCrBg[None, _nt, k_block_idx],
                                gate_acc[None, _mt, _nt],
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFBu[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                up_acc[None, _mt, _nt],
                                tCrA_tile[None, _mt, k_block_idx],
                                tCrBu[None, _nt, k_block_idx],
                                up_acc[None, _mt, _nt],
                            )
                if cutlass.const_expr(self.stamps):
                    if Int32(tidx) == Int32(0):
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_item + Int32(1)),
                                cute.arch.globaltimer(),
                            )

                # ============================================================
                # Activation + quant into sA2 / sSFA2 (stock arithmetic)
                # ============================================================
                gs_value = global_scale[weight_expert_idx].to(cutlass.Float32)
                if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                    0.0
                ):
                    if self.fast_math:
                        gs_value = rcp_approx_ftz(gs_value)
                    else:
                        gs_value = cutlass.Float32(1.0) / gs_value

                # the scatter caches written above must be visible to every
                # MMA warp before the FC2 scatter reads them; the barrier
                # below (before the sC reads) provides that
                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    epi_m_valid = (
                        valid_rows
                        - tile_m_base
                        - Int32(epi_m) * Int32(self.epi_tile[0])
                    )
                    silu_epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                    if epi_m_valid > Int32(0):
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                mma_n = mma_n_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                up_slice = tRS_rUp[(None, mma_m, mma_n)]
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
                        acc_vec = tRS_rD.load()
                        acc_vec = acc_vec.to(cutlass.BFloat16)
                        tRS_rD_out.store(acc_vec)
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, silu_epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                    epi_rows = epi_m_valid
                    if epi_rows > Int32(self.epi_tile[0]):
                        epi_rows = Int32(self.epi_tile[0])
                    if epi_rows < Int32(0):
                        epi_rows = Int32(0)
                    quant_idx = Int32(tidx)
                    while quant_idx < epi_rows * sf_blocks_per_row_tile:
                        local_row = quant_idx // sf_blocks_per_row_tile
                        row = sa_row_base + rows_offset + local_row
                        sf_block = quant_idx - local_row * sf_blocks_per_row_tile
                        block_start = sf_block * Int32(self.sf_vec_size)

                        values = cute.make_rmem_tensor(
                            (self.sf_vec_size,), cutlass.Float32
                        )
                        block_max = cutlass.Float32(0.0)
                        for elem_idx in cutlass.range_constexpr(self.sf_vec_size):
                            value = cutlass.Float32(
                                sC[local_row, block_start + elem_idx, silu_epi_buffer]
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
                        # K-major SW64 A tile: byte (row, kb) sits at
                        # row * 64 + (kb ^ (((row >> 1) & 3) << 4)); the recast
                        # tensor's flat index is colexicographic over
                        # (a2_rows, packed_cols)
                        xor_bits = ((row >> Int32(1)) & Int32(0x3)) << Int32(4)
                        for byte_idx in cutlass.range_constexpr(self.sf_vec_size // 2):
                            src_pcol = packed_base + Int32(byte_idx)
                            dst_flat = (src_pcol ^ xor_bits) * a2_rows + row
                            byte_val = Uint8(
                                (packed_lo >> Uint64(byte_idx * 8)) & Uint64(0xFF)
                            )
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
                # PHASE B: FC2 sweep over the fc2 pipeline, A from sA2/sSFA2
                # ============================================================
                csA2_p = csA2[None, None, None, 0]
                fz_csSFA2_p = cute.filter_zeros(csSFA2[None, None, None, 0])
                for _kb in cutlass.range_constexpr(num_k_blocks):
                    cute.copy(smem_copy_A, csA2_p[None, None, _kb], crA2[None, None, _kb])
                    cute.copy(
                        smem_copy_SFA,
                        fz_csSFA2_p[None, None, _kb],
                        fz_crSFA2[None, None, _kb],
                    )

                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    fc2_peek = fc2_pipeline.consumer_try_wait(fc2_cons_state)
                    fc2_pipeline.consumer_wait(fc2_cons_state, fc2_peek)
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
                                smem_copy_B,
                                csB2_p[None, None, k_next],
                                crB2[None, None, k_next],
                            )
                            cute.copy(
                                smem_copy_SFB,
                                fz_csSFB2_p[None, None, k_next],
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

                    tile_n_base_cur = output_tile_idx * Int32(self.tile_shape_mnk[1])
                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_n = mma_n_in_epi
                                mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                down_epi_acc_slice = down_acc[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    tRS_rD_slice[elem_idx] = (
                                        down_alpha_value * down_epi_acc_slice[elem_idx]
                                    )

                        acc_vec = tRS_rD.load()
                        acc_vec = acc_vec.to(cutlass.BFloat16)
                        tRS_rD_out.store(acc_vec)
                        epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                        warp_epi_rows = (
                            valid_rows - tile_m_base - rows_offset - warp_m_base
                        )
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
                            cached_row = rows_offset + warp_m_base + local_row
                            tok = ld_shared_i32_relaxed(
                                scatter_tok_base_addr + cached_row * Int32(4)
                            )
                            wv = _ld_shared_f32(
                                scatter_weight_base_addr + cached_row * Int32(4)
                            )
                            sc_v0 = cutlass.Float32(
                                sC[warp_m_base + local_row, local_col, epi_buffer]
                            )
                            sc_v1 = cutlass.Float32(
                                sC[warp_m_base + local_row, local_col + Int32(1), epi_buffer]
                            )
                            sc_v2 = cutlass.Float32(
                                sC[warp_m_base + local_row, local_col + Int32(2), epi_buffer]
                            )
                            sc_v3 = cutlass.Float32(
                                sC[warp_m_base + local_row, local_col + Int32(3), epi_buffer]
                            )
                            sc_v4 = cutlass.Float32(
                                sC[warp_m_base + local_row, local_col + Int32(4), epi_buffer]
                            )
                            sc_v5 = cutlass.Float32(
                                sC[warp_m_base + local_row, local_col + Int32(5), epi_buffer]
                            )
                            sc_v6 = cutlass.Float32(
                                sC[warp_m_base + local_row, local_col + Int32(6), epi_buffer]
                            )
                            sc_v7 = cutlass.Float32(
                                sC[warp_m_base + local_row, local_col + Int32(7), epi_buffer]
                            )
                            scatter_add_v4_bf16x2(
                                get_ptr_as_int64(
                                    scatter_output, tok * scatter_N + global_col
                                ),
                                wv * sc_v0,
                                wv * sc_v1,
                                wv * sc_v2,
                                wv * sc_v3,
                                wv * sc_v4,
                                wv * sc_v5,
                                wv * sc_v6,
                                wv * sc_v7,
                            )
                            vec_idx += Int32(self.num_threads_per_warp)

                        # all warps finish reading sC before the next output
                        # tile overwrites it
                        self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.stamps):
                    if Int32(tidx) == Int32(0):
                        if item_no < Int32(STAMP_ITEMS):
                            _st_global_i64(
                                get_ptr_as_int64(stamps, stamp_item + Int32(3)),
                                cute.arch.globaltimer(),
                            )

                # next item: the next FC1 loads target the stage buffers,
                # never sA2/sSFA2/sC, so no item-boundary barrier is needed
                item_no += Int32(1)
                current_work_linear_idx += num_persistent_clusters
                tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                    _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_shape_mnk[0]),
                        num_tiles_n=Int32(self.output_tile_count_n),
                        cluster_shape_mn=cluster_shape_mn,
                        current_work_linear_idx=current_work_linear_idx,
                        current_local_expert_idx=current_local_expert_idx,
                        accum_tile_m=accum_tile_m,
                        cta_id_in_cluster=cta_id_in_cluster,
                    )
                )
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
        # DMA WARP (warp 4): FC1 stages then FC2 stages, item after item
        # ===================================================================
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            num_persistent_clusters = Int32(gdim_z)
            cluster_shape_mn = (
                Int32(self.cluster_shape_mn[0]),
                Int32(self.cluster_shape_mn[1]),
            )
            cta_id_in_cluster = (
                Int32(bidx % cluster_shape_mn[0]),
                Int32(bidy % cluster_shape_mn[1]),
                Int32(0),
            )
            current_work_linear_idx = Int32(bidz)
            current_local_expert_idx = Int32(0)
            accum_tile_m = Int32(0)
            item_no = Int32(0)
            is_dma_lane0 = Int32(tidx) == Int32(self.tma_load_warp_id * 32)
            tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                _compact_static_get_work_tile(
                    row_counts,
                    active_expert_count,
                    tile_m=Int32(self.tile_shape_mnk[0]),
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

                sa_tile_coord_m = tc[0] // self.sa_tiles_per_block
                tAgA_mk = tAgA[(None, sa_tile_coord_m, None, local_expert_idx)]
                sfa_tile_coord_m = tc[0] // self.sfa_tiles_per_block
                tAgSFA_mk = tAgSFA[(None, sfa_tile_coord_m, None, local_expert_idx)]

                # w13 is [up, gate] along N: up slice s, gate slice s + gate_tile_cnt
                tBgB_up_nk = tBgB_w13[(None, intermediate_slice, None, weight_expert_idx)]
                sfb_up_tile_coord = intermediate_slice // self.sfb_tiles_per_block
                tBgSFB_up_nk = tBgSFB_w13[
                    (None, sfb_up_tile_coord, None, weight_expert_idx)
                ]
                gate_slice = intermediate_slice + gate_tile_cnt
                tBgB_gate_nk = tBgB_w13[(None, gate_slice, None, weight_expert_idx)]
                sfb_gate_tile_coord = gate_slice // self.sfb_tiles_per_block
                tBgSFB_gate_nk = tBgSFB_w13[
                    (None, sfb_gate_tile_coord, None, weight_expert_idx)
                ]

                # ---- FC1: A + gate + up per k-tile on the fc1 pipeline ----
                for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    fc1_pipeline.producer_acquire(fc1_prod_state)
                    bar = fc1_pipeline.producer_get_barrier(fc1_prod_state)
                    cute.copy(
                        tma_a,
                        tAgA_mk[(None, k_tile)],
                        tAsA[(None, fc1_prod_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_b_w13,
                        tBgB_gate_nk[(None, k_tile)],
                        tBsBg[(None, fc1_prod_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_b_w13,
                        tBgB_up_nk[(None, k_tile)],
                        tBsBu[(None, fc1_prod_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_sfa,
                        tAgSFA_mk[(None, k_tile)],
                        tAsSFA[(None, fc1_prod_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_sfb_w13,
                        tBgSFB_gate_nk[(None, k_tile)],
                        tBsSFBg[(None, fc1_prod_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_sfb_w13,
                        tBgSFB_up_nk[(None, k_tile)],
                        tBsSFBu[(None, fc1_prod_state.index)],
                        tma_bar_ptr=bar,
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

                # ---- FC2: the item's down tiles on the fc2 pipeline ----
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    fc2_pipeline.producer_acquire(fc2_prod_state)
                    bar2 = fc2_pipeline.producer_get_barrier(fc2_prod_state)
                    cute.copy(
                        tma_b_down,
                        tBgB_down[
                            (
                                None,
                                output_tile_idx,
                                intermediate_slice,
                                weight_expert_idx,
                            )
                        ],
                        tBsB2[(None, fc2_prod_state.index)],
                        tma_bar_ptr=bar2,
                    )
                    cute.copy(
                        tma_sfb_down,
                        tBgSFB_down[
                            (
                                None,
                                output_tile_idx // self.sfb_tiles_per_block,
                                intermediate_slice,
                                weight_expert_idx,
                            )
                        ],
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
                tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = (
                    _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_shape_mnk[0]),
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
        return


__all__ = ["MoEStaticKernelV2", "STAMP_SLOTS", "STAMP_ITEMS", "STAMP_MMA_END", "STAMP_DMA_BASE"]
