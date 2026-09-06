"""
MoEStaticKernelV4 -- v2 with FC1 streamed as two 64-wide N halves over
256-wide K stages, so every w13 TMA box row is 128 B (a full L2 line)
instead of the 64 B half line the stock kernel and v2 read.

Why (35차 §7 stamps): in v2 the FC1 phase streams w13 at 4.5 GB/s per CTA
(216 GB/s aggregate) while the FC2 phase streams w2 at 5.05 GB/s per CTA
(242 GB/s) with the same pipeline structure. w13 rows are 2,048 B apart and
each k-tile box takes 64 B of every row; w2's 256 B rows are read whole by
the four slice CTAs. Doubling the FC1 K tile to 256 fp4 elements makes the
w13 segment 128 B; halving the FC1 N tile to 64 keeps a stage at 26 KB
(A 4 + gate 8 + up 8 + scales 6) so two stages plus FC2's three fit smem.

Item structure (unchanged: one (m_tile, 128-wide intermediate slice, expert)
per item, 160 items at the served decode shape):

    FC1 half 0: k loop over 16 stages of (A 32x256, gate 64x256, up 64x256)
                -> activation -> fp4 quant into sA2 columns [0, 64)
    FC1 half 1: the same for columns [64, 128)
    FC2:        as v2 (32 down tiles of 128x128 from sA2/sSFA2, scatter)

Two tiled MMAs live in the kernel: (32, 64, 256) for FC1 and (32, 128, 128)
for FC2; the quantized intermediate keeps v2's K-major SW64 layout, so the
quant store formula and the FC2 side are v2's verbatim. Numerics: the FC1
accumulation is the same mma atom over the same k order (k-blocks of 64),
so results are bit-identical to v2 up to the bf16 atomic scatter order.

Selected by ``VLLM_GLM53_B12X_STATIC_V2`` spec cell ``w`` (tile_m 32, static
schedule; ``f``/``g`` set the FC1/FC2 stage counts; ``s`` stamps as in v2).
"""

from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint64
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


class MoEStaticKernelV4:
    """v3 with 256 B w13 row segments: FC1 halves over 512-wide K stages,
    gate and up in separate stages (32 KB each: A 8 + B 16 + SFA 4 + SFB 4),
    FC2 2 stages by default (smem)."""

    def __init__(
        self,
        sf_vec_size: int,
        output_tile_count_n: int,
        *,
        fc1_stages: int = 2,
        fc2_stages: int = 2,
        stamps: bool = False,
        even: bool = False,
        split: bool = False,
        skip_sf: bool = False,
        skip_a: bool = False,
        a_ring: bool = False,
        sf_pack: bool = False,
        sf_pack_raw: bool = False,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
        swiglu_alpha: float = 1.702,
        swiglu_beta: float = 1.0,
        swiglu_limit: float | None = None,
    ):
        if activation not in {"silu", "gelu_tanh", "swigluoai_uninterleave"}:
            raise ValueError(f"unsupported activation {activation!r} (v4 is gated only)")
        if sf_vec_size != _SF_VEC_SIZE:
            raise ValueError("v4 is the NVFP4 (sf_vec_size=16) lane")
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
        # even waves: only the largest CTA count in {48, 44, 40, 36, 32} that
        # leaves the fewest empty item slots takes items, so the last wave is
        # full (U=40: 40 CTAs x 4 items instead of 48 x 3.33). The item total
        # is accumulated in next_item[0] by the routing phase.
        self.even = bool(even)
        # split: when the last (partial) wave has p items and 2p CTAs are
        # available, each of those items runs on two CTAs -- the striding
        # owner streams FC1 half 0, the helper (bidz + p) half 1, each zeroes
        # the other half of the intermediate and runs the full FC2 (atomic
        # scatter adds the two partial sums). The 16-item wave at U=40 then
        # streams from 32 CTAs instead of 16.
        self.split = bool(split)
        if self.even and self.split:
            raise ValueError("e and k are exclusive (the split assumes gdim_z striding)")
        # probe-only timing variants (numerics are garbage): skip the FC1
        # SFB boxes (skip_sf) and/or the A + SFA boxes (skip_a) so the stamps
        # say what the small boxes cost the FC1 stream
        self.skip_sf = bool(skip_sf)
        self.skip_a = bool(skip_a)
        # a_ring: A + SFA ride their own 2-deep ring (own mbarriers) loaded
        # once per k tile and shared by the gate and the up stage, instead of
        # once per stage -- halves the L2->smem A traffic v4 doubled (v3's
        # `xa` diagnostic priced those loads at ~3%). Same smem: the A/SFA
        # staged buffers already exist per stage.
        self.a_ring = bool(a_ring)
        if self.a_ring and self.skip_a:
            raise ValueError("xa (skip A) and the A ring are exclusive")
        # sf_pack (cell q, 39차 §4c): the FC1 weight scales arrive 6-bit packed
        # (base + index per 4 KB block, two byte-aligned planes and the base in
        # a 16 B tail = 3088 B a stage instead of 4096), and the MMA warps
        # expand them IN PLACE in the stage's own scale buffer before reading
        # the fragment. Scales are 7.7% of the item's traffic and the kernel is
        # bandwidth-bound (§3e), so -25% of them is worth ~1.5% of the call
        # (§4d measured the whole box at 6.2%); the expansion rides in the DMA's
        # shadow, where the MMA warps are already waiting.
        self.sf_pack = bool(sf_pack)
        # sf_pack_raw (cell q0, diagnosis): same bulk-copy path and the same
        # block index, but the stage is the UNPACKED 4096 B block and nothing
        # is expanded. It splits a numerics failure in two -- if q0 passes, the
        # DMA address and the block index are right and the fault is in the
        # expansion; if q0 fails, it is the address.
        self.sf_pack_raw = bool(sf_pack_raw)
        self.sf_stage_bytes = _SF_BLOCK_BYTES if self.sf_pack_raw else _SF_STAGE_BYTES
        if self.sf_pack and self.skip_sf:
            raise ValueError("xs (skip the FC1 SFB boxes) and sf_pack are exclusive")
        if self.sf_pack and self.split:
            raise ValueError(
                "sf_pack needs every MMA warp at every FC1 stage; the split "
                "roles send the two halves to different warps"
            )
        # FC1: (32, 64, 512), one B (gate or up) per stage; FC2: (32, 128, 128)
        self.fc1_tile_shape_mnk = (_TILE_M, _FC1_TILE_N, _FC1_TILE_K)
        self.tile_shape_mnk = (_TILE_M, _FC2_TILE_N, _FC2_TILE_K)
        self.sa1_tile_shape_mk = (_TILE_M, _FC1_TILE_K)
        self.sfa1_tile_shape_mk = (128, _FC1_TILE_K)   # SF blocks are 128 rows
        self.sfa_tiles_per_block = 128 // _TILE_M
        # SFB gmem tiles are 128-row blocks: a 64-row box is not expressible
        # (39차 §3b -- the block interleaves its four 32-row groups at 4 B, so
        # half the rows is 8 B of every 16 and TMA wants 16 B contiguous)
        self.sfb1_tile_shape_nk = (128, _FC1_TILE_K)
        self.sfb1_tiles_per_block = 128 // _FC1_TILE_N   # 2 halves share a block
        self.sfb_tile_shape_nk = (128, _FC2_TILE_K)
        self.output_tile_count_n = output_tile_count_n
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi1_tile = (_TILE_M, _FC1_TILE_N)
        self.epi_tile = (_TILE_M, _FC2_TILE_N)
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
        # the 4 MMA warps expand a packed scale stage together: read, barrier,
        # write. Barrier 1 is this kernel's epilogue sync, and the stock dense
        # class it borrows helpers from names 1 and 2 (mma_sync / epilog_sync),
        # so this one takes 3 -- a collision would be a hang, not an error.
        self.sf_expand_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 32
        self.mma_register_requirement = 232
        self.smem_bytes = 0

    # the dense-kernel SF helpers read tiled_mma attributes only
    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFA(self, sfa_tensor, tiled_mma)

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFB(self, sfb_tensor, tiled_mma)

    def _get_layoutSFA_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFA_TV(self, tiled_mma)  # type: ignore[arg-type]

    def _get_layoutSFB_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFB_TV(self, tiled_mma)  # type: ignore[arg-type]

    def _make_a_smem_layout(self, rows: int, tile_k: int, stages: int):
        import cutlass.utils.hopper_helpers as sm90_utils

        a_is_k_major = self.a_layout.is_k_major_a()
        tile = (rows, tile_k)
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

    def _staged_layouts(self, tile_shape_mnk, epi_tile, tiled_mma, stages: int):
        (
            _,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
            epi_smem_staged,
        ) = self._dense_cls._make_smem_layouts(
            tile_shape_mnk,
            epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            stages,
            cutlass.BFloat16,
            self.c_layout,
            1,
            self.sf_vec_size,
            tiled_mma,
        )
        return b_smem_staged, sfa_smem_staged, sfb_smem_staged, epi_smem_staged

    def _sf_expand_stage(self, stage_addr, tidx):
        """Expand one packed FC1 scale stage in place (39차 §4c).

        The stage holds 2048 B of low nibbles, 1024 B of high 2-bit fields and
        the block's e4m3 base in a 16 B tail; the MMA fragment wants the 4096 B
        the stock TMA box would have written, in the same byte order (the block
        is one contiguous run -- §3b). Each of the 128 MMA threads owns 32
        output bytes and reads its own 24 B of input FIRST; the barrier between
        the two halves is what makes it safe in place, since thread 0's output
        lands on plane A bytes thread 1 has not read yet.

        The reads are VOLATILE for that reason: a plain ld.shared has no side
        effects, so nothing stops the compiler from sinking it past the barrier
        intrinsic, and then a thread reads bytes another thread has already
        overwritten. Same trap 35차 §8 recorded for the claim slot; here it
        showed up as scales wrong everywhere and different between runs (cell
        q's first GPU gate), while q0 -- the same DMA with no expansion --
        passed.
        """
        a = []
        for w in range(4):
            a.append(_ld_shared_i32_volatile(stage_addr + Int32(16) * tidx + Int32(4 * w)))
        b = []
        for w in range(2):
            b.append(_ld_shared_i32_volatile(
                stage_addr + Int32(_SF_PLANE_A) + Int32(8) * tidx + Int32(4 * w)))
        base = _ld_shared_i32_volatile(stage_addr + Int32(_SF_BASE_OFF)) & Int32(0xFF)
        # (plain Python loops: this helper is inlined into the traced kernel at
        # trace time, so every index below is a constant by the time IR is emitted)
        self.sf_expand_barrier.arrive_and_wait()
        for j in range(8):
            word = Int32(0)
            for m in range(4):
                i = 4 * j + m
                nib = (a[i >> 3] >> Int32(8 * ((i >> 1) & 3) + 4 * (i & 1))) & Int32(0xF)
                hi = (b[i >> 4] >> Int32(8 * ((i >> 2) & 3) + 2 * (i & 3))) & Int32(0x3)
                val = (base + nib + (hi << Int32(4))) & Int32(0xFF)
                word = word | (val << Int32(8 * m))
            _st_shared_i32(stage_addr + Int32(32) * tidx + Int32(4 * j), word)

    def _smem_bytes_estimate(self) -> int:
        def _align_up(value: int, align: int) -> int:
            return ((value + align - 1) // align) * align

        offset = (
            2 * 4
            + (self.fc1_stages + self.fc2_stages) * 2 * 8
            + self.fc1_stages * 2 * 8          # a_bars (always allocated)
            + _COMPACT_STATIC_TILE_M * 4
            + _COMPACT_STATIC_TILE_M * 4
        )
        buffers = [
            cute.size_in_bytes(self.a_dtype, self.a1_smem_layout_staged),
            cute.size_in_bytes(self.b_dtype, self.b1_smem_layout_staged),
            cute.size_in_bytes(self.sf_dtype, self.sfa1_smem_layout_staged),
            cute.size_in_bytes(self.sf_dtype, self.sfb1_smem_layout_staged),
            cute.size_in_bytes(self.b_dtype, self.b2_smem_layout_staged),
            cute.size_in_bytes(self.sf_dtype, self.sfb2_smem_layout_staged),
            cute.size_in_bytes(self.a_dtype, self.a2_smem_layout),
            cute.size_in_bytes(self.sf_dtype, self.sfa2_smem_layout),
            cute.size_in_bytes(cutlass.BFloat16, self.epi1_smem_layout_staged),
            cute.size_in_bytes(cutlass.BFloat16, self.epi_smem_layout_staged),
        ]
        for size in buffers:
            offset = _align_up(offset, self.buffer_align_bytes) + size
        return offset

    def _make_tiled_mma(self, tile_shape_mnk):
        import cutlass.utils.blackwell_helpers as sm120_utils

        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_layout = cute.make_layout((2, 2, 1))
        permutation_mnk = sm120_utils.get_permutation_mnk(
            tile_shape_mnk,
            self.sf_vec_size,
            False,
        )
        return mma_op, cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )

    def _setup_attributes(self, hidden_size: int):
        self._hidden_size = hidden_size
        mma_op, self.tiled_mma1 = self._make_tiled_mma(self.fc1_tile_shape_mnk)
        _, self.tiled_mma = self._make_tiled_mma(self.tile_shape_mnk)
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_m_tiles = _TILE_M // 32
        self.num_n_tiles1 = _FC1_TILE_N // 16
        self.num_k_blocks1 = _FC1_TILE_K // 64
        self.num_n_tiles = _FC2_TILE_N // 16
        self.num_k_blocks = _FC2_TILE_K // 64

        self.a1_smem_layout_staged = self._make_a_smem_layout(
            _TILE_M, _FC1_TILE_K, self.fc1_stages
        )
        (
            self.b1_smem_layout_staged,
            self.sfa1_smem_layout_staged,
            self.sfb1_smem_layout_staged,
            self.epi1_smem_layout_staged,
        ) = self._staged_layouts(
            self.fc1_tile_shape_mnk, self.epi1_tile, self.tiled_mma1, self.fc1_stages
        )
        (
            self.b2_smem_layout_staged,
            _,
            self.sfb2_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._staged_layouts(
            self.tile_shape_mnk, self.epi_tile, self.tiled_mma, self.fc2_stages
        )
        self.a2_smem_layout = self._make_a_smem_layout(_TILE_M, _FC2_TILE_K, 1)
        self.sfa2_smem_layout = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        self.smem_bytes = self._smem_bytes_estimate()
        if self.smem_bytes > self.smem_capacity:
            raise ValueError(
                f"v4 smem {self.smem_bytes} B exceeds {self.smem_capacity} B "
                f"(fc1 {self.fc1_stages} x fc2 {self.fc2_stages} stages)"
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
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
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
        next_item: cute.Tensor,   # even: item total (else unused)
        sfb1_packed: cute.Tensor,   # cell q: 6-bit FC1 scales (u8), dummy otherwise
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
        sfb_down_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_down.shape, self.sf_vec_size
        )
        sfb_down_tensor = cute.make_tensor(sfb_down_ptr, sfb_down_layout)

        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a, self.a1_smem_layout_staged, self.sa1_tile_shape_mk, 1
        )
        tma_sfa, gSFA = self._dense_cls._make_tma_atoms_and_tensors(
            sfa_tensor, self.sfa1_smem_layout_staged, self.sfa1_tile_shape_mk, 1,
            internal_type=cutlass.Int16,
        )
        tma_b_w13, gB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            b_w13, self.b1_smem_layout_staged, (_FC1_TILE_N, _FC1_TILE_K), 1
        )
        tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_w13_tensor, self.sfb1_smem_layout_staged, self.sfb1_tile_shape_nk, 1,
            internal_type=cutlass.Int16,
        )
        tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
            b_down, self.b2_smem_layout_staged, (_FC2_TILE_N, _FC2_TILE_K), 1
        )
        tma_sfb_down, gSFB_down = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_down_tensor, self.sfb2_smem_layout_staged, self.sfb_tile_shape_nk, 1,
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
            self.tiled_mma1,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a1_smem_layout_staged,
            self.b1_smem_layout_staged,
            self.sfa1_smem_layout_staged,
            self.sfb1_smem_layout_staged,
            self.epi1_smem_layout_staged,
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
            next_item,
            sfb1_packed,
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
        sfb1_packed: cute.Tensor,   # sf_pack: (stage bytes, blocks/expert, E) u8
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
            if cutlass.const_expr(self.sf_pack):
                fc1_tma_bytes += self.sf_stage_bytes
            else:
                fc1_tma_bytes += cute.size_in_bytes(self.sf_dtype, sfb1_smem_one)
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
        sfb1_base_addr = shared_ptr_to_u32(storage.sSFB1.data_ptr())
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
            scatter_output[j // cols, j % cols] = cutlass.BFloat16(0.0)
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
                if cutlass.const_expr(self.even or self.split):
                    if row % Int32(_TILE_M) == Int32(0):
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
        gSFB_w13_tiled = cute.local_tile(
            mSFB_w13, self.sfb1_tile_shape_nk, (None, None, None)
        )
        gB_down = cute.local_tile(
            mB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
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
        sSFB1_1 = cute.local_tile(sSFB1, sfb1_tile, (1, 0, None))
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
        tCrSFB1_0 = self._dense_cls._partition_fragment_SFB(
            self, sSFB1_0[None, None, 0], thr_mma1, tidx)  # type: ignore[arg-type]
        tCrSFB1_1 = self._dense_cls._partition_fragment_SFB(
            self, sSFB1_1[None, None, 0], thr_mma1, tidx)  # type: ignore[arg-type]

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
        tCrSFB2 = self._dense_cls._partition_fragment_SFB(
            self, sSFB2[None, None, 0], thr_mma, tidx  # type: ignore[arg-type]
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
            mma_tile_m1 = _TILE_M // cute.size(tRS_rGate, mode=[1])
            mma_tile_n1 = _FC1_TILE_N // cute.size(tRS_rGate, mode=[2])
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
            mma_tile_m = _TILE_M // cute.size(tRS_rDown, mode=[1])
            mma_tile_n = _FC2_TILE_N // cute.size(tRS_rDown, mode=[2])
            MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
            MmaNPerEpiN = self.epi_tile[1] // mma_tile_n

            scatter_N = Int32(scatter_output.shape[1])
            lane_id = Int32(tidx) & Int32(31)
            warp_in_tile = Int32(tidx) >> Int32(5)
            warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)
            warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)
            a2_rows = Int32(_TILE_M)
            sA2_u8 = cute.recast_tensor(sA2[None, None, 0], cutlass.Uint8)
            sf_blocks_per_half = Int32(_FC1_TILE_N // self.sf_vec_size)   # 4

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
                    tile_m=Int32(_TILE_M),
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
                tile_m_base = tile_coord[0] * Int32(_TILE_M)
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
                if valid_tile_rows > Int32(_TILE_M):
                    valid_tile_rows = Int32(_TILE_M)
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
                for h in cutlass.range_constexpr(2):
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
                                if cutlass.const_expr(self.sf_pack and not self.sf_pack_raw):
                                    self._sf_expand_stage(
                                        sfb1_base_addr
                                        + fc1_cons_state.index * Int32(_SF_BLOCK_BYTES),
                                        Int32(tidx),
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
                        if epi_rows > Int32(_TILE_M):
                            epi_rows = Int32(_TILE_M)
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

                    tile_n_base_cur = output_tile_idx * Int32(_FC2_TILE_N)
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
                        scatter_add_v4_bf16x2(
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
                        tile_m=Int32(_TILE_M),
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
                    tile_m=Int32(_TILE_M),
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
                for h in cutlass.range_constexpr(2):
                    if (role - Int32(2)) * (role - Int32(h)) == Int32(0):
                        up_tile = intermediate_slice * Int32(2) + Int32(h)
                        gate_tile = gate_tile_cnt + up_tile
                        tBgB_up_nk = tBgB_w13[(None, up_tile, None, weight_expert_idx)]
                        tBgB_gate_nk = tBgB_w13[(None, gate_tile, None, weight_expert_idx)]
                        # the SFB gmem tile is the 128-row block both halves
                        # share (39차 §3b: a 64-row box is not expressible)
                        sfb_up_idx = up_tile // Int32(self.sfb1_tiles_per_block)
                        sfb_gate_idx = gate_tile // Int32(self.sfb1_tiles_per_block)
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
                                    if cutlass.const_expr(self.sf_pack):
                                        # 3088 B of packed scales, one request,
                                        # into the stage's own 4 KB buffer; the
                                        # MMA warps expand it there (39차 §4c)
                                        if is_dma_lane0:
                                            if cutlass.const_expr(gu == 0):
                                                sfb_blk = sfb_gate_idx
                                            else:
                                                sfb_blk = sfb_up_idx
                                            _bulk_g2s(
                                                sfb1_base_addr
                                                + fc1_prod_state.index * Int32(_SF_BLOCK_BYTES),
                                                sfb1_packed_base
                                                + (Int64(weight_expert_idx) * sf_blocks_per_expert
                                                   + Int64(sfb_blk) * Int64(k_tile_cnt1)
                                                   + Int64(k_tile)) * Int64(self.sf_stage_bytes),
                                                Int32(self.sf_stage_bytes),
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
                        tile_m=Int32(_TILE_M),
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


__all__ = ["MoEStaticKernelV4"]
