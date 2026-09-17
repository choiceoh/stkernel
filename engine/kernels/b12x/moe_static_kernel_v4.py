"""Persistent GLM-5.3 NVFP4 MoE with streamed FC1 and FC2 pipelines.

Default t geometry: FC1 M32/N64/K512 over two 64-column halves; FC2
M32/N128/K128. Each work item owns one expert's 128-column intermediate
slice, retaining BF16 activation and per-slice output rounding.

The optional t,r decode reform uses M16/N128/K256 for FC1 and M16/N256/K128
for FC2, with four warps distributed along N. It removes duplicated FC1
scale-block reads, halves padded MMA work and halves FC2 output-tile count.
Weight storage and the 128-column intermediate boundary are unchanged.
Runtime dispatch specializes this bundle only for 1<=M<=8; larger batches
keep the default tile geometry. See MEASUREMENTS.md for measured evidence.
The optional SF6-v1 storage is read directly by both geometries: ordinary
FC1 gathers two packed K256 blocks, while FC2 gathers one N128 row half.
Expansion occurs only in the existing shared scale stages.
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
from .fp4_scale_search import quantize_block_fp4_search
from .moe_micro_kernel import (
    scatter_add_bf16x2_to_f32, scatter_add_bf16x4_to_f32, scatter_store_bf16x2_to_f32,
    scatter_add_bf16x8_from_smem_to_f32,
)

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
from .moe_w4a16_fp4_helpers import add_u8x4
from ._moe_dynamic.gated import load_global_bf16x16_to_f32x16
from .moe_static_common import (
    _bulk_g2s,
    _bulk_prefetch_l2,
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
    _ld_shared_u8_volatile,
    _ld_shared_u16_volatile,
    _spin_wait_global_eq_i32,
    _st_global_i64,
    _st_global_release_i32,
    _st_shared_f32,
    _st_shared_i32,
    _st_shared_u64,
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
        scatter_fp32: bool = False,
        route_scatter: bool = False,
        direct_scatter: bool = False,
        scatter_reuse: bool = False,
        fc2_prefetch: bool = False,
        fc1_stages: int = 2,
        fc2_stages: int = 2,
        l2_prefetch: int = 0,
        l2_prefetch_fc1: bool = True,
        bulk_b: bool = False,
        input_vec16: bool = False,
        input_reuse: int = 0,
        fc2_scale_search: int = 0,
        stamps: bool = False,
        decode_reform: bool = False,
        even: bool = False,
        split: bool = False,
        skip_sf: bool = False,
        skip_a: bool = False,
        a_ring: bool = False,
        sf_pack: bool = False,
        reform_sf_pack: bool = False,
        sf6_separate: bool = True,
        sf6_word_expand: bool = True,
        sf6_fc2_word_expand: bool = True,
        packed_activation_store: bool = True,
        fc1_reuse_a: bool = True,
        compact_staging: bool = True,
        sf6_registers: bool = True,
        sync_cleanup: bool = True,
        scatter_vec4: bool = True,
        scatter_packed_load: bool = True,
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
        self.scatter_fp32 = bool(scatter_fp32)
        self.route_scatter = bool(route_scatter)
        self.direct_scatter = bool(direct_scatter)
        # What private scatter needs is the packed FP32 output and an unsplit epilogue:
        # `_validate_direct_scatter_layout` binds register pairs to the epilogue tile
        # (`epi_tile = (tile_m, fc2_tile_n)`) and reads no scale state at all. reform_sf_pack is
        # the FC1 *scale* packing on the input side -- it sizes sf1_block_bytes/sf1_stage_bytes and
        # nothing the scatter touches. Requiring it here refused the companion lane that a
        # mixed-provenance checkpoint builds beside every sf6 lane, and at m=16 (where `batch`
        # turns direct scatter on) that refusal killed the boot in warmup_decode_experts
        # (measurements/st_hybrid_boot_block_20260916). route_scatter keeps the original pairing:
        # it re-indexes the output by route and has only ever been built on the sf6 tile.
        if self.route_scatter and not (scatter_fp32 and reform_sf_pack and not split):
            raise ValueError("route scatter requires packed FP32 output without split work")
        if self.direct_scatter and not (scatter_fp32 and not split):
            raise ValueError("private scatter requires packed FP32 output without split work")
        self.sf_vec_size = sf_vec_size
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.activation = activation
        self.is_gated = is_gated_activation(activation)
        assert self.is_gated
        self.fast_math = bool(fast_math)
        if fc2_scale_search not in (0, 1, 2):
            raise ValueError("FC2 scale search radius must be 0, 1 or 2")
        self.fc2_scale_search = int(fc2_scale_search)
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.swiglu_limit = float(swiglu_limit) if swiglu_limit is not None else None
        self.fc1_stages = int(fc1_stages)
        self.fc2_stages = int(fc2_stages)
        # l<n> (2026-09-16): the DMA warp asks L2 for the B stage n stages ahead of the one it lands
        # (cp.async.bulk.prefetch.L2, one request per contiguous 16 KB run, no smem). A CTA's two
        # 16 KB FC1 stages in flight over ~6 us of DRAM latency stream ~5 GB/s (the stamps of
        # st_c2_moe_chunk_20260915: FC1 5.0, FC2 5.5 GB/s per CTA, 48 CTAs = 215-220 of 273 GB/s),
        # and the ring cannot deepen: the CTA's smem is spent. The prefetch raises the bytes in
        # flight without a byte of smem. The MMA reads the same bytes, so numerics are the kernel's
        # own. Declared for the M16 reform tile over tile-major storage whose chunk equals the FC1 K
        # tile (the dispatcher checks): only then is a box one contiguous run.
        self.l2_prefetch = int(l2_prefetch)
        if self.l2_prefetch < 0 or self.l2_prefetch > 16:
            raise ValueError("l2 prefetch depth must be 0..16 stages")
        if self.l2_prefetch and not decode_reform:
            raise ValueError("l2 prefetch is declared for the M16 reform tile (t,r) only")
        # lf<n>: the FC1 boxes are left alone (the first ticket, c2l2prefetch2-0916, measured FC1 slower
        # under its own prefetch and FC2 faster), only the item's FC2 boxes -- across the seam and inside
        # the FC2 loop -- are asked for ahead.
        self.l2_prefetch_fc1 = bool(l2_prefetch_fc1)
        # z (2026-09-17): the B stages arrive as one 1-D cp.async.bulk each (16 KB) from storage whose
        # boxes are pre-swizzled into the stage's own byte order, instead of a 2-D TMA box of 128 (FC1)
        # or 256 (FC2) row segments -- the fewest requests a stage can be (38차 §8 read the path as
        # L2-request-rate bound; the original z on the t tile measured -2.5% with a permutation bug).
        # The mbarrier accounting is the TMA's: the same bytes complete on the same barrier. Declared
        # for the reform tile over tile-major storage whose chunk is the FC1 K tile (the dispatcher checks).
        self.bulk_b = bool(bulk_b)
        if self.bulk_b and not decode_reform:
            raise ValueError("bulk B stages are declared for the M16 reform tile (t,r) only")
        self.stamps = bool(stamps)
        self.decode_reform = bool(decode_reform)
        # One integrated C=1 tile: halve padded M work, consume both FC1
        # halves together, and double FC2 output width. Keep weight storage
        # and the 128-wide intermediate/rounding boundary unchanged.
        self.tile_m = 16 if self.decode_reform else _TILE_M
        self.fc1_tile_n = 128 if self.decode_reform else _FC1_TILE_N
        self.fc1_tile_k = 256 if self.decode_reform else _FC1_TILE_K
        self.fc2_tile_n = 256 if self.decode_reform else _FC2_TILE_N
        self.fc2_tile_k = _FC2_TILE_K
        self.fc1_halves = self.fc2_tile_k // self.fc1_tile_n
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
        self.reform_sf_pack = bool(reform_sf_pack)
        if self.reform_sf_pack and any((self.sf_pack, self.split, self.a_ring)):
            raise ValueError("sf6 requires the unmodified t or t,r geometry")
        # C=1 keeps FC1 packed input disjoint from expanded MMA scales.
        # Each pipeline slot owns both buffers until
        # consumer_release; expansion then needs only its publication barrier.
        self.sf6_separate = bool(sf6_separate and self.reform_sf_pack and self.decode_reform)
        self.sf6_word_expand = bool(sf6_word_expand and self.sf6_separate)
        self.sf6_fc2_word_expand = bool(sf6_fc2_word_expand and self.reform_sf_pack and self.decode_reform)
        self.packed_activation_store = bool(packed_activation_store and self.decode_reform)
        # The full K256 A/SFA fragment already has one register slice per
        # K64 block. Retain gate's slices through up; neither MMA writes them.
        # Gate alone loads A/SFA; the original B/SFB stage ring is unchanged.
        self.fc1_reuse_a = bool(fc1_reuse_a and self.decode_reform and self.reform_sf_pack)
        # Gate always occupies an even B/SFB slot when the ring is even.
        # Up reads registers, so only gate slots need A/SFA storage. The
        # original gate-slot release protects reuse of its compact input.
        # Spend part of that storage on disjoint FC2 packed scales, removing
        # the in-place read-before-write barrier without growing the CTA.
        self.compact_staging = bool(compact_staging and self.fc1_reuse_a
                                    and self.sf6_separate and self.fc1_stages % 2 == 0)
        # Load only the MMA lane's own scale words from the packed ring.
        # No expanded shared writes or cross-warp publication are needed.
        self.sf6_registers = bool(sf6_registers and self.compact_staging)
        self.scatter_reuse = bool(scatter_reuse)
        self.input_vec16 = bool(input_vec16)
        self.input_reuse = int(input_reuse)
        if self.input_reuse not in (0, 1, 2, 3, 4) or (self.input_reuse and not self.input_vec16):
            raise ValueError("input reuse requires a bounded BF16x16 cache cell")
        if self.input_vec16 and sf_vec_size != 16:
            raise ValueError("vector input loads require complete BF16x16 scale groups")
        if self.scatter_reuse and not (self.direct_scatter and self.sf6_registers
                                       and self.decode_reform and not self.route_scatter):
            raise ValueError("scatter reuse requires the M16 register SF6 atomic output path")
        self.fc2_prefetch = bool(fc2_prefetch)
        if self.fc2_prefetch:
            if not self.scatter_reuse or self.fc2_stages != 2:
                raise ValueError("FC2 prefetch requires the two-stage retained scatter path")
            # Direct scatter never reads or writes sC. Spend its 8 KiB and
            # the remaining CTA budget on a third B/SFB pipeline slot.
            # The existing TMA consumer release owns every slot's lifetime.
            self.fc2_stages = 3
        self.fc1_input_stages = self.fc1_stages // 2 if self.compact_staging else self.fc1_stages
        # SF6 cannot use the legacy A ring. Initialize only the two used
        # pipelines and publish their barriers together before Phase 0.
        self.sync_cleanup = bool(sync_cleanup and self.decode_reform and self.reform_sf_pack)
        # Each staged scatter lane owns eight contiguous columns. Combine
        # its four pairwise reductions into two 16-byte vector reductions.
        self.scatter_vec4 = bool(scatter_vec4 and self.scatter_fp32 and self.decode_reform
                                 and self.reform_sf_pack and not self.direct_scatter
                                 and not self.route_scatter)
        self.scatter_packed_load = bool(scatter_packed_load and self.scatter_vec4)
        self.a_barrier_count = 0 if self.sync_cleanup else self.fc1_stages * 2
        # Scatter only consumes rows in this M16 tile. Avoid initializing
        # 112 unused token/weight entries per item and reclaim their storage.
        self.scatter_cache_rows = self.tile_m if self.sf6_separate else _COMPACT_STATIC_TILE_M
        # skip_sf / skip_a are compile-time omissions of a TMA issue, not a layout: the branches below are
        # `const_expr(not self.skip_*)` around the box descriptors alone, so they compose with any scale
        # geometry. They stayed excluded here only because nothing had asked -- and what asks is the one
        # question the stamps exist for: which of the served recipe's boxes holds the fixed cost
        # (45차 §23 조사 17차: 13.0 ms a layer against a 4.3 ms bank read).
        # SF6-v1 stays 2048 raw bytes -> 1552 packed bytes for both tile
        # geometries. Ordinary FC1 K512 needs two adjacent K256 blocks.
        # Ordinary FC2 N128 uses one row half of a packed N256 block.
        self.sf1_block_bytes = 2048 if self.reform_sf_pack and self.decode_reform else _SF_BLOCK_BYTES
        self.sf1_packed_blocks = self.fc1_tile_k // 256
        self.sf1_stage_bytes = 1552 if self.reform_sf_pack else _SF_STAGE_BYTES
        self.sf6_packed_bytes = self.fc1_stages * self.sf1_stage_bytes if self.sf6_separate else 0
        self.sf2_block_bytes = 2048 if self.decode_reform else 1024
        # Gather just one N128 row half in its existing 1024-byte stage:
        # low512 + high256 + base/alignment16. No extra shared/global buffer.
        self.sf2_stage_bytes = 1552 if self.decode_reform else 784
        self.sf2_packed_bytes = self.fc2_stages * self.sf2_stage_bytes if self.compact_staging else 0
        if self.sf_pack and self.skip_sf:
            raise ValueError("xs (skip the FC1 SFB boxes) and sf_pack are exclusive")
        if self.sf_pack and self.split:
            raise ValueError(
                "sf_pack needs every MMA warp at every FC1 stage; the split "
                "roles send the two halves to different warps"
            )
        # FC1: (32, 64, 512), one B (gate or up) per stage; FC2: (32, 128, 128)
        self.fc1_tile_shape_mnk = (self.tile_m, self.fc1_tile_n, self.fc1_tile_k)
        self.tile_shape_mnk = (self.tile_m, self.fc2_tile_n, self.fc2_tile_k)
        self.sa1_tile_shape_mk = (self.tile_m, self.fc1_tile_k)
        self.sfa1_tile_shape_mk = (128, self.fc1_tile_k)   # SF blocks are 128 rows
        self.sfa_tiles_per_block = 128 // self.tile_m
        # SFB gmem tiles are 128-row blocks: a 64-row box is not expressible
        # (39차 §3b -- the block interleaves its four 32-row groups at 4 B, so
        # half the rows is 8 B of every 16 and TMA wants 16 B contiguous)
        self.sfb1_tile_shape_nk = (128, self.fc1_tile_k)
        self.sfb1_tiles_per_block = 128 // self.fc1_tile_n   # 2 halves share a block
        self.sfb_tile_shape_nk = (self.fc2_tile_n, self.fc2_tile_k)
        self.output_tile_count_n = output_tile_count_n
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        if self.sync_cleanup and (self.fc1_halves != 1 or self.cluster_shape_mnk != (1, 1, 1)):
            raise ValueError("sync cleanup requires one FC1 half and a single-CTA cluster")
        self.epi1_tile = (self.tile_m, self.fc1_tile_n)
        self.epi_tile = (self.tile_m, self.fc2_tile_n)
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

    def _partition_fragment_SFB(self, tensor, thr_mma, tidx):
        fragment = self._dense_cls._partition_fragment_SFB(self, tensor, thr_mma, tidx)
        if self.decode_reform and cute.rank(fragment) == 2:
            # The legacy helper folds N and K together when the M atom extent
            # is one. Restore (values, N tiles, K blocks) without changing the
            # scale bytes or their lane assignment.
            shape, stride = fragment.shape, fragment.stride
            assert len(shape[1]) == 2
            fragment = cute.make_tensor(fragment.iterator, cute.make_layout(
                (shape[0], shape[1][0], shape[1][1]),
                stride=(stride[0], stride[1][0], stride[1][1])))
        assert cute.rank(fragment) == 3
        return fragment

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

    def _sf6_expand_word(self, low, high, base):
        """Spread four 6-bit codes, then add the broadcast base modulo 256."""
        low = (low | (low << Int32(8))) & Int32(0x00FF00FF)
        low = (low | (low << Int32(4))) & Int32(0x0F0F0F0F)
        high = (high | (high << Int32(12))) & Int32(0x000F000F)
        high = (high | (high << Int32(6))) & Int32(0x03030303)
        return Int32(add_u8x4(low | (high << Int32(4)), base))

    def _sf_expand_stage(self, stage_addr, tidx, block_bytes=4096, *, packed_addr=None,
                         word_expand=None):
        """Exact MMA-stage expansion shared by q, sf6 and the device gate.

        The 1024-byte form expands the selected FC2 row half gathered as
        low512 + high256 + base/tail16, with the original full-stage base.

        A separate packed_addr must name storage disjoint from the entire
        expanded stage. Only that form can omit the read-before-write barrier.
        Volatile reads cannot sink across the in-place read-before-write barrier.
        The post-write barrier publishes every owner's bytes before peers
        load MMA fragments (39-sf-pack-kernel correctness fixes).
        word_expand overrides only the integer reconstruction. In-place FC2
        retains both barriers and all volatile packed reads in either form.
        """
        if block_bytes not in (1024, 2048, 4096):
            raise ValueError("unsupported scale expansion stage")
        per_thread = block_bytes // 128
        plane_a = block_bytes // 2
        base_offset = block_bytes * 3 // 4
        source_addr = stage_addr if packed_addr is None else packed_addr
        a = []
        for w in range(per_thread // 8):
            a.append(_ld_shared_i32_volatile(
                source_addr + Int32(per_thread // 2) * tidx + Int32(4 * w)))
        b = []
        if per_thread == 8:
            # Two adjacent threads share an aligned high-plane word, but
            # each keeps only its own two bytes before in-place expansion.
            b.append(_ld_shared_i32_volatile(
                source_addr + Int32(plane_a) + (tidx // Int32(2)) * Int32(4))
                >> ((tidx & Int32(1)) * Int32(16)))
        else:
            for w in range(per_thread // 16):
                b.append(_ld_shared_i32_volatile(
                    source_addr + Int32(plane_a) + Int32(per_thread // 4) * tidx
                    + Int32(4 * w)))
        base = _ld_shared_i32_volatile(source_addr + Int32(base_offset)) & Int32(0xFF)
        if packed_addr is None:
            self.sf_expand_barrier.arrive_and_wait()
        if word_expand is None:
            word_expand = self.sf6_word_expand and packed_addr is not None
        if word_expand:
            packed_base = base * Int32(0x01010101)
        for j in range(per_thread // 4):
            if word_expand:
                word = self._sf6_expand_word(
                    (a[j // 2] >> Int32(16 * (j % 2))) & Int32(0xFFFF),
                    (b[j // 4] >> Int32(8 * (j % 4))) & Int32(0xFF), packed_base)
            else:
                word = Int32(0)
                for m in range(4):
                    i = 4 * j + m
                    nib = (a[i >> 3] >> Int32(8 * ((i >> 1) & 3) + 4 * (i & 1))) & Int32(0xF)
                    hi = (b[i >> 4] >> Int32(8 * ((i >> 2) & 3) + 2 * (i & 3))) & Int32(0x3)
                    val = (base + nib + (hi << Int32(4))) & Int32(0xFF)
                    word = word | (val << Int32(8 * m))
            _st_shared_i32(stage_addr + Int32(per_thread) * tidx + Int32(4 * j), word)
        self.sf_expand_barrier.arrive_and_wait()

    def _sf6_prepare_stage(self, packed_base, tidx):
        """Keep invariant scale metadata for all K64 fragments until release."""
        base = _ld_shared_u8_volatile(packed_base, 1536)
        base = base * Int32(0x01010101)
        quad_lane = Int32(tidx) & Int32(3)
        row = ((Int32(tidx) & Int32(31)) >> Int32(2)) * Int32(16)
        row += (Int32(tidx) & Int32(32)) * Int32(8) + (Int32(tidx) & Int32(64)) * Int32(2)
        # Each quad member owns exactly two low-plane bytes and one high-plane
        # byte. Select them once here instead of shifting a shared word at
        # every K fragment. The copy-layout oracle also checks this lane map.
        low = packed_base + (row >> Int32(1)) + quad_lane * Int32(2)
        high = packed_base + Int32(1024) + (row >> Int32(2)) + quad_lane
        return (low, high, base, Int32(tidx) & Int32(28))

    def _sf6_load_fragment(self, dest, stage, kind, k_block):
        """Reconstruct exact SFB operands cooperatively within each lane quad.

        _verify_sf6_register_layout proves these offsets against the ordinary
        shared-to-register copy before compiling. Pipeline wait/release owns
        the prepared metadata and packed reads; there are no shared writes.
        """
        dst = cute.recast_tensor(dest, Int32)
        offsets = self.sf6_register_offsets[kind][0][k_block]
        assert cute.size(dst) == len(offsets) and len(offsets) % 4 == 0
        low_base, high_base, base, quad_base = stage
        for group in range(len(offsets) // 4):
            # Each group is 16-byte aligned in the verified ordinary view.
            offset = offsets[group * 4]
            low = _ld_shared_u16_volatile(low_base, offset // 2)
            high = _ld_shared_u8_volatile(high_base, offset // 4)
            word = self._sf6_expand_word(low, high, base)
            for lane in range(4):
                dst[group * 4 + lane] = cute.arch.shuffle_sync(word, quad_base + Int32(lane))

    def _verify_sf6_register_layout(self):
        """Check each ordinary copy word against the real CuTe SFB layouts.

        An arithmetic identity iterator exposes physical byte offsets without
        allocating or reading a GPU tensor. Keep every lane, including the
        duplicated scale owners, in the receipt used by the CPU byte oracle.
        """
        records = {}
        for kind, layout, mma, tile_shape in (
            ("fc1", self.sfb1_smem_layout_staged, self.tiled_mma1, self.fc1_tile_shape_mnk),
            ("fc2", self.sfb2_smem_layout_staged, self.tiled_mma, self.tile_shape_mnk),
        ):
            identity = cute.make_identity_tensor(cute.cosize(layout))
            physical = cute.make_tensor(identity.iterator, layout)
            tile = cute.local_tile(physical, cute.slice_(tile_shape, (0, None, None)), (0, 0, None))
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
            copy = cute.make_tiled_copy(atom, self._get_layoutSFB_TV(mma),
                (cute.size(mma.permutation_mnk[1]), cute.size(mma.permutation_mnk[2])))
            lanes = []
            for tid in range(128):
                source = cute.filter_zeros(copy.get_slice(tid).partition_S(tile)[None, None, None, 0])
                blocks = []
                for kb in range(cute.size(source, mode=[2])):
                    block = source[None, None, kb]
                    offsets = [int(block[i]) for i in range(cute.size(block))]
                    assert len(offsets) % 16 == 0
                    words = []
                    for i in range(0, len(offsets), 4):
                        word = offsets[i]
                        assert word % 4 == 0 and 0 <= word <= 2044, (kind, tid, offsets)
                        assert offsets[i:i+4] == list(range(word, word+4)), (kind, tid, offsets)
                        words.append(word)
                    blocks.append(words)
                lanes.append(blocks)
            for tid, blocks in enumerate(lanes):
                for kb, words in enumerate(blocks):
                    row = ((tid & 31) // 4)*16 + (tid & 32)*8 + (tid & 64)*2
                    assert words == [b + row for b in lanes[0][kb]], (kind, tid, kb, words)
                    for i in range(0, len(words), 4):
                        assert words[i] % 16 == 0
                        assert words[i:i+4] == list(range(words[i], words[i]+16, 4))
            records[kind] = lanes
        self.sf6_register_offsets = records

    def _store_packed_activation(self, base_addr, layout, row, packed_base, packed):
        """Preserve the quantizer's eight bytes and the FC2 consumer swizzle.

        The M16 layout maps each aligned eight-byte block contiguously:
        S<2,4,3> only changes bits 4 and 5, leaving bits 0..2 intact.
        _setup_attributes verifies this against the real consumer layout.
        """
        if self.packed_activation_store:
            fp4_offset = cute.crd2idx((row, packed_base * Int32(2), 0), layout.outer)
            byte_offset = fp4_offset // Int32(2)
            byte_offset = byte_offset ^ ((byte_offset >> Int32(3)) & Int32(0x30))
            _st_shared_u64(base_addr + byte_offset, packed)
        else:
            for byte_idx in range(self.sf_vec_size // 2):
                src_pcol = packed_base + Int32(byte_idx)
                byte_val = Uint8((packed >> Uint64(byte_idx * 8)) & Uint64(0xFF))
                fp4_offset = cute.crd2idx((row, src_pcol * Int32(2), 0), layout.outer)
                byte_offset = fp4_offset // Int32(2)
                byte_offset = byte_offset ^ ((byte_offset >> Int32(3)) & Int32(0x30))
                st_shared_u8(base_addr + byte_offset, byte_val)

    def _sf6_copy_fc2_half(self, dest_addr, source_addr, bar_addr, row_half):
        """Gather the selected SF6-v1 row half into a 784-byte shared stage.

        The original [K64][row128][512] byte order is retained. Five aligned
        bulk loads copy low2x256, high2x128 and the unchanged base/tail16.
        Their total must match the FC2 pipeline's 784 expected transaction
        bytes. The MMA consumer expands only after that barrier completes.
        """
        for k64 in range(2):
            _bulk_g2s(dest_addr + Int32(k64 * 256),
                      source_addr + Int64(k64 * 512) + Int64(row_half) * Int64(256),
                      Int32(256), bar_addr)
            _bulk_g2s(dest_addr + Int32(512 + k64 * 128),
                      source_addr + Int64(1024 + k64 * 256) + Int64(row_half) * Int64(128),
                      Int32(128), bar_addr)
        _bulk_g2s(dest_addr + Int32(768), source_addr + Int64(1536), Int32(16), bar_addr)

    def _smem_bytes_estimate(self) -> int:
        def _align_up(value: int, align: int) -> int:
            return ((value + align - 1) // align) * align

        offset = (
            2 * 4
            + (self.fc1_stages + self.fc2_stages) * 2 * 8
            + self.a_barrier_count * 8
            + self.scatter_cache_rows * 4
            + self.scatter_cache_rows * 4
        )
        # Placing FC1 packed input in the header also consumes existing
        # padding before the first 1024-byte-aligned tensor. A separate
        # allocation after Storage would lose this padding reuse.
        offset = _align_up(offset, 16) + self.sf6_packed_bytes
        offset = _align_up(offset, 16) + self.sf2_packed_bytes
        buffers = [
            cute.size_in_bytes(self.a_dtype, self.a1_smem_layout_staged),
            cute.size_in_bytes(self.b_dtype, self.b1_smem_layout_staged),
            cute.size_in_bytes(self.sf_dtype, self.sfa1_smem_layout_staged),
            0 if self.sf6_registers else cute.size_in_bytes(self.sf_dtype, self.sfb1_smem_layout_staged),
            cute.size_in_bytes(self.b_dtype, self.b2_smem_layout_staged),
            0 if self.sf6_registers else cute.size_in_bytes(self.sf_dtype, self.sfb2_smem_layout_staged),
            cute.size_in_bytes(self.a_dtype, self.a2_smem_layout),
            cute.size_in_bytes(self.sf_dtype, self.sfa2_smem_layout),
            cute.size_in_bytes(cutlass.BFloat16, self.epi1_smem_layout_staged),
            0 if self.fc2_prefetch else cute.size_in_bytes(cutlass.BFloat16, self.epi_smem_layout_staged),
        ]
        for size in buffers:
            offset = _align_up(offset, self.buffer_align_bytes) + size
        return offset

    def _fc1_input_slot(self, stage):
        # An even ring starts at gate slot zero and advances in gate/up
        # pairs, including across work items. Odd rings keep their layout.
        return stage // Int32(2) if self.compact_staging else stage

    def _make_tiled_mma(self, tile_shape_mnk):
        import cutlass.utils.blackwell_helpers as sm120_utils

        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_layout = cute.make_layout((1, 4, 1) if self.decode_reform else (2, 2, 1))
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

    def _validate_fc1_reuse_fragment(self, fragment, name):
        """Every K64 block must survive the next block's register loads."""
        if cute.rank(fragment) != 3 or cute.size(fragment, mode=[2]) != self.fc1_tile_k // 64:
            raise ValueError(f"FC1 reuse requires a full K tile in {name} registers")
        identity = cute.make_identity_tensor(fragment.shape)
        blocks = [set() for _ in range(self.fc1_tile_k // 64)]
        for i in range(cute.size(fragment)):
            point = tuple(identity[i])
            blocks[int(point[2])].add(int(cute.crd2idx(point, fragment.layout)))
        seen = set()
        for block in blocks:
            if not block or block & seen:
                raise ValueError(f"FC1 reuse found aliased K blocks in {name} registers")
            seen.update(block)
        return tuple(len(block) for block in blocks)

    def _validate_direct_scatter_layout(self):
        """Bind register pairs to the actual copy layout before native compile."""
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
        st = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 2), cutlass.BFloat16)
        copy = cute.make_tiled_copy_S(atom, cute.make_tiled_copy_C_atom(st, self.tiled_mma))
        identity = cute.make_identity_tensor((*self.epi_tile, 1))
        staged = cute.make_identity_tensor(cute.shape(self.epi_smem_layout_staged.outer))
        seen = set()
        for tid in range(128):
            thread = copy.get_slice(tid)
            if cute.size(thread.partition_D(staged), mode=[3]) != 1:
                raise ValueError("direct scatter requires one output tile buffer")
            coords = thread.partition_D(identity)[None, None, None, 0]
            registers = cute.make_layout(cute.shape(thread.partition_S(staged))[:3])
            if cute.size(coords) != cute.size(registers) or cute.size(coords) % 2:
                raise ValueError("direct scatter register/coordinate size mismatch")
            rows, row_pairs, pair_rows = [], [], []
            for pair in range(cute.size(coords) // 2):
                a, b = tuple(coords[2 * pair]), tuple(coords[2 * pair + 1])
                if not (a[0] == b[0] and a[1] % 2 == 0 and b[1] == a[1] + 1
                        and a[2] == b[2] == 0):
                    raise ValueError("direct scatter needs adjacent BF16 register pairs")
                if a[0] not in rows:
                    rows.append(a[0])
                    row_pairs.append(pair)
                pair_rows.append(rows.index(a[0]))
                for point in (a, b):
                    if point in seen:
                        raise ValueError("direct scatter duplicate output coordinate")
                    seen.add(point)
            if self.scatter_reuse:
                if len(rows) != 2:
                    raise ValueError("M16 scatter reuse requires two output rows per lane")
                mapping = (tuple(row_pairs), tuple(pair_rows))
                if tid == 0:
                    self.scatter_row_pairs, self.scatter_pair_rows = mapping
                elif mapping != (self.scatter_row_pairs, self.scatter_pair_rows):
                    raise ValueError("scatter row ownership differs between MMA lanes")
        if seen != {(r, c, 0) for r in range(self.epi_tile[0]) for c in range(self.epi_tile[1])}:
            raise ValueError("direct scatter incomplete output coverage")

    def _scatter_smem_byte_offset(self, element_offset):
        """S<3,4,3> applies to bytes; the outer epilogue layout counts BF16."""
        offset = element_offset * 2
        return offset ^ ((offset >> 3) & 0x70)

    def _verify_scatter_packed_layout(self):
        """Bind every aligned load to the ordinary BF16 shared consumer map."""
        layout = self.epi_smem_layout_staged
        if self.epi_tile != (16, 256) or layout.inner != cute.make_swizzle(3, 4, 3):
            raise ValueError("packed scatter requires the M16/N256 BF16 S<3,4,3> layout")
        byte_swizzle = cute.make_composed_layout(layout.inner, 0, cute.make_layout(16*256*2))
        seen = set()
        for row in range(self.epi_tile[0]):
            for col in range(0, self.epi_tile[1], 8):
                elements = [int(cute.crd2idx((row, col+j, 0), layout.outer)) for j in range(8)]
                ordinary = [int(cute.crd2idx(2 * offset, byte_swizzle)) for offset in elements]
                start = self._scatter_smem_byte_offset(elements[0])
                if start % 16 or ordinary != [start+2*j for j in range(8)]:
                    raise ValueError("packed scatter load differs from the ordinary shared layout")
                for offset in ordinary:
                    if offset in seen:
                        raise ValueError("packed scatter shared output coordinates overlap")
                    seen.add(offset)
        if seen != set(range(0, 16*256*2, 2)):
            raise ValueError("packed scatter shared output coverage is incomplete")

    def _setup_attributes(self, hidden_size: int):
        self._hidden_size = hidden_size
        if self.scatter_vec4 and hidden_size % 4:
            raise ValueError("vector scatter requires 16-byte-aligned FP32 output rows")
        mma_op, self.tiled_mma1 = self._make_tiled_mma(self.fc1_tile_shape_mnk)
        _, self.tiled_mma = self._make_tiled_mma(self.tile_shape_mnk)
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_m_tiles = self.tile_m // (16 if self.decode_reform else 32)
        self.num_n_tiles1 = self.fc1_tile_n // (32 if self.decode_reform else 16)
        self.num_k_blocks1 = self.fc1_tile_k // 64
        self.num_n_tiles = self.fc2_tile_n // (32 if self.decode_reform else 16)
        self.num_k_blocks = self.fc2_tile_k // 64

        self.a1_smem_layout_staged = self._make_a_smem_layout(
            self.tile_m, self.fc1_tile_k, self.fc1_input_stages
        )
        (
            self.b1_smem_layout_staged,
            self.sfa1_smem_layout_staged,
            self.sfb1_smem_layout_staged,
            self.epi1_smem_layout_staged,
        ) = self._staged_layouts(
            self.fc1_tile_shape_mnk, self.epi1_tile, self.tiled_mma1, self.fc1_stages
        )
        if self.compact_staging:
            original_sfa = self.sfa1_smem_layout_staged
            self.sfa1_smem_layout_staged = sm120_make_smem_layout_sfa(
                self.tiled_mma1, self.fc1_tile_shape_mnk, self.sf_vec_size, self.fc1_input_stages)
            original_a = self._make_a_smem_layout(self.tile_m, self.fc1_tile_k, self.fc1_stages)
            for original, compact in ((original_a, self.a1_smem_layout_staged),
                                      (original_sfa, self.sfa1_smem_layout_staged)):
                if (cute.slice_(original, (None, None, 0)) != cute.slice_(compact, (None, None, 0))
                        or cute.cosize(original) != 2 * cute.cosize(compact)
                        or cute.size(compact, mode=[2]) != self.fc1_input_stages):
                    raise ValueError("compact FC1 input changed the per-stage consumer layout")
        (
            self.b2_smem_layout_staged,
            _,
            self.sfb2_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._staged_layouts(
            self.tile_shape_mnk, self.epi_tile, self.tiled_mma, self.fc2_stages
        )
        self.a2_smem_layout = self._make_a_smem_layout(self.tile_m, self.fc2_tile_k, 1)
        if self.direct_scatter:
            self._validate_direct_scatter_layout()
        if self.scatter_packed_load:
            self._verify_scatter_packed_layout()
        self.sfa2_smem_layout = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        if self.decode_reform:
            for mma, shape in ((self.tiled_mma1, (self.tile_m, self.fc1_tile_n)),
                               (self.tiled_mma, (self.tile_m, self.fc2_tile_n))):
                ident = cute.make_identity_tensor(shape)
                seen = set()
                for tid in range(128):
                    coords = mma.get_slice(tid).partition_C(ident)
                    for i in range(cute.size(coords)):
                        point = tuple(coords[i])
                        assert point not in seen, point
                        seen.add(point)
                assert seen == {(m,n) for m in range(shape[0]) for n in range(shape[1])}
            seen_bytes = set()
            # Shared pointers apply S<2,4,3> to BYTE offsets, whereas the
            # FP4 outer layout counts nibbles. Check against the consumer
            # mapping, not just a bijection (the wrong map is also bijective).
            assert self.a2_smem_layout.inner == cute.make_swizzle(2, 4, 3)
            for row in range(self.tile_m):
                for col in range(0, self.fc2_tile_k, 2):
                    lo = int(cute.crd2idx((row,col,0), self.a2_smem_layout.outer))
                    hi = int(cute.crd2idx((row,col+1,0), self.a2_smem_layout.outer))
                    assert lo % 2 == 0 and hi == lo + 1, (row,col,lo,hi)
                    byte = lo // 2
                    byte ^= (byte >> 3) & 0x30
                    consumer = row * (self.fc2_tile_k // 2) + ((col // 2) ^ (((row >> 1) & 3) << 4))
                    assert byte == consumer, (row, col, byte, consumer)
                    assert byte not in seen_bytes
                    seen_bytes.add(byte)
                    if self.packed_activation_store:
                        block_col = col - col % self.sf_vec_size
                        block = int(cute.crd2idx((row, block_col, 0), self.a2_smem_layout.outer)) // 2
                        block ^= (block >> 3) & 0x30
                        assert block % 8 == 0 and byte == block + (col - block_col) // 2
            print('DECODE_REFORM_LAYOUT_PASS', len(seen_bytes), flush=True)
        if self.reform_sf_pack:
            self._verify_reform_sf_layout(hidden_size)
        if self.sf6_registers:
            self._verify_sf6_register_layout()
        self.smem_bytes = self._smem_bytes_estimate()
        if self.smem_bytes > self.smem_capacity:
            raise ValueError(
                f"v4 smem {self.smem_bytes} B exceeds {self.smem_capacity} B "
                f"(fc1 {self.fc1_stages} x fc2 {self.fc2_stages} stages)"
            )

    def _verify_reform_sf_layout(self, hidden_size):
        """Compile-time full-byte proof of the packer's actual MMA consumer map.

        A bijection alone is insufficient: the former FP4 swizzle regression
        was bijective too. Check source AND destination offsets from CuTe's
        real layouts, including the two separated FC2 row blocks.
        """
        from .moe_reform_sf_pack import stage_shape, stage_source_offset
        intermediate = self.output_tile_count_n * 128
        for kind, rows, k, rn, kn, smem_layout in (
            ("fc1", intermediate*2, hidden_size, 128, self.fc1_tile_k, self.sfb1_smem_layout_staged),
            ("fc2", hidden_size, intermediate, self.fc2_tile_n, 128, self.sfb2_smem_layout_staged),
        ):
            stage_shape(rows, k, kind)
            nr, nk = rows // rn, k // kn
            block_bytes = rn * kn // 16
            source = blockscaled_utils.tile_atom_to_shape_SF((rows, k, 2), 16)
            assert int(cute.crd2idx((0, 0, 1), smem_layout)) == block_bytes, kind
            covered = set()
            for row in range(rn):
                for col in range(kn // 16):
                    dest = int(cute.crd2idx((row, col*16, 0), smem_layout))
                    assert dest not in covered, (kind, row, col, dest)
                    covered.add(dest)
                    for e, rt, kt in ((0, 0, 0), (0, min(1, nr-1), 0), (1, nr-1, nk-1)):
                        actual = int(cute.crd2idx((rt*rn+row, kt*kn+col*16, e), source))
                        if not self.decode_reform and kind == "fc1":
                            expected = stage_source_offset(rows, k, kind, e, rt,
                                                           kt*2 + dest//2048, dest%2048)
                        elif not self.decode_reform:
                            packed_byte = (dest//512)*1024 + (rt%2)*512 + dest%512
                            expected = stage_source_offset(rows, k, kind, e, rt//2, kt, packed_byte)
                        else:
                            expected = stage_source_offset(rows, k, kind, e, rt, kt, dest)
                        assert actual == expected, (kind, e, rt, kt, row, col, actual, expected)
            assert covered == set(range(block_bytes)), kind
        print(f"REFORM_SF6_LAYOUT_PASS FC1={self.sf1_block_bytes} FC2={self.sf2_block_bytes}", flush=True)

    @cute.jit
    def _prepare_token_routes(self, tid: Int32, smem, ids: cute.Tensor,
                              weights: cute.Tensor, counts: cute.Tensor,
                              active: cute.Tensor, experts: cute.Tensor,
                              global_to_local: cute.Tensor, tokens: cute.Tensor,
                              route_weights: cute.Tensor, metadata: cute.Tensor,
                              topk: Int32):
        # One CTA prepares the small K7 route table while the other resident
        # CTAs quantize inputs. The ordinary phase-0 grid fence publishes both.
        # First-occurrence order gives every expert a compact ID and each
        # route a unique row, without global CAS, spins or row-count atomics.
        pairs = Int32(ids.shape[0])
        expert = Int32(-1)
        row = Int32(0)
        count = Int32(0)
        first = tid
        if tid < pairs:
            expert = ids[tid].to(Int32)
            _st_shared_i32(smem + tid * Int32(4), expert)
        cute.arch.sync_threads()
        if tid < pairs:
            j = Int32(0)
            while j < pairs:
                other = _ld_shared_i32(smem + j * Int32(4))
                if other == expert:
                    count += Int32(1)
                    if j < tid:
                        row += Int32(1)
                    if j < first:
                        first = j
                j += Int32(1)
            if cutlass.const_expr(self.input_reuse == 4):
                # Every active warp is full (64/128 routes). Publish one mask
                # per warp instead of making each route scan 64/128 flags.
                first_mask = cute.arch.vote_ballot_sync(row == Int32(0))
                if tid % Int32(32) == Int32(0):
                    _st_shared_i32(smem + (pairs + tid // Int32(32)) * Int32(4), first_mask.to(Int32))
            else:
                _st_shared_i32(smem + (pairs + tid) * Int32(4), Int32(row == 0))
        cute.arch.sync_threads()
        if tid < pairs:
            local = Int32(0)
            total = Int32(0)
            if cutlass.const_expr(self.input_reuse == 4):
                for warp in cutlass.range_constexpr(cute.size(ids) // 32):
                    mask = _ld_shared_i32(smem + (pairs + Int32(warp)) * Int32(4)).to(cutlass.Uint32)
                    total += cute.arch.popc(mask).to(Int32)
                    if Int32(warp) < first // Int32(32):
                        local += cute.arch.popc(mask).to(Int32)
                    elif Int32(warp) == first // Int32(32):
                        preceding = (cutlass.Uint32(1) << (first % Int32(32))) - cutlass.Uint32(1)
                        local += cute.arch.popc(mask & preceding).to(Int32)
            else:
                j = Int32(0)
                while j < pairs:
                    flag = _ld_shared_i32(smem + (pairs + j) * Int32(4))
                    total += flag
                    if j < first:
                        local += flag
                    j += Int32(1)
            if tid == Int32(0):
                active[Int32(0)] = total
            if row == Int32(0):
                counts[local] = count
                experts[local] = expert
                global_to_local[expert] = local
            tokens[local, row] = tid // topk
            route_weights[local, row] = weights[tid]
            metadata[tid, 0] = local
            metadata[tid, 1] = row

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
        sfb1_packed: cute.Tensor,   # packed FC1 scales; dummy off lane
        sfb2_packed: cute.Tensor,   # sf6 FC2 scales; dummy off lane
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
        if cutlass.const_expr(not self.reform_sf_pack):
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
            b_w13, self.b1_smem_layout_staged, (self.fc1_tile_n, self.fc1_tile_k), 1
        )
        if cutlass.const_expr(self.reform_sf_pack):
            # Typed placeholders only: all SFB descriptor use is compile-time
            # eliminated. No descriptor can retain freed original scales.
            tma_sfb_w13, gSFB_w13 = tma_sfa, gSFA
        else:
            tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
                sfb_w13_tensor, self.sfb1_smem_layout_staged, self.sfb1_tile_shape_nk, 1,
                internal_type=cutlass.Int16,
            )
        tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
            b_down, self.b2_smem_layout_staged, (self.fc2_tile_n, self.fc2_tile_k), 1
        )
        if cutlass.const_expr(self.reform_sf_pack):
            tma_sfb_down, gSFB_down = tma_sfa, gSFA
        else:
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
            sfb2_packed,
            b_w13,
            b_down,
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
        sfb1_packed: cute.Tensor,   # (E, blocks/expert, stage bytes) u8
        sfb2_packed: cute.Tensor,
        b_w13_raw: cute.Tensor,     # the weight storage itself: the l<n> prefetch addresses (unused off lane)
        b_down_raw: cute.Tensor,
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
        if cutlass.const_expr(not self.skip_a and not self.a_ring and not self.fc1_reuse_a):
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
            a_bars: cute.struct.MemRange[cutlass.Int64, self.a_barrier_count]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, self.scatter_cache_rows
            ]
            scatter_weight_cache: cute.struct.MemRange[
                cutlass.Float32, self.scatter_cache_rows
            ]
            # Zero elements in the unchanged lanes; native compile checks
            # those layouts too. Smaller scatter caches and header padding
            # hold the 3104-byte ring within a 2048-byte total increase.
            packed_fc1: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.sf6_packed_bytes], 16
            ]
            packed_fc2: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.sf2_packed_bytes], 16
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
                cute.struct.MemRange[self.sf_dtype, 0 if self.sf6_registers else cute.cosize(sfb1_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b2_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, 0 if self.sf6_registers else cute.cosize(sfb2_smem_staged)],
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
                cute.struct.MemRange[cutlass.BFloat16, 0 if self.fc2_prefetch else cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]

        if cutlass.const_expr(self.sf6_registers):
            assert Storage.__sizeof__() == self.smem_bytes
            assert cute.size_in_bytes(self.b_dtype, b1_smem_staged) >= cute.size_in_bytes(self.sf_dtype, sfb1_smem_staged)
            assert cute.size_in_bytes(self.b_dtype, b2_smem_staged) >= cute.size_in_bytes(self.sf_dtype, sfb2_smem_staged)
        storage = smem.allocate(Storage)
        if cutlass.const_expr(self.sf6_separate):
            sf1_input_base_addr = shared_ptr_to_u32(storage.packed_fc1.data_ptr())
        else:
            sf1_input_base_addr = shared_ptr_to_u32(storage.sSFB1.data_ptr())
        if cutlass.const_expr(self.compact_staging):
            sf2_input_base_addr = shared_ptr_to_u32(storage.packed_fc2.data_ptr())
        else:
            sf2_input_base_addr = shared_ptr_to_u32(storage.sSFB2.data_ptr())

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
            defer_sync=self.sync_cleanup,
        )
        fc2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.fc2_stages,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=fc2_tma_bytes,
            barrier_storage=storage.fc2_bars.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=self.sync_cleanup,
        )
        if cutlass.const_expr(not self.sync_cleanup):
            a_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.fc1_stages,
                producer_group=prod_group,
                consumer_group=cons_group,
                tx_count=a_tma_bytes,
                barrier_storage=storage.a_bars.data_ptr(),
                cta_layout_vmnk=cta_layout_vmnk,
            )
        else:
            # create(defer_sync=True) performs every mbarrier.init but skips
            # its fence/sync. One fence + the existing CTA sync publishes
            # BOTH rings. This kernel's cluster is exactly one CTA.
            cute.arch.mbarrier_init_fence()

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
        if cutlass.const_expr(self.sf6_registers):
            # Layout-only views create ordinary MMA register fragments. No
            # pointer in either view is read or written on this route. Use
            # existing B backing so the dead expanded-scale rings cost zero.
            sSFB1 = cute.make_tensor(cute.recast_ptr(storage.sB1.data_ptr(), dtype=self.sf_dtype), sfb1_smem_staged)
            sSFB2 = cute.make_tensor(cute.recast_ptr(storage.sB2.data_ptr(), dtype=self.sf_dtype), sfb2_smem_staged)
        else:
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
        if cutlass.const_expr(self.fc2_prefetch):
            # Layout-only view: direct scatter uses this solely to construct
            # accumulator/copy fragments. No sC load/store is reachable.
            sC = cute.make_tensor(cute.recast_ptr(storage.sB2.data_ptr(), dtype=cutlass.BFloat16),
                                 epi_smem_staged)
        else:
            sC = storage.sC.get_tensor(
                epi_smem_staged.outer, swizzle=epi_smem_staged.inner
            )
        sfa2_base_addr = shared_ptr_to_u32(storage.sSFA2.data_ptr())
        a2_base_addr = shared_ptr_to_u32(storage.sA2.data_ptr())
        sfb1_base_addr = shared_ptr_to_u32(storage.sSFB1.data_ptr())
        # bulk_b: the stage rings' byte bases (the fp4 pointers recast to bytes before the address is taken)
        sb1_base_addr = shared_ptr_to_u32(cute.recast_ptr(storage.sB1.data_ptr(), dtype=cutlass.Uint8))
        sb2_base_addr = shared_ptr_to_u32(cute.recast_ptr(storage.sB2.data_ptr(), dtype=cutlass.Uint8))
        sfb2_base_addr = shared_ptr_to_u32(storage.sSFB2.data_ptr())
        ctrl_base_addr = shared_ptr_to_u32(storage.ctrl.data_ptr())
        scatter_tok_base_addr = shared_ptr_to_u32(storage.scatter_tok_cache.data_ptr())
        if cutlass.const_expr(self.scatter_packed_load):
            scatter_smem_base_addr = shared_ptr_to_u32(storage.sC.data_ptr())
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
        if cutlass.const_expr(self.input_reuse):
            # Compact expert IDs are < routed rows. The host proves this tail
            # cannot overlap any active expert's packed input, even if every
            # routed slot selects a different expert. No new allocation or
            # barrier: phase 0 publishes the cache at the existing grid fence.
            reuse_bytes = (Int32(0) if cutlass.const_expr(self.input_reuse == 4) else
                           num_tokens * (cols // Int32(2) + sf_blocks_per_row))
            if cutlass.const_expr(self.input_reuse in (3, 4)):
                reuse_bytes += total_pairs * Int32(8)
            reuse_base = packed_a_storage.iterator + (Int32(packed_a_storage.shape[0]) - reuse_bytes)
            reuse_layout = cute.make_layout((num_tokens, sf_blocks_per_row), stride=(sf_blocks_per_row, 1))
            reuse_packed = cute.make_tensor(cute.recast_ptr(reuse_base, dtype=Uint64), reuse_layout)
            reuse_scales = cute.make_tensor(reuse_base + num_tokens * (cols // Int32(2)), reuse_layout)
            if cutlass.const_expr(self.input_reuse in (3, 4)):
                route_base = reuse_base
                if cutlass.const_expr(self.input_reuse != 4):
                    route_base += num_tokens * (cols // Int32(2) + sf_blocks_per_row)
                reuse_routes = cute.make_tensor(cute.recast_ptr(route_base, dtype=Int32),
                    cute.make_layout((total_pairs, 2), stride=(2, 1)))

        # ------------------------------------------------------------------
        # Phase 0 / Phase 1 (stock frontend)
        # ------------------------------------------------------------------
        if cutlass.const_expr(self.input_reuse not in (3, 4)):
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
        # Each route/128-wide intermediate part owns a complete output row.
        # All routes, including zero weights, are computed below. The private
        # reduction runs after this kernel, so no clear/atomic scatter is needed.
        scatter_total = Int32(0) if cutlass.const_expr(self.route_scatter) else num_tokens * cols
        j = flat_tid
        while j < scatter_total:
            if cutlass.const_expr(self.scatter_fp32):
                scatter_output[j // cols, j % cols] = cutlass.Float32(0.0)
            else:
                scatter_output[j // cols, j % cols] = cutlass.BFloat16(0.0)
            j += flat_stride
        if cutlass.const_expr(self.input_reuse in (1, 2, 3)):
            reuse_idx = flat_tid
            if cutlass.const_expr(self.input_reuse in (2, 3)):
                reuse_idx = (Int32(tidx) // Int32(32) * Int32(gdim_z) + Int32(bidz)) * Int32(32) + Int32(tidx) % Int32(32)
            while reuse_idx < num_tokens * sf_blocks_per_row:
                token_idx = reuse_idx // sf_blocks_per_row
                block_idx = reuse_idx % sf_blocks_per_row
                reference_expert = topk_ids[token_idx * num_topk].to(Int32)
                gs = input_global_scale[reference_expert].to(cutlass.Float32)
                if self.input_scales_are_reciprocal and gs != cutlass.Float32(0.0):
                    if self.fast_math:
                        gs = rcp_approx_ftz(gs)
                    else:
                        gs = cutlass.Float32(1.0) / gs
                values = cute.make_rmem_tensor((self.sf_vec_size,), cutlass.Float32)
                loaded = load_global_bf16x16_to_f32x16(get_ptr_as_int64(
                    a_input, token_idx * cols + block_idx * Int32(self.sf_vec_size)))
                block_max = cutlass.Float32(0.0)
                for elem_idx in cutlass.range_constexpr(self.sf_vec_size):
                    values[elem_idx] = loaded[elem_idx]
                    block_max = fmax_f32(block_max, fabs_f32(loaded[elem_idx]))
                packed = Uint64(0)
                scale = Uint8(0)
                if self.fast_math:
                    packed, scale = quantize_block_fp4_fast(values, block_max, gs)
                else:
                    packed, scale = quantize_block_fp4(values, block_max, gs)
                reuse_packed[token_idx, block_idx] = packed
                reuse_scales[token_idx, block_idx] = scale
                reuse_idx += flat_stride
        if cutlass.const_expr(self.input_reuse in (3, 4)):
            assert cute.size_in_bytes(self.b_dtype, b2_smem_staged) >= 8 * cute.size(topk_ids)
            if Int32(bidz) == Int32(0):
                # FC2 weights have not been loaded yet. Its stage storage
                # exists in both cells, unlike C2's eliminated sC buffer.
                self._prepare_token_routes(Int32(tidx), shared_ptr_to_u32(storage.sB2.data_ptr()),
                    topk_ids, topk_weights, row_counts, active_expert_count,
                    weight_expert_ids, global_to_local_expert, token_map, token_weights,
                    reuse_routes, num_topk)
                # Retire generic shared accesses before the async TMA proxy
                # overwrites this stage. The existing CTA barrier follows.
                cute.arch.fence_proxy("async.shared", space="cta")
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

        if cutlass.const_expr(self.input_reuse == 4):
            # Keep one quantized block in registers and fan it out directly.
            # Only the route metadata crosses the first grid barrier; there is
            # no global quantization cache to write and reread eight times.
            reuse_idx = (Int32(tidx) // Int32(32) * Int32(gdim_z) + Int32(bidz)) * Int32(32) + Int32(tidx) % Int32(32)
            while reuse_idx < num_tokens * sf_blocks_per_row:
                token_idx = reuse_idx // sf_blocks_per_row
                sf_idx = reuse_idx % sf_blocks_per_row
                first_pair = token_idx * num_topk
                reference_expert = topk_ids[first_pair].to(Int32)
                raw_gs = input_global_scale[reference_expert].to(cutlass.Float32)
                gs = raw_gs
                if self.input_scales_are_reciprocal and gs != cutlass.Float32(0.0):
                    if self.fast_math:
                        gs = rcp_approx_ftz(gs)
                    else:
                        gs = cutlass.Float32(1.0) / gs
                values = cute.make_rmem_tensor((self.sf_vec_size,), cutlass.Float32)
                loaded = load_global_bf16x16_to_f32x16(get_ptr_as_int64(
                    a_input, token_idx * cols + sf_idx * Int32(self.sf_vec_size)))
                block_max = cutlass.Float32(0.0)
                for elem_idx in cutlass.range_constexpr(self.sf_vec_size):
                    values[elem_idx] = loaded[elem_idx]
                    block_max = fmax_f32(block_max, fabs_f32(loaded[elem_idx]))
                packed = Uint64(0)
                scale = Uint8(0)
                if self.fast_math:
                    packed, scale = quantize_block_fp4_fast(values, block_max, gs)
                else:
                    packed, scale = quantize_block_fp4(values, block_max, gs)
                route = Int32(0)
                while route < num_topk:
                    pair_idx = first_pair + route
                    expert_id = topk_ids[pair_idx].to(Int32)
                    local_expert_id = reuse_routes[pair_idx, 0]
                    row = reuse_routes[pair_idx, 1]
                    packed_lo = packed
                    scale_byte = scale
                    other_gs = input_global_scale[expert_id].to(cutlass.Float32)
                    if other_gs != raw_gs:
                        if self.input_scales_are_reciprocal and other_gs != cutlass.Float32(0.0):
                            if self.fast_math:
                                other_gs = rcp_approx_ftz(other_gs)
                            else:
                                other_gs = cutlass.Float32(1.0) / other_gs
                        if self.fast_math:
                            packed_lo, scale_byte = quantize_block_fp4_fast(values, block_max, other_gs)
                        else:
                            packed_lo, scale_byte = quantize_block_fp4(values, block_max, other_gs)
                    output_offset = (local_expert_id * max_rows * output_bytes_per_row
                                     + row * output_bytes_per_row + sf_idx * Int32(self.sf_vec_size // 2))
                    st_global_u64(get_ptr_as_int64(packed_a_storage, output_offset), packed_lo)
                    scale_offset = (local_expert_id * expert_scale_stride
                        + (row // Int32(128)) * num_k_tiles * Int32(512)
                        + (sf_idx // Int32(4)) * Int32(512)
                        + (row % Int32(32)) * Int32(16)
                        + ((row % Int32(128)) // Int32(32)) * Int32(4) + sf_idx % Int32(4))
                    scale_storage[scale_offset] = scale_byte
                    route += Int32(1)
                reuse_idx += flat_stride
        else:
            pair_idx = Int32(bidz)
            while pair_idx < total_pairs:
                expert_id = topk_ids[pair_idx].to(Int32)
                token_idx = pair_idx // num_topk
                weight = topk_weights[pair_idx].to(cutlass.Float32)
                local_expert_id = Int32(0)
                row = Int32(0)
                if cutlass.const_expr(self.input_reuse in (3, 4)):
                    local_expert_id = reuse_routes[pair_idx, 0]
                    row = reuse_routes[pair_idx, 1]
                else:
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
                        scatter_row = pair_idx if cutlass.const_expr(self.route_scatter) else token_idx
                        st_global_i32(get_ptr_as_int64(token_map, map_idx), scatter_row)
                        st_global_f32(get_ptr_as_int64(token_weights, map_idx), weight)
                        _st_shared_i32(ctrl_base_addr + Int32(0), local_expert_id)
                        _st_shared_i32(ctrl_base_addr + Int32(4), row)
                    cute.arch.sync_threads()
                    local_expert_id = _ld_shared_i32(ctrl_base_addr + Int32(0))
                    row = _ld_shared_i32(ctrl_base_addr + Int32(4))

                gs_value = input_global_scale[expert_id].to(cutlass.Float32)
                reuse_this = Int32(0)
                if cutlass.const_expr(self.input_reuse):
                    reference_expert = topk_ids[token_idx * num_topk].to(Int32)
                    reuse_this = Int32(gs_value == input_global_scale[reference_expert].to(cutlass.Float32))
                if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(0.0):
                    if self.fast_math:
                        gs_value = rcp_approx_ftz(gs_value)
                    else:
                        gs_value = cutlass.Float32(1.0) / gs_value
                sf_idx = Int32(tidx)
                while sf_idx < sf_blocks_per_row:
                    scale_byte = Uint8(0)
                    packed_lo = Uint64(0)
                    if reuse_this != Int32(0):
                        if cutlass.const_expr(self.input_reuse):
                            packed_lo = reuse_packed[token_idx, sf_idx].to(Uint64)
                            scale_byte = reuse_scales[token_idx, sf_idx].to(Uint8)
                    else:
                        block_start = sf_idx * Int32(self.sf_vec_size)
                        values = cute.make_rmem_tensor((self.sf_vec_size,), cutlass.Float32)
                        block_max = cutlass.Float32(0.0)
                        if cutlass.const_expr(self.input_vec16):
                            loaded = load_global_bf16x16_to_f32x16(
                                get_ptr_as_int64(a_input, token_idx * cols + block_start))
                        for elem_idx in cutlass.range_constexpr(self.sf_vec_size):
                            if cutlass.const_expr(self.input_vec16):
                                value = loaded[elem_idx]
                            else:
                                value = cutlass.Float32(a_input[token_idx, block_start + Int32(elem_idx)])
                            values[elem_idx] = value
                            block_max = fmax_f32(block_max, fabs_f32(value))
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

                if cutlass.const_expr(self.input_reuse not in (3, 4)):
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
        if cutlass.const_expr(not self.sync_cleanup):
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
            if cutlass.const_expr(self.direct_scatter):
                ep_identity = cute.make_identity_tensor((*self.epi_tile, 1))
                ep_tRS_coords = thr_copy_r2s.partition_D(ep_identity)
                ep_coords = ep_tRS_coords[None, None, None, 0]
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
                if cutlass.const_expr(self.fc1_reuse_a):
                    self._validate_fc1_reuse_fragment(tCrA1, "A")
                    self._validate_fc1_reuse_fragment(tCrSFA1_tile, "SFA")
                valid_tile_rows = valid_rows - tile_m_base
                if valid_tile_rows > Int32(self.tile_m):
                    valid_tile_rows = Int32(self.tile_m)
                if valid_tile_rows < Int32(0):
                    valid_tile_rows = Int32(0)

                cache_row = Int32(tidx)
                if cache_row < Int32(self.scatter_cache_rows):
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
                                if cutlass.const_expr(self.reform_sf_pack and not self.sf6_registers):
                                    for sf_block in cutlass.range_constexpr(self.sf1_packed_blocks):
                                        sf1_dest = (sfb1_base_addr
                                            + fc1_cons_state.index * Int32(self.sf1_block_bytes)
                                            + Int32(sf_block * 2048))
                                        if cutlass.const_expr(self.sf6_separate):
                                            self._sf_expand_stage(sf1_dest, Int32(tidx), 2048,
                                                packed_addr=sf1_input_base_addr
                                                    + fc1_cons_state.index * Int32(self.sf1_stage_bytes))
                                        else:
                                            self._sf_expand_stage(sf1_dest, Int32(tidx), 2048)
                                elif cutlass.const_expr(self.sf_pack):
                                    self._sf_expand_stage(
                                        sfb1_base_addr
                                        + fc1_cons_state.index * Int32(self.sf1_block_bytes),
                                        Int32(tidx), self.sf1_block_bytes,
                                    )
                                if cutlass.const_expr(self.a_ring):
                                    a_slot = a_cons_state.index
                                else:
                                    a_slot = self._fc1_input_slot(fc1_cons_state.index)
                                csA_p = csA1[None, None, None, a_slot]
                                csB_p = csB1[None, None, None, fc1_cons_state.index]
                                fz_csSFA_p = cute.filter_zeros(
                                    csSFA1_tile[None, None, None, a_slot]
                                )
                                fz_csSFB_p = cute.filter_zeros(
                                    csSFB1[None, None, None, fc1_cons_state.index]
                                )
                                if cutlass.const_expr(not self.fc1_reuse_a or gu == 0):
                                    cute.copy(smem_copy_A1, csA_p[None, None, 0], crA1[None, None, 0])
                                cute.copy(smem_copy_B1, csB_p[None, None, 0], crB1[None, None, 0])
                                if cutlass.const_expr(not self.fc1_reuse_a or gu == 0):
                                    cute.copy(
                                        smem_copy_SFA1, fz_csSFA_p[None, None, 0],
                                        fz_crSFA1_tile[None, None, 0],
                                    )
                                if cutlass.const_expr(self.sf6_registers):
                                    sf1_register_stage = self._sf6_prepare_stage(
                                        sf1_input_base_addr + fc1_cons_state.index * Int32(self.sf1_stage_bytes), tidx)
                                    self._sf6_load_fragment(fz_crSFB1[None, None, 0], sf1_register_stage, "fc1", 0)
                                else:
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
                                        if cutlass.const_expr(not self.fc1_reuse_a or gu == 0):
                                            cute.copy(
                                                smem_copy_A1, csA_p[None, None, k_next],
                                                crA1[None, None, k_next],
                                            )
                                        cute.copy(
                                            smem_copy_B1, csB_p[None, None, k_next],
                                            crB1[None, None, k_next],
                                        )
                                        if cutlass.const_expr(not self.fc1_reuse_a or gu == 0):
                                            cute.copy(
                                                smem_copy_SFA1, fz_csSFA_p[None, None, k_next],
                                                fz_crSFA1_tile[None, None, k_next],
                                            )
                                        if cutlass.const_expr(self.sf6_registers):
                                            self._sf6_load_fragment(fz_crSFB1[None, None, k_next],
                                                sf1_register_stage, "fc1", k_next)
                                        else:
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
                            if cutlass.const_expr(self.fc2_scale_search > 0):
                                packed_lo, scale_byte = quantize_block_fp4_search(
                                    values, block_max, gs_value, self.fc2_scale_search, self.fast_math
                                )
                            elif self.fast_math:
                                packed_lo, scale_byte = quantize_block_fp4_fast(
                                    values, block_max, gs_value
                                )
                            else:
                                packed_lo, scale_byte = quantize_block_fp4(
                                    values, block_max, gs_value
                                )
                            packed_base = sf_block * Int32(self.sf_vec_size // 2)
                            if cutlass.const_expr(self.decode_reform):
                                self._store_packed_activation(
                                    a2_base_addr, a2_smem_layout, row, packed_base, packed_lo)
                            else:
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
                        # The M16 reform tile (C1 rows, C2 batch) has only one
                        # half. Its final fence + publication barrier below
                        # already protects A2/SFA2 reads, C2's retained route
                        # metadata and completion of sC1 reads before the next
                        # work item.
                        # Stamped runs retain the earlier completion point:
                        # stamp +1 must not precede another warp's last write.
                        if cutlass.const_expr(not self.sync_cleanup or self.stamps):
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
                if cutlass.const_expr(self.scatter_reuse):
                    # The completed FC1 publication barrier also publishes
                    # route metadata. Each MMA lane owns only two output
                    # rows; retain their destination/weight for the sweep.
                    ep_bases = cute.make_rmem_tensor((2,), Int32)
                    ep_weights = cute.make_rmem_tensor((2,), cutlass.Float32)
                    for row_slot in cutlass.range_constexpr(2):
                        cached_ep_row = Int32(ep_coords[2 * self.scatter_row_pairs[row_slot]][0])
                        ep_bases[row_slot] = Int32(0)
                        ep_weights[row_slot] = cutlass.Float32(0.0)
                        if cached_ep_row < valid_tile_rows:
                            ep_bases[row_slot] = _ld_shared_i32_volatile(
                                scatter_tok_base_addr + cached_ep_row * Int32(4)) * scatter_N
                            ep_weights[row_slot] = _ld_shared_i32_volatile(
                                scatter_weight_base_addr + cached_ep_row * Int32(4)).bitcast(cutlass.Float32)
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
                    if cutlass.const_expr(self.reform_sf_pack and not self.sf6_registers):
                        if cutlass.const_expr(self.decode_reform):
                            sf2_packed_addr = None
                            if cutlass.const_expr(self.compact_staging):
                                sf2_packed_addr = (sf2_input_base_addr
                                    + fc2_cons_state.index * Int32(self.sf2_stage_bytes))
                            self._sf_expand_stage(
                                sfb2_base_addr + fc2_cons_state.index * Int32(2048),
                                Int32(tidx), 2048,
                                packed_addr=sf2_packed_addr,
                                word_expand=self.sf6_fc2_word_expand,
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
                    if cutlass.const_expr(self.sf6_registers):
                        sf2_register_stage = self._sf6_prepare_stage(
                            sf2_input_base_addr + fc2_cons_state.index * Int32(self.sf2_stage_bytes), tidx)
                        self._sf6_load_fragment(fz_crSFB2[None, None, 0], sf2_register_stage, "fc2", 0)
                    else:
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
                            if cutlass.const_expr(self.sf6_registers):
                                self._sf6_load_fragment(fz_crSFB2[None, None, k_next],
                                    sf2_register_stage, "fc2", k_next)
                            else:
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
                    if cutlass.const_expr(self.direct_scatter):
                        for ep_pair in cutlass.range_constexpr(cute.size(tRS_rD_out) // 2):
                            ep_coord = ep_coords[2 * ep_pair]
                            ep_row = Int32(ep_coord[0])
                            if ep_row < valid_tile_rows:
                                if cutlass.const_expr(self.scatter_reuse):
                                    ep_base = ep_bases[self.scatter_pair_rows[ep_pair]]
                                    ep_weight = ep_weights[self.scatter_pair_rows[ep_pair]]
                                else:
                                    ep_tok = ld_shared_i32_relaxed(scatter_tok_base_addr + ep_row * Int32(4))
                                    ep_weight = _ld_shared_f32(scatter_weight_base_addr + ep_row * Int32(4))
                                    if cutlass.const_expr(self.route_scatter):
                                        ep_tok = ep_tok * Int32(self.output_tile_count_n) + Int32(tile_coord[1])
                                    ep_base = ep_tok * scatter_N
                                ep_v0 = cutlass.Float32(tRS_rD_out[2 * ep_pair])
                                ep_v1 = cutlass.Float32(tRS_rD_out[2 * ep_pair + 1])
                                ep_ptr = get_ptr_as_int64(
                                    scatter_output, ep_base + tile_n_base_cur + Int32(ep_coord[1]))
                                if cutlass.const_expr(self.route_scatter):
                                    scatter_store_bf16x2_to_f32(ep_ptr, ep_weight * ep_v0, ep_weight * ep_v1)
                                else:
                                    scatter_add_bf16x2_to_f32(ep_ptr, ep_weight * ep_v0, ep_weight * ep_v1)
                    else:
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
                            if cutlass.const_expr(self.scatter_packed_load):
                                # sC's outer layout counts BF16 elements. Its
                                # pointer swizzle applies to byte addresses.
                                sc_offset = Int32(sC.layout((cached_row, local_col, 0)))
                                sc_addr = scatter_smem_base_addr + self._scatter_smem_byte_offset(sc_offset)
                                scatter_add_bf16x8_from_smem_to_f32(
                                    get_ptr_as_int64(scatter_output, tok * scatter_N + global_col),
                                    sc_addr, wv)
                            else:
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
                                if cutlass.const_expr(self.route_scatter):
                                    route_row = tok * Int32(self.output_tile_count_n) + Int32(tile_coord[1])
                                    out_ptr = get_ptr_as_int64(scatter_output, route_row * scatter_N + global_col)
                                    scatter_store_bf16x2_to_f32(out_ptr + Int64(0), wv * sc_v0, wv * sc_v1)
                                    scatter_store_bf16x2_to_f32(out_ptr + Int64(8), wv * sc_v2, wv * sc_v3)
                                    scatter_store_bf16x2_to_f32(out_ptr + Int64(16), wv * sc_v4, wv * sc_v5)
                                    scatter_store_bf16x2_to_f32(out_ptr + Int64(24), wv * sc_v6, wv * sc_v7)
                                elif cutlass.const_expr(self.scatter_fp32):
                                    if cutlass.const_expr(self.scatter_vec4):
                                        scatter_add_bf16x4_to_f32(
                                            get_ptr_as_int64(scatter_output, tok * scatter_N + global_col),
                                            wv * sc_v0, wv * sc_v1, wv * sc_v2, wv * sc_v3)
                                        scatter_add_bf16x4_to_f32(
                                            get_ptr_as_int64(scatter_output, tok * scatter_N + global_col + Int32(4)),
                                            wv * sc_v4, wv * sc_v5, wv * sc_v6, wv * sc_v7)
                                    else:
                                        scatter_add_bf16x2_to_f32(
                                            get_ptr_as_int64(scatter_output, tok * scatter_N + global_col + Int32(0)),
                                            wv * sc_v0, wv * sc_v1)
                                        scatter_add_bf16x2_to_f32(
                                            get_ptr_as_int64(scatter_output, tok * scatter_N + global_col + Int32(2)),
                                            wv * sc_v2, wv * sc_v3)
                                        scatter_add_bf16x2_to_f32(
                                            get_ptr_as_int64(scatter_output, tok * scatter_N + global_col + Int32(4)),
                                            wv * sc_v4, wv * sc_v5)
                                        scatter_add_bf16x2_to_f32(
                                            get_ptr_as_int64(scatter_output, tok * scatter_N + global_col + Int32(6)),
                                            wv * sc_v6, wv * sc_v7)
                                else:
                                    scatter_add_v4_bf16x2(
                                        get_ptr_as_int64(
                                            scatter_output, tok * scatter_N + global_col
                                        ),
                                        wv * sc_v0, wv * sc_v1, wv * sc_v2, wv * sc_v3,
                                        wv * sc_v4, wv * sc_v5, wv * sc_v6, wv * sc_v7,
                                    )
                            vec_idx += Int32(self.num_threads_per_warp)
                    if cutlass.const_expr(not self.scatter_reuse):
                        self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.scatter_reuse):
                    # B/SFB ring reuse is governed by the FC2 pipeline.
                    # Direct output has no per-column shared epilogue. One
                    # retirement barrier protects A2 and route metadata before
                    # any MMA warp can begin the next work item.
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
            if cutlass.const_expr(self.l2_prefetch > 0 or self.bulk_b):
                # Tile-major storage (rows, K_in, K_tiles, E), K-major within the chunk, K_in == the
                # kernel's K tile (the dispatcher checks): a (tile rows x K_in) box is one contiguous run
                # at base + e * expert + k_tile * ktile + n_tile * box bytes. Row and box bytes are the
                # tile's own constants; the byte view of the fp4 storage gives the base address.
                w13_base = get_ptr_as_int64(cute.recast_tensor(b_w13_raw, cutlass.Uint8), Int32(0))
                w2_base = get_ptr_as_int64(cute.recast_tensor(b_down_raw, cutlass.Uint8), Int32(0))
                fc1_box_i64 = Int64(self.fc1_tile_n * (self.fc1_tile_k // 2))
                fc2_box_i64 = Int64(self.fc2_tile_n * (self.fc2_tile_k // 2))
                fc1_box_bytes = Int32(self.fc1_tile_n * (self.fc1_tile_k // 2))
                fc2_box_bytes = Int32(self.fc2_tile_n * (self.fc2_tile_k // 2))
                w13_ktile_bytes = Int64(cute.size(b_w13_raw.shape[0])) * Int64(self.fc1_tile_k // 2)
                w13_expert_bytes = w13_ktile_bytes * Int64(cute.size(b_w13_raw.shape[2]))
                w2_ktile_bytes = Int64(cute.size(b_down_raw.shape[0])) * Int64(self.fc2_tile_k // 2)
                w2_expert_bytes = w2_ktile_bytes * Int64(cute.size(b_down_raw.shape[2]))
                prefetch_ahead = Int32(self.l2_prefetch)
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
                            if cutlass.const_expr(self.l2_prefetch > 0):
                                # the gate and up boxes n stages ahead, then -- past FC1's end -- the
                                # item's first FC2 boxes, so the stream of requests runs on across
                                # the FC1/FC2 seam of the same item
                                if is_dma_lane0:
                                    ahead = k_tile + prefetch_ahead
                                    if ahead < k_tile_cnt1:
                                        if cutlass.const_expr(self.l2_prefetch_fc1):
                                            w13_at = (w13_base + Int64(weight_expert_idx) * w13_expert_bytes
                                                      + Int64(ahead) * w13_ktile_bytes)
                                            _bulk_prefetch_l2(w13_at + Int64(gate_tile) * fc1_box_i64,
                                                              fc1_box_bytes)
                                            _bulk_prefetch_l2(w13_at + Int64(up_tile) * fc1_box_i64,
                                                              fc1_box_bytes)
                                    else:
                                        down_tile = ahead - k_tile_cnt1
                                        if down_tile < output_tile_cnt:
                                            _bulk_prefetch_l2(
                                                w2_base + Int64(weight_expert_idx) * w2_expert_bytes
                                                + Int64(intermediate_slice) * w2_ktile_bytes
                                                + Int64(down_tile) * fc2_box_i64,
                                                fc2_box_bytes)
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
                                if cutlass.const_expr(self.fc1_reuse_a and not self.skip_a and gu == 0):
                                    # The base arrival expects B/SFB (>0 bytes).
                                    # Add A/SFA exactly once BEFORE any TMA issue;
                                    # up then needs no input DMA or separate A ring.
                                    if is_dma_lane0:
                                        cute.arch.mbarrier_expect_tx(bar, a_tma_bytes)
                                if cutlass.const_expr(not self.skip_a and not self.a_ring
                                                      and (not self.fc1_reuse_a or gu == 0)):
                                    cute.copy(
                                        tma_a, tAgA_mk[(None, k_tile)],
                                        tAsA[(None, self._fc1_input_slot(fc1_prod_state.index))], tma_bar_ptr=bar,
                                    )
                                if cutlass.const_expr(self.bulk_b):
                                    # one bulk copy of the pre-swizzled 16 KB box; the bytes and the barrier
                                    # transaction are exactly the TMA box's
                                    if is_dma_lane0:
                                        if cutlass.const_expr(gu == 0):
                                            b_tile = gate_tile
                                        else:
                                            b_tile = up_tile
                                        _bulk_g2s(
                                            sb1_base_addr + fc1_prod_state.index * fc1_box_bytes,
                                            w13_base + Int64(weight_expert_idx) * w13_expert_bytes
                                            + Int64(k_tile) * w13_ktile_bytes + Int64(b_tile) * fc1_box_i64,
                                            fc1_box_bytes, shared_ptr_to_u32(bar))
                                elif cutlass.const_expr(gu == 0):
                                    cute.copy(
                                        tma_b_w13, tBgB_gate_nk[(None, k_tile)],
                                        tBsB1[(None, fc1_prod_state.index)], tma_bar_ptr=bar,
                                    )
                                else:
                                    cute.copy(
                                        tma_b_w13, tBgB_up_nk[(None, k_tile)],
                                        tBsB1[(None, fc1_prod_state.index)], tma_bar_ptr=bar,
                                    )
                                if cutlass.const_expr(not self.skip_a and not self.a_ring
                                                      and (not self.fc1_reuse_a or gu == 0)):
                                    cute.copy(
                                        tma_sfa, tAgSFA_mk[(None, k_tile)],
                                        tAsSFA[(None, self._fc1_input_slot(fc1_prod_state.index))], tma_bar_ptr=bar,
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
                                                if cutlass.const_expr(self.sf6_separate):
                                                    sf1_dest = (sf1_input_base_addr
                                                        + fc1_prod_state.index * Int32(self.sf1_stage_bytes))
                                                else:
                                                    sf1_dest = (sfb1_base_addr
                                                        + fc1_prod_state.index * Int32(self.sf1_block_bytes)
                                                        + Int32(sf_block * 2048))
                                                _bulk_g2s(
                                                    sf1_dest,
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
                    if cutlass.const_expr(self.l2_prefetch > 0):
                        if is_dma_lane0:
                            down_ahead = output_tile_idx + prefetch_ahead
                            if down_ahead < output_tile_cnt:
                                _bulk_prefetch_l2(
                                    w2_base + Int64(weight_expert_idx) * w2_expert_bytes
                                    + Int64(intermediate_slice) * w2_ktile_bytes
                                    + Int64(down_ahead) * fc2_box_i64,
                                    fc2_box_bytes)
                    fc2_pipeline.producer_acquire(fc2_prod_state)
                    bar2 = fc2_pipeline.producer_get_barrier(fc2_prod_state)
                    if cutlass.const_expr(self.bulk_b):
                        if is_dma_lane0:
                            _bulk_g2s(
                                sb2_base_addr + fc2_prod_state.index * fc2_box_bytes,
                                w2_base + Int64(weight_expert_idx) * w2_expert_bytes
                                + Int64(intermediate_slice) * w2_ktile_bytes
                                + Int64(output_tile_idx) * fc2_box_i64,
                                fc2_box_bytes, shared_ptr_to_u32(bar2))
                    else:
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
                                if cutlass.const_expr(self.compact_staging):
                                    sf2_dest = (sf2_input_base_addr
                                        + fc2_prod_state.index * Int32(self.sf2_stage_bytes))
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


__all__ = ["MoEStaticKernelV4"]
