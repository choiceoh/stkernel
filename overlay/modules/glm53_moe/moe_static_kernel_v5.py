"""
MoEStaticKernelV5 -- v4 over TILE-MAJOR expert weights (spec cell ``t``).

What changes and why (39차): the v4 kernel streams w13 as TMA boxes of 64
rows x 256 B, and every one of those 64 rows sits 2,048 B from the next in
the stock [E, N, K] row-major storage -- DRAM sees 64 separate 256 B
segments per box. The vector-load streamer (35차 §6) rated that access class
at 222 GB/s against 239 for a linear read, and the kernel's own stamps put
FC1 at 200-206 GB/s (38차 §3, §5) while FC2 -- whose four slice CTAs happen
to cover whole 256 B w2 rows between them -- streams at 237. Widening the
box further is blocked by smem (38차 §7: 512 B segments need 32-row N tiles
and waste the 128-row SF boxes).

The re-layout removes the stride instead of widening the box: the dispatcher
stores each expert's w13 as [K/512 k-tiles][N rows][256 B] and w2 as
[I_tp/128 k-tiles][H rows][64 B] (``moe_dispatch._get_weight_views(tiled=
True)``), so a (64 rows x 512 K) box is one contiguous 16 KB run and a
(128 rows x 128 K) down box one contiguous 8 KB run. The TMA descriptor is
4-D -- (K_in, rows, k_tile, expert) with strides (1, 256 B, N x 256 B,
K x N / 2 B) -- which this class builds by grouping the two K modes of the
4-D tensors the dispatcher hands over into one hierarchical K mode:
``local_tile`` with the v4 tile shapes then produces exactly v4's
(box, n_tile, k_tile, expert) coordinates, and the kernel body is v4's,
untouched (same tiles, same smem layouts, same MMA order: results are
bit-identical to v4 up to the bf16 atomic scatter order). The SF tensors keep
their stock layout: the SM120 block-scaled layout already stores a 128-row
block's scales for one 512-wide k tile as a contiguous 4 KB.

Cost of the layout: the storage is permuted once when the weight views are
built (a copy in the probe; in-place at weight post-processing in serving),
and every consumer of the expert weights must read the tiled layout -- the
stock static kernel and the dynamic (prefill) kernel included, which the
dispatcher enforces (a tiled view never reaches a kernel compiled for the
row-major layout).

Selected by spec cell ``t`` (v4 geometry: tile_m 32, FC1 K-512 gate/up
stages, FC2 2 stages); ``v,t`` adds the A ring, ``t,s`` the stamps.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from .moe_static_kernel_v4 import (
    MoEStaticKernelV4,
    _FC1_TILE_K,
    _FC1_TILE_N,
    _FC2_TILE_K,
    _FC2_TILE_N,
)


# The tiled storage's inner chunk per row, in fp4 elements: one FC1 / FC2 k
# tile. The dispatcher's re-layout and the 4-D fake tensors it compiles
# against use these two numbers; a kernel tile that did not divide them
# would silently read the wrong bytes, so the kernel pins them here.
TILED_W13_K_IN = _FC1_TILE_K   # 512 fp4 = 256 B per row per k tile
TILED_W2_K_IN = _FC2_TILE_K    # 128 fp4 = 64 B per row per k tile


class MoEStaticKernelV5(MoEStaticKernelV4):
    """v4 whose weight tensors arrive tile-major: b_w13 as (N, 512, K/512, E)
    and b_down as (H, 128, I_tp/128, E) fp4, K-major within the chunk."""

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
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = packed_a.element_type
        self.b_dtype = b_w13.element_type
        self.sf_dtype = sfa_ptr.dtype
        self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
        # K_in is the stride-1 mode of the 4-D tensor: K-major B, as v4's
        self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        hidden_size = a_input.shape[1]
        self._setup_attributes(hidden_size=hidden_size)

        # the scale tensors are laid out for the flat (rows, K, E) shape --
        # their storage is the stock one -- so their layouts come from the
        # flat shape, not from the tiled weight tensor's
        w13_rows = b_w13.shape[0]
        w13_k = b_w13.shape[1] * b_w13.shape[2]
        w13_e = b_w13.shape[3]
        down_rows = b_down.shape[0]
        down_k = b_down.shape[1] * b_down.shape[2]
        down_e = b_down.shape[3]
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            packed_a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_w13_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (w13_rows, w13_k, w13_e), self.sf_vec_size
        )
        sfb_w13_tensor = cute.make_tensor(sfb_w13_ptr, sfb_w13_layout)
        sfb_down_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (down_rows, down_k, down_e), self.sf_vec_size
        )
        sfb_down_tensor = cute.make_tensor(sfb_down_ptr, sfb_down_layout)

        # (N, K_in, K_tiles, E) -> (N, (K_in, K_tiles), E): one hierarchical
        # K mode whose inner extent is the k tile, so the v4 tile shapes
        # divide it and the TMA map's innermost box dim is the contiguous
        # 256 B / 64 B chunk with the row stride right behind it
        b_w13_h = cute.group_modes(b_w13, 1, 3)
        b_down_h = cute.group_modes(b_down, 1, 3)

        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a, self.a1_smem_layout_staged, self.sa1_tile_shape_mk, 1
        )
        tma_sfa, gSFA = self._dense_cls._make_tma_atoms_and_tensors(
            sfa_tensor, self.sfa1_smem_layout_staged, self.sfa1_tile_shape_mk, 1,
            internal_type=cutlass.Int16,
        )
        tma_b_w13, gB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            b_w13_h, self.b1_smem_layout_staged, (_FC1_TILE_N, _FC1_TILE_K), 1
        )
        tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_w13_tensor, self._sfb1_tma_smem_layout(), self.sfb1_tile_shape_nk, 1,
            internal_type=cutlass.Int16,
        )
        tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
            b_down_h, self.b2_smem_layout_staged, (_FC2_TILE_N, _FC2_TILE_K), 1
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
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            cooperative=True,
            stream=stream,
        )


__all__ = ["MoEStaticKernelV5", "TILED_W13_K_IN", "TILED_W2_K_IN"]
