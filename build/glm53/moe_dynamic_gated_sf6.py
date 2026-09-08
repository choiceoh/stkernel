# SPDX-License-Identifier: Apache-2.0
"""Direct SF6 expert-scale loads for the stock tiled dynamic prefill kernel.

Provenance: FlashInfer _moe_dynamic/gated.py from immutable image
sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211,
source SHA256 993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445.
The vendor file remains byte-identical for the independent #368 contract.

Only __call__/kernel are forked to carry packed tensors and remove raw SFB
TMA descriptors/transaction bytes. Q0/Q1, queue ownership, gate/up MMA order,
tail handling, consumer release, FC2 reduction and atomic scatter are inherited.
The producer warp expands each physical N128/K128 scale tile (1024 bytes)
from the existing 2048-to-1552 SF6 encoding directly into its existing shared
stage. There is no global raw-scale allocation, reconstruction or descriptor.
FC1 selects one K128 half; FC2 selects one N128 half across both K64 groups.

A producer first waits for the old consumer to release the stage. After
ordinary shared stores, warp synchronization precedes the TMA full-barrier
release arrival. Only A/B/SFA DMA bytes are then expected. A consumer cannot
observe a half-written scale stage, even when the TMA copies complete early.
This is functional implementation evidence, not a measured speedup claim.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from flashinfer.cute_dsl.fp4_common import get_ptr_as_int64, shared_ptr_to_u32
from ._moe_dynamic import gated as _stock
from ._moe_dynamic.gated import (
    DynamicLaunchParams, _TASK_SLICE_CHUNK, _ld_global_acquire_i32,
    _ld_shared_i32, _st_shared_i32,
)
from .moe_dynamic_gated_tiled import MoEGatedDynamicKernelTiled
from .moe_static_common import _sf6_unpack_u8x4

STOCK_GATED_SHA256 = "993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445"
SF6_STAGE_BYTES = 1552
DYNAMIC_SF_BYTES = 1024


@lru_cache(maxsize=1)
def stock_contract_matches() -> bool:
    """Read-only inherited ABI check, including every consumer and queue helper."""
    try:
        return hashlib.sha256(Path(_stock.__file__).read_bytes()).hexdigest() == STOCK_GATED_SHA256
    except (OSError, TypeError):
        return False


def dynamic_sf6_byte_index(kind: str, tile_half: int, byte: int) -> int:
    """CPU oracle: original 1024-byte dynamic tile to decoded SF6-stage byte."""
    if kind not in ("fc1", "fc2") or tile_half not in (0, 1) or not 0 <= byte < 1024:
        raise ValueError("invalid dynamic scale tile coordinate")
    if kind == "fc1":
        return tile_half * 1024 + byte
    return (byte // 512) * 1024 + tile_half * 512 + byte % 512


def dynamic_sf6_stage_index(kind: str, rows: int, k: int, expert: int,
                            row_tile: int, k_tile: int) -> tuple[int, int]:
    """CPU oracle for [expert, packed-stage, 1552] and the selected half."""
    nr, nk = (128, 256) if kind == "fc1" else (256, 128)
    if (kind not in ("fc1", "fc2") or rows <= 0 or k <= 0 or rows % nr or k % nk
            or expert < 0 or not 0 <= row_tile < rows // 128 or not 0 <= k_tile < k // 128):
        raise ValueError("invalid dynamic scale plane geometry")
    if kind == "fc1":
        return expert * (rows // 128) * (k // 256) + row_tile * (k // 256) + k_tile // 2, k_tile % 2
    return expert * (rows // 256) * (k // 128) + (row_tile // 2) * (k // 128) + k_tile, row_tile % 2


@dsl_user_op
def _sf6_ld_global_u32(addr: Int64, *, loc=None, ip=None):
    # Every selected offset is four-byte aligned, including the 1552-byte
    # stage stride. Volatile prevents movement across publication boundaries.
    return Int32(llvm.inline_asm(
        T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.volatile.global.b32 $0, [$1];", "=r,l", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


@cute.jit
def _sf6_expand_dynamic_tile(stage_addr: Int64, destination: Int32,
                              half: Int32, lane: Int32, fc2: cutlass.Constexpr):
    """One producer warp writes 1024 disjoint shared bytes; no global stores."""
    first = lane * Int32(32)
    decoded = first + half * Int32(1024)
    if cutlass.const_expr(fc2):
        decoded = ((first // Int32(512)) * Int32(1024)
                   + half * Int32(512) + first % Int32(512))
    lows = cute.make_rmem_tensor((4,), Int32)
    highs = cute.make_rmem_tensor((2,), Int32)
    for word in cutlass.range_constexpr(4):
        lows[word] = _sf6_ld_global_u32(stage_addr + Int64(decoded // Int32(2) + Int32(word * 4)))
    for word in cutlass.range_constexpr(2):
        highs[word] = _sf6_ld_global_u32(stage_addr + Int64(1024) + Int64(decoded // Int32(4) + Int32(word * 4)))
    base = _sf6_ld_global_u32(stage_addr + Int64(1536)) & Int32(255)
    base_word = base * Int32(0x01010101)
    for word in cutlass.range_constexpr(8):
        low4 = lows[word // 2] >> Int32((word % 2) * 16)
        high4 = highs[word // 4] >> Int32((word % 4) * 8)
        value = _sf6_unpack_u8x4(low4, high4, base_word)
        _st_shared_i32(destination + first + Int32(word * 4), value)


class MoEGatedDynamicKernelSF6(MoEGatedDynamicKernelTiled):
    """Pinned stock dynamic arithmetic with optional direct SF6 scale storage."""

    def __init__(self, *args, reform_sf_pack: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reform_sf_pack = bool(reform_sf_pack)
        if self.reform_sf_pack:
            if not stock_contract_matches():
                raise RuntimeError("dynamic SF6 inherited gated source has drifted")
            # Eight math warps still reserve 232 registers; producer 64 keeps
            # the CTA below the 64K register file while holding unpack words.
            self.load_register_requirement = 64

    def _check_sf6_shapes(self, w13, down, packed1, packed2):
        hidden = cute.size(w13.shape[1])
        intermediate = cute.size(down.shape[1])
        experts = w13.shape[2]
        if (len(w13.shape) != 3 or len(down.shape) != 3 or hidden % 256
                or intermediate % 128 or down.shape[0] != hidden
                or w13.shape[0] != 2 * intermediate or down.shape[2] != experts
                or self.tile_shape_mnk != (128, 128, 128)):
            raise ValueError("dynamic SF6 requires stock M128/N128/K128 gated tiles and aligned weight planes")
        expected = ((experts, (2 * intermediate // 128) * (hidden // 256), 1552),
                    (experts, (hidden // 256) * (intermediate // 128), 1552))
        for tensor, shape in ((packed1, expected[0]), (packed2, expected[1])):
            if tuple(tensor.shape) != shape or tensor.element_type != cutlass.Uint8:
                raise ValueError("dynamic SF6 packed plane shape/type does not match weights")


    @cute.jit
    def load_fc1_tma_slice(
        self, intermediate_slice: Int32, wait_for_prior_slice: Int32,
        task_expert_idx: Int32, gate_tile_cnt, fc1_k_tile_cnt, prod_state,
        ml_pipeline, up_prod_state, up_pipeline, tma_inputs,
        gmem_partitions, smem_partitions,
    ):
        tma_a, tma_b_w13, tma_sfa = tma_inputs
        tAgA_mk, tAgSFA_mk, tBgB_w13, packed = gmem_partitions
        (tAsA, tAsSFA, tBsB_w13, tBsB_w13_up,
         sSFB_gate, sSFB_up, sSFB_up_extra) = smem_partitions
        lane = Int32(cute.arch.thread_idx()[0]) & Int32(31)
        packed_base = get_ptr_as_int64(packed, Int32(0))
        blocks_per_expert = Int64(cute.size(packed.shape[1]))
        k256_tiles = Int64(fc1_k_tile_cnt // Int32(2))
        prod_state.reset_count()
        gate_wait_pending = wait_for_prior_slice
        for fc1_half in cutlass.range_constexpr(2):
            native_up_slice_idx = intermediate_slice * Int32(2) + Int32(fc1_half)
            native_gate_slice_idx = (intermediate_slice + gate_tile_cnt) * Int32(2) + Int32(fc1_half)
            tBgB_gate_nk = tBgB_w13[(None, native_gate_slice_idx, None, task_expert_idx)]
            tBgB_up_nk = tBgB_w13[(None, native_up_slice_idx, None, task_expert_idx)]
            for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                # Preserve stock's third-stage alias wait before any writes.
                if gate_wait_pending > Int32(0) and prod_state.index == Int32(self.ab_storage_stage):
                    self.pass_gate_barrier.wait_unaligned()
                    gate_wait_pending = Int32(0)
                # The base operation waits only. A TMA acquire would publish
                # its release arrival too early, before regular shared writes.
                pipeline.PipelineAsync.producer_acquire(ml_pipeline, prod_state)
                gate_addr = shared_ptr_to_u32(sSFB_gate[None, None, prod_state.index].iterator)
                up_addr = shared_ptr_to_u32(sSFB_up_extra.iterator)
                if prod_state.index < Int32(self.ab_storage_stage):
                    up_addr = shared_ptr_to_u32(sSFB_up[None, None, prod_state.index].iterator)
                block = Int64(task_expert_idx) * blocks_per_expert + Int64(k_tile // Int32(2))
                gate_block = block + Int64(intermediate_slice + gate_tile_cnt) * k256_tiles
                up_block = block + Int64(intermediate_slice) * k256_tiles
                _sf6_expand_dynamic_tile(
                    packed_base + gate_block * Int64(1552), gate_addr,
                    Int32(k_tile) & Int32(1), lane, False,
                )
                _sf6_expand_dynamic_tile(
                    packed_base + up_block * Int64(1552), up_addr,
                    Int32(k_tile) & Int32(1), lane, False,
                )
                # Warp stores happen-before the elected producer's release
                # arrival; consumer_wait acquires both those stores and DMA.
                cute.arch.sync_warp()
                ml_pipeline.producer_acquire(prod_state, try_acquire_token=True)
                barrier = ml_pipeline.producer_get_barrier(prod_state)
                cute.copy(tma_a, tAgA_mk[(None, k_tile)], tAsA[(None, prod_state.index)], tma_bar_ptr=barrier)
                cute.copy(tma_b_w13, tBgB_gate_nk[(None, k_tile)], tBsB_w13[(None, prod_state.index)], tma_bar_ptr=barrier)
                cute.copy(tma_b_w13, tBgB_up_nk[(None, k_tile)], tBsB_w13_up[(None, prod_state.index)], tma_bar_ptr=barrier)
                cute.copy(tma_sfa, tAgSFA_mk[(None, k_tile)], tAsSFA[(None, prod_state.index)], tma_bar_ptr=barrier)
                ml_pipeline.producer_commit(prod_state)
                prod_state.advance()
        return prod_state, up_prod_state

    @cute.jit
    def load_fc2_tma_tile(
        self, intermediate_slice: Int32, output_tile_idx: Int32,
        task_expert_idx: Int32, phase2_prod_state, phase2_pipeline,
        tma_inputs, gmem_partitions, smem_partitions,
    ):
        (tma_b_down,) = tma_inputs
        tBgB_down, packed = gmem_partitions
        tBsB_down, tBsB_down_extra, sSFB = smem_partitions
        lane = Int32(cute.arch.thread_idx()[0]) & Int32(31)
        packed_base = get_ptr_as_int64(packed, Int32(0))
        blocks_per_expert = Int64(cute.size(packed.shape[1]))
        # Each packed N256 block combines two physical output tiles.
        output_pairs = Int64(self._hidden_size // 256)
        k128_tiles = blocks_per_expert // output_pairs
        block = (Int64(task_expert_idx) * blocks_per_expert
                 + Int64(output_tile_idx // Int32(2)) * k128_tiles
                 + Int64(intermediate_slice))
        pipeline.PipelineAsync.producer_acquire(phase2_pipeline, phase2_prod_state)
        destination = shared_ptr_to_u32(sSFB[None, None, phase2_prod_state.index].iterator)
        _sf6_expand_dynamic_tile(
            packed_base + block * Int64(1552), destination,
            output_tile_idx & Int32(1), lane, True,
        )
        cute.arch.sync_warp()
        phase2_pipeline.producer_acquire(phase2_prod_state, try_acquire_token=True)
        barrier = phase2_pipeline.producer_get_barrier(phase2_prod_state)
        if phase2_prod_state.index < Int32(self.ab_storage_stage):
            cute.copy(
                tma_b_down,
                tBgB_down[(None, output_tile_idx, intermediate_slice, task_expert_idx)],
                tBsB_down[(None, phase2_prod_state.index)], tma_bar_ptr=barrier,
            )
        else:
            cute.copy(
                tma_b_down,
                tBgB_down[(None, output_tile_idx, intermediate_slice, task_expert_idx)],
                tBsB_down_extra, tma_bar_ptr=barrier,
            )
        phase2_pipeline.producer_commit(phase2_prod_state)
        phase2_prod_state.advance()
        return phase2_prod_state


    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,  # [num_tokens, K] bf16
        topk_ids: cute.Tensor,  # [num_tokens * topk] int32
        topk_weights: cute.Tensor,  # [num_tokens * topk] float32
        packed_a: cute.Tensor,  # [rows_padded, K, 1] fp4x2 view for compute
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,  # flat uint8 backing packed_a
        scale_storage: cute.Tensor,  # flat uint8 backing sfa_ptr
        barrier_count: cute.Tensor,  # [1] int32 (host-zeroed)
        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)
        pair_head: cute.Tensor,  # [1] int32
        task_head: cute.Tensor,  # [1] int32
        task_tail: cute.Tensor,  # [1] int32
        task_expert: cute.Tensor,  # [max_tasks] int32
        task_valid_rows: cute.Tensor,  # [max_tasks] int32
        b_w13: cute.Tensor,  # [2*I_tp, K, E] (gated) or [I_tp, K, E] (relu2)
        sfb_w13_ptr: cute.Pointer,  # scale factors for w13
        b_down: cute.Tensor,  # [K, I_tp, E]
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,  # expert row histogram [E]
        expert_write_rows: cute.Tensor,  # route/pack write cursors [E]
        expert_tile_base: cute.Tensor,  # compact physical-tile prefix [E + 1]
        input_global_scale: cute.Tensor,  # [E] per-expert FC1 input scale
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,  # [num_tokens, K]
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        sfb1_packed: cute.Tensor,
        sfb2_packed: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(len(b_w13.shape) == 4):
            b_w13 = cute.group_modes(b_w13, 1, 3)
        if cutlass.const_expr(len(b_down.shape) == 4):
            b_down = cute.group_modes(b_down, 1, 3)
        if cutlass.const_expr(not self.reform_sf_pack):
            MoEGatedDynamicKernelTiled.__call__(
                self, a_input, topk_ids, topk_weights, packed_a, sfa_ptr,
                packed_a_storage, scale_storage, barrier_count, barrier_epoch,
                pair_head, task_head, task_tail, task_expert, task_valid_rows,
                b_w13, sfb_w13_ptr, b_down, sfb_down_ptr, row_counts,
                expert_write_rows, expert_tile_base, input_global_scale, alpha,
                down_alpha, global_scale, scatter_output, token_map, token_weights,
                max_active_clusters, stream,
            )
        else:
            self._check_sf6_shapes(b_w13, b_down, sfb1_packed, sfb2_packed)
            self.a_dtype = packed_a.element_type
            self.b_dtype = b_w13.element_type
            self.sf_dtype = sfa_ptr.dtype
            self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
            self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
            # Dynamic never materializes the intermediate C tensor. Preserve the
            # original row-major epilogue layout without carrying a dead memref.
            self.c_layout = utils.LayoutEnum.ROW_MAJOR

            hidden_size = a_input.shape[1]
            if cutlass.const_expr(
                hidden_size > self.tile_shape_mnk[0] * self.tile_shape_mnk[1]
            ):
                raise ValueError(
                    "the gated dynamic kernel requires one BF16 input row to fit "
                    "in its 16384-element Q0 staging buffer"
                )
            self._setup_attributes(hidden_size=hidden_size)

            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                packed_a.shape, self.sf_vec_size
            )
            sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

            # TMA descriptors
            tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
                packed_a,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                1,
            )
            tma_sfa, gSFA = self._dense_cls._make_tma_atoms_and_tensors(
                sfa_tensor,
                self.sfa_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                1,
                internal_type=cutlass.Int16,
            )
            # FC1 B uses a true N64 descriptor.  Each logical N128 slice is two
            # consecutive native B tiles; Up precedes Gate in global w13 storage.
            tma_b_w13, gB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
                b_w13,
                self.fc1_b_smem_layout_staged,
                (self.fc1_tile_shape_mnk[1], self.fc1_tile_shape_mnk[2]),
                1,
            )
            # FC2 weight TMA; expert scales are loaded directly from SF6.
            tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
                b_down,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
            )
            # W13 concatenates equally-sized Gate and Up branches along N.
            gate_tile_cnt_static = b_w13.shape[0] // self.tile_shape_mnk[1] // 2
            if cutlass.const_expr(gate_tile_cnt_static > _TASK_SLICE_CHUNK):
                raise ValueError(
                    "the gated dynamic kernel retains at most four intermediate "
                    "slices per task"
                )
            gate_tile_cnt = Int32(gate_tile_cnt_static)
            launch_params = DynamicLaunchParams(row_counts, gate_tile_cnt)
            grid = (*self.cluster_shape_mn, max_active_clusters)
            self.kernel(
                a_input,
                topk_ids,
                topk_weights,
                packed_a_storage,
                scale_storage,
                barrier_count,
                barrier_epoch,
                pair_head,
                task_head,
                task_tail,
                task_expert,
                task_valid_rows,
                tma_a,
                gA,
                tma_sfa,
                gSFA,
                tma_b_w13,
                gB_w13,
                sfb1_packed,
                tma_b_down,
                gB_down,
                sfb2_packed,
                self.tiled_mma,
                self.fc1_tiled_mma,
                self.mma_atom,
                self.mma_atom,
                self.cta_layout_mnk,
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.phase2_b_smem_layout_staged,
                self.fc1_b_smem_layout_staged,
                self.sfa_smem_layout_staged,
                self.sfb_smem_layout_staged,
                self.phase2_sfb_smem_layout_staged,
                self.fc1_sfb_smem_layout_staged,
                self.fc1_sfb_smem_layout_storage,
                self.epi_smem_layout_staged,
                launch_params,
                expert_write_rows,
                expert_tile_base,
                input_global_scale,
                alpha,
                down_alpha,
                global_scale,
                scatter_output,
                token_map,
                token_weights,
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
        pair_head: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_b_w13: cute.CopyAtom,
        mB_w13: cute.Tensor,
        sfb1_packed: cute.Tensor,
        tma_b_down: cute.CopyAtom,
        mB_down: cute.Tensor,
        sfb2_packed: cute.Tensor,
        tiled_mma: cute.TiledMma,
        fc1_tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        mma_atom_tail: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        phase2_b_smem_staged: cute.ComposedLayout,
        fc1_b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
        phase2_sfb_smem_staged: cute.Layout,
        fc1_sfb_smem_staged: cute.Layout,
        fc1_sfb_smem_layout_storage: cute.Layout,
        epi_smem_staged: cute.ComposedLayout,
        launch_params: DynamicLaunchParams,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
    ):
        """Kernel entry point."""
        tidx, _, _ = cute.arch.thread_idx()
        _bidx, _, bidz = cute.arch.block_idx()
        _, _, gdim_z = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_id = Int32(tidx) & Int32(31)
        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_b_w13)
            cpasync.prefetch_descriptor(tma_b_down)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        a_smem_one = cute.slice_(a_smem_staged, (None, None, 0))
        b_smem_one = cute.slice_(b_smem_staged, (None, None, 0))
        fc1_b_smem_one = cute.slice_(fc1_b_smem_staged, (None, None, 0))
        sfa_smem_one = cute.slice_(sfa_smem_staged, (None, None, 0))
        sfb_smem_one = cute.slice_(sfb_smem_staged, (None, None, 0))
        fc1_sfb_smem_one = cute.slice_(fc1_sfb_smem_staged, (None, None, 0))
        sequential_branch_compact = cutlass.const_expr(
            getattr(self, "sequential_branch_compact", False)
        )
        fc1_storage_alias = cutlass.const_expr(
            getattr(self, "fc1_storage_alias", sequential_branch_compact)
        )
        fc1_tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_one)
            + cute.size_in_bytes(self.b_dtype, fc1_b_smem_one)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_one)
        )
        fc1_branch_tma_copy_bytes = fc1_tma_copy_bytes
        if cutlass.const_expr(not sequential_branch_compact):
            fc1_tma_copy_bytes += cute.size_in_bytes(
                self.b_dtype, fc1_b_smem_one
            )
        phase2_tma_copy_bytes = cute.size_in_bytes(
            self.b_dtype, b_smem_one
        )

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class StorageGated:
            # ctrl layout (16 x Int32, accessed via raw shared memory PTX):
            #   [0] has_task     [4] done          [8]  expert_idx
            #   [12] m_tile_idx  [16] slice_begin   [20] slice_count
            #   [24] valid_rows  [28] batch_base
            #   [32] next_has    [36] next_done     [40] next_expert
            #   [44] next_mtile  [48] next_begin    [52] next_count
            #   [56] next_rows   [60] reserved
            ctrl: cute.struct.MemRange[cutlass.Int32, 16]
            # Startup-only route cache aliases the unused sC backing.
            route_phys_rows: cute.struct.MemRange[cutlass.Int32, 0]
            route_expert_ids: cute.struct.MemRange[cutlass.Int32, 0]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            up_pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            phase2_pipeline_array: cute.struct.MemRange[
                cutlass.Int64, self.phase2_stage * 2
            ]
            q0_bulk_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, self.tile_shape_mnk[0] * 2
            ]
            scatter_weight_cache: cute.struct.MemRange[cutlass.Float32, 0]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            # During FC1, the first B-sized part of sC is the contiguous
            # third N128 B stage.  FC1 releases it before activation writes.
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            # Gate and Up occupy disjoint N64 halves of the N128 sB stage.
            sB_up: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, 0],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB_phase2_extra: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_one)],
                self.buffer_align_bytes,
            ]
            # SM120 packs each logical N64 SFB half in a physical-N128 block;
            # the Up branch therefore needs a distinct physical backing.
            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    (
                        0
                        if fc1_storage_alias
                        else cute.cosize(fc1_sfb_smem_layout_storage)
                    ),
                ],
                self.buffer_align_bytes,
            ]

        storage = smem.allocate(StorageGated)

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        ml_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=fc1_tma_copy_bytes,
            barrier_storage=storage.pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        up_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=fc1_branch_tma_copy_bytes,
            barrier_storage=storage.up_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        phase2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.phase2_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=phase2_tma_copy_bytes,
            barrier_storage=storage.phase2_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        # FC2 retains its two physical B stages and addresses dead sA[0]
        # separately as phase2 stage2.  Conversely FC1 sees one contiguous
        # three-stage N128 backing spanning sB[0:2] and the beginning of sC.
        phase2_b_extra_ptr = cute.recast_ptr(
            storage.sA.data_ptr(),
            b_smem_one.inner,
            dtype=self.b_dtype,
        )
        sB_phase2_extra = cute.make_tensor(phase2_b_extra_ptr, b_smem_one.outer)
        sB_fc1_all = storage.sB.get_tensor(
            phase2_b_smem_staged.outer,
            swizzle=phase2_b_smem_staged.inner,
        )
        # While FC1 is live, split the N128 FC2 backing into two N64 views.
        sB_fc1 = cute.local_tile(
            sB_fc1_all,
            cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
            (0, 0, None),
        )
        sB_up_fc1 = cute.local_tile(
            sB_fc1_all,
            cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
            (1, 0, None),
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_staged)
        # sSFB_phase2_extra is the immediately following aligned field, so
        # expose the existing two-plus-one backing as one staged tensor.
        sSFB_phase2 = storage.sSFB.get_tensor(phase2_sfb_smem_staged)
        # Gate gets a contiguous third SFB stage from the existing phase2
        # extra.  Up keeps its two allocated stages and uses a disjoint
        # one-stage alias in sC immediately after the FC1 B-stage2 bytes.
        sSFB_fc1 = storage.sSFB.get_tensor(fc1_sfb_smem_staged)
        sSFB_up_fc1 = (
            sSFB_fc1
            if fc1_storage_alias
            else storage.sSFB_up.get_tensor(fc1_sfb_smem_layout_storage)
        )
        fc1_sfb_smem_one = cute.slice_(fc1_sfb_smem_staged, (None, None, 0))
        fc1_b_stage_bytes = cute.size_in_bytes(self.b_dtype, b_smem_one)
        sSFB_up_fc1_extra_ptr = cute.recast_ptr(
            storage.sC.data_ptr() + fc1_b_stage_bytes // 2,
            dtype=self.sf_dtype,
        )
        sSFB_up_fc1_extra = cute.make_tensor(sSFB_up_fc1_extra_ptr, fc1_sfb_smem_one)
        sC = storage.sC.get_tensor(
            epi_smem_staged.outer,
            swizzle=epi_smem_staged.inner,
        )
        sfa_base_addr = shared_ptr_to_u32(storage.sSFA.data_ptr())
        sfa_stage_elements = Int32(cute.cosize(sfa_smem_one))
        ctrl_base_addr = shared_ptr_to_u32(storage.ctrl.data_ptr())
        # Q0 uses raw-linear sC as an eight-token BF16 staging buffer.
        # Move both 288-entry route caches to startup-idle sA.
        route_phys_rows_addr = shared_ptr_to_u32(storage.sA.data_ptr())
        route_expert_ids_addr = route_phys_rows_addr + Int32(
            (self.num_mma_warps + 1) * 32 * 4
        )
        q0_input_stage_base_addr = shared_ptr_to_u32(storage.sC.data_ptr())
        q0_bulk_barrier_addr = shared_ptr_to_u32(storage.q0_bulk_barrier.data_ptr())
        scatter_tok_base_addr = shared_ptr_to_u32(storage.scatter_tok_cache.data_ptr())
        scatter_weight_base_addr = scatter_tok_base_addr + Int32(4)

        self.initialize_route_q0_and_publish(
            (tidx, bidz, gdim_z, warp_idx, is_cta_leader),
            (a_input, topk_ids, topk_weights, input_global_scale),
            (
                packed_a_storage,
                scale_storage,
                scatter_output,
                token_map,
                token_weights,
            ),
            (expert_write_rows, expert_tile_base, pair_head),
            (
                task_head,
                task_tail,
                task_expert,
                task_valid_rows,
            ),
            (barrier_count, barrier_epoch),
            (
                ctrl_base_addr,
                route_phys_rows_addr,
                route_expert_ids_addr,
                q0_input_stage_base_addr,
                q0_bulk_barrier_addr,
            ),
            launch_params,
        )

        # Deferred publication is complete after the resident-grid barrier
        # inside initialize_route_q0_and_publish.  Cache the immutable tail in
        # the otherwise streaming-only ctrl[28] slot; the claim loop uses a
        # side-effecting shared load to preserve phase ordering.
        if is_cta_leader > Int32(0):
            stable_task_tail = _ld_global_acquire_i32(
                get_ptr_as_int64(task_tail, Int32(0))
            )
            _st_shared_i32(ctrl_base_addr + Int32(28), stable_task_tail)
            _st_shared_i32(ctrl_base_addr + Int32(32), Int32(0))
            _st_shared_i32(ctrl_base_addr + Int32(36), Int32(0))

        gA = cute.local_tile(
            mA, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        # B is tiled at the native N64 compute granularity.  SFB is tiled at
        # the physical N128 scale-factor block granularity and replayed for
        # the two B halves.
        gB_w13_tiled = cute.local_tile(
            mB_w13,
            cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA = cute.local_tile(
            mSFA, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(tidx)
        fc1_thr_mma = fc1_tiled_mma.get_slice(tidx)

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

        # w13 FC1 partitions: N64 B payload plus physical-N128 SFB payload.
        tBsB_w13, tBgB_w13 = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_fc1, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsB_w13_up, _tBgB_w13_up = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_up_fc1, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        # Existing raw shared layouts remain consumer-owned stage storage.
        # Only their producer changes; no raw global scale descriptor exists.
        tBsSFB_w13, tBgSFB_w13 = sSFB_fc1, sfb1_packed
        tBsSFB_w13_up = sSFB_up_fc1
        tBsSFB_w13_up_extra = sSFB_up_fc1_extra

        # B_down TMA partitions
        gB_down = cute.local_tile(
            mB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        tBsB_down, tBgB_down = cpasync.tma_partition(
            tma_b_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_down, 0, 2),
        )
        tBsB_down_extra, _tBgB_down_extra = cpasync.tma_partition(
            tma_b_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_phase2_extra, 0, 2),
            cute.group_modes(gB_down, 0, 2),
        )
        tBsSFB_down, tBgSFB_down = sSFB_phase2, sfb2_packed

        # FC2 fragment partitions retain the original N128 contract.
        tCsA = thr_mma.partition_A(sA)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrSFA = self._dense_cls._partition_fragment_SFA(
            self,  # type: ignore[arg-type]
            sSFA[None, None, 0],
            thr_mma,
            tidx,
        )

        # FC1 has an independent N64 MMA/permutation and aliases the same A/SFA
        # storage.  Its SFB fragment is created per half below because a
        # physical N128 SFB block contains two logical N64 scale tiles.
        tCsA_fc1 = fc1_thr_mma.partition_A(sA)
        tCrA_fc1 = fc1_tiled_mma.make_fragment_A(tCsA_fc1[None, None, None, 0])
        tCrSFA_fc1 = self._dense_cls._partition_fragment_SFA(
            self,  # type: ignore[arg-type]
            sSFA[None, None, 0],
            fc1_thr_mma,
            tidx,
        )
        tCsB_fc1 = fc1_thr_mma.partition_B(sB_fc1)
        tCrB_fc1 = fc1_tiled_mma.make_fragment_B(tCsB_fc1[None, None, None, 0])
        tCsB_up_fc1 = fc1_thr_mma.partition_B(sB_up_fc1)
        tCrB_up_fc1 = fc1_tiled_mma.make_fragment_B(tCsB_up_fc1[None, None, None, 0])
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFB = self._dense_cls._partition_fragment_SFB(
            self,  # type: ignore[arg-type]
            sSFB[None, None, 0],
            thr_mma,
            tidx,
        )

        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        sub_shape = tCsC_for_shape.shape[:3]
        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        k_tile_cnt = cute.size(gA, mode=[3])
        fc1_k_tile_cnt = k_tile_cnt
        # gB is native-N64 while tasks and FC2 remain logical-N128.
        native_fc1_tile_cnt = cute.size(gB_w13_tiled, mode=[2]) // Int32(2)
        gate_tile_cnt = native_fc1_tile_cnt // Int32(2)
        output_tile_cnt = cute.size(gB_down, mode=[2])

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        up_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        up_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        phase2_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.phase2_stage
        )
        phase2_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.phase2_stage
        )

        num_k_blocks = cute.size(tCrA, mode=[2])
        fc1_num_k_blocks = cute.size(tCrA_fc1, mode=[2])

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
        atom_ld_SF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
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
        smem_copy_A_fc1 = cute.make_tiled_copy_A(atom_ld_A, fc1_tiled_mma)
        smem_copy_B_fc1 = cute.make_tiled_copy_B(atom_ld_B, fc1_tiled_mma)
        smem_copy_SFA_fc1 = cute.make_tiled_copy(
            atom_ld_SF,
            self._dense_cls._get_layoutSFA_TV(self, fc1_tiled_mma),  # type: ignore[arg-type]
            (
                cute.size(fc1_tiled_mma.permutation_mnk[0]),
                cute.size(fc1_tiled_mma.permutation_mnk[2]),
            ),
        )
        smem_copy_SFB_fc1 = cute.make_tiled_copy(
            atom_ld_SF,
            self._dense_cls._get_layoutSFB_TV(self, fc1_tiled_mma),  # type: ignore[arg-type]
            (
                cute.size(fc1_tiled_mma.permutation_mnk[1]),
                cute.size(fc1_tiled_mma.permutation_mnk[2]),
            ),
        )

        thr_ld_A = smem_copy_A.get_slice(tidx)
        thr_ld_B = smem_copy_B.get_slice(tidx)
        csA = thr_ld_A.partition_S(sA)
        crA = thr_ld_A.retile(tCrA)
        csB = thr_ld_B.partition_S(sB)
        csB_phase2_extra = thr_ld_B.partition_S(sB_phase2_extra)
        crB = thr_ld_B.retile(tCrB)

        thr_ld_SFA = smem_copy_SFA.get_slice(tidx)
        thr_ld_SFB = smem_copy_SFB.get_slice(tidx)
        csSFA = thr_ld_SFA.partition_S(sSFA)
        crSFA = thr_ld_SFA.retile(tCrSFA)
        csSFB = thr_ld_SFB.partition_S(sSFB_phase2)
        crSFB = thr_ld_SFB.retile(tCrSFB)

        thr_ld_A_fc1 = smem_copy_A_fc1.get_slice(tidx)
        thr_ld_B_fc1 = smem_copy_B_fc1.get_slice(tidx)
        csA_fc1 = thr_ld_A_fc1.partition_S(sA)
        crA_fc1 = thr_ld_A_fc1.retile(tCrA_fc1)
        csB_fc1 = thr_ld_B_fc1.partition_S(sB_fc1)
        crB_fc1 = thr_ld_B_fc1.retile(tCrB_fc1)
        csB_up_fc1 = thr_ld_B_fc1.partition_S(sB_up_fc1)
        crB_up_fc1 = thr_ld_B_fc1.retile(tCrB_up_fc1)

        thr_ld_SFA_fc1 = smem_copy_SFA_fc1.get_slice(tidx)
        thr_ld_SFB_fc1 = smem_copy_SFB_fc1.get_slice(tidx)
        csSFA_fc1 = thr_ld_SFA_fc1.partition_S(sSFA)
        crSFA_fc1 = thr_ld_SFA_fc1.retile(tCrSFA_fc1)

        # ===================================================================
        # Per-warp setup for the consumer steady state
        # ===================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ===================================================================
        # Consumer steady state: pop one ready task per CTA, then let
        # the MMA warps and DMA warp cooperate on that task.
        # ===================================================================
        consumer_live = Int32(1)
        while consumer_live > Int32(0):
            has_task, is_done = self.claim_and_cache_task(
                tidx,
                warp_idx,
                is_cta_leader,
                ctrl_base_addr,
                task_head,
                task_expert,
                task_valid_rows,
                token_map,
                token_weights,
                scatter_tok_base_addr,
                scatter_weight_base_addr,
            )
            if has_task == Int32(0):
                if is_done > Int32(0):
                    consumer_live = Int32(0)
            elif warp_idx < self.num_mma_warps:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
                task_m_tile_idx = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_slice_begin_idx = _ld_shared_i32(ctrl_base_addr + Int32(16))
                task_slice_count_val = _ld_shared_i32(ctrl_base_addr + Int32(20))
                task_valid_rows_val = _ld_shared_i32(ctrl_base_addr + Int32(24))

                alpha_value = alpha[task_expert_idx].to(cutlass.Float32)
                valid_rows = task_valid_rows_val
                # atom_layout=(4,2,1): two M16 fragments per warp, separated
                # by 64 rows. Full tasks use the original branch-free method.
                warp_m_coord = Int32(warp_idx) & Int32(3)

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
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_C_Atom
                )
                fc1_tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                    copy_atom_C, fc1_tiled_mma
                )
                fc1_tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, fc1_tiled_copy_C_Atom
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                fc1_thr_copy_r2s = fc1_tiled_copy_r2s.get_slice(tidx)
                fc1_tRS_sD = fc1_thr_copy_r2s.partition_D(sC)
                down_alpha_value = down_alpha[task_expert_idx].to(cutlass.Float32)
                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]

                fc1_m_tiles = cute.size(tCrA_fc1, mode=[1])
                fc1_n_tiles = cute.size(tCrB_fc1, mode=[1])
                deferred_a_words = cute.make_rmem_tensor((8,), cutlass.Uint32)
                deferred_a_words.fill(0)
                deferred_sfa_words = cute.make_rmem_tensor((2,), cutlass.Uint32)
                deferred_sfa_words.fill(0)
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    if valid_rows == Int32(self.tile_shape_mnk[0]):
                        cons_state, up_cons_state = self.fc1_gate_up_swiglu_to_sC(
                            tidx,
                            ml_pipeline,
                            cons_state,
                            up_pipeline,
                            up_cons_state,
                            fc1_tiled_mma,
                            mma_atom,
                            sSFB_fc1,
                            sSFB_up_fc1,
                            sSFB_up_fc1_extra,
                            fc1_thr_mma,
                            thr_ld_SFB_fc1,
                            csA_fc1,
                            csB_fc1,
                            csB_up_fc1,
                            csSFA_fc1,
                            crA_fc1,
                            crB_fc1,
                            crB_up_fc1,
                            crSFA_fc1,
                            tCrA_fc1,
                            tCrB_fc1,
                            tCrB_up_fc1,
                            tCrSFA_fc1,
                            smem_copy_A_fc1,
                            smem_copy_B_fc1,
                            smem_copy_SFA_fc1,
                            smem_copy_SFB_fc1,
                            fc1_tiled_copy_r2s,
                            fc1_tRS_sD,
                            fc1_k_tile_cnt,
                            fc1_num_k_blocks,
                            fc1_m_tiles,
                            fc1_n_tiles,
                            alpha_value,
                            valid_rows,
                            task_expert_idx,
                            global_scale,
                            sC,
                            sA,
                            sfa_base_addr,
                            epi_rest_m,
                        )
                    else:
                        cons_state, up_cons_state = self.fc1_gate_up_swiglu_to_sC_tail(
                            tidx,
                            ml_pipeline,
                            cons_state,
                            up_pipeline,
                            up_cons_state,
                            fc1_tiled_mma,
                            mma_atom_tail,
                            sSFB_fc1,
                            sSFB_up_fc1,
                            sSFB_up_fc1_extra,
                            fc1_thr_mma,
                            thr_ld_SFB_fc1,
                            csA_fc1,
                            csB_fc1,
                            csB_up_fc1,
                            csSFA_fc1,
                            crA_fc1,
                            crB_fc1,
                            crB_up_fc1,
                            crSFA_fc1,
                            tCrA_fc1,
                            tCrB_fc1,
                            tCrB_up_fc1,
                            tCrSFA_fc1,
                            smem_copy_A_fc1,
                            smem_copy_B_fc1,
                            smem_copy_SFA_fc1,
                            smem_copy_SFB_fc1,
                            fc1_tiled_copy_r2s,
                            fc1_tRS_sD,
                            fc1_k_tile_cnt,
                            fc1_num_k_blocks,
                            fc1_m_tiles,
                            fc1_n_tiles,
                            alpha_value,
                            valid_rows,
                            warp_m_coord,
                            task_expert_idx,
                            global_scale,
                            sC,
                            sA,
                            sfa_base_addr,
                            epi_rest_m,
                        )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    q1_a_stage_idx = Int32(3)
                    defer_a = Int32(0)
                    if slice_idx == Int32(1):
                        q1_a_stage_idx = Int32(4)
                    elif slice_idx == Int32(2):
                        q1_a_stage_idx = Int32(0)
                        defer_a = Int32(1)
                    elif slice_idx == Int32(3):
                        q1_a_stage_idx = Int32(1)

                    q1_sfa_stage_idx = Int32(3)
                    defer_sfa = Int32(0)
                    deferred_sfa_slot = Int32(0)
                    if slice_idx == Int32(1):
                        q1_sfa_stage_idx = Int32(0)
                        defer_sfa = Int32(1)
                    elif slice_idx == Int32(2):
                        q1_sfa_stage_idx = Int32(0)
                        defer_sfa = Int32(1)
                        deferred_sfa_slot = Int32(1)
                    elif slice_idx == Int32(3):
                        q1_sfa_stage_idx = Int32(0)
                    self.quantize_q1_sC_to_sA_sSFA(
                        tidx,
                        valid_rows,
                        task_expert_idx,
                        global_scale,
                        sC,
                        sA,
                        fc1_tRS_sD,
                        sfa_base_addr,
                        sfa_stage_elements,
                        q1_a_stage_idx,
                        defer_a,
                        deferred_a_words,
                        q1_sfa_stage_idx,
                        defer_sfa,
                        deferred_sfa_words,
                        deferred_sfa_slot,
                        epi_rest_m,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    self.pass_gate_barrier.arrive_unaligned()

                    # Q1 has finished reading sC, so the following FC1 slice
                    # may safely reuse it as the third B/SFB stage.
                    slice_idx += Int32(1)

                # The final FC1 pass has released A/SFA stages0:2.  Materialize
                # the exact packed Q1 bytes deferred to registers.
                self.flush_deferred_q1_a(
                    tidx,
                    valid_rows,
                    deferred_a_words,
                    sA,
                    Int32(2),
                )
                self.flush_deferred_q1_sfa(
                    tidx,
                    valid_rows,
                    deferred_sfa_words[0],
                    sfa_base_addr,
                    sfa_stage_elements,
                    Int32(1),
                )
                self.flush_deferred_q1_sfa(
                    tidx,
                    valid_rows,
                    deferred_sfa_words[1],
                    sfa_base_addr,
                    sfa_stage_elements,
                    Int32(2),
                )
                self.epilog_sync_barrier.arrive_and_wait()

                phase2_cons_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    physical_output_tile_idx = (
                        Int32(output_tile_idx) + task_expert_idx
                    ) % Int32(output_tile_cnt)
                    down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
                    down_acc.fill(0.0)
                    slice_idx = Int32(0)
                    while slice_idx < task_slice_count_val:
                        q1_a_stage_idx = Int32(3)
                        if slice_idx == Int32(1):
                            q1_a_stage_idx = Int32(4)
                        elif slice_idx == Int32(2):
                            q1_a_stage_idx = Int32(2)
                        elif slice_idx == Int32(3):
                            q1_a_stage_idx = Int32(1)

                        q1_sfa_stage_idx = Int32(3)
                        if slice_idx == Int32(1):
                            q1_sfa_stage_idx = Int32(1)
                        elif slice_idx == Int32(2):
                            q1_sfa_stage_idx = Int32(2)
                        elif slice_idx == Int32(3):
                            q1_sfa_stage_idx = Int32(0)
                        self.load_fc2_a_fragments(
                            num_k_blocks,
                            q1_a_stage_idx,
                            q1_sfa_stage_idx,
                            (csA, csSFA),
                            (crA, crSFA),
                            (smem_copy_A, smem_copy_SFA),
                        )
                        if valid_rows == Int32(self.tile_shape_mnk[0]):
                            phase2_cons_state = self.fc2_accumulate_slice(
                                num_k_blocks,
                                mma_atom,
                                down_acc,
                                (phase2_pipeline, phase2_cons_state),
                                (csB, csB_phase2_extra, csSFB),
                                (tCrA, tCrB, tCrSFA, tCrSFB, crB, crSFB),
                                (smem_copy_B, smem_copy_SFB),
                            )
                        else:
                            phase2_cons_state = self.fc2_accumulate_slice_tail(
                                num_k_blocks,
                                mma_atom_tail,
                                down_acc,
                                valid_rows,
                                warp_m_coord,
                                (phase2_pipeline, phase2_cons_state),
                                (csB, csB_phase2_extra, csSFB),
                                (tCrA, tCrB, tCrSFA, tCrSFB, crB, crSFB),
                                (smem_copy_B, smem_copy_SFB),
                            )
                        slice_idx += Int32(1)

                    self.fc2_epilogue_to_sC(
                        acc_shape,
                        down_alpha_value,
                        down_acc,
                        sC,
                        tiled_copy_r2s,
                        thr_copy_r2s,
                        tRS_sD,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    if (warp_idx & Int32(2)) == Int32(0):
                        self.fc2_group_a_barrier.arrive_and_wait()
                    else:
                        self.fc2_group_b_barrier.arrive_and_wait()
                    self.scatter_sC_to_gmem(
                        tidx,
                        physical_output_tile_idx,
                        valid_rows,
                        sC,
                        tRS_sD,
                        scatter_output,
                        scatter_tok_base_addr,
                        scatter_weight_base_addr,
                        down_alpha_value,
                    )
                    if (warp_idx & Int32(2)) == Int32(0):
                        self.fc2_group_a_barrier.arrive_and_wait()
                    else:
                        self.fc2_group_b_barrier.arrive_and_wait()

                # All output tiles have consumed every retained Q1 slice.
                self.pass_final_barrier.arrive_and_wait()

            elif warp_idx == self.tma_load_warp_id:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
                task_m_tile_idx = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_slice_begin_idx = _ld_shared_i32(ctrl_base_addr + Int32(16))
                task_slice_count_val = _ld_shared_i32(ctrl_base_addr + Int32(20))

                tAgA_mk = tAgA[(None, task_m_tile_idx, None, Int32(0))]
                tAgSFA_mk = tAgSFA[(None, task_m_tile_idx, None, Int32(0))]
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    intermediate_slice = task_slice_begin_idx + slice_idx
                    wait_for_prior_slice = Int32(0)
                    if slice_idx > Int32(0):
                        wait_for_prior_slice = Int32(1)
                    prod_state, up_prod_state = self.load_fc1_tma_slice(
                        intermediate_slice,
                        wait_for_prior_slice,
                        task_expert_idx,
                        gate_tile_cnt,
                        fc1_k_tile_cnt,
                        prod_state,
                        ml_pipeline,
                        up_prod_state,
                        up_pipeline,
                        (tma_a, tma_b_w13, tma_sfa),
                        (tAgA_mk, tAgSFA_mk, tBgB_w13, tBgSFB_w13),
                        (
                            tAsA,
                            tAsSFA,
                            tBsB_w13,
                            tBsB_w13_up,
                            tBsSFB_w13,
                            tBsSFB_w13_up,
                            tBsSFB_w13_up_extra,
                        ),
                    )
                    slice_idx += Int32(1)

                # The final FC1 MMA release is narrower than pass_gate:
                # clone the state so we can prove every FC1 A/B/SF stage is
                # empty without advancing the live producer state.  FC2
                # weights do not alias sC, so they may prefetch while final
                # activation/Q1 is still using sC.
                fc1_drain_state = prod_state.clone()
                ml_pipeline.producer_tail(fc1_drain_state)

                phase2_prod_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    physical_output_tile_idx = (
                        Int32(output_tile_idx) + task_expert_idx
                    ) % Int32(output_tile_cnt)
                    slice_idx = Int32(0)
                    while slice_idx < task_slice_count_val:
                        intermediate_slice = task_slice_begin_idx + slice_idx
                        phase2_prod_state = self.load_fc2_tma_tile(
                            intermediate_slice,
                            physical_output_tile_idx,
                            task_expert_idx,
                            phase2_prod_state,
                            phase2_pipeline,
                            (tma_b_down,),
                            (tBgB_down, tBgSFB_down),
                            (
                                tBsB_down,
                                tBsB_down_extra,
                                tBsSFB_down,
                            ),
                        )
                        slice_idx += Int32(1)

                # Warp8 has finished issuing current-task FC2 weights while
                # math warps still own FC2 MMA/epilogue/scatter. Reserve and
                # cache one next descriptor in disjoint shared control state.
                self.prefetch_next_task_descriptor(
                    lane_id,
                    ctrl_base_addr,
                    task_head,
                    task_expert,
                    task_valid_rows,
                )

                # Consume the final slice's activation/Q1 arrival before the
                # task handoff. Earlier arrivals were consumed lazily at the
                # first next-slice Stage2 overwrite.
                self.pass_gate_barrier.wait_unaligned()

                # Keep A4/Q1 and sC alive until all output tiles complete.
                self.pass_final_barrier.wait_unaligned()

        if warp_idx == self.tma_load_warp_id:
            ml_pipeline.producer_tail(prod_state)
            if cutlass.const_expr(getattr(self, "sequential_branch_compact", False)):
                up_pipeline.producer_tail(up_prod_state)
            phase2_pipeline.producer_tail(phase2_prod_state)
        return



__all__ = ["MoEGatedDynamicKernelSF6", "stock_contract_matches"]
