# SPDX-License-Identifier: Apache-2.0
"""Optional word decoder for the pinned TP SF6 long-prefill producer.

Only the two load methods are specialized. Q0, queue publication, MMA,
scatter and every decode/short-prefill class retain their original source.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int32, Int64
from flashinfer.cute_dsl.fp4_common import get_ptr_as_int64, shared_ptr_to_u32
from ._moe_dynamic.gated import _st_shared_i32
from . import moe_dynamic_gated_sf6 as _sf6
from .moe_dynamic_gated_sf6 import _sf6_ld_global_u32

PARENT_SHA256 = '6efb0a2ec044dfbaeb43af92f569b6c130a99bee751fb5a129f78dac1183300e'


@lru_cache(maxsize=1)
def stock_contract_matches():
    return (_sf6.stock_contract_matches()
            and hashlib.sha256(Path(_sf6.__file__).read_bytes()).hexdigest() == PARENT_SHA256)


def _sf6_unpack_word(low, high, base_lo, base_hi):
    """Four exact byte lanes from the validated SF6 packer's base/deltas.

    The word arithmetic is shared with the existing EP scale decoder; this
    producer still uses its own global reads and shared-stage publication.
    Splitting the base avoids any carry between neighboring byte lanes.
    """
    low = cutlass.Uint32(low) & cutlass.Uint32(0xFFFF)
    low = (low | (low << cutlass.Uint32(8))) & cutlass.Uint32(0x00FF00FF)
    low = (low | (low << cutlass.Uint32(4))) & cutlass.Uint32(0x0F0F0F0F)
    high = cutlass.Uint32(high) & cutlass.Uint32(0xFF)
    high = (high | (high << cutlass.Uint32(12))) & cutlass.Uint32(0x000F000F)
    high = (high | (high << cutlass.Uint32(6))) & cutlass.Uint32(0x03030303)
    return ((low | (high << cutlass.Uint32(4))) + base_lo) ^ base_hi


@cute.jit
def _sf6_expand_dynamic_tile(stage_addr: Int64, destination: Int32,
                              half: Int32, lane: Int32, fc2: cutlass.Constexpr,
                              word_unpack: cutlass.Constexpr = False):
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
    if cutlass.const_expr(word_unpack):
        base_lo = cutlass.Uint32(base & Int32(127)) * cutlass.Uint32(0x01010101)
        base_hi = cutlass.Uint32(base & Int32(128)) * cutlass.Uint32(0x01010101)
    for word in cutlass.range_constexpr(8):
        if cutlass.const_expr(word_unpack):
            value = Int32(_sf6_unpack_word(
                lows[word // 2] >> Int32((word % 2) * 16),
                highs[word // 4] >> Int32((word % 4) * 8), base_lo, base_hi))
        else:
            value = Int32(0)
            for byte in cutlass.range_constexpr(4):
                index = word * 4 + byte
                low = (lows[index // 8] >> Int32((index % 8) * 4)) & Int32(15)
                high = (highs[index // 16] >> Int32((index % 16) * 2)) & Int32(3)
                value = value | ((base + low + (high << Int32(4))) << Int32(byte * 8))
        _st_shared_i32(destination + first + Int32(word * 4), value)


class MoEGatedDynamicKernelSF6Words(_sf6.MoEGatedDynamicKernelSF6):
    def __init__(self, *args, **kwargs):
        if not stock_contract_matches():
            raise RuntimeError('long-prefill SF6 word decoder parent source drifted')
        super().__init__(*args, **kwargs)
        if not self.reform_sf_pack:
            raise ValueError('prefill word unpack requires prepared SF6 scales')
        self.prefill_word_unpack = True

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
                    Int32(k_tile) & Int32(1), lane, False, self.prefill_word_unpack,
                )
                _sf6_expand_dynamic_tile(
                    packed_base + up_block * Int64(1552), up_addr,
                    Int32(k_tile) & Int32(1), lane, False, self.prefill_word_unpack,
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
            output_tile_idx & Int32(1), lane, True, self.prefill_word_unpack,
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
