"""GLM TP4 prefill reuse candidate for the M128 gated NVFP4 backend.

Selected only by VLLM_GLM53_B12X_PREFILL_REUSE=1 and the exact E288/K4096/
I512/top8/SM121 contract in moe_dispatch. Small dynamic tiles and every
static/micro call retain their existing implementation.

Changes relative to the pinned upstream body:
* Q0 aliases its eight-row (64 KiB) staging region across startup-idle sC
  and sA; route metadata moves to startup-idle sB. All eight math warps
  work instead of four, with only the 8-byte q0_bulk_barrier shared word
  added on top of aliased storage.
* A nine-warp scan replaces the 288-expert serial prefix loop, and eight
  lanes reserve a token's expert rows concurrently.
* Per-token route scales are compared once, outside the Q0 block loop.
* Each of four Q1 A/SFA register fragments is loaded once per task and
  reused across all 32 FC2 output tiles, instead of 32 times per slice.

Weight/activation quantization, MMA order, BF16 conversion, barriers and
weighted atomic scatter are preserved. Route reservation order may change
BF16 atomic addition order, as it already does between stock executions.
Extra FC2 register pressure can offset the removed loads; GPU correctness,
spill/occupancy and throughput validation remain mandatory before promotion.
No numerical or speed claim follows from source-level load-count reductions.

The inherited FC1/FC2 helpers and storage contract are pinned to the source
hash below. Dispatch fails closed to stock when that source changes.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync
from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32, fabs_f32, fmax_f32, rcp_approx_ftz,
    quantize_block_fp4, quantize_block_fp4_fast, get_ptr_as_int64,
    ld_shared_i32_relaxed, st_global_f32, st_global_i32,
    shared_ptr_to_u32, st_shared_i32, st_global_v4_u32,
)
from ._moe_dynamic import gated as _stock
from ._moe_dynamic.gated import (
    DynamicLaunchParams, MoEGatedDynamicKernel, _TASK_SLICE_CHUNK,
    _ld_shared_i32, _st_shared_i32, _ld_global_acquire_i32,
    _threadfence, atomic_add_shared_i32,
    q0_bulk_barrier_init, q0_cp_async_bulk,
    q0_bulk_arrive_expect_tx, q0_bulk_try_wait,
    load_shared_bf16x16_to_f32x16, st_global_u64_adaptive_l2,
)

STOCK_GATED_SHA256 = "993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445"


@lru_cache(maxsize=1)
def stock_contract_matches() -> bool:
    """No CUDA work: inherited methods must match the reviewed storage ABI."""
    try:
        return hashlib.sha256(Path(_stock.__file__).read_bytes()).hexdigest() == STOCK_GATED_SHA256
    except (OSError, TypeError):
        return False


class MoEGatedPrefillReuseKernel(MoEGatedDynamicKernel):
    """Pinned gated backend with startup and task-local operand reuse."""

    def _setup_attributes(self, hidden_size: int):
        # This also rejects direct construction outside the guarded dispatcher:
        # the startup alias below has capacity for eight 4096-wide BF16 rows.
        if hidden_size != 4096 or self.tile_shape_mnk != (128, 128, 128):
            raise ValueError("prefill reuse requires K4096 and M128/N128/K128 tiles")
        if not stock_contract_matches():
            raise RuntimeError("prefill reuse inherited gated source has drifted")
        super()._setup_attributes(hidden_size)

    @cute.jit
    def initialize_route_q0_and_publish(
        self,
        thread_info,
        route_inputs,
        route_outputs,
        routing_state,
        task_queue,
        resident_barriers,
        shared_addresses,
        launch_params: DynamicLaunchParams,
    ):
        tidx, bidz, gdim_z, warp_idx, is_cta_leader = thread_info
        a_input, topk_ids, topk_weights, input_global_scale = route_inputs
        (
            packed_a_storage,
            scale_storage,
            scatter_output,
            token_map,
            token_weights,
        ) = route_outputs
        expert_write_rows, expert_tile_base, pair_head = routing_state
        (
            task_head,
            task_tail,
            task_expert,
            task_valid_rows,
        ) = task_queue
        barrier_count, barrier_epoch = resident_barriers
        (
            ctrl_base_addr,
            route_phys_rows_addr,
            route_expert_ids_addr,
            q0_input_stage_base_addr,
            q0_bulk_barrier_addr,
        ) = shared_addresses

        num_tokens = Int32(a_input.shape[0])
        cols = Int32(a_input.shape[1])
        scatter_base = scatter_output.iterator.toint()
        row_counts = launch_params.row_counts
        num_experts = Int32(row_counts.shape[0])
        sf_blocks_per_row = cols // Int32(16)
        output_bytes_per_row = cols // Int32(2)
        cols_u32 = cols // Int32(2)
        scatter_output_u32 = cute.recast_tensor(scatter_output, cutlass.Uint32)
        total_pairs = Int32(topk_ids.shape[0])
        num_topk = total_pairs // num_tokens
        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)
        num_k_tiles = (cols + Int32(63)) // Int32(64)
        route_gate_tile_cnt = launch_params.gate_tile_cnt
        task_slice_chunk = Int32(_TASK_SLICE_CHUNK)
        if num_tokens <= Int32(2048):
            task_slice_chunk = Int32(2)

        # Phase 0: cooperative init — zero routing state, queue state, and output.
        i = flat_tid
        while i < num_experts:
            row_counts[i] = Int32(0)
            expert_write_rows[i] = Int32(0)
            i += flat_stride
        if flat_tid < num_experts + Int32(1):
            expert_tile_base[flat_tid] = Int32(0)

        scatter_total_u32 = num_tokens * cols_u32
        scatter_vecs = scatter_total_u32 // Int32(4)
        zero_u32 = Uint32(0)
        zv = flat_tid
        while zv < scatter_vecs:
            st_global_v4_u32(
                scatter_base + Int64(zv) * Int64(16),
                zero_u32,
                zero_u32,
                zero_u32,
                zero_u32,
            )
            zv += flat_stride

        j = scatter_vecs * Int32(4) + flat_tid
        while j < scatter_total_u32:
            scatter_output_u32[j // cols_u32, j % cols_u32] = Uint32(0)
            j += flat_stride

        if flat_tid == Int32(0):
            pair_head[Int32(0)] = Int32(0)
            task_head[Int32(0)] = Int32(0)
            task_tail[Int32(0)] = Int32(0)

        cute.arch.sync_threads()
        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # Phase 1: aggregate routed rows per CTA before publishing the
        # 288 expert subtotals globally. The first 2304 bytes of startup-idle
        # sB hold route caches; the following 1152 bytes hold the histogram.
        route_hist_addr = route_expert_ids_addr + Int32(
            (self.num_mma_warps + 1) * 32 * 4
        )
        hist_bin = tidx
        while hist_bin < num_experts:
            st_shared_i32(route_hist_addr + hist_bin * Int32(4), Int32(0))
            hist_bin += Int32((self.num_mma_warps + 1) * 32)
        cute.arch.sync_threads()

        hist_idx = flat_tid
        while hist_idx < total_pairs:
            expert_id = topk_ids[hist_idx].to(Int32)
            atomic_add_shared_i32(route_hist_addr + expert_id * Int32(4), Int32(1))
            hist_idx += flat_stride
        cute.arch.sync_threads()

        hist_bin = tidx
        while hist_bin < num_experts:
            subtotal = ld_shared_i32_relaxed(route_hist_addr + hist_bin * Int32(4))
            if subtotal > Int32(0):
                atomic_add_global_i32(get_ptr_as_int64(row_counts, hist_bin), subtotal)
            hist_bin += Int32((self.num_mma_warps + 1) * 32)

        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        if num_experts == Int32(288) and bidz == Int32(0):
            # 288 experts exactly fill this 9-warp CTA. Only block zero
            # enters, and all of its threads participate in both barriers.
            prefix_lane = Int32(tidx) & Int32(31)
            rows = row_counts[tidx]
            tile_count = (rows + Int32(127)) // Int32(128)
            inclusive = tile_count
            for scan_stage in cutlass.range_constexpr(5):
                offset = Int32(1 << scan_stage)
                prior = cute.arch.shuffle_sync(inclusive, prefix_lane - offset)
                if prefix_lane >= offset:
                    inclusive += prior
            exclusive = inclusive - tile_count
            if prefix_lane == Int32(31):
                st_shared_i32(route_hist_addr + warp_idx * Int32(4), inclusive)
            cute.arch.sync_threads()
            if warp_idx == Int32(0):
                total = Int32(0)
                if prefix_lane < Int32(9):
                    total = ld_shared_i32_relaxed(route_hist_addr + prefix_lane * Int32(4))
                warp_prefix = total
                for scan_stage in cutlass.range_constexpr(5):
                    offset = Int32(1 << scan_stage)
                    prior = cute.arch.shuffle_sync(warp_prefix, prefix_lane - offset)
                    if prefix_lane >= offset:
                        warp_prefix += prior
                if prefix_lane < Int32(9):
                    st_shared_i32(route_hist_addr + prefix_lane * Int32(4), warp_prefix - total)
                if prefix_lane == Int32(8):
                    _st_shared_i32(ctrl_base_addr + Int32(0), warp_prefix)
            cute.arch.sync_threads()
            base = ld_shared_i32_relaxed(route_hist_addr + warp_idx * Int32(4))
            expert_tile_base[tidx] = base + exclusive
            if tidx == Int32(0):
                expert_tile_base[num_experts] = _ld_shared_i32(ctrl_base_addr + Int32(0))
        elif (
            num_experts == Int32(256)
            and bidz == Int32(0)
            and warp_idx < Int32(self.num_mma_warps)
        ):
            prefix_lane = Int32(tidx) & Int32(31)
            rows = row_counts[tidx]
            tile_count = (rows + Int32(self.tile_shape_mnk[0]) - Int32(1)) // Int32(
                self.tile_shape_mnk[0]
            )

            warp_inclusive = tile_count
            for scan_stage in cutlass.range_constexpr(5):
                scan_offset = Int32(1 << scan_stage)
                scan_value = cute.arch.shuffle_sync(
                    warp_inclusive, prefix_lane - Int32(scan_offset)
                )
                if prefix_lane >= Int32(scan_offset):
                    warp_inclusive += scan_value
            warp_exclusive = warp_inclusive - tile_count

            if prefix_lane == Int32(31):
                st_shared_i32(
                    route_hist_addr + warp_idx * Int32(4),
                    warp_inclusive,
                )
            self.epilog_sync_barrier.arrive_and_wait()

            if warp_idx == Int32(0):
                warp_total = Int32(0)
                if prefix_lane < Int32(self.num_mma_warps):
                    warp_total = ld_shared_i32_relaxed(
                        route_hist_addr + prefix_lane * Int32(4)
                    )
                warp_sum_inclusive = warp_total
                for scan_stage in cutlass.range_constexpr(5):
                    scan_offset = Int32(1 << scan_stage)
                    scan_value = cute.arch.shuffle_sync(
                        warp_sum_inclusive,
                        prefix_lane - Int32(scan_offset),
                    )
                    if prefix_lane >= Int32(scan_offset):
                        warp_sum_inclusive += scan_value
                if prefix_lane < Int32(self.num_mma_warps):
                    st_shared_i32(
                        route_hist_addr + prefix_lane * Int32(4),
                        warp_sum_inclusive - warp_total,
                    )
                if prefix_lane == Int32(self.num_mma_warps - 1):
                    _st_shared_i32(ctrl_base_addr + Int32(0), warp_sum_inclusive)
            self.epilog_sync_barrier.arrive_and_wait()

            warp_base = ld_shared_i32_relaxed(route_hist_addr + warp_idx * Int32(4))
            expert_tile_base[tidx] = warp_base + warp_exclusive
            if tidx == Int32(0):
                expert_tile_base[num_experts] = _ld_shared_i32(
                    ctrl_base_addr + Int32(0)
                )
        elif num_experts != Int32(256) and flat_tid == Int32(0):
            tile_acc = Int32(0)
            expert_idx = Int32(0)
            while expert_idx < num_experts:
                expert_tile_base[expert_idx] = tile_acc
                rows = row_counts[expert_idx]
                tile_acc += (rows + Int32(self.tile_shape_mnk[0]) - Int32(1)) // Int32(
                    self.tile_shape_mnk[0]
                )
                expert_idx += Int32(1)
            expert_tile_base[num_experts] = tile_acc

        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # Phase 2: the TMA warp stages eight contiguous BF16 rows across
        # startup-idle sC and sA. Each of the eight math warps quantizes one.
        if tidx == Int32(0):
            q0_bulk_barrier_init(q0_bulk_barrier_addr)
        cute.arch.sync_threads()
        lane_id = Int32(tidx) & Int32(31)
        _num_cta_warps = Int32(self.num_mma_warps + 1)
        # pair_head is a token counter; eight complete rows fit in this
        # subclass's checked 64 KiB startup staging span.
        producer_batch_tokens = Int32(8)
        # The subclass's storage alias covers eight complete K=4096 rows;
        # every producer warp now owns one token in the batch.
        if producer_batch_tokens > Int32(self.num_mma_warps):
            producer_batch_tokens = Int32(self.num_mma_warps)
        shared_input_gs_value = cutlass.Float32(0.0)
        if cutlass.const_expr(self.share_input_across_experts):
            shared_input_gs_value = input_global_scale[Int32(0)].to(cutlass.Float32)
            if (
                self.input_scales_are_reciprocal
                and shared_input_gs_value != cutlass.Float32(0.0)
            ):
                if self.fast_math:
                    shared_input_gs_value = rcp_approx_ftz(shared_input_gs_value)
                else:
                    shared_input_gs_value = cutlass.Float32(1.0) / shared_input_gs_value
        pair_idx = Int32(0)
        expert_id = Int32(0)
        token_idx = Int32(0)
        weight = cutlass.Float32(0.0)
        row = Int32(0)
        phys_tile = Int32(0)
        phys_row = Int32(0)
        produce_active = Int32(1)
        q0_bulk_phase = Int32(0)
        while produce_active > Int32(0):
            batch_base = Int32(0)
            if is_cta_leader > Int32(0):
                claim_count = producer_batch_tokens
                batch_base = atomic_add_global_i32(
                    get_ptr_as_int64(pair_head, Int32(0)),
                    claim_count,
                )
                _st_shared_i32(ctrl_base_addr + Int32(28), batch_base)
            cute.arch.sync_threads()
            batch_base = _ld_shared_i32(ctrl_base_addr + Int32(28))
            producer_limit = num_tokens
            if batch_base >= producer_limit:
                produce_active = Int32(0)
            else:
                staged_tokens = num_tokens - batch_base
                if staged_tokens > producer_batch_tokens:
                    staged_tokens = producer_batch_tokens
                first_copy_tokens = staged_tokens
                if first_copy_tokens > Int32(4):
                    first_copy_tokens = Int32(4)
                second_copy_tokens = staged_tokens - first_copy_tokens
                first_copy_bytes = first_copy_tokens * cols * Int32(2)
                second_copy_bytes = second_copy_tokens * cols * Int32(2)
                if warp_idx == Int32(self.num_mma_warps):
                    if lane_id == Int32(0):
                        input_batch_addr = Int64(a_input.iterator.toint()) + Int64(
                            batch_base
                        ) * Int64(cols) * Int64(2)
                        if first_copy_bytes > Int32(0):
                            q0_cp_async_bulk(
                                q0_input_stage_base_addr,
                                input_batch_addr,
                                first_copy_bytes,
                                q0_bulk_barrier_addr,
                            )
                        if second_copy_bytes > Int32(0):
                            q0_cp_async_bulk(
                                q0_input_stage_base_addr + Int32(4) * cols * Int32(2),
                                input_batch_addr + Int64(4) * Int64(cols) * Int64(2),
                                second_copy_bytes,
                                q0_bulk_barrier_addr,
                            )
                        q0_bulk_arrive_expect_tx(
                            q0_bulk_barrier_addr,
                            first_copy_bytes + second_copy_bytes,
                        )

                if cutlass.const_expr(self.share_input_across_experts):
                    token_idx = batch_base + warp_idx
                    if warp_idx < producer_batch_tokens and token_idx < num_tokens:
                        route_slot_base = warp_idx * Int32(32)
                        # Each lane reserves a different top-k expert. The
                        # same warp barrier publishes all eight route slots.
                        topk_slot = lane_id
                        if topk_slot < num_topk:
                            pair_idx = token_idx * num_topk + topk_slot
                            expert_id = topk_ids[pair_idx].to(Int32)
                            weight = topk_weights[pair_idx].to(cutlass.Float32)
                            row = atomic_add_global_i32(
                                get_ptr_as_int64(expert_write_rows, expert_id),
                                Int32(1),
                            )
                            phys_tile = expert_tile_base[expert_id] + row // Int32(
                                self.tile_shape_mnk[0]
                            )
                            phys_row = phys_tile * Int32(
                                self.tile_shape_mnk[0]
                            ) + row % Int32(self.tile_shape_mnk[0])
                            st_global_i32(
                                get_ptr_as_int64(token_map, phys_row), token_idx
                            )
                            st_global_f32(
                                get_ptr_as_int64(token_weights, phys_row), weight
                            )
                            slot = route_slot_base + topk_slot
                            _st_shared_i32(
                                route_phys_rows_addr + slot * Int32(4), phys_row
                            )
                            _st_shared_i32(
                                route_expert_ids_addr + slot * Int32(4), expert_id
                            )
                        cute.arch.sync_warp()
                        q0_ready = q0_bulk_try_wait(q0_bulk_barrier_addr, q0_bulk_phase)
                        while q0_ready == Int32(0):
                            q0_ready = q0_bulk_try_wait(
                                q0_bulk_barrier_addr, q0_bulk_phase
                            )

                        gs_value = shared_input_gs_value
                        if num_topk == Int32(8):
                            route_output_base = cute.make_rmem_tensor((8,), Int32)
                            route_scale_base = cute.make_rmem_tensor((8,), Int32)
                            for cache_slot in cutlass.range_constexpr(8):
                                slot = route_slot_base + Int32(cache_slot)
                                phys_row = _ld_shared_i32(
                                    route_phys_rows_addr + slot * Int32(4)
                                )
                                phys_tile = phys_row // Int32(self.tile_shape_mnk[0])
                                tile_row = phys_row - phys_tile * Int32(
                                    self.tile_shape_mnk[0]
                                )
                                route_output_base[cache_slot] = (
                                    phys_row * output_bytes_per_row
                                )
                                route_scale_base[cache_slot] = (
                                    phys_tile * num_k_tiles * Int32(32 * 4 * 4)
                                    + (tile_row % Int32(32)) * Int32(4 * 4)
                                    + ((tile_row % Int32(32 * 4)) // Int32(32))
                                    * Int32(4)
                                )

                            sf_idx = lane_id
                            while sf_idx < sf_blocks_per_row:
                                block_start = sf_idx * Int32(16)
                                loaded_values = load_shared_bf16x16_to_f32x16(
                                    q0_input_stage_base_addr
                                    + warp_idx * cols * Int32(2)
                                    + block_start * Int32(2)
                                )
                                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = loaded_values[elem_idx]
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                k_tile_idx = sf_idx // Int32(4)
                                scale_k_base = k_tile_idx * Int32(32 * 4 * 4) + (
                                    sf_idx % Int32(4)
                                )
                                for cache_slot in cutlass.range_constexpr(8):
                                    output_offset = route_output_base[
                                        cache_slot
                                    ] + sf_idx * Int32(8)
                                    st_global_u64_adaptive_l2(
                                        num_tokens,
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    scale_storage[
                                        route_scale_base[cache_slot] + scale_k_base
                                    ] = scale_byte
                                sf_idx += Int32(32)
                        else:
                            sf_idx = lane_id
                            while sf_idx < sf_blocks_per_row:
                                block_start = sf_idx * Int32(16)
                                loaded_values = load_shared_bf16x16_to_f32x16(
                                    q0_input_stage_base_addr
                                    + warp_idx * cols * Int32(2)
                                    + block_start * Int32(2)
                                )
                                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = loaded_values[elem_idx]
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                topk_slot = Int32(0)
                                while topk_slot < num_topk:
                                    slot = route_slot_base + topk_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr + slot * Int32(4)
                                    )
                                    phys_tile = phys_row // Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    tile_row = phys_row - phys_tile * Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    output_offset = (
                                        phys_row * output_bytes_per_row
                                        + sf_idx * Int32(8)
                                    )
                                    st_global_u64_adaptive_l2(
                                        num_tokens,
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    k_tile_idx = sf_idx // Int32(4)
                                    outer_m_idx = tile_row % Int32(32)
                                    inner_m_idx = (tile_row % Int32(32 * 4)) // Int32(
                                        32
                                    )
                                    inner_k_idx = sf_idx % Int32(4)
                                    scale_offset = (
                                        phys_tile * num_k_tiles * Int32(32 * 4 * 4)
                                        + k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    scale_storage[scale_offset] = scale_byte
                                    topk_slot += Int32(1)
                                sf_idx += Int32(32)

                else:
                    # Each math warp owns one token and handles all of its
                    # routes.  Keep a 16-entry register cache so both the
                    # Qwen topk=8 and topk=10 shapes use this shared-load path.
                    token_idx = batch_base + warp_idx
                    if warp_idx < producer_batch_tokens and token_idx < num_tokens:
                        route_slot_base = warp_idx * Int32(32)
                        # Each lane reserves a different top-k expert. The
                        # same warp barrier publishes all eight route slots.
                        topk_slot = lane_id
                        if topk_slot < num_topk:
                            pair_idx = token_idx * num_topk + topk_slot
                            expert_id = topk_ids[pair_idx].to(Int32)
                            weight = topk_weights[pair_idx].to(cutlass.Float32)
                            row = atomic_add_global_i32(
                                get_ptr_as_int64(expert_write_rows, expert_id),
                                Int32(1),
                            )
                            phys_tile = expert_tile_base[expert_id] + row // Int32(
                                self.tile_shape_mnk[0]
                            )
                            phys_row = phys_tile * Int32(
                                self.tile_shape_mnk[0]
                            ) + row % Int32(self.tile_shape_mnk[0])
                            st_global_i32(
                                get_ptr_as_int64(token_map, phys_row), token_idx
                            )
                            st_global_f32(
                                get_ptr_as_int64(token_weights, phys_row), weight
                            )

                            route_slot = route_slot_base + topk_slot
                            _st_shared_i32(
                                route_phys_rows_addr + route_slot * Int32(4),
                                phys_row,
                            )
                            _st_shared_i32(
                                route_expert_ids_addr + route_slot * Int32(4),
                                expert_id,
                            )

                        cute.arch.sync_warp()
                        q0_ready = q0_bulk_try_wait(q0_bulk_barrier_addr, q0_bulk_phase)
                        while q0_ready == Int32(0):
                            q0_ready = q0_bulk_try_wait(
                                q0_bulk_barrier_addr, q0_bulk_phase
                            )

                        # Preserve the baseline's per-lane scale load and
                        # reciprocal work.  Hoist it out of the block loop,
                        # but do not introduce a 32x broadcast optimization.
                        route_gs = cute.make_rmem_tensor((16,), cutlass.Float32)
                        cache_slot = Int32(0)
                        while cache_slot < num_topk:
                            route_slot = route_slot_base + cache_slot
                            expert_id = _ld_shared_i32(
                                route_expert_ids_addr + route_slot * Int32(4)
                            )
                            gs_value = input_global_scale[expert_id].to(cutlass.Float32)
                            if (
                                self.input_scales_are_reciprocal
                                and gs_value != cutlass.Float32(0.0)
                            ):
                                if self.fast_math:
                                    gs_value = rcp_approx_ftz(gs_value)
                                else:
                                    gs_value = cutlass.Float32(1.0) / gs_value
                            route_gs[cache_slot] = gs_value
                            cache_slot += Int32(1)

                        # Quantized payload is identical only when all
                        # selected experts use the same input global scale.
                        route_scales_equal = Int32(1)
                        scale_idx = Int32(1)
                        while scale_idx < num_topk:
                            if route_gs[scale_idx] != route_gs[0]:
                                route_scales_equal = Int32(0)
                            scale_idx += Int32(1)

                        sf_idx = lane_id
                        while sf_idx < sf_blocks_per_row:
                            block_start = sf_idx * Int32(16)
                            loaded_values = load_shared_bf16x16_to_f32x16(
                                q0_input_stage_base_addr
                                + warp_idx * cols * Int32(2)
                                + block_start * Int32(2)
                            )
                            values = cute.make_rmem_tensor((16,), cutlass.Float32)
                            block_max = cutlass.Float32(0.0)
                            for elem_idx in cutlass.range_constexpr(16):
                                value = loaded_values[elem_idx]
                                values[elem_idx] = value
                                block_max = fmax_f32(block_max, fabs_f32(value))

                            if route_scales_equal > Int32(0):
                                gs_value = route_gs[0]
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                cache_slot = Int32(0)
                                while cache_slot < num_topk:
                                    route_slot = route_slot_base + cache_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr + route_slot * Int32(4)
                                    )
                                    phys_tile = phys_row // Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    tile_row = phys_row - phys_tile * Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    output_offset = (
                                        phys_row * output_bytes_per_row
                                        + sf_idx * Int32(8)
                                    )
                                    st_global_u64_adaptive_l2(
                                        num_tokens,
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    k_tile_idx = sf_idx // Int32(4)
                                    outer_m_idx = tile_row % Int32(32)
                                    inner_m_idx = (tile_row % Int32(32 * 4)) // Int32(
                                        32
                                    )
                                    inner_k_idx = sf_idx % Int32(4)
                                    scale_offset = (
                                        phys_tile * num_k_tiles * Int32(32 * 4 * 4)
                                        + k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    scale_storage[scale_offset] = scale_byte
                                    cache_slot += Int32(1)
                            else:
                                # Preserve independent quant/store operations;
                                # only the BF16 load and absmax are shared.
                                cache_slot = Int32(0)
                                while cache_slot < num_topk:
                                    route_slot = route_slot_base + cache_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr + route_slot * Int32(4)
                                    )
                                    phys_tile = phys_row // Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    tile_row = phys_row - phys_tile * Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    gs_value = route_gs[cache_slot]

                                    packed64 = Uint64(0)
                                    scale_byte = Uint8(0)
                                    if self.fast_math:
                                        packed64, scale_byte = quantize_block_fp4_fast(
                                            values, block_max, gs_value
                                        )
                                    else:
                                        packed64, scale_byte = quantize_block_fp4(
                                            values, block_max, gs_value
                                        )

                                    output_offset = (
                                        phys_row * output_bytes_per_row
                                        + sf_idx * Int32(8)
                                    )
                                    st_global_u64_adaptive_l2(
                                        num_tokens,
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    k_tile_idx = sf_idx // Int32(4)
                                    outer_m_idx = tile_row % Int32(32)
                                    inner_m_idx = (tile_row % Int32(32 * 4)) // Int32(
                                        32
                                    )
                                    inner_k_idx = sf_idx % Int32(4)
                                    scale_offset = (
                                        phys_tile * num_k_tiles * Int32(32 * 4 * 4)
                                        + k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    scale_storage[scale_offset] = scale_byte
                                    cache_slot += Int32(1)
                            sf_idx += Int32(32)

                q0_bulk_phase = Int32(1) - q0_bulk_phase

        cute.arch.sync_threads()
        # Conservative publish fence before the last-producer CTA flushes any
        # partial tiles. All producer threads in the CTA must have ordered
        # their global writes before lane 0 can publish work.
        _threadfence()
        cute.arch.sync_threads()

        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        total_m_tiles = expert_tile_base[num_experts]
        split_groups = (route_gate_tile_cnt + Int32(1)) // Int32(2)
        extra_per_split = split_groups - Int32(1)
        split_tile_count = Int32(0)
        if extra_per_split > Int32(0):
            if num_tokens > Int32(256):
                if num_tokens <= Int32(4096):
                    target_task_count = Int32(4) * Int32(gdim_z)
                    if num_tokens > Int32(2048):
                        target_task_count = (
                            Int32(125) * Int32(gdim_z) + Int32(31)
                        ) // Int32(32)
                    missing_tasks = target_task_count - total_m_tiles
                    if missing_tasks > Int32(0):
                        split_tile_count = (
                            missing_tasks + extra_per_split - Int32(1)
                        ) // extra_per_split
                        if split_tile_count > total_m_tiles:
                            split_tile_count = total_m_tiles

        if is_cta_leader > Int32(0):
            expert_flush = Int32(bidz)
            while expert_flush < num_experts:
                rows_remaining = row_counts[expert_flush]
                m_tile_offset = Int32(0)
                while rows_remaining > Int32(0):
                    valid_rows = rows_remaining
                    if valid_rows > Int32(self.tile_shape_mnk[0]):
                        valid_rows = Int32(self.tile_shape_mnk[0])
                    if num_tokens <= Int32(256):
                        self.publish_uniform_deferred_tasks(
                            task_expert,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            task_slice_chunk,
                            expert_flush,
                            expert_tile_base[expert_flush] + m_tile_offset,
                            valid_rows,
                        )
                    elif num_tokens <= Int32(4096):
                        self.publish_variable_deferred_tasks(
                            task_expert,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            split_tile_count,
                            expert_flush,
                            expert_tile_base[expert_flush] + m_tile_offset,
                            valid_rows,
                        )
                    else:
                        self.publish_uniform_deferred_tasks(
                            task_expert,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            task_slice_chunk,
                            expert_flush,
                            expert_tile_base[expert_flush] + m_tile_offset,
                            valid_rows,
                        )
                    rows_remaining -= Int32(self.tile_shape_mnk[0])
                    m_tile_offset += Int32(1)
                expert_flush += Int32(gdim_z)

        if flat_tid == Int32(0):
            uniform_groups = (
                route_gate_tile_cnt + task_slice_chunk - Int32(1)
            ) // task_slice_chunk
            published_task_count = expert_tile_base[num_experts] * uniform_groups
            if num_tokens > Int32(256):
                if num_tokens <= Int32(4096):
                    published_task_count = (
                        expert_tile_base[num_experts]
                        + split_tile_count * extra_per_split
                    )
            st_global_i32(
                get_ptr_as_int64(task_tail, Int32(0)),
                published_task_count,
            )

        self.resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
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
        tma_sfb_w13: cute.CopyAtom,
        mSFB_w13: cute.Tensor,
        tma_b_down: cute.CopyAtom,
        mB_down: cute.Tensor,
        tma_sfb_down: cute.CopyAtom,
        mSFB_down: cute.Tensor,
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
            cpasync.prefetch_descriptor(tma_sfb_w13)
            cpasync.prefetch_descriptor(tma_b_down)
            cpasync.prefetch_descriptor(tma_sfb_down)

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
            + cute.size_in_bytes(self.sf_dtype, fc1_sfb_smem_one)
        )
        fc1_branch_tma_copy_bytes = fc1_tma_copy_bytes
        if cutlass.const_expr(not sequential_branch_compact):
            fc1_tma_copy_bytes += cute.size_in_bytes(
                self.b_dtype, fc1_b_smem_one
            ) + cute.size_in_bytes(self.sf_dtype, fc1_sfb_smem_one)
        phase2_tma_copy_bytes = cute.size_in_bytes(
            self.b_dtype, b_smem_one
        ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)

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
            # N64 lane: FC1 B is two N64 halves of sB (sB_fc1_all); sC stays
            # the FC2/epilogue buffer until activation writes it. N128 lane:
            # sC[0:16KiB] is Up's two complete N128 B stages, Gate keeps sB.
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
        if cutlass.const_expr(getattr(self, "prefill_fc1_n128", False)):
            if cutlass.const_expr(
                self.ab_stage != 2
                or cute.size_in_bytes(self.b_dtype, fc1_b_smem_staged) != 16384
                or cute.size_in_bytes(self.b_dtype, b_smem_staged) < 16384
                or cute.size_in_bytes(cutlass.BFloat16, epi_smem_staged) < 16384
            ):
                raise ValueError("prefill N128 FC1 shared-storage contract changed")
            # The wide FC1 variant uses two complete N128 B stages per
            # branch. Gate owns sB[0:16KiB], Up owns sC[0:16KiB]. Its producer
            # waits before next-slice Stage0; the consumer waits for every
            # math warp before activation/Q1 reclaims sC. No allocation grows.
            sB_fc1 = storage.sB.get_tensor(
                fc1_b_smem_staged.outer, swizzle=fc1_b_smem_staged.inner,
            )
            # sC is allocated as BF16; recast its pointer before applying
            # the NVFP4 B layout so its offsets remain half-byte elements.
            sB_up_fc1_ptr = cute.recast_ptr(
                storage.sC.data_ptr(), fc1_b_smem_staged.inner,
                dtype=self.b_dtype,
            )
            sB_up_fc1 = cute.make_tensor(
                sB_up_fc1_ptr, fc1_b_smem_staged.outer,
            )
        else:
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
        # extra. The sC "extra" one-stage alias below is selected only when
        # the FC1 pipeline index exceeds the two allocated Up SFB stages --
        # the N64 lane's third stage, into sC while FC1 still owns it. The
        # N128 lane pins ab_storage_stage = 2, so its index cycles 0..1 and
        # the alias is unreachable by construction; it MUST stay unreachable
        # there, because sC is live Up-B backing in that lane.
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
        # Startup Q0 alone owns these aliases. sC (32 KiB) followed by
        # sA (40 KiB) is a contiguous 72 KiB span: stage eight K4096 BF16
        # rows (64 KiB). Metadata (3.5 KiB including histogram) occupies
        # idle sB (16 KiB). The resident publish barrier ends all aliases
        # before the original FC1/FC2 storage lifetimes begin.
        if cutlass.const_expr(
            cute.size_in_bytes(cutlass.BFloat16, epi_smem_staged) != 32768
            or cute.size_in_bytes(self.a_dtype, a_smem_staged) < 32768
            or cute.size_in_bytes(self.b_dtype, b_smem_staged) < 3456
        ):
            raise ValueError("prefill reuse Q0 shared-storage contract changed")
        route_phys_rows_addr = shared_ptr_to_u32(storage.sB.data_ptr())
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
        gSFB_w13_tiled = cute.local_tile(
            mSFB_w13,
            self.fc1_sfb_tile_shape_nk,
            (None, None, None),
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
        tBsSFB_w13, tBgSFB_w13 = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_fc1, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13_up, _tBgSFB_w13_up = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_up_fc1, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13_up_extra, _tBgSFB_w13_up_extra = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_up_fc1_extra, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13 = cute.filter_zeros(tBsSFB_w13)
        tBgSFB_w13 = cute.filter_zeros(tBgSFB_w13)
        tBsSFB_w13_up = cute.filter_zeros(tBsSFB_w13_up)
        tBsSFB_w13_up_extra = cute.filter_zeros(tBsSFB_w13_up_extra)

        # B_down TMA partitions
        gB_down = cute.local_tile(
            mB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFB_down = cute.local_tile(
            mSFB_down,
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
        tBsSFB_down, tBgSFB_down = cpasync.tma_partition(
            tma_sfb_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_phase2, 0, 2),
            cute.group_modes(gSFB_down, 0, 2),
        )
        tBsSFB_down = cute.filter_zeros(tBsSFB_down)
        tBgSFB_down = cute.filter_zeros(tBgSFB_down)

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
        if cutlass.const_expr(getattr(self, "prefill_fc1_n128", False)):
            gate_tile_cnt = cute.size(gB_w13_tiled, mode=[2]) // Int32(2)
        else:
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

                # Q1 is now immutable for this task. Preserve four packed
                # A/SFA fragments in registers across every FC2 output tile.
                # Loading all slices before the loop also keeps the original
                # shared aliases dead before producer stage2 starts to reuse A0.
                cached_a = (
                    tCrA,
                    cute.make_rmem_tensor(tCrA.layout, self.a_dtype),
                    cute.make_rmem_tensor(tCrA.layout, self.a_dtype),
                    cute.make_rmem_tensor(tCrA.layout, self.a_dtype),
                )
                cached_sfa = (
                    tCrSFA,
                    cute.make_rmem_tensor(tCrSFA.layout, self.sf_dtype),
                    cute.make_rmem_tensor(tCrSFA.layout, self.sf_dtype),
                    cute.make_rmem_tensor(tCrSFA.layout, self.sf_dtype),
                )
                for cache_idx in cutlass.range_constexpr(4):
                    if task_slice_count_val > Int32(cache_idx):
                        # These are the same retained-Q1 slots as stock.
                        q1_a_stage_idx = Int32((3, 4, 2, 1)[cache_idx])
                        q1_sfa_stage_idx = Int32((3, 1, 2, 0)[cache_idx])
                        self.load_fc2_a_fragments(
                            num_k_blocks,
                            q1_a_stage_idx,
                            q1_sfa_stage_idx,
                            (csA, csSFA),
                            (thr_ld_A.retile(cached_a[cache_idx]),
                             thr_ld_SFA.retile(cached_sfa[cache_idx])),
                            (smem_copy_A, smem_copy_SFA),
                        )

                phase2_cons_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    physical_output_tile_idx = (
                        Int32(output_tile_idx) + task_expert_idx
                    ) % Int32(output_tile_cnt)
                    down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
                    down_acc.fill(0.0)
                    # Compile-time selection avoids dynamic indexing into
                    # register arrays; slice arithmetic order stays 0..3.
                    for slice_idx in cutlass.range_constexpr(4):
                        if task_slice_count_val > Int32(slice_idx):
                            # 39차: mma_atom.set(SFA/SFB) rebinds the atom's IR
                            # value in place. Stock runs the slices in one
                            # dynamic while-loop region, which the DSL threads;
                            # this static unroll makes four sibling scf.if
                            # regions, and the value set in one does not
                            # dominate the next ("operand #0 does not dominate
                            # this use" at atom.set_value, P2D/P2D2). A fresh
                            # atom per region keeps every set/gemm chain local.
                            slice_mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
                                self.a_dtype, self.acc_dtype, self.sf_dtype,
                            )
                            slice_atom = cute.make_mma_atom(slice_mma_op)
                            # then/else below are sibling regions too: an atom
                            # mutated in one must not be reused in the other
                            # (stock passes mma_atom / mma_atom_tail for the
                            # same reason).
                            slice_atom_tail = cute.make_mma_atom(slice_mma_op)
                            if valid_rows == Int32(self.tile_shape_mnk[0]):
                                phase2_cons_state = self.fc2_accumulate_slice(
                                    num_k_blocks,
                                    slice_atom,
                                    down_acc,
                                    (phase2_pipeline, phase2_cons_state),
                                    (csB, csB_phase2_extra, csSFB),
                                    (cached_a[slice_idx], tCrB,
                                     cached_sfa[slice_idx], tCrSFB, crB, crSFB),
                                    (smem_copy_B, smem_copy_SFB),
                                )
                            else:
                                phase2_cons_state = self.fc2_accumulate_slice_tail(
                                    num_k_blocks,
                                    slice_atom_tail,
                                    down_acc,
                                    valid_rows,
                                    warp_m_coord,
                                    (phase2_pipeline, phase2_cons_state),
                                    (csB, csB_phase2_extra, csSFB),
                                    (cached_a[slice_idx], tCrB,
                                     cached_sfa[slice_idx], tCrSFB, crB, crSFB),
                                    (smem_copy_B, smem_copy_SFB),
                                )

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
                        (tma_a, tma_b_w13, tma_sfa, tma_sfb_w13),
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
                            (tma_b_down, tma_sfb_down),
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


__all__ = ["MoEGatedPrefillReuseKernel", "stock_contract_matches"]
