"""TP SF6 Q0 cache with FP32 accumulation of rounded BF16 contributions.

The stock initialization/histogram/prefix and final task publication are retained. The
four-row Q0 producer reuses within-call scale/equality/address metadata; all
original eight TP routes, including zero-weight routes, remain allocated.
The shared dynamic workspace owns the FP32 scatter plane. Each weighted BF16
contribution is widened before atomic addition, with one final BF16 output cast.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32, Uint64
from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32, fabs_f32, fmax_f32, rcp_approx_ftz,
    quantize_block_fp4, quantize_block_fp4_fast, get_ptr_as_int64,
    ld_shared_i32_relaxed, st_global_f32, st_global_i32,
    st_shared_i32, st_shared_f32, st_global_v4_u32, st_global_u64,
)
from ._moe_dynamic.gated import (
    DynamicLaunchParams, _TASK_SLICE_CHUNK, _ld_shared_i32, _st_shared_i32,
    _threadfence, atomic_add_shared_i32, q0_bulk_barrier_init,
    q0_cp_async_bulk, q0_bulk_arrive_expect_tx, q0_bulk_try_wait,
    load_shared_bf16x16_to_f32x16,
)
from . import moe_dynamic_gated_sf6 as _sf6
from .moe_dynamic_ep_local import MoEGatedEPLocalKernel

SF6_SOURCE_SHA256 = "6efb0a2ec044dfbaeb43af92f569b6c130a99bee751fb5a129f78dac1183300e"


@lru_cache(maxsize=1)
def stock_contract_matches():
    return (_sf6.stock_contract_matches()
            and hashlib.sha256(Path(_sf6.__file__).read_bytes()).hexdigest() == SF6_SOURCE_SHA256)


class MoEGatedDynamicKernelSF6Q0(_sf6.MoEGatedDynamicKernelSF6):
    # The M128 epilogue has the same shared-memory layout in TP and EP.
    # Widen the atomic sum, preserving each route's BF16 multiply/rounding.
    scatter_sC_to_gmem = MoEGatedEPLocalKernel.scatter_sC_to_gmem

    def _setup_attributes(self, hidden_size):
        if (hidden_size != 4096 or self.tile_shape_mnk != (128,128,128)
                or not self.reform_sf_pack or self.share_input_across_experts
                or (self.activation,self.swiglu_alpha,self.swiglu_beta,self.swiglu_limit)
                   != ("swigluoai_uninterleave",1.,0.,10.)
                or self.num_mma_warps != 8 or self.threads_per_cta != 288):
            raise ValueError("TP SF6 Q0 requires exact unshared M128 GLM geometry")
        if not stock_contract_matches():
            raise RuntimeError("TP SF6 Q0 inherited source changed")
        super()._setup_attributes(hidden_size)
        # SF6.kernel aliases two 288-entry route caches into startup-idle sA;
        # the new 288-entry scale cache reuses its dead histogram directly
        # after them. No extra shared allocation or launch shape is needed.
        if (cute.size_in_bytes(self.a_dtype,self.a_smem_layout_staged) < 3*288*4
                or cute.size_in_bytes(cutlass.BFloat16,self.epi_smem_layout_staged) < 4*4096*2):
            raise ValueError("TP SF6 Q0 startup aliases exceed shared backing")

    def _check_sf6_shapes(self, w13, down, packed1, packed2):
        super()._check_sf6_shapes(w13,down,packed1,packed2)
        if (w13.shape[0] != 1024 or cute.size(w13.shape[1]) != 4096
                or w13.shape[2] != 288 or down.shape[0] != 4096
                or cute.size(down.shape[1]) != 512 or down.shape[2] != 288):
            raise ValueError("TP SF6 Q0 requires E288/H4096/I512")

    @cute.jit
    def initialize_route_q0_and_publish(self, thread_info, route_inputs, route_outputs, routing_state, task_queue, resident_barriers, shared_addresses, launch_params: DynamicLaunchParams):
        tidx, bidz, gdim_z, warp_idx, is_cta_leader = thread_info
        a_input, topk_ids, topk_weights, input_global_scale = route_inputs
        packed_a_storage, scale_storage, scatter_output, token_map, token_weights = route_outputs
        expert_write_rows, expert_tile_base, pair_head = routing_state
        task_head, task_tail, task_expert, task_valid_rows = task_queue
        barrier_count, barrier_epoch = resident_barriers
        ctrl_base_addr, route_phys_rows_addr, route_expert_ids_addr, q0_input_stage_base_addr, q0_bulk_barrier_addr = shared_addresses
        num_tokens = Int32(a_input.shape[0])
        cols = Int32(a_input.shape[1])
        scatter_base = scatter_output.iterator.toint()
        row_counts = launch_params.row_counts
        num_experts = Int32(row_counts.shape[0])
        sf_blocks_per_row = cols // Int32(16)
        output_bytes_per_row = cols // Int32(2)
        cols_u32 = cols  # FP32 scatter, including the whole zeroed output plane
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
            st_global_v4_u32(scatter_base + Int64(zv) * Int64(16), zero_u32, zero_u32, zero_u32, zero_u32)
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
        self.resident_grid_barrier(barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader)
        route_hist_addr = route_expert_ids_addr + Int32((self.num_mma_warps + 1) * 32 * 4)
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
        self.resident_grid_barrier(barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader)
        if num_experts == Int32(256) and bidz == Int32(0) and (warp_idx < Int32(self.num_mma_warps)):
            prefix_lane = Int32(tidx) & Int32(31)
            rows = row_counts[tidx]
            tile_count = (rows + Int32(self.tile_shape_mnk[0]) - Int32(1)) // Int32(self.tile_shape_mnk[0])
            warp_inclusive = tile_count
            for scan_stage in cutlass.range_constexpr(5):
                scan_offset = Int32(1 << scan_stage)
                scan_value = cute.arch.shuffle_sync(warp_inclusive, prefix_lane - Int32(scan_offset))
                if prefix_lane >= Int32(scan_offset):
                    warp_inclusive += scan_value
            warp_exclusive = warp_inclusive - tile_count
            if prefix_lane == Int32(31):
                st_shared_i32(route_hist_addr + warp_idx * Int32(4), warp_inclusive)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == Int32(0):
                warp_total = Int32(0)
                if prefix_lane < Int32(self.num_mma_warps):
                    warp_total = ld_shared_i32_relaxed(route_hist_addr + prefix_lane * Int32(4))
                warp_sum_inclusive = warp_total
                for scan_stage in cutlass.range_constexpr(5):
                    scan_offset = Int32(1 << scan_stage)
                    scan_value = cute.arch.shuffle_sync(warp_sum_inclusive, prefix_lane - Int32(scan_offset))
                    if prefix_lane >= Int32(scan_offset):
                        warp_sum_inclusive += scan_value
                if prefix_lane < Int32(self.num_mma_warps):
                    st_shared_i32(route_hist_addr + prefix_lane * Int32(4), warp_sum_inclusive - warp_total)
                if prefix_lane == Int32(self.num_mma_warps - 1):
                    _st_shared_i32(ctrl_base_addr + Int32(0), warp_sum_inclusive)
            self.epilog_sync_barrier.arrive_and_wait()
            warp_base = ld_shared_i32_relaxed(route_hist_addr + warp_idx * Int32(4))
            expert_tile_base[tidx] = warp_base + warp_exclusive
            if tidx == Int32(0):
                expert_tile_base[num_experts] = _ld_shared_i32(ctrl_base_addr + Int32(0))
        elif num_experts != Int32(256) and flat_tid == Int32(0):
            tile_acc = Int32(0)
            expert_idx = Int32(0)
            while expert_idx < num_experts:
                expert_tile_base[expert_idx] = tile_acc
                rows = row_counts[expert_idx]
                tile_acc += (rows + Int32(self.tile_shape_mnk[0]) - Int32(1)) // Int32(self.tile_shape_mnk[0])
                expert_idx += Int32(1)
            expert_tile_base[num_experts] = tile_acc
        self.resident_grid_barrier(barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader)
        expert_scales_addr = route_hist_addr
        route_scales_addr = route_expert_ids_addr
        if tidx < num_experts:
            gs_value = input_global_scale[tidx].to(cutlass.Float32)
            if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(0.0):
                if self.fast_math:
                    gs_value = rcp_approx_ftz(gs_value)
                else:
                    gs_value = cutlass.Float32(1.0) / gs_value
            st_shared_f32(expert_scales_addr + tidx * Int32(4), gs_value)
        if tidx == Int32(0):
            q0_bulk_barrier_init(q0_bulk_barrier_addr)
        cute.arch.sync_threads()
        lane_id = Int32(tidx) & Int32(31)
        _num_cta_warps = Int32(self.num_mma_warps + 1)
        producer_batch_tokens = Int32(self.tile_shape_mnk[0] * self.tile_shape_mnk[1]) // cols
        if producer_batch_tokens > Int32(self.num_mma_warps):
            producer_batch_tokens = Int32(self.num_mma_warps)
        pair_idx = Int32(0)
        expert_id = Int32(0)
        token_idx = Int32(0)
        weight = cutlass.Float32(0.0)
        row = Int32(0)
        phys_row = Int32(0)
        produce_active = Int32(1)
        q0_bulk_phase = Int32(0)
        while produce_active > Int32(0):
            batch_base = Int32(0)
            if is_cta_leader > Int32(0):
                claim_count = producer_batch_tokens
                batch_base = atomic_add_global_i32(get_ptr_as_int64(pair_head, Int32(0)), claim_count)
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
                        input_batch_addr = Int64(a_input.iterator.toint()) + Int64(batch_base) * Int64(cols) * Int64(2)
                        if first_copy_bytes > Int32(0):
                            q0_cp_async_bulk(q0_input_stage_base_addr, input_batch_addr, first_copy_bytes, q0_bulk_barrier_addr)
                        if second_copy_bytes > Int32(0):
                            q0_cp_async_bulk(q0_input_stage_base_addr + Int32(4) * cols * Int32(2), input_batch_addr + Int64(4) * Int64(cols) * Int64(2), second_copy_bytes, q0_bulk_barrier_addr)
                        q0_bulk_arrive_expect_tx(q0_bulk_barrier_addr, first_copy_bytes + second_copy_bytes)
                token_idx = batch_base + warp_idx
                if warp_idx < producer_batch_tokens and token_idx < num_tokens:
                    route_slot_base = warp_idx * Int32(32)
                    if lane_id == Int32(0):
                        topk_slot = Int32(0)
                        local_topk = Int32(0)
                        producer_first_gs = cutlass.Float32(0.0)
                        producer_scales_equal = Int32(1)
                        while topk_slot < num_topk:
                            pair_idx = token_idx * num_topk + topk_slot
                            expert_id = topk_ids[pair_idx].to(Int32)
                            weight = topk_weights[pair_idx].to(cutlass.Float32)
                            row = atomic_add_global_i32(get_ptr_as_int64(expert_write_rows, expert_id), Int32(1))
                            phys_row = expert_tile_base[expert_id] * Int32(self.tile_shape_mnk[0]) + row
                            st_global_i32(get_ptr_as_int64(token_map, phys_row), token_idx)
                            st_global_f32(get_ptr_as_int64(token_weights, phys_row), weight)
                            route_slot = route_slot_base + local_topk
                            _st_shared_i32(route_phys_rows_addr + route_slot * Int32(4), phys_row)
                            route_scale_row_base = Int32((Uint32(phys_row) >> Uint32(7)) * Uint32(num_k_tiles * Int32(512)) + (Uint32(phys_row) & Uint32(31)) * Uint32(16) + (Uint32(phys_row) >> Uint32(5) & Uint32(3)) * Uint32(4))
                            _st_shared_i32(route_phys_rows_addr + (route_slot + Int32(8)) * Int32(4), route_scale_row_base)
                            selected_scale_bits = _ld_shared_i32(expert_scales_addr + expert_id * Int32(4))
                            _st_shared_i32(route_scales_addr + route_slot * Int32(4), selected_scale_bits)
                            selected_gs = Uint32(selected_scale_bits).bitcast(cutlass.Float32)
                            if local_topk == Int32(0):
                                producer_first_gs = selected_gs
                            elif selected_gs != producer_first_gs:
                                producer_scales_equal = Int32(0)
                            local_topk += Int32(1)
                            topk_slot += Int32(1)
                        _st_shared_i32(route_expert_ids_addr + (route_slot_base + Int32(31)) * Int32(4), local_topk | producer_scales_equal << Int32(4))
                    cute.arch.sync_warp()
                    q0_ready = q0_bulk_try_wait(q0_bulk_barrier_addr, q0_bulk_phase)
                    while q0_ready == Int32(0):
                        q0_ready = q0_bulk_try_wait(q0_bulk_barrier_addr, q0_bulk_phase)
                    route_state = _ld_shared_i32(route_expert_ids_addr + (route_slot_base + Int32(31)) * Int32(4))
                    local_topk = route_state & Int32(15)
                    if local_topk > Int32(0):
                        first_gs = Uint32(_ld_shared_i32(route_scales_addr + route_slot_base * Int32(4))).bitcast(cutlass.Float32)
                        route_scales_equal = route_state >> Int32(4)
                        sf_idx = lane_id
                        while sf_idx < sf_blocks_per_row:
                            block_start = sf_idx * Int32(16)
                            loaded_values = load_shared_bf16x16_to_f32x16(q0_input_stage_base_addr + warp_idx * cols * Int32(2) + block_start * Int32(2))
                            values = cute.make_rmem_tensor((16,), cutlass.Float32)
                            block_max = cutlass.Float32(0.0)
                            for elem_idx in cutlass.range_constexpr(16):
                                value = loaded_values[elem_idx]
                                values[elem_idx] = value
                                block_max = fmax_f32(block_max, fabs_f32(value))
                            if route_scales_equal > Int32(0):
                                gs_value = first_gs
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(values, block_max, gs_value)
                                else:
                                    packed64, scale_byte = quantize_block_fp4(values, block_max, gs_value)
                                cache_slot = Int32(0)
                                while cache_slot < local_topk:
                                    route_slot = route_slot_base + cache_slot
                                    phys_row = _ld_shared_i32(route_phys_rows_addr + route_slot * Int32(4))
                                    output_offset = phys_row * output_bytes_per_row + sf_idx * Int32(8)
                                    st_global_u64(get_ptr_as_int64(packed_a_storage, output_offset), packed64)
                                    scale_row_base = _ld_shared_i32(route_phys_rows_addr + (route_slot + Int32(8)) * Int32(4))
                                    scale_offset = Int32(Uint32(scale_row_base) + (Uint32(sf_idx) >> Uint32(2)) * Uint32(512) + (Uint32(sf_idx) & Uint32(3)))
                                    scale_storage[scale_offset] = scale_byte
                                    cache_slot += Int32(1)
                            else:
                                cache_slot = Int32(0)
                                while cache_slot < local_topk:
                                    route_slot = route_slot_base + cache_slot
                                    phys_row = _ld_shared_i32(route_phys_rows_addr + route_slot * Int32(4))
                                    gs_value = Uint32(_ld_shared_i32(route_scales_addr + route_slot * Int32(4))).bitcast(cutlass.Float32)
                                    packed64 = Uint64(0)
                                    scale_byte = Uint8(0)
                                    if self.fast_math:
                                        packed64, scale_byte = quantize_block_fp4_fast(values, block_max, gs_value)
                                    else:
                                        packed64, scale_byte = quantize_block_fp4(values, block_max, gs_value)
                                    output_offset = phys_row * output_bytes_per_row + sf_idx * Int32(8)
                                    st_global_u64(get_ptr_as_int64(packed_a_storage, output_offset), packed64)
                                    scale_row_base = _ld_shared_i32(route_phys_rows_addr + (route_slot + Int32(8)) * Int32(4))
                                    scale_offset = Int32(Uint32(scale_row_base) + (Uint32(sf_idx) >> Uint32(2)) * Uint32(512) + (Uint32(sf_idx) & Uint32(3)))
                                    scale_storage[scale_offset] = scale_byte
                                    cache_slot += Int32(1)
                            sf_idx += Int32(32)
                q0_bulk_phase = Int32(1) - q0_bulk_phase
        cute.arch.sync_threads()
        _threadfence()
        cute.arch.sync_threads()
        self.resident_grid_barrier(barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader)
        total_m_tiles = expert_tile_base[num_experts]
        split_groups = (route_gate_tile_cnt + Int32(1)) // Int32(2)
        extra_per_split = split_groups - Int32(1)
        split_tile_count = Int32(0)
        if extra_per_split > Int32(0):
            if num_tokens > Int32(256):
                if num_tokens <= Int32(4096):
                    target_task_count = Int32(4) * Int32(gdim_z)
                    if num_tokens > Int32(2048):
                        target_task_count = (Int32(125) * Int32(gdim_z) + Int32(31)) // Int32(32)
                    missing_tasks = target_task_count - total_m_tiles
                    if missing_tasks > Int32(0):
                        split_tile_count = (missing_tasks + extra_per_split - Int32(1)) // extra_per_split
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
                        self.publish_uniform_deferred_tasks(task_expert, task_valid_rows, route_gate_tile_cnt, task_slice_chunk, expert_flush, expert_tile_base[expert_flush] + m_tile_offset, valid_rows)
                    elif num_tokens <= Int32(4096):
                        self.publish_variable_deferred_tasks(task_expert, task_valid_rows, route_gate_tile_cnt, split_tile_count, expert_flush, expert_tile_base[expert_flush] + m_tile_offset, valid_rows)
                    else:
                        self.publish_uniform_deferred_tasks(task_expert, task_valid_rows, route_gate_tile_cnt, task_slice_chunk, expert_flush, expert_tile_base[expert_flush] + m_tile_offset, valid_rows)
                    rows_remaining -= Int32(self.tile_shape_mnk[0])
                    m_tile_offset += Int32(1)
                expert_flush += Int32(gdim_z)
        if flat_tid == Int32(0):
            uniform_groups = (route_gate_tile_cnt + task_slice_chunk - Int32(1)) // task_slice_chunk
            published_task_count = expert_tile_base[num_experts] * uniform_groups
            if num_tokens > Int32(256):
                if num_tokens <= Int32(4096):
                    published_task_count = expert_tile_base[num_experts] + split_tile_count * extra_per_split
            st_global_i32(get_ptr_as_int64(task_tail, Int32(0)), published_task_count)
        self.resident_grid_barrier(barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader)
