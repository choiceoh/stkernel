"""Full-token E72 expert-local prefill: skip remote routes before row allocation.

Pinned stock M128 FC1/Q1/FC2/scatter math and task descriptors are inherited.
The entry point admits I2048 with uniformly bounded four-slice tasks.
Route/Q0 preparation changes: histogram valid local pairs, compact a
warp's <=8 local routes into its existing shared cache, and quantize the
original token once per block/scale. No expanded pair_x/pair_out, nonzero,
CPU route-count synchronization or external index_add is needed.

The original [T,8] remap uses E72 as its out-of-range zero-weight sentinel.
Every sentinel is rejected before deriving a histogram/scale/row address.
No-local-route tokens keep their prezeroed output; all CTA barriers remain.
This is experimental; no GPU correctness or prefill speed claim is made.
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
    st_shared_i32, st_global_v4_u32,
)
from ._moe_dynamic import gated as _stock
from ._moe_dynamic.gated import (
    DynamicLaunchParams, MoEGatedDynamicKernel, _TASK_SLICE_CHUNK,
    blockscaled_utils, utils, cuda,
    _ld_shared_i32, _st_shared_i32, _threadfence, atomic_add_shared_i32,
    q0_bulk_barrier_init, q0_cp_async_bulk, q0_bulk_arrive_expect_tx, q0_bulk_try_wait,
    load_shared_bf16x16_to_f32x16, st_global_u64_adaptive_l2,
)

STOCK_GATED_SHA256 = "993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445"


@lru_cache(maxsize=1)
def stock_contract_matches():
    return hashlib.sha256(Path(_stock.__file__).read_bytes()).hexdigest() == STOCK_GATED_SHA256


class MoEGatedEPLocalKernel(MoEGatedDynamicKernel):
    def _setup_attributes(self, hidden_size):
        if hidden_size != 4096 or self.tile_shape_mnk != (128, 128, 128):
            raise ValueError("expert-local prefill requires H4096 and M128/N128/K128")
        if self.share_input_across_experts or not stock_contract_matches():
            raise ValueError("expert-local prefill inherited source/scale contract differs")
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
        # local expert subtotals globally.  The first 2304 bytes of sC
        # hold route caches; the following aligned 1 KiB is idle here.
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
            weight = topk_weights[hist_idx].to(cutlass.Float32)
            if expert_id >= Int32(0) and expert_id < num_experts and weight != cutlass.Float32(0.0):
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

        if flat_tid == Int32(0):
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

        # Phase 2: the TMA warp stages only as many contiguous BF16 rows as
        # fit in the aliased sC backing.  The remaining math warps stay idle
        # during Q0 but remain active in FC1/Q1/FC2/scatter.
        if tidx == Int32(0):
            q0_bulk_barrier_init(q0_bulk_barrier_addr)
        cute.arch.sync_threads()
        lane_id = Int32(tidx) & Int32(31)
        _num_cta_warps = Int32(self.num_mma_warps + 1)
        # pair_head is a token counter in the token-major overlay.  sC holds
        # tile_M * tile_N BF16 values, so tile-area/cols is its full-token
        # capacity.  Keep this expression region-local for CuTe isolation.
        producer_batch_tokens = (
            Int32(self.tile_shape_mnk[0] * self.tile_shape_mnk[1]) // cols
        )
        if producer_batch_tokens > Int32(self.num_mma_warps):
            producer_batch_tokens = Int32(self.num_mma_warps)
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

                # Each math warp owns one token with at most eight local
                # routes, as required by the exact top8 dispatcher gate.
                token_idx = batch_base + warp_idx
                if warp_idx < producer_batch_tokens and token_idx < num_tokens:
                    route_slot_base = warp_idx * Int32(32)
                    if lane_id == Int32(0):
                        topk_slot = Int32(0)
                        local_topk = Int32(0)
                        while topk_slot < num_topk:
                            pair_idx = token_idx * num_topk + topk_slot
                            expert_id = topk_ids[pair_idx].to(Int32)
                            weight = topk_weights[pair_idx].to(cutlass.Float32)
                            if expert_id >= Int32(0) and expert_id < num_experts and weight != cutlass.Float32(0.0):
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

                                route_slot = route_slot_base + local_topk
                                _st_shared_i32(
                                    route_phys_rows_addr + route_slot * Int32(4),
                                    phys_row,
                                )
                                _st_shared_i32(
                                    route_expert_ids_addr + route_slot * Int32(4),
                                    expert_id,
                                )

                                local_topk += Int32(1)
                            topk_slot += Int32(1)
                        _st_shared_i32(route_expert_ids_addr + (route_slot_base + Int32(31)) * Int32(4), local_topk)
                    cute.arch.sync_warp()
                    q0_ready = q0_bulk_try_wait(q0_bulk_barrier_addr, q0_bulk_phase)
                    while q0_ready == Int32(0):
                        q0_ready = q0_bulk_try_wait(
                            q0_bulk_barrier_addr, q0_bulk_phase
                        )

                    local_topk = _ld_shared_i32(route_expert_ids_addr + (route_slot_base + Int32(31)) * Int32(4))
                    if local_topk > Int32(0):
                        # Preserve the baseline's per-lane scale load and
                        # reciprocal work.  Hoist it out of the block loop,
                        # but do not introduce a 32x broadcast optimization.
                        route_gs = cute.make_rmem_tensor((8,), cutlass.Float32)
                        cache_slot = Int32(0)
                        while cache_slot < local_topk:
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

                        # Expert scales are constant across this token's SF
                        # blocks. Compare once, not once per 16-column block.
                        route_scales_equal = Int32(1)
                        scale_idx = Int32(1)
                        while scale_idx < local_topk:
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

                            # Quantized payload is identical only when all
                            # selected experts use the same input global scale.
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
                                while cache_slot < local_topk:
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
                                while cache_slot < local_topk:
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

        # Every I2048 task retains at most four N128 slices. The stock
        # variable-task policy can retain the whole intermediate at T4096,
        # so this wide producer always uses the uniform four-slice policy.
        if is_cta_leader > Int32(0):
            expert_flush = Int32(bidz)
            while expert_flush < num_experts:
                rows_remaining = row_counts[expert_flush]
                m_tile_offset = Int32(0)
                while rows_remaining > Int32(0):
                    valid_rows = rows_remaining
                    if valid_rows > Int32(self.tile_shape_mnk[0]):
                        valid_rows = Int32(self.tile_shape_mnk[0])
                    self.publish_uniform_deferred_tasks(
                        task_expert, task_valid_rows, route_gate_tile_cnt,
                        task_slice_chunk, expert_flush,
                        expert_tile_base[expert_flush] + m_tile_offset, valid_rows)
                    rows_remaining -= Int32(self.tile_shape_mnk[0])
                    m_tile_offset += Int32(1)
                expert_flush += Int32(gdim_z)

        if flat_tid == Int32(0):
            uniform_groups = (
                route_gate_tile_cnt + task_slice_chunk - Int32(1)
            ) // task_slice_chunk
            published_task_count = expert_tile_base[num_experts] * uniform_groups
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
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
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

        # SF tensor for w13 (gated: gate+up concatenated; relu2: single W1)
        sfb_w13_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_w13.shape, self.sf_vec_size
        )
        sfb_w13_tensor = cute.make_tensor(sfb_w13_ptr, sfb_w13_layout)

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
        # SFB is different from B: the SM120 helper physically packs scale
        # factors in N128 blocks.  Both N64 halves replay the same physical
        # block and select half 0/1 from its shared-memory view.
        tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_w13_tensor,
            self.fc1_sfb_smem_layout_staged,
            self.fc1_sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )
        # B_down TMA
        sfb_down_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_down.shape, self.sf_vec_size
        )
        sfb_down_tensor = cute.make_tensor(sfb_down_ptr, sfb_down_layout)
        tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
            b_down,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_sfb_down, gSFB_down = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_down_tensor,
            self.sfb_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
            internal_type=cutlass.Int16,
        )

        # W13 concatenates equally-sized Gate and Up branches along N.
        gate_tile_cnt_static = b_w13.shape[0] // self.tile_shape_mnk[1] // 2
        # I2048 is sixteen logical N128 slices, published as four tasks
        # retaining four slices each. Never expand the inherited Q1 storage.
        if cutlass.const_expr(
            gate_tile_cnt_static != 16 or b_w13.shape[2] != 72
            or b_down.shape[1] != 2048 or row_counts.shape[0] != 72
        ):
            raise ValueError("expert-local wide entry point requires E72/I2048")
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
            tma_sfb_w13,
            gSFB_w13,
            tma_b_down,
            gB_down,
            tma_sfb_down,
            gSFB_down,
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
