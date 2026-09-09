"""Full-token E72 expert-local prefill: skip remote routes before row allocation.

Pinned stock M128 FC1/Q1/FC2 math and task descriptors are inherited.
Weighted BF16 contributions accumulate in FP32 before one BF16 output cast.
The entry point admits I2048 with uniformly bounded four-slice tasks.
Row-major and in-place tile-major weights share the same compute body;
tile-major changes only the weight TMA views, not raw MMA scale storage.
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
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32, Uint64, dsl_user_op
from cutlass._mlir.dialects import llvm
from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32, fabs_f32, fmax_f32, rcp_approx_ftz, get_smem_ptr_as_int32,
    quantize_block_fp4, quantize_block_fp4_fast, get_ptr_as_int64,
    ld_shared_i32_relaxed, st_global_f32, st_global_i32,
    st_shared_i32, st_shared_f32, st_global_v4_u32, st_global_u64,
)
from ._moe_dynamic import gated as _stock
from ._moe_dynamic.gated import (
    DynamicLaunchParams, MoEGatedDynamicKernel, _TASK_SLICE_CHUNK,
    blockscaled_utils, utils, cuda,
    _ld_shared_i32, _st_shared_i32, _threadfence, atomic_add_shared_i32,
    q0_bulk_barrier_init, q0_cp_async_bulk, q0_bulk_arrive_expect_tx, q0_bulk_try_wait,
    load_shared_bf16x16_to_f32x16, load_shared_i32_f32_pair,
)

STOCK_GATED_SHA256 = "993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445"


@lru_cache(maxsize=1)
def stock_contract_matches():
    return hashlib.sha256(Path(_stock.__file__).read_bytes()).hexdigest() == STOCK_GATED_SHA256


@dsl_user_op
def scatter_add_weighted_bf16x8_to_f32(
    addr, smem_addr, route_weight, down_alpha, *, loc=None, ip=None,
):
    """Preserve the stock BF16 contribution, widen only its atomic sum.

    Global FP32 RED rounds to nearest even and flushes FP32 subnormals,
    unlike stock BF16 RED's noftz. This does not promise deterministic sums.
    """
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip),
         Int32(smem_addr).ir_value(loc=loc, ip=ip),
         route_weight.ir_value(loc=loc, ip=ip),
         down_alpha.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 p0,p1,p2,p3,w2; .reg .b16 w,h0,h1,h2,h3,h4,h5,h6,h7;"
        " .reg .f32 combined_scale,f0,f1,f2,f3,f4,f5,f6,f7; .reg .b64 cp,next;"
        " ld.shared.v4.u32 {p0,p1,p2,p3}, [$1];"
        " mul.rn.f32 combined_scale, $2, $3;"
        " cvt.rn.bf16.f32 w, combined_scale;"
        " mov.b32 w2, {w,w};"
        " mul.rn.bf16x2 p0, p0, w2;"
        " mul.rn.bf16x2 p1, p1, w2;"
        " mul.rn.bf16x2 p2, p2, w2;"
        " mul.rn.bf16x2 p3, p3, w2;"
        " mov.b32 {h0,h1}, p0; mov.b32 {h2,h3}, p1;"
        " mov.b32 {h4,h5}, p2; mov.b32 {h6,h7}, p3;"
        " cvt.f32.bf16 f0, h0; cvt.f32.bf16 f1, h1;"
        " cvt.f32.bf16 f2, h2; cvt.f32.bf16 f3, h3;"
        " cvt.f32.bf16 f4, h4; cvt.f32.bf16 f5, h5;"
        " cvt.f32.bf16 f6, h6; cvt.f32.bf16 f7, h7;"
        " createpolicy.fractional.L2::evict_last.b64 cp, 1.0;"
        " red.global.add.L2::cache_hint.v4.f32 [$0], {f0,f1,f2,f3}, cp;"
        " add.u64 next, $0, 16;"
        " red.global.add.L2::cache_hint.v4.f32 [next], {f4,f5,f6,f7}, cp; }",
        "l,r,f,f", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


class MoEGatedEPLocalKernel(MoEGatedDynamicKernel):
    def _setup_attributes(self, hidden_size):
        if hidden_size != 4096 or self.tile_shape_mnk != (128, 128, 128):
            raise ValueError("expert-local prefill requires H4096 and M128/N128/K128")
        if self.share_input_across_experts or not stock_contract_matches():
            raise ValueError("expert-local prefill inherited source/scale contract differs")
        super()._setup_attributes(hidden_size)

    @cute.jit
    def scatter_sC_to_gmem(
        self,
        tidx,
        output_tile_idx,
        valid_rows: Int32,
        sC: cute.Tensor,
        tRS_sD: cute.Tensor,
        scatter_output: cute.Tensor,
        scatter_tok_base_addr: Int32,
        scatter_weight_base_addr: Int32,
        down_alpha_value,
    ):
        epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
        scatter_N = Int32(scatter_output.shape[1])
        lane_id = Int32(tidx) & Int32(31)
        warp_in_tile = Int32(tidx) >> Int32(5)
        warp_m_base = (warp_in_tile >> Int32(1)) * Int32(32)
        warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)

        # Scatter using precomputed metadata (no redundant gmem loads)
        tile_n_base_cur = output_tile_idx * Int32(self.tile_shape_mnk[1])
        for epi_m in cutlass.range_constexpr(epi_rest_m):
            epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])

            # Per-warp scatter: all eight math warps cover one disjoint
            # sC strip (32 M-rows x 64 N-cols).
            warp_epi_rows = valid_rows - rows_offset - warp_m_base
            if warp_epi_rows > Int32(32):
                warp_epi_rows = Int32(32)
            if warp_epi_rows < Int32(0):
                warp_epi_rows = Int32(0)

            if scatter_output.shape[0] <= Int32(2048):
                # One work item owns two adjacent N8 vectors from the same
                # M row.  Relative to the original 256-vector round-robin loop,
                # this keeps all 32 lanes active while sharing one token/weight
                # metadata load across two reductions.  Unlike row ownership, a
                # lane never serializes all eight reductions of one row.
                tile_pair_cols = Int32(64) // Int32(16)
                pair_idx = lane_id
                while pair_idx < warp_epi_rows * tile_pair_cols:
                    local_row = pair_idx // tile_pair_cols
                    local_pair_col = pair_idx - local_row * tile_pair_cols
                    local_col_base = warp_n_base + local_pair_col * Int32(16)
                    cached_row = rows_offset + warp_m_base + local_row
                    tok, wv = load_shared_i32_f32_pair(
                        scatter_tok_base_addr + cached_row * Int32(8)
                    )
                    for pair_half in cutlass.range_constexpr(2):
                        local_col = local_col_base + Int32(pair_half) * Int32(8)
                        global_col = tile_n_base_cur + local_col
                        # Preserve the K_SW128 address transform independently
                        # for both N8 reductions in the pair.
                        sc_element_offset = Int32(
                            sC.layout(
                                (
                                    warp_m_base + local_row,
                                    local_col,
                                    epi_buffer,
                                )
                            )
                        )
                        sc_element_offset = sc_element_offset ^ (
                            (sc_element_offset & Int32(0x1C0)) >> Int32(3)
                        )
                        sc_smem_addr = get_smem_ptr_as_int32(
                            sC,
                            sc_element_offset,
                        )
                        scatter_add_weighted_bf16x8_to_f32(
                            get_ptr_as_int64(
                                scatter_output, tok * scatter_N + global_col
                            ),
                            sc_smem_addr,
                            wv,
                            down_alpha_value,
                        )
                    pair_idx += Int32(self.num_threads_per_warp)
            else:
                tile_vec_cols = Int32(64) // Int32(8)
                vec_idx = lane_id
                while vec_idx < warp_epi_rows * tile_vec_cols:
                    local_row = vec_idx // tile_vec_cols
                    local_vec_col = vec_idx - local_row * tile_vec_cols
                    local_col = warp_n_base + local_vec_col * Int32(8)
                    global_col = tile_n_base_cur + local_col
                    cached_row = rows_offset + warp_m_base + local_row
                    tok, wv = load_shared_i32_f32_pair(
                        scatter_tok_base_addr + cached_row * Int32(8)
                    )
                    # Preserve the K_SW128 address transform: compute the
                    # unswizzled outer offset through sC.layout, then explicitly
                    # apply S<3,4,3> in BF16 element units before stripping the
                    # SMEM pointer metadata.  A raw pointer does not retain CuTe's
                    # swizzle transform.
                    sc_element_offset = Int32(
                        sC.layout(
                            (
                                warp_m_base + local_row,
                                local_col,
                                epi_buffer,
                            )
                        )
                    )
                    sc_element_offset = sc_element_offset ^ (
                        (sc_element_offset & Int32(0x1C0)) >> Int32(3)
                    )
                    sc_smem_addr = get_smem_ptr_as_int32(
                        sC,
                        sc_element_offset,
                    )
                    scatter_add_weighted_bf16x8_to_f32(
                        get_ptr_as_int64(scatter_output, tok * scatter_N + global_col),
                        sc_smem_addr,
                        wv,
                        down_alpha_value,
                    )
                    vec_idx += Int32(self.num_threads_per_warp)


    @cute.jit
    def publish_ep_local_uniform_tasks(
        self,
        task_expert,
        task_valid_rows,
        gate_tile_cnt: Int32,
        slice_chunk: Int32,
        expert_idx: Int32,
        m_tile_idx: Int32,
        valid_rows: Int32,
    ):
        # E72/I2048 publishes exactly four four-slice tasks per M128 tile.
        # Their expert words are identical and their valid-row words differ
        # only in slice_begin.  Keep the same single publisher and slot order,
        # but write each four-word descriptor array with one vector store.
        # The normal workspace allocations are aligned; retain scalar stores
        # for a direct caller that only satisfies the pointer ABI's 4B align.
        first_slot = m_tile_idx * Int32(4)
        expert_addr = get_ptr_as_int64(task_expert, first_slot)
        rows_addr = get_ptr_as_int64(task_valid_rows, first_slot)
        if (
            gate_tile_cnt == Int32(16)
            and slice_chunk == Int32(4)
            and ((expert_addr | rows_addr) & Int64(15)) == Int64(0)
        ):
            expert_word = Uint32(expert_idx | (m_tile_idx << Int32(16)))
            rows_word = Uint32(valid_rows | (Int32(4) << Int32(20)))
            st_global_v4_u32(
                expert_addr, expert_word, expert_word, expert_word, expert_word
            )
            st_global_v4_u32(
                rows_addr,
                rows_word,
                rows_word | Uint32(4 << 8),
                rows_word | Uint32(8 << 8),
                rows_word | Uint32(12 << 8),
            )
        else:
            self.publish_uniform_deferred_tasks(
                task_expert, task_valid_rows, gate_tile_cnt, slice_chunk,
                expert_idx, m_tile_idx, valid_rows,
            )

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
        # The EP-only scatter pointer is FP32; packed Q0 rows remain FP4.
        cols_u32 = cols
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
        # local expert subtotals globally.  The first 2304 bytes of sA
        # hold route caches; the next 72 * 4 bytes hold this histogram.
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
            # Remote/sentinel routes have no histogram contribution, even
            # when their unused weight storage contains NaNs or other bits.
            if expert_id >= Int32(0) and expert_id < num_experts:
                weight = topk_weights[hist_idx].to(cutlass.Float32)
                if weight != cutlass.Float32(0.0):
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
                # E72/top8/T<=16384 gives 0<=rows<=131072. M128 is
                # admitted above, so unsigned ceil has no signed correction.
                tile_acc += Int32((Uint32(rows) + Uint32(127)) >> Uint32(7))
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
        # The histogram is dead after the preceding resident-grid barriers.
        # Reuse its 72 * 4 bytes in sA for per-expert transformed input scales;
        # sA is otherwise idle until all Q0 producers finish.  Each expert is
        # prepared once per CTA instead of once per token/route in every lane.
        # The existing Q0-init CTA barrier publishes these shared stores.
        expert_scales_addr = route_hist_addr
        # Row allocation is the last use of each selected expert ID.  Reuse
        # its existing route slot for the transformed scale's raw bits, so
        # all lanes can broadcast it without a per-thread local-memory array.
        route_scales_addr = route_expert_ids_addr
        if tidx < num_experts:
            gs_value = input_global_scale[tidx].to(cutlass.Float32)
            if (
                self.input_scales_are_reciprocal
                and gs_value != cutlass.Float32(0.0)
            ):
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
                        producer_first_gs = cutlass.Float32(0.0)
                        producer_scales_equal = Int32(1)
                        while topk_slot < num_topk:
                            pair_idx = token_idx * num_topk + topk_slot
                            expert_id = topk_ids[pair_idx].to(Int32)
                            if expert_id >= Int32(0) and expert_id < num_experts:
                                weight = topk_weights[pair_idx].to(cutlass.Float32)
                                if weight != cutlass.Float32(0.0):
                                    row = atomic_add_global_i32(
                                        get_ptr_as_int64(expert_write_rows, expert_id),
                                        Int32(1),
                                    )
                                    # The tile quotient and remainder recombine
                                    # into row; avoid signed division in the route
                                    # allocator without changing its physical row.
                                    phys_row = expert_tile_base[expert_id] * Int32(
                                        self.tile_shape_mnk[0]
                                    ) + row
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
                                    # Slots 0..7 retain physical rows. The
                                    # unused 8..15 slots cache their row-only
                                    # M128 scale offsets once per route, before
                                    # every lane visits its eight SF blocks.
                                    # H4096/top8/T<=16384 keeps these exact
                                    # nonnegative byte offsets below 2**31.
                                    route_scale_row_base = Int32(
                                        (Uint32(phys_row) >> Uint32(7))
                                        * Uint32(num_k_tiles * Int32(512))
                                        + (Uint32(phys_row) & Uint32(31)) * Uint32(16)
                                        + ((Uint32(phys_row) >> Uint32(5)) & Uint32(3))
                                        * Uint32(4)
                                    )
                                    _st_shared_i32(
                                        route_phys_rows_addr
                                        + (route_slot + Int32(8)) * Int32(4),
                                        route_scale_row_base,
                                    )
                                    selected_scale_bits = _ld_shared_i32(
                                        expert_scales_addr + expert_id * Int32(4)
                                    )
                                    _st_shared_i32(
                                        route_scales_addr + route_slot * Int32(4),
                                        selected_scale_bits,
                                    )
                                    # Reuse this already-loaded word while the
                                    # input copy is in flight. Compare Float32,
                                    # preserving signed-zero equality and the
                                    # single-NaN case (never compare the first
                                    # selected scale with itself).
                                    selected_gs = Uint32(selected_scale_bits).bitcast(
                                        cutlass.Float32
                                    )
                                    if local_topk == Int32(0):
                                        producer_first_gs = selected_gs
                                    elif selected_gs != producer_first_gs:
                                        producer_scales_equal = Int32(0)

                                    local_topk += Int32(1)
                            topk_slot += Int32(1)
                        # Only this Q0 producer/consumer uses slot 31. Top8
                        # leaves bit 4 free above the four-bit route count;
                        # publish equality in the existing count word, with
                        # no extra shared store or slot. The same warp barrier
                        # publishes both this state and the raw scale slots.
                        _st_shared_i32(
                            route_expert_ids_addr + (route_slot_base + Int32(31)) * Int32(4),
                            local_topk | (producer_scales_equal << Int32(4)),
                        )
                    cute.arch.sync_warp()
                    q0_ready = q0_bulk_try_wait(q0_bulk_barrier_addr, q0_bulk_phase)
                    while q0_ready == Int32(0):
                        q0_ready = q0_bulk_try_wait(
                            q0_bulk_barrier_addr, q0_bulk_phase
                        )

                    route_state = _ld_shared_i32(
                        route_expert_ids_addr + (route_slot_base + Int32(31)) * Int32(4)
                    )
                    local_topk = route_state & Int32(15)
                    if local_topk > Int32(0):
                        # Every lane reloads published state; lane 0's
                        # registers are not shared. The next batch's CTA
                        # barrier still protects the raw selected-scale slots.
                        first_gs = Uint32(_ld_shared_i32(
                            route_scales_addr + route_slot_base * Int32(4)
                        )).bitcast(cutlass.Float32)
                        route_scales_equal = route_state >> Int32(4)

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
                                gs_value = first_gs
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
                                    output_offset = (
                                        phys_row * output_bytes_per_row
                                        + sf_idx * Int32(8)
                                    )
                                    # This EP entry admits T>=4096, so the
                                    # stock T<=2048 L2-retention arm is dead.
                                    st_global_u64(
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    scale_row_base = _ld_shared_i32(
                                        route_phys_rows_addr
                                        + (route_slot + Int32(8)) * Int32(4)
                                    )
                                    # The existing warp barrier publishes the
                                    # cached row field. SF fields are disjoint;
                                    # retain the exact M128 byte permutation.
                                    scale_offset = Int32(
                                        Uint32(scale_row_base)
                                        + (Uint32(sf_idx) >> Uint32(2)) * Uint32(512)
                                        + (Uint32(sf_idx) & Uint32(3))
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
                                    gs_value = Uint32(_ld_shared_i32(
                                        route_scales_addr + route_slot * Int32(4)
                                    )).bitcast(cutlass.Float32)

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
                                    st_global_u64(
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    scale_row_base = _ld_shared_i32(
                                        route_phys_rows_addr
                                        + (route_slot + Int32(8)) * Int32(4)
                                    )
                                    scale_offset = Int32(
                                        Uint32(scale_row_base)
                                        + (Uint32(sf_idx) >> Uint32(2)) * Uint32(512)
                                        + (Uint32(sf_idx) & Uint32(3))
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
                    self.publish_ep_local_uniform_tasks(
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
        b_w13: cute.Tensor,  # [4096,4096,72] or tiled [4096,512,8,72]
        sfb_w13_ptr: cute.Pointer,  # scale factors for w13
        b_down: cute.Tensor,  # [4096,2048,72] or tiled [4096,128,16,72]
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
        if cutlass.const_expr(scatter_output.element_type != cutlass.Float32):
            raise ValueError("expert-local v2 scatter requires FP32 accumulation storage")
        # Match the stock tiled dynamic adapter: grouping is a view, retaining
        # each packed tensor's pointer and strides. Both native K128 TMA boxes
        # divide the inner K512/K128 modes, so no tile crosses a storage chunk.
        # Reject mixed or different 4-D layouts before any TMA descriptor exists.
        if cutlass.const_expr(len(b_w13.shape) == 4 or len(b_down.shape) == 4):
            if cutlass.const_expr(
                b_w13.shape != (4096, 512, 8, 72)
                or b_down.shape != (4096, 128, 16, 72)
            ):
                raise ValueError("expert-local tiled weights require E72/H4096/I2048 v5 views")
            b_w13 = cute.group_modes(b_w13, 1, 3)
            b_down = cute.group_modes(b_down, 1, 3)
        # Raw MMA scales keep their original logical shape, independent of the
        # hierarchical weight K. In particular, do not tile or repack SFB here.
        w13_logical_shape = (b_w13.shape[0], cute.size(b_w13.shape[1]), b_w13.shape[2])
        down_logical_shape = (b_down.shape[0], cute.size(b_down.shape[1]), b_down.shape[2])
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
            w13_logical_shape, self.sf_vec_size
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
            down_logical_shape, self.sf_vec_size
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
            or down_logical_shape[1] != 2048 or row_counts.shape[0] != 72
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
