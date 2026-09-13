"""Static, pinned Q0 fork: eight rows, nine-warp prefix, parallel top8 allocation."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'engine/kernels/b12x/moe_dynamic_gated_sf6_q0.py'
PIN = '759a519145c3edecac42ea917abbd8003b142f02ba4d41f86fc2d58e939a51ba'


def build():
    source = (ROOT/SOURCE).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != PIN:
        raise ValueError('Q0 parent changed; review arithmetic and publication before regenerating')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == 'initialize_route_q0_and_publish')
    imports = '\n'.join(ast.get_source_segment(source,n) for n in tree.body
                        if isinstance(n,(ast.Import,ast.ImportFrom)))
    method = '\n'.join(source.splitlines()[node.lineno-2:node.end_lineno])
    old = '        num_tokens = Int32(a_input.shape[0])'
    remap = '''        # Startup-idle sB precedes sC, followed by sA, in both pinned kernels.
        # Move all route caches out of sA before staging eight rows over sC+sA.
        route_phys_rows_addr = q0_input_stage_base_addr - Int32(self.q0_route_shift)
        route_expert_ids_addr = route_phys_rows_addr + Int32(9 * 32 * 4)
'''
    assert method.count(old)==1
    method = method.replace(old, remap+old)
    old = 'producer_batch_tokens = Int32(self.tile_shape_mnk[0] * self.tile_shape_mnk[1]) // cols'
    assert method.count(old)==1
    method = method.replace(old,'producer_batch_tokens = Int32(8)')
    old = '        if num_experts == Int32(256) and bidz == Int32(0) and (warp_idx < Int32(self.num_mma_warps)):'
    prefix = '''        if num_experts == Int32(288) and bidz == Int32(0):
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
                    _st_shared_i32(ctrl_base_addr, warp_prefix)
            cute.arch.sync_threads()
            base = ld_shared_i32_relaxed(route_hist_addr + warp_idx * Int32(4))
            expert_tile_base[tidx] = base + exclusive
            if tidx == Int32(0):
                expert_tile_base[num_experts] = _ld_shared_i32(ctrl_base_addr)
'''
    assert method.count(old)==1
    method = method.replace(old,prefix+old.replace('        if ', '        elif ',1))
    start = method.index('                    if lane_id == Int32(0):\n                        topk_slot = Int32(0)')
    end = method.index('                    cute.arch.sync_warp()',start)
    reserve = '''                    # All 32 lanes participate in the equality reduction. The first
                    # eight reserve independent routes, including duplicate/zero-weight routes.
                    selected_gs = cutlass.Float32(0.0)
                    topk_slot = lane_id
                    if topk_slot < num_topk:
                        pair_idx = token_idx * num_topk + topk_slot
                        expert_id = topk_ids[pair_idx].to(Int32)
                        weight = topk_weights[pair_idx].to(cutlass.Float32)
                        row = atomic_add_global_i32(get_ptr_as_int64(expert_write_rows, expert_id), Int32(1))
                        phys_row = expert_tile_base[expert_id] * Int32(128) + row
                        st_global_i32(get_ptr_as_int64(token_map, phys_row), token_idx)
                        st_global_f32(get_ptr_as_int64(token_weights, phys_row), weight)
                        route_slot = route_slot_base + topk_slot
                        _st_shared_i32(route_phys_rows_addr + route_slot * Int32(4), phys_row)
                        route_scale_row_base = Int32((Uint32(phys_row) >> Uint32(7)) * Uint32(num_k_tiles * Int32(512)) + (Uint32(phys_row) & Uint32(31)) * Uint32(16) + (Uint32(phys_row) >> Uint32(5) & Uint32(3)) * Uint32(4))
                        _st_shared_i32(route_phys_rows_addr + (route_slot + Int32(8)) * Int32(4), route_scale_row_base)
                        selected_scale_bits = _ld_shared_i32(expert_scales_addr + expert_id * Int32(4))
                        _st_shared_i32(route_scales_addr + route_slot * Int32(4), selected_scale_bits)
                        selected_gs = Uint32(selected_scale_bits).bitcast(cutlass.Float32)
                    producer_first_gs = cute.arch.shuffle_sync(selected_gs, Int32(0))
                    mismatch = Int32((lane_id < num_topk) & (selected_gs != producer_first_gs))
                    for stage in cutlass.range_constexpr(5):
                        mismatch = mismatch | cute.arch.shuffle_sync(mismatch, lane_id ^ Int32(1 << stage))
                    if lane_id == Int32(0):
                        _st_shared_i32(route_expert_ids_addr + (route_slot_base + Int32(31)) * Int32(4), num_topk | Int32(mismatch == Int32(0)) << Int32(4))
'''
    method = method[:start]+reserve+method[end:]
    result = ('# SPDX-License-Identifier: Apache-2.0\n'
              '# Generated by measurements/st_prefill_oracle_20260913/generate_q0_batch8.py.\n'
              '# Quantization, FP32 zeroing and final task publication retain the pinned Q0 body.\n'
              +imports+'\n\nclass Q0Batch8Body:\n'+method+'\n')
    ast.parse(result)
    return result


if __name__=='__main__':
    (ROOT/'engine/kernels/b12x/_prefill_q0_batch8.py').write_text(build())
