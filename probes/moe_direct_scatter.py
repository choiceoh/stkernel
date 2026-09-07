#!/usr/bin/env python3
"""Private MoE FC2 epilogues: preserve BF16 rounding without shared staging.

The pair variant scatters each accumulator pair directly. The vector variant
gathers four adjacent lane pairs and retains the stock 16-byte reduction.
Neither changes production dispatch, packed weights, or the FC1/FC2 MMA order.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile

VARIANTS = ('pair', 'vector', 'warp')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'expected one source anchor: {old[:90]!r}')
    return source.replace(old, new)


PAIR_HELPER = '''
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def _scatter_pair(addr, v0, v1, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip),
         v0.ir_value(loc=loc, ip=ip), v1.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 p; cvt.rn.satfinite.bf16x2.f32 p, $2, $1; "
        "red.global.add.noftz.bf16x2 [$0], p; }",
        "l,f,f", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


'''

LAYOUT_CHECK = '''    def _check_direct_scatter_layout(self):
        # Compile-time exhaustive proof, independent of any runtime data.
        # A pair must have the same row and adjacent columns; adjacent lanes
        # in a quad must cover one aligned 8-column vector. Retiling or MMA
        # geometry drift therefore fails before this kernel can launch.
        ident = cute.make_identity_tensor((_TILE_M, _FC2_TILE_N))
        seen = set()
        for tid in range(128):
            coords = self.tiled_mma.get_slice(tid).partition_C(ident)
            leader = self.tiled_mma.get_slice(tid - tid % 4).partition_C(ident)
            for i in range(0, cute.size(coords), 2):
                r, c = coords[i]
                r1, c1 = coords[i + 1]
                lr, lc = leader[i]
                assert r == r1 and c1 == c + 1 and c % 2 == 0
                assert r == lr and c == lc + 2 * (tid % 4) and lc % 8 == 0
                for rr, cc in ((r, c), (r1, c1)):
                    assert (rr, cc) not in seen
                    seen.add((rr, cc))
        assert seen == {(r, c) for r in range(_TILE_M) for c in range(_FC2_TILE_N)}
        print('DIRECT_SCATTER_LAYOUT_PASS', len(seen), flush=True)

'''


def render(source, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant == 'warp':
        # Only M<=8: each valid output row is written and scattered by the
        # same warp. Larger batches retain the original CTA-wide barriers.
        check = LAYOUT_CHECK.replace(
            '                r1, c1 = coords[i + 1]',
            '                if r < 8:\n'
            '                    assert tid // 32 == 2 * ((c // 16) % 2), (tid, r, c)\n'
            '                r1, c1 = coords[i + 1]')
        source = replace_once(source, '    def _make_tiled_mma(self, tile_shape_mnk):',
                              check + '    def _make_tiled_mma(self, tile_shape_mnk):')
        source = replace_once(source, '        self.mma_atom = cute.make_mma_atom(mma_op)',
                              '        self._check_direct_scatter_layout()\n'
                              '        self.mma_atom = cute.make_mma_atom(mma_op)')
        source = replace_once(source,
            '            warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)\n'
            '            warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)',
            '            if cutlass.const_expr(a_input.shape[0] <= 8):\n'
            '                warp_m_base = (warp_in_tile & Int32(1)) * Int32(64)\n'
            '                warp_n_base = (warp_in_tile >> Int32(1)) * Int32(16)\n'
            '            else:\n'
            '                warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)\n'
            '                warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)')
        begin = source.index('                    tile_n_base_cur = output_tile_idx * Int32(_FC2_TILE_N)')
        end = source.index('\n                if cutlass.const_expr(self.stamps):', begin)
        body = source[begin:end]
        body = replace_once(body,
            '                        local_col = warp_n_base + local_vec_col * Int32(8)',
            '                        if cutlass.const_expr(a_input.shape[0] <= 8):\n'
            '                            local_col = (warp_n_base + (local_vec_col % Int32(2)) * Int32(8)\n'
            '                                         + (local_vec_col // Int32(2)) * Int32(32))\n'
            '                        else:\n'
            '                            local_col = warp_n_base + local_vec_col * Int32(8)')
        barrier = '                    self.epilog_sync_barrier.arrive_and_wait()'
        assert body.count(barrier) == 2
        body = body.replace(barrier,
            '                    if cutlass.const_expr(a_input.shape[0] <= 8):\n'
            '                        cute.arch.sync_warp()\n'
            '                    else:\n'
            '                        self.epilog_sync_barrier.arrive_and_wait()')
        body += ('\n                if cutlass.const_expr(a_input.shape[0] <= 8):\n'
                 '                    self.epilog_sync_barrier.arrive_and_wait()\n')
        return source[:begin] + body + source[end:]
    source = replace_once(source, 'class MoEStaticKernelV4:', PAIR_HELPER + 'class MoEStaticKernelV4:')
    source = replace_once(source, '    def _make_tiled_mma(self, tile_shape_mnk):',
                          LAYOUT_CHECK + '    def _make_tiled_mma(self, tile_shape_mnk):')
    source = replace_once(source, '        self.mma_atom = cute.make_mma_atom(mma_op)',
                          '        self._check_direct_scatter_layout()\n'
                          '        self.mma_atom = cute.make_mma_atom(mma_op)')
    source = replace_once(source, '            down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)',
                          '            down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)\n'
                          '            down_coords = thr_mma.partition_C(\n'
                          '                cute.make_identity_tensor((_TILE_M, _FC2_TILE_N)))')
    begin = source.index('                    tile_n_base_cur = output_tile_idx * Int32(_FC2_TILE_N)')
    end = source.index('\n                if cutlass.const_expr(self.stamps):', begin)
    body = '''                    tile_n_base_cur = output_tile_idx * Int32(_FC2_TILE_N)
                    for pair in cutlass.range_constexpr(cute.size(down_acc) // 2):
                        row, col = down_coords[2 * pair]
                        # Preserve the stock intermediate BF16 conversion,
                        # then FP32 route multiplication and satfinite BF16.
                        v0 = cutlass.Float32(cutlass.BFloat16(down_alpha_value * down_acc[2 * pair]))
                        v1 = cutlass.Float32(cutlass.BFloat16(down_alpha_value * down_acc[2 * pair + 1]))
'''
    if variant == 'vector':
        body += '''                        quad = lane_id & Int32(-4)
                        v2 = cute.arch.shuffle_sync(v0, quad + Int32(1))
                        v3 = cute.arch.shuffle_sync(v1, quad + Int32(1))
                        v4 = cute.arch.shuffle_sync(v0, quad + Int32(2))
                        v5 = cute.arch.shuffle_sync(v1, quad + Int32(2))
                        v6 = cute.arch.shuffle_sync(v0, quad + Int32(3))
                        v7 = cute.arch.shuffle_sync(v1, quad + Int32(3))
                        if row < valid_tile_rows and (lane_id & Int32(3)) == Int32(0):
                            tok = ld_shared_i32_relaxed(scatter_tok_base_addr + row * Int32(4))
                            wv = _ld_shared_f32(scatter_weight_base_addr + row * Int32(4))
                            scatter_add_v4_bf16x2(
                                get_ptr_as_int64(scatter_output, tok * scatter_N + tile_n_base_cur + col),
                                wv*v0, wv*v1, wv*v2, wv*v3, wv*v4, wv*v5, wv*v6, wv*v7)
'''
    else:
        body += '''                        if row < valid_tile_rows:
                            tok = ld_shared_i32_relaxed(scatter_tok_base_addr + row * Int32(4))
                            wv = _ld_shared_f32(scatter_weight_base_addr + row * Int32(4))
                            _scatter_pair(
                                get_ptr_as_int64(scatter_output, tok * scatter_N + tile_n_base_cur + col),
                                wv*v0, wv*v1)
'''
    # One barrier per item retains the cache ownership handoff: a fast warp
    # must not overwrite scatter token/weight metadata while peers use it.
    body += '                self.epilog_sync_barrier.arrive_and_wait()\n'
    return source[:begin] + body + source[end:]


def install(md, variant, directory=None):
    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x as package
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_static_kernel_v4 as original
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_static_kernel_v5 as tiled
    dst = Path(directory or tempfile.mkdtemp(prefix='moe-direct-'))
    dst.mkdir(parents=True, exist_ok=True)
    source = render(Path(original.__file__).read_text(), variant)
    digest = hashlib.sha256(source.encode()).hexdigest()
    leaf = f'moe_direct_{variant}_{digest[:12]}'
    path = dst / (leaf + '.py')
    path.write_text(source)
    package.__path__.append(str(dst))

    def module(name, path):
        spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    mod = module(leaf, path)
    tiled_source = replace_once(Path(tiled.__file__).read_text(),
                               'from .moe_static_kernel_v4 import (', f'from .{leaf} import (')
    tiled_path = dst / (leaf + '_tiled.py')
    tiled_path.write_text(tiled_source)
    mod_t = module(leaf + '_tiled', tiled_path)
    md.MoEStaticKernelV4, md.MoEStaticKernelV5 = mod.MoEStaticKernelV4, mod_t.MoEStaticKernelV5
    old_sources = md._kernel_source_files
    md._kernel_source_files = lambda: (*old_sources(), str(path), str(tiled_path))
    md._STATIC_V2_KERNEL_CACHE.clear()
    print(f'candidate={variant} source_sha256={digest}', flush=True)
    return dict(variant=variant, source_sha256=digest, files=[str(path), str(tiled_path)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=('baseline', *VARIANTS), required=True)
    args = ap.parse_args()
    import b12x_static_compile_check  # device-free target queries
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    if args.variant != 'baseline':
        install(md, args.variant)
    md.get_num_sm = lambda dev=None: 48
    md.get_max_active_clusters = lambda n=1: 48
    cfg = md._parse_glm53_static_v2('t', probe=True)
    md._get_static_kernel_v2(288, 288, 6, 4096, 512, 8, 512,
        config=cfg, mac_override=48, activation='swigluoai_uninterleave',
        swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.0)
    print(f'VERDICT: PASS ({args.variant} CPU compile only)', flush=True)


if __name__ == '__main__':
    main()
