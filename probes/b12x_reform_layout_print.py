#!/usr/bin/env python3
"""Print the served M16 reform tile's B stage smem layouts (swizzle included) on the CPU, and check whether a
host pre-swizzle of the tile-major storage would be the canonical 128 B / 64 B row swizzle -- the permutation a
one-request `cp.async.bulk` per B stage needs (the z cell redone for the reform tile, 2026-09-17).

    CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a PYTHONPATH=/repo python3 probes/b12x_reform_layout_print.py

The trace runs on the CPU; nothing is launched. For every (row, 16-byte chunk) of stage 0 the byte offset the
consumer reads is enumerated through the composed layout, and compared with
    byte(row, chunk) = row * pitch + ((chunk ^ (row % 8)) * 16)        # Swizzle<3,4,3> over 128 B rows (B1)
    byte(row, chunk) = row * pitch + ((chunk ^ ((row // 2) % 4)) * 16)  # Swizzle<2,4,3> over 64 B rows (B2)
"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import patch

import torch

with patch.object(torch.cuda, "is_available", return_value=True), \
        patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)):
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels.b12x.moe_static_kernel_v4 import MoEStaticKernelV4


def kernel_for_served_m16():
    config = md._static_v2_decode_config(md._parse_glm53_static_v2("t,r,sf6,batch"), 16)
    reform = config["decode_reform"]
    k = MoEStaticKernelV4(
        scatter_fp32=True, route_scatter=False,
        direct_scatter=bool(config.get("c2_direct_scatter")), scatter_reuse=bool(config["c2_scatter_reuse"]),
        fc2_prefetch=bool(config["c2_fc2_prefetch"]), a_ring=False, sf_pack=False, decode_reform=reform,
        reform_sf_pack=bool(config.get("reform_sf_pack", False)), sf6_separate=bool(config["sf6_separate"]),
        sf6_word_expand=bool(config["sf6_word_expand"]), sf6_fc2_word_expand=bool(config["sf6_fc2_word_expand"]),
        packed_activation_store=bool(config["packed_activation_store"]), fc1_reuse_a=bool(config["fc1_reuse_a"]),
        compact_staging=bool(config["compact_staging"]), sf6_registers=bool(config["sf6_registers"]),
        sync_cleanup=bool(config["sync_cleanup"]), scatter_vec4=bool(config["scatter_vec4"]),
        scatter_packed_load=bool(config["scatter_packed_load"]), sf_vec_size=16, output_tile_count_n=4,
        fc1_stages=int(config["fc1"]), fc2_stages=int(config["fc2"]), stamps=False,
        activation="swigluoai_uninterleave", swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
    k.a_dtype = cutlass.Float4E2M1FN
    k.b_dtype = cutlass.Float4E2M1FN
    k.sf_dtype = cutlass.Float8E4M3FN
    k.a_layout = utils.LayoutEnum.ROW_MAJOR
    k.b_layout = utils.LayoutEnum.ROW_MAJOR
    k.c_layout = utils.LayoutEnum.ROW_MAJOR
    return k


def main() -> int:
    k = kernel_for_served_m16()
    rows1, k1 = k.fc1_tile_n, k.fc1_tile_k      # 128 rows x 256 fp4 = 128 B rows
    rows2, k2 = k.fc2_tile_n, k.fc2_tile_k      # 256 rows x 128 fp4 = 64 B rows
    print(f"reform tile: fc1 (tile_n {rows1}, tile_k {k1}) fc2 (tile_n {rows2}, tile_k {k2})")

    @cute.jit
    def show(dummy: cute.Tensor):
        k._setup_attributes(4096)
        for name in ("b1_smem_layout_staged", "b2_smem_layout_staged", "a1_smem_layout_staged", "a2_smem_layout"):
            print(f"{name}: {getattr(k, name)}")
        for label, layout, rows, kk, pitch, kind in (("B1", k.b1_smem_layout_staged, rows1, k1, k1 // 2, 1),
                                                      ("B2", k.b2_smem_layout_staged, rows2, k2, k2 // 2, 2)):
            chunks = pitch // 16
            bad = 0
            first = ""
            for r in cutlass.range_constexpr(rows):
                for c in cutlass.range_constexpr(chunks):
                    got = int(cute.crd2idx((r, c * 32, 0), layout)) // 2     # fp4 elements -> bytes
                    if cutlass.const_expr(kind == 1):
                        want = r * pitch + ((c ^ (r % 8)) * 16)
                    else:
                        want = r * pitch + ((c ^ ((r // 2) % 4)) * 16)
                    if cutlass.const_expr(got != want):
                        bad += 1
                        first = first or f"(row {r}, chunk {c}): layout byte {got}, canonical {want}"
            print(f"MAP {label}: {rows} rows x {chunks} chunks of 16 B, pitch {pitch} B -- "
                  f"{bad} of {rows * chunks} chunks off the canonical swizzle"
                  + (f"; first {first}" if first else "; the host pre-swizzle is the canonical one"))
            # the raw map of the first sixteen rows, for the record
            for r in cutlass.range_constexpr(16):
                offs = []
                for c in cutlass.range_constexpr(chunks):
                    offs.append(int(cute.crd2idx((r, c * 32, 0), layout)) // 2)
                print(f"RAW {label} row {r}: " + " ".join(str(o) for o in offs))

    dummy = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    cute.compile(show, dummy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
