#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the b12x static kernel's smem layouts (swizzle included) on the CPU.

    MK_PROBE_NO_GPU=1 bash probes/run_mk_probe.sh probes/b12x_static_layout_print.py

The tile-major weight layout (cell t) can be pre-swizzled in global memory so
a stage's B tile is one 1-D cp.async.bulk copy; the permutation to apply on
the host is exactly the ComposedLayout printed here.
"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.environ.get("MK_PKG_PATH", "/usr/local/lib/python3.12/dist-packages"))
import torch  # noqa: E402

torch.cuda.is_available = lambda: True  # type: ignore[assignment]
torch.cuda.get_device_capability = lambda *a, **k: (12, 1)  # type: ignore[assignment]

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cutlass.utils as utils  # noqa: E402
import cutlass.utils.blockscaled_layout as blockscaled_utils  # noqa: E402
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_static_kernel_v4 import (  # noqa: E402
    MoEStaticKernelV4,
)


def _runs(offsets):
    """the offsets, sorted, folded into (start, end) contiguous runs"""
    out = []
    for o in sorted(offsets):
        if out and o == out[-1][1] + 1:
            out[-1][1] = o
        else:
            out.append([o, o])
    return [(a, b) for a, b in out]


def main() -> int:
    dump = "--dump" in sys.argv
    sf = "--sf" in sys.argv
    verify = "--verify-swz" in sys.argv
    k = MoEStaticKernelV4(sf_vec_size=16, output_tile_count_n=4,
                          activation="swigluoai_uninterleave", swiglu_alpha=1.0,
                          swiglu_beta=0.0, swiglu_limit=10.0)
    k.a_dtype = cutlass.Float4E2M1FN
    k.b_dtype = cutlass.Float4E2M1FN
    k.sf_dtype = cutlass.Float8E4M3FN
    k.a_layout = utils.LayoutEnum.ROW_MAJOR
    k.b_layout = utils.LayoutEnum.ROW_MAJOR
    k.c_layout = utils.LayoutEnum.ROW_MAJOR
    # layouts need an MLIR context: build them inside a traced jit function
    # (the trace runs on the CPU; nothing is launched)
    @cute.jit
    def show(dummy: cute.Tensor):
        k._setup_attributes(4096)
        for name in ("a1_smem_layout_staged", "b1_smem_layout_staged",
                     "sfa1_smem_layout_staged", "sfb1_smem_layout_staged",
                     "b2_smem_layout_staged", "sfb2_smem_layout_staged",
                     "a2_smem_layout", "sfa2_smem_layout"):
            print(f"{name}: {getattr(k, name)}")
        print("smem bytes:", k.smem_bytes)
        if dump:
            # every (row, k) of stage 0 -> element offset, swizzle applied: the
            # permutation a host pre-swizzle must reproduce for a 1-D bulk copy
            for r in cutlass.range_constexpr(64):
                offs = []
                for c in cutlass.range_constexpr(32):
                    offs.append(int(cute.crd2idx((r, c * 16, 0), k.b1_smem_layout_staged)))
                print("MAP B1 " + str(r) + " " + " ".join(str(o) for o in offs))
            for r in cutlass.range_constexpr(128):
                offs = []
                for c in cutlass.range_constexpr(8):
                    offs.append(int(cute.crd2idx((r, c * 16, 0), k.b2_smem_layout_staged)))
                print("MAP B2 " + str(r) + " " + " ".join(str(o) for o in offs))

        if sf:
            # Can a TMA box read HALF the rows of an SF block (cell h: one FC1
            # stage wants only the 64 N rows its own half consumes)? The SM120
            # block-scaled layout stores a 128-row x 512-k block as 4 KB in
            # which the four 32-row groups are INTERLEAVED at 4 B: the row mode
            # is ((32,4),..) with stride ((16,4),..), so rows 0-31 own bytes
            # +0..3 of every 16, rows 32-63 +4..7, and so on. A TMA box's
            # innermost dimension must be contiguous and a multiple of 16 B.
            sfb = blockscaled_utils.tile_atom_to_shape_SF((512, 4096, 288),
                                                          k.sf_vec_size)
            print("global SFB(w13) layout:", sfb)
            per_row = {}
            for r in cutlass.range_constexpr(128):
                offs = []
                for c in cutlass.range_constexpr(32):
                    offs.append(int(cute.crd2idx((r, c * 16, 0), sfb)))
                per_row[r] = offs
            for rows in (128, 64, 32):
                cover = [o for r in range(rows) for o in per_row[r]]
                runs = _runs(cover)
                lens = sorted({b - a + 1 for a, b in runs})
                print(f"SF box rows={rows:3d}: {len(cover)} bytes in "
                      f"{len(runs)} contiguous runs, run lengths {lens}, "
                      f"first runs {runs[:4]} -> innermost "
                      f"{'OK (>= 16 B)' if min(lens) >= 16 else 'ILLEGAL (< 16 B)'}")

        if verify:
            # cell z end to end, on the CPU: the byte the kernel reads at smem
            # byte d must be the plain tile-major byte moe_dispatch's host
            # permutation put there. perm[d] is a GATHER index: the swizzled
            # storage is built as swz[d] = plain[perm[d]], and the bulk copy
            # lands swz linearly, so smem byte d holds plain[perm[d]].
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
                _swizzle_perms,
            )
            perm16k, perm8k = _swizzle_perms(torch.device("cpu"))
            bad = 0
            first = ""
            for r in cutlass.range_constexpr(64):
                for j in cutlass.range_constexpr(256):   # bytes per row per k tile
                    e = int(cute.crd2idx((r, 2 * j, 0), k.b1_smem_layout_staged))
                    want = r * 256 + j
                    got = int(perm16k[e // 2])
                    if cutlass.const_expr(got != want):
                        bad += 1
                        first = first or (
                            f"B1 (row {r}, byte {j}) -> smem byte {e // 2}: "
                            f"host perm says plain byte {got}, kernel wants {want}")
            print(f"VERIFY B1: {bad} / {64 * 256} bytes disagree"
                  + (f" -- first: {first}" if first else ""))
            bad2 = 0
            first2 = ""
            for r in cutlass.range_constexpr(128):
                for j in cutlass.range_constexpr(64):
                    e = int(cute.crd2idx((r, 2 * j, 0), k.b2_smem_layout_staged))
                    want = r * 64 + j
                    got = int(perm8k[e // 2])
                    if cutlass.const_expr(got != want):
                        bad2 += 1
                        first2 = first2 or (
                            f"B2 (row {r}, byte {j}) -> smem byte {e // 2}: "
                            f"host perm says plain byte {got}, kernel wants {want}")
            print(f"VERIFY B2: {bad2} / {128 * 64} bytes disagree"
                  + (f" -- first: {first2}" if first2 else ""))

    dummy = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    cute.compile(show, dummy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
