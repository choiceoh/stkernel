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
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_static_kernel_v4 import (  # noqa: E402
    MoEStaticKernelV4,
)


def main() -> int:
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

    dummy = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    cute.compile(show, dummy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
