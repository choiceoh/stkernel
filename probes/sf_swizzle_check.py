"""Hold modules/nvfp4_sf.swizzle_sf to flashinfer's block_scale_interleave (the
layout the b12x MoE lane eats), inside the glm53 image:
    bash probes/run_mk_probe.sh probes/sf_swizzle_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.modules.nvfp4_sf import swizzle_sf, unswizzle_sf


def main() -> int:
    """The served layer converts through vllm.utils.flashinfer.flashinfer_convert_sf_to_mma_layout,
    a lazy wrapper over flashinfer.fp4_quantization.block_scale_interleave (CUDA, per arch);
    the trtllm op the older utils.swizzle_sf names is not registered in this image."""
    from flashinfer.fp4_quantization import block_scale_interleave
    ok = True
    for E, m, k in ((3, 1024, 4096), (2, 4096, 512), (1, 256, 2048), (2, 200, 1024)):
        s = k // 16
        sf = torch.randint(0, 255, (E, m, s), dtype=torch.uint8, device="cuda")
        theirs = block_scale_interleave(sf).reshape(-1)
        ours = torch.cat([swizzle_sf(sf[e]) for e in range(E)]).reshape(-1)
        same = theirs.numel() == ours.numel() and torch.equal(theirs, ours)
        back = all(torch.equal(unswizzle_sf(swizzle_sf(sf[e]), m, s), sf[e]) for e in range(E))
        print(f"  E={E} [{m}, {k}] (sf [{m}, {s}]): identical to flashinfer {same} ({theirs.numel()} vs {ours.numel()} bytes), inverse {back}")
        ok = ok and same and back
    print("\n  " + ("swizzle == flashinfer block_scale_interleave" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
