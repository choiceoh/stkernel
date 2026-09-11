"""Hold modules/sparse_attention.mla_sparse_mqa to GLM-5.3's SERVED sparse-MLA
lane: our megakernel's `mla_decode` (CUDA) and its own torch twin
`mla_decode_ref`, with the overlay mounted on its target inside the glm53
image and the persistent build cache at /cache. D4 at kernel level.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.modules.sparse_attention import mla_sparse_mqa


def main() -> int:
    from vllm.model_executor.layers import glm53_megakernel as m
    torch.manual_seed(0); dev = "cuda"
    T, H, D, S, K = 8, m.MLA_H, m.MLA_D, 4096, 2048           # GLM: 16 heads/rank, 512 latent, top-k 2048
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    ckv_scale = 2.0
    cache8 = (torch.randn(S, D, device=dev) / ckv_scale).to(torch.float8_e4m3fn)
    slots = torch.stack([torch.randperm(S, device=dev)[:K] for _ in range(T)]).to(torch.int32).contiguous()
    lens = torch.tensor([K, 1500, 64, 1, K, 700, 2047, 300], device=dev, dtype=torch.int32)
    scale = 256 ** -0.5                                         # qk_head_dim 256 (nope), as served
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
    ours = mla_sparse_mqa(q, cache8, slots, lens, scale, ckv_scale)
    twin = m.mla_decode_ref(q, cache8, slots, lens, scale, ckv_scale)
    print(f"  ours vs megakernel torch twin (mla_decode_ref): rel {rel(ours, twin):.2e}")
    ok = rel(ours, twin) < 2e-2
    try:
        m.maybe_arm()
        armed = bool(m._ARMED.get("mla"))
        print(f"  megakernel MLA lane armed: {armed} (ENABLE_MLA={getattr(m, 'ENABLE_MLA', None)})")
        if armed:
            t0 = time.perf_counter()
            lane = m.mla_decode(q.contiguous(), cache8.view(torch.uint8), slots, lens, scale, ckv_scale)
            torch.cuda.synchronize()
            print(f"  ours vs SERVED mla_decode (CUDA lane): rel {rel(ours, lane):.2e}  ({(time.perf_counter() - t0) * 1e3:.1f} ms incl. first launch)")
            ok = ok and rel(ours, lane) < 2e-2
    except Exception as e:
        print(f"  lane unavailable: {type(e).__name__}: {str(e)[:200]}")
    print("\n  " + ("MLA reference == served lane" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
