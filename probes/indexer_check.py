"""Hold modules/sparse_indexer.indexer_logits to the served DeepGEMM op
`fp8_fp4_mqa_logits` (FP8 path: q fp8 with its per-token scale folded into
`weights`, k fp8 with per-row fp32 scales). Run inside the glm53 image."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.modules.sparse_indexer import indexer_logits


def main() -> int:
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    torch.manual_seed(0); dev = "cuda"
    M, H, D, N = 64, 32, 128, 1024                       # GLM indexer: 32 heads x 128, 2048-token windows
    q = torch.randn(M, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(N, D, device=dev, dtype=torch.bfloat16)
    w = torch.rand(M, H, device=dev, dtype=torch.float32)
    # served quantisation: q per token (amax/448), k per row (amax/448); q's scale folds into weights
    q_s = q.float().abs().amax(dim=(1, 2), keepdim=True) / 448.0
    q8 = (q.float() / q_s).to(torch.float8_e4m3fn)
    k_s = k.float().abs().amax(dim=1) / 448.0
    k8 = (k.float() / k_s[:, None]).to(torch.float8_e4m3fn)
    w_folded = w * q_s.view(M, 1)
    cu_q = torch.arange(M + 1, device=dev, dtype=torch.int32) * 0            # single sequence
    try:
        served = fp8_fp4_mqa_logits((q8, None), (k8, k_s.contiguous()), w_folded,
                                    torch.zeros(M, device=dev, dtype=torch.int32),          # cu_seqlen_ks (start)
                                    torch.full((M,), N, device=dev, dtype=torch.int32),      # cu_seqlen_ke (end)
                                    clean_logits=False)
    except TypeError as e:
        print(f"  signature differs: {e}"); import inspect; print(inspect.signature(fp8_fp4_mqa_logits)); return 1
    # reference on the SAME quantised values (the judge is the formula, not the quantiser)
    qd = (q8.float() * q_s).to(torch.bfloat16); kd = (k8.float() * k_s[:, None]).to(torch.bfloat16)
    ours = indexer_logits(qd, kd, w)
    served_f = served.float()[:, :N]
    rel = ((ours - served_f).abs().max() / served_f.abs().max().clamp_min(1e-6)).item()
    print(f"  indexer logits vs served fp8_fp4_mqa_logits: rel {rel:.2e}  (|logits| max {served_f.abs().max().item():.1f})")
    ok = rel < 3e-2
    print("\n  " + ("indexer reference == served op" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
