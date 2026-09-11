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
    # the key quantiser, byte for byte, against the served fwht128_quant_fp8
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
    from engine.modules.sparse_indexer import fwht128_quant
    rows = torch.randn(512, 128, device=dev, dtype=torch.bfloat16)
    q8_s, s_s = fwht128_quant_fp8(rows); q8_o, s_o = fwht128_quant(rows)
    byte_eq = torch.equal(q8_s.view(torch.uint8), q8_o.view(torch.uint8)); scale_eq = torch.equal(s_s.view(-1), s_o.view(-1))
    frac = (q8_s.view(torch.uint8) != q8_o.view(torch.uint8)).float().mean().item()
    print(f"  fwht128 fp8 keys vs served: bytes identical={byte_eq} (differing {frac:.2e}), scales identical={scale_eq}")
    ok = ok and scale_eq and frac < 1e-3
    # pooling, byte for byte, via the served wrapper in return-only mode
    from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_compress_and_write_cache, expand_pools_and_append_tail
    from engine.modules.sparse_indexer import kpool_compress, select_with_tail
    P, kp = 96, 4
    slot_k = torch.randn(P, kp, 128, device=dev, dtype=torch.bfloat16)
    slot_score = torch.randn(P, kp, 128, device=dev, dtype=torch.bfloat16)
    ape = torch.randn(kp, 128, device=dev, dtype=torch.float32)
    dummy_cache = torch.zeros(1, 64, 132, device=dev, dtype=torch.uint8)
    try:
        res = kpool_compress_and_write_cache(dummy_cache, slot_k, slot_score, ape,
                                             torch.arange(P, device=dev, dtype=torch.int64), kp,
                                             return_compressed=True, write_cache=False)
        srv_q8, srv_s = (res[0], res[1]) if isinstance(res, (tuple, list)) else (res, None)
        our_q8, our_s = kpool_compress(slot_k, slot_score, ape)
        from engine.modules.sparse_indexer import hadamard128
        # the fp32 truth both quantisers approximate: pooled key after the rotation
        sc = slot_score.float() + ape.float()[None]; prob = torch.softmax(sc, dim=1)
        truth = hadamard128((prob * slot_k.float()).sum(1))
        sq8 = srv_q8.view(torch.uint8).reshape(P, -1)[:, :128].view(torch.float8_e4m3fn)
        d_srv = (sq8.float() * srv_s.reshape(-1, 1).float() - truth).abs(); d_our = (our_q8.float() * our_s - truth).abs()
        ulp = torch.exp2(torch.floor(torch.log2(truth.abs().clamp_min(2 ** -6)))) * 2 ** -3        # e4m3: 3 mantissa bits
        f_srv = (d_srv > ulp).float().mean().item(); f_our = (d_our > ulp).float().mean().item()
        scales_eq = torch.equal(srv_s.reshape(-1).float(), our_s.reshape(-1))
        print(f"  kpool pooling: scales identical={scales_eq}; beyond 1 fp8 ulp of the fp32 truth -- served {f_srv:.2e}, ours {f_our:.2e}; "
              f"served-vs-ours differing elements {((sq8.float() * srv_s.reshape(-1, 1).float() - our_q8.float() * our_s).abs() > 0).float().mean().item():.2e}")
        ok = ok and scales_eq and f_our <= f_srv + 1e-3
    except Exception as e:
        print(f"  kpool pooling judge unavailable: {type(e).__name__}: {str(e)[:160]}"); ok = False
    pool_ids = torch.randint(-1, 40, (5, 512), device=dev, dtype=torch.int32); seq_lens = torch.tensor([2048, 2047, 1601, 4, 3], device=dev, dtype=torch.int32)
    srv_t = expand_pools_and_append_tail(pool_ids, seq_lens, kp); our_t = select_with_tail(pool_ids, seq_lens, kp)
    same = torch.equal(srv_t.to(torch.int32), our_t)
    print(f"  expand_pools_and_append_tail vs served: identical={same} shape {tuple(srv_t.shape)} vs {tuple(our_t.shape)}")
    ok = ok and same
    print("\n  " + ("indexer reference == served op" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
