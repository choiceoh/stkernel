"""Hold engine/kernels.py to what the TileLang sources say, without tilelang.

Each check compares against something derived a DIFFERENT way -- a dense
attention, an unquantized matmul, the doubly-stochastic property -- rather than
against the same code path. A round-trip that only checks itself passes for a
kernel that is uniformly wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import kernels as K


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)
    good = True

    # 1. fp8 round trip: error must be inside e4m3's own resolution (2^-3 rel).
    x = torch.randn(64, 256, device=dev, dtype=torch.float32) * 3
    y, s = K.act_quant(x, block_size=128)
    deq = (y.float().unflatten(-1, (2, 128)) * s.float().unsqueeze(-1)).flatten(-2)
    rel = ((deq - x).abs() / x.abs().clamp_min(1e-3)).max().item()
    good &= check("act_quant round trip", rel < 0.08, f"max rel {rel:.4f}")

    # 2. fp4 pack/unpack must be exact on table values, and place the EVEN
    #    element in the low nibble -- swap them and this is the check that dies.
    table = K.FP4_TABLE.to(dev)
    vals = table[torch.randint(0, 8, (16, 64), device=dev)]
    packed, ps = K.fp4_act_quant(vals, block_size=32)
    back = K._unpack_fp4(packed)
    back = (back.unflatten(-1, (2, 32)) * ps.float().unsqueeze(-1)).flatten(-2)
    good &= check("fp4 pack/unpack exact on table values",
                  torch.allclose(back, vals.float(), atol=1e-6),
                  f"max err {(back - vals.float()).abs().max().item():.2e}")

    # 3. fp8_gemm against an unquantized matmul of the same dequantized values.
    a = torch.randn(32, 256, device=dev, dtype=torch.float32)
    b = torch.randn(128, 256, device=dev, dtype=torch.float32)
    aq, a_s = K.act_quant(a, 128)
    bq, b_s = K.act_quant(b.unflatten(0, (1, 128)).squeeze(0), 128)
    b_s_block = b_s[::128] if b_s.size(0) >= 128 else b_s[:1].expand(1, b_s.size(1))
    got = K.fp8_gemm(aq, a_s, bq, b_s_block, block_size=128)
    ad = (aq.float().unflatten(-1, (2, 128)) * a_s.float().unsqueeze(-1)).flatten(-2)
    bd = (bq.float().unflatten(-1, (2, 128))
          * b_s_block.float().repeat_interleave(128, 0)[:128].unsqueeze(-1)).flatten(-2)
    want = ad @ bd.T
    err = ((got.float() - want).abs() / want.abs().clamp_min(1e-2)).max().item()
    good &= check("fp8_gemm == dequantized matmul", err < 0.02, f"max rel {err:.4f}")

    # 4. sparse_attn against a dense attention over the SAME gathered positions.
    torch.set_default_dtype(torch.bfloat16)
    b_, m, h, d, n, topk = 2, 5, 4, 32, 64, 8
    q = torch.randn(b_, m, h, d, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(b_, n, d, device=dev, dtype=torch.bfloat16)
    sink = torch.randn(h, device=dev, dtype=torch.float32)
    idxs = torch.randint(0, n, (b_, m, topk), device=dev, dtype=torch.int32)
    idxs[0, 0, 3:] = -1                                   # a partially masked row
    scale = d ** -0.5
    got = K.sparse_attn(q, kv, sink, idxs, scale)

    ref = torch.zeros_like(got, dtype=torch.float32)
    for bb in range(b_):
        for mm in range(m):
            keep = [int(i) for i in idxs[bb, mm].tolist() if i != -1]
            if not keep:
                continue
            kk = kv[bb, keep].float()                      # [k, d]
            sc = (q[bb, mm].float() @ kk.T) * scale        # [h, k]
            mx = sc.amax(dim=-1, keepdim=True)
            w = torch.exp(sc - mx)
            den = w.sum(-1) + torch.exp(sink - mx.squeeze(-1))
            ref[bb, mm] = (w @ kk) / den.unsqueeze(-1)
    err = (got.float() - ref).abs().max().item()
    good &= check("sparse_attn == dense attention on the kept positions",
                  err < 5e-2, f"max abs {err:.2e}")

    # 5. the all-(-1) row: zero, not NaN. This is the -1e30 seed's whole job.
    idxs2 = torch.full((1, 1, topk), -1, device=dev, dtype=torch.int32)
    out = K.sparse_attn(q[:1, :1], kv[:1], sink, idxs2, scale)
    good &= check("all-(-1) row is zero, not NaN",
                  torch.isfinite(out).all().item() and out.abs().max().item() == 0)

    # 6. Sinkhorn: comb must come out doubly stochastic.
    hc, iters = 4, 20
    mixes = torch.randn(1, 7, (2 + hc) * hc, device=dev, dtype=torch.float32)
    pre, post, comb = K.hc_split_sinkhorn(
        mixes, torch.ones(3, device=dev), torch.zeros((2 + hc) * hc, device=dev), hc, iters)
    rows = comb.sum(-1)
    cols = comb.sum(-2)
    good &= check("sinkhorn comb is doubly stochastic",
                  (rows - 1).abs().max() < 2e-2 and (cols - 1).abs().max() < 2e-2,
                  f"row dev {(rows - 1).abs().max():.3e} col dev {(cols - 1).abs().max():.3e}")
    good &= check("pre in (eps, 1+eps), post in (0, 2)",
                  bool((pre > 0).all() and (pre < 1.01).all()
                       and (post >= 0).all() and (post <= 2).all()))
    print("\n  " + ("ALL PASS" if good else "SOMETHING FAILED"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
