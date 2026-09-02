#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cost of quantising the MLA query to e4m3, measured where it matters.

The score s = q.c decides which of the indexer's top-k slots dominate the
softmax, so the question is not the value error but whether the ATTENTION
OUTPUT and the effective ranking move. Compares, on the serving geometry:
  bf16 q  (what the kernel and FlashInfer both use)  <- reference
  e4m3 q  (what the k32 fp8 mma would require)
against an fp32 reference, over realistic activation scales."""
import sys, torch
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
torch.manual_seed(0)
dev = "cuda"
D, H, W = 512, 16, 2048
def e4m3(x):
    # per-tensor scale so the max lands near e4m3's 448 ceiling (best case)
    s = x.abs().amax().clamp_min(1e-6) / 448.0
    return (x / s).to(torch.float8_e4m3fn).float() * s
rows = []
for trial in range(8):
    q32 = torch.randn(8, H, D, device=dev) * 0.3
    c32 = torch.randn(W, D, device=dev) * 0.5
    c8 = c32.to(torch.float8_e4m3fn).float()          # the cache is e4m3 either way
    sm = D ** -0.5
    def attend(qq):
        s = (qq @ c8.T) * sm
        p = torch.softmax(s.float(), dim=-1)
        return p @ c8, s
    ref, s_ref = attend(q32)                                   # fp32 q
    bf, s_bf = attend(q32.to(torch.bfloat16).float())          # bf16 q (today)
    fp, s_fp = attend(e4m3(q32))                               # e4m3 q (fp8 mma)
    def stat(o, s):
        rel = ((o - ref).norm() / ref.norm()).item()
        top1 = (s.argmax(-1) != s_ref.argmax(-1)).float().mean().item()
        t8 = lambda a: a.topk(8, -1).indices.sort(-1).values
        top8 = (t8(s) != t8(s_ref)).any(-1).float().mean().item()
        # attention mass captured by the reference's top-32 slots
        idx = s_ref.topk(32, -1).indices
        pm = torch.softmax(s.float(), -1).gather(-1, idx).sum(-1).mean().item()
        return rel, top1, top8, pm
    rows.append((stat(bf, s_bf), stat(fp, s_fp)))
import statistics as st
for name, i in (("bf16 q (today)", 0), ("e4m3 q (fp8 mma)", 1)):
    rel = st.mean(r[i][0] for r in rows); t1 = st.mean(r[i][1] for r in rows)
    t8 = st.mean(r[i][2] for r in rows); pm = st.mean(r[i][3] for r in rows)
    print(f"{name:18s} out rel {rel:.2e} | top-1 바뀜 {t1:6.2%} | top-8 집합 바뀜 {t8:6.2%} | 기준 top-32 질량 {pm:.4f}")
