#!/usr/bin/env python3
"""Diagnose the T=129 conv-layout mismatch of the strided QK l2norm and try fixes.
  bash probes/run_mk_probe.sh probes/qknorm_diag.py
"""
import os
os.environ.setdefault("VLLM_GLM53_KDA_PREFILL_QK_NORM", "1")
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from kda_prefill_bench import H, K, build_pkg  # noqa: E402

kda = build_pkg()["kda"]
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(3)


def conv_pair(T):
    base = torch.randn(1, H, K, T, generator=g, device=dev, dtype=torch.float32).mul(0.8).to(torch.bfloat16)
    return base.permute(0, 3, 1, 2), base.permute(0, 3, 1, 2).clone()


def stock(q):
    return kda.l2norm_fwd(q.contiguous())


def diff_rows(got, ref):
    d = (got.float() != ref.float()).any(-1)[0]          # [T, H]
    idx = d.nonzero()
    return idx.shape[0], idx[:8].tolist()


# --- variant A: the strided (kernel2-shaped) kernel on the conv layout
def variant_strided(q, k):
    rows = q.shape[1] * 16
    qo = torch.empty(q.shape, dtype=q.dtype, device=q.device); ko = torch.empty_like(qo)
    kda._glm53_qk_l2norm_strided_kernel[(triton.cdiv(rows, 32), 2)](
        q, k, qo, ko, 1e-6, rows, QT=q.stride(1), QH=q.stride(2), QD=q.stride(3),
        KT=k.stride(1), KH=k.stride(2), KD=k.stride(3), H=16, N=128, BD=128, MBLOCK=32, num_warps=4)
    return qo, ko


# --- variant B: channel-major load, transpose to [BT, 128], reduce axis=1 like kernel2
@triton.jit
def _cm_trans_kernel(Q, K, QY, KY, TOKENS, QH: tl.constexpr, QD: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr, BT: tl.constexpr):
    token = (tl.program_id(0) * BT + tl.arange(0, BT)[None, :]).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    dim = tl.arange(0, 128)[:, None].to(tl.int64)
    if tl.program_id(2) == 0:
        xs = tl.load(Q + token + head * QH + dim * QD, token < TOKENS, other=0).to(tl.float32); out = QY
    else:
        xs = tl.load(K + token + head * KH + dim * KD, token < TOKENS, other=0).to(tl.float32); out = KY
    xt = tl.trans(xs)                                   # [BT, 128]
    tmask = tl.trans(token < TOKENS + tl.zeros_like(dim))  # [BT, 128] mask
    square = tl.broadcast_to(xt * xt, [BT, 128])
    ssum = tl.sum(tl.where(tmask, square, 0), 1)[:, None]
    inv = tl.rsqrt(ssum + 1.0e-6)
    tok_t = tl.trans(token + tl.zeros_like(dim))       # [BT, 128] token index
    dim_t = tl.trans(dim + tl.zeros_like(token))
    tl.store(out + (tok_t * 16 + head) * 128 + dim_t, xt * inv, tmask)


def variant_trans(q, k, warps=4):
    qo = torch.empty(q.shape, dtype=q.dtype, device=q.device); ko = torch.empty_like(qo)
    _cm_trans_kernel[(triton.cdiv(q.shape[1], 32), 16, 2)](
        q, k, qo, ko, q.shape[1], QH=q.stride(2), QD=q.stride(3), KH=k.stride(2), KD=k.stride(3), BT=32, num_warps=warps)
    return qo, ko


# --- variant C: the shipped channel-major kernel with other warp counts
def variant_cm(q, k, warps):
    qo = torch.empty(q.shape, dtype=q.dtype, device=q.device); ko = torch.empty_like(qo)
    kda._glm53_qk_l2norm_channel_major_kernel[(triton.cdiv(q.shape[1], 32), 16, 2)](
        q, k, qo, ko, q.shape[1], QH=q.stride(2), QD=q.stride(3), KH=k.stride(2), KD=k.stride(3), BT=32, num_warps=warps)
    return qo, ko


lengths = [1, 31, 32, 33, 63, 127, 128, 129, 130, 160, 255, 256, 257, 1000, 8185, 8192]
print("T      shipped(cm,w4)   strided-on-conv   trans(w4)   trans(w1)   cm(w1)  cm(w8)   [mismatching (token,head) rows of q for the shipped kernel]")
for T in lengths:
    q, k = conv_pair(T)
    rq, rk = stock(q), stock(k)
    res = {}
    for name, fn in (("cm4", lambda: kda._glm53_qk_l2norm_strided(q, k)), ("strided", lambda: variant_strided(q, k)),
                     ("trans4", lambda: variant_trans(q, k, 4)), ("trans1", lambda: variant_trans(q, k, 1)),
                     ("cm1", lambda: variant_cm(q, k, 1)), ("cm8", lambda: variant_cm(q, k, 8))):
        try:
            qo, ko = fn()
            nq, where = diff_rows(qo, rq); nk, _ = diff_rows(ko, rk)
            res[name] = (nq + nk, where)
        except Exception as e:  # noqa: BLE001
            res[name] = (f"ERR {type(e).__name__}", [])
    def fmt(n):
        v = res[n][0]; return ("ok" if v == 0 else str(v)).rjust(6)
    print(f"{T:6d} {fmt('cm4'):>14s} {fmt('strided'):>16s} {fmt('trans4'):>11s} {fmt('trans1'):>11s} {fmt('cm1'):>7s} {fmt('cm8'):>7s}   {res['cm4'][1] if res['cm4'][0] else ''}")
