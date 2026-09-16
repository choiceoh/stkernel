"""How much room is there for the draft FC bias? An upper bound, with no capture.

`draft_fc_bias` fits a per-output-channel MEAN correction to the drafter's FP8 decode FC. That mean
is (W_q - W) @ mu, where mu is the average aux input -- which only the (unbuilt) pair collection can
supply. What does NOT need mu is the bound: |(W_q - W) mu| <= ||W_q - W|| ||mu||, so if the FP8
weight error is negligible the correction is negligible whatever mu turns out to be.

This reports the FP8 round-trip error of fc.weight under the repo's own quantizer, per output row
and relative to the row's own scale. CPU only, no CUDA context.
"""
import json
import os
import struct
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.environ.get("MK_PKG_PATH", "/usr/local/lib/python3.12/dist-packages"))
sys.path.insert(0, "/repo")

import torch  # noqa: E402

RAW = "/cache/draft-fc-weight.bin"   # fc.weight, extracted beside the cache (no model mount here)


def load():
    meta = json.loads(open(RAW + ".json").read())
    with open(RAW, "rb") as f:
        raw = f.read()
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*meta["shape"])


def main() -> int:
    from engine.kernels.dense.packing import fp8_rtn, FP8_BLOCK
    W = load()
    print(f"fc.weight {tuple(W.shape)} {W.dtype}, FP8_BLOCK={FP8_BLOCK}", flush=True)
    q8, s8 = fp8_rtn(W)
    per = s8.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    Wq = (q8.float() * per)[: W.shape[0], : W.shape[1]]
    D = Wq - W.float()
    row_err = D.norm(dim=1)
    row_ref = W.float().norm(dim=1)
    rel = (row_err / row_ref.clamp(min=1e-12))
    q = torch.tensor([0.5, 0.9, 0.99, 1.0])
    print("per-output-row relative FP8 error  ||dW|| / ||W||")
    print("   median %.4e   p90 %.4e   p99 %.4e   max %.4e"
          % tuple(torch.quantile(rel, q).tolist()))
    print("   frobenius %.4e" % (D.norm() / W.float().norm()).item())
    print()
    # The bound above is the worst case, where dW aligns with mu. Quantization error is not aligned,
    # so what actually couples to a real mean input is its DC part: with mu_j ~ mubar for all j,
    # (dW mu)_i -> mubar * sum_j dW_ij. That row sum is computable without mu, and it is what a mean
    # correction would remove. Compare it against the same contraction of W itself.
    dw_sum, w_sum = D.sum(dim=1), W.float().sum(dim=1)
    dc = (dw_sum.abs() / w_sum.abs().clamp(min=1e-12))
    q = torch.tensor([0.5, 0.9, 0.99, 1.0])
    print("DC coupling  |sum_j dW_ij| / |sum_j W_ij|   (what a uniform mean input would see)")
    print("   median %.4e   p90 %.4e   p99 %.4e   max %.4e" % tuple(torch.quantile(dc, q).tolist()))
    # and the isotropic expectation, for contrast: a random-direction error projects down by sqrt(k)
    iso = (row_err / row_ref.clamp(min=1e-12)) / (W.shape[1] ** 0.5)
    print("   isotropic expectation for contrast: median %.4e" % iso.median().item())
    print()
    print("aligned worst case %.2e   DC coupling %.2e   isotropic %.2e"
          % (rel.median().item(), dc.median().item(), iso.median().item()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
