"""Hold modules/hyper_connection.mhc_pre/mhc_post to GLM-5.3's SERVED mHC
kernels -- our TileLang fork, mounted on its target inside the glm53 image
(the same way probes/kda_check.py judges KDA). Run there, not on the host.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.modules.hyper_connection import mhc_pre, mhc_post


def main() -> int:
    import vllm.model_executor.layers.mhc as vmhc            # registers torch.ops.vllm.mhc_*_tilelang
    import importlib
    mhc_torch = importlib.import_module("vllm.model_executor.kernels.mhc.torch")   # the package __init__ shadows the name with real torch
    torch.manual_seed(0); dev = "cuda"
    T, hc, hidden = 96, 4, 4096                              # GLM: hc_mult 4, hidden 4096
    mix_hc = (2 + hc) * hc
    residual = torch.randn(T, hc, hidden, device=dev, dtype=torch.bfloat16)
    fn = (torch.randn(mix_hc, hc * hidden, device=dev) * 0.02).float()
    hc_scale = torch.tensor([1.0, 1.0, 1.0], device=dev); hc_base = (torch.randn(mix_hc, device=dev) * 0.1).float()
    args = (1e-6, 1e-6, 1e-6, 2.0, 20)                      # rms_eps, pre_eps, sinkhorn_eps, post_mult, repeat
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
    ours = mhc_pre(residual, fn, hc_scale, hc_base, *args)
    ref = mhc_torch.mhc_pre_torch(residual, fn, hc_scale, hc_base, *args)
    e_ref = [rel(a, b) for a, b in zip(ours, ref)]
    # CustomOp instantiation asserts a vLLM config context; the registered op
    # itself does not need one, so call it as forward_cuda does (mhc.py:62-74).
    served = torch.ops.vllm.mhc_pre_tilelang(residual, fn, hc_scale, hc_base, *args, 1, None, 0.0)
    e_served = [rel(a, b) for a, b in zip(ours, served)]
    print(f"  mhc_pre  vs vLLM torch reference: post {e_ref[0]:.1e} comb {e_ref[1]:.1e} layer_input {e_ref[2]:.1e}")
    print(f"  mhc_pre  vs SERVED tilelang:      post {e_served[0]:.1e} comb {e_served[1]:.1e} layer_input {e_served[2]:.1e}")
    x = torch.randn(T, hidden, device=dev, dtype=torch.bfloat16)
    o_ours = mhc_post(x, residual, served[0], served[1])
    o_ref = mhc_torch.mhc_post_torch(x, residual, served[0], served[1])
    o_served = torch.ops.vllm.mhc_post_tilelang(x, residual, served[0], served[1])
    print(f"  mhc_post vs torch reference {rel(o_ours, o_ref):.1e}, vs SERVED tilelang {rel(o_ours, o_served):.1e}")
    ok = max(e_ref) < 1e-3 and max(e_served) < 2e-2 and rel(o_ours, o_served) < 2e-2
    print("\n  " + ("mHC reference == served kernels" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
