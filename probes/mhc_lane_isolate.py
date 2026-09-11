"""Why does the served mHC pre lane (tf32_hc_prenorm_gemm inside our tilelang
fork) raise CUDA_ERROR_INVALID_VALUE under the ST engine's check, when
probes/mhc_check.py passed? Three suspects, one run each, inside the glm53
image (bash probes/run_mk_probe.sh probes/mhc_lane_isolate.py):

  T=512 with norm_weight         (the probe ran without a norm weight)
  fn as an ARENA VIEW            (a uint8 buffer viewed as fp32 at a 256 B offset, as the loader carves it)
  called from a worker thread    (LocalTP runs the four ranks as threads)
"""
from __future__ import annotations

import threading
import traceback

import torch


def main() -> int:
    import vllm.model_executor.layers.mhc  # noqa: F401  registers torch.ops.vllm.mhc_*_tilelang
    dev = "cuda"; torch.manual_seed(0)
    hc, H, mix = 4, 4096, 24
    T = 512
    res = torch.randn(T, hc, H, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(mix, hc * H, device=dev) * 0.02
    scale = torch.tensor([1.0, 1.0, 1.0], device=dev); base = torch.zeros(mix, device=dev)
    norm_w = torch.ones(H, device=dev, dtype=torch.bfloat16)
    args = (1e-5, 1e-6, 1e-6, 2.0, 20)

    def call(fn_t, nw, label):
        try:
            out = torch.ops.vllm.mhc_pre_tilelang(res, fn_t, scale, base, *args, 1, nw, 1e-5 if nw is not None else 0.0)
            torch.cuda.synchronize()
            print(f"  {label:<44} OK  layer_input |x| {out[2].float().abs().mean().item():.4f}")
            return True
        except Exception as e:
            print(f"  {label:<44} FAIL {type(e).__name__}: {str(e)[:100]}")
            return False

    ok = call(fn, None, "main thread, fresh fn, no norm")
    ok &= call(fn, norm_w, "main thread, fresh fn, norm_weight")
    buf = torch.zeros(256 + fn.numel() * 4 + 4096, dtype=torch.uint8, device=dev)
    fn_view = buf[256:256 + fn.numel() * 4].view(torch.float32).view(mix, hc * H); fn_view.copy_(fn)
    ok &= call(fn_view, norm_w, "main thread, arena-view fn (256 B offset)")
    fn_view2 = buf[4096 - 256:4096 - 256 + fn.numel() * 4].view(torch.float32).view(mix, hc * H); fn_view2.copy_(fn)
    ok &= call(fn_view2, norm_w, "main thread, arena-view fn (3840 B offset)")
    result = {}
    def worker():
        result["ok"] = call(fn, norm_w, "worker thread, fresh fn, norm_weight")
    t = threading.Thread(target=worker); t.start(); t.join()
    ok &= result.get("ok", False)
    for T2 in (6, 64, 256, 2304):
        res2 = torch.randn(T2, hc, H, device=dev, dtype=torch.bfloat16)
        try:
            torch.ops.vllm.mhc_pre_tilelang(res2, fn, scale, base, *args, 1, norm_w, 1e-5); torch.cuda.synchronize()
            print(f"  {'main thread, T=' + str(T2):<44} OK")
        except Exception as e:
            print(f"  {'main thread, T=' + str(T2):<44} FAIL {type(e).__name__}: {str(e)[:100]}"); ok = False
    print("\n  " + ("every form runs" if ok else "some form FAILS -- see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
