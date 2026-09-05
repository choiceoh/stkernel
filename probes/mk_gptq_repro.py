#!/usr/bin/env python3
"""Reproduce the GPTQ pack build on a REAL calibration dump (chain 24: 135 of
180 target packs failed silently at boot). Loads the dumped Hessian, the
checkpoint's layer weight of the same k, and runs the packer with the
Hessian handed over -- the exception, if any, is printed in full."""
import os, sys, time, traceback, torch
sys.path.insert(0, "/repo/probes")
from mk_pack_accuracy import real_weights
DEV = "cuda"
os.environ["VLLM_GLM53_MK_PACK_CACHE"] = "off"
os.environ["VLLM_GLM53_MK_PACK_ROWSHIFT"] = "1"
os.environ["VLLM_GLM53_MK_PACK_GPTQ"] = "1"
from vllm.model_executor.layers import glm53_megakernel as mk
ws = real_weights(1)
cases = [("model.layers.1.self_attn.in_proj_qkvbfg_a", ws["q_proj"]),
         ("model.layers.1.mlp.shared_experts.down_proj", ws["se.down"] if "se.down" in ws else None)]
for name, w in cases:
    p = f"/mkcache/calib/{name}.pt"
    blob = torch.load(p, map_location="cpu")
    H, ntok = blob["H"], int(blob["ntok"])
    print(f"== {name}: H {tuple(H.shape)} {H.dtype} ntok={ntok} finite={bool(torch.isfinite(H).all())} "
          f"diag min/max={float(H.diag().min()):.3e}/{float(H.diag().max()):.3e}", flush=True)
    if w is None:
        print("   (no weight of that name in the checkpoint slice; skipped)"); continue
    print(f"   weight {tuple(w.shape)} {w.dtype}", flush=True)
    w = w.to(DEV)
    mk._CALIB_OVERRIDE = (H, ntok)
    t0 = time.time()
    try:
        pk = mk.build_mk_weight_w4(w, name=name)
        print(f"   OK in {time.time() - t0:.1f}s: {mk.pack_stats_line()}", flush=True)
    except Exception:
        print(f"   FAILED after {time.time() - t0:.1f}s:", flush=True)
        traceback.print_exc()
    mk._CALIB_OVERRIDE = None
    print(f"   peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
