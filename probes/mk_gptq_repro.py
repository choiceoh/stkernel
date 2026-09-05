#!/usr/bin/env python3
"""GPTQ pack build on REAL calibration dumps (/mkcache/calib2/*.pt): the
weight is the checkpoint's layer-1 projection of the same k when there is
one, else N(0, 0.02). Prints the outcome and the error vs the bf16 truth on
synthetic inputs drawn from the dump's own channel scales."""
import glob, os, sys, time, traceback, torch
sys.path.insert(0, "/repo/probes")
from mk_pack_accuracy import real_weights, _rel
DEV = "cuda"
os.environ["VLLM_GLM53_MK_PACK_CACHE"] = "off"
os.environ["VLLM_GLM53_MK_PACK_ROWSHIFT"] = "1"
os.environ["VLLM_GLM53_MK_PACK_GPTQ"] = "1"
from vllm.model_executor.layers import glm53_megakernel as mk
ws = real_weights(1)
for p in sorted(glob.glob("/mkcache/calib2/*.pt")):
    blob = torch.load(p, map_location="cpu")
    H, ntok = blob["H"], int(blob["ntok"])
    k = H.shape[0]
    name = os.path.basename(p)[:-3]
    print(f"== {name}: H {tuple(H.shape)} ntok={ntok} finite={bool(torch.isfinite(H).all())} "
          f"diag min/max={float(H.diag().min()):.3e}/{float(H.diag().max()):.3e} "
          f"sym={float((H - H.T).abs().max()):.2e}", flush=True)
    w = None
    for tag, t in ws.items():
        if t.shape[1] == k:
            w = t; break
    if w is None:
        torch.manual_seed(0); w = (torch.randn(4096, k) * 0.02).to(torch.bfloat16)
    w = w.to(DEV)
    # inputs with the dump's FULL second moment (x ~ N(0, H/ntok)): GPTQ's
    # objective is E|(W - Q) x|^2 under exactly this covariance, so this is
    # the fair offline test (diagonal-only inputs drop the correlations GPTQ
    # compensates for and read it worse than RTN -- an artefact)
    C = (H.double() / ntok); C = 0.5 * (C + C.T)
    # the real H is indefinite by fp32 accumulation rounding (negative
    # eigenvalues far below the damping the packer adds): sample through the
    # eigendecomposition with the negative part clipped
    evals, evecs = torch.linalg.eigh(C)
    print(f"   eig min/max {float(evals.min()):.3e}/{float(evals.max()):.3e} "
          f"negative {int((evals < 0).sum())}/{k}", flush=True)
    root = evecs * torch.sqrt(evals.clamp(min=0.0))[None, :]
    x = (torch.randn(256, k, dtype=torch.float64) @ root.T).to(DEV, torch.bfloat16)
    s = torch.sqrt(H.diag().float() / ntok).clamp(min=1e-6)
    x_diag = (torch.randn(256, k) * s[None, :]).to(DEV, torch.bfloat16)
    truth = x.float() @ w.float().T
    truth_diag = x_diag.float() @ w.float().T
    for arm, gptq, lr in (("rtn", False, 0), ("gptq", True, 0), ("rtn+lorc16(plainSVD)", False, 16),
                          ("gptq+lorc16", True, 16), ("gptq+lorc32", True, 32)):
        os.environ["VLLM_GLM53_MK_PACK_GPTQ"] = "1" if gptq else "0"
        os.environ["VLLM_GLM53_MK_PACK_LORC"] = str(lr)
        # the low-rank SVD is activation-aware through the Hessian: hand it over
        # for the RTN+lorc arm too (GPTQ=0 keeps the rounding RTN)
        mk._CALIB_OVERRIDE = (H, ntok) if (gptq or lr) else None
        t0 = time.time()
        try:
            pk = mk.build_mk_weight_w4(w, name=name)
            out = mk.mk_pack_twin(x, pk, w.shape[0])
            out_d = mk.mk_pack_twin(x_diag, pk, w.shape[0])
            print(f"   {arm}: OK {time.time() - t0:.1f}s err/truth full-cov {_rel(out, truth):.4e} "
                  f"diag-only {_rel(out_d, truth_diag):.4e}", flush=True)
        except Exception:
            print(f"   {arm}: FAILED after {time.time() - t0:.1f}s"); traceback.print_exc()
        mk._CALIB_OVERRIDE = None
