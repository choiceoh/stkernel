#!/usr/bin/env python3
"""Warm flashinfer's mm_fp4 JIT (the cutlass backend) for the fleet's dense
prefill shapes OUTSIDE a serving boot, with a bounded compile parallelism.

Why: the first NVFP4P boot (28차/32차) built the nvfp4 pairs inside the
serve process and srv3 went from ~25 GiB free to 5 GiB (earlyoom) while
the first mm_fp4 shape compiled -- a 26차-class cliff. The JIT cache lives
under FLASHINFER_WORKSPACE_BASE (/cache), so one warm per node makes the
next serve boot compile nothing. Run through probes/run_mk_probe.sh on each
node (no serving container up), MAX_JOBS=4."""
import os
import sys
import time

os.environ.setdefault("MAX_JOBS", "4")
import torch  # noqa: E402

sys.path.insert(0, os.environ.get("MK_PKG_PATH", ""))
from vllm.model_executor.layers import glm53_fp8_dense as fd  # noqa: E402

# per-rank TP=4 dense shapes of GLM-5.3-Flash (N, K): KDA in_proj, o_proj,
# shared gate_up, shared down, MLA q_a / kv_a / q_b / o
SHAPES = [(6144, 4096), (4096, 4096), (1024, 4096), (4096, 512),
          (1536, 4096), (576, 4096), (4096, 1536), (4096, 1024)]
M = int(os.environ.get("NVFP4_WARM_M", "4096"))
dev = "cuda"
worst = 0.0
for n, k in SHAPES:
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.05)
    x = torch.randn(M, k, dtype=torch.bfloat16, device=dev)
    t0 = time.time()
    wq, wsf, w_gs = fd._quantize_nvfp4(w)
    got = None
    for alpha in (1.0, -1.0):
        try:
            out = fd._nvfp4_dense_gemm(x, wq, wsf, w_gs, n, alpha)
            torch.cuda.synchronize()
            ref = x.float() @ w.float().t()
            rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-6)).item()
            if rel < 0.5:
                got = (alpha, rel); break
        except Exception as e:  # noqa: BLE001
            print(f"  [{n}x{k}] alpha={alpha}: {type(e).__name__}: {str(e)[:80]}")
    dt = time.time() - t0
    free, total = torch.cuda.mem_get_info()
    print(f"[{n:>5}x{k:<5}] M={M} {'alpha=%s rel=%.3e' % got if got else 'FAILED'}  {dt:6.1f}s  cuda free {free/2**30:.1f}/{total/2**30:.1f} GiB")
    worst = max(worst, got[1] if got else 9.9)
    del w, x, wq, wsf
    torch.cuda.empty_cache()
print("VERDICT", "PASS" if worst < 0.5 else "FAIL", f"worst rel {worst:.3e}")
