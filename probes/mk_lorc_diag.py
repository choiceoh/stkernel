#!/usr/bin/env python3
"""33차 lever 4 diagnostic: the v2 kernel's low-rank correction against
the pack's twin per forced slice count (ksr), per row count (m), and
across back-to-back launches (a stale or unreset scratch shows as a
second launch disagreeing with the first).

    mkprobe.sh probes/mk_lorc_diag.py [--shapes 1024:4096,6416:4096,4096:512]
"""
import argparse, os, sys, torch

DEV = "cuda"


def _rel(a, b):
    a = a.float(); b = b.float(); n = b.norm()
    return float((a - b).norm() / n) if n > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1024:4096,2048:4096,6416:4096,4096:512")
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()
    os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
    os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
    os.environ.setdefault("VLLM_GLM53_MK_GEMM2", "1")
    os.environ["VLLM_GLM53_MK_PACK_CACHE"] = "off"
    os.environ["VLLM_GLM53_MK_PACK_ROWSHIFT"] = "1"
    os.environ["VLLM_GLM53_MK_PACK_GPTQ"] = "0"
    from vllm.model_executor.layers import glm53_megakernel as mk
    ext = mk._build()
    print(f"{'shape':<16}{'m':>3}{'ksr':>5}{'plan':>12}{'gap#1':>10}{'gap#2':>10}{'gap#3 (uncorr between)':>24}")
    ok = True
    for sh in args.shapes.split(","):
        n, k = (int(v) for v in sh.split(":"))
        torch.manual_seed(0)
        w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
        os.environ["VLLM_GLM53_MK_PACK_LORC"] = str(args.rank)
        pk = mk.build_mk_weight_w4(w)
        os.environ["VLLM_GLM53_MK_PACK_LORC"] = "0"
        pk0 = mk.build_mk_weight_w4(w)
        assert pk[4] is not None and pk0[4] is None
        deq = mk.mk_pack_dequant(pk, n).float()
        for m in (8, 32):
            x = torch.randn(m, k, device=DEV).to(torch.bfloat16)
            twin = (mk._mk_quant_x_ref(x) @ deq.T).to(torch.bfloat16)
            for ksr in (0, 1, 2, 4, 8):
                ext.set_gemm2(1, ksr, -1)
                try:
                    plan = list(ext.gemm2_plan(m, n, k))
                except Exception:
                    plan = None
                g1 = mk._gemm_call(x, pk, n); torch.cuda.synchronize()
                g2 = mk._gemm_call(x, pk, n); torch.cuda.synchronize()
                mk._gemm_call(x, pk0, n); torch.cuda.synchronize()   # an uncorrected launch between
                g3 = mk._gemm_call(x, pk, n); torch.cuda.synchronize()
                gaps = [_rel(g, twin) for g in (g1, g2, g3)]
                good = all(v < 5e-3 for v in gaps)
                ok &= good
                print(f"{'!' if not good else ' '}{n}x{k:<10}{m:>3}{ksr:>5}{str(plan[:3]) if plan else '-':>12}"
                      f"{gaps[0]:>10.2e}{gaps[1]:>10.2e}{gaps[2]:>24.2e}")
        ext.set_gemm2(1, 0, -1)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
