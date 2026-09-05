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
        for m in (8, 32):
            x = torch.randn(m, k, device=DEV).to(torch.bfloat16)
            twin = mk.mk_pack_twin(x, pk, n).to(torch.bfloat16)
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
    # ---- isolate the correction: corrected launch minus uncorrected launch (same
    # nibbles / scales) against the twin's x (A B)^T. NOTE: the difference of two
    # bf16 outputs carries bf16 rounding noise of the OUTPUT (~2^-9), which is
    # 15-30% of a 1-2% correction -- the "rel gap" here is dominated by that;
    # the verdict is the full-output gap above (<= 1e-4 = correct)
    print()
    print(f"{'shape':<16}{'m':>3}{'|corr|/|out|':>14}{'kern/twin':>11}{'rel gap':>9}"
          f"{'row-shift?':>11}{'col-shift?':>11}{'zero?':>7}")
    for sh in args.shapes.split(","):
        n, k = (int(v) for v in sh.split(":"))
        torch.manual_seed(0)
        w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
        os.environ["VLLM_GLM53_MK_PACK_LORC"] = str(args.rank)
        pk = mk.build_mk_weight_w4(w)
        os.environ["VLLM_GLM53_MK_PACK_LORC"] = "0"
        pk0 = mk.build_mk_weight_w4(w)
        AB = (pk[4][:n].float() @ pk[5].float())
        ext.set_gemm2(1, 0, -1)
        for m in (8, 32):
            x = torch.randn(m, k, device=DEV).to(torch.bfloat16)
            tw = x.float() @ AB.T                            # the twin's correction (fp32, on x itself)
            kc = mk._gemm_call(x, pk, n).float() - mk._gemm_call(x, pk0, n).float()
            torch.cuda.synchronize()
            out = mk._gemm_call(x, pk0, n).float()
            ratio = float(kc.norm() / tw.norm()) if tw.norm() > 0 else float("nan")
            gap = _rel(kc, tw)
            # misplacement tests: does the kernel's term match the twin's with rows or
            # 128-column tiles shifted?
            rs = min(_rel(kc[1:], tw[:-1]), _rel(kc[:-1], tw[1:])) if m > 1 else float("nan")
            cs = min(_rel(kc[:, 128:], tw[:, :-128]), _rel(kc[:, :-128], tw[:, 128:]))
            zero = float(kc.norm() / out.norm())
            print(f"{n}x{k:<10}{m:>3}{float(tw.norm() / out.norm()):>14.3e}{ratio:>11.3f}{gap:>9.2e}"
                  f"{rs:>11.2e}{cs:>11.2e}{zero:>7.1e}")
            # per-tile gap: which 128-column tiles are wrong
            tiles = [(_rel(kc[:, t*128:(t+1)*128], tw[:, t*128:(t+1)*128])) for t in range(min(n // 128, 16))]
            print("   tile gaps:", " ".join(f"{g:.1e}" for g in tiles))
    # ---- timeline (VLLM_GLM53_MK_PHASE_TS=1 builds): when the reducer
    # blocks are dispatched, pass the PDL wait and publish, against the GEMM
    # blocks' entry / loop end / exit -- late reducers or a long reducer
    # body both show here as the gap the epilogue waits through
    ts_ok = bool(list(ext.read_ts2()) or True)
    print()
    print(f"{'shape':<16}{'m':>3}{'blocks':>7}{'red disp':>9}{'red wait':>9}{'red done':>9}"
          f"{'gemm ent':>9}{'gemm loop':>10}{'gemm exit':>10}{'span':>7}{'ev us':>7}")
    for sh in args.shapes.split(","):
        n, k = (int(v) for v in sh.split(":"))
        torch.manual_seed(0)
        w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
        os.environ["VLLM_GLM53_MK_PACK_LORC"] = str(args.rank)
        pk = mk.build_mk_weight_w4(w)
        ext.set_gemm2(1, 0, -1)
        for m in (8, 32):
            x = torch.randn(m, k, device=DEV).to(torch.bfloat16)
            for _ in range(3):
                mk._gemm_call(x, pk, n)
            torch.cuda.synchronize()
            ext.read_ts2()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); mk._gemm_call(x, pk, n); e.record(); torch.cuda.synchronize()
            ev = s.elapsed_time(e) * 1e3
            ts = list(ext.read_ts2())
            if not ts:
                print("  (no stamps: build with VLLM_GLM53_MK_PHASE_TS=1)")
                break
            plan = list(ext.gemm2_plan(m, n, k))
            nblk = plan[2] + 16
            rows = [ts[u * 4:u * 4 + 4] for u in range(nblk)]
            red = [r for r in rows[:16] if r[0] > 0]
            gem = [r for r in rows[16:] if r[0] > 0 and r[3] > 0]
            if not red or not gem:
                print(f"{n}x{k:<10}{m:>3} incomplete stamps red={len(red)} gemm={len(gem)}")
                continue
            t0 = min(min(r[0] for r in red), min(r[0] for r in gem))
            us = lambda v: (v - t0) / 1e3
            rd = sorted(us(r[0]) for r in red); rw = sorted(us(r[1]) for r in red); rdn = sorted(us(r[3]) for r in red)
            ge = sorted(us(r[0]) for r in gem); gl = sorted(us(r[2]) for r in gem); gx = sorted(us(r[3]) for r in gem)
            print(f"{n}x{k:<10}{m:>3}{nblk:>7}{rd[0]:>4.0f}-{rd[-1]:<4.0f}{rw[0]:>4.0f}-{rw[-1]:<4.0f}{rdn[0]:>4.0f}-{rdn[-1]:<4.0f}"
                  f"{ge[0]:>4.0f}-{ge[-1]:<4.0f}{gl[len(gl)//2]:>5.0f}/{gl[-1]:<4.0f}{gx[len(gx)//2]:>5.0f}/{gx[-1]:<4.0f}{gx[-1]:>7.0f}{ev:>7.0f}")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
