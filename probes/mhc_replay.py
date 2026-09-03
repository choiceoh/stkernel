"""Pin down an mhc replay-stability failure.

megakernel_glm53_bench.py only reports one number (`rep`), which says a
repeated call diverged but not which output, on which call, or whether the
inputs survived. This walks the same call N times and prints all three.
"""
import os
import torch

os.environ.setdefault("VLLM_GLM53_MK_MHC", "1")
DEV = "cuda"
NAMES = ("residual_out", "post_mix", "comb_mix", "layer_input")


def _rel(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d / max(s, 1e-30)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    # The bench's `rep` cell -- the failure this tool exists to pin down -- is
    # measured at the driver's SINKHORN_SERVED, so walking the same call at a
    # different count would reproduce a longer sinkhorn chain's drift as
    # clean. Unset means that constant; pass a number to compare bases.
    ap.add_argument("--sinkhorn", type=int, default=None)
    args = ap.parse_args()

    from vllm.model_executor.layers import glm53_megakernel as mk

    sk = mk.SINKHORN_SERVED if args.sinkhorn is None else args.sinkhorn
    print(f"sinkhorn_repeat={sk}"
          f"{' (driver default)' if args.sinkhorn is None else ''}")
    torch.cuda.init()
    mk._build()

    bad = False
    for T in (8, 32):
        torch.manual_seed(0)
        x = torch.randn(T, 4096, dtype=torch.bfloat16, device=DEV) * 0.1
        res = torch.randn(T, 4, 4096, dtype=torch.bfloat16, device=DEV) * 0.1
        pm = torch.rand(T, 4, dtype=torch.float32, device=DEV)
        cm = torch.rand(T, 4, 4, dtype=torch.float32, device=DEV)
        fn = torch.randn(24, 16384, dtype=torch.float32, device=DEV) * 0.02
        nw = torch.randn(4096, dtype=torch.bfloat16, device=DEV)
        # Inputs are cloned so a call that writes through one of its own
        # arguments shows up as a nonzero input drift rather than as a
        # mystery output drift on the NEXT call.
        keep = {"x": x.clone(), "res": res.clone(), "pm": pm.clone(),
                "cm": cm.clone(), "fn": fn.clone(), "nw": nw.clone()}

        def call():
            return mk._mhc_call(x, res, pm, cm.reshape(T, 16).contiguous(), fn,
                                mk.hc_scale_ones(), mk.hc_base_zeros(), nw, T,
                                1e-6, 1e-6, 1e-6, 1.0, 1e-6, sk)

        base = tuple(g.clone() for g in call())
        torch.cuda.synchronize()
        print(f"--- T={T} ---")
        for i in range(2, 8):
            got = tuple(g.clone() for g in call())
            torch.cuda.synchronize()
            devs = [_rel(g, b) for g, b in zip(got, base)]
            flag = "!" if max(devs) > 1e-6 else " "
            cells = "  ".join(f"{n}={d:.1e}" for n, d in zip(NAMES, devs))
            print(f"{flag}call {i}: {cells}")
            bad |= max(devs) > 1e-6
        drift = {k: _rel(v, keep[k]) for k, v in
                 (("x", x), ("res", res), ("pm", pm), ("cm", cm),
                  ("fn", fn), ("nw", nw))}
        worst = max(drift.values())
        print(f" 입력 변조: {'  '.join(f'{k}={v:.1e}' for k, v in drift.items())}")
        if worst > 0:
            print("  -> 호출이 자기 입력을 덮어쓴다 (재현성 실패의 원인)")
    print("REPLAY:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
