"""How thick does a calibration have to be? ROWS_TARGET decides, and one boot's blob is all a rank ever gets.

`adapter.housekeeping` files the sums the moment `progress() >= ROWS_TARGET` and disarms the observer, and a
filed blob reads as present on the next boot -- there is no path that adds tokens to a blob that exists. So the
constant is not a checkpoint interval, it is the thickness a rank lives with until someone moves its blobs aside
and spends another fleet window. Picking it by guess is picking it for good.

Three real Hessians of the same sites exist at three thicknesses, so the curve is measurable without a boot:

    17,189   the production blob        -- what every pack in production was built from
    86,620   calib-v2-heldout
    329,580  calib-v2-fit               -- the thickest, used here as the stand-in for the true distribution

Everything is SCORED on the 329,580 blob, and the arms differ only in what GPTQ was given to compensate against.
RTN is the no-Hessian ceiling; calibrating on the scoring blob itself is in-sample and is printed as a floor, not
as a result. Where the 86,620 arm falls between them says whether ROWS_TARGET wants to be ~90K or ~330K.

    python3 thickness_curve.py --dir /work/sitetable --fit /work/sitetable-fit --heldout /work/sitetable-ho
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def per_row_energy(M, H):
    Md = M.double()
    return ((Md @ H) * Md).sum(1)


def sym(H):
    H = H.float()
    return 0.5 * (H + H.T)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/sitetable")
    ap.add_argument("--fit", default="/work/sitetable-fit")
    ap.add_argument("--heldout", default="/work/sitetable-ho")
    ap.add_argument("--out", default="/work/thickness_curve.json")
    ap.add_argument("--repo", default="/repo")
    args = ap.parse_args(argv)
    sys.path.insert(0, args.repo)
    from engine.kernels.dense.packing import (_E2M1_GRID, _E2M1_MIDS, _w4_row_shift, _w4_gptq_codes,
                                              _w4_rtn_codes, mk_w4_dequant_rowmajor, gptq_factor)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mids = torch.tensor(_E2M1_MIDS, device=dev)
    grid = torch.tensor(_E2M1_GRID, device=dev)
    subset = [e for e in json.load(open(f"{args.fit}/subset.json"))
              if os.path.isfile(f"{args.dir}/{e['key']}.w.npy")
              and os.path.isfile(f"{args.dir}/{e['key']}.h.pt")
              and os.path.isfile(f"{args.heldout}/{e['key']}.ho.pt")
              and os.path.isfile(f"{args.fit}/{e['key']}.fit.pt")]
    out, t0 = [], time.time()
    print(f"{'site':<22}{'RTN':>11}{'17K':>11}{'87K':>11}{'330K*':>11}   {'87K buys':>9}")
    for e in subset:
        key = e["key"]
        u16 = torch.from_numpy(np.load(f"{args.dir}/{key}.w.npy"))
        W = ((u16.to(torch.int32) << 16).view(torch.float32)).to(dev)
        Wb = W.to(torch.bfloat16)
        n, k = W.shape
        blobs = {t: torch.load(p, map_location=dev) for t, p in
                 (("17K", f"{args.dir}/{key}.h.pt"), ("87K", f"{args.heldout}/{key}.ho.pt"),
                  ("330K", f"{args.fit}/{key}.fit.pt"))}
        truth = sym(blobs["330K"]["H"]).double()            # the thickest stands in for the distribution
        den = per_row_energy(W, truth).sum()
        need, shift, _ = _w4_row_shift(Wb, (n + 127) // 128 * 128, k // 16, True)

        def score(codes, scales):
            nib = codes[:, ::2] | (codes[:, 1::2] << 4)
            D = W - mk_w4_dequant_rowmajor(nib, scales, wgs=1.0, rgs=torch.exp2(-shift))[:n]
            return float((per_row_energy(D, truth).sum() / den).clamp(min=0).sqrt())

        row = dict(e, ntok={t: int(b["ntok"]) for t, b in blobs.items()})
        row["rtn"] = score(*_w4_rtn_codes(Wb, shift, need, mids, grid))
        for tag in ("17K", "87K", "330K"):
            H = sym(blobs[tag]["H"])
            factor = gptq_factor(H, percdamp=0.01, act_order=True, factor_device="cpu")
            row[tag] = score(*_w4_gptq_codes(Wb, shift, need, H, mids, grid, act_order=True, factor=factor))
            del H, factor
        span = row["17K"] - row["330K"]                      # 330K is in-sample: a floor, not a result
        row["fraction"] = (row["17K"] - row["87K"]) / span if span > 0 else float("nan")
        out.append(row)
        print(f"{key:<22}{row['rtn']:>11.3e}{row['17K']:>11.3e}{row['87K']:>11.3e}{row['330K']:>11.3e}"
              f"{100 * row['fraction']:>8.0f}%")
        del W, Wb, blobs, truth
        if dev == "cuda":
            torch.cuda.empty_cache()
        json.dump(out, open(args.out, "w"))
    ok = sorted(r["fraction"] for r in out if r["fraction"] == r["fraction"])
    print(f"\n* the 330K column is IN-SAMPLE (scored on its own blob): a floor, not an achievable number.")
    print(f"median fraction of the 17K->330K span that 87K already buys: {100 * ok[len(ok) // 2]:.0f}%"
          f"   ({len(out)} sites, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
