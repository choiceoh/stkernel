"""What a fresh calibration would be worth, priced without a fleet window.

Production's blobs are a thin sum: ntok 17,189, and `_gptq_inverse_factor`'s damping ladder steps to 10% on
them, which is the packer saying the Hessian is not positive-definite at 1%. A recalibration boot would sum a
thicker one -- but a boot costs production, so the question is whether it pays before spending it.

Both calibrations already exist on disk for the same sites, so the comparison is free:

    thin    the production blob, ntok 17,189      -- what every pack in production was built from
    thick   calib-v2-fit, ntok 329,580 (19x)      -- the band a fresh sum would land in

Both are scored on the SAME held-out blob (calib-v2-heldout), so the only thing that varies is what GPTQ was
given to compensate against. The gap is the ceiling on what recalibration can buy.

Honest confound: `calib_run.sh` drops the NVIDIA rank file's page cache, so calib-v2 was very likely summed on
the ModelOpt arm rather than production's. This therefore mixes a token-count effect with an arm effect, and
today's provenance measurement says the arm effect alone runs +43..114% in the wrong direction. So a thick
number that beats thin is a LOWER bound on recalibration's value (it wins despite the arm handicap); a thick
number that loses says nothing clean. Read it that way.

    python3 recalibration_value.py --dir /work/sitetable --fit /work/sitetable-fit --heldout /work/sitetable-ho
"""
import argparse
import json
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
    ap.add_argument("--out", default="/work/recalibration_value.json")
    ap.add_argument("--repo", default="/repo")
    args = ap.parse_args(argv)
    sys.path.insert(0, args.repo)
    from engine.kernels.dense.packing import (_E2M1_GRID, _E2M1_MIDS, _w4_row_shift, _w4_gptq_codes,
                                              _w4_rtn_codes, mk_w4_dequant_rowmajor, gptq_factor)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mids = torch.tensor(_E2M1_MIDS, device=dev)
    grid = torch.tensor(_E2M1_GRID, device=dev)
    subset = json.load(open(f"{args.fit}/subset.json"))
    import os
    have = [e for e in subset if os.path.isfile(f"{args.dir}/{e['key']}.w.npy")
            and os.path.isfile(f"{args.dir}/{e['key']}.h.pt")
            and os.path.isfile(f"{args.heldout}/{e['key']}.ho.pt")
            and os.path.isfile(f"{args.fit}/{e['key']}.fit.pt")]
    if len(have) != len(subset):
        print(f"({len(have)} of {len(subset)} subset sites present; the rest are not staged)")
    subset = have
    out, t0 = [], time.time()
    print(f"{'site':<22}{'rtn':>11}{'thin GPTQ':>12}{'thick GPTQ':>12}{'thick gain':>12}{'ntok thin':>11}")
    for e in subset:
        key = e["key"]
        u16 = torch.from_numpy(np.load(f"{args.dir}/{key}.w.npy"))
        W = ((u16.to(torch.int32) << 16).view(torch.float32)).to(dev)
        Wb = W.to(torch.bfloat16)
        n, k = W.shape
        thin_b = torch.load(f"{args.dir}/{key}.h.pt", map_location=dev)
        thick_b = torch.load(f"{args.fit}/{key}.fit.pt", map_location=dev)
        ho = torch.load(f"{args.heldout}/{key}.ho.pt", map_location=dev)
        Hho64 = sym(ho["H"]).double()
        den = per_row_energy(W, Hho64).sum()
        need, shift, _ = _w4_row_shift(Wb, (n + 127) // 128 * 128, k // 16, True)

        def score(codes, scales):
            nib = codes[:, ::2] | (codes[:, 1::2] << 4)
            D = W - mk_w4_dequant_rowmajor(nib, scales, wgs=1.0, rgs=torch.exp2(-shift))[:n]
            return float((per_row_energy(D, Hho64).sum() / den).clamp(min=0).sqrt())

        rtn = score(*_w4_rtn_codes(Wb, shift, need, mids, grid))
        row = dict(e, rtn=rtn, ntok_thin=int(thin_b["ntok"]), ntok_thick=int(thick_b["ntok"]))
        for tag, blob in (("thin", thin_b), ("thick", thick_b)):
            H = sym(blob["H"])
            factor = gptq_factor(H, percdamp=0.01, act_order=True, factor_device="cpu")
            row[tag] = score(*_w4_gptq_codes(Wb, shift, need, H, mids, grid, act_order=True, factor=factor))
            del H, factor
        row["gain"] = 1.0 - row["thick"] / row["thin"]
        out.append(row)
        print(f"{key:<22}{rtn:>11.3e}{row['thin']:>12.3e}{row['thick']:>12.3e}"
              f"{100 * row['gain']:>11.0f}%{row['ntok_thin']:>11,}")
        del W, Wb, thin_b, thick_b, ho, Hho64
        if dev == "cuda":
            torch.cuda.empty_cache()
        json.dump(out, open(args.out, "w"))
    good = [r for r in out if r["thin"] > 0]
    print(f"\nmedian thick-vs-thin gain: {100 * sorted(r['gain'] for r in good)[len(good) // 2]:.0f}%"
          f"   ({len(out)} sites, {time.time() - t0:.0f}s)")
    worse = [r["key"] for r in out if r["thin"] > r["rtn"]]
    if worse:
        print(f"sites where the THIN calibration made GPTQ worse than RTN: {len(worse)} -- {worse[:6]}")


if __name__ == "__main__":
    main()
