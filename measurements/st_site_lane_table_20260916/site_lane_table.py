"""What each dense site costs in the W4 decode lane versus the FP8 lane, in bytes and in injected error.

The engine assigns lanes by rule -- DenseLinear sends <=32 rows to W4A8 and everything above to FP8 -- so no
site's decode format was ever chosen by what it costs. This table is the per-site evidence for choosing: for
every weight rank 3 packs, the bytes each lane holds and the error each lane injects, measured with the site's
own real calibration Hessian (the production blobs, summed under the weights production serves, so the
calibration is matched -- see store.fits_weights).

The score is ABSOLUTE injected energy on a HELD-OUT Hessian, not the relative in-sample error:

    E_lane = tr(D H_heldout D^T),   D = W - dequant(pack_lane(W))

Calibration and scoring are separated on purpose. GPTQ minimises tr(D H D^T) on the H it was given, and
production's blobs are a thin sum (ntok 17,189, and the damping ladder fires at 10% on them), so scoring on
that same H flatters the pack -- and flatters W4 MORE than FP8, because the coarser grid leaves more error to
overfit, which would bias the whole allocation against moving bytes to FP8. So the pack is calibrated on the
production blob (the weights production serves, so the calibration is matched) and scored on calib-v2-heldout
(19x the tokens, a different stream). Both numbers are recorded: `rel_w4` / `rel_fp8` are held out,
`rel_w4_in` / `rel_fp8_in` are in-sample, and the gap between them is how much the thin blob was overfit.

Relative error normalises each site by its own output energy, which makes a site that contributes little to the
residual stream look as important as one that dominates it -- and 2026-09-16 showed where that leads: the GPTQ
expert hybrid halved its blocks' relative error and made end-to-end NLL worse by 0.050 nats. Absolute energy is
in one unit across sites, so the sites that write into the same residual stream are directly comparable.

It is still a proxy. It does not know how a site's error propagates to the loss, and nothing offline does. The
end-to-end channel is the head-NLL rig (measurements/st_hybrid_head_nll_20260916), which needs a boot per
assignment -- so this table PROPOSES an assignment and a fleet window DISPOSES of it. Read the caveats in the
README before spending bytes on it.

    python3 site_lane_table.py --dir /work/sitetable --out /work/site_lane_table.json
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

W4_BYTES_PER = 0.5625          # 4-bit element + one e4m3 scale byte per 16
FP8_BYTES_PER = 1.0            # e4m3 element; the 128x128 UE8M0 block scales are ~0.002% on top


def per_row_energy(M, H):
    Md = M.double()
    return ((Md @ H) * Md).sum(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/sitetable")
    ap.add_argument("--heldout", default="/work/sitetable-ho", help="blobs to SCORE on; calibration stays --dir")
    ap.add_argument("--out", default="/work/site_lane_table.json")
    ap.add_argument("--repo", default="/repo")
    ap.add_argument("--limit", type=int, default=0, help="stop after N sites (a smoke run)")
    args = ap.parse_args(argv)
    sys.path.insert(0, args.repo)
    from engine.kernels.dense.packing import (_E2M1_GRID, _E2M1_MIDS, _w4_row_shift, _w4_gptq_codes,
                                              mk_w4_dequant_rowmajor, gptq_factor, fp8_gptq, FP8_BLOCK)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    index = json.load(open(f"{args.dir}/index.json"))
    if args.limit:
        index = index[:args.limit]
    mids = torch.tensor(_E2M1_MIDS, device=dev)
    grid = torch.tensor(_E2M1_GRID, device=dev)
    rows_out = []
    t_start = time.time()
    for i, entry in enumerate(index):
        key = entry["key"]
        u16 = torch.from_numpy(np.load(f"{args.dir}/{key}.w.npy"))
        W = ((u16.to(torch.int32) << 16).view(torch.float32)).to(dev)
        Wb = W.to(torch.bfloat16)
        n, k = W.shape
        blob = torch.load(f"{args.dir}/{key}.h.pt", map_location=dev)
        H = blob["H"].float()
        H = 0.5 * (H + H.T)
        H64 = H.double()                                   # in-sample: what GPTQ optimised on
        ho = torch.load(f"{args.heldout}/{key}.ho.pt", map_location=dev)
        Hho = ho["H"].float()
        Hho64 = (0.5 * (Hho + Hho.T)).double()             # held out: what the score is read from
        del Hho
        signal = float(per_row_energy(W, Hho64).sum())
        signal_in = float(per_row_energy(W, H64).sum())

        factor = gptq_factor(H, percdamp=0.01, act_order=True, factor_device="cpu")   # shared by both lanes
        need, shift, _ = _w4_row_shift(Wb, (n + 127) // 128 * 128, k // 16, True)
        codes, scales = _w4_gptq_codes(Wb, shift, need, H, mids, grid, act_order=True, factor=factor)
        nib = codes[:, ::2] | (codes[:, 1::2] << 4)
        D4 = W - mk_w4_dequant_rowmajor(nib, scales, wgs=1.0, rgs=torch.exp2(-shift))[:n]
        e_w4 = float(per_row_energy(D4, Hho64).sum())
        e_w4_in = float(per_row_energy(D4, H64).sum())
        del codes, scales, nib, D4

        q8, s8 = fp8_gptq(Wb, H, factor=factor)
        per = s8.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
        D8 = W - (q8.float() * per)[:n]
        e_fp8 = float(per_row_energy(D8, Hho64).sum())
        e_fp8_in = float(per_row_energy(D8, H64).sum())
        del q8, s8, per, D8, factor, H, H64, Hho64

        rows_out.append(dict(entry, params=n * k, ntok=int(blob["ntok"]), ntok_ho=int(ho["ntok"]),
                             signal=signal, e_w4=e_w4, e_fp8=e_fp8,
                             rel_w4=(e_w4 / signal) ** 0.5, rel_fp8=(e_fp8 / signal) ** 0.5,
                             e_w4_in=e_w4_in, e_fp8_in=e_fp8_in,
                             rel_w4_in=(e_w4_in / signal_in) ** 0.5, rel_fp8_in=(e_fp8_in / signal_in) ** 0.5,
                             bytes_w4=n * k * W4_BYTES_PER, bytes_fp8=n * k * FP8_BYTES_PER))
        del W, Wb, blob, ho
        if dev == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i + 1 == len(index):
            print(f"  {i + 1}/{len(index)} sites, {time.time() - t_start:.0f}s", flush=True)
        json.dump(rows_out, open(args.out, "w"))       # resumable evidence: written as it goes
    print(f"wrote {args.out}: {len(rows_out)} sites in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
