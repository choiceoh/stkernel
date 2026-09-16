"""What the W4A8 decode lane and the FP8 prefill lane actually lose on the KDA projections.

Both lanes are already running on these weights -- DenseLinear dispatches <=32 rows to W4A8
and everything above to FP8 -- so this is not a proposal, it is a reading of the two lanes.

33차 answered a version of this twice and got different answers: a synthetic Hessian said
GPTQ -13%, the real dumps said -69~74%. The difference was the probe's design, so this one
takes both halves from real calibration and keeps them apart: GPTQ is CALIBRATED on the
`fit` dump and SCORED on the `heldout` dump, which are different streams. Scoring on the
calibration Hessian flatters GPTQ; scoring on an unrelated distribution punishes it. The
in-sample column is printed beside the held-out one so the gap is visible.

The packs are the fleet's own (engine/kernels/dense/packing.py), not a reimplementation --
the RTN arm reproduces 33차's recorded ~8.3% weight error, which is what makes the rest
readable. `--production` uses the served options (act_order=True = GPTQ_ACT_ORDER, per-row
shift); the plain-order arm is printed beside it because on these two weights it is better.

Channel smoothing is not applied: its factors are POW2, which kernels/dense/smoothing records
as exactly nothing on the FP8 and W4A8 lanes (45차 §23 조사 9차). Per-row shift is 33차's
lever 3, measured there as zero.

Score is the exact expected relative output error under the held-out distribution,
    sqrt( tr(D H D^T) / tr(W H W^T) ),   D = W - dequant(pack(W))
i.e. E||xD^T||^2 / E||xW^T||^2 for x with second moment H. No sampling noise, and separable
by row, so the in_proj row groups come out of one matmul.

Run (ost-97x, x86_64 sm_120 check image; any CUDA box with the packer will do):

    docker run --rm --gpus all -v ~/kda-err:/work st-engine:glm53-sm120-x86 \
        /work/kda_pack_error.py --layer 1 --dir /work

`--dir` holds, for layer L:
    kda_l<L>.npz               {"L<L>.kda.in_proj", "L<L>.kda.o_proj"} as uint16 bf16 bits
    L<L>.fit.<h>.pt            calib-v2-fit/mkcalib/rank<r>/Glm5NextForCausalLM/
    L<L>.heldout.<h>.pt          model.layers.<L>.self_attn.{in_proj_qkvbfg_a,o_proj}.pt
with <h> in (in_proj_qkvbfg_a, o_proj). See README.md for the extraction.
"""
import argparse
import sys
import time

import numpy as np
import torch

# in_proj rows are 3*Hl*D + Hl + 2*D (specs.py:125), split by tree_decode.py:169 as
# q/k/v | beta | fa | ga. Only the 272 gate rows are true bf16 -- see kda_weight_bits.py.
IN_PROJ_GROUPS = [("q/k/v", 0, 6144), ("beta", 6144, 6160), ("fa", 6160, 6288),
                  ("ga", 6288, 6416), ("ALL", 0, 6416)]
HESSIAN = {"in_proj": "in_proj_qkvbfg_a", "o_proj": "o_proj"}


def w4_dequant(weight, hessian=None, act_order=False, per_row=True):
    """The served W4 pack of `weight`, dequantised to fp32 [rows, cols]. RTN when hessian is None."""
    from engine.kernels.dense.packing import (_E2M1_GRID, _E2M1_MIDS, _w4_row_shift, _w4_rtn_codes,
                                              _w4_gptq_codes, mk_w4_dequant_rowmajor, gptq_factor)
    rows, cols = weight.shape
    mids = torch.tensor(_E2M1_MIDS, device=weight.device)
    grid = torch.tensor(_E2M1_GRID, device=weight.device)
    need, shift, _clamped = _w4_row_shift(weight, (rows + 127) // 128 * 128, cols // 16, per_row)
    if hessian is None:
        codes, scales = _w4_rtn_codes(weight, shift, need, mids, grid)
    else:
        factor = gptq_factor(hessian, percdamp=0.01, act_order=act_order, factor_device="cpu")
        codes, scales = _w4_gptq_codes(weight, shift, need, hessian, mids, grid,
                                       act_order=act_order, factor=factor)
    nibbles = codes[:, ::2] | (codes[:, 1::2] << 4)
    rgs = torch.exp2(-shift) if per_row else None
    wgs = 1.0 if per_row else float(torch.exp2(-shift[0]))
    return mk_w4_dequant_rowmajor(nibbles, scales, wgs=wgs, rgs=rgs)[:rows]


def fp8_dequant(weight, hessian=None):
    """The FP8 lane's weights (DeepGEMM 128x128 UE8M0 block scale), dequantised to fp32."""
    from engine.kernels.dense.packing import FP8_BLOCK, fp8_rtn, fp8_gptq, gptq_factor
    rows = weight.shape[0]
    if hessian is None:
        q, scale = fp8_rtn(weight)
    else:
        q, scale = fp8_gptq(weight, hessian, factor=gptq_factor(hessian, 0.01, True, "cpu"))
    per = scale.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    return (q.float() * per)[:rows]


def per_row_energy(M, H):
    """(M H M^T)_rr for every row r, in fp64 -- the row-separable half of the score."""
    Md = M.double()
    return ((Md @ H) * Md).sum(1)


def load_hessian(path, device):
    blob = torch.load(path, map_location=device)
    H = blob["H"].float()
    return 0.5 * (H + H.T), int(blob["ntok"])            # fp32 accumulation leaves it slightly asymmetric


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dir", default="/work")
    ap.add_argument("--repo", default=None, help="repo root, when engine/ is not already importable")
    args = ap.parse_args(argv)
    if args.repo:
        sys.path.insert(0, args.repo)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    raw = np.load(f"{args.dir}/kda_l{args.layer}.npz")
    print(f"device {dev} | torch {torch.__version__} | layer {args.layer}")

    for site in ("in_proj", "o_proj"):
        key = f"L{args.layer}.kda.{site}"
        u16 = torch.from_numpy(raw[key].astype(np.uint16))
        W = ((u16.to(torch.int32) << 16).view(torch.float32)).to(dev)      # bf16 bits -> fp32
        Wb = W.to(torch.bfloat16)                                          # the packers take bf16
        groups = IN_PROJ_GROUPS if site == "in_proj" else [("ALL", 0, W.shape[0])]
        h = HESSIAN[site]
        Hf, nf = load_hessian(f"{args.dir}/L{args.layer}.fit.{h}.pt", dev)
        Hh, nh = load_hessian(f"{args.dir}/L{args.layer}.heldout.{h}.pt", dev)
        ev = torch.linalg.eigvalsh(Hh.double()).clamp(min=0)
        participation = float(ev.sum() ** 2 / (ev ** 2).sum())
        print(f"\n== {key} {tuple(W.shape)} | fit {nf:,} tok, heldout {nh:,} tok")
        print(f"   heldout H participation ratio {participation:.0f} of {W.shape[1]} dims")

        Hh64, Hf64 = Hh.double(), Hf.double()
        den_h, den_f = per_row_energy(W, Hh64), per_row_energy(W, Hf64)
        arms = (("W4  RTN", lambda w: w4_dequant(w)),
                ("W4  GPTQ act_order [served]", lambda w: w4_dequant(w, Hf, act_order=True)),
                ("W4  GPTQ plain order", lambda w: w4_dequant(w, Hf, act_order=False)),
                ("FP8 RTN", lambda w: fp8_dequant(w)),
                ("FP8 GPTQ [served]", lambda w: fp8_dequant(w, Hf)))
        print("   " + f"{'arm':<30}" + "".join(f"{g[0]:>10}" for g in groups)
              + f"{'in-sample':>12}{'||D||/||W||':>13}")
        for label, pack in arms:
            t0 = time.time()
            Q = pack(Wb)
            D = W - Q
            num_h = per_row_energy(D, Hh64)
            cells = "".join(
                f"{float((num_h[a:b].sum() / den_h[a:b].sum()).clamp(min=0).sqrt()):>10.3e}"
                for _, a, b in groups)
            ins = float((per_row_energy(D, Hf64).sum() / den_f.sum()).clamp(min=0).sqrt())
            fro = float(D.norm() / W.norm())
            print(f"   {label:<30}{cells}{ins:>12.3e}{fro:>13.3e}   ({time.time() - t0:.0f}s)")
            del Q, D, num_h
            if dev == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
