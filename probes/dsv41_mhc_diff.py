#!/usr/bin/env python3
"""Does our mHC mixing equal DeepSeek's? No GPU, no weights.

`dsv41_mhc` exists because V4 and V4.1 PAIR the mixing coefficients
differently -- a sublayer collapses its input with the PREVIOUS sublayer's
`pre`, not its own -- and the image's V4 kernel bakes in the other pairing.
Getting the pairing right is worthless if the coefficients themselves are
wrong, so this holds the math to the reference.

The reference's `hc_mixes`, `hc_pre` and `hc_post` are plain torch inside
`inference/model.py` and are run verbatim. Its `hc_split_sinkhorn` is TileLang
(`inference/kernel.py`), which is not installed here; the kernel's own
comments give the torch equivalent, and that transcription is what
`dsv41_mhc.hc_split_sinkhorn` is. So:

  1. `hc_pre` and `hc_post` -- bit-identical against the reference's methods.
  2. `hc_mixes` -- bit-identical against the reference's method with OUR
     sinkhorn substituted into it, which isolates the projection and the
     rsqrt from the sinkhorn.
  3. the sinkhorn's own structure, checked as properties rather than against
     a copy of itself: the first pass is softmax-over-rows then a column
     normalisation, and only the remaining iterations are the symmetric pair.

The mutations this was checked against:

    hc_post summing comb over the second index      transposes the residual
                                                    mixing; shapes all check
    sinkhorn ending on a row pass                   unit rows instead of unit
                                                    columns; all finite
    hc_mixes normalising per copy, not per token    plausible coefficients
    hc_pre fusing multiply and sum into an FMA      last-bit differences

    python3 probes/dsv41_mhc_diff.py --model-py .../inference/model.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_vllm"))

import dsv41_mhc as ours  # noqa: E402


class RefBlock:
    """The reference's four methods, exec'd from model.py with our sinkhorn."""

    def __init__(self, model_py: Path, hc_mult, norm_eps, hc_eps, iters):
        text = model_py.read_text()
        start = text.index("    def hc_mixes(")
        # through the end of Block.forward: the pairing check runs the
        # reference's own forward, so cutting at `def forward` -- which the
        # first version did -- leaves exactly the method under test out.
        end = text.index("\nclass ParallelHead", start)
        src = "class _R:\n" + text[start:end]
        ns = {"torch": torch, "F": F,
              "hc_split_sinkhorn": ours.hc_split_sinkhorn}
        exec(compile(src, str(model_py), "exec"), ns)
        self._r = ns["_R"]()
        self._r.hc_mult = hc_mult
        self._r.norm_eps = norm_eps
        self._r.hc_eps = hc_eps
        self._r.hc_sinkhorn_iters = iters

    def __getattr__(self, name):
        return getattr(self._r, name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-py",
                    default="/home/choiceoh/models/DeepSeek-V4.1-Flash/"
                            "inference/model.py")
    ap.add_argument("--hc-mult", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--iters", type=int, default=20, help="hc_sinkhorn_iters")
    args = ap.parse_args()

    torch.manual_seed(20260911)
    hc, d, t = args.hc_mult, args.dim, args.tokens
    norm_eps, hc_eps = 1e-20, 1e-6
    ref = RefBlock(Path(args.model_py), hc, norm_eps, hc_eps, args.iters)

    x = (torch.randn(1, t, hc, d) * 0.4).to(torch.bfloat16)
    y = (torch.randn(1, t, d) * 0.4).to(torch.bfloat16)
    mix_hc = (2 + hc) * hc
    hc_fn = torch.randn(mix_hc, hc * d, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, dtype=torch.float32) * 0.1
    ok = True

    # -- 1. hc_mixes -------------------------------------------------------
    r_pre, r_post, r_comb = ref.hc_mixes(x, hc_fn, hc_scale, hc_base)
    o_pre, o_post, o_comb = ours.hc_mixes(
        x, hc_fn, hc_scale, hc_base, hc_mult=hc, norm_eps=norm_eps,
        sinkhorn_iters=args.iters, hc_eps=hc_eps)
    for name, a, b in (("pre", r_pre, o_pre), ("post", r_post, o_post),
                       ("comb", r_comb, o_comb)):
        same = torch.equal(a, b)
        ok &= same
        print(f"  hc_mixes {name:5s} {'OK' if same else 'MISMATCH'}  "
              f"{tuple(b.shape)}")

    # -- 2. hc_pre / hc_post ----------------------------------------------
    same = torch.equal(ref.hc_pre(x, r_pre), ours.hc_pre(x, r_pre))
    ok &= same
    print(f"  hc_pre         {'OK' if same else 'MISMATCH'}")
    r_out = ref.hc_post(y, x, r_post, r_comb)
    o_out = ours.hc_post(y, x, r_post, r_comb)
    same = torch.equal(r_out, o_out)
    ok &= same
    print(f"  hc_post        {'OK' if same else 'MISMATCH'}")
    # the control: summing comb over the other index must DIFFER, or the
    # axis was never exercised
    wrong = (r_post.unsqueeze(-1) * y.unsqueeze(-2)
             + torch.sum(r_comb.unsqueeze(-1) * x.unsqueeze(-3), dim=-2)
             ).type_as(y)
    moved = not torch.equal(r_out, wrong)
    ok &= moved
    print(f"  hc_post control: the transposed sum "
          f"{'differs, good' if moved else 'is identical -- axis untested'}")

    # -- 3. sinkhorn structure --------------------------------------------
    # A WIDE input on purpose. With a narrow one the iteration count is
    # generous enough that the matrix converges to doubly stochastic and every
    # structural question stops having an answer -- the first probe written
    # here did that and reported that the ending axis did not matter.
    mixes = torch.randn(1, t, mix_hc, dtype=torch.float32) * 3.0
    _, _, comb = ours.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc,
                                        args.iters, hc_eps)
    col_err = (comb.sum(dim=-2) - 1).abs().max().item()
    row_err = (comb.sum(dim=-1) - 1).abs().max().item()
    print(f"  sinkhorn       ends on a COLUMN pass: |col sum - 1| "
          f"{col_err:.1e}, |row sum - 1| {row_err:.1e} at {args.iters} iters")
    ok &= col_err < 1e-4
    if row_err < 1e-4:
        print("  FAIL: the rows converged too, so nothing below is a control")
        ok = False

    # Starting with the symmetric pair is NOT a control: the softmax has
    # already normalised the rows, so the extra row pass divides by
    # 1 + hc*eps. Measured rather than assumed.
    alt = torch.softmax(
        (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
         ).reshape(1, t, hc, hc), dim=-1) + hc_eps
    for _ in range(args.iters):
        alt = alt / (alt.sum(-1, keepdim=True) + hc_eps)
        alt = alt / (alt.sum(-2, keepdim=True) + hc_eps)
    print(f"  sinkhorn start symmetric-first agrees to "
          f"{(alt - comb).abs().max():.1e} -- the softmax already normalised "
          f"the rows, so it is not a control")

    # The control that IS one: end on a row pass instead.
    ended_on_row = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
    delta = (ended_on_row - comb).abs().max().item()
    ok &= delta > 1e-3
    print(f"  sinkhorn axis  ending on a row pass moves it by {delta:.1e} "
          f"{'-- the ending axis is load-bearing' if delta > 1e-3 else '-- FAIL'}")

    # And how far from converged it is, as a function of iterations, because
    # "20 is enough" is a claim about the inputs and not about the number.
    print("  sinkhorn convergence (wide input):")
    for it in (1, 4, 8, 20):
        _, _, c = ours.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, it,
                                         hc_eps)
        print(f"      {it:3d} iters   |row sum - 1| "
              f"{(c.sum(-1) - 1).abs().max():.2e}   |col sum - 1| "
              f"{(c.sum(-2) - 1).abs().max():.2e}")

    # -- 4. the PAIRING, against the reference's own Block.forward ---------
    # This is the difference between V4 and V4.1 and the reason a separate
    # kernel exists upstream. Two identity sublayers make the comparison about
    # the mixing alone: anything else would let a sublayer's own arithmetic
    # hide a mis-paired coefficient.
    ident = lambda t: t                                        # noqa: E731
    ref2 = RefBlock(Path(args.model_py), hc, norm_eps, hc_eps, args.iters)
    ref2._r.hc_attn_fn, ref2._r.hc_ffn_fn = hc_fn, hc_fn * 0.7
    ref2._r.hc_attn_scale = ref2._r.hc_ffn_scale = hc_scale
    ref2._r.hc_attn_base = ref2._r.hc_ffn_base = hc_base
    ref2._r.attn_norm = ref2._r.ffn_norm = ident
    ref2._r.attn = lambda t, *a: t
    ref2._r.ffn = lambda t, *a: t

    carried = ours.identity_pre_mix(x, hc)
    r_out, r_pre_out = ref2._r.forward(x, 0, carried, None)
    o = x
    o, attn_pre = ours.sublayer_pair(
        o, carried, ref2._r.hc_attn_fn, hc_scale, hc_base, ident, ident,
        hc_mult=hc, norm_eps=norm_eps, sinkhorn_iters=args.iters,
        hc_eps=hc_eps)
    o, ffn_pre = ours.sublayer_pair(
        o, attn_pre, ref2._r.hc_ffn_fn, hc_scale, hc_base, ident, ident,
        hc_mult=hc, norm_eps=norm_eps, sinkhorn_iters=args.iters,
        hc_eps=hc_eps)
    same = torch.equal(r_out, o) and torch.equal(r_pre_out, ffn_pre)
    ok &= same
    print(f"  pairing        {'OK' if same else 'MISMATCH'} -- output and the "
          f"pre handed forward both match Block.forward")

    # The control: V4's pairing, i.e. collapse with the pre just computed.
    v4 = x
    for fn in (ref2._r.hc_attn_fn, ref2._r.hc_ffn_fn):
        own, post, comb = ours.hc_mixes(v4, fn, hc_scale, hc_base,
                                        hc_mult=hc, norm_eps=norm_eps,
                                        sinkhorn_iters=args.iters,
                                        hc_eps=hc_eps)
        v4 = ours.hc_post(ours.hc_pre(v4, own), v4, post, comb)
    moved = not torch.equal(r_out, v4)
    ok &= moved
    d = (r_out.float() - v4.float()).abs().max().item()
    print(f"  pairing control V4's pairing (collapse with the pre just "
          f"computed) {'differs' if moved else 'is IDENTICAL -- FAIL'}, "
          f"max |d| {d:.3e}")

    print("\n" + ("MHC PASS" if ok else "MHC FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
