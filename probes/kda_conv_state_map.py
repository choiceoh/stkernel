#!/usr/bin/env python3
"""Where does each conv_state slot come from? -- the KDA gate's open item.

The megakernel's KDA segment passes rec_state (1.6e-2) and fails on
conv_state (~0.5) and out (~4.2). This probe answers the question that
decides the fix, and it answers it by MEASUREMENT rather than by reading two
kernels against each other:

  1. map   -- for each of the three state slots, which input position (or
              initial-state slot) holds that exact value in the stock arm
  2. cmp   -- MK vs stock per channel, with the worst channel printed
  3. seg   -- are the mismatches confined to the q / k / v thirds

What it has already established (2026-09-01, srv2 scratch container):

  * acc=1 stock = [init[1], init[2], x[0]] -- a clean left-shift by one, and
    exactly what the MK writes. ch0 matches to 4 decimals.
  * acc=3,5,8 stock still ends with x[0], and its two older slots hold
    values that appear in NEITHER the inputs NOR the initial state. So the
    reference itself does not advance by `acc` tokens the way the MK
    assumes, and "fix the MK to match" is not yet a settled instruction.
  * The mismatch is uniform across q/k/v (~64% each at acc=1), so it is not
    a channel-range indexing bug.
  * phase 3 is guarded (`head < KDA_H`), the grid barrier has the release/
    acquire fence pair, and every phase boundary has one -- so the obvious
    race and OOB candidates are already excluded.

conv_state is now exact (2.2e-06). The remaining `out` gap (~3-4) is
localised by `stock_run(debug=True)`, which returns the pipeline split:

    attn  -- the recurrence readout (phase 3)
    core  -- after the gated RMSNorm (phase 4)
    out   -- after o_proj (phase 5)

Measured 2026-09-01, narrowing the `out` gap phase by phase:

    g1      9.9e-08   phase 1 -- exact
    g2      0.0e+00   phase 1 -- exact
    core    1.9-2.5   phase 4 output
    out     2.8-3.8   phase 5 output

phase 5 only carries the error (core is already wrong), and phase 1 is
exact, so the gates are not it. `core = norm(attn, g2)` with an exact g2
and a norm whose formula matches the stock one leaves the recurrence
READOUT (`attn`, phase 3) as the remaining candidate.

The state is separately fine: rec_state passes at acc=8 (1.6e-2). Note the
asymmetry -- `attn` is written for EVERY query token while rec_state is
written only at `j == acc - 1`, and rec_state's error grows as acc shrinks
(1.6e-2 at acc=8, 1.2e-1 at acc=1). The fixture sets ssm_state_indices to
the same slot for all 8 positions, so the stock arm stores its state at
every token and ends with the full-sequence state, while the MK stores once
at the accepted boundary. Those agree only at acc == T, which is exactly
what the numbers show -- so the fixture's index tensor is itself worth
checking against what production passes before reading rec_state at low
acc as a kernel defect.

Run it in the scratch container that probes/run_megakernel_bench.sh builds.
"""
import sys

import torch

from vllm.model_executor.layers import glm53_megakernel as mk
from vllm.model_executor.layers.glm53_fp8_dense import _fp8_dense_gemm

SLOT = mk._KdaFixture.SLOT


def _qkv(fx):
    """The conv input the stock arm sees, rebuilt from the fixture."""
    sq, sws, rin, cin = mk._stock_fp8_pair(fx.w_in)
    proj = _fp8_dense_gemm(fx.x, sq, sws, rin, cin)
    return proj.split([mk.KDA_QKV, mk.KDA_H, mk.KDA_D, mk.KDA_D], dim=-1)[0]


def probe_map(accs, ch=0):
    for acc in accs:
        fx = mk._KdaFixture(acc=acc)
        qkv = _qkv(fx)
        ref = fx.stock_run()["conv_state"][SLOT]
        init = fx.conv_st[SLOT]
        print(f"\n=== map acc={acc} ch{ch} ===")
        print(f"  input  {[round(v, 4) for v in qkv[:, ch].float().tolist()]}")
        print(f"  init   {[round(v, 4) for v in init[ch].tolist()]}")
        print(f"  stock  {[round(v, 4) for v in ref[ch].tolist()]}")
        for i in range(ref.shape[1]):
            val = ref[ch][i].item()
            hits = [f"x[{t}]" for t in range(qkv.shape[0])
                    if abs(qkv[t, ch].float().item() - val) < 2e-3]
            hits += [f"init[{j}]" for j in range(init.shape[1])
                     if abs(init[ch][j].item() - val) < 2e-3]
            print(f"    slot[{i}] {val:+.4f} -> {hits or 'NO MATCH'}")


def probe_cmp(accs):
    for acc in accs:
        fx = mk._KdaFixture(acc=acc)
        ref = fx.stock_run()["conv_state"][SLOT].float()
        got = fx.mk_run()["conv_state"][SLOT].float()
        per_ch = (ref - got).abs().max(dim=1).values
        worst = int(per_ch.argmax())
        print(f"\n=== cmp acc={acc} ===")
        print(f"  mismatched {int((per_ch > 2e-2).sum())}/{per_ch.numel()}")
        for c in (0, worst):
            print(f"  ch{c:<5} stock {[round(v, 4) for v in ref[c].tolist()]}")
            print(f"  {'':<7} mk    {[round(v, 4) for v in got[c].tolist()]}")


def probe_seg(accs):
    h, d = mk.KDA_H, mk.KDA_D
    for acc in accs:
        fx = mk._KdaFixture(acc=acc)
        ref = fx.stock_run()["conv_state"][SLOT].float()
        got = fx.mk_run()["conv_state"][SLOT].float()
        bad = (ref - got).abs().max(dim=1).values > 2e-2
        print(f"\n=== seg acc={acc}  {int(bad.sum())}/{bad.numel()} ===")
        for name, lo, hi in (("q", 0, h * d), ("k", h * d, 2 * h * d),
                             ("v", 2 * h * d, 3 * h * d)):
            seg = bad[lo:hi]
            print(f"  {name} [{lo}:{hi}] {int(seg.sum()):5d}"
                  f"  ({100 * seg.float().mean():.1f}%)")


def main() -> int:
    mk._build()
    accs = [int(a) for a in sys.argv[1:]] or [1, 3, 8]
    probe_map(accs)
    probe_cmp(accs)
    probe_seg(accs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
