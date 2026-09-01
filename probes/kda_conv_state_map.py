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

RESOLVED 2026-09-01 -- two readout defects, both found by that narrowing:

  * rec_state was stored TRANSPOSED. Stock writes element (v, k) at
    v * K + k (fused_recurrent.py, `o_v[:, None] * K + o_k[None, :]`); the
    kernel used k * KDA_D + v. The recurrence is transpose-equivariant so it
    stayed self-consistent internally, but the buffer is SHARED with the
    stock path in production. Measured at acc == T: as-is 1.005 / transposed
    6.6e-06 before, exactly swapped after.
  * The readout had no q scale. Stock does
        b_q = b_q / sqrt(sum(b_q*b_q) + 1e-6);  b_q = b_q * scale
    and kda.py:165 defaults scale to k.shape[-1] ** -0.5, so the readout ran
    sqrt(128) = 11.3x hot. The scale lands on q ONLY, which is why rec_state
    matched while attn/core/out did not -- that asymmetry is what pointed
    here.

At acc == T every component now passes: conv 2.2e-06, rec 6.6e-06,
core 1.4e-03, out 3.9e-03 against a 2e-2 gate.

WHAT IS LEFT: acc < T (core ~1.0). This is a SEMANTIC mismatch, not a bug
to hunt. Three parties disagree about the spec state slots:

  production  gdn_attn.py:266 -- spec_state_indices_tensor is
              block_table_tensor[mask, : num_spec + 1], i.e. a DISTINCT
              physical slot per draft position
  stock       stores b_h at ssm_state_indices[n, t] for EVERY token t whose
              index is > 0 (NULL_BLOCK_ID = 0 suppresses a store)
  this kernel stores once, at j == acc - 1
  the fixture torch.full((1, 8), SLOT) -- one slot for all eight, so every
              stock store collapses onto it and the last one wins

They coincide only at acc == T, which is exactly where the numbers agree.
Deciding this needs the consumer's contract (which slot the next step
gathers from), not another measurement -- and the fixture has to model
distinct slots before any acc < T number means anything. Note the
conv_state width was the same shape of trap: the FIXTURE was wrong and the
kernel was being blamed.

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


# ---------------------------------------------------------------------------
# GEMM 블록 활용률 -- 측정된 낭비와, split-K 가 왜 답이 아닌지
# ---------------------------------------------------------------------------
# mk_gemm_phase gives every block ONE 128-column tile across the whole k
# range, so the kernel's wall time is one tile-time regardless of n. Measured
# with L2 flushed (probes/megakernel_glm53_bench.py):
#
#     n      tiles  blocks busy   mk_us   stock_us
#     1024      8       8/48      210.7      60.1
#     2048     16      16/48      222.9      94.9
#     4096     32      32/48      222.9     163.6
#     6416     51      48/48      314.0     201.4
#
# The obvious fix -- give idle blocks k-slices (split-K) -- does NOT apply to
# the shapes this kernel actually runs. ks = MK_GRID / (n / 128):
#
#     phase 0 in_proj  n=6528  51 tiles  ks=1   <- 2 rounds, see below
#     phase 5 o_proj   n=4096  32 tiles  ks=1
#
# Only the probe's small-n shapes (1024, 2048) get ks = 6 and 3, and nothing
# in the model has those. A uniform split-K is dead code here.
#
# The REAL waste is phase 0. 51 tiles on 48 blocks takes TWO rounds, and the
# second round carries only 3 tiles -- 45 blocks idle for a full tile-time.
# The kernel spends ~2 tile-times doing 1.06 tiles of work, and phase 0 is
# ~43% of the KDA segment (314 us of 730).
#
# The fix that fits is a REMAINDER split, not a uniform one: run tiles
# 0..47 as today, then split the 3 leftover tiles' k 16 ways so the second
# round costs 1/16 of a tile instead of a whole one. Expected KDA 730 -> ~580
# us (2.44x -> ~3.1x vs stock). It needs the atomicAdd reduction path (an
# fp32 [m, n_orig] accumulator, zeroed under the existing grid barrier) for
# the remainder tiles only, while the full tiles keep writing bf16 directly.
