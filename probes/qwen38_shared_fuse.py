#!/usr/bin/env python3
"""Is fusing the shared expert into the routed batch EXACT? No GPU, no weights.

The fused lane replaces

    out = sum_{e in topk} w_e * E_e(x)  +  g * S(x)

with one grouped GEMM over topk + 1 slots. That is only admissible if it is the
same number, so this runs both and requires equality -- and then attacks the
three ways it can be subtly wrong:

  1. NORMALISATION ORDER. `norm_topk_prob` normalises the routed weights across
     the top-k. The shared gate is an independent sigmoid. Appending the slot
     BEFORE normalising renormalises every routed weight against a value that
     is not part of their distribution -- every shape correct, every value
     finite, every token slightly wrong.
  2. THE ALL-TO-ALL. Every token uses the shared expert, so routing it as a
     real id sends the whole batch to one rank. `dispatch_ids` must remove it,
     and the probe checks the dispatch is unchanged by the fusion.
  3. THE LOCAL INDEX. The shared slot must resolve to the same replicated
     local index on EVERY rank, while a routed id resolves only on its owner.

    python3 probes/qwen38_shared_fuse.py

Shapes follow Qwen3.8-Flash-Next scaled down: intermediate == shared
intermediate (the property that makes fusion possible at all), top-k routed
plus one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/qwen38_moe"))

from qwen38_shared_fuse import (                                # noqa: E402
    NO_EXPERT_ID, SHARED_SLOT_ID, check_fused_routing, dispatch_ids,
    fuse_expert_weights, fuse_routing, local_ids, local_shared_index,
)


def expert_mlp(x, w13, w2, limit):
    """One expert: clamped SwiGLU, the shape Qwen ships (swiglu_limit 10.0)."""
    gate_up = x @ w13.T
    gate, up = gate_up.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(min=-limit, max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (F.silu(gate) * up) @ w2.T


def run_moe(x, ids, weights, w13, w2, limit):
    """Dense simulation of a grouped GEMM over whatever slots it is given."""
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=x.dtype)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e == NO_EXPERT_ID:
                continue
            out[t] += weights[t, k] * expert_mlp(x[t:t + 1], w13[e], w2[e],
                                                 limit)[0]
    return out


def main() -> int:
    torch.manual_seed(20260911)
    T, H, I, E, K, W = 12, 32, 16, 8, 3, 4      # tokens, hidden, inter, experts
    limit = 10.0
    ok = True

    x = torch.randn(T, H, dtype=torch.float32) * 0.3
    w13 = torch.randn(E, 2 * I, H) * 0.05
    w2 = torch.randn(E, H, I) * 0.05
    s13 = torch.randn(2 * I, H) * 0.05
    s2 = torch.randn(H, I) * 0.05

    logits = torch.randn(T, E)
    raw_w, ids = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
    ids = ids.to(torch.int32)
    # norm_topk_prob: normalise ACROSS the top-k, before anything is appended
    weights = raw_w / raw_w.sum(dim=-1, keepdim=True)
    gate = torch.sigmoid(torch.randn(T))

    # -- the oracle: routed + gated shared, as two separate computations ----
    routed = run_moe(x, ids, weights, w13, w2, limit)
    shared = expert_mlp(x, s13, s2, limit)
    want = routed + gate.unsqueeze(-1) * shared

    # -- the fused lane ----------------------------------------------------
    fw13, fw2 = fuse_expert_weights(w13, w2, s13, s2)
    fids, fweights = fuse_routing(ids, weights, gate)
    check_fused_routing(fids, num_experts=E, where="probe")
    # the grouped GEMM addresses the shared slot by its local index
    gemm_ids = fids.clone()
    gemm_ids[gemm_ids == SHARED_SLOT_ID] = E
    got = run_moe(x, gemm_ids, fweights, fw13, fw2, limit)

    same = torch.equal(want, got)
    ok &= same
    d = (want - got).abs().max().item()
    print(f"  exact          {'OK' if same else 'MISMATCH'}   max |d| {d:.3e}")

    # -- 1. the normalisation-order control --------------------------------
    raw_plus = torch.cat([raw_w, gate.unsqueeze(-1)], dim=-1)
    bad_weights = raw_plus / raw_plus.sum(dim=-1, keepdim=True)
    bad = run_moe(x, gemm_ids, bad_weights, fw13, fw2, limit)
    moved = not torch.allclose(want, bad, atol=1e-6)
    ok &= moved
    print(f"  order control  appending BEFORE normalising "
          f"{'differs, good' if moved else 'is identical -- FAIL'}"
          f"   max |d| {(want - bad).abs().max().item():.3e}")

    # -- 2. the dispatch must not see the shared slot ----------------------
    disp = dispatch_ids(fids)
    kept = torch.equal(disp[:, :-1], ids)
    gone = bool((disp[:, -1] == NO_EXPERT_ID).all())
    twice = torch.equal(dispatch_ids(disp), disp)
    ok &= kept and gone and twice
    print(f"  dispatch       routed ids unchanged: {kept}; shared slot "
          f"removed: {gone}; idempotent: {twice}")
    routed_hits = int((disp != NO_EXPERT_ID).sum())
    fused_hits = int((fids != NO_EXPERT_ID).sum())
    print(f"                 all-to-all carries {routed_hits} of "
          f"{fused_hits} slots -- the {T} shared ones stay local")

    # -- 3. local indices, on every rank -----------------------------------
    per_rank = E // W
    shared_local = local_shared_index(per_rank)
    seen = []
    for r in range(W):
        li = local_ids(fids, rank=r, num_local_experts=per_rank, num_experts=E)
        seen.append(bool((li[:, -1] == shared_local).all()))
        owned = li[:, :-1] != NO_EXPERT_ID
        expect = (ids >= r * per_rank) & (ids < (r + 1) * per_rank)
        ok &= torch.equal(owned, expect)
    ok &= all(seen)
    print(f"  local index    shared slot resolves to {shared_local} on all "
          f"{W} ranks: {all(seen)}; routed ids resolve only on their owner")

    # -- 4. the contract refuses what a grouped GEMM cannot mean -----------
    def refuses(label, fn):
        nonlocal ok
        try:
            fn()
        except (ValueError, TypeError) as exc:
            print(f"  refuses {label:34s} OK  ({str(exc)[:44]})")
            return
        print(f"  refuses {label:34s} FAIL -- accepted it")
        ok = False

    drifted = fids.clone()
    drifted[0, -1], drifted[0, 0] = drifted[0, 0], SHARED_SLOT_ID
    refuses("a shared slot not in the last column",
            lambda: check_fused_routing(drifted, num_experts=E))
    doubled = torch.cat([fids, fids[:, -1:]], dim=1)
    refuses("two shared slots on a token",
            lambda: check_fused_routing(doubled, num_experts=E))
    oob = fids.clone()
    oob[0, 0] = E + 5
    refuses("a routed id past the expert table",
            lambda: check_fused_routing(oob, num_experts=E))
    refuses("a shared expert of a different shape",
            lambda: fuse_expert_weights(w13, w2, s13[:, :H - 1], s2))

    print("\n" + ("SHARED FUSE PASS" if ok else "SHARED FUSE FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
