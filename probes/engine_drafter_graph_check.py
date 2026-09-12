"""Real DFlash2 weights: proposal and accepted-prefix graph equivalence.

Target embedding/head use rank 0 arithmetic in isolation. This checks the
drafter implementation and cache ownership, not full-model acceptance.
"""
import argparse
import json

import torch

from engine.base.instruments import Recorder
from engine.profiles.glm53 import drafter as drafter_mod, facts
from engine.profiles.glm53.boot import build
from engine.profiles.glm53.lanes import served
from probes.engine_decode_graph_check import IsolatedRank


def unpadded_attention(drafter, layer, x, positions, ring, context):
    """Independent FP64 SDPA oracle over only the live context positions."""
    import torch.nn.functional as fn
    from engine.profiles.glm53.drafter import rmsnorm, rope
    F, p = drafter.F, drafter.p
    prefix = f"layers.{layer}.self_attn."
    def project(kind, heads, normalized):
        value = fn.linear(x, p[prefix+kind+"_proj.weight"]).view(-1, heads, F.head_dim)
        if normalized:
            value = rope(rmsnorm(value, p[prefix+kind+"_norm.weight"], F.rms_eps), positions, F.rope_theta)
        return value
    q = project("q", F.heads, True)
    k = project("k", F.kv_heads, True)
    v = project("v", F.kv_heads, False)
    live = torch.arange(max(0, context-F.window), context, device=x.device) % F.window
    k = torch.cat((ring[layer, 0, live], k)).repeat_interleave(F.heads//F.kv_heads, 1)
    v = torch.cat((ring[layer, 1, live], v)).repeat_interleave(F.heads//F.kv_heads, 1)
    out = fn.scaled_dot_product_attention(q.transpose(0, 1).double(),
                                          k.transpose(0, 1).double(), v.transpose(0, 1).double())
    return fn.linear(out.transpose(0, 1).reshape(x.shape[0], -1).to(x.dtype), p[prefix+"o_proj.weight"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Defaults that work on a node: the fleet queue invokes an admitted ST check with NO arguments
    # (bench/fleet_onepass.ST_FLAGS admits a handful and --drafter-dir and --tier-dir are not among
    # them), so a required argument here is a check the queue can start and never run (45차 §95).
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--ckpt-meta", default=str(facts.CKPT))
    ap.add_argument("--drafter-dir", default=str(drafter_mod.DRAFTER))
    args = ap.parse_args()
    torch.manual_seed(19)
    _, _, caches, engine, _ = build(IsolatedRank(), [0, 3], served(), args.ranks, .25, 2,
                                    True, Recorder("draft-graph"), ckpt_meta=args.ckpt_meta,
                                    drafter_dir=args.drafter_dir)
    drafter = engine.drafter
    drafter.capture_decode(caches)
    slots = [caches.slots.take(i) for i in range(2)]
    rows = []
    for position in (0, 1, 5, 17, 2047, 2048, 2057):
        slot = slots[position % 2]
        caches.reset_slot(slot)
        ring = caches.draft_ring(slot)
        context = torch.arange(max(0, position-drafter.F.window), position, device="cuda")
        aux = torch.randn(context.numel(), 20480, device="cuda", dtype=torch.bfloat16) * .02
        drafter.observe(ring, context, aux)
        attention_error = 0.
        x = torch.randn(6, drafter.F.hidden, device="cuda", dtype=torch.bfloat16)*.02
        positions = position + torch.arange(6, device="cuda")
        for layer in range(drafter.F.layers):
            oracle = unpadded_attention(drafter, layer, x, positions, ring, position)
            padded = drafter._attn(layer, x, positions, ring, position)
            error = ((oracle.float()-padded.float()).abs().max()/oracle.float().abs().max()).item()
            assert torch.isfinite(padded).all() and error <= .01, (position, layer, error)
            attention_error = max(attention_error, error)
        anchor = torch.full((1,), 1234, device="cuda", dtype=torch.int64)
        expected = drafter.propose_tensor(anchor, position, ring).clone()
        actual = drafter.decode_graphs.propose(1234, position, ring).clone()
        same = torch.equal(expected, actual)
        row = dict(context=position, slot=slot, proposal_equal=same,
                   unpadded_fp64_attention_relative=attention_error,
                   eager=expected.tolist(), graph=actual.tolist())
        print(json.dumps(row), flush=True)
        assert same, row
        rows.append(row)
        saved = ring.clone()
        for accepted in range(1, drafter.k+2):
            positions = position + torch.arange(accepted, device="cuda")
            new_aux = torch.randn(accepted, 20480, device="cuda", dtype=torch.bfloat16) * .02
            ring.copy_(saved)
            drafter.observe(ring, positions, new_aux)
            expected_ring = ring.clone()
            ring.copy_(saved)
            drafter.observe_decode(ring, positions, new_aux)
            assert torch.equal(ring, expected_ring), (position, accepted)
    drafter.decode_graphs.proposals.close()
    drafter.decode_graphs.observations.close()
    print(json.dumps(dict(passed=True, proposals=len(rows), accepted_prefixes=len(rows)*6)), flush=True)


if __name__ == "__main__":
    main()
