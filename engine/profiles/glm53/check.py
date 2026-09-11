"""Run the engine's GLM-5.3 on real weights at TP=4, one box, no vLLM (profile check).

    python3 engine/profiles/glm53/check.py --layers 0-4 --tokens 512 --chunk 256
    python3 engine/profiles/glm53/check.py --lanes served          # inside the glm53 image

The four ranks are four threads on this GB10 (base/comm.LocalTP), each
binding ITS slice of the layers from its rank file (facts.RANKS, the
preshard's output) into its own arena, so the row/column splits and every
all-reduce run exactly as on the fleet -- a wrong split shows up here, not
on four nodes. Two things are judged:

  ranks agree     after every all-reduce the activations are replicated, so
                  the four final hidden states must be byte-identical
  two chunks ==   a prefill in two chunks against one prefill, per block,
  one prefill     through every cache the model keeps (KDA conv/recurrent
                  state, fp8 latent, kpool keys+scales, the tail ring) --
                  held to the noise floor the first chunk already shows,
                  because bf16 GEMMs round differently at M=512 and M=256
                  and the MoE router turns that into a different top-8 on a
                  few tokens (measured 30/256), in any engine.

It says nothing about the algebra being GLM's; the served layer judge does.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch                                                     # noqa: E402

from engine.base.arena import Arena                              # noqa: E402
from engine.base.comm import LocalTP                             # noqa: E402
from engine.base.instruments import Recorder                     # noqa: E402
from engine.base.loader import RankLoader                        # noqa: E402
from engine.base.params import total_bytes                       # noqa: E402
from engine.profiles.glm53 import facts, lanes as lane_tables    # noqa: E402
from engine.profiles.glm53.net import BF16, E4M3, F32, Glm53Net  # noqa: E402

GIB = 1 << 30


class ChainCaches:
    """The `Caches` protocol for one sequence in one slot, contiguous: slot ids
    are positions (block table = identity). Carved from this rank's arena."""

    def __init__(self, arena: Arena, F, net: Glm53Net, capacity: int, slots: int = 2):
        kp = F.kpool
        self._kda, self._lat, self._pk, self._ps, self._tail = {}, {}, {}, {}, {}
        for L in net.layers:
            if F.is_dsa(L):
                self._lat[L] = arena.carve(capacity * F.kv_lora, f"L{L} latent").view(E4M3).view(capacity, F.kv_lora)
                self._pk[L] = arena.carve(capacity // kp * F.idx_dim, f"L{L} pool keys").view(E4M3).view(capacity // kp, F.idx_dim)
                self._ps[L] = arena.carve(capacity // kp * 4, f"L{L} pool scales").view(F32)
                self._tail[L] = arena.carve(slots * kp * 2 * F.idx_dim * 2, f"L{L} tail").view(BF16).view(slots, kp, 2, F.idx_dim)
            else:
                c = 3 * net.Hk * F.kda_dim
                conv = arena.carve(slots * c * (F.conv - 1) * 2, f"L{L} conv state").view(BF16).view(slots, c, F.conv - 1)
                rec = arena.carve(slots * net.Hk * F.kda_dim * F.kda_dim * 4, f"L{L} recurrent state").view(F32).view(slots, net.Hk, F.kda_dim, F.kda_dim)
                self._kda[L] = (conv, rec)

    def reset(self):
        for conv, rec in self._kda.values():
            conv.zero_(); rec.zero_()
        for t in list(self._lat.values()) + list(self._pk.values()) + list(self._tail.values()):
            t.view(torch.uint8).zero_()
        for t in self._ps.values():
            t.zero_()

    def kda(self, layer, slot): conv, rec = self._kda[layer]; return conv[slot], rec[slot]
    def latent(self, layer): return self._lat[layer]
    def pool_keys(self, layer): return self._pk[layer]
    def pool_scales(self, layer): return self._ps[layer]
    def tail(self, layer, slot): return self._tail[layer][slot]
    def token_slots(self, seq, positions): return positions.to(torch.int32)
    def pool_slots(self, seq, pool_ids): return pool_ids.to(torch.int32)


def parse_layers(spec: str):
    out = []
    for piece in spec.split(","):
        if "-" in piece:
            lo, hi = piece.split("-"); out += list(range(int(lo), int(hi) + 1))
        elif piece:
            out.append(int(piece))
    return sorted(set(out))


def tok_rel(x, y):
    return (x.float() - y.float()).abs().amax(-1) / y.float().abs().amax(-1).clamp_min(1e-6)


def rank_main(comm, a, F, layers, lanes, ids):
    """One rank's whole check: bind, prefill whole, prefill in two chunks."""
    rec = Recorder(f"rank{comm.rank}")
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    cap = -(-a.tokens // F.block) * F.block
    with rec.phase("arena"):
        arena = Arena(total_bytes(specs) + 256 * len(specs) + len(layers) * (64 << 20) + cap * 8192)   # weights + alignment + states + paged
    with rec.phase("load"):
        views = RankLoader(Path(a.ranks) / f"rank{comm.rank}of{facts.TP}.safetensors").load([s.name for s in specs], arena=arena, recorder=rec)
        net.bind(views)
    caches = ChainCaches(arena, F, net, cap)
    blocks = {}
    net.probe = lambda name, L, out: blocks.setdefault((name, L), []).append(out)
    with rec.phase("prefill whole"):
        h_whole = net.prefill(ids, 0, seq=0, slot=1, caches=caches)
        torch.cuda.synchronize()
    logits = net.head(h_whole[-1:])
    whole = {k: v[0] for k, v in blocks.items()}
    blocks.clear(); caches.reset()
    with rec.phase("prefill chunked"):
        h_a = net.prefill(ids[: a.chunk], 0, seq=0, slot=1, caches=caches)
        h_b = net.prefill(ids[a.chunk:], a.chunk, seq=0, slot=1, caches=caches)
        torch.cuda.synchronize()
    return {"rec": rec, "arena": arena, "h_whole": h_whole, "logits": logits, "whole": whole,
            "chunked": {k: v for k, v in blocks.items()}, "h_a": h_a, "h_b": h_b}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--layers", default="0-4")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--lanes", choices=["reference", "served"], default="reference")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    F = facts.load()
    print(f"  box: {facts.check_box()}")
    layers = parse_layers(a.layers)
    if a.chunk % F.kpool or a.tokens <= a.chunk:
        raise SystemExit(f"--chunk must be a multiple of kpool {F.kpool} and below --tokens")
    lanes = lane_tables.reference() if a.lanes == "reference" else lane_tables.served(expert_lane="reference")
    if a.lanes == "served":
        print("  NOTE: served lanes with the REFERENCE expert lane (b12x is not bound yet) -- a lane judge, not a boot")
    torch.manual_seed(a.seed)
    ids = torch.randint(0, 100_000, (a.tokens,), device="cuda")
    t0 = time.perf_counter()
    outs = LocalTP(facts.TP).run(rank_main, a, F, layers, lanes, ids)
    wall = time.perf_counter() - t0
    r0 = outs[0]
    print(r0["rec"].table())
    print(f"  chain {layers[0]}-{layers[-1]} ({len(layers)} layers: {sum(F.is_dsa(L) for L in layers)} dsa, "
          f"{sum(not F.is_dsa(L) for L in layers)} kda; {sum(F.is_moe(L) for L in layers)} moe), TP={facts.TP} in-process, lanes={lanes.name}, "
          f"{a.tokens} tokens, arena {r0['arena'].used / GIB:.2f} GiB per rank x {facts.TP}, wall {wall:.1f} s")
    finite = all(bool(torch.isfinite(o["h_whole"]).all() and torch.isfinite(o["logits"]).all()) for o in outs)
    top = r0["logits"][0].float().topk(5)
    print(f"  hidden |h| mean {r0['h_whole'].float().abs().mean().item():.4f} max {r0['h_whole'].float().abs().max().item():.3f}, "
          f"finite {finite}; last-token top-5 ids {top.indices.tolist()} logits {[round(v, 2) for v in top.values.tolist()]}")
    # ranks agree: replicated activations must be identical on every rank
    agree = all(torch.equal(o["h_whole"], r0["h_whole"]) and torch.equal(o["logits"], r0["logits"]) for o in outs[1:])
    max_dev = max((o["h_whole"].float() - r0["h_whole"].float()).abs().max().item() for o in outs[1:])
    print(f"  ranks agree (hidden and logits identical on all {facts.TP}): {agree} (max |diff| {max_dev:.1e})")
    # two chunks == one prefill, per block, against the first chunk's own noise floor
    rows, worst_first, worst_second, ok = [], 0.0, 0.0, finite and agree
    for (name, L), pair in r0["chunked"].items():
        first, second = tok_rel(pair[0], r0["whole"][(name, L)][: a.chunk]), tok_rel(pair[1], r0["whole"][(name, L)][a.chunk:])
        f50, s50 = first.median().item(), second.median().item()
        rows.append(f"    L{L:<3}{name:<6} first chunk p50 {f50:.1e} max {first.max().item():.1e} | second chunk (via caches) p50 {s50:.1e} max {second.max().item():.1e}")
        if name in ("kda", "dsa"):
            worst_first, worst_second = max(worst_first, first.max().item()), max(worst_second, second.max().item())
            ok = ok and second.max().item() <= max(1.5 * first.max().item(), 3e-2) and s50 <= max(1.5 * f50, 3e-3)
    print("  per block, chunked vs whole (per-token rel):")
    print("\n".join(rows))
    print(f"  attention blocks, second chunk through every cache: worst {worst_second:.2e} (first-chunk noise floor {worst_first:.2e})")
    r_tail = tok_rel(r0["h_b"], r0["h_whole"][a.chunk:])
    print(f"  final hidden, second chunk: p50 {r_tail.median().item():.1e}, {(r_tail > 5e-2).sum().item()} of {a.tokens - a.chunk} tokens over 5e-2 (router flips)")
    print("\n  " + ("PASS: four ranks agree, and the caches add nothing a whole prefill does not have" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
