"""Run the engine's GLM-5.3 on real weights, one node, no vLLM (profile check).

    python3 engine/profiles/glm53/check.py --layers 0-4 --tokens 512 --chunk 256
    python3 engine/profiles/glm53/check.py --lanes served          # inside the glm53 image

Binds a chain of layers from a dev rank file (preshard.py --world 1) into
one arena, carves the chain's caches from the same arena, and asks the one
question a composition can answer about itself before a served oracle is
put next to it: is a prefill in two chunks the same computation as one?
That crosses every cache the model keeps -- the KDA conv and recurrent
state, the fp8 latent, the kpool pool keys and their scales, the tail ring
-- and the block-table arithmetic around them. It says nothing about the
algebra being GLM's; the served layer judge (next) does that.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch                                                     # noqa: E402

from engine.base.arena import Arena                              # noqa: E402
from engine.base.comm import Comm                                # noqa: E402
from engine.base.instruments import Recorder                     # noqa: E402
from engine.base.loader import RankLoader                        # noqa: E402
from engine.profiles.glm53 import facts, lanes as lane_tables    # noqa: E402
from engine.profiles.glm53.net import BF16, E4M3, F32, Glm53Net  # noqa: E402

DEV_RANK = Path("/home/choiceoh/models/glm53-redhat-nvfp4-dev/rank0of1.safetensors")
GIB = 1 << 30


class ChainCaches:
    """The `Caches` protocol for one sequence in one slot, contiguous: slot ids
    are positions (block table = identity). Carved from the arena."""

    def __init__(self, arena: Arena, F, net: Glm53Net, capacity: int, slots: int = 2):
        self.F = F
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


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank-file", default=str(DEV_RANK))
    ap.add_argument("--layers", default="0-4")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--lanes", choices=["reference", "served"], default="reference")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    F = facts.load()
    layers = parse_layers(a.layers)
    if a.chunk % F.kpool or a.tokens <= a.chunk:
        raise SystemExit(f"--chunk must be a multiple of kpool {F.kpool} and below --tokens")
    rec = Recorder("glm53 check")
    comm = Comm.init(rank=0, world=1)
    lanes = lane_tables.reference() if a.lanes == "reference" else lane_tables.served()
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    from engine.base.params import total_bytes
    weights = total_bytes(specs)
    cap = -(-a.tokens // F.block) * F.block
    with rec.phase("arena"):
        arena = Arena(weights + 256 * len(specs) + 64 * GIB // 1024 + cap * 4096)      # weights + carve alignment + caches
    with rec.phase("load"):
        loader = RankLoader(a.rank_file)
        views = loader.load([s.name for s in specs], arena=arena, recorder=rec)
        net.bind(views)
    caches = ChainCaches(arena, F, net, cap)
    rec.gauge("arena_GiB", round(arena.used / GIB, 3))
    torch.manual_seed(a.seed)
    ids = torch.randint(0, 100_000, (a.tokens,), device="cuda")
    blocks = {}
    net.probe = lambda name, L, out: blocks.setdefault((name, L), []).append(out)
    torch.cuda.synchronize()
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
    finite = bool(torch.isfinite(h_whole).all() and torch.isfinite(logits).all())
    top = logits[0].float().topk(5)
    print(rec.table())
    print(f"  chain {layers[0]}-{layers[-1]} ({len(layers)} layers: {sum(F.is_dsa(L) for L in layers)} dsa, "
          f"{sum(not F.is_dsa(L) for L in layers)} kda; {sum(F.is_moe(L) for L in layers)} moe), lanes={lanes.name}, "
          f"{a.tokens} tokens, arena {arena.used / GIB:.2f} GiB ({len(arena.regions)} regions)")
    print(f"  hidden |h| mean {h_whole.float().abs().mean().item():.4f} max {h_whole.float().abs().max().item():.3f}, "
          f"finite {finite}; last-token top-5 ids {top.indices.tolist()} logits {[round(v, 2) for v in top.values.tolist()]}")

    # The judgement. bf16 GEMMs round differently at M=512 and M=256, and the
    # MoE router turns that into a different top-8 on a few tokens (measured:
    # 30/256 at layer 3) -- so a chunked prefill cannot equal a whole one
    # token for token past an MoE, in any engine. What the caches must not do
    # is ADD to that: every attention block's output on the second chunk (the
    # only place the KDA state, the latent, the pools and the tail are read)
    # is held to the same noise floor as the first chunk, whose caches were
    # empty either way.
    def tok_rel(x, y):
        return (x.float() - y.float()).abs().amax(-1) / y.float().abs().amax(-1).clamp_min(1e-6)
    rows, worst_first, worst_second, ok = [], 0.0, 0.0, finite
    for (name, L), outs in blocks.items():
        first, second = tok_rel(outs[0], whole[(name, L)][: a.chunk]), tok_rel(outs[1], whole[(name, L)][a.chunk:])
        f50, s50 = first.median().item(), second.median().item()
        rows.append(f"    L{L:<3}{name:<6} first chunk p50 {f50:.1e} max {first.max().item():.1e} | second chunk (via caches) p50 {s50:.1e} max {second.max().item():.1e}")
        if name in ("kda", "dsa"):
            worst_first, worst_second = max(worst_first, first.max().item()), max(worst_second, second.max().item())
            # the caches may not add to what the block already inherits from upstream (an MoE's flips included)
            ok = ok and second.max().item() <= max(1.5 * first.max().item(), 3e-2) and s50 <= max(1.5 * f50, 3e-3)
    print("  per block, chunked vs whole (per-token rel):")
    print("\n".join(rows))
    print(f"  attention blocks, second chunk through every cache: worst {worst_second:.2e} (first-chunk noise floor {worst_first:.2e})")
    r_tail = tok_rel(h_b, h_whole[a.chunk:])
    print(f"  final hidden, second chunk: p50 {r_tail.median().item():.1e}, {(r_tail > 5e-2).sum().item()} of {a.tokens - a.chunk} tokens over 5e-2 (router flips)")
    print("\n  " + ("PASS: the caches add nothing a whole prefill does not have" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
