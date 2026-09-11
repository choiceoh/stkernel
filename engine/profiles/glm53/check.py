"""Run GLM-5.3 on real weights at TP=4 with paged caches and the common runner.

    python3 engine/profiles/glm53/check.py --layers 0-4 --tokens 512 --chunk 256
    python3 engine/profiles/glm53/check.py --lanes served          # inside the glm53 image
    python3 engine/profiles/glm53/check.py --distributed           # one process per fleet node

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
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch                                                     # noqa: E402

from engine.base.arena import Arena                              # noqa: E402
from engine.base.comm import Comm, LocalTP                       # noqa: E402
from engine.base.instruments import Recorder                     # noqa: E402
from engine.base.loader import RankLoader                        # noqa: E402
from engine.base.params import total_bytes                       # noqa: E402
from engine.base.scheduler import Contract                       # noqa: E402
from engine.profiles.glm53 import facts, lanes as lane_tables    # noqa: E402
from engine.profiles.glm53.caches import Glm53Caches, layout       # noqa: E402
from engine.profiles.glm53.net import BF16, E4M3, F32, Glm53Net, Step  # noqa: E402
from engine.profiles.glm53.runtime import Glm53Runtime            # noqa: E402

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
                width = kp - 1 + F.spec_k
                self._tail[L] = arena.carve(slots * width * 2 * F.idx_dim * 2, f"L{L} tail").view(BF16).view(slots, width, 2, F.idx_dim)
            else:
                c, wc, wr = 3 * net.Hk * F.kda_dim, net.conv_ring, net.rec_ring
                conv = arena.carve(slots * c * wc * 2, f"L{L} conv ring").view(BF16).view(slots, c, wc)
                rec = arena.carve(slots * wr * net.Hk * F.kda_dim * F.kda_dim * 4, f"L{L} recurrent ring").view(F32).view(slots, wr, net.Hk, F.kda_dim, F.kda_dim)
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
    def token_slots(self, layer, seq, positions): return positions.to(torch.int32)
    def pool_slots(self, layer, seq, pool_ids): return pool_ids.to(torch.int32)


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


class PairedMoe:
    """Isolate cache equality from the served MoE's unordered BF16 scatter.

    Both passes execute the real expert lane. On the paired pass every
    activation, selected expert and routing weight must match exactly.
    Forward the first expert output to subsequent layers so atomic-sum
    rounding cannot hide a cache mismatch behind a numerical tolerance.
    Other checks, including generation, use the unmodified expert lane.
    """
    def __init__(self, fn):
        self.fn, self.records, self.replaying, self.position = fn, [], False, 0
        self.max_rel = 0.0

    def __call__(self, *args):
        if not self.replaying:
            out = self.fn(*args)
            self.records.append((tuple(t.clone() for t in args[:3]),
                                 tuple(t.data_ptr() for t in args[3:7]), args[7], out.clone()))
            return out
        inputs, weights, limit, expected = self.records[self.position]
        if not all(torch.equal(a, b) for a, b in zip(args[:3], inputs)):
            raise AssertionError("paged cache changed MoE activations, selected experts or routing weights")
        if weights != tuple(t.data_ptr() for t in args[3:7]) or limit != args[7]:
            raise AssertionError("paired cache check changed MoE weights or activation limit")
        out = self.fn(*args)
        if not torch.isfinite(out).all():
            raise AssertionError("served MoE returned nonfinite output")
        self.max_rel = max(self.max_rel, tok_rel(out, expected).max().item())
        self.position += 1
        return expected


def rank_main(comm, a, F, layers, lanes, ids, garbage):
    """One rank's whole check: bind, then four runs -- whole prefill; two
    chunks; prefill + one verify step; prefill + a verify step whose last four
    drafts are garbage, 'accept 2', and the next verify step on the truth."""
    rec = Recorder(f"rank{comm.rank}")
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    cap = -(-a.tokens // F.block) * F.block
    num_blocks = cap // F.block + 2
    cache_bytes = layout(F, layers).nbytes(num_blocks, 2)
    with rec.phase("arena"):
        arena = Arena(total_bytes(specs) + 256 * len(specs) + len(layers) * (64 << 20) + cap * 8192 + cache_bytes)
    with rec.phase("load"):
        views = RankLoader(Path(a.ranks) / f"rank{comm.rank}of{facts.TP}.safetensors").load(
            [s.name for s in specs], arena=arena, recorder=rec, max_run=128 << 20)
        net.bind(views)
    chain = ChainCaches(arena, F, net, cap)
    caches = Glm53Caches(arena, F, layers, num_blocks, 2)
    caches.pool.reserve(1, F.block)                    # force the tested sequence off physical block 0
    caches.pool.reserve(0, cap)
    caches.slots.take(0)
    blocks = {}
    net.probe = lambda name, L, out: blocks.setdefault((name, L), []).append(out)
    T, K1 = a.tokens, F.spec_k + 1
    runs = {}

    def run(name, steps, cache=caches):
        blocks.clear(); cache.reset()
        with rec.phase(name):
            hs = []
            for st in steps:
                if cache is caches:
                    cache.prepare(st)
                hs.append(net.forward(st, cache))
            torch.cuda.synchronize()
        runs[name] = {"h": hs, "blocks": {k: list(v) for k, v in blocks.items()}}

    paired = PairedMoe(lanes.moe)
    net.lanes = replace(lanes, moe=paired)
    run("whole", [Step.prefill(ids, 0, 0, 1)], cache=chain)
    logits = net.head(runs["whole"]["h"][0][-1:])
    paired.replaying = True
    run("paged", [Step.prefill(ids, 0, 0, 1)])
    assert paired.position == len(paired.records), "paged pass skipped an expert layer"
    net.lanes = lanes
    paired.records.clear()
    run("chunked", [Step.prefill(ids[: a.chunk], 0, 0, 1), Step.prefill(ids[a.chunk:], a.chunk, 0, 1)])
    run("verify", [Step.prefill(ids[: T - K1], 0, 0, 1), Step.decode([(ids[T - K1:], T - K1, 0, 1)])])
    c0 = T - 2 * K1                                                        # prefill to c0, verify 2 true + 4 garbage, accept 2, verify the truth
    drafts = torch.cat([ids[c0: c0 + 2], garbage])
    run("rollback", [Step.prefill(ids[: c0], 0, 0, 1), Step.decode([(drafts, c0, 0, 1)]), Step.decode([(ids[c0 + 2: c0 + 2 + K1], c0 + 2, 0, 1)])])
    # Bind the same real model to the common scheduler/runner and finish two
    # requests with unequal limits. All four ranks execute the same schedule.
    net.probe = None
    caches.pool.release(0); caches.pool.release(1); caches.slots.give(1)
    runtime = Glm53Runtime(net, caches, Contract(F.kpool, a.chunk, 0, 0., 2))
    runtime.submit(0, ids[:a.chunk], 3, now=0.)
    runtime.submit(1, ids[-a.chunk:], 1, now=0.)
    with rec.phase("runtime"):
        for tick in range(16):
            if runtime.step(now=tick + 1.) is None:
                break
        else:
            raise AssertionError("runtime did not finish the bounded requests")
    generated = [runtime.take_result(s) for s in (0, 1)]
    assert [len(x) for x in generated] == [3, 1]
    assert caches.pool.available == num_blocks and caches.slots.available == 2
    return {"rec": rec, "arena": arena, "logits": logits, "runs": runs, "generated": generated,
            "moe_repeat_rel": paired.max_rel}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--layers", default="0-4")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--lanes", choices=["reference", "served"], default="reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distributed", action="store_true", help="one rank per GB10; RANK/WORLD_SIZE/MASTER_* from the launcher")
    a = ap.parse_args(argv)
    F = facts.load()
    print(f"  box: {facts.check_box()}")
    layers = parse_layers(a.layers)
    if not layers or layers[0] < 0 or layers[-1] >= F.layers:
        raise SystemExit(f"--layers must select layers inside 0..{F.layers - 1}")
    if a.chunk <= 0 or a.chunk % F.kpool or a.tokens <= max(a.chunk, 2 * (F.spec_k + 1)):
        raise SystemExit(f"--chunk must be a multiple of kpool {F.kpool} and below --tokens")
    lanes = lane_tables.reference() if a.lanes == "reference" else lane_tables.served()
    torch.manual_seed(a.seed)
    ids = torch.randint(0, 100_000, (a.tokens,), device="cuda")
    garbage = torch.randint(0, 100_000, (F.spec_k + 1 - 2,), device="cuda")
    comm = Comm.init(world=facts.TP) if a.distributed else None
    try:
        return judge(a, F, layers, lanes, ids, garbage, comm)
    finally:
        if comm is not None:
            comm.close()


def judge(a, F, layers, lanes, ids, garbage, comm):
    t0 = time.perf_counter()
    tp = None if comm is not None else LocalTP(facts.TP)
    lane_tables.bind_tp(tp)
    outs = ([rank_main(comm, a, F, layers, lanes, ids, garbage)] if comm is not None
            else tp.run(rank_main, a, F, layers, lanes, ids, garbage))
    wall = time.perf_counter() - t0
    r0 = outs[0]
    print(r0["rec"].table())
    print(f"  chain {layers[0]}-{layers[-1]} ({len(layers)} layers: {sum(F.is_dsa(L) for L in layers)} dsa, "
          f"{sum(not F.is_dsa(L) for L in layers)} kda; {sum(F.is_moe(L) for L in layers)} moe), TP={facts.TP} {'fleet' if comm else 'in-process'}, lanes={lanes.name}, "
          f"{a.tokens} tokens, arena {r0['arena'].used / GIB:.2f} GiB per rank x {facts.TP}, wall {wall:.1f} s")
    whole = r0["runs"]["whole"]
    h_whole = whole["h"][0]
    finite = all(bool(torch.isfinite(o["runs"]["whole"]["h"][0]).all() and torch.isfinite(o["logits"]).all()) for o in outs)
    top = r0["logits"][0].float().topk(5)
    print(f"  hidden |h| mean {h_whole.float().abs().mean().item():.4f} max {h_whole.float().abs().max().item():.3f}, "
          f"finite {finite}; last-token top-5 ids {top.indices.tolist()} logits {[round(v, 2) for v in top.values.tolist()]}")
    # ranks agree: replicated activations must be identical on every rank
    peer_h = ([o["runs"]["whole"]["h"][0] for o in outs] if comm is None
              else list(comm.all_gather(h_whole, dim=0).chunk(facts.TP, dim=0)))
    peer_logits = ([o["logits"] for o in outs] if comm is None
                   else list(comm.all_gather(r0["logits"], dim=0).chunk(facts.TP, dim=0)))
    agree = all(torch.equal(h, h_whole) and torch.equal(l, r0["logits"]) for h, l in zip(peer_h, peer_logits))
    max_dev = max((h.float() - h_whole.float()).abs().max().item() for h in peer_h)
    print(f"  ranks agree (hidden and logits identical on all {facts.TP}): {agree} (max |diff| {max_dev:.1e})")
    paged_exact = all(torch.equal(o["runs"]["whole"]["h"][0], o["runs"]["paged"]["h"][0]) for o in outs)
    generated_agree = all(o["generated"] == r0["generated"] for o in outs)
    if comm is not None:
        generated = torch.tensor([t for seq in r0["generated"] for t in seq], device="cuda", dtype=torch.int64)
        generated_agree = all(torch.equal(row, generated) for row in comm.all_gather(generated, dim=0).chunk(facts.TP))
    print(f"  paged KV == contiguous oracle (exact MoE inputs, first expert outputs forwarded): {paged_exact}; "
          f"runner generated {list(map(len, r0['generated']))} tokens, ranks agree: {generated_agree}, all resources returned")
    print(f"  actual expert lane executed in both paired passes; max relative repeat difference: {r0['moe_repeat_rel']:.3e}")
    # every other run against the whole prefill, per block, per token, against the first chunk's own noise floor
    T, K1 = a.tokens, F.spec_k + 1
    windows = {"chunked": [(0, a.chunk), (a.chunk, T)],
               "verify": [(0, T - K1), (T - K1, T)],
               "rollback": [(0, T - 2 * K1), None, (T - 2 * K1 + 2, T - K1 + 2)]}       # None: the garbage step is not comparable
    floor = {}
    ok = finite and agree and paged_exact and generated_agree
    rows = []
    for run_name, wins in windows.items():
        for (name, L), outs_b in r0["runs"][run_name]["blocks"].items():
            ref = whole["blocks"][(name, L)][0]
            for i, win in enumerate(wins):
                if win is None:
                    continue
                r = tok_rel(outs_b[i], ref[win[0]: win[1]])
                p50, mx = r.median().item(), r.max().item()
                if run_name == "chunked" and i == 0:
                    floor[(name, L)] = (p50, mx)                                   # the first chunk: no cache was read
                    continue
                f50, fmx = floor[(name, L)]
                judged = name in ("kda", "dsa")
                passed = (mx <= max(1.5 * fmx, 3e-2) and p50 <= max(1.5 * f50, 3e-3)) if judged else True
                ok = ok and passed
                rows.append(f"    {run_name:<9} step {i}  L{L:<3}{name:<6} p50 {p50:.1e} max {mx:.1e}  (floor p50 {f50:.1e} max {fmx:.1e}){'' if passed else '  <-- FAIL'}")
    print("  per block vs the whole prefill (per-token rel; floor = first chunk, no cache read):")
    print("\n".join(rows))
    if comm is not None:
        ok = comm.all_reduce(torch.tensor([int(ok)], device="cuda", dtype=torch.int32)).item() == facts.TP
    print("\n  " + ("PASS: four ranks agree; paged caches and runtime; chunked, verify and rollback checks" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
