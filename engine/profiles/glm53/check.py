"""Run the engine's GLM-5.3 on real weights at TP=4, one box, no vLLM (profile check).

    python3 engine/profiles/glm53/check.py --layers 0-4 --tokens 512 --chunk 256
    python3 engine/profiles/glm53/check.py --lanes served          # inside the glm53 image

The four ranks are four threads on this GB10 (base/comm.LocalTP), each
binding ITS slice of the layers from its rank file (facts.RANKS, the
preshard's output) into its own arena, so the row/column splits and every
all-reduce run exactly as on the fleet -- a wrong split shows up here, not
on four nodes. The caches are the profile's real ones (caches.py: block and
slot pools carved from the same arena, block-table addressing). Judged:

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
from engine.profiles.glm53.caches import Glm53Caches, block_bytes, slot_bytes   # noqa: E402
from engine.profiles.glm53.net import Glm53Net, Step             # noqa: E402

GIB = 1 << 30


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


def rank_main(comm, a, F, layers, lanes, ids, garbage):
    """One rank's whole check: bind, then four runs -- whole prefill; two
    chunks; prefill + one verify step; prefill + a verify step whose last four
    drafts are garbage, 'accept 2', and the next verify step on the truth."""
    rec = Recorder(f"rank{comm.rank}")
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    nb, ns = -(-a.tokens // F.block) + 1, 2                                          # blocks for the prompt (+1 spare), null slot + one
    with rec.phase("arena"):
        arena = Arena(total_bytes(specs) + 256 * (len(specs) + 64) + nb * block_bytes(F, layers) + ns * slot_bytes(F, layers, net.Hk))
    with rec.phase("load"):
        views = RankLoader(Path(a.ranks) / f"rank{comm.rank}of{facts.TP}.safetensors").load([s.name for s in specs], arena=arena, recorder=rec)
        net.bind(views)
    caches = Glm53Caches(arena, F, layers, net.Hk, nb, ns, max_seqs=1)
    blocks = {}
    net.probe = lambda name, L, out: blocks.setdefault((name, L), []).append(out)
    T, K1 = a.tokens, F.spec_k + 1
    runs = {}

    def run(name, steps):
        blocks.clear()
        if caches.blocks.tokens[0]:
            caches.release(0)
        slot = caches.slots.take(0); caches.clear_slot(slot)
        caches.reserve(0, T + F.spec_k)                                                # the whole prompt plus a step's drafts, up front
        with rec.phase(name):
            hs = [net.forward(st, caches) for st in steps]
            torch.cuda.synchronize()
        caches.slots.give(slot)
        runs[name] = {"h": hs, "blocks": {k: list(v) for k, v in blocks.items()}}

    run("whole", [Step.prefill(ids, 0, 0, 1)])
    logits = net.head(runs["whole"]["h"][0][-1:])
    if lanes.name.startswith("served"):                                    # the lane judge: the same composition, reference kernels
        ref_net = Glm53Net(F, comm, lane_tables.reference(), layers); ref_net.p = net.p
        ref_net.probe = net.probe
        blocks.clear()
        if caches.blocks.tokens[0]:
            caches.release(0)
        slot = caches.slots.take(0); caches.clear_slot(slot); caches.reserve(0, T + F.spec_k)
        with rec.phase("whole (reference lanes)"):
            h_ref = ref_net.forward(Step.prefill(ids, 0, 0, 1), caches)
            torch.cuda.synchronize()
        caches.slots.give(slot)
        runs["whole_ref"] = {"h": [h_ref], "blocks": {k: list(v) for k, v in blocks.items()}}
    run("chunked", [Step.prefill(ids[: a.chunk], 0, 0, 1), Step.prefill(ids[a.chunk:], a.chunk, 0, 1)])
    run("verify", [Step.prefill(ids[: T - K1], 0, 0, 1), Step.decode([(ids[T - K1:], T - K1, 0, 1)])])
    c0 = T - 2 * K1                                                        # prefill to c0, verify 2 true + 4 garbage, accept 2, verify the truth
    drafts = torch.cat([ids[c0: c0 + 2], garbage])
    run("rollback", [Step.prefill(ids[: c0], 0, 0, 1), Step.decode([(drafts, c0, 0, 1)]), Step.decode([(ids[c0 + 2: c0 + 2 + K1], c0 + 2, 0, 1)])])
    return {"rec": rec, "arena": arena, "logits": logits, "runs": runs}


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
    lanes = lane_tables.reference() if a.lanes == "reference" else lane_tables.served()
    tp = LocalTP(facts.TP)
    lane_tables.bind_tp(tp)
    torch.manual_seed(a.seed)
    ids = torch.randint(0, 100_000, (a.tokens,), device="cuda")
    garbage = torch.randint(0, 100_000, (F.spec_k + 1 - 2,), device="cuda")
    t0 = time.perf_counter()
    outs = tp.run(rank_main, a, F, layers, lanes, ids, garbage)
    wall = time.perf_counter() - t0
    r0 = outs[0]
    print(r0["rec"].table())
    print(f"  chain {layers[0]}-{layers[-1]} ({len(layers)} layers: {sum(F.is_dsa(L) for L in layers)} dsa, "
          f"{sum(not F.is_dsa(L) for L in layers)} kda; {sum(F.is_moe(L) for L in layers)} moe), TP={facts.TP} in-process, lanes={lanes.name}, "
          f"{a.tokens} tokens, arena {r0['arena'].used / GIB:.2f} GiB per rank x {facts.TP}, wall {wall:.1f} s")
    whole = r0["runs"]["whole"]
    h_whole = whole["h"][0]
    finite = all(bool(torch.isfinite(o["runs"]["whole"]["h"][0]).all() and torch.isfinite(o["logits"]).all()) for o in outs)
    top = r0["logits"][0].float().topk(5)
    print(f"  hidden |h| mean {h_whole.float().abs().mean().item():.4f} max {h_whole.float().abs().max().item():.3f}, "
          f"finite {finite}; last-token top-5 ids {top.indices.tolist()} logits {[round(v, 2) for v in top.values.tolist()]}")
    # ranks agree: replicated activations must be identical on every rank
    agree = all(torch.equal(o["runs"]["whole"]["h"][0], h_whole) and torch.equal(o["logits"], r0["logits"]) for o in outs[1:])
    max_dev = max((o["runs"]["whole"]["h"][0].float() - h_whole.float()).abs().max().item() for o in outs[1:])
    print(f"  ranks agree (hidden and logits identical on all {facts.TP}): {agree} (max |diff| {max_dev:.1e})")
    # every other run against the whole prefill, per block, per token, against the first chunk's own noise floor
    T, K1 = a.tokens, F.spec_k + 1
    windows = {"chunked": [(0, a.chunk), (a.chunk, T)],
               "verify": [(0, T - K1), (T - K1, T)],
               "rollback": [(0, T - 2 * K1), None, (T - 2 * K1 + 2, T - K1 + 2)]}       # None: the garbage step is not comparable
    floor = {}
    ok = finite and agree
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
    if "whole_ref" in r0["runs"]:
        # served kernels vs reference kernels on the SAME composition and inputs: the lane adapters' judge (D4 at lane level)
        print("  served lanes vs reference lanes, per block (per-token rel):")
        for (name, L), outs_b in whole["blocks"].items():
            r = tok_rel(outs_b[0], r0["runs"]["whole_ref"]["blocks"][(name, L)][0])
            p50, mx = r.median().item(), r.max().item()
            judged = name in ("kda", "dsa")
            passed = p50 <= 3e-2 if judged else True
            ok = ok and passed
            print(f"    L{L:<3}{name:<6} p50 {p50:.1e} max {mx:.1e}{'' if passed else '  <-- FAIL'}")
        r = tok_rel(h_whole, r0["runs"]["whole_ref"]["h"][0])
        print(f"    final hidden p50 {r.median().item():.1e} max {r.max().item():.1e} (MoE router flips included)")
    print("\n  " + ("PASS: four ranks agree; chunked, verify and rollback steps read the caches and add nothing" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
