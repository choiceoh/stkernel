"""The first token of a real prompt through layers 0..k, on the reference lanes, with per-layer hidden stats.

45차 §21: the fleet's text is garbage on both lane tables and every self-consistency judge passes. This
probe puts a real prompt through the same composition (Glm53Net + the reference lanes, TP=4 as threads on
one GB10 -- rank files for all four ranks must be on this node) and prints what an external reference
(vLLM's Glm5Next on the same layers) can be compared against: the embedding, every block's output, the
final hidden state and the top-10 next-token logits.

    PYTHONPATH=. python3 probes/glm53_first_token.py --layers 0-2 --prompt "The capital of France is"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                                     # noqa: E402

from engine.base.arena import Arena                              # noqa: E402
from engine.base.comm import LocalTP                             # noqa: E402
from engine.base.instruments import Recorder                     # noqa: E402
from engine.base.params import total_bytes                       # noqa: E402
from engine.profiles.glm53 import facts, lanes as lane_tables    # noqa: E402
from engine.profiles.glm53.check import ChainCaches              # noqa: E402
from engine.profiles.glm53.net import Glm53Net, Step             # noqa: E402
from engine.profiles.glm53.weights import rank_loader             # noqa: E402


def stats(t: torch.Tensor) -> str:
    f = t.float()
    return f"|mean| {f.abs().mean().item():.4f} rms {f.pow(2).mean().sqrt().item():.4f} max {f.abs().max().item():.3f}"


def rank_main(comm, a, F, layers, ids):
    rec = Recorder(f"rank{comm.rank}")
    net = Glm53Net(F, comm, lane_tables.reference(), layers)
    specs = net.specs()
    cap = -(-len(ids) // F.block) * F.block
    with rec.phase("arena"):
        arena = Arena(total_bytes(specs) + 256 * len(specs) + len(layers) * (64 << 20) + cap * 8192 + (256 << 20))
    with rec.phase("load"):
        views = rank_loader(Path(a.ranks) / f"rank{comm.rank}of{facts.TP}.safetensors").load(
            [s.name for s in specs], arena=arena, recorder=rec, max_run=128 << 20)
        net.bind(views)
    chain = ChainCaches(arena, F, net, cap)
    blocks = {}
    net.probe = lambda name, L, out: blocks.setdefault((name, L), out.detach().clone())
    emb = net.embed(ids)
    h = net.forward(Step.prefill(ids, 0, 0, 1), chain)
    logits = net.head(h[-1:])[0].float()
    torch.cuda.synchronize()
    return {"emb": emb, "blocks": blocks, "h": h, "logits": logits}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", default="0-2")
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ids", default="", help="comma-separated token ids instead of --prompt")
    ap.add_argument("--dump", default="", help="directory: save embed / block outputs / final hidden / logits as .pt for an exact diff")
    a = ap.parse_args(argv)
    F = facts.load()
    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    if a.ids:
        ids_list = [int(x) for x in a.ids.split(",")]
    else:
        from engine.profiles.glm53.boot import tokenizer
        tok = tokenizer()                                         # the door's tokenizer: no inherited truncation
        ids_list = tok.encode(a.prompt, add_special_tokens=False).ids
    print(f"  prompt {a.prompt!r} -> {len(ids_list)} ids {ids_list}")
    ids = torch.tensor(ids_list, dtype=torch.int64, device="cuda")
    tp = LocalTP(facts.TP)
    outs = tp.run(rank_main, a, F, layers, ids)
    r0 = outs[0]
    print(f"  layers {layers[0]}-{layers[-1]} ({len(layers)}), reference lanes, TP={facts.TP} in-process")
    print(f"  embed[last]     {stats(r0['emb'][-1])}   embed[all] {stats(r0['emb'])}")
    for (name, L), out in sorted(r0["blocks"].items(), key=lambda kv: (kv[0][1], kv[0][0] != "kda" and kv[0][0] != "dsa")):
        print(f"  L{L:<3}{name:<6} out[last] {stats(out[-1])}   out[all] {stats(out)}")
    print(f"  final h[last]   {stats(r0['h'][-1])}")
    agree = all(torch.equal(o["h"], r0["h"]) for o in outs)
    top = r0["logits"].topk(10)
    print(f"  ranks agree {agree}; top-10 ids {top.indices.tolist()}")
    print(f"  top-10 logits {[round(v, 3) for v in top.values.tolist()]}")
    try:
        from engine.profiles.glm53.boot import tokenizer
        tok = tokenizer()
        print(f"  top-10 tokens {[tok.id_to_token(i) for i in top.indices.tolist()]}")
    except Exception:
        pass
    if a.dump:
        d = Path(a.dump); d.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": ids_list, "emb": r0["emb"].cpu(), "h": r0["h"].cpu(), "logits": r0["logits"].cpu(),
                    "blocks": {f"{name}.{L}": v.cpu() for (name, L), v in r0["blocks"].items()}}, d / "st_first_token.pt")
        print(f"  dumped to {d / 'st_first_token.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
