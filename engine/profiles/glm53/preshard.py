"""Cut the HF checkpoint into rank files in the engine's layout (profile tool).

    python3 engine/profiles/glm53/preshard.py                       # -> facts.RANKS/rank{0..3}of4.safetensors
    python3 engine/profiles/glm53/preshard.py --layers 0-4 --out /some/dev/dir

Runs once, offline. What the fleet boots from afterwards is `rank{r}of{W}.safetensors`
read by base/loader.RankLoader with coalesced range reads into the arena --
no per-tensor work at boot, which is the difference between the served
340 s load and a disk-bound one (D1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from engine.base.checkpoint import Checkpoint            # noqa: E402
from engine.base.preshard import write_ranks             # noqa: E402
from engine.profiles.glm53 import facts, specs           # noqa: E402
from engine.profiles.glm53.weights import WEIGHT_LAYOUT  # noqa: E402


def parse_layers(spec: str, n: int):
    if spec == "all":
        return list(range(n))
    out = []
    for piece in spec.split(","):
        if "-" in piece:
            lo, hi = piece.split("-"); out += list(range(int(lo), int(hi) + 1))
        elif piece:
            out.append(int(piece))
    return sorted(set(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--out", default=str(facts.RANKS))
    ap.add_argument("--ckpt", default=str(facts.CKPT))
    a = ap.parse_args(argv)
    F = facts.load(a.ckpt)
    layers = parse_layers(a.layers, F.layers)
    ck = Checkpoint(a.ckpt)
    out = Path(a.out)
    paths = [out / f"rank{r}of{facts.TP}.safetensors" for r in range(facts.TP)]
    print(f"  glm53 preshard: TP {facts.TP}, layers {layers[0]}..{layers[-1]} ({len(layers)}), -> {out}")
    sizes = write_ranks(specs.groups(F, layers), paths, lambda keys: ck.load(keys), facts.TP,
                        metadata={"model": "glm53", "world": facts.TP, "layers": a.layers,
                                  "layout": "engine.profiles.glm53.specs", "weight_layout": WEIGHT_LAYOUT})
    print(f"  done: {sizes[0] / 2**30:.2f} GiB per rank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
