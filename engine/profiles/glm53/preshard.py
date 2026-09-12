"""Cut the HF checkpoint into rank files in the engine's layout (profile tool).

    python3 engine/profiles/glm53/preshard.py                       # -> facts.RANKS/rank{0..3}of4.safetensors
    python3 engine/profiles/glm53/preshard.py --layers 0-4 --out /some/dev/dir
    python3 engine/profiles/glm53/preshard.py --vision               # -> facts.RANKS/vision.safetensors (the tower, whole; seconds)
    python3 engine/profiles/glm53/preshard.py --drafter-w4           # -> facts.RANKS/drafter-w4.safetensors (the DFlash2 GEMMs int4; minutes, CPU)

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
    ap.add_argument("--vision", action="store_true", help="write only vision.safetensors: the vision tower, whole, for every rank (45차 §23 A7)")
    ap.add_argument("--drafter-w4", action="store_true",
                    help="write only drafter-w4.safetensors: the DFlash2 drafter with its 47 GEMMs packed int4, one file for every rank (45차 §23 GPU 판정 5차)")
    ap.add_argument("--drafter", default=None, help="the DFlash2 checkpoint directory (default: drafter.DRAFTER)")
    a = ap.parse_args(argv)
    if a.drafter_w4:
        from engine.profiles.glm53 import drafter as drafter_mod
        src = a.drafter or drafter_mod.DRAFTER
        print(f"  glm53 preshard: drafter W4 {src} -> {Path(a.out) / drafter_mod.W4_FILE}")
        size = drafter_mod.write_w4_file(src, a.out)
        print(f"  done: {size / 2**30:.2f} GiB")
        return 0
    if a.vision:
        from engine.profiles.glm53 import vision
        print(f"  glm53 preshard: vision tower -> {Path(a.out) / vision.FILE}")
        size = vision.write_file(a.ckpt, a.out)
        print(f"  done: {size / 2**30:.2f} GiB")
        return 0
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
