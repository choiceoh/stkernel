"""Do the bytes a rank's weights hold IN THE ARENA equal the file's? (45차 §21)

The rank-file audit compared the file with the builders through the loader's CPU path; the fleet reads
through the arena path (coalesced O_DIRECT ranges, one H2D copy per run, tensors as views of the block).
Load the same names both ways and compare every byte.

    bash probes/run_engine_probe.sh probes/glm53_arena_bytes_check.py --rank 0 --layers 0,3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                                     # noqa: E402

from engine.base.arena import Arena                              # noqa: E402
from engine.base.loader import RankLoader                        # noqa: E402
from engine.base.params import total_bytes                       # noqa: E402
from engine.profiles.glm53 import facts, specs                   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--ranks", default=str(facts.RANKS))
    a = ap.parse_args(argv)
    F = facts.load()
    layers = [int(x) for x in a.layers.split(",") if x]
    group = specs.top_specs(F) + [s for L in layers for s in specs.layer_specs(F, L)]
    names = [s.name for s in group]
    loader = RankLoader(Path(a.ranks) / f"rank{a.rank}of{facts.TP}.safetensors")
    arena = Arena(total_bytes(group) + 256 * (len(group) + 64) + (64 << 20))
    gpu = loader.load(names, arena=arena, max_run=128 << 20)
    cpu = loader.load(names, device="cpu", max_run=128 << 20)
    bad = 0
    for s in group:
        g, c = gpu[s.name], cpu[s.name]
        if tuple(g.shape) != tuple(c.shape) or g.dtype != c.dtype:
            print(f"  MISMATCH {s.name}: arena {g.dtype} {tuple(g.shape)} vs cpu {c.dtype} {tuple(c.shape)}"); bad += 1; continue
        gb = g.contiguous().view(torch.uint8).flatten().cpu(); cb = c.contiguous().view(torch.uint8).flatten()
        if not torch.equal(gb, cb):
            bad += 1
            print(f"  MISMATCH {s.name}: {(gb != cb).sum().item():,} of {gb.numel():,} bytes differ; first diff at {int((gb != cb).nonzero()[0])}")
    print(f"  {'PASS' if not bad else 'FAIL'}: arena path == cpu path for {len(group)} tensors (rank {a.rank}, layers {layers} + top), {bad} different")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
