"""Are the rank files what specs.py says they are? (45차 §21: garbage text from both lane tables)

For every non-expert Spec of the chosen layers (and the top trio) on rank r, rebuild the tensor from
the HF checkpoint with the spec's own builder and compare it byte for byte with what the rank file
holds. A file cut by another tree's builders passes the name/shape check and the layout marker but
carries different bytes -- this is the only judge for that.

    python3 probes/glm53_rankfile_audit.py --rank 0 --layers 0,3 [--ranks DIR] [--experts]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                                     # noqa: E402

from engine.base.checkpoint import Checkpoint                    # noqa: E402
from engine.base.loader import RankLoader                        # noqa: E402
from engine.profiles.glm53 import facts, specs                   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--ckpt", default=str(facts.CKPT))
    ap.add_argument("--experts", action="store_true", help="also the packed NVFP4 experts (slow, large)")
    a = ap.parse_args(argv)
    F = facts.load(a.ckpt)
    layers = [int(x) for x in a.layers.split(",") if x]
    ck = Checkpoint(a.ckpt)
    rank_file = RankLoader(Path(a.ranks) / f"rank{a.rank}of{facts.TP}.safetensors")
    groups = [("top", specs.top_specs(F))] + [(f"layer {L}", specs.layer_specs(F, L)) for L in layers]
    bad = same = 0
    for label, group in groups:
        if not a.experts:
            group = [s for s in group if not (s.name.endswith(".moe.w13") or s.name.endswith(".moe.w13_sf")
                                             or s.name.endswith(".moe.w2") or s.name.endswith(".moe.w2_sf"))]
        keys = sorted({k for s in group for k in s.sources})
        t0 = time.perf_counter()
        src = ck.load(keys, device="cpu")
        got = rank_file.load([s.name for s in group], device="cpu")
        for s in group:
            want = s.build(src, a.rank, facts.TP)
            have = got[s.name]
            if want.dtype != have.dtype or tuple(want.shape) != tuple(have.shape):
                print(f"  MISMATCH {s.name}: built {want.dtype} {tuple(want.shape)} vs file {have.dtype} {tuple(have.shape)}")
                bad += 1
                continue
            wb, hb = want.contiguous().view(torch.uint8).flatten(), have.contiguous().view(torch.uint8).flatten()
            if torch.equal(wb, hb):
                same += 1
                continue
            bad += 1
            diff = (wb != hb).sum().item()
            note = ""
            if want.dtype in (torch.bfloat16, torch.float32):
                rel = ((want.float() - have.float()).abs().max() / (want.float().abs().max() + 1e-12)).item()
                note = f" max rel {rel:.2e}"
            print(f"  MISMATCH {s.name}: {diff:,} of {wb.numel():,} bytes differ{note}")
        print(f"{label}: {len(group)} specs checked in {time.perf_counter() - t0:.1f} s (identical so far {same}, different {bad})")
    print(f"\n  {'PASS' if not bad else 'FAIL'}: rank {a.rank} file == specs.py builders for {[g[0] for g in groups]}: {same} identical, {bad} different")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
