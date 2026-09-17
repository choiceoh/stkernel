#!/usr/bin/env python3
"""How tight was the choice? Margin profile of the incident's own logit captures.

The engine is deterministic -- a repeat of one arm is bit-identical over 90 prefill
stages and mode 5 vs mode 0 agree to 0.0 -- so when two arms produce different text,
the difference is caused by whatever constant or code changed between them, not by a
race. What decides whether such a change flips the answer is the *margin* at the
token it flips. This reads the incident's `incident-logits/*.pt` captures (one
154,880-wide logit row per (admission, generation)) and reports the chosen token's
probability and its margin over the runner-up, so an arm comparison can say *where*
the generation first differs and how close that choice was.

    python3 tools/incident_logit_margin.py CAPTURE_DIR [CAPTURE_DIR ...]
    python3 tools/incident_logit_margin.py --compare DIR_A DIR_B

Private logits stay outside the repository: only counts, probabilities and ids print.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PREFIX_FIELDS = ("prefix_sha256", "input_prefix_sha256", "prefix_hash", "prompt_sha256", "prefix")
UNIFORM_FIELDS = ("uniform", "uniforms", "u", "draw", "draws", "seed_uniform")


def _scalar(node, fields):
    """The first value under any of `fields`, whether it sits at the top level or one dict down."""
    for source in (node, *(v for v in node.values() if isinstance(v, dict))):
        for field in fields:
            if field in source and not hasattr(source[field], "shape"):
                return source[field]
    return None


def read_capture(path: Path):
    """(admission, generation, top1, top2, margin, chosen probability) or None."""
    import torch

    match = re.search(r"admit(\d+)-gen(\d+)", path.name)
    if not match:
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                                                        # noqa: BLE001 -- a truncated capture is skipped, not fatal
        return None
    row = None
    for value in (obj.values() if isinstance(obj, dict) else [obj]):
        if hasattr(value, "shape") and value.dim() >= 1 and value.numel() >= 2:
            if row is None or value.numel() > row.numel():
                row = value
    if row is None:
        return None
    flat = row.float().flatten()
    top = torch.topk(flat, 2)
    probability = float(torch.softmax(flat, dim=-1)[top.indices[0]])
    prefix = _scalar(obj, PREFIX_FIELDS) if isinstance(obj, dict) else None
    uniform = _scalar(obj, UNIFORM_FIELDS) if isinstance(obj, dict) else None
    return dict(admission=int(match.group(1)), generation=int(match.group(2)), width=int(flat.numel()),
                top1=int(top.indices[0]), top2=int(top.indices[1]),
                margin=float(top.values[0] - top.values[1]), chosen_probability=probability,
                prefix=None if prefix is None else str(prefix),
                uniform=None if uniform is None else float(uniform),
                source=path.name)


def profile(root: Path):
    rows = [row for row in (read_capture(path) for path in sorted(root.glob("*.pt"))) if row]
    return {(row["admission"], row["generation"]): row for row in rows}


def flips(left, right, *, prefixes=True):
    """Shared rows whose top-1 differs, with each side's margin.

    A row is only compared when both sides were fed the same prefix -- the capture's
    prefix hash is the guard, because logits from different prefixes differ by
    construction and a conclusion drawn across them is not about the engine. A row
    whose uniforms also differ has a *draw* difference on top: the choice moved
    because the random number moved, not because the distribution did.
    """
    out, skipped = [], []
    for key in sorted(set(left) & set(right)):
        l, r = left[key], right[key]
        if prefixes and l["prefix"] is not None and r["prefix"] is not None and l["prefix"] != r["prefix"]:
            skipped.append(key)
            continue
        if l["top1"] == r["top1"]:
            continue
        tight = min(l["margin"], r["margin"])
        same_uniform = (None if l["uniform"] is None or r["uniform"] is None
                        else abs(l["uniform"] - r["uniform"]) < 1e-12)
        out.append(dict(admission=key[0], generation=key[1], left=l, right=r,
                        tightest_margin=tight, tight=tight < 1.0, same_uniform=same_uniform))
    return out, skipped


def report(rows, label):
    if not rows:
        print(f"{label}: no captures")
        return
    ordered = sorted(rows.values(), key=lambda r: (r["admission"], r["generation"]))
    probabilities = sorted(r["chosen_probability"] for r in ordered)
    print(f"{label}: {len(ordered)} captures; chosen probability median "
          f"{probabilities[len(probabilities) // 2]:.3f}, under 0.5: "
          f"{sum(1 for p in probabilities if p < 0.5)}/{len(ordered)}")
    for row in ordered[:6]:
        print(f"   admit{row['admission']} gen{row['generation']:5d}: top1 {row['top1']:6d} "
              f"p {row['chosen_probability']:.4f} margin {row['margin']:.3f}  ({row['source']})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", help="directories of incident-logits captures")
    parser.add_argument("--compare", action="store_true", help="report the shared rows' flips and their margins")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--allow-mixed-prefixes", action="store_true",
                        help="compare rows even when the two sides' prefix hashes differ (off by default)")
    args = parser.parse_args()

    profiles = [(Path(path), profile(Path(path))) for path in args.captures]
    for path, rows in profiles:
        report(rows, str(path))
    if args.compare and len(profiles) >= 2:
        (left_path, left), (right_path, right) = profiles[0], profiles[1]
        found, skipped = flips(left, right, prefixes=not args.allow_mixed_prefixes)
        print(f"shared rows {len(set(left) & set(right))}; compared {len(found)}; "
              f"skipped for differing prefixes {len(skipped)}")
        print("  a flip at a tight margin is the knife edge; at a wide one a different computation;")
        print("  same_uniform=False means the draw moved, not the distribution")
        for row in found[:args.limit]:
            uniform = row["same_uniform"]
            print(f"   admit{row['admission']} gen{row['generation']:5d}: margins "
                  f"{row['left']['margin']:.3f} / {row['right']['margin']:.3f} "
                  f"(min {row['tightest_margin']:.3f}) -> {'TIGHT' if row['tight'] else 'wide'}"
                  f"{'' if uniform is None else (' same-uniform' if uniform else ' DIFFERENT-UNIFORM')}")
        if found:
            tight = sorted(row["tightest_margin"] for row in found)
            print(f"  flip margins: min {tight[0]:.3f} median {tight[len(tight) // 2]:.3f} max {tight[-1]:.3f}")
            draws = [row for row in found if row["same_uniform"] is False]
            if draws:
                print(f"  flips whose uniforms differ: {len(draws)}/{len(found)} -- the draw moved, check the RNG stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
