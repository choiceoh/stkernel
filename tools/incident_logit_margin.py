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
    return dict(admission=int(match.group(1)), generation=int(match.group(2)), width=int(flat.numel()),
                top1=int(top.indices[0]), top2=int(top.indices[1]),
                margin=float(top.values[0] - top.values[1]), chosen_probability=probability,
                source=path.name)


def profile(root: Path):
    rows = [row for row in (read_capture(path) for path in sorted(root.glob("*.pt"))) if row]
    return {(row["admission"], row["generation"]): row for row in rows}


def flips(left, right):
    """Shared (admission, generation) rows whose top-1 differs, with each side's margin."""
    shared = sorted(set(left) & set(right))
    out = []
    for key in shared:
        if left[key]["top1"] == right[key]["top1"]:
            continue
        tight = min(left[key]["margin"], right[key]["margin"])
        out.append(dict(admission=key[0], generation=key[1], left=left[key], right=right[key],
                        tightest_margin=tight, tight=tight < 1.0))
    return out


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
    args = parser.parse_args()

    profiles = [(Path(path), profile(Path(path))) for path in args.captures]
    for path, rows in profiles:
        report(rows, str(path))
    if args.compare and len(profiles) >= 2:
        (left_path, left), (right_path, right) = profiles[0], profiles[1]
        found = flips(left, right)
        print(f"shared rows {len(set(left) & set(right))}; top-1 flips {len(found)}")
        print("  a flip at a tight margin is the knife edge; a flip at a wide one is a different computation")
        for row in found[:args.limit]:
            print(f"   admit{row['admission']} gen{row['generation']:5d}: margins "
                  f"{row['left']['margin']:.3f} / {row['right']['margin']:.3f} "
                  f"(min {row['tightest_margin']:.3f}) -> {'TIGHT' if row['tight'] else 'wide'}")
        if found:
            tight = sorted(row["tightest_margin"] for row in found)
            print(f"  flip margins: min {tight[0]:.3f} median {tight[len(tight) // 2]:.3f} max {tight[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
