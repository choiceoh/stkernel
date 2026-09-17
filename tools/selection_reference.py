#!/usr/bin/env python3
"""Recompute the sparse indexer's selection from a captured dump and compare it.

The incident's last unverified path is the selection: the selector *kernels* are
held to torch.topk on synthetic logits and the KDA operands were audited on real
ones, but the pool ids the indexer picks for a real long prefix have never been
compared with a reference recomputed from the very operands that produced them.
`engine/modules/selection_capture.py` writes those operands (ST_SELECTION_CAPTURE);
this compares them.

    ST_SELECTION_CAPTURE=/tmp/selection python3 -m unittest ...   # in the boot
    python3 tools/selection_reference.py /tmp/selection/*.pt --out report.json

The reference is `engine/modules/sparse_indexer.indexer_logits` -- the bf16
reference of the served DeepGEMM op (`probes/indexer_check.py` judges it against
that op): the served read takes raw fp8 queries with the per-token scale already
folded into the weights, and dequantises the keys with their per-row fp32 scale.

A difference on a row whose k-th and (k+1)-th scores are equal is a tie rule, not
a wrong score: the margin at the cut is reported beside every differing row so the
two cannot be confused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def select_reference(dump):
    """The reference selection for a captured dump, in the engine's own semantics.

    The `index_kpool_always_select_tail` pin is part of that policy, so the
    reference applies it too; `pin_report` measures how often it mattered.
    """
    import torch
    from engine.modules.sparse_indexer import indexer_logits, pin_pools_in_logits, tail_pin_pools, topk_positions

    q8, keys, scales = dump["q8"], dump["keys"], dump["scales"]
    w_eff, ke, k = dump["w_eff"], dump["ke"], int(dump["k"])
    keys_dequant = keys.float() * scales.float()[:, None]
    logits = torch.einsum("mhd,nd->mhn", q8.float(), keys_dequant).relu_()
    logits = torch.einsum("mhn,mh->mn", logits, w_eff.float())
    pin = tail_pin(torch, dump)
    unbiased = topk_positions(logits.clone(), k, valid=ke)
    pin_pools_in_logits(logits, pin)
    return logits, topk_positions(logits, k, valid=ke), unbiased, pin


def tail_pin(torch, dump):
    """The dump's pin, or None when it predates the raw horizon fields."""
    from engine.modules.sparse_indexer import tail_pin_pools

    seq, pool = dump.get("seq_lens"), dump.get("pool")
    if seq is None or pool is None:
        return None
    return tail_pin_pools(seq.to(torch.int32), int(pool))


def pin_report(dump, pin, unbiased) -> dict:
    """How often recency would have lost the top-k: rows the pin holds, of which
    the pinned pool is absent from the unbiased selection."""
    import torch
    if pin is None:
        return dict(available=False)
    live = pin >= 0
    rows = int(live.sum().item())
    held = 0
    for row in torch.nonzero(live, as_tuple=False).flatten().tolist():
        if int(pin[row]) not in {int(x) for x in unbiased[row].tolist() if x >= 0}:
            held += 1
    return dict(available=True, pinned_rows=rows, would_have_been_dropped=held)


def compare(dump, *, sample_rows=8):
    """Per-row set comparison of the captured selection against the reference."""
    import torch

    logits, reference, unbiased, pin = select_reference(dump)
    captured, ke, k = dump["selected"], dump["ke"], int(dump["k"])
    rows = logits.shape[0]
    if tuple(captured.shape) != (rows, k):
        raise SystemExit(f"captured selection is {tuple(captured.shape)}, expected {(rows, k)}")
    differing = []
    for row in range(rows):
        got = {int(x) for x in captured[row].tolist() if x >= 0}
        want = {int(x) for x in reference[row].tolist() if x >= 0}
        if got == want:
            continue
        horizon = int(ke[row].item()) if torch.is_tensor(ke[row]) else int(ke[row])
        width = min(k + 1, max(horizon, 1))
        scores = torch.topk(logits[row, :horizon].float(), width, sorted=True).values
        margin = float(scores[k - 1] - scores[k]) if scores.numel() > k else float("inf")
        # A tie INSIDE the top-k is not visible at the cut: pair the swapped members by
        # rank and report how far apart their scores are. Zero is a tie rule; large is a
        # score the two paths disagree about.
        exchanged = sorted(got ^ want)
        captured_scores = sorted((float(logits[row, p]) for p in exchanged if p in got), reverse=True)
        reference_scores = sorted((float(logits[row, p]) for p in exchanged if p in want), reverse=True)
        tie_gap = max((abs(a - b) for a, b in zip(captured_scores, reference_scores)), default=0.0)
        differing.append(dict(row=row, horizon=horizon, shared=len(got & want),
                              only_captured=sorted(got - want)[:sample_rows],
                              only_reference=sorted(want - got)[:sample_rows],
                              margin_at_k=margin, tie_gap=tie_gap))
    return dict(layer=int(dump["layer"]), rows=rows, k=k, n_cand=int(dump["n_cand"]),
                heads=int(dump["heads"]), width=int(dump["width"]),
                differing_rows=len(differing), compared_rows=rows, differing=differing[:sample_rows],
                tail_pin=pin_report(dump, pin, unbiased))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+", help="selection-L*.pt files from ST_SELECTION_CAPTURE")
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    import torch

    reports = [compare(torch.load(path, map_location="cpu")) for path in args.dumps]
    for report in reports:
        ties = [row for row in report["differing"] if abs(row["margin_at_k"]) < 1e-6]
        pin = report["tail_pin"]
        pin_text = (f" tail-pin {pin['would_have_been_dropped']}/{pin['pinned_rows']} rows"
                    if pin.get("available") else "")
        print(f"L{report['layer']:>3} rows={report['compared_rows']:>6} k={report['k']} "
              f"n_cand={report['n_cand']:>6} differing={report['differing_rows']}"
              f"{f' (ties at the cut: {len(ties)})' if ties else ''}{pin_text}")
        if not args.summary_only:
            for row in report["differing"]:
                print(f"    row {row['row']}: shared {row['shared']} of {report['k']}, "
                      f"only captured {row['only_captured'][:4]}, only reference {row['only_reference'][:4]}, "
                      f"margin at k {row['margin_at_k']:.3e}")
    if args.out:
        Path(args.out).write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
