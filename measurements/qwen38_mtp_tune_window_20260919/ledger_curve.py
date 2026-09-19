#!/usr/bin/env python3
"""The draft ledger's curve (fleet --draft-ledger, adapter.ServedMTP.record): for each draft depth, how often the
target kept the pick as a function of the head's probability of it, among rows whose earlier drafts were kept -- and
what a threshold would have cut and cost, replayed on the same rows (a cut at depth j drops drafts j.. of that row).

    ledger_curve.py LEDGER.jsonl [LEDGER.jsonl ...]    rows with every draft proposed (threshold off) are the curve's

Replay (per row, K drafts proposed): kept = matched; with a threshold t the proposal ends before the first pick with
p < t, so kept_t = min(matched, cut_t) and the verify width is 1 + cut_t instead of 1 + K. The cost model in tokens
is the one this campaign's default came from (#1228: a verify position ~1.75 ms, a token ~14 ms at K=3) -- printed
beside, not a measurement.
"""
import json
import sys

BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97, 1.01]
rows = []
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        if r.get("probs") and r["picks"] and r["proposed"] == len(r["picks"]):
            rows.append(r)
if not rows:
    raise SystemExit("no row with every draft proposed and its probabilities")
K = max(len(r["picks"]) for r in rows)
print(json.dumps({"rows": len(rows), "k": K, "mean_matched": round(sum(r["matched"] for r in rows) / len(rows), 4)}))
for j in range(K):
    counts = [[0, 0] for _ in BINS[:-1]]
    for r in rows:
        if len(r["probs"]) <= j or r["matched"] < j:          # an earlier draft was rejected: this one never counted
            continue
        p = r["probs"][j]
        b = next(i for i in range(len(BINS) - 1) if BINS[i] <= p < BINS[i + 1])
        counts[b][0] += 1
        counts[b][1] += int(r["matched"] > j)
    print(json.dumps({"depth": j + 1, "bins": [{"p": f"{BINS[i]:.2f}-{BINS[i + 1]:.2f}", "n": n, "kept": round(k / n, 3) if n else None}
                                                  for i, (n, k) in enumerate(counts) if n]}))
for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
    kept = width = 0
    for r in rows:
        cut = next((j for j, p in enumerate(r["probs"]) if p < t), len(r["probs"]))
        kept += min(r["matched"], cut)
        width += 1 + cut
    base_kept = sum(r["matched"] for r in rows)
    base_width = sum(1 + len(r["probs"]) for r in rows)
    lost = (base_kept - kept) / len(rows)
    saved = (base_width - width) / len(rows)
    print(json.dumps({"threshold": t, "tokens_lost_a_row": round(lost, 4), "verify_positions_saved_a_row": round(saved, 4),
                      "model_ms_a_row": round(saved * 1.75 - lost * 14.0, 3)}))
