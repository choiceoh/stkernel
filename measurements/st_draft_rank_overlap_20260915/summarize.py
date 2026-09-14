"""Summaries of the two first-rejection probes (main 9c45086a + debug branches, C=1 K=7 greedy, onepass JSON 12).

    python3 summarize.py rank-rows.jsonl.gz overlap-rows.jsonl.gz [--moe-share 0.456]

rank rows    (debug-draft-candidate-rank cc3ab619): one `draft_rejection` row per decode step on rank 0 -- reason,
             accepted prefix, and at the first rejection the target's and the walk's rank among the drafter's 16
             candidates (ordered by the drafter's own score).
overlap rows (debug-sibling-expert-overlap ef2e5b6e): one `route_overlap` row per decode step -- distinct (layer, expert)
             routes of the 8 verify rows, each draft row's leave-one-out routes, and the next step's anchor routes the
             previous step's rows did not read (after a rejection that anchor is exactly the right sibling).

Everything below "estimates" is arithmetic on these counts, not a measurement of a tree.
"""
import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict

K = 7            # draft width: 8 verify rows
CANDIDATES = 16  # sel_top_k
UNIFORM_8_OF_288 = 288 * (1 - (280 / 288) ** 8)


def rows_of(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return [json.loads(line) for line in f]


def pct(x):
    return f"{100 * x:.1f}%"


def siblings(r, width):
    """the best-ranked candidates other than the walk's own pick: what a tree adding `width` siblings would draft"""
    return [x for x in range(CANDIDATES) if x != r["draft_rank"]][:width]


def rank_summary(rows, moe_share, sibling_reads):
    steps = [r for r in rows if r["reason"] != "output_boundary"]
    n = len(steps)
    reasons = Counter(r["reason"] for r in rows)
    print(f"## rank probe: {len(rows)} steps ({reasons['output_boundary']} at the output boundary, left out below)")
    rejected = [r for r in steps if r["reason"] != "all_accepted"]
    covered = [r for r in rejected if r["reason"] == "selector_miss"]
    print(f"all accepted {reasons['all_accepted']} ({pct(reasons['all_accepted'] / n)}), "
          f"selector_miss {reasons['selector_miss']} ({pct(reasons['selector_miss'] / n)}), "
          f"candidate_miss {reasons['candidate_miss']} ({pct(reasons['candidate_miss'] / n)})")
    accepted = [r["accepted_prefix"] for r in steps]
    tps = statistics.mean(accepted) + 1
    print(f"tokens/step {tps:.3f}")
    print(f"first rejections {len(rejected)}: the target is among the {CANDIDATES} candidates in {len(covered)} ({pct(len(covered) / len(rejected))})")
    hist = Counter(r["target_rank"] for r in covered)
    cum, parts = 0, []
    for rank in range(CANDIDATES):
        cum += hist.get(rank, 0)
        parts.append(f"<={rank + 1} {pct(cum / len(rejected))}")
    print("target's candidate rank (1 = drafter top-1), cumulative share of all first rejections: " + ", ".join(parts[:4] + parts[7:8] + parts[15:]))
    print("target rank histogram (1-based): " + ", ".join(f"{k + 1}:{hist[k]}" for k in sorted(hist)))
    walk = Counter(r["draft_rank"] for r in rejected)
    print("the walk's own pick at the rejected position (1-based): " + ", ".join(f"{k + 1}:{walk[k]}" for k in sorted(walk)))
    joint = Counter((r["target_rank"] == 0, r["draft_rank"] == 0) for r in covered)
    print(f"selector misses: walk took the drafter top-1, target ranked lower {joint[(False, True)]} ({pct(joint[(False, True)] / len(rejected))}); "
          f"target was the drafter top-1, walk took another {joint[(True, False)]} ({pct(joint[(True, False)] / len(rejected))}); "
          f"neither top-1 {joint[(False, False)]} ({pct(joint[(False, False)] / len(rejected))})")
    for width in (1, 2, 3, 7):
        caught = sum(r["target_rank"] in siblings(r, width) for r in covered)
        print(f"  {width} sibling(s) by drafter rank would hold the target at {caught} of {len(rejected)} first rejections ({pct(caught / len(rejected))})")

    print("\nby position (p = accepted prefix; the rejected draft is p+1):")
    print("| p | reached | rejected | cond. accept | target in candidates | 1 sibling right | target is drafter top-1 |")
    print("|---|---|---|---|---|---|---|")
    cond = []
    for p in range(K):
        reach = sum(1 for a in accepted if a >= p)
        rej = [r for r in rejected if r["accepted_prefix"] == p]
        cov = [r for r in rej if r["reason"] == "selector_miss"]
        one = sum(r["target_rank"] in siblings(r, 1) for r in cov)
        top1 = sum(r["target_rank"] == 0 for r in cov)
        cond.append(1 - len(rej) / reach)
        print(f"| {p} | {reach} | {len(rej)} | {cond[-1]:.3f} | {pct(len(cov) / len(rej))} | {pct(one / len(rej))} | {pct(top1 / len(rej))} |")
    model = 1 + sum(math.prod(cond[:k]) for k in range(1, K + 1))
    print(f"tokens/step from the conditional acceptances {model:.3f} (identity check against {tps:.3f})")

    print("\n## estimates (arithmetic on the counts above)")
    p_all = reasons["all_accepted"] / n
    print(f"the deepest chain node (draft 7) adds a token only when all 7 accept: +{p_all:.3f} tokens/step ({pct(p_all / tps)})")
    wasted = sum(K - r["accepted_prefix"] for r in rejected) / n
    print(f"draft rows whose input is the rejected token or follows it: {wasted:.2f} of {K} per step")
    print("a right sibling leaf adds exactly one token (it is the correction, and its row yields the bonus):")
    for width in (1, 2, 3):
        gains = [sum(r["target_rank"] in siblings(r, width) for r in rejected if r["reason"] == "selector_miss" and r["accepted_prefix"] == p) / n
                 for p in range(K)]
        best = max(range(K), key=lambda p: gains[p])
        print(f"  {width} sibling(s) at p: " + ", ".join(f"p{p} +{g:.3f}" for p, g in enumerate(gains))
              + f"; best p{best} +{gains[best]:.3f} tokens/step ({pct(gains[best] / tps)}), at every position +{sum(gains):.3f}")
    best1 = max(sum(r["target_rank"] in siblings(r, 1) for r in rejected if r["reason"] == "selector_miss" and r["accepted_prefix"] == p) / n
                for p in range(K))
    if sibling_reads is not None:
        cost = sibling_reads * moe_share
        print(f"one sibling as a 9th row: +{pct(best1 / tps)} tokens/step against +{pct(sibling_reads)} expert reads x MoE share {moe_share:.3f} "
              f"= +{pct(cost)} step time from MoE alone; break-even needs <= {pct(best1 / tps / moe_share)} new reads")
    print(f"in a fixed 8-node budget the sibling replaces the deepest chain node: +{best1:.3f} against +{p_all:.3f} tokens/step")
    for share in (0.25, 0.5, 1.0):
        # a better drafter: `share` of the misses whose target was the drafter's 2nd choice become accepts at that position,
        # later positions keep their conditional acceptance
        lifted = []
        for p in range(K):
            reach = sum(1 for a in accepted if a >= p)
            second = sum(1 for r in rejected if r["reason"] == "selector_miss" and r["accepted_prefix"] == p and r["target_rank"] == 1)
            lifted.append(cond[p] + share * second / reach)
        new = 1 + sum(math.prod(lifted[:k]) for k in range(1, K + 1))
        print(f"  if {share:.0%} of the 2nd-choice misses were drafted right: mean cond. accept {statistics.mean(cond):.3f} -> "
              f"{statistics.mean(lifted):.3f}, tokens/step {tps:.2f} -> {new:.2f} ({pct(new / tps - 1)})")


def overlap_summary(rows):
    rows = [r for r in rows if r.get("rows") == K + 1]
    layers = rows[0]["layers"]
    slots = 8 * layers
    print(f"\n## overlap probe: {len(rows)} steps, {layers} MoE layers, top-8; stale route sets {sum(r['stale'] for r in rows)}; "
          f"tokens/step {statistics.mean(r['accepted'] for r in rows) + 1:.3f}")
    print(f"distinct experts per layer read by the 8 verify rows: mean {statistics.mean(r['union_total'] / layers for r in rows):.1f} "
          f"(uniform random top-8 of 288 per row: {UNIFORM_8_OF_288:.1f}; one row: 8)")
    print(f"a chain draft row's routes no other row reads: {pct(statistics.mean(r['loo_draft_mean'] / slots for r in rows))} of its {slots}, "
          f"= +{pct(statistics.mean(r['loo_draft_mean'] / r['union_total'] for r in rows))} of the step's reads")
    cont = [r for r in rows if "anchor_new" in r]
    out = {}
    for name, sel in (("after a rejection (anchor = the right sibling)", [r for r in cont if r["prev_accepted"] < r["prev_drafted"]]),
                      ("after all 7 accepted (anchor = the continuation)", [r for r in cont if r["prev_accepted"] == r["prev_drafted"]])):
        new = [r["anchor_new"] / slots for r in sel]
        growth = [r["anchor_new"] / r["prev_union_total"] for r in sel]
        out[name] = statistics.mean(growth)
        print(f"{name}: {len(sel)} steps; new routes {pct(statistics.mean(new))} of the row's {slots} (median {pct(statistics.median(new))}), "
              f"adding the row = +{pct(statistics.mean(growth))} reads")
    by_pos = defaultdict(list)
    for r in cont:
        if r["prev_accepted"] < r["prev_drafted"]:
            by_pos[r["prev_accepted"]].append(r["anchor_new"] / r["prev_union_total"])
    print("sibling at p (accepted prefix before the rejection): " + ", ".join(
        f"p{p} +{pct(statistics.mean(v))} (n={len(v)})" for p, v in sorted(by_pos.items())))
    return statistics.mean(by_pos[0]) if by_pos[0] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rank_rows")
    ap.add_argument("overlap_rows")
    ap.add_argument("--moe-share", type=float, default=0.456, help="MoE expert share of a C=1 K=7 step: 22.03 of 48.30 kernel ms, main 9c45086a (#965)")
    a = ap.parse_args()
    sibling_reads = overlap_summary(rows_of(a.overlap_rows))
    print()
    rank_summary(rows_of(a.rank_rows), a.moe_share, sibling_reads)


if __name__ == "__main__":
    main()
