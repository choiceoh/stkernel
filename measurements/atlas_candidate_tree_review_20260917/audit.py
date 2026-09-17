"""Read-only arithmetic audit of fixed spine-plus-leaf shapes; no GPU/model use.

All shapes retain the baseline spine and have only leaf alternatives at spine
depths. Oracle coverage is an upper bound, not a predictor or measured tree.
The older rank capture also permits its recorded unary-order sibling policy.
"""
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
K = 7


def identity(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def shape_stats(counts, histogram, gains):
    length = len(counts)
    n = sum(histogram.values())
    baseline = sum(p * count for p, count in histogram.items())
    lost = sum(max(0, p - length) * count for p, count in histogram.items())
    rescued = sum(gains[p][count - 1] for p, count in enumerate(counts))
    accepted = baseline - lost + rescued
    # Existing Tree.state_updates carry count for spine-first DFS traversal.
    passes = 1 + sum(counts) + sum((p + 1) * (count - 1)
                                  for p, count in enumerate(counts))
    return {"counts_by_depth": counts, "spine": length,
            "verification_rows": 1 + sum(counts),
            "baseline_accepted_total": baseline, "tail_tokens_lost": lost,
            "leaf_tokens_rescued": rescued, "accepted_total": accepted,
            "mean_accepted": accepted / n, "mean_emitted_unclipped": 1 + accepted / n,
            "emitted_ratio_to_chain7": (accepted + n) / (baseline + n),
            "step_time_reduction_needed_for_tie": 1 - (accepted + n) / (baseline + n),
            "current_tree_kda_full_state_passes": passes}


def enumerate_shapes(histogram, gains):
    all_shapes = []
    for length in range(1, K + 1):
        for counts in itertools.product(range(1, 5), repeat=length):
            if sum(counts) <= K:
                all_shapes.append(shape_stats(counts, histogram, gains))
    best = []
    for length in range(2, K + 1):
        exact8 = [s for s in all_shapes if s["spine"] == length and s["verification_rows"] == 8]
        best.append(max(exact8, key=lambda s: (s["accepted_total"],
                    -s["current_tree_kda_full_state_passes"])))
    chain = next(s for s in all_shapes if s["spine"] == 7)
    return {"shape_count": len(all_shapes), "baseline_chain7": chain,
            "best_fixed_8row_by_spine_length": best,
            "any_fixed_shape_strictly_beats_chain7_acceptance":
                any(s["accepted_total"] > chain["accepted_total"] for s in all_shapes)}


def oracle_gains(histogram):
    # One or more perfect leaf alternatives rescue at most one token at the
    # first rejection. Extra leaves at that depth cannot rescue two tokens.
    return {p: [0] + [histogram[p]] * 3 for p in range(K)}


def main():
    current_path = ROOT / "measurements/st_draft_sensitivity_20260916/live/rank0-validation.json"
    current = json.loads(current_path.read_text())["cases"]
    prefixes = []
    for case in current:
        assert case["k"] == K and len(case["target"]) == len(case["baseline_drafts"]) == K
        prefixes.append(next((p for p, pair in enumerate(zip(case["baseline_drafts"], case["target"]))
                              if pair[0] != pair[1]), K))
    current_hist = Counter(prefixes)
    assert len(current) == 16 and sum(prefixes) == 46

    old_path = ROOT / "measurements/st_draft_rank_overlap_20260915/rank-rows.jsonl.gz"
    with gzip.open(old_path, "rt") as source:
        raw = [json.loads(line) for line in source]
    old = [r for r in raw if r["reason"] != "output_boundary"]
    assert len(raw) == 6291 and len(old) == 6285
    assert all(r["draft_width"] == K and 0 <= r["accepted_prefix"] <= K for r in old)
    old_hist = Counter(r["accepted_prefix"] for r in old)
    unary_gains = {p: [0, 0, 0, 0] for p in range(K)}
    for row in old:
        if row["reason"] != "selector_miss":
            continue
        siblings = [r for r in range(16) if r != row["draft_rank"]]
        for width in range(1, 4):
            unary_gains[row["accepted_prefix"]][width] += row["target_rank"] in siblings[:width]

    report = {
        "scope": "Fixed-baseline-state arithmetic; no tree GPU execution, timing, or deployment verdict",
        "assumptions": ["Unchanged baseline spine; alternatives have no descendants",
                        "Static shape shared by all cases, at most 7 draft nodes plus anchor",
                        "Exact target verification and no EOS/output-budget clipping",
                        "Oracle may choose a correct leaf per case: unattainable upper bound",
                        "Captures are different builds/workloads and are not pooled"],
        "capture_20260916": {"source": identity(current_path), "cases": len(current),
            "requests": len({c["request_key"] for c in current}),
            "prefixes": prefixes, "prefix_histogram": dict(sorted(current_hist.items())),
            "perfect_leaf_upper_bound": enumerate_shapes(current_hist, oracle_gains(current_hist))},
        "rank_capture_20260915": {"source": identity(old_path), "raw_steps": len(raw),
            "eligible_steps": len(old), "excluded_output_boundaries": len(raw) - len(old),
            "prefix_histogram": dict(sorted(old_hist.items())),
            "perfect_leaf_upper_bound": enumerate_shapes(old_hist, oracle_gains(old_hist)),
            "recorded_unary_sibling_policy": enumerate_shapes(old_hist, unary_gains),
            "policy_limits": "Unary rank excludes recorded walk pick; synchronous rows used a rank-1 placeholder. Not bilinear alternative ranking."},
    }
    # Check exact integer arithmetic for the two principal claims.
    cur4 = report["capture_20260916"]["perfect_leaf_upper_bound"]["best_fixed_8row_by_spine_length"][2]
    assert cur4["spine"] == 4 and cur4["accepted_total"] == 41
    assert cur4["tail_tokens_lost"] == 15 and cur4["leaf_tokens_rescued"] == 10
    old6 = report["rank_capture_20260915"]["recorded_unary_sibling_policy"]["best_fixed_8row_by_spine_length"][4]
    assert old6["spine"] == 6 and old6["tail_tokens_lost"] == 1738
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
