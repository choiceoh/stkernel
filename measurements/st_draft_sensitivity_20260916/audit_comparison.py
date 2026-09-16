#!/usr/bin/env python3
"""Recompute fixed-state prefix and timing results from the retained raw data."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "comparison"


def read(path):
    return json.loads(path.read_text())


def prefix(proposal, target):
    assert len(proposal) == len(target) == 7
    return next((i for i, (a, b) in enumerate(zip(proposal, target)) if a != b), 7)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9), (a, b)


def main():
    validations = [read(ROOT / f"live/rank{rank}-validation.json") for rank in range(4)]
    cases = validations[0]["cases"]
    baseline = [prefix(c["baseline_drafts"], c["target"]) for c in cases]
    assert len(cases) == 16 and len({c["request_key"] for c in cases}) == 8
    for validation in validations[1:]:
        assert [(c["case_id"], c["request_key"], c["target"], c["baseline_drafts"])
                for c in validation["cases"]] == [
                    (c["case_id"], c["request_key"], c["target"], c["baseline_drafts"])
                    for c in cases]
    runtimes = [read(OUT / f"{arm}-runtime.json") for arm in ("fp8", "bf16")]
    for key in ("revision", "engine_revision", "env", "rounds", "capture", "checkpoint"):
        assert runtimes[0][key] == runtimes[1][key], key
    assert [r["image"] for r in runtimes[0]["runtime"]] == [r["image"] for r in runtimes[1]["runtime"]]
    assert all(r["engine_booted"] is False for r in runtimes)
    summary = dict(
        scope="fixed-state C1 greedy proposal sensitivity; no live tok/s or output quality verdict",
        controller_revision=runtimes[0]["revision"], engine_revision=runtimes[0]["engine_revision"],
        cases=16, requests=8, baseline_prefixes=baseline,
        baseline_mean_prefix=statistics.mean(baseline), arms={})
    source_hashes = set()
    for arm, runtime in zip(("fp8", "bf16"), runtimes):
        raw = read(OUT / f"{arm}.json")
        assert raw["precision"] == runtime["precision"]
        assert raw["rank_state_sha256"] == [v["state_sha256"] for v in validations]
        source_hashes.add(raw["replay_source_sha256"])
        assert [r["reader"] for r in raw["results"]] == validations[0]["readers"]
        assert len(raw["results"]) == 30
        rows = []
        changes = dict(before_first_rejection=0, at_first_rejection=0, after_first_rejection=0)
        for result in raw["results"]:
            assert result["cases"] == result["labels_complete"] == 16
            assert result["requests"] == 8
            assert len(result["proposals"]) == len(result["timing"]) == 16
            prefixes = [prefix(p, c["target"]) for p, c in zip(result["proposals"], cases)]
            gains = [a - b for a, b in zip(prefixes, baseline)]
            for key, value in (("baseline_prefix_bounds", statistics.mean(baseline)),
                               ("candidate_prefix_bounds", statistics.mean(prefixes)),
                               ("prefix_gain_bounds", statistics.mean(gains))):
                assert result[key] == [value, value]
            for key, values in (("prefix_survival_baseline", baseline),
                                ("prefix_survival_candidate", prefixes)):
                assert result[key] == [[sum(p >= i for p in values) / 16] * 2 for i in range(1, 8)]
            changed = 0
            for proposal, case, accepted in zip(result["proposals"], cases, baseline):
                for i, (a, b) in enumerate(zip(proposal, case["baseline_drafts"])):
                    if a != b:
                        changed += 1
                        key = ("before_first_rejection" if i < accepted else
                               "at_first_rejection" if i == accepted else "after_first_rejection")
                        changes[key] += 1
            assert changed == result["changed_draft_positions"]
            for pair in result["timing"]:
                for side in ("baseline", "candidate"):
                    samples = pair[f"{side}_samples_us"]
                    assert len(samples) == 2 * runtime["rounds"]
                    assert all(math.isfinite(x) and x > 0 for x in samples)
                    close(pair[f"{side}_us"], statistics.median(samples))
            delta = statistics.mean(t["candidate_us"] - t["baseline_us"] for t in result["timing"])
            close(delta, result["proposal_delta_us"])
            assert len(result["active_weight_byte_delta_per_rank"]) == 4
            rows.append(dict(reader=result["reader"], prefix_by_case=prefixes, gain_by_case=gains,
                             prefix_gain=statistics.mean(gains), proposal_delta_us=delta,
                             changed_draft_positions=changed,
                             active_weight_byte_delta_per_rank=result["active_weight_byte_delta_per_rank"],
                             baseline_mean_proposal_us=statistics.mean(t["baseline_us"] for t in result["timing"])))
        gains = [g for r in rows for g in r["gain_by_case"]]
        summary["arms"][arm] = dict(
            readers=30, paired_case_comparisons=len(gains),
            improved=sum(g > 0 for g in gains), regressed=sum(g < 0 for g in gains),
            unchanged=sum(g == 0 for g in gains), draft_changes=changes,
            proposal_delta_us_range=[min(r["proposal_delta_us"] for r in rows), max(r["proposal_delta_us"] for r in rows)],
            baseline_dominates_point_estimates=all(r["prefix_gain"] <= 0 and r["proposal_delta_us"] > 0
                and all(b > 0 for b in r["active_weight_byte_delta_per_rank"]) for r in rows),
            results=rows)
        fleet_log = (OUT / f"{arm}-fleet.log").read_text()
        assert "TP4 replay complete:" in fleet_log and '"reason": "release"' in fleet_log
    assert len(source_hashes) == 1
    summary["replay_source_sha256"] = source_hashes.pop()
    summary["raw_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ("summary.json", "README.md")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"verified": True, "cases": len(cases), "arms": {
        a: {k: v for k, v in d.items() if k != "results"} for a, d in summary["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
