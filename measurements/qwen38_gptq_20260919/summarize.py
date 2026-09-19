"""Recompute public GPTQ experiment summaries without accessing private text."""
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def projection_summary():
    rows, ranks = [], []
    for rank in range(4):
        fit = read(ROOT / f"fit-audit-rank{rank}.json")
        held = read(ROOT / f"heldout-audit-rank{rank}.json")
        boot = read(ROOT / "compare" / f"Bpack-audit-rank{rank}.json")
        score = read(ROOT / "compare" / f"projection-rank{rank}.json")
        assert fit["weights_id"] == held["weights_id"] == boot["weights_id"] == score["weights_id"]
        assert boot["serving_gptq_verified"] and boot["w4_sites"] == 192 and boot["fp8_sites"] == 193
        assert fit["minimum_rows"] == fit["maximum_rows"] == 131184
        assert held["minimum_rows"] == held["maximum_rows"] == 50512
        assert len(score["cases"]) == len({(r["name"], r["lane"]) for r in score["cases"]}) == 385
        assert all(r["heldout_rows"] == 50512 for r in score["cases"])
        rows.extend(dict(r, rank=rank) for r in score["cases"])
        ranks.append(dict(rank=rank, **score["summary"]))
    lanes = {}
    for lane in ("w4", "fp8"):
        cases = [r for r in rows if r["lane"] == lane]
        ratios = [r["gptq"]["relative_rmse"] / r["rtn"]["relative_rmse"] for r in cases]
        lanes[lane] = dict(
            sites=len(cases), improved=sum(x < 1 for x in ratios), worsened=sum(x > 1 for x in ratios),
            median_rtn_rmse=statistics.median(r["rtn"]["relative_rmse"] for r in cases),
            median_gptq_rmse=statistics.median(r["gptq"]["relative_rmse"] for r in cases),
            median_paired_rmse_ratio=statistics.median(ratios),
            minimum_paired_rmse_ratio=min(ratios), maximum_paired_rmse_ratio=max(ratios))
    head = [dict(rank=r["rank"], rtn=r["rtn"]["relative_rmse"], gptq=r["gptq"]["relative_rmse"])
            for r in rows if r["key"] == "head"]
    return dict(scope="Weight-packing projection error on held-out real-input Gram statistics; not whole-model accuracy.",
                lanes=lanes, ranks=ranks, head=head)


def main():
    result = projection_summary()
    (ROOT / "projection-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
