"""Verify all six tile experiments and summarize them without filtering runs."""
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = digest(HERE / "baseline-256.py")
    assert digest(ROOT / "engine/kernels/state.py") == baseline, "rejected tiles reached production"
    cases = []
    for tile, prefix in ((1024, "tiles"), (512, "tiles512")):
        candidate = digest(HERE / f"candidate-{tile}.py")
        groups = {}
        for run in (1, 2, 3):
            name = f"{prefix}-{run}.json"
            report = json.loads((HERE / name).read_text())
            assert report["baseline_sha256"] == baseline, name
            assert report["state_sha256"] == candidate, name
            assert len(report["results"]) == 4, name
            assert {(c["kda_layers"], c["active_sequences"]) for c in report["results"]} == {
                (1, 1), (1, 4), (34, 1), (34, 4)}
            for c in report["results"]:
                assert c["exact"] is True
                assert set(c["writes"]) == {"1", "6"}
                for operation, timing in [("history", c["history"]), *c["writes"].items()]:
                    for variant in ("baseline", "optimized"):
                        t = timing[variant]
                        assert len(t["samples_us"]) == 5
                        assert all(len(r) == 30 and all(v > 0 for v in r) for r in t["samples_us"])
                        assert t["median_us"] == statistics.median(sum(t["samples_us"], []))
                    a, b = [timing[v]["median_us"] for v in ("baseline", "optimized")]
                    key = c["kda_layers"], c["active_sequences"], operation
                    groups.setdefault(key, []).append(dict(file=name, baseline_us=a, candidate_us=b,
                                                          reduction_percent=100 * (1 - b / a)))
        for (layers, n, operation), runs in groups.items():
            reduction = [r["reduction_percent"] for r in runs]
            cases.append(dict(tile=tile, kda_layers=layers, active_sequences=n, operation=operation,
                              reduction_range_percent=[min(reduction), max(reduction)], runs=runs))
    gpu = (HERE / "gpu-tests.log").read_text()
    assert re.search(r"Ran 170 tests in .*\n\nOK\n", gpu), "GPU suite failed, incomplete or skipped"
    lines = (HERE / "graph-check.log").read_text().splitlines()
    checks = [s for s in lines if s.startswith(("case ", "rejected-future "))]
    assert len(checks) == 17 and "PASS" in lines
    assert all("relative 0.0 state_exact True paged_exact True" in s for s in checks)
    runtime = json.loads((HERE / "runtime-final.json").read_text())
    assert runtime["passed"] and not runtime["vllm_present"]
    for path, sha in runtime["source_files"].items():
        assert digest(ROOT / "engine" / path) == sha, path
    result = dict(decision="retain main's 256-element tile; neither candidate promoted",
                  scope="direct-state transfers only; no full-model or TP4 speedup claim",
                  gpu_tests=170, isolated_real_weight_graph_conditions=17, cases=cases)
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print(case["tile"], case["kda_layers"], case["active_sequences"], case["operation"],
              [round(v, 2) for v in case["reduction_range_percent"]])


if __name__ == "__main__":
    main()
