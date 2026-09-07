"""Reproduce the serving comparison; keep raw records unchanged.

The first two records predate the SPEC_K alias fingerprint fix. Normalize
only that known default, after checking the archived boot arguments, engine
counters and deployed profile. No timings or quality/proof outcomes change.
"""
import copy
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
raw = [json.loads(line) for line in (HERE / "records.raw.jsonl").read_text().splitlines()]
assert [r["name"] for r in raw] == ["M8SERVB1", "M8SERVB2", "M8SERVA1", "M8SERVA2", "M8SERVB3"]
assert "SPEC_K=5" in (HERE / "deployed-profile.txt").read_text().splitlines()
excluded = raw[-1]
excluded_log = (HERE / "boot-M8SERVB3.log").read_text()
assert "Running: 2 reqs" in excluded_log
# Non-benchmark requests overlap its 128K phase. Global step/acceptance
# counters are contaminated; exclude the whole arm from the primary pair.
rows = copy.deepcopy(raw[:-1])
for r in rows:
    assert r["overlay"] == "1af1a93f83b5"
    assert r["decode"]["num_spec"] == 5
    assert r["quality"] == {"ok": 9, "total": 9}
    assert r["korean"]["dirty"] == 0 and r["korean"]["n"] == 5
    assert r["workload"] == raw[0]["workload"]
    log = (HERE / f"boot-{r['name']}.log").read_text()
    assert "'num_speculative_tokens': 5" in log
    assert "'tensor_parallel_size': 4" in log
    if r["name"] in ("M8SERVB1", "M8SERVB2"):
        assert r["knobs"] == {"VLLM_GLM53_SPEC_K": "5"}
        r["metadata_correction"] = {
            "original_knobs": r["knobs"],
            "reason": "Launcher alias matches deployed SPEC_K=5, boot args and measured engine counters",
            "fix_commit": "47f9e3c",
        }
        r["knobs"] = {}
        r["proof"].pop("VLLM_GLM53_SPEC_K", None)
    if "M8SERVA" in r["name"]:
        assert r["knobs"] == {"VLLM_GLM53_MK_FP8_PACK2": "1", "VLLM_GLM53_MK_GEMM_TRANSPOSE_M8": "2"}
        assert r["proof_ok"] == "2/2" and all(r["proof"].values())
        assert "N=4096 K=2048 compact=True" in log
        assert "N=6144 K=4096 compact=True" in log
    else:
        assert r["knobs"] == {}

(HERE / "records.normalized.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
per_run = []
for r in rows:
    entry = {"name": r["name"], "step_s": r["decode"]["windows_med"],
             "tokens_per_step": r["decode"]["tokens_per_step"], "by_ctx": {}}
    for ctx in (2000, 32000, 128000):
        req = [q for q in r["requests"] if q["ctx"] == ctx]
        assert req and all(q["decode_tok_s"] > 0 for q in req)
        entry["by_ctx"][str(ctx)] = {key: median(q[key] for q in req)
                                     for key in ("decode_tok_s", "tpot_ms")}
        entry["by_ctx"][str(ctx)]["completion_tokens"] = [q["completion_tokens"] for q in req]
    per_run.append(entry)

comparison = {}
for ctx in ("2000", "32000", "128000"):
    by_arm = {arm: [r["by_ctx"][ctx] for r in per_run if r["name"].startswith("M8SERV" + arm)]
              for arm in ("A", "B")}
    stats = {arm: {key: median(q[key] for q in reqs) for key in ("decode_tok_s", "tpot_ms")}
             for arm, reqs in by_arm.items()}
    bases = [q["decode_tok_s"] for q in by_arm["B"]]
    stats["speed_change_pct"] = 100 * (stats["A"]["decode_tok_s"] / stats["B"]["decode_tok_s"] - 1)
    stats["baseline_spread_pct"] = 100 * (max(bases) - min(bases)) / median(bases)
    comparison[ctx] = stats
window_comparison = {}
for ctx in ("2000", "32000", "128000", "all"):
    stats = {}
    for arm in ("B", "A"):
        windows = [r["decode"]["windows"] if ctx == "all" else r["decode"]["windows_by_ctx"][ctx]
                   for r in rows if r["name"].startswith("M8SERV" + arm)]
        stats[arm] = {"step_s": median(median(w) for w in windows),
                      "windows": sum(len(w) for w in windows), "boots": 1}
    stats["step_change_pct"] = 100 * (stats["A"]["step_s"] / stats["B"]["step_s"] - 1)
    window_comparison[ctx] = stats
(HERE / "summary.json").write_text(json.dumps({"per_run": per_run, "comparison": comparison,
    "window_comparison": window_comparison,
    "assessment": {"status": "inconclusive", "candidate": "retained", "default_promotion": "deferred",
                   "reason": "Too few windows, one clean boot per arm, acceptance variation and missing clean reversal"},
    "excluded": {"name": excluded["name"], "reason": "External requests overlapped 128K; Running: 2 reqs"}}, indent=2) + "\n")
for ctx, s in comparison.items():
    print(f"{ctx}: {s['B']['decode_tok_s']:.3f} -> {s['A']['decode_tok_s']:.3f} tok/s "
          f"({s['speed_change_pct']:+.2f}%), {s['B']['tpot_ms']:.3f} -> {s['A']['tpot_ms']:.3f} ms/token; "
          f"baseline spread {s['baseline_spread_pct']:.2f}%")
