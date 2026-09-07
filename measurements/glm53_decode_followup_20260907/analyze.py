"""Summarize fixed-workload ABBA; each boot, not each window, is a replicate."""
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
NAMES = ("MHCBF16LB1", "MHCBF16LA1", "MHCBF16LA2", "MHCBF16LB2")
KNOBS = {"VLLM_GLM53_MK_FP8_PACK2": "1", "VLLM_GLM53_MK_GEMM_TRANSPOSE_M8": "2",
         "VLLM_GLM53_MK_M8_FASTPATH": "1", "VLLM_GLM53_MK_MHC_BF16": "1"}
BASE_KNOBS = {"VLLM_GLM53_FP8_CACHE": "0"}


def summarize(rows):
    assert [r["name"] for r in rows] == list(NAMES), "need four distinct, complete boots in ABBA order"
    assert len({r["boot_id"] for r in rows}) == 4, "each arm must be an independent boot"
    out = []
    for r in rows:
        log = (HERE / ("boot-" + r["name"] + ".log")).read_text()
        assert "source md5=554fa17b" in log
        assert "'num_speculative_tokens': 5" in log and "'tensor_parallel_size': 4" in log
        assert not r.get("rehearsal") and not r.get("evidence_issues")
        assert r["overlay"] == rows[0]["overlay"]
        assert r["endpoint"] == {"completion": "http://127.0.0.1:18000/v1/chat/completions", "metrics": "http://127.0.0.1:18000/metrics"}
        assert all(r.get(k) == rows[0].get(k) for k in ("workload", "runtime", "harness", "thinking", "doc_lang"))
        assert r["quality"]["total"] == 24 and r["korean"]["n"] == 10
        quality_issues = []
        if r["quality"]["ok"] != r["quality"]["total"]:
            quality_issues.append("retrieval quality failed")
        if r["korean"]["dirty"]:
            quality_issues.append(f"Korean corruption {r['korean']['dirty']}/{r['korean']['n']}")
        assert not r["traffic"]["issues"]
        assert r["traffic"]["after"]["finished"] - r["traffic"]["before"]["finished"] == 10
        arm = "A" if r["name"].startswith("MHCBF16LA") else "B"
        assert r["knobs"] == ({**BASE_KNOBS, **KNOBS} if arm == "A" else BASE_KNOBS)
        if arm == "A":
            assert r["proof_ok"] == "4/4" and all(r["proof"].get(k) is True for k in KNOBS)
            assert "m8-fastpath CAPTURED M=6" in log
            assert "mhc-bf16 CAPTURED T=6" in log
            assert "selftest mhc-bf16 all outputs bit equal -> ARM" in log
            assert "N=4096 K=2048 compact=True" in log and "N=6144 K=4096 compact=True" in log
        else:
            assert "m8-fastpath CAPTURED" not in log
            assert "mhc-bf16 CAPTURED" not in log
        d = r["decode"]
        assert d["primary"] == "fixed-2K" and d["num_spec"] == 5
        windows = d["fixed_intervals"]
        assert len(windows) >= 20 and all(w["seconds"] > 0 and w["steps"] >= 0 for w in windows)
        rate = sum(w["steps"] for w in windows) / sum(w["seconds"] for w in windows)
        assert math.isclose(rate, d["fixed_pooled_step_s"])
        reqs = [q for q in r["requests"] if q.get("fixed_decode")]
        assert len(reqs) == 5 and all(q["completion_tokens"] == 2048 for q in reqs)
        first_reqs = [q for q in rows[0]["requests"] if q.get("fixed_decode")]
        assert [(q["request_sha256"], q["prompt_tokens"], q["seed"]) for q in reqs] == [
                (q["request_sha256"], q["prompt_tokens"], q["seed"]) for q in first_reqs]
        decoded = sum(q["completion_tokens"] - 1 for q in reqs)
        duration = sum(q["decode_s"] for q in reqs)
        gaps = [g for q in reqs for g in q["chunk_gaps_ms"]]
        out.append({"name": r["name"], "arm": arm, "windows": len(windows),
                    "median_chunk_gap_ms": median(gaps),
                    "quality": r["quality"], "korean": r["korean"], "quality_issues": quality_issues,
                    "step_s": rate, "window_median_step_s": d["windows_med"],
                    "tok_s": median(q["decode_tok_s"] for q in reqs),
                    "tpot_ms": median(q["tpot_ms"] for q in reqs),
                    "pooled_tok_s": decoded / duration,
                    "pooled_tpot_ms": 1000 * duration / decoded,
                    "request_tok_s": [q["decode_tok_s"] for q in reqs],
                    "acceptance_all_requests": d["acc_raw"],
                    "tokens_per_step_all_requests": d["tokens_per_step"]})
    metrics = ("step_s", "tok_s", "tpot_ms", "pooled_tok_s", "pooled_tpot_ms")
    stats = {arm: {key: median(r[key] for r in out if r["arm"] == arm)
                   for key in metrics} for arm in ("B", "A")}
    paired = [{"candidate": out[a]["name"], "baseline": out[b]["name"],
               **{key + "_change_pct": 100 * (out[a][key] / out[b][key] - 1)
                  for key in ("step_s", "tok_s", "pooled_tok_s")}} for b, a in ((0, 1), (3, 2))]
    spread = {}
    for arm in ("B", "A"):
        spread[arm] = {}
        for key in ("step_s", "tok_s", "pooled_tok_s"):
            values = [r[key] for r in out if r["arm"] == arm]
            spread[arm][key] = 100 * (max(values) - min(values)) / median(values)
    output_matches = []
    for i, j in ((0, 3), (1, 2), (0, 1), (3, 2)):
        left = [q for q in rows[i]["requests"] if q.get("fixed_decode")]
        right = [q for q in rows[j]["requests"] if q.get("fixed_decode")]
        output_matches.append({"left": rows[i]["name"], "right": rows[j]["name"],
                               "same_output": sum(a["output_sha256"] == b["output_sha256"]
                                                  for a, b in zip(left, right)), "requests": 5})
    quality_pass = not any(r["quality_issues"] for r in out)
    return {"per_boot": out, "arms": stats, "paired": paired,
            "all_quality_gates_passed": quality_pass,
            "promotion_blockers": [] if quality_pass else ["Quality guard failed; timings remain descriptive only."],
            "change_pct": {key: 100 * (stats["A"][key] / stats["B"][key] - 1)
                           for key in metrics},
            "within_arm_boot_spread_pct": spread,
            "fixed_output_matches": output_matches,
            "independent_boots_per_arm": 2,
            "assessment": {"status": "inconclusive", "candidate": "retained", "default_promotion": "deferred",
                           "reason": "Engine-step pairs are positive, but pooled output pairs disagree, output gain is below baseline variation, and baseline quality guards failed."},
            "runtime_baseline_knobs": BASE_KNOBS,
            "candidate_knobs": KNOBS,
            "baseline_scope": "Matched runtime baseline with disk FP8 cache disabled; not a reusable profile-default baseline",
            "note": "Windows within one boot are correlated; two boots per arm do not support a narrow confidence interval."}


if __name__ == "__main__":
    rows = [json.loads(line) for line in (HERE / "records.raw.jsonl").read_text().splitlines()]
    result = summarize(rows)
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
