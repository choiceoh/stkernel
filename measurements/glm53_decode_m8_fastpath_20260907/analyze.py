"""Summarize fixed-workload ABBA; each boot, not each window, is a replicate."""
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
NAMES = ("M8NEXTB1", "M8NEXTA1", "M8NEXTA2", "M8NEXTB2")
KNOBS = {"VLLM_GLM53_MK_FP8_PACK2": "1", "VLLM_GLM53_MK_GEMM_TRANSPOSE_M8": "2",
         "VLLM_GLM53_MK_M8_FASTPATH": "1"}


def summarize(rows):
    assert [r["name"] for r in rows] == list(NAMES), "need four distinct, complete boots in ABBA order"
    out = []
    for r in rows:
        assert not r.get("rehearsal") and not r.get("evidence_issues")
        assert r["overlay"] == rows[0]["overlay"]
        assert all(r.get(k) == rows[0].get(k) for k in ("workload", "runtime", "harness", "thinking", "doc_lang"))
        assert r["quality"] == {"ok": 18, "total": 18}
        assert r["korean"]["dirty"] == 0 and r["korean"]["n"] == 8
        assert not r["traffic"]["issues"]
        assert r["traffic"]["after"]["finished"] - r["traffic"]["before"]["finished"] == 8
        arm = "A" if r["name"].startswith("M8NEXTA") else "B"
        assert r["knobs"] == (KNOBS if arm == "A" else {})
        if arm == "A":
            assert r["proof_ok"] == "3/3" and all(r["proof"].get(k) is True for k in KNOBS)
        d = r["decode"]
        assert d["primary"] == "fixed-2K" and d["num_spec"] == 5
        windows = d["fixed_intervals"]
        assert len(windows) >= 20 and all(w["seconds"] > 0 and w["steps"] > 0 for w in windows)
        rate = sum(w["steps"] for w in windows) / sum(w["seconds"] for w in windows)
        assert math.isclose(rate, d["fixed_pooled_step_s"])
        reqs = [q for q in r["requests"] if q.get("fixed_decode")]
        assert len(reqs) == 3 and all(q["completion_tokens"] == 2048 for q in reqs)
        out.append({"name": r["name"], "arm": arm, "windows": len(windows),
                    "step_s": rate, "window_median_step_s": d["windows_med"],
                    "tok_s": median(q["decode_tok_s"] for q in reqs),
                    "tpot_ms": median(q["tpot_ms"] for q in reqs),
                    "request_tok_s": [q["decode_tok_s"] for q in reqs],
                    "acceptance_all_requests": d["acc_raw"],
                    "tokens_per_step_all_requests": d["tokens_per_step"]})
    stats = {arm: {key: median(r[key] for r in out if r["arm"] == arm)
                   for key in ("step_s", "tok_s", "tpot_ms")} for arm in ("B", "A")}
    paired = [{"candidate": out[a]["name"], "baseline": out[b]["name"],
               **{key + "_change_pct": 100 * (out[a][key] / out[b][key] - 1)
                  for key in ("step_s", "tok_s")}} for b, a in ((0, 1), (3, 2))]
    spread = {}
    for arm in ("B", "A"):
        spread[arm] = {}
        for key in ("step_s", "tok_s"):
            values = [r[key] for r in out if r["arm"] == arm]
            spread[arm][key] = 100 * (max(values) - min(values)) / median(values)
    return {"per_boot": out, "arms": stats, "paired": paired,
            "change_pct": {key: 100 * (stats["A"][key] / stats["B"][key] - 1)
                           for key in ("step_s", "tok_s", "tpot_ms")},
            "within_arm_boot_spread_pct": spread,
            "independent_boots_per_arm": 2,
            "note": "Windows within one boot are correlated; two boots per arm do not support a narrow confidence interval."}


if __name__ == "__main__":
    rows = [json.loads(line) for line in (HERE / "records.raw.jsonl").read_text().splitlines()]
    result = summarize(rows)
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
