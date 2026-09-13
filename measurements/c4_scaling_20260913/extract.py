"""Compact summary of one onepass run's C=1/C=4 recordings (bench/onepass_recording.py artifacts)."""
import json, sys, os, collections, statistics
root = sys.argv[1]
out = {"run": os.path.basename(root)}
rec = json.load(open(os.path.join(root, "record.json")))
out["git"] = rec.get("git"); out["arm_sha"] = rec.get("arm_sha"); out["image"] = rec.get("image")
out["harness"] = rec.get("harness"); out["generation_budget"] = rec.get("generation_budget")
keep = ("ctx", "question", "client", "ttft_s", "decode_tok_s", "decode_s", "completion_tokens", "prompt_tokens",
        "elapsed_s", "tpot_ms", "finish_reason", "cached_tokens")
out["c1_requests"] = [{k: q.get(k) for k in keep} for q in rec["requests"]]
out["prefill_c1"] = [{k: r.get(k) for k in ("ctx", "tok", "cold_s", "warm_s", "reused_frac", "combined")} for r in rec["prefill"]]
out["decode_c1"] = {k: rec["decode"].get(k) for k in ("gen_tokens", "wall_s", "tokens_per_step", "acc_raw", "num_spec")}
out["c4"] = []
for c in rec.get("c4", []):
    out["c4"].append({"ctx": c["ctx"], "elapsed_s": c["elapsed_s"], "aggregate_output_tok_s": c["aggregate_output_tok_s"],
                      "valid": c.get("valid"), "issues": c.get("issues"), "phase": c.get("latency_artifacts"),
                      "requests": [{k: q.get(k) for k in keep} for q in c["requests"]]})


def rows(path, rank=0):
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("rank", rank) != rank:
                continue
            yield r


phases = {}
for phase in sorted(os.listdir(root)):
    lat = os.path.join(root, phase, "latency.jsonl")
    if not os.path.isfile(lat):
        continue
    host = collections.defaultdict(list); dev = collections.defaultdict(lambda: collections.defaultdict(list))
    waits = collections.defaultdict(list); seq = []; kinds = collections.Counter()
    for r in rows(lat):
        k = r.get("kind"); kinds[k] += 1
        if k == "host_step":
            n = len(r.get("rows") or [])
            host[(r["phase"], n, r["tokens"] if r["phase"] == "prefill" else 0)].append(r["duration_us"])
            seq.append((r["phase"][0], n, r["tokens"]))
        elif k == "device_stage":
            dev[len(r.get("rows") or [])][r["operation"]].append(r["duration_us"])
        elif k == "host_wait":
            waits[len(r.get("rows") or [])].append(r["duration_us"])
    comp = []
    for ph, n, t in seq:
        if comp and comp[-1][0] == ph and comp[-1][1] == n and (ph == "d" or comp[-1][2] == t):
            comp[-1][3] += 1
        else:
            comp.append([ph, n, t, 1])
    entry = {"row_kinds": dict(kinds),
             "host_step_ms": {f"{k[0]}/rows{k[1]}/tok{k[2]}": dict(n=len(v), mean=statistics.mean(v) / 1e3,
                                                                    median=statistics.median(v) / 1e3, max=max(v) / 1e3)
                              for k, v in sorted(host.items())},
             "device_stage_ms_by_rows": {str(n): {op: dict(mean=statistics.mean(v) / 1e3, n=len(v)) for op, v in sorted(ops.items())}
                                         for n, ops in sorted(dev.items())},
             "host_wait_resolve_ms_by_rows": {str(n): dict(n=len(v), mean=statistics.mean(v) / 1e3, median=statistics.median(v) / 1e3)
                                              for n, v in sorted(waits.items())},
             "step_sequence": [f"{p}{n}/{t}x{c}" for p, n, t, c in comp]}
    summ = os.path.join(root, phase, "latency-summary.json")
    if os.path.isfile(summ):
        s = json.load(open(summ))
        ops = [o for o in s["operations"] if o.get("kind") == "gpu_activity" and o.get("rank") == 0]
        byk = collections.defaultdict(lambda: [0.0, 0])
        for o in ops:
            key = (o.get("phase"), (o.get("kernel") or "?")[:110])
            byk[key][0] += o["sum_us"]; byk[key][1] += o["samples"]
        top = sorted(byk.items(), key=lambda x: -x[1][0])[:60]
        if top:
            entry["gpu_kernels_top_ms"] = [dict(phase=k[0], kernel=k[1], ms=v[0] / 1e3, launches=v[1]) for k, v in top]
            entry["gpu_kernels_total_ms"] = {ph: sum(v[0] for k, v in byk.items() if k[0] == ph) / 1e3 for ph in ("prefill", "decode")}
        entry["device_intervals"] = s.get("device_intervals")
    phases[phase] = entry
out["phases"] = phases
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[2], "phases", list(phases))
