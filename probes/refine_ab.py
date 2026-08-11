#!/usr/bin/env python3
"""REFINE_PASS A/B harness. Snapshots spec-decode counters around a fixed
tg128 workload to get BOTH tok/s and accepted-tokens-per-step for the same
run — so acceptance gain and latency cost are read on identical work.
Run once with REFINE=0 (baseline), once with REFINE=1, diff the JSON."""
import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
URL = BASE + "/v1/chat/completions"
MODEL = "deepseek-v4-flash"


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
        txt = r.read().decode()
    out = {}
    for key in ["num_drafts", "num_draft_tokens", "num_accepted_tokens",
                "num_generation_tokens", "iteration_tokens_total"]:
        m = re.search(rf'vllm:spec_decode_{key}_total\{{[^}}]*\}}\s+([\d.eE+]+)', txt)
        if not m:
            m = re.search(rf'vllm:{key}_total\{{[^}}]*\}}\s+([\d.eE+]+)', txt)
        out[key] = float(m.group(1)) if m else 0.0
    # per-position accepted
    pos = {}
    for m in re.finditer(
            r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([\d.eE+]+)', txt):
        pos[int(m.group(1))] = float(m.group(2))
    out["per_pos"] = pos
    return out


def one(max_tok, seed):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Write a long detailed story. ({seed})"}],
        "max_tokens": max_tok, "min_tokens": max_tok, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0.8, "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    tfirst = None
    tlast = time.time()
    ct = 0          # streamed deltas (content + reasoning), for timing
    usage_ct = 0    # authoritative completion_tokens from usage
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            p = line[6:]
            if p == "[DONE]":
                break
            ev = json.loads(p)
            u = ev.get("usage")
            if u and u.get("completion_tokens"):
                usage_ct = u["completion_tokens"]
            ch = ev.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if tfirst is None:
                        tfirst = time.time()
                    tlast = time.time()
                    ct += 1
    gen = tlast - (tfirst or tlast)
    toks = usage_ct or ct  # prefer authoritative usage count
    return toks, ((toks - 1) / gen if gen > 0 else 0.0)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    # warm
    one(64, 0)
    m0 = metrics()
    rates = []
    t0 = time.time()
    for r in range(reps):
        ct, rate = one(128, r + 1)
        rates.append(rate)
        print(f"  rep{r+1}: {ct} tok {rate:6.1f} tok/s", flush=True)
    wall = time.time() - t0
    m1 = metrics()
    d_draft = m1["num_drafts"] - m0["num_drafts"]
    d_dtok = m1["num_draft_tokens"] - m0["num_draft_tokens"]
    d_acc = m1["num_accepted_tokens"] - m0["num_accepted_tokens"]
    rates.sort()
    n = len(rates)
    acc_per_step = d_acc / d_draft if d_draft > 0 else 0
    acc_rate = d_acc / d_dtok if d_dtok > 0 else 0
    dpos = {k: m1["per_pos"].get(k, 0) - m0["per_pos"].get(k, 0)
            for k in sorted(m1["per_pos"])}
    posrate = {k: (dpos[k] / d_draft if d_draft > 0 else 0) for k in dpos}
    res = {
        "tag": tag,
        "tokps_mean": round(sum(rates) / n, 2),
        "tokps_median": round(rates[n // 2], 2),
        "tokps_min": round(rates[0], 2),
        "tokps_max": round(rates[-1], 2),
        "accepted_per_step": round(acc_per_step, 4),
        "accept_rate": round(acc_rate, 4),
        "draft_sets": int(d_draft),
        "draft_tokens": int(d_dtok),
        "accepted_tokens": int(d_acc),
        "per_pos_acceptrate": {k: round(v, 3) for k, v in posrate.items()},
        "wall_s": round(wall, 1),
    }
    print("RESULT " + json.dumps(res))
