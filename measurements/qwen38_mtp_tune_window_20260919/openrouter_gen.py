#!/usr/bin/env python3
"""The MTP fine-tune's text from OpenRouter's Qwen3.8 (operator 2026-09-19: "데이터 수집은 오픈라우터 동일모델이랑 같이
하면 되지않나"): the train prompts, a few samples each, at the prompts' own temperature and hotter; thinking where the
prompt asks for it. What the fleet then does with each is a prefill (the target's own streams at every position) --
the labels are the served model's distribution there, so this text is only the prefixes. Stdlib only.

    OPENROUTER_API_KEY=... openrouter_gen.py prompts.jsonl samples.jsonl [SAMPLES=3] [CONCURRENCY=32] [MODEL]
"""
import json
import os
import sys
import threading
import time
import urllib.request

PROMPTS, OUT = sys.argv[1], sys.argv[2]
SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CONCURRENCY = int(sys.argv[4]) if len(sys.argv) > 4 else 32
MODEL = sys.argv[5] if len(sys.argv) > 5 else "qwen/qwen3.8-flash"
KEY = os.environ.get("OPENROUTER_API_KEY")
if not KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set")

prompts = [json.loads(line) for line in open(PROMPTS)]
jobs = [(p, s) for p in prompts if p["split"] == "train" for s in range(SAMPLES)]
done = set()
if os.path.exists(OUT):                                   # a rerun resumes
    for line in open(OUT):
        r = json.loads(line)
        if "content" in r:
            done.add((r["id"], r["sample"]))
jobs = [(p, s) for p, s in jobs if (p["id"], s) not in done]
lock, state = threading.Lock(), {"next": 0, "ok": 0, "failed": 0, "tokens": 0}
out = open(OUT, "a")


def ask(p, sample):
    temperature = (p["temperature"], 0.7, 0.9)[sample % 3]
    body = {"model": MODEL, "messages": [{"role": "user", "content": p["content"]}], "max_tokens": p["max_tokens"],
            "temperature": temperature, "reasoning": {"enabled": bool(p.get("thinking"))}}
    if temperature > 0:
        body["top_p"] = 0.95
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}",
                                  "X-Title": "stkernel mtp fine-tune data"})
    with urllib.request.urlopen(req, timeout=300) as r:
        answer = json.loads(r.read().decode())
    message = answer["choices"][0]["message"]
    return {"id": p["id"], "sample": sample, "prompt": p["content"], "thinking": bool(p.get("thinking")),
            "temperature": temperature, "content": message.get("content") or "", "reasoning": message.get("reasoning") or "",
            "finish": answer["choices"][0].get("finish_reason"), "usage": answer.get("usage", {}),
            "provider": answer.get("provider"), "model": answer.get("model")}


def worker():
    while True:
        with lock:
            i = state["next"]
            state["next"] += 1
        if i >= len(jobs):
            return
        p, sample = jobs[i]
        for attempt in range(3):
            try:
                row = ask(p, sample)
                break
            except Exception as exc:                          # noqa: BLE001 -- rate limits, provider hiccups
                row = {"id": p["id"], "sample": sample, "error": repr(exc)[:200]}
                time.sleep(2 + 4 * attempt)
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if "content" in row:
                state["ok"] += 1
                state["tokens"] += row["usage"].get("completion_tokens", 0)
            else:
                state["failed"] += 1


threads = [threading.Thread(target=worker, daemon=True) for _ in range(CONCURRENCY)]
began = time.time()
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(20)
    with lock:
        print(json.dumps({"elapsed_s": round(time.time() - began), "jobs": len(jobs), **state}), flush=True)
print(json.dumps({"final": state, "elapsed_s": round(time.time() - began)}), flush=True)
