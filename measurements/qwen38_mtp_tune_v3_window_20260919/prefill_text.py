#!/usr/bin/env python3
"""Rendered conversations (render_all.py) prefilled as raw text: each POSTed to /v1/completions with max_tokens 1, so the
tap records the target's streams at every computed position. Sent in the order of their opening text, so texts that
share a system prompt and tools follow each other and the prefix cache keeps the shared part computed once.

    prefill_text.py URL LABEL TEXTS.jsonl [CONCURRENCY=6] > requests.jsonl
"""
import json
import sys
import threading
import time
import urllib.request

BASE, LABEL, PATH = sys.argv[1], sys.argv[2], sys.argv[3]
CONCURRENCY = int(sys.argv[4]) if len(sys.argv) > 4 else 6
rows = sorted((json.loads(line) for line in open(PATH)), key=lambda r: r["text"][:2000])


def wait_door():
    began = time.time()
    while True:
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=3) as r:
                return json.loads(r.read())["data"][0]["id"], round(time.time() - began, 1)
        except Exception:                                            # noqa: BLE001
            time.sleep(5)


model, waited = wait_door()
print(json.dumps({"label": LABEL, "door_open_after_s": waited, "model": model, "jobs": len(rows)}), flush=True)
lock, state = threading.Lock(), {"next": 0, "done": 0, "errors": 0, "prompt_tokens": 0, "cached_tokens": 0}


def feed():
    while True:
        with lock:
            i = state["next"]
            state["next"] += 1
        if i >= len(rows):
            return
        body = {"model": model, "prompt": rows[i]["text"], "max_tokens": 1, "temperature": 0}
        try:
            req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                usage = json.loads(r.read()).get("usage", {})
            with lock:
                state["done"] += 1
                state["prompt_tokens"] += usage.get("prompt_tokens", 0)
                state["cached_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        except Exception as exc:                                     # noqa: BLE001
            with lock:
                state["errors"] += 1
            print(json.dumps({"label": LABEL, "cid": rows[i]["cid"], "error": repr(exc)[:300]}), flush=True)


threads = [threading.Thread(target=feed, daemon=True) for _ in range(CONCURRENCY)]
for t in threads:
    t.start()
began = time.time()
while any(t.is_alive() for t in threads):
    time.sleep(30)
    with lock:
        print(json.dumps({"label": LABEL, "elapsed_s": round(time.time() - began), **state}), flush=True)
print(json.dumps({"label": LABEL, "final": state, "elapsed_s": round(time.time() - began)}), flush=True)
