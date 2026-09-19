#!/usr/bin/env python3
"""The rendered tool conversations (render_tools.py) prefilled as raw text beside the window's data3c boot: waits for
window.log to open that boot and its door to answer, then POSTs each text to /v1/completions (max_tokens 1) at
concurrency 2 -- the chat path answers tools with 400 on this door (no grammar compiler bound).

    side_prefill.py WINDOW_LOG RENDERED.jsonl URL > side-requests.jsonl
"""
import json
import sys
import threading
import time
import urllib.request

LOG, RENDERED, BASE = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(line) for line in open(RENDERED)]

while "== boot data3c" not in open(LOG).read():
    time.sleep(5)
while True:
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=3) as r:
            model = json.loads(r.read())["data"][0]["id"]
        break
    except Exception:                                                # noqa: BLE001
        time.sleep(5)
print(json.dumps({"door": "up", "model": model, "jobs": len(rows)}), flush=True)
lock, state = threading.Lock(), {"next": 0, "done": 0, "errors": 0, "prompt_tokens": 0}


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
            with urllib.request.urlopen(req, timeout=900) as r:
                usage = json.loads(r.read()).get("usage", {})
            with lock:
                state["done"] += 1
                state["prompt_tokens"] += usage.get("prompt_tokens", 0)
        except Exception as exc:                                     # noqa: BLE001
            with lock:
                state["errors"] += 1
            print(json.dumps({"cid": rows[i]["cid"], "error": repr(exc)[:300]}), flush=True)


threads = [threading.Thread(target=feed, daemon=True) for _ in range(2)]
for t in threads:
    t.start()
began = time.time()
while any(t.is_alive() for t in threads):
    time.sleep(20)
    with lock:
        print(json.dumps({"elapsed_s": round(time.time() - began), **state}), flush=True)
print(json.dumps({"final": state, "elapsed_s": round(time.time() - began)}), flush=True)
