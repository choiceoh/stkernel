#!/usr/bin/env python3
"""Co-ingest smoothness probe: stream one decode request; mid-generation,
fire a ~34K prefill alongside. Records the decode stream's inter-token gaps
before / during / after the ingest."""
import json
import os
import random
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4-flash")
WORDS = ["reactor", "harbor", "lattice", "quarry", "ember", "meridian",
         "syntax", "granite", "voltage", "cirrus", "tundra", "beacon"]

ingest_t0 = ingest_t1 = None


def ingest():
    global ingest_t0, ingest_t1
    rng = random.Random(time.time_ns())
    text = " ".join(rng.choice(WORDS) for _ in range(24600))
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text + " End."}],
        "max_tokens": 1,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    ingest_t0 = time.time()
    urllib.request.urlopen(req, timeout=900).read()
    ingest_t1 = time.time()


body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content":
                  "지열 발전의 원리에 대해 1500 토큰 분량으로 아주 자세히 "
                  "설명해줘."}],
    "max_tokens": 1500,
    "temperature": 0.95,
    "chat_template_kwargs": {"thinking": False},
    "stream": True,
}).encode()
req = urllib.request.Request(URL, data=body,
                             headers={"Content-Type": "application/json"})

stamps = []
t_start = time.time()
th = None
with urllib.request.urlopen(req, timeout=900) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        ev = json.loads(line[6:])
        ch = ev.get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content"):
            stamps.append(time.time())
        if th is None and stamps and time.time() - t_start > 6:
            th = threading.Thread(target=ingest, daemon=True)
            th.start()
if th:
    th.join(timeout=300)

gaps = [(stamps[i] - stamps[i - 1], stamps[i]) for i in range(1, len(stamps))]


def rate(lo, hi):
    n = sum(1 for t in stamps if lo <= t <= hi)
    return n / max(hi - lo, 1e-9)


print(f"decode chunks: {len(stamps)}, wall {stamps[-1] - t_start:.1f}s")
if ingest_t0:
    print(f"ingest window: {ingest_t0 - t_start:.1f}s .. "
          f"{ingest_t1 - t_start:.1f}s ({ingest_t1 - ingest_t0:.1f}s)")
    print(f"decode rate BEFORE ingest: {rate(t_start, ingest_t0):6.1f} chunk/s")
    print(f"decode rate DURING ingest: {rate(ingest_t0, ingest_t1):6.1f} chunk/s")
    print(f"decode rate AFTER  ingest: {rate(ingest_t1, stamps[-1]):6.1f} chunk/s")
big = sorted(gaps, reverse=True)[:8]
print("largest token gaps (s @ t):")
for g, t in big:
    print(f"  {g:6.2f}s @ {t - t_start:6.1f}s")
