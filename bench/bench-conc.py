#!/usr/bin/env python3
"""Concurrency sweep for the hy4 stack: aggregate decode + prefill at C=1,2,4,8.

Single-stream numbers are what we measured all day; an agent workload runs
overlapping requests, so aggregate is the number that actually matters.
Unique random prompts per request (no prefix-cache hits) => conservative.
"""
import json, time, urllib.request, random, os, sys
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8000/v1/chat/completions"
W = ["태양광","발전소","인버터","모듈","효율","계통","연계","전압","주파수","모니터링",
     "데이터","분석","예측","유지보수","진단","출력","손실","온도","일사량","가동률"]

def call(msg, max_tokens, to=900):
    b = json.dumps({"model":"deepseek-v4-flash","messages":[{"role":"user","content":msg}],
                    "max_tokens":max_tokens}).encode()
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(
        urllib.request.Request(BASE, b, {"Content-Type":"application/json"}), timeout=to).read())
    return d["usage"], time.time()-t0

def decode_job(seed):
    r = random.Random(seed)
    tag = " ".join(r.choice(W) for _ in range(5))   # unique -> no cache hit
    u, el = call(f"[{tag}] 1부터 30까지 각 숫자가 소수인지 한 줄씩 설명하라.", 500)
    return u["completion_tokens"], el

def prefill_job(seed):
    r = random.Random(seed)
    txt = " ".join(r.choice(W) for _ in range(12000))
    u, el = call(txt, 1)
    return u["prompt_tokens"], el

for kind, job in (("decode", decode_job), ("prefill", prefill_job)):
    print(f"=== {kind} ===")
    for C in (1, 2, 4):
        seeds = [os.getpid()*7919 + int(time.time()*1000) % 90000 + i*1301 for i in range(C)]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=C) as ex:
            res = list(ex.map(job, seeds))
        wall = time.time() - t0
        total = sum(n for n, _ in res)
        per = [n/e for n, e in res]
        print(f"  C={C}: 집계 {total/wall:>7,.0f} tok/s | 스트림당 {sum(per)/len(per):>6,.0f} tok/s | wall {wall:.1f}s")
