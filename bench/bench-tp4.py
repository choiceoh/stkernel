#!/usr/bin/env python3
"""Fresh-prompt prefill+decode bench for the hy4 stack (prefix-cache proof: unique seed each run)."""
import json, time, urllib.request, random, os, sys

# Served name of the model under test; same env as bench-dec.
MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4-flash")

BASE = "http://localhost:8000/v1/chat/completions"
W = ["태양광","발전소","인버터","모듈","효율","계통","연계","전압","주파수","모니터링",
     "데이터","분석","예측","유지보수","진단","출력","손실","온도","일사량","가동률"]

def call(msg, max_tokens, to=900):
    b = json.dumps({"model": MODEL,"messages":[{"role":"user","content":msg}],
                    "max_tokens":max_tokens}).encode()
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(
        urllib.request.Request(BASE, b, {"Content-Type":"application/json"}), timeout=to).read())
    return d, time.time()-t0

label = sys.argv[1] if len(sys.argv) > 1 else "run"
random.seed(os.getpid() * 104729 + int(time.time()) % 100000)   # unique -> no prefix-cache hit

d, el = call("1부터 30까지 각 숫자가 소수인지 한 줄씩 설명하라.", 800)
dec = d["usage"]["completion_tokens"] / el

txt = " ".join(random.choice(W) for _ in range(25000))
d, el = call(txt + " 끝.", 1)
pt = d["usage"]["prompt_tokens"]
pre = pt / el

print(f"[{label}] decode {dec:6.1f} tok/s | prefill {pt:,} tok in {el:5.1f}s => {pre:,.0f} tok/s")
