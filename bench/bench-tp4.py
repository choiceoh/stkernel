#!/usr/bin/env python3
"""Fresh-prompt prefill+decode bench for the hy4 stack (prefix-cache proof: unique seed each run)."""
import json, time, urllib.request, random, os, sys

BASE = "http://localhost:8000/v1/chat/completions"


def _resolve_model(default: str) -> str:
    """The served name, asked of the server, not assumed.

    The literal default is the dsv4 lane's name. Pointed at the glm53 server
    it 404s, which has silently voided prefill and decode runs in this lane
    more than once -- the harness raised, the boot script's grep found no
    SUMMARY line, and the section read as "measured nothing" rather than
    "never ran". Ask; fall back to the literal only if the server cannot say.
    """
    named = os.environ.get("BENCH_MODEL")
    if named:
        return named
    try:
        base = BASE.split("/v1/", 1)[0]
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode())["data"]
        for entry in data:
            if entry["id"] == default:
                return default
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return default


# Served name of the model under test; same env as bench-dec.
MODEL = _resolve_model("deepseek-v4-flash")
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
