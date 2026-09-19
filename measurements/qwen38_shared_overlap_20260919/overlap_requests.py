#!/usr/bin/env python3
"""A fleet window's requests for the shared-expert overlap A/B (carry M5): wait for the door, then the same fixed set
every boot -- C=1 greedy with thinking off, then C=4 (four requests at once, twice) -- and give each its decode
ms/step and tokens/step from /metrics deltas (the engine's own st:step_seconds, as window_requests.py of
measurements/qwen38_serve_window_20260919 does) and a hash of its full text, so two boots' answers can be compared.
Stdlib only; runs on the head node."""
import hashlib
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.10.10.2:8000"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "boot"
DOOR_S = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
C1 = [("warm", "17 곱하기 23 은? 숫자만 답해.", 16),
      ("sky", "하늘이 파란 이유를 세 문단으로 설명해 주세요.", 256),
      ("transformer", "Explain the transformer architecture in about 350 words.", 512),
      ("solar", "태양광 발전소 인허가 절차를 단계별로 자세히 설명해 주세요.", 512),
      ("transformer-again", "Explain the transformer architecture in about 350 words.", 512),
      ("sky-again", "하늘이 파란 이유를 세 문단으로 설명해 주세요.", 256)]
C4 = [("c4-physics", "Explain Newton's three laws of motion with examples.", 256),
      ("c4-wind", "풍력 발전기의 구조와 원리를 설명해 주세요.", 256),
      ("c4-ess", "Explain how a grid-scale battery energy storage system is operated.", 256),
      ("c4-history", "조선 시대의 과거 제도에 대해 설명해 주세요.", 256)]


def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read().decode()


def metrics():
    out = {}
    for line in get("/metrics").splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return out


def pick(m, *needles):
    return sum(v for k, v in m.items() if all(n in k for n in needles))


def ask(model, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", body,
                                                       {"Content-Type": "application/json"}), timeout=600) as r:
        answer = json.loads(r.read().decode())
    text = answer["choices"][0]["message"].get("content") or ""
    usage = answer.get("usage", {})
    return dict(wall_s=round(time.time() - t0, 2), completion_tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens"), sha=hashlib.sha256(text.encode()).hexdigest()[:16],
                finish=answer["choices"][0].get("finish_reason"), text=text[:100])


def block(name, fn):
    before = metrics()
    t0 = time.time()
    result = fn()
    wall = time.time() - t0
    after = metrics()
    steps = pick(after, "st:steps_decode_total") - pick(before, "st:steps_decode_total")
    seconds = pick(after, "st:step_seconds_sum", 'kind="decode"') - pick(before, "st:step_seconds_sum", 'kind="decode"')
    drafted = pick(after, "spec_decode_num_draft_tokens") - pick(before, "spec_decode_num_draft_tokens")
    accepted = pick(after, "spec_decode_num_accepted_tokens") - pick(before, "spec_decode_num_accepted_tokens")
    buckets = {k.split("le=")[1].split('"')[1]: round(after[k] - before.get(k, 0.0))
               for k in after if k.startswith("st:step_seconds_bucket") and 'kind="decode"' in k and "le=" in k}
    made = sum(r["completion_tokens"] for r in result) if isinstance(result, list) else result["completion_tokens"]
    row = {"label": LABEL, "block": name, "wall_s": round(wall, 2), "decode_steps": steps, "completion_tokens": made,
           "ms_a_step": round(seconds / steps * 1e3, 3) if steps else None,
           "tokens_a_step": round(made / steps, 3) if steps else None,
           "decode_tok_s": round(made / seconds, 1) if seconds else None,
           "accepted": accepted, "drafted": drafted, "step_buckets": buckets}
    row["requests"] = result if isinstance(result, list) else [result]
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


began = time.time()
while True:
    try:
        model = json.loads(get("/v1/models"))["data"][0]["id"]
        break
    except Exception as exc:                                  # noqa: BLE001 -- the door is not up yet
        if time.time() - began > DOOR_S:
            print(json.dumps({"label": LABEL, "door": "never opened", "waited_s": round(time.time() - began, 1),
                              "last": repr(exc)}))
            sys.exit(1)
        time.sleep(3)
print(json.dumps({"label": LABEL, "door_open_after_s": round(time.time() - began, 1), "model": model}), flush=True)

for name, prompt, max_tokens in C1:
    try:
        block(name, lambda: ask(model, prompt, max_tokens))
    except Exception as exc:                                  # noqa: BLE001
        print(json.dumps({"label": LABEL, "block": name, "error": repr(exc)[:300]}), flush=True)


def four():
    results = [None] * len(C4)

    def one(i, prompt, max_tokens):
        try:
            results[i] = dict(name=C4[i][0], **ask(model, prompt, max_tokens))
        except Exception as exc:                              # noqa: BLE001
            results[i] = dict(name=C4[i][0], error=repr(exc)[:200], completion_tokens=0)
    threads = [threading.Thread(target=one, args=(i, p, m)) for i, (_, p, m) in enumerate(C4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


for rep in (1, 2):
    try:
        block(f"c4 run {rep}", four)
    except Exception as exc:                                  # noqa: BLE001
        print(json.dumps({"label": LABEL, "block": f"c4 run {rep}", "error": repr(exc)[:300]}), flush=True)
final = metrics()
print(json.dumps({"label": LABEL, "stall_counters": {k: v for k, v in final.items() if "stall" in k.lower() and v}}),
      flush=True)
