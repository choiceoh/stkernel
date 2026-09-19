#!/usr/bin/env python3
"""A fleet window's requests: wait for the door, send a fixed set (greedy, thinking off, C=1), and give each request's
wall time and, from /metrics deltas, its decode ms/step and tokens/step. Stdlib only; runs on the head node."""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.10.10.2:8000"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "boot"
DOOR_S = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
LONG = ("다음 글을 세 문장으로 요약해 주세요.\n\n" + "태양광 발전소의 인허가 절차는 발전사업허가, 개발행위허가, 환경영향평가 협의, 계통연계 신청, 공사계획 신고, "
        "사용전검사의 순서로 진행되며 각 단계마다 관할 기관과 제출 서류, 처리 기간이 다릅니다. 풍력은 여기에 해상교통안전진단과 군 전파영향 협의가 더해집니다. " * 40)
PROMPTS = [("17x23", "17 곱하기 23 은? 숫자만 답해.", 32), ("capital", "대한민국의 수도는 어디이고 인구는 대략 얼마인가요? 한 문장으로.", 64),
           ("sky", "하늘이 파란 이유를 세 문단으로 설명해 주세요.", 256), ("transformer", "Explain the transformer architecture in about 350 words.", 512),
           ("long-summary", LONG, 96), ("transformer-again", "Explain the transformer architecture in about 350 words.", 512),
           ("sky-again", "하늘이 파란 이유를 세 문단으로 설명해 주세요.", 256)]


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


began = time.time()
while True:
    try:
        models = json.loads(get("/v1/models"))
        model = models["data"][0]["id"]
        break
    except Exception as exc:                                  # noqa: BLE001 -- the door is not up yet
        if time.time() - began > DOOR_S:
            print(json.dumps({"label": LABEL, "door": "never opened", "waited_s": round(time.time() - began, 1), "last": repr(exc)}))
            sys.exit(1)
        time.sleep(3)
print(json.dumps({"label": LABEL, "door_open_after_s": round(time.time() - began, 1), "model": model}), flush=True)

for name, prompt, max_tokens in PROMPTS:
    before = metrics()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", body,
                                                           {"Content-Type": "application/json"}), timeout=600) as r:
            answer = json.loads(r.read().decode())
    except Exception as exc:                                  # noqa: BLE001
        print(json.dumps({"label": LABEL, "request": name, "error": repr(exc)[:300], "wall_s": round(time.time() - t0, 2)}), flush=True)
        continue
    wall = time.time() - t0
    after = metrics()
    steps = pick(after, "st:steps_decode_total") - pick(before, "st:steps_decode_total")
    seconds = pick(after, "st:step_seconds_sum", 'kind="decode"') - pick(before, "st:step_seconds_sum", 'kind="decode"')
    usage = answer.get("usage", {})
    made = usage.get("completion_tokens", 0)
    text = answer["choices"][0]["message"].get("content") or ""
    print(json.dumps({"label": LABEL, "request": name, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": made,
                      "wall_s": round(wall, 2), "decode_steps": steps, "ms_a_step": round(seconds / steps * 1e3, 2) if steps else None,
                      "tokens_a_step": round(made / steps, 3) if steps else None,
                      "decode_tok_s": round(made / seconds, 1) if seconds else None,
                      "tok_s_incl_prefill": round(made / wall, 1) if wall else None,
                      "finish": answer["choices"][0].get("finish_reason"), "text": text[:160]}, ensure_ascii=False), flush=True)
stalls = {k: v for k, v in metrics().items() if "stall" in k.lower() and v}
print(json.dumps({"label": LABEL, "stall_counters": stalls}), flush=True)
final = metrics()
print(json.dumps({"label": LABEL, "spec": {k: v for k, v in sorted(final.items()) if "spec" in k.lower()}}), flush=True)
print(json.dumps({"label": LABEL, "steps": {k: v for k, v in sorted(final.items()) if k.startswith("st:steps") or "step_seconds" in k}}), flush=True)
