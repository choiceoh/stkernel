#!/usr/bin/env python3
"""The MTP window's requests (stdlib only, on the head node).

    mtp_requests.py eval URL LABEL PROMPTS            the held-out prompts, C=1, greedy, thinking off: each request's
                                                      decode ms/step and tokens/step from /metrics deltas
    mtp_requests.py data URL LABEL PROMPTS SECONDS C  the train prompts at concurrency C until SECONDS pass, cycling
                                                      (a later pass samples at 0.8): the tap's data
"""
import json
import sys
import threading
import time
import urllib.request

MODE, BASE, LABEL, PROMPTS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


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


def wait_door(limit_s=900.0):
    began = time.time()
    while True:
        try:
            return json.loads(get("/v1/models"))["data"][0]["id"], round(time.time() - began, 1)
        except Exception as exc:                                  # noqa: BLE001 -- the door is not up yet
            if time.time() - began > limit_s:
                print(json.dumps({"label": LABEL, "door": "never opened", "last": repr(exc)[:200]}), flush=True)
                sys.exit(1)
            time.sleep(3)


def ask(model, p, temperature=None):
    t = p["temperature"] if temperature is None else temperature
    body = {"model": model, "messages": [{"role": "user", "content": p["content"]}], "max_tokens": p["max_tokens"],
            "temperature": t, "chat_template_kwargs": {"enable_thinking": bool(p.get("thinking"))}}
    if t > 0:
        body["top_p"] = 0.95
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


prompts = [json.loads(line) for line in open(PROMPTS)]
model, waited = wait_door()
print(json.dumps({"label": LABEL, "door_open_after_s": waited, "model": model}), flush=True)

if MODE == "eval":
    total_steps = total_seconds = total_tokens = 0.0
    for p in (p for p in prompts if p["split"] == "eval"):
        before = metrics()
        t0 = time.time()
        try:
            answer = ask(model, p)
        except Exception as exc:                                  # noqa: BLE001
            print(json.dumps({"label": LABEL, "id": p["id"], "error": repr(exc)[:300]}), flush=True)
            continue
        wall = time.time() - t0
        after = metrics()
        steps = pick(after, "st:steps_decode_total") - pick(before, "st:steps_decode_total")
        seconds = pick(after, "st:step_seconds_sum", 'kind="decode"') - pick(before, "st:step_seconds_sum", 'kind="decode"')
        made = answer.get("usage", {}).get("completion_tokens", 0)
        total_steps, total_seconds, total_tokens = total_steps + steps, total_seconds + seconds, total_tokens + made
        print(json.dumps({"label": LABEL, "id": p["id"], "completion_tokens": made, "wall_s": round(wall, 2),
                          "decode_steps": steps, "ms_a_step": round(seconds / steps * 1e3, 2) if steps else None,
                          "tokens_a_step": round(made / steps, 3) if steps else None,
                          "finish": answer["choices"][0].get("finish_reason"),
                          "text": (answer["choices"][0]["message"].get("content") or "")[:120]}, ensure_ascii=False), flush=True)
    print(json.dumps({"label": LABEL, "summary": {"steps": total_steps, "tokens": total_tokens,
                                                  "ms_a_step": round(total_seconds / total_steps * 1e3, 2) if total_steps else None,
                                                  "tokens_a_step": round(total_tokens / total_steps, 3) if total_steps else None,
                                                  "decode_tok_s": round(total_tokens / total_seconds, 1) if total_seconds else None}}),
          flush=True)
elif MODE == "prefill":
    # PROMPTS is openrouter_gen.py's samples: each conversation prefilled whole (the answer continued, one token
    # drawn), so the tap records the target's streams at every position of the text
    concurrency = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    # a thinking answer cut by max_tokens has reasoning and no content: keep it, its answer is " " (the v2 driver's
    # rule). Window 3 ran `if s.get("content")` and so skipped 29 of 1,470 samples (6.9% of the text), all long
    # thinking -- fixed on review (PR #1275), the record says what ran
    samples = [s for s in prompts if s.get("content") or s.get("reasoning")]
    lock, state = threading.Lock(), {"next": 0, "done": 0, "prompt_tokens": 0, "errors": 0}

    def feed():
        while True:
            with lock:
                i = state["next"]
                state["next"] += 1
            if i >= len(samples):
                return
            s = samples[i]
            answer = (f"<think>\n{s['reasoning'].strip()}\n</think>\n\n" if s.get("reasoning") else "") + (s.get("content") or " ")
            body = {"model": model, "messages": [{"role": "user", "content": s["prompt"]}, {"role": "assistant", "content": answer}],
                    "max_tokens": 1, "temperature": 0, "add_generation_prompt": False, "continue_final_message": True,
                    "chat_template_kwargs": {"enable_thinking": bool(s.get("thinking"))}}
            req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    usage = json.loads(r.read().decode()).get("usage", {})
                with lock:
                    state["done"] += 1
                    state["prompt_tokens"] += usage.get("prompt_tokens", 0)
            except Exception as exc:                          # noqa: BLE001
                with lock:
                    state["errors"] += 1
                print(json.dumps({"label": LABEL, "id": s["id"], "sample": s.get("sample"), "error": repr(exc)[:200]}),
                      flush=True)

    threads = [threading.Thread(target=feed, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    began = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        with lock:
            print(json.dumps({"label": LABEL, "elapsed_s": round(time.time() - began), "of": len(samples), **state}), flush=True)
    print(json.dumps({"label": LABEL, "final": state, "elapsed_s": round(time.time() - began)}), flush=True)
else:
    seconds, concurrency = float(sys.argv[5]), int(sys.argv[6])
    train = [p for p in prompts if p["split"] == "train"]
    lock, state = threading.Lock(), {"next": 0, "done": 0, "tokens": 0, "errors": 0}
    deadline = time.time() + seconds

    def worker():
        while time.time() < deadline:
            with lock:
                i = state["next"]
                state["next"] += 1
            p = train[i % len(train)]
            temperature = p["temperature"] if i < len(train) else 0.8
            try:
                answer = ask(model, p, temperature)
                made = answer.get("usage", {}).get("completion_tokens", 0)
                with lock:
                    state["done"] += 1
                    state["tokens"] += made
            except Exception as exc:                              # noqa: BLE001
                with lock:
                    state["errors"] += 1
                print(json.dumps({"label": LABEL, "id": p["id"], "error": repr(exc)[:200]}), flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    began = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with lock:
            print(json.dumps({"label": LABEL, "elapsed_s": round(time.time() - began), **state}), flush=True)
    print(json.dumps({"label": LABEL, "final": state, "elapsed_s": round(time.time() - began)}), flush=True)
