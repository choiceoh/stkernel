#!/usr/bin/env python3
"""The MTP window's requests (stdlib only, on the head node).

    mtp_requests.py eval URL LABEL PROMPTS            the held-out prompts, C=1, greedy, thinking off: each request's
                                                      decode ms/step and tokens/step from /metrics deltas
    mtp_requests.py data URL LABEL PROMPTS SECONDS C  the train prompts at concurrency C until SECONDS pass, cycling
                                                      (a later pass samples at 0.8): the tap's data
    mtp_requests2.py eval URL LABEL PROMPTS T,K,P     the held-out prompts at a sampler instead of greedy (T=1 arms)
    mtp_requests2.py prefill2 URL LABEL CONVOS C WHICH [RAW]
                                                      datagen2.py's conversations whose answers were greedy / sampled /
                                                      all (WHICH), each prefilled whole through the chat template (system,
                                                      turns, tools, the last turn's thinking), then RAW's documents as
                                                      raw text (/v1/completions): the tap's data
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


SAMPLER = None                                                # eval's T,K,P override (the T=1 arms)
if MODE == "eval" and len(sys.argv) > 5:
    _t, _k, _p = sys.argv[5].split(",")
    SAMPLER = (float(_t), int(_k), float(_p))


def ask(model, p, temperature=None):
    t = p["temperature"] if temperature is None else temperature
    top_k, top_p = None, 0.95
    if SAMPLER is not None:
        t, top_k, top_p = SAMPLER
    # a v2 held-out prompt carries its conversation up to the last user turn (system prompt, earlier turns)
    body = {"model": model, "messages": p.get("messages") or [{"role": "user", "content": p["content"]}],
            "max_tokens": p["max_tokens"],
            "temperature": t, "chat_template_kwargs": {"enable_thinking": bool(p.get("thinking"))}}
    if t > 0:
        body["top_p"] = top_p
        if top_k:
            body["top_k"] = top_k
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


prompts = [json.loads(line) for line in open(PROMPTS)]


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
model, waited = wait_door()
print(json.dumps({"label": LABEL, "door_open_after_s": waited, "model": model}), flush=True)

if MODE == "eval":
    # the first completion token is the prefill step's; the decode steps and their seconds made the rest
    total_steps = total_seconds = total_tokens = total_decoded = 0.0
    # the T=1 arms (best-t1-*) run the hand-written and Deneb prompts only (operator: "그렇게해", the window's clock)
    t1_subset = LABEL.startswith("best-t1")
    for p in (p for p in prompts if p["split"] == "eval"
              and (not t1_subset or p["id"].startswith(("s-", "d-", "r-")))):   # 80: operator "80개까지만 하자"
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
        decoded = max(made - 1, 0)
        total_steps, total_seconds, total_tokens = total_steps + steps, total_seconds + seconds, total_tokens + made
        total_decoded += decoded
        print(json.dumps({"label": LABEL, "id": p["id"], "completion_tokens": made, "wall_s": round(wall, 2),
                          "decode_steps": steps, "ms_a_step": round(seconds / steps * 1e3, 2) if steps else None,
                          "tokens_a_step": round(decoded / steps, 3) if steps else None,
                          "finish": answer["choices"][0].get("finish_reason"),
                          "text": (answer["choices"][0]["message"].get("content") or "")[:120]}, ensure_ascii=False), flush=True)
    print(json.dumps({"label": LABEL, "summary": {"steps": total_steps, "tokens": total_tokens, "decoded_tokens": total_decoded,
                                                  "ms_a_step": round(total_seconds / total_steps * 1e3, 2) if total_steps else None,
                                                  "tokens_a_step": round(total_decoded / total_steps, 3) if total_steps else None,
                                                  "decode_tok_s": round(total_decoded / total_seconds, 1) if total_seconds else None}}),
          flush=True)
elif MODE == "prefill2":
    # PROMPTS is datagen2.py's conversations.jsonl: the ones whose answers were greedy / sampled / all, each prefilled
    # whole (system, turns, tool calls and results, the last turn's thinking; the answer continued, one token drawn),
    # then RAW's documents as raw text -- the tap records the target's streams at every position
    concurrency, which = int(sys.argv[5]), sys.argv[6]
    raw_path = sys.argv[7] if len(sys.argv) > 7 else None
    greedy = lambda r: (r.get("final") or {}).get("temperature") == 0.0
    convos = [r for r in prompts if which == "all" or (which == "greedy") == greedy(r)]
    convos = [r for r in convos if (r["messages"][-1].get("content") or "").strip()
              or r["messages"][-1].get("reasoning_content")]          # a reasoning cut by max_tokens: its answer is " "
    jobs = [("chat", r) for r in convos]
    if raw_path:
        jobs += [("raw", json.loads(line)) for line in open(raw_path)]
    lock, state = threading.Lock(), {"next": 0, "done": 0, "prompt_tokens": 0, "errors": 0, "chat": 0, "raw": 0}

    def feed():
        while True:
            with lock:
                i = state["next"]
                state["next"] += 1
            if i >= len(jobs):
                return
            kind, r = jobs[i]
            try:
                if kind == "chat":
                    body = {"model": model, "messages": r["messages"], "max_tokens": 1, "temperature": 0,
                            "add_generation_prompt": False, "continue_final_message": True,
                            "chat_template_kwargs": {"enable_thinking": bool(r.get("thinking"))}}
                    if r.get("tools"):
                        body["tools"] = r["tools"]
                    usage = post("/v1/chat/completions", body).get("usage", {})
                else:
                    usage = post("/v1/completions", {"model": model, "prompt": r["text"], "max_tokens": 1,
                                                     "temperature": 0}).get("usage", {})
                with lock:
                    state["done"] += 1
                    state[kind] += 1
                    state["prompt_tokens"] += usage.get("prompt_tokens", 0)
            except Exception as exc:                          # noqa: BLE001
                with lock:
                    state["errors"] += 1
                print(json.dumps({"label": LABEL, "kind": kind, "id": r.get("cid") or r.get("rid"), "error": repr(exc)[:300]}),
                      flush=True)

    threads = [threading.Thread(target=feed, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    began = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        with lock:
            print(json.dumps({"label": LABEL, "elapsed_s": round(time.time() - began), "of": len(jobs), **state}), flush=True)
    print(json.dumps({"label": LABEL, "final": state, "elapsed_s": round(time.time() - began)}), flush=True)
elif MODE == "prefill":
    # PROMPTS is openrouter_gen.py's samples: each conversation prefilled whole (the answer continued, one token
    # drawn), so the tap records the target's streams at every position of the text
    concurrency = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    # a thinking answer cut by max_tokens has reasoning and no content: keep it, its answer is " " (review, PR #1275)
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
