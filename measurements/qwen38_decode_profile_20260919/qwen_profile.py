"""srv2: where a Qwen3.8 decode step's time goes on the booted door (:8000) -- measurements/st_decode_profile_20260914's
method (GLM-5.3, 09-14) on this model.

Long greedy decodes keep the engine in steady steps. During each, POST /v1/engine/profile profiles the next 32 decode
steps on every rank and GET returns rank 0's kernel table (device time per kernel, calls). Around each profiled window,
and around an unprofiled window beside it, /metrics gives the host-seen step seconds and the drafted/accepted counters,
so the table can be put per step and the profiler's own cost is visible. A Qwen3.8 step is one verification (K drafts,
K+1 tokens a row), so a step and an iteration are the same thing here.

    python3 qwen_profile.py --out <json> --label <label> [--c4]
"""
import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request

URL = "http://10.10.10.2:8000"
PROMPT = ("Write a very long, detailed technical essay (at least 5000 words) on the history of numerical linear "
          "algebra, from Gaussian elimination to modern GPU kernels. Use many sections and subsections, explain each "
          "algorithm step by step, and do not stop early.")


def http(path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and path == "/v1/engine/profile" and body is None:
            return None
        raise
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode()


def metrics():
    out = {}
    for line in http("/metrics").splitlines():
        m = re.match(r'^(st:step_seconds_(?:sum|count))\{engine="st",kind="decode"\} (\S+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
        for name in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                     "vllm:spec_decode_num_drafts_total", "st:steps_decode_total", "vllm:generation_tokens_total"):
            if line.startswith(name + "{") or line.startswith(name + " "):
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def delta(a, b):
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b}
    steps = d.get("st:step_seconds_count", 0.0)
    drafted = d.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    d["step_ms"] = 1000 * d.get("st:step_seconds_sum", 0.0) / steps if steps else None
    d["acceptance"] = d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) / drafted if drafted else None
    made = d.get("vllm:generation_tokens_total", 0.0)
    d["tokens_a_step"] = made / steps if steps else None
    return d


def decode(model, n, stop):
    """n concurrent long greedy requests; each restarts until `stop` is set, so decode never pauses."""
    def one():
        while not stop.is_set():
            body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 6000,
                    "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
            try:
                http("/v1/chat/completions", body, timeout=900)
            except Exception as exc:  # noqa: BLE001 -- keep the load up; the table says what ran
                print("request error:", exc, flush=True)
                time.sleep(1)
    threads = [threading.Thread(target=one, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    return threads


def window(profile, seconds_hint):
    before = metrics()
    table = None
    if profile:
        previous = http("/v1/engine/profile")
        http("/v1/engine/profile", {"steps": 32})
        deadline = time.time() + 120
        table = previous
        while time.time() < deadline:
            time.sleep(2)
            table = http("/v1/engine/profile")
            if isinstance(table, dict) and table.get("kernels") and table != previous:
                break
    else:
        time.sleep(seconds_hint)
    after = metrics()
    return table, delta(before, after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--c4", action="store_true", help="also one profiled window at C=4")
    a = ap.parse_args()
    model = http("/v1/models")["data"][0]["id"]
    http("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16,
                                  "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
    result = {"label": a.label, "model": model, "runs": []}
    plan = [(1, 2)] + ([(4, 1)] if a.c4 else [])
    for concurrency, repeats in plan:
        stop = threading.Event()
        threads = decode(model, concurrency, stop)
        time.sleep(15)                                   # past the prefill, into steady steps
        for r in range(repeats):
            table, prof = window(True, 0)
            _, plain = window(False, 8)
            run = {"concurrency": concurrency, "repeat": r, "profiled": prof, "unprofiled": plain,
                   "device_us_per_step": (table or {}).get("device_us_per_step"), "steps": (table or {}).get("steps"),
                   "kernels": (table or {}).get("kernels", [])}
            result["runs"].append(run)
            print(f"C={concurrency} #{r}: device {run['device_us_per_step']} us/step over {run['steps']} steps; step ms "
                  f"profiled {prof.get('step_ms')} vs plain {plain.get('step_ms')}; tokens/step {plain.get('tokens_a_step')}; "
                  f"accept {plain.get('acceptance')}; kernels {len(run['kernels'])}", flush=True)
            with open(a.out, "w") as f:
                json.dump(result, f, indent=1)
        stop.set()
        for t in threads:
            t.join(timeout=900)
    print("PROFILE done", flush=True)


if __name__ == "__main__":
    main()
