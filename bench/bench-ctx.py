#!/usr/bin/env python3
"""Long-context decode bench. Fresh (uncached) prompt each run so prefix caching
cannot inflate the numbers; decode rate is measured over the generated tokens
only, using the server's own usage counters."""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-v4-flash"
WORDS = [
    "reactor", "harbor", "lattice", "quarry", "ember", "meridian", "syntax",
    "granite", "voltage", "cirrus", "tundra", "beacon", "ledger", "prism",
    "cobalt", "willow", "cascade", "anvil", "nocturne", "vellum",
]


def filler(approx_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    # ~1.3 tokens per word for this vocabulary; overshoot slightly, the exact
    # length is reported back by the server anyway.
    n = int(approx_tokens / 1.3)
    return " ".join(rng.choice(WORDS) for _ in range(n))


def run(ctx_tokens: int, max_out: int, seed: int):
    body = json.dumps({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": (
                f"Below is a word list. Ignore it entirely.\n\n"
                f"{filler(ctx_tokens, seed)}\n\n"
                f"Now write exactly {max_out} tokens of continuous prose about "
                f"how tides work. Do not mention the word list."
            ),
        }],
        "max_tokens": max_out,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    t_last = t0
    ct = 0
    pt = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                pt = ev["usage"].get("prompt_tokens", pt)
                ct = ev["usage"].get("completion_tokens", ct)
            ch = ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if t_first is None:
                    t_first = time.time()
                t_last = time.time()
    ttft = (t_first or t_last) - t0
    gen = max(t_last - (t_first or t_last), 1e-9)
    return pt, ct, ttft, gen


if __name__ == "__main__":
    seed_base = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for ctx, out_toks in ((2_000, 256), (32_000, 256), (128_000, 256)):
        pt, ct, ttft, gen = run(ctx, out_toks, seed_base * 977 + ctx)
        print(f"ctx~{ctx//1000:>3}K | prompt {pt:>7,} | out {ct:>4} | "
              f"TTFT {ttft:>6.1f}s | decode {(ct - 1) / gen:>6.1f} tok/s",
              flush=True)
