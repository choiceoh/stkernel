#!/usr/bin/env python3
"""256K needle probe — check-quality.py pattern extended to ctx=256K.
Three planted facts at 25/50/75% depth, greedy retrieval, substring match."""
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
]
FACTS = [
    ("The calibration constant for the Meridian array is 8127.",
     "What is the calibration constant for the Meridian array?", ["8127"]),
    ("Dr. Halvorsen signed the tundra survey on 14 March 1997.",
     "Who signed the tundra survey, and on what date?",
     ["halvorsen", "14 march 1997"]),
    ("The quarry's emergency shutoff is valve K-42 on the north wall.",
     "Which valve is the quarry's emergency shutoff, and where is it?",
     ["k-42", "north"]),
]


def filler(n_tokens, rng):
    return " ".join(rng.choice(WORDS) for _ in range(int(n_tokens / 1.3)))


def build(ctx_tokens, seed):
    rng = random.Random(seed)
    chunks, per = [], ctx_tokens / 4.0
    for sent, _, _ in FACTS:
        chunks.append(filler(per, rng))
        chunks.append(sent)
    chunks.append(filler(per, rng))
    return "\n".join(chunks)


def ask(body_text, question):
    body = json.dumps({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": (f"Document:\n{body_text}\n\n"
                        f"Answer from the document only, in one short "
                        f"sentence. {question}"),
        }],
        "max_tokens": 120,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    el = time.time() - t0
    return (out["choices"][0]["message"]["content"] or ""), el, \
        out["usage"]["prompt_tokens"]


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    doc = build(256_000, seed)
    ok = 0
    for i, (_, q, keys) in enumerate(FACTS):
        ans, el, pt = ask(doc, q)
        hit = all(k in ans.lower() for k in keys)
        ok += hit
        print(f"needle {i + 1} (depth {(i + 1) * 25}%): "
              f"{'o' if hit else 'X'}  [{el:.0f}s, {pt:,} tok] {ans[:70]!r}",
              flush=True)
    print(f"=> {ok}/3 correct at 256K")
