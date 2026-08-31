#!/usr/bin/env python3
"""Correctness probe. Facts are planted at a known depth inside a long filler
body so a wrong index stride (the class of bug PR #51042 fixes) shows up as a
retrieval miss rather than as merely-degraded prose. Runs at 2K / 32K / 128K."""
import json
import os
import random
import sys
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4-flash")
WORDS = [
    "reactor", "harbor", "lattice", "quarry", "ember", "meridian", "syntax",
    "granite", "voltage", "cirrus", "tundra", "beacon", "ledger", "prism",
]

# (planted sentence, question, accepted answer substrings)
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
    # plant the three facts at 25% / 50% / 75% depth (4 equal filler chunks)
    chunks, per = [], ctx_tokens / 4.0
    for i, (sent, _, _) in enumerate(FACTS):
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
                        f"Answer from the document only, in one short sentence. "
                        f"{question}"),
        }],
        "max_tokens": 120,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"] or ""


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    total = ok = 0
    for ctx in (2_000, 32_000, 128_000):
        doc = build(ctx, seed + ctx)
        hits = []
        for _, q, expect in FACTS:
            ans = ask(doc, q).lower()
            good = all(e in ans for e in expect)
            hits.append("o" if good else "X")
            total += 1
            ok += good
            if not good:
                print(f"    MISS ctx~{ctx // 1000}K q={q!r} -> {ans[:110]!r}")
        print(f"ctx~{ctx // 1000:>3}K  {' '.join(hits)}", flush=True)
    print(f"=> {ok}/{total} correct")
