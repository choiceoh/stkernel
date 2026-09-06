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
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from bench_common import resolve_model as _resolve_model
MODEL = _resolve_model("deepseek-v4-flash")
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


# 39차 (operator: "한국어 본문을 프리필해서 품질·프리필·디코드를 한 번에"): the
# same three facts planted in KOREAN prose, so the document IS the Korean
# workload -- prefill on real Korean text, answers decoded from a Korean
# context, corruption scanned on those answers -- instead of an English
# gibberish body plus a separate Korean prompt set. The filler is a fixed
# pool of Korean Wikipedia paragraphs (bench/ko_filler.txt, CC BY-SA) drawn
# in seeded order; the English word-salad body stays as lang="en" (dsv4's
# gate and the pre-39차 records).
FACTS_KO = [
    ("메리디안 배열의 보정 상수는 8127이다.",
     "메리디안 배열의 보정 상수는 얼마인가?", ["8127"]),
    ("할보르센 박사는 1997년 3월 14일에 툰드라 조사 보고서에 서명했다.",
     "툰드라 조사 보고서에는 누가, 언제 서명했는가?",
     ["할보르센", "1997년 3월 14일"]),
    ("채석장의 비상 차단 밸브는 북쪽 벽에 있는 K-42 밸브이다.",
     "채석장의 비상 차단 밸브는 무엇이고 어디에 있는가?", ["k-42", "북쪽"]),
]
KO_CHARS_PER_TOKEN = 1.24      # GLM-5.3 tokenizer on the pool (39차: 170,026 chars -> 137,416 tokens)
_KO_POOL = None


def _ko_pool():
    global _KO_POOL
    if _KO_POOL is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ko_filler.txt")
        with open(path, encoding="utf-8") as f:
            _KO_POOL = [p for p in f.read().split("\n") if p]
    return _KO_POOL


def filler_ko(n_tokens, rng):
    pool = _ko_pool()
    want = int(n_tokens * KO_CHARS_PER_TOKEN)
    out, got = [], 0
    while got < want:
        p = pool[rng.randrange(len(pool))]
        if got + len(p) > want + 40:            # trim the last paragraph at a sentence end
            cut = p.rfind("다.", 0, max(want - got, 1) + 2)
            p = p[:cut + 2] if cut > 0 else p[:want - got]
        out.append(p)
        got += len(p) + 1
    return "\n".join(out)


def facts(lang="en"):
    return FACTS_KO if lang == "ko" else FACTS


def build(ctx_tokens, seed, lang="en"):
    rng = random.Random(seed)
    fill = filler_ko if lang == "ko" else filler
    # plant the three facts at 25% / 50% / 75% depth (4 equal filler chunks)
    chunks, per = [], ctx_tokens / 4.0
    for i, (sent, _, _) in enumerate(facts(lang)):
        chunks.append(fill(per, rng))
        chunks.append(sent)
    chunks.append(fill(per, rng))
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
    # QUALITY_CTX=2000,32000 -> the quick leg (an exploration arm skips the
    # 128K prefill, ~3 min of the 25-min arm); the default is the full ladder
    ctxs = tuple(int(c) for c in os.environ.get("QUALITY_CTX", "2000,32000,128000").split(","))
    lang = os.environ.get("QUALITY_LANG", "en")   # ko = Korean prose body (39차); en = the word salad
    total = ok = 0
    for ctx in ctxs:
        doc = build(ctx, seed + ctx, lang)
        hits = []
        for _, q, expect in facts(lang):
            ans = ask(doc, q).lower()
            good = all(e in ans for e in expect)
            hits.append("o" if good else "X")
            total += 1
            ok += good
            if not good:
                print(f"    MISS ctx~{ctx // 1000}K q={q!r} -> {ans[:110]!r}")
        print(f"ctx~{ctx // 1000:>3}K  {' '.join(hits)}", flush=True)
    print(f"=> {ok}/{total} correct")
