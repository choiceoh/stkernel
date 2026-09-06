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
# 39차 (operator: "한국어 본문을 프리필해서 품질·프리필·디코드를 한 번에",
# "모든 테스트는 저거 한 개로 통일"): the document IS the Korean workload --
# three facts planted in Korean prose drawn in seeded order from a fixed pool
# of Korean Wikipedia paragraphs (bench/ko_filler.txt, CC BY-SA). The English
# word-salad body (14 words, assumed 1.3 tokens/word, actually ~1.0) is gone.
# (planted sentence, question, accepted answer substrings)
FACTS = [
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


def filler(n_tokens, rng):
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
            "content": (f"문서:\n{body_text}\n\n"
                        f"문서의 내용만 근거로 한국어 한 문장으로 답해줘. 이름·숫자·날짜·장비 번호는 문서에 적힌 그대로 인용해줘.\n"
                        f"질문: {question}"),
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
    total = ok = 0
    for ctx in ctxs:
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
