#!/usr/bin/env python3
"""Count corrupted Korean output, so "간헐적" becomes a rate instead of an impression.

The operator sees individual Korean tokens break occasionally. Occasionally is
the hard part: one bad sample proves the bug, but no bad sample in three tries
proves nothing, and neither tells you whether a change helped. So generate a
fixed corpus and report how many characters and how many responses are hit.

What counts as corruption, in order of how sure we can be:

  replacement  U+FFFD. A decoder gave up on a byte sequence. Unambiguous.
  lone_jamo    Hangul Jamo / Compatibility Jamo outside a syllable block. Korean
               text renders as precomposed syllables (AC00-D7A3); a bare ㄱ or ᅡ
               between them means a syllable was assembled from the wrong pieces.
  cjk_mixed    Han characters inside a Korean response. The model was asked to
               answer in Korean, so a stray 漢 is a token id that landed in the
               wrong part of the vocabulary -- the signature of a bad sample,
               not of a bad decode.
  control      C0/C1 controls other than tab/newline.

Prompts are fixed and seeded so two runs compare directly; temperature 0 so a
difference is the stack's, not the sampler's.
"""
import json
import os
import re
import sys
import unicodedata
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "glm-5.3-flash")

PROMPTS = [
    "조력 발전의 원리를 한국어로 자세히 설명해줘.",
    "한국의 전력 계통에서 주파수를 유지하는 방법을 설명해줘.",
    "다음을 한국어로 요약해줘: 변압기의 냉각 방식에는 유입 자냉식, 유입 풍냉식, 송유 수냉식이 있다.",
    "서울에서 부산까지 기차로 가는 방법을 한국어로 안내해줘.",
    "한국어 맞춤법에서 띄어쓰기가 어려운 사례를 다섯 개 들어줘.",
    "태양광 인버터의 MPPT 제어를 한국어로 설명해줘.",
    "김치를 담그는 과정을 순서대로 한국어로 설명해줘.",
    "한국의 사계절 기후 특징을 한국어로 서술해줘.",
]

SYL = re.compile(r"[가-힣]")
JAMO = re.compile(r"[ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]")
HAN = re.compile(r"[一-鿿㐀-䶿]")


def scan(text: str) -> dict:
    hits = {"replacement": 0, "lone_jamo": 0, "cjk_mixed": 0, "control": 0}
    hits["replacement"] = text.count("�")
    hits["lone_jamo"] = len(JAMO.findall(text))
    hits["cjk_mixed"] = len(HAN.findall(text))
    hits["control"] = sum(
        1 for c in text
        if unicodedata.category(c) == "Cc" and c not in "\t\n\r"
    )
    return hits


def ask(prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    # GLM-style models stream the chain of thought into `reasoning` and leave
    # `content` empty until it ends. Both are model output and both can break,
    # so scan whichever arrived rather than reporting a clean zero.
    parts = [m.get(k) or "" for k in ("content", "reasoning", "reasoning_content")]
    return "\n".join(p for p in parts if p)


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    total = {"replacement": 0, "lone_jamo": 0, "cjk_mixed": 0, "control": 0}
    chars = 0
    dirty = []
    n = 0
    for r in range(rounds):
        for i, p in enumerate(PROMPTS):
            text = ask(p, max_tokens)
            n += 1
            chars += len(text)
            h = scan(text)
            for k in total:
                total[k] += h[k]
            if any(h.values()):
                worst = max(h, key=lambda k: h[k])
                idx = -1
                if worst == "replacement":
                    idx = text.find("�")
                elif worst == "lone_jamo":
                    m = JAMO.search(text)
                    idx = m.start() if m else -1
                elif worst == "cjk_mixed":
                    m = HAN.search(text)
                    idx = m.start() if m else -1
                ctx = text[max(0, idx - 40):idx + 40].replace("\n", " ") if idx >= 0 else ""
                dirty.append((r, i, h, ctx))

    print(f"응답 {n}개 · 문자 {chars:,}")
    if chars == 0:
        print("  판정 불가: 응답에 텍스트가 없다 (max_tokens 부족이거나 필드 불일치)")
        return 2
    print(f"  깨진 응답: {len(dirty)}/{n} ({100*len(dirty)/n:.0f}%)")
    for k, v in total.items():
        rate = f"{1e6*v/chars:.1f}/백만자" if chars else "-"
        print(f"  {k:12s} {v:>5}   {rate}")
    for r, i, h, ctx in dirty[:6]:
        kinds = ",".join(f"{k}={v}" for k, v in h.items() if v)
        print(f"\n  [round {r} prompt {i}] {kinds}")
        print(f"    …{ctx}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
