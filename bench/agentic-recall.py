#!/usr/bin/env python3
"""Does a long agentic context still get *recalled*? The glyph counters cannot see this.

The 2026-09-17 incident was judged by counters that count damage -- U+FFFD, a
jamo welded to a syllable, a Han run standing in for Korean. Every one of its
derailed answers scores clean on those counters while the text says the wrong
thing: invented words, the wrong referent, a topic that drifts one sentence in.
`bench/korean-corruption.py` answers "is the script intact"; nothing answers
"did the answer use the context it was given".

This tool asks the second question, deterministically and without a judge. It
builds one long agentic context in the shape the product sends -- system prompt,
memory/session records, `[ctx] [assistant]` reasoning records, tool JSON,
document text -- buries one exact fact (a code, a date) in the oldest third, adds
three decoys in the newest third, and requires the answer to state that fact and
neither decoy. Glyph counters are reported beside it, so one run shows both.

    python3 bench/agentic-recall.py 2 40000 --url http://127.0.0.1:8000

The exit status is about the tool, not the boot: a boot that cannot recall the
fact is a failed gate and prints one line per round saying so.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"
DECOYS = 3
QUESTION = "위 기록들에서 이번 분기 예산 코드와 확정일 두 값만 정확히 알려줘. 다른 값과 섞지 말고 한 줄로."
SYSTEM = "You are a Korean assistant working from the session records below. Answer in Korean."

RECORD_TEMPLATES = (
    '- source=session ref="main#{n}/assistant" confidence={conf} age={age}d date=2026-0{m}-{dd}',
    '  cl:main#{n}/assistant {korean}',
    '[ctx] [assistant] The user asks "{ask}" — recall from the records above, no tools needed.',
    '**[assistant]** {korean}',
    '<tool_call>{{"name": "blackboard", "action": "put", "key": "draft.{n}", "value": "{korean}"}}</tool_call>',
    '- source=session ref="main#{n}/user" confidence=high age={age}d date=2026-0{m}-{dd}',
    '  cl:main#{n}/user {korean}',
    '<tool_call>{{"name": "sessions", "action": "search", "query": "{ask}"}}</tool_call>',
)
KOREAN = (
    "이번 주 안으로 표를 정리해서 공유하기로 했다. 금액과 기한은 담당자 확인 뒤 확정한다.",
    "지난 분기와 비교해서 세 항목만 다르게 잡았다. 나머지는 그대로 유지한다.",
    "회의는 다음 주 화요일로 옮기고, 자료는 하루 전까지 올린다.",
    "견적 단가는 유지하되 수량 구간만 조정한다. 조건은 문서에 적어 둔다.",
    "확인이 필요한 항목 세 개를 목록으로 남겼다. 답이 오면 바로 반영한다.",
)
ASKS = ("예산 코드가 뭐였지", "확정일 언제였지", "이번 분기 조건 정리해줘", "지난 회의 결론이 뭐였지")


def build(seed: int, chars: int) -> dict:
    """A deterministic long agentic context carrying one fact and three decoys."""
    rng = random.Random(seed)
    code = f"TS-{rng.randrange(1000, 10000)}"
    date = "2026-11-03"
    facts = [(code, date)]
    for _ in range(DECOYS):
        facts.append((f"TS-{rng.randrange(1000, 10000)}", f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"))
    records, n, written, buried = [], rng.randrange(100, 999), 0, [False] * len(facts)
    # the fact's line is the OLDEST third; each decoy sits in the newest third,
    # so an answer that skims the tail finds a decoy and not the fact
    burial_points = [chars // 3] + [chars - 200 * (i + 1) for i in range(DECOYS)]
    while written < chars:
        n += rng.randrange(1, 40)
        template = rng.choice(RECORD_TEMPLATES)
        line = template.format(n=n, conf=rng.choice(("low", "medium", "high")), age=rng.randrange(1, 400),
                               m=rng.randrange(1, 13), dd=rng.randrange(1, 29), korean=rng.choice(KOREAN),
                               ask=rng.choice(ASKS))
        records.append(line + "\n")
        written += len(line) + 1
        for index, point in enumerate(burial_points):
            if not buried[index] and written >= point:
                buried[index] = True
                fact_code, fact_date = facts[index]
                records.append(f'  cl:main#{n}/user 이번 분기 예산 코드는 {fact_code}, 확정일은 {fact_date}로 확정한다.\n')
    body = "".join(records)
    return dict(code=code, date=date, decoys=[c for c, _ in facts[1:]], decoy_dates=[d for _, d in facts[1:]],
                system=SYSTEM, body=body, question=QUESTION)


def grade(case: dict, answer: str) -> dict:
    """The exact fact stated, no decoy stated. No judge, no tolerance."""
    hit_code = case["code"] in answer
    hit_date = case["date"] in answer
    decoy = [c for c in case["decoys"] if c in answer] + [d for d in case["decoy_dates"] if d in answer]
    return dict(recalled=hit_code and hit_date, code=hit_code, date=hit_date, decoys_stated=sorted(decoy))


def glyph_scan(text: str) -> dict:
    """The counters `bench/korean-corruption.py` gates on, reported for context."""
    welded = len(re.findall(r"[\uac00-\ud7a3][\u1100-\u11ff\u3130-\u318f]", text))
    return dict(replacement=text.count("\ufffd"), welded_jamo=welded,
                han=len(re.findall(r"[\u4e00-\u9fff]", text)),
                cyrillic=len(re.findall(r"[\u0400-\u04ff]", text)),
                thai=len(re.findall(r"[\u0e00-\u0e7f]", text)),
                control=sum(1 for c in text if unicodedata.category(c) == "Cc" and c not in "\t\n\r"))


def ask(url: str, case: dict, prompt_tokens: int, temperature: float, max_tokens: int) -> tuple:
    body = json.dumps({"model": "glm-5.3-flash", "temperature": temperature, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": case["system"] + "\n\n" + case["body"]},
                                    {"role": "user", "content": case["question"]}]}).encode()
    request = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)
    message = payload["choices"][0]["message"]
    answer = message.get("content") or ""
    usage = payload.get("usage") or {}
    return answer, usage.get("prompt_tokens", prompt_tokens), usage.get("completion_tokens", 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rounds", nargs="?", type=int, default=1)
    parser.add_argument("chars", nargs="?", type=int, default=120000, help="target context characters")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows, recalled = [], 0
    for r in range(args.rounds):
        case = build(args.seed + r, args.chars)
        answer, prompt_tokens, completion = ask(args.url, case, 0, args.temperature, args.max_tokens)
        verdict = grade(case, answer)
        recalled += bool(verdict["recalled"])
        row = dict(round=r, seed=args.seed + r, prompt_tokens=prompt_tokens, completion_tokens=completion,
                   recalled=verdict["recalled"], code=verdict["code"], date=verdict["date"],
                   decoys_stated=verdict["decoys_stated"], glyph=glyph_scan(answer),
                   answer_sha256=hashlib.sha256(answer.encode()).hexdigest(), seconds=None)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print(f"recall {recalled}/{args.rounds} · 문맥 {args.chars:,}자 · T={args.temperature}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
    return 0 if recalled == args.rounds else 1


if __name__ == "__main__":
    raise SystemExit(main())
