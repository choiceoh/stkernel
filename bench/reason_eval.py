#!/usr/bin/env python3
"""37차: a quick logical-reasoning check of the served model (operator: "모델 논리 능력
테스트해봐"). Four small sets, temperature 0, the server's default thinking mode:

  gsm8k      openai/gsm8k test, exact numeric match (English word problems)
  mmlu_logic cais/mmlu formal_logic test, 4-way MCQ letter match
  kmmlu      HAERAE-HUB/KMMLU (a few subjects), 4-way MCQ (Korean), skipped if not loadable
  ko_puzzles 20 hand-written Korean logic puzzles with fixed answers

Prints accuracy per set, tokens per answer and latency, then the first failures.
    BENCH_URL=http://10.10.10.2:8000/v1/chat/completions python3 bench/reason_eval.py --n 60 --conc 4
"""
import argparse, json, os, re, sys, time, random
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import resolve_model  # noqa: E402

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = resolve_model(os.environ.get("BENCH_MODEL", "glm-5.3-flash"), URL)

KO_PUZZLES = [
    ("철수는 영희보다 키가 크고, 영희는 민수보다 키가 크다. 세 명 중 가장 작은 사람은?", "민수"),
    ("모든 고래는 포유류이고, 모든 포유류는 폐로 숨을 쉰다. 그러면 고래는 폐로 숨을 쉬는가? 예/아니오로 답하라.", "예"),
    ("어떤 마을에서 기사는 항상 참말만 하고 악당은 항상 거짓말만 한다. A가 \"우리 둘 다 악당이다\"라고 말했다. A는 기사인가 악당인가?", "악당"),
    ("5명이 원탁에 앉는다. 갑은 을의 바로 왼쪽, 을은 병의 바로 왼쪽에 앉는다. 병의 바로 오른쪽에 앉은 사람은? (왼쪽/오른쪽은 시계 반대/시계 방향)", "을"),
    ("1, 4, 9, 16, 25 다음에 오는 수는?", "36"),
    ("어제의 이틀 뒤가 목요일이라면 오늘은 무슨 요일인가?", "수요일"),
    ("사과 3개와 배 2개의 값이 1,300원이고, 사과 1개와 배 1개의 값이 500원이면 사과 1개의 값은?", "300"),
    ("한 상자에 빨간 공 3개, 파란 공 2개가 있다. 공을 하나 꺼냈을 때 빨간 공일 확률을 기약분수로 쓰면?", "3/5"),
    ("모든 새는 날 수 있다는 명제가 거짓임을 보이는 반례로 알맞은 것은? (1) 참새 (2) 펭귄 (3) 독수리 (4) 비둘기. 번호로 답하라.", "2"),
    ("A는 B의 아버지이고 B는 C의 어머니다. A는 C에게 무엇인가?", "할아버지"),
    ("3진법 수 102를 십진법으로 바꾸면?", "11"),
    ("두 수의 합이 20이고 차가 4일 때 큰 수는?", "12"),
    ("만약 비가 오면 땅이 젖는다. 땅이 젖지 않았다. 비가 왔는가? 예/아니오로 답하라.", "아니오"),
    ("한 줄에 6명이 서 있다. 지수는 앞에서 두 번째, 뒤에서 몇 번째인가?", "5"),
    ("12시를 가리키는 시계가 있다. 시침이 90도 돌면 몇 시인가?", "3"),
    ("빨강, 파랑, 노랑 모자를 쓴 세 사람 중 A는 빨강이 아니고 B는 파랑이 아니며 C는 빨강이다. B의 모자 색은?", "노랑"),
    ("어떤 수에 3을 곱한 뒤 7을 빼면 20이다. 그 수는?", "9"),
    ("영어 문장 'No cats are dogs'의 대우(contrapositive)와 같은 뜻의 문장은? (1) 어떤 개는 고양이다 (2) 어떤 개도 고양이가 아니다 (3) 모든 고양이는 개다. 번호로 답하라.", "2"),
    ("10명이 서로 한 번씩 악수하면 악수는 총 몇 번인가?", "45"),
    ("동전을 두 번 던져 적어도 한 번 앞면이 나올 확률을 기약분수로 쓰면?", "3/4"),
]


def ask(prompt, max_tokens):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning") or m.get("reasoning_content") or ""), \
        d.get("usage", {}).get("completion_tokens", 0), time.time() - t, d["choices"][0].get("finish_reason")


def last_number(s):
    s = s.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None


def norm_num(x):
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return (x or "").strip()


def letter(s):
    m = re.findall(r"\b([ABCD])\b", s.strip().upper())
    return m[-1] if m else None


def load_sets(n, seed):
    rng = random.Random(seed)
    sets = []
    try:
        from datasets import load_dataset
        g = load_dataset("openai/gsm8k", "main", split="test")
        idx = rng.sample(range(len(g)), min(n, len(g)))
        items = []
        for i in idx:
            q, a = g[i]["question"], g[i]["answer"].split("####")[-1].strip()
            items.append((q + "\n\nSolve step by step, then end with a final line 'Answer: <number>'.", a, "num"))
        sets.append(("gsm8k", items))
    except Exception as e:  # noqa: BLE001
        print("gsm8k unavailable:", repr(e)[:120], flush=True)
    try:
        from datasets import load_dataset
        m = load_dataset("cais/mmlu", "formal_logic", split="test")
        idx = rng.sample(range(len(m)), min(n, len(m)))
        items = []
        for i in idx:
            r = m[i]
            opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", r["choices"]))
            items.append((f"{r['question']}\n{opts}\n\nThink it through, then end with a final line 'Answer: <letter>'.",
                          "ABCD"[r["answer"]], "letter"))
        sets.append(("mmlu_formal_logic", items))
    except Exception as e:  # noqa: BLE001
        print("mmlu unavailable:", repr(e)[:120], flush=True)
    try:
        from datasets import load_dataset
        items = []
        for subj in ("Math", "Law", "Korean-History"):
            try:
                k = load_dataset("HAERAE-HUB/KMMLU", subj, split="test")
            except Exception:
                continue
            for i in rng.sample(range(len(k)), min(max(n // 3, 5), len(k))):
                r = k[i]
                opts = "\n".join(f"{L}. {r[L]}" for L in "ABCD")
                items.append((f"{r['question']}\n{opts}\n\n단계적으로 생각한 뒤 마지막 줄에 '정답: <A/B/C/D>' 형식으로 답하라.",
                              "ABCD"[int(r["answer"]) - 1], "letter"))
        if items:
            sets.append(("kmmlu", items))
    except Exception as e:  # noqa: BLE001
        print("kmmlu unavailable:", repr(e)[:120], flush=True)
    sets.append(("ko_puzzles", [(q + "\n\n짧게 추론한 뒤 마지막 줄에 '정답: <답>' 형식으로 답하라.", a, "text") for q, a in KO_PUZZLES]))
    return sets


def grade(kind, content, gold):
    tail = content.strip().splitlines()[-1] if content.strip() else ""
    ans_line = tail
    for line in reversed(content.strip().splitlines()):
        if re.search(r"(Answer|정답)\s*[:：]", line):
            ans_line = line.split(":", 1)[-1].split("：", 1)[-1]
            break
    if kind == "num":
        return norm_num(last_number(ans_line)) == norm_num(gold), ans_line.strip()
    if kind == "letter":
        return letter(ans_line) == gold, ans_line.strip()
    g = gold.replace(" ", "")
    a = ans_line.replace(" ", "").replace("**", "")
    if re.fullmatch(r"[\d/.]+", g):
        return (g in a) or (norm_num(last_number(a)) == norm_num(g)), ans_line.strip()
    return g in a, ans_line.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.expanduser("~/glm53-logs/reason-eval.jsonl"))
    args = ap.parse_args()
    sets = load_sets(args.n, args.seed)
    print(f"sets: {[(n, len(i)) for n, i in sets]} conc={args.conc} url={URL}", flush=True)
    results = {}
    for name, items in sets:
        t0 = time.time()
        with ThreadPoolExecutor(args.conc) as ex:
            outs = list(ex.map(lambda it: ask(it[0], args.max_tokens), items))
        rows = []
        for (prompt, gold, kind), (content, reasoning, ntok, dt, fin) in zip(items, outs):
            ok, ans = grade(kind, content, gold)
            rows.append({"ok": ok, "gold": gold, "ans": ans[:80], "tokens": ntok, "s": round(dt, 1),
                         "finish": fin, "thinking_chars": len(reasoning), "q": prompt[:100]})
        acc = sum(r["ok"] for r in rows) / max(len(rows), 1)
        toks = sum(r["tokens"] for r in rows) / max(len(rows), 1)
        lat = sum(r["s"] for r in rows) / max(len(rows), 1)
        trunc = sum(1 for r in rows if r["finish"] == "length")
        results[name] = {"n": len(rows), "acc": round(acc, 3), "tokens_mean": round(toks), "latency_mean_s": round(lat, 1),
                         "truncated": trunc, "wall_s": round(time.time() - t0)}
        print(f"{name:<18} n={len(rows):<4} acc={acc*100:5.1f}%  tokens/answer={toks:6.0f}  latency={lat:5.1f}s  "
              f"truncated={trunc}  wall={time.time()-t0:.0f}s", flush=True)
        for r in [r for r in rows if not r["ok"]][:4]:
            print(f"   miss: gold={r['gold']!r} got={r['ans']!r} finish={r['finish']} | {r['q'][:70]!r}", flush=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "set": name, "summary": results[name], "rows": rows}, ensure_ascii=False) + "\n")
    print("SUMMARY", json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
