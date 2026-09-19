"""Three synthetic prompts whose no-thinking greedy answers had domain errors, re-asked with thinking on and a large budget.
Key from the environment (gen2.sh style), never printed. Output: reasoning length, finish reason, answer text (names masked)."""
import json, re, sys, threading
sys.path.insert(0, "/home/choiceoh/q38mtp-gen2")
import datagen2
names = "오선택 오형석 박환민 김성훈 이윤상 차남두 김건호 박화숙 조은실 백창선 공명한 오용현 박민수".split()
mask = lambda s: re.sub("|".join(names), "○○", s or "")
rows = [json.loads(l) for l in open(sys.argv[1])]
out = [None] * len(rows)
def work(i):
    r = rows[i]
    try:
        a = datagen2.chat(r["messages"], temperature=0.0, max_tokens=8192, thinking=True)
        c = a["choices"][0]
        out[i] = {"cid": r["cid"], "finish": c.get("finish_reason"), "usage": a.get("usage"),
                  "reasoning": c["message"].get("reasoning") or "", "content": c["message"].get("content") or ""}
    except Exception as exc:
        out[i] = {"cid": r["cid"], "error": repr(exc)[:200]}
ts = [threading.Thread(target=work, args=(i,)) for i in range(len(rows))]
[t.start() for t in ts]; [t.join() for t in ts]
for o in out:
    print("=" * 90)
    if "error" in o:
        print(o); continue
    print("[", o["cid"], "| finish", o["finish"], "| reasoning chars", len(o["reasoning"]), "| content chars", len(o["content"]), "| usage", {k: v for k, v in (o["usage"] or {}).items() if "tokens" in k and isinstance(v, int)}, "]")
    print("--- REASONING (head 1200):", mask(o["reasoning"])[:1200].replace("\n", " "))
    print("--- ANSWER:"); print(mask(o["content"])[:2600])
