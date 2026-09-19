#!/usr/bin/env python3
"""Live-eval prompts from Deneb's own text (operator 2026-09-19: "데네브 대화도 써도 돼. 기록도 써도되고") -- decoded on
the fleet, never sent out, never committed. Runs on srv4 beside deneb_extract.py, whose parsing it reuses.

    deneb_eval.py V2_DIR [CONVOS=30] [DOCS=17] > deneb_eval.jsonl

- conversations: sessions that no data boot used (deneb_convos.jsonl's cids hold each used session's file name), one
  prompt a session -- up to two earlier exchanges and the user turn that the transcript answered next, under Deneb's
  system prompt. No tool calls or results anywhere in the context: this window's fleet answers tools with 400 (#1282
  is not in its tree). Ids "d-<n>" (no session names).
- records: wiki pages, files and code that deneb_raw.jsonl does not hold (matched on their opening text), 3-12K
  characters, each with a task a person would ask of it (summary, decisions, a newcomer's explanation, a table of
  figures, review). Ids "r-<n>".
"""
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter

d = sys.argv[1]
N_CONVOS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
N_DOCS = int(sys.argv[3]) if len(sys.argv) > 3 else 17
rng = random.Random(20260920)

# deneb_extract.py runs its extraction at import: load only its definitions
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deneb_extract.py")).read()
src = src.split("\nprint(json.dumps({\"transcripts\"")[0]
mod = {"__name__": "deneb_extract_defs"}
sys_argv, sys.argv = sys.argv, ["deneb_extract.py", "/tmp/deneb-eval-unused"]
exec(compile(src, "deneb_extract.py", "exec"), mod)
sys.argv = sys_argv
HOME, system, blocks_to_messages, read, cap = mod["HOME"], mod["system"], mod["blocks_to_messages"], mod["read"], mod["cap"]
SKIP_DIRS, pdf_text, docx_text, hwpx_text = mod["SKIP_DIRS"], mod["pdf_text"], mod["docx_text"], mod["hwpx_text"]


def korean(text):
    return len(re.findall(r"[가-힣]", text)) > 0.2 * max(1, len(re.sub(r"\W", "", text)))


used_sessions = {json.loads(l)["cid"][len("deneb-"):].rsplit("-", 1)[0] for l in open(f"{d}/deneb_convos.jsonl")}
used_heads = {json.loads(l)["text"][:300] for l in open(f"{d}/deneb_raw.jsonl")}

# -- conversations ---------------------------------------------------------------------------------------------------------
files = sorted(f for f in os.listdir(os.path.join(HOME, "transcripts")) if f.endswith(".jsonl"))
rng.shuffle(files)
convos, skipped = [], Counter()
for f in files:
    if len(convos) >= N_CONVOS:
        break
    if f[:40] in used_sessions:
        skipped["used session"] += 1
        continue
    records = []
    for line in open(os.path.join(HOME, "transcripts", f), errors="replace"):
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    messages, _ = blocks_to_messages(records)
    ends = [i for i, m in enumerate(messages[:-1]) if m["role"] == "user" and messages[i + 1]["role"] == "assistant"
            and len((m.get("content") or "").strip()) >= 12]
    rng.shuffle(ends)
    pick = None
    for i in ends:
        users = [j for j in range(i + 1) if messages[j]["role"] == "user"]
        start = users[-3] if len(users) >= 3 else users[0]           # up to two earlier exchanges
        ctx = messages[start:i + 1]
        if any(m["role"] == "tool" or m.get("tool_calls") for m in ctx):
            continue
        size = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") for m in ctx)
        if size > 12000:
            continue
        pick = ctx
        break
    if pick is None:
        skipped["no clean turn"] += 1
        continue
    answer = messages[messages.index(pick[-1]) + 1] if pick[-1] in messages else None
    text = " ".join(m.get("content") or "" for m in pick)
    convos.append({"id": f"d-{len(convos):02d}", "kind": "d-chat", "suite": "deneb",
                   "messages": ([{"role": "system", "content": system}] if system else []) + pick,
                   "content": pick[-1]["content"], "max_tokens": 512,
                   "thinking": bool(answer and answer.get("reasoning_content")), "temperature": 0.0, "split": "eval",
                   "lang": "ko" if korean(text) else "en", "category": "deneb_" + f.split(":")[0],
                   "system": bool(system)})

# -- records ---------------------------------------------------------------------------------------------------------------
TASKS = {
    "ko": ["다음 문서를 다섯 줄로 요약하고, 정해진 일이나 해야 할 일이 있으면 따로 목록으로 뽑아줘.",
           "이 문서를 처음 보는 신입에게 설명하듯 핵심만 정리해줘.",
           "이 문서에 나오는 중요한 숫자와 날짜를 표로 정리해줘.",
           "이 문서 내용으로 확인 질문 세 개와 모범 답을 만들어줘."],
    "en": ["Summarize this document in five lines, then list any decisions or action items separately.",
           "Explain the key points of this document to someone new to the team.",
           "Put the important figures and dates in this document into a table.",
           "Write three check questions about this document, with model answers."],
    "code": {"ko": "이 코드가 하는 일을 설명하고, 고치거나 개선할 점 세 가지를 짚어줘.",
             "en": "Explain what this code does and point out three things to fix or improve."},
}
budget = {"wiki": N_DOCS * 7 // 17, "files": N_DOCS * 6 // 17}
budget["code"] = N_DOCS - budget["wiki"] - budget["files"]
code_ext = {"go", "kt", "tsx", "ts", "sql", "sh", "py", "js", "rs", "java"}
docs = []
for source, want in budget.items():
    paths = []
    for dirpath, _dirs, names in os.walk(os.path.join(HOME, source)):
        if SKIP_DIRS.search(dirpath + "/"):
            continue
        paths += [os.path.join(dirpath, n) for n in names]
    rng.shuffle(paths)
    got = 0
    for p in paths:
        if got >= want:
            break
        ext = p.rsplit(".", 1)[-1].lower() if "." in os.path.basename(p) else ""
        if source == "wiki" and ext != "md":
            continue
        if source == "code" and ext not in code_ext:
            continue
        if source == "files":
            text = {"pdf": pdf_text, "docx": docx_text, "hwpx": hwpx_text}.get(ext, lambda q: read(q) if ext in ("txt", "md", "csv") else "")(p)
        else:
            text = read(p)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not 3000 <= len(text) <= 12000 or "\x00" in text or re.search(r"[A-Za-z0-9+/=]{400,}", text):
            continue
        if cap(text, 8000)[:300] in used_heads:
            skipped["doc prefilled"] += 1
            continue
        lang = "ko" if korean(text) else "en"
        task = TASKS["code"][lang] if source == "code" else TASKS[lang][got % 4]
        body = f"{task}\n\n```{ext}\n{text}\n```" if source == "code" else f"{task}\n\n---\n{text}"
        docs.append({"id": f"r-{len(docs):02d}", "kind": f"r-{source}", "suite": "deneb_records",
                     "messages": [{"role": "user", "content": body}], "content": body, "max_tokens": 512,
                     "thinking": False, "temperature": 0.0, "split": "eval", "lang": lang,
                     "category": f"deneb_{source}", "system": False})
        got += 1

for r in convos + docs:
    print(json.dumps(r, ensure_ascii=False))
print(json.dumps({"convos": len(convos), "docs": len(docs), "skipped": dict(skipped),
                  "convo_lang": dict(Counter(r["lang"] for r in convos)),
                  "convo_channels": dict(Counter(r["category"] for r in convos)),
                  "convo_thinking": sum(r["thinking"] for r in convos),
                  "convo_chars_max": max((sum(len(m.get("content") or "") for m in r["messages"]) for r in convos), default=0),
                  "doc_kinds": dict(Counter(r["kind"] for r in docs)), "doc_lang": dict(Counter(r["lang"] for r in docs))}),
      file=sys.stderr)
