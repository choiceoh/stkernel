#!/usr/bin/env python3
"""Self-distribution data v3 for the MTP head (operator 2026-09-19: "2번 작업으로 빨리 넘어가는게 나을것 같기도", then "a" --
Deneb's prompts may go to OpenRouter's qwen/qwen3.8-flash). Window 5 showed why: the head drafts only inside the served
model's own answers, and the v2 Deneb prefill was mostly other text (user turns, tool results, another model's answers).
Here real prompts are answered by the served model itself; the fleet prefills them with the tap on and the trainer can
keep the answer positions.

    selfgen3.py prompts V2_DIR OUT_DIR [PER_SESSION=4] [DOCS=200]
        deneb_prompts.jsonl  Deneb sessions outside the live-eval set: user turns the transcript answered, each with up
                             to two earlier exchanges (tool calls and results kept) under Deneb's system prompt and the
                             tools the session used; tasks over wiki/files/code outside the eval documents
    selfgen3.py answer OUT_DIR IN.jsonl OUT.jsonl
        each prompt answered once at a served setting (greedy 60%, T=1 top-k 20 top-p 0.95 40%), thinking as the
        transcript's answer had it -- resumable (ids already in OUT are skipped)

Runs on srv4 beside deneb_extract.py and datagen2.py (OPENROUTER_API_KEY from the environment, never printed). Prints
counts only: Deneb's text stays on srv4/srv2 and out of the repo.
"""
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def extract_defs():
    src = open(os.path.join(HERE, "deneb_extract.py")).read().split("\nprint(json.dumps({\"transcripts\"")[0]
    mod = {"__name__": "deneb_extract_defs"}
    argv, sys.argv = sys.argv, ["deneb_extract.py", "/tmp/deneb-selfgen-unused"]
    exec(compile(src, "deneb_extract.py", "exec"), mod)
    sys.argv = argv
    return mod


def korean(text):
    return len(re.findall(r"[가-힣]", text)) > 0.2 * max(1, len(re.sub(r"\W", "", text)))


def prompts(v2_dir, out_dir, per_session=4, n_docs=200):
    x = extract_defs()
    rng = random.Random(20260921)
    os.makedirs(out_dir, exist_ok=True)
    # the live-eval set stays unseen: sessions holding any of its user turns, and its documents
    eval_rows = [json.loads(l) for l in open(os.path.join(v2_dir, "deneb_eval.jsonl"))]
    eval_users = {m["content"].strip() for r in eval_rows if r["id"].startswith("d-")
                  for m in r["messages"] if m["role"] == "user" and (m.get("content") or "").strip()}
    eval_doc_heads = {r["content"].split("\n", 1)[1][:400] for r in eval_rows if r["id"].startswith("r-")}
    files = sorted(f for f in os.listdir(os.path.join(x["HOME"], "transcripts")) if f.endswith(".jsonl"))
    sessions, schemas, skipped = [], defaultdict(dict), Counter()
    for f in files:
        records = []
        for line in open(os.path.join(x["HOME"], "transcripts", f), errors="replace"):
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
        messages, sch = x["blocks_to_messages"](records)
        for k, v in sch.items():
            schemas[k].update(v)
        if any((m.get("content") or "").strip() in eval_users for m in messages if m["role"] == "user"):
            skipped["eval session"] += 1
            continue
        sessions.append((f, messages))
    tools_all = {n: {"type": "function", "function": {"name": n, "description": "", "parameters": {
        "type": "object", "properties": {k: {"type": t} for k, t in props.items()}}}} for n, props in schemas.items()}
    out = []
    for f, messages in sessions:
        ends = [i for i, m in enumerate(messages[:-1]) if m["role"] == "user" and messages[i + 1]["role"] == "assistant"
                and len((m.get("content") or "").strip()) >= 12]
        if not ends:
            skipped["no answered turn"] += 1
            continue
        picks = sorted(rng.sample(ends, min(per_session, len(ends))))
        used = sorted({c["function"]["name"] for m in messages for c in m.get("tool_calls") or []})
        for i in picks:
            users = [j for j in range(i + 1) if messages[j]["role"] == "user"]
            start = users[-3] if len(users) >= 3 else users[0]
            ctx = messages[start:i + 1]
            size = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") +
                       sum(len(c["function"]["arguments"]) for c in m.get("tool_calls") or []) for m in ctx)
            if size > 16000:
                skipped["context over 16K chars"] += 1
                continue
            answer = messages[i + 1]
            text = " ".join(m.get("content") or "" for m in ctx)
            out.append({"id": f"dp-{len(out):05d}", "kind": "deneb", "category": "deneb_" + f.split(":")[0],
                        "lang": "ko" if korean(text) else "en",
                        "thinking": bool(answer.get("reasoning_content")),
                        "messages": ([{"role": "system", "content": x["system"]}] if x["system"] else []) + ctx,
                        "tools": [tools_all[n] for n in used if n in tools_all] or None})
    # tasks over Deneb's records, outside the eval documents
    tasks = {
        "ko": ["다음 문서를 다섯 줄로 요약해줘.", "이 문서에서 정해진 일과 해야 할 일을 목록으로 뽑아줘.",
               "이 문서를 처음 보는 사람에게 설명하듯 핵심을 정리해줘.", "이 문서의 중요한 숫자와 날짜를 표로 정리해줘.",
               "이 문서 내용으로 확인 질문 세 개와 모범 답을 만들어줘.", "이 문서의 문제점이나 빠진 부분을 짚어줘.",
               "이 문서를 바탕으로 관련 부서에 보낼 짧은 공지를 써줘.", "이 문서를 영어로 요약해줘."],
        "en": ["Summarize this document in five lines.", "List the decisions and action items in this document.",
               "Explain the key points to someone new to the team.", "Put the important figures and dates into a table.",
               "Write three check questions about this document, with model answers.",
               "Point out weaknesses or missing pieces in this document.",
               "Draft a short announcement to the relevant team based on this document.", "Summarize this document in Korean."],
        "code": {"ko": ["이 코드가 하는 일을 설명해줘.", "이 코드의 버그 가능성과 개선점을 짚어줘.", "이 코드에 대한 테스트를 작성해줘."],
                 "en": ["Explain what this code does.", "Point out likely bugs and improvements in this code.",
                        "Write tests for this code."]},
    }
    code_ext = {"go", "kt", "tsx", "ts", "sql", "sh", "py", "js", "rs", "java"}
    docs = 0
    for source, share in (("wiki", 0.4), ("files", 0.35), ("code", 0.25)):
        want, got = int(n_docs * share), 0
        paths = []
        for dirpath, _dirs, names in os.walk(os.path.join(x["HOME"], source)):
            if x["SKIP_DIRS"].search(dirpath + "/"):
                continue
            paths += [os.path.join(dirpath, n) for n in names]
        rng.shuffle(paths)
        for p in paths:
            if got >= want:
                break
            ext = p.rsplit(".", 1)[-1].lower() if "." in os.path.basename(p) else ""
            if (source == "wiki" and ext != "md") or (source == "code" and ext not in code_ext):
                continue
            if source == "files":
                text = {"pdf": x["pdf_text"], "docx": x["docx_text"], "hwpx": x["hwpx_text"]}.get(
                    ext, lambda q: x["read"](q) if ext in ("txt", "md", "csv") else "")(p)
            else:
                text = x["read"](p)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if not 2000 <= len(text) <= 14000 or "\x00" in text or re.search(r"[A-Za-z0-9+/=]{400,}", text):
                continue
            body_head = (f"```{ext}\n{text}\n```" if source == "code" else f"---\n{text}")[:400]
            if any(body_head[:200] in h for h in eval_doc_heads):
                skipped["eval document"] += 1
                continue
            lang = "ko" if korean(text) else "en"
            pool = tasks["code"][lang] if source == "code" else tasks[lang]
            task = pool[got % len(pool)]
            body = f"{task}\n\n```{ext}\n{text}\n```" if source == "code" else f"{task}\n\n---\n{text}"
            out.append({"id": f"rp-{docs:04d}", "kind": f"record-{source}", "category": f"deneb_{source}", "lang": lang,
                        "thinking": False, "messages": [{"role": "user", "content": body}], "tools": None})
            got += 1
            docs += 1
    with open(os.path.join(out_dir, "deneb_prompts.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    conv = [r for r in out if r["kind"] == "deneb"]
    print(json.dumps({"sessions": len(files), "usable_sessions": len(sessions), "prompts": len(out),
                      "conversation_prompts": len(conv), "record_prompts": docs, "skipped": dict(skipped),
                      "with_tools": sum(1 for r in conv if r["tools"]), "thinking": sum(r["thinking"] for r in conv),
                      "lang": dict(Counter(r["lang"] for r in out)),
                      "channels": dict(Counter(r["category"] for r in conv).most_common(8)),
                      "context_chars_p50": sorted(sum(len(m.get("content") or "") for m in r["messages"]) for r in out)[len(out) // 2]}))


def answer(out_dir, in_path, out_path):
    import datagen2
    rows = [json.loads(l) for l in open(in_path)]
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["cid"] for l in open(out_path)}
    jobs = [r for r in rows if r["id"] not in done]

    def work(r):
        h = int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16)
        setting = datagen2.SERVED[0] if h % 10 < 6 else datagen2.SERVED[1]      # greedy 60%, T=1 40%
        rng = random.Random(h)
        msg, meta = datagen2.served_answer(r["messages"], rng, thinking=r["thinking"], tools=r.get("tools"),
                                           max_tokens=1536, setting=setting)
        if not ((msg.get("content") or "").strip() or msg.get("reasoning_content") or msg.get("tool_calls")):
            raise ValueError("empty answer")
        if msg.get("reasoning_content") and not (msg.get("content") or "").strip() and not msg.get("tool_calls"):
            msg["content"] = " "                                   # a reasoning cut by max_tokens: kept, answer " "
        return [{"cid": r["id"], "shape": r["kind"], "category": r["category"], "lang": r["lang"], "thinking": r["thinking"],
                 "messages": r["messages"] + [msg], "tools": r.get("tools"), "final": meta}]

    print(json.dumps({"stage": "answer", "jobs": len(jobs), "already": len(done)}), flush=True)
    datagen2.parallel(jobs, work, out_path, "answer")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "prompts":
        prompts(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4,
                int(sys.argv[5]) if len(sys.argv) > 5 else 200)
    elif stage == "answer":
        answer(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
