#!/usr/bin/env python3
"""Deneb's own text for the MTP head (operator 2026-09-19: "데네브에 통과시켜서 원문 일부 뽑는건?" -- wiki, transcripts,
code and files approved; used on the fleet for prefill only, never sent out). Runs on srv4, stdlib + pdftotext.

    deneb_extract.py OUT_DIR [CONVO_CHARS=1300000] [RAW_CHARS=950000]

    deneb_convos.jsonl   transcript windows as chat conversations: Deneb's system prompt (its SOUL/IDENTITY/USER/AGENTS/
                         TOOLS files), the tools it called (schemas rebuilt from the calls), user turns, the assistant's
                         thinking, text and tool calls, tool results -- each window ending on an assistant turn
    deneb_raw.jsonl      wiki pages, source files and document text, as raw text
"""
import json
import os
import random
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict

HOME = os.path.expanduser("~/.deneb")
OUT = sys.argv[1]
CONVO_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 1300000
RAW_CHARS = int(sys.argv[3]) if len(sys.argv) > 3 else 950000
TOOL_RESULT_CAP, TEXT_CAP, WINDOW_CAP = 4000, 8000, 16000
rng = random.Random(20260919)
os.makedirs(OUT, exist_ok=True)


def read(path):
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def cap(text, n):
    return text if len(text) <= n else text[:n] + "\n...(truncated)"


# -- Deneb's system prompt ------------------------------------------------------------------------------------------------
system = "\n\n".join(read(os.path.join(HOME, f)).strip() for f in ("SOUL.md", "IDENTITY.md", "USER.md", "AGENTS.md", "TOOLS.md")
                     if read(os.path.join(HOME, f)).strip())


# -- transcripts -> conversations -----------------------------------------------------------------------------------------
def blocks_to_messages(records):
    """Anthropic-style records -> OpenAI messages: thinking -> reasoning_content, tool_use -> tool_calls, tool_result ->
    role tool."""
    out, schemas = [], defaultdict(dict)
    for r in records:
        role, content = r.get("role"), r.get("content")
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str):
            if content.strip():
                out.append({"role": role, "content": cap(content, TEXT_CAP)})
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            text, reasoning, calls = [], [], []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text" and b.get("text"):
                    text.append(b["text"])
                elif t == "thinking":
                    th = b.get("thinking") or b.get("text") or ""
                    if th:
                        reasoning.append(th)
                elif t == "tool_use" and b.get("name"):
                    args = b.get("input") if isinstance(b.get("input"), dict) else {}
                    calls.append({"id": str(b.get("id") or f"call_{len(calls)}"), "type": "function",
                                  "function": {"name": b["name"], "arguments": cap(json.dumps(args, ensure_ascii=False), TEXT_CAP)}})
                    for k, v in args.items():
                        schemas[b["name"]][k] = {bool: "boolean", int: "integer", float: "number", list: "array",
                                                 dict: "object"}.get(type(v), "string")
            m = {"role": "assistant", "content": cap("\n\n".join(text), TEXT_CAP)}
            if reasoning:
                m["reasoning_content"] = cap("\n\n".join(reasoning), TEXT_CAP)
            if calls:
                m["tool_calls"] = calls
            if m["content"] or reasoning or calls:
                out.append(m)
        else:
            texts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                    out.append({"role": "tool", "tool_call_id": str(b.get("tool_use_id") or ""), "content": cap(str(c or ""), TOOL_RESULT_CAP)})
                elif b.get("type") == "text" and b.get("text"):
                    texts.append(b["text"])
            if texts:
                out.append({"role": "user", "content": cap("\n\n".join(texts), TEXT_CAP)})
    return out, schemas


def windows(messages):
    """Windows of whole exchanges: start at a user turn, end on an assistant turn with text or thinking, at most
    WINDOW_CAP characters -- a tool message never starts one, and every tool call's result stays with it."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
    for s in starts:
        size, end = 0, None
        for j in range(s, len(messages)):
            m = messages[j]
            size += len(m.get("content") or "") + len(m.get("reasoning_content") or "") + \
                sum(len(c["function"]["arguments"]) for c in m.get("tool_calls") or [])
            if size > WINDOW_CAP:
                break
            nxt = messages[j + 1] if j + 1 < len(messages) else None
            if m["role"] == "assistant" and not m.get("tool_calls") and (m.get("content") or m.get("reasoning_content")) \
                    and (nxt is None or nxt["role"] == "user"):
                end = j
        if end is not None:
            yield messages[s:end + 1], size


def transcripts():
    files = sorted(f for f in os.listdir(os.path.join(HOME, "transcripts")) if f.endswith(".jsonl"))
    rng.shuffle(files)
    picked, total, all_schemas = [], 0, defaultdict(dict)
    per_file = Counter()
    candidates = []
    for f in files:
        records = []
        for line in open(os.path.join(HOME, "transcripts", f), errors="replace"):
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
        messages, schemas = blocks_to_messages(records)
        for k, v in schemas.items():
            all_schemas[k].update(v)
        ws = list(windows(messages))
        rng.shuffle(ws)
        candidates += [(f, w, size) for w, size in ws[:6]]           # at most six windows a session: many sessions
    rng.shuffle(candidates)
    tools = [{"type": "function", "function": {"name": n, "description": "",
                                               "parameters": {"type": "object", "properties": {k: {"type": t} for k, t in props.items()}}}}
             for n, props in sorted(all_schemas.items())]
    for f, w, size in candidates:
        if total >= CONVO_CHARS:
            break
        if per_file[f] >= 3:
            continue
        per_file[f] += 1
        total += size
        used = {c["function"]["name"] for m in w for c in m.get("tool_calls") or []}
        final = w[-1]
        if not (final.get("content") or "").strip():
            final["content"] = " "                                   # thinking only: resumed after one space
        text = " ".join(m.get("content") or "" for m in w)
        picked.append({"cid": f"deneb-{f[:40]}-{per_file[f]}", "shape": "deneb", "category": f.split(":")[0],
                       "lang": "ko" if len(re.findall(r"[가-힣]", text)) > 0.2 * max(1, len(re.sub(r"\W", "", text))) else "en",
                       "thinking": any(m.get("reasoning_content") for m in w),
                       "messages": ([{"role": "system", "content": system}] if system else []) + w,
                       "tools": [t for t in tools if t["function"]["name"] in used] or None,
                       "final": {"temperature": None}})
    with open(os.path.join(OUT, "deneb_convos.jsonl"), "w") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"sessions": len(files), "windows": len(picked), "chars": total, "sessions_used": len(per_file),
            "tools_rebuilt": len(tools), "system_prompt_chars": len(system)}


# -- raw text -------------------------------------------------------------------------------------------------------------
SKIP_DIRS = re.compile(r"/(node_modules|\.git|vendor|dist|build|target|__pycache__|\.venv|generated|gen|third_party)/")


def docx_text(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
    except Exception:                                                # noqa: BLE001
        return ""
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)


def hwpx_text(path):
    try:
        with zipfile.ZipFile(path) as z:
            parts = [z.read(n).decode("utf-8", "replace") for n in z.namelist() if re.match(r"Contents/section\d+\.xml", n)]
    except Exception:                                                # noqa: BLE001
        return ""
    xml = re.sub(r"</hp:p>", "\n", "".join(parts))
    return re.sub(r"<[^>]+>", "", xml)


def pdf_text(path):
    try:
        return subprocess.run(["pdftotext", "-l", "6", "-q", path, "-"], capture_output=True, text=True, timeout=30).stdout
    except Exception:                                                # noqa: BLE001
        return ""


def raw():
    docs = []
    budgets = {"wiki": 0.35, "code": 0.35, "files": 0.30}
    code_ext = {"go", "kt", "tsx", "ts", "sql", "sh", "py", "md", "js", "rs", "java", "yaml", "yml"}
    for source, share in budgets.items():
        root = os.path.join(HOME, source)
        paths = []
        for dirpath, _dirs, names in os.walk(root):
            if SKIP_DIRS.search(dirpath + "/"):
                continue
            for n in names:
                paths.append(os.path.join(dirpath, n))
        rng.shuffle(paths)
        if source == "code":                                          # every language gets its turn
            by = defaultdict(list)
            for p in paths:
                ext = p.rsplit(".", 1)[-1].lower() if "." in os.path.basename(p) else ""
                if ext in code_ext and os.path.getsize(p) <= 40000:
                    by[ext].append(p)
            paths = []
            while any(by.values()):
                for ext in sorted(by):
                    if by[ext]:
                        paths.append(by[ext].pop())
        budget, used = RAW_CHARS * share, 0
        for p in paths:
            if used >= budget:
                break
            ext = p.rsplit(".", 1)[-1].lower() if "." in os.path.basename(p) else ""
            if source == "wiki" and ext != "md":
                continue
            if source == "files":
                text = {"pdf": pdf_text, "docx": docx_text, "hwpx": hwpx_text}.get(ext, lambda q: read(q) if ext in ("txt", "md", "csv") else "")(p)
            else:
                text = read(p)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) < 400 or "\x00" in text or re.search(r"[A-Za-z0-9+/=]{400,}", text):   # empty, binary, base64 blobs
                continue
            text = cap(text, 8000)
            docs.append({"rid": f"deneb-{source}-{len(docs)}", "category": f"deneb_{source}", "lang":
                         "ko" if len(re.findall(r"[가-힣]", text)) > 0.2 * max(1, len(re.sub(r"\W", "", text))) else "en",
                         "text": text})
            used += len(text)
    with open(os.path.join(OUT, "deneb_raw.jsonl"), "w") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return {"docs": len(docs), "by_source": Counter(d["category"] for d in docs), "chars": sum(len(d["text"]) for d in docs)}


print(json.dumps({"transcripts": transcripts(), "raw": raw()}, ensure_ascii=False, default=str))
