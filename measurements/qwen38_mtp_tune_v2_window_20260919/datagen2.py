#!/usr/bin/env python3
"""The MTP head's second data set, built for quality (operator 2026-09-19: "프롬프트 생성기나 학습데이터 품질이 아주 중요해").

The head learns the target's distribution on whatever text the fleet prefills, so the text should look like what the
served model will see: real users' wording (not templates), system prompts, multi-turn conversations, tool calls with
their results, long pasted documents, thinking where a request calls for it, the model's own answers at the served
settings. Stages (stdlib only; OPENROUTER_API_KEY from the environment, never printed):

    datagen2.py seeds  OUT_DIR [CALLS]     a writer model drafts realistic user requests, cell by cell of a taxonomy
                                           (language x category x style), and tool requests and documents with questions
    datagen2.py convos OUT_DIR [LIMIT]     each seed becomes a conversation: single turn, multi-turn (the writer drafts
                                           the follow-ups), tool use (the model calls, the writer answers as the tool),
                                           a long document -- the served model (Qwen3.8-Flash-Next) writes every answer
    datagen2.py filter OUT_DIR             drop failures, loops, template debris and near-duplicates -> conversations.jsonl
    datagen2.py stats  OUT_DIR             what the set holds, and a sample to read

Each stage resumes (it skips what its output already holds).

The mix (operator: "적절히 섞어야해 무조건 서빙환경으로만 학습하면 일반 상황에서 과적합될수 있어서"): about 40% plain
single-turn chat with no system prompt, about 50% served shapes (a system prompt, multi-turn, tools, a long document), and
the documents alone as raw text (`raw.jsonl`, prefilled without a chat template) -- and the earlier sets stay in the
training mix beside this one.
"""
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

MODEL = os.environ.get("MODEL", "qwen/qwen3.8-flash")          # the served model's own weights (Qwen/Qwen3.8-Flash-Next)
WRITER = os.environ.get("WRITER", MODEL)
CONCURRENCY = int(os.environ.get("CONCURRENCY", "32"))
SERVED = [(0.0, None, None), (1.0, 20, 0.95)]                  # greedy, and the door's default (generation_config.json)
GREEDY_SHARE = 0.7        # operator: "t=1 학습데이터보다 t=0 학습데이터가 더 좋을수도" -- greedy-heavy, some T=1 kept

LANGS = [("ko", 0.6), ("en", 0.35), ("mixed", 0.05)]
CATEGORIES = [   # name, weight, what it covers (the writer reads this), thinking share
    ("general_qa", 8, "everyday factual questions people ask an assistant", 0.1),
    ("explain", 9, "asking for an explanation of a concept, at the asker's level", 0.15),
    ("howto", 6, "practical step-by-step help with a real task", 0.1),
    ("coding_write", 9, "asking for code: functions, scripts, small programs, with concrete requirements", 0.4),
    ("coding_debug", 6, "pasting code or an error message or a stack trace and asking what is wrong", 0.5),
    ("coding_review", 3, "asking for a review, refactor or explanation of pasted code", 0.3),
    ("data_sql", 4, "SQL, spreadsheets, pandas, data analysis questions with concrete tables", 0.4),
    ("math", 5, "math problems from arithmetic to calculus and probability, word problems", 0.7),
    ("reasoning", 3, "logic puzzles, judgement calls, planning under constraints", 0.7),
    ("writing_business", 7, "emails, reports, official notices, proposals, meeting minutes", 0.05),
    ("writing_creative", 3, "stories, poems, slogans, speeches", 0.05),
    ("summarize", 5, "pasting a text (an article, a meeting transcript, a report) and asking for a summary or key points", 0.1),
    ("translate_edit", 4, "translation between Korean and English, proofreading, changing tone or formality", 0.05),
    ("business_finance", 5, "management, marketing, pricing, accounting, investment questions", 0.3),
    ("legal_tax_admin", 4, "practical legal, tax and administrative questions (Korean context: 세무, 계약, 인허가, 노무)", 0.3),
    ("energy_solar", 6, "solar and renewable energy business and engineering: plants, inverters, permits, REC/SMP, O&M, ESS", 0.3),
    ("tech_ops", 4, "servers, networking, cloud, DevOps, security questions", 0.3),
    ("ai_ml", 3, "questions about AI, machine learning, LLMs, their use and limits", 0.2),
    ("life_health", 4, "cooking, travel, fitness, health habits, shopping, relationships", 0.05),
    ("chitchat", 2, "casual conversation, opinions, small talk", 0.0),
]
STYLES = {
    "ko": ["짧고 구어체인 반말 섞인 질문", "정중하고 자세한 존댓말 요청", "업무 메신저처럼 간결한 요청", "상황 설명을 길게 한 뒤의 질문",
           "오타나 띄어쓰기 실수가 조금 있는 급한 질문", "코드, 로그, 표, 숫자를 붙여넣은 요청"],
    "en": ["short casual question", "polite detailed request", "terse work-chat request", "long context then a question",
           "hurried question with small typos", "request with pasted code, logs, a table or numbers"],
    "mixed": ["영어 기술 용어를 섞은 한국어 요청", "Korean request that quotes English error messages or docs"],
}
SHAPES = [("single", 0.55), ("multi", 0.25), ("tool", 0.10), ("longdoc", 0.10)]
SYSTEMS = {
    "ko": ["당신은 친절하고 정확한 AI 어시스턴트입니다. 한국어로 답합니다.",
           "당신은 태양광 발전 회사의 사내 업무 도우미입니다. 사내 직원의 질문에 실무적으로 답하세요.",
           "당신은 숙련된 소프트웨어 엔지니어입니다. 코드는 코드 블록으로, 설명은 간결하게 작성하세요.",
           "당신은 에너지 회사의 고객 상담원입니다. 공손하게 답하고, 모르는 것은 확인 후 안내하겠다고 말하세요.",
           "답변은 핵심만 세 줄 이내로 하세요.", "당신은 데이터 분석가입니다. 가능하면 표와 수치로 답하세요.",
           "사용자의 질문에 단계별로 생각한 뒤 결론을 분명히 제시하세요."],
    "en": ["You are a helpful, accurate assistant.", "You are a senior software engineer. Prefer code blocks and short explanations.",
           "You are a customer support agent for a renewable energy company. Be polite and precise.",
           "Answer concisely. Use bullet points when listing.", "You are a data analyst. Show calculations when you use numbers.",
           "You are a tutor. Explain step by step and check understanding at the end."],
}
AGENT_SYSTEMS = {                                              # a tool conversation's system prompt says tools exist
    "ko": ["당신은 사내 업무를 돕는 AI 에이전트입니다. 필요하면 도구를 호출해 정확한 정보를 확인한 뒤 답하세요.",
           "당신은 태양광 발전소 운영을 지원하는 에이전트입니다. 발전량과 설비 상태는 반드시 도구로 조회하고, 결과를 근거로 답하세요.",
           "도구를 사용할 수 있는 개인 비서입니다. 일정, 이메일, 검색 도구를 적절히 사용하세요."],
    "en": ["You are an assistant with access to tools. Call them when you need facts you do not have.",
           "You are an operations agent for a solar portfolio. Query generation and inverter status with the tools before answering.",
           "You are a personal assistant with calendar, email and search tools. Use them to act on the user's requests."],
}
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather and forecast for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web and return the top results",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_plant_generation", "description": "Daily generation (kWh) of a solar plant over a date range",
     "parameters": {"type": "object", "properties": {"plant_id": {"type": "string"}, "start": {"type": "string", "description": "YYYY-MM-DD"},
                                                     "end": {"type": "string", "description": "YYYY-MM-DD"}}, "required": ["plant_id", "start", "end"]}}},
    {"type": "function", "function": {"name": "get_inverter_status", "description": "Status, alarms and output of the inverters at a plant",
     "parameters": {"type": "object", "properties": {"plant_id": {"type": "string"}}, "required": ["plant_id"]}}},
    {"type": "function", "function": {"name": "create_ticket", "description": "Open a maintenance or support ticket",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                                                     "assignee": {"type": "string"}, "details": {"type": "string"}}, "required": ["title", "priority"]}}},
    {"type": "function", "function": {"name": "query_database", "description": "Run a read-only SQL query on the company database",
     "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "list_calendar_events", "description": "List calendar events in a date range",
     "parameters": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}}, "required": ["start", "end"]}}},
    {"type": "function", "function": {"name": "create_calendar_event", "description": "Create a calendar event",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"},
                                                     "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "start", "end"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "Exchange rate between two currencies",
     "parameters": {"type": "object", "properties": {"base": {"type": "string"}, "quote": {"type": "string"}}, "required": ["base", "quote"]}}},
]
DEBRIS = re.compile(r"을\(를\)|이\(가\)|와\(과\)|은\(는\)|\{[a-z_]+\}")
HAN = re.compile(r"[\u4e00-\u9fff]")          # Qwen leaks Chinese characters into Korean at T=1 (e.g. "재편加速")


def han_leak(text: str, lang: str, *, allowed: int) -> bool:
    """Korean text with more Chinese characters than `allowed`: a writer's glitch, not what a user types or pastes."""
    return lang in ("ko", "mixed") and len(HAN.findall(text)) > allowed


def key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return k


def chat(messages, *, model=MODEL, temperature=1.0, top_k=None, top_p=None, max_tokens=1024, thinking=False, tools=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "reasoning": {"enabled": bool(thinking)}}
    if temperature > 0 and top_p is not None:
        body["top_p"] = top_p
    if temperature > 0 and top_k is not None:
        body["top_k"] = top_k
    if tools:
        body["tools"] = tools
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key()}",
                                  "X-Title": "stkernel mtp fine-tune data v2"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                answer = json.loads(r.read().decode())
            if "choices" not in answer:
                raise ValueError(str(answer)[:200])
            return answer
        except Exception as exc:                                   # noqa: BLE001 -- rate limits, provider hiccups
            last = exc
            time.sleep(2 + 5 * attempt)
    raise RuntimeError(repr(last)[:300])


def parallel(jobs, work, out_path, label):
    """Run `work(job) -> list of rows` over jobs on CONCURRENCY threads, appending rows to out_path as they land."""
    lock, state = threading.Lock(), {"next": 0, "ok": 0, "failed": 0, "rows": 0}
    out = open(out_path, "a")

    def worker():
        while True:
            with lock:
                i = state["next"]
                state["next"] += 1
            if i >= len(jobs):
                return
            try:
                rows = work(jobs[i])
                with lock:
                    for row in rows:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    state["ok"] += 1
                    state["rows"] += len(rows)
            except Exception as exc:                               # noqa: BLE001
                with lock:
                    state["failed"] += 1
                    print(json.dumps({"stage": label, "error": repr(exc)[:200]}), flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(CONCURRENCY)]
    began = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(20)
        with lock:
            print(json.dumps({"stage": label, "elapsed_s": round(time.time() - began), "jobs": len(jobs), **state}), flush=True)
    print(json.dumps({"stage": label, "final": state, "elapsed_s": round(time.time() - began)}), flush=True)


def weighted(rng, pairs):
    r, acc = rng.random() * sum(w for _, w in pairs), 0.0
    for item, w in pairs:
        acc += w
        if r <= acc:
            return item
    return pairs[-1][0]


def parse_list(text):
    """The writer's JSON array of strings (or of objects), tolerant of a code fence around it."""
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        got = json.loads(m.group(0))
    except ValueError:
        return []
    return got if isinstance(got, list) else []


# -- seeds ------------------------------------------------------------------------------------------------------------
LANG_NAME = {"ko": "Korean", "en": "English", "mixed": "Korean mixed with English technical terms"}


def seed_job(job):
    rng = random.Random(job["seed"])
    lang, cat, desc, style = job["lang"], job["category"], job["desc"], job["style"]
    n = 6
    ask = (f"Write {n} different messages that real users send to an AI assistant, in {LANG_NAME[lang]}.\n"
           f"Topic area: {desc}.\nWriting style: {style}.\n"
           "Make them realistic and specific: concrete names, numbers, situations, pasted snippets where natural. "
           "Vary the length (from one line to a few paragraphs) and the sub-topic. Do not number them, do not add "
           "explanations, do not start every message the same way. Output only a JSON array of strings.")
    answer = chat([{"role": "user", "content": ask}], model=WRITER, temperature=1.0, top_p=0.95, max_tokens=3000)
    items = [s.strip() for s in parse_list(answer["choices"][0]["message"].get("content") or "") if isinstance(s, str)]
    return [{"kind": "seed", "lang": lang, "category": cat, "style": style, "text": s, "sid": f"{job['seed']}-{i}"}
            for i, s in enumerate(items) if 8 <= len(s) <= 6000 and not han_leak(s, lang, allowed=2)]


def tool_job(job):
    lang = job["lang"]
    tools = job["tools"]
    names = ", ".join(t["function"]["name"] + ": " + t["function"]["description"] for t in tools)
    ask = (f"An AI assistant can call these tools: {names}.\nWrite 5 different realistic user messages in {LANG_NAME[lang]} "
           "that need one or more of these tools to answer (concrete places, dates in 2026, plant ids like 'PV-031', "
           "people and amounts). Output only a JSON array of strings.")
    answer = chat([{"role": "user", "content": ask}], model=WRITER, temperature=1.0, top_p=0.95, max_tokens=2000)
    items = [s.strip() for s in parse_list(answer["choices"][0]["message"].get("content") or "") if isinstance(s, str)]
    return [{"kind": "tool_seed", "lang": lang, "category": "tool", "tools": [t["function"]["name"] for t in tools],
             "text": s, "sid": f"{job['seed']}-{i}"} for i, s in enumerate(items) if len(s) >= 8]


def doc_job(job):
    lang, cat, desc = job["lang"], job["category"], job["desc"]
    kind = random.Random(job["seed"]).choice(["a news article", "a meeting transcript", "an internal report", "a technical document",
                                              "a policy or regulation excerpt", "a customer email thread", "a product manual section"])
    ask = (f"Write {kind} in {LANG_NAME[lang]} about {desc}, 600 to 1500 words, realistic and specific (names, numbers, dates). "
           "Then, after a line containing only ###, write 3 different questions or requests a reader would ask an AI "
           "assistant about this text, as a JSON array of strings.")
    answer = chat([{"role": "user", "content": ask}], model=WRITER, temperature=0.8, top_p=0.95, max_tokens=4000)
    text = answer["choices"][0]["message"].get("content") or ""
    if "###" not in text or han_leak(text, lang, allowed=3):
        return []
    doc, tail = text.split("###", 1)
    questions = [q.strip() for q in parse_list(tail) if isinstance(q, str)]
    doc = doc.strip()
    if len(doc) < 1200 or not questions:
        return []
    return [{"kind": "doc_seed", "lang": lang, "category": cat, "doc": doc, "text": q, "sid": f"{job['seed']}-{i}"}
            for i, q in enumerate(questions[:2])]


RAW_KINDS = ["a news article", "an encyclopedia entry", "a technical documentation page", "a product manual section",
             "a meeting transcript", "a legal or policy text excerpt", "a short story", "a blog post", "a research abstract and introduction",
             "an email thread", "a Python module with docstrings", "a TypeScript file", "a SQL migration and queries", "a YAML configuration with comments",
             "a Markdown README", "a report with a table of figures", "a Go source file", "a shell script with comments", "an FAQ page",
             "a speech transcript"]


def rawdoc_job(job):
    rng = random.Random(job["seed"])
    lang = job["lang"]
    kind = rng.choice(RAW_KINDS)
    code = any(w in kind for w in ("Python", "TypeScript", "SQL", "YAML", "Go ", "shell"))
    topic = rng.choice([d for _, _, d, _ in CATEGORIES])
    ask = (f"Write {kind} " + ("" if code else f"in {LANG_NAME[lang]} ") + f"related to: {topic}. "
           + ("Comments and strings in " + LANG_NAME[lang] + ". " if code else "")
           + "It should read like a real document someone wrote, specific and detailed, 500 to 1200 words (or 150 to 300 "
           "lines of code). Output only the document itself, with no preface.")
    answer = chat([{"role": "user", "content": ask}], model=WRITER, temperature=0.8, top_p=0.95, max_tokens=3500)
    text = (answer["choices"][0]["message"].get("content") or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text)
    if len(text) < 1200 or looping(text) or han_leak(text, lang, allowed=3):
        return []
    return [{"rid": job["seed"], "lang": lang, "category": kind, "text": text}]


def rawdocs(out_dir, calls):
    rng = random.Random(20260920)
    path = os.path.join(out_dir, "rawdocs.jsonl")
    done = {json.loads(l)["rid"] for l in open(path)} if os.path.exists(path) else set()
    jobs = [{"seed": str(2000000 + i), "lang": "ko" if rng.random() < 0.55 else "en"} for i in range(calls)]
    parallel([j for j in jobs if j["seed"] not in done], rawdoc_job, path, "rawdocs")


def seeds(out_dir, calls):
    rng = random.Random(20260919)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "seeds.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["sid"].split("-")[0] for l in open(path)}
    cats = [(c, w) for c, w, _, _ in CATEGORIES]
    desc = {c: d for c, _, d, _ in CATEGORIES}
    jobs = []
    for i in range(calls):
        lang = weighted(rng, LANGS)
        cat = weighted(rng, cats)
        style = rng.choice(STYLES[lang])
        what = weighted(rng, [("seed", 0.8), ("tool", 0.1), ("doc", 0.1)])
        job = {"seed": str(1000000 + i), "lang": "ko" if lang == "mixed" and what != "seed" else lang, "category": cat,
               "desc": desc[cat], "style": style, "what": what, "tools": rng.sample(TOOLS, rng.randint(2, 4))}
        if job["seed"] not in done:
            jobs.append(job)
    work = {"seed": seed_job, "tool": tool_job, "doc": doc_job}
    parallel(jobs, lambda j: work[j["what"]](j), path, "seeds")


# -- conversations ------------------------------------------------------------------------------------------------------
def served_answer(messages, rng, *, thinking, tools=None, max_tokens=1536, setting=None):
    temperature, top_k, top_p = setting or rng.choice(SERVED)
    answer = chat(messages, model=MODEL, temperature=temperature, top_k=top_k, top_p=top_p,
                  max_tokens=max_tokens * (2 if thinking else 1), thinking=thinking, tools=tools)
    choice = answer["choices"][0]
    message = choice["message"]
    out = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("reasoning"):
        out["reasoning_content"] = message["reasoning"]
    if message.get("tool_calls"):
        out["tool_calls"] = [{"id": c.get("id") or f"call_{i}", "type": "function",
                              "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"}}
                             for i, c in enumerate(message["tool_calls"])]
    return out, {"temperature": temperature, "top_k": top_k, "top_p": top_p, "finish": choice.get("finish_reason"),
                 "usage": answer.get("usage", {})}


def follow_up(messages, lang):
    """The writer as the user: the next message a real user would send after this exchange."""
    shown = "\n\n".join(f"[{m['role']}]\n{m['content'][:3000]}" for m in messages if m["role"] in ("user", "assistant"))
    ask = (f"Here is a conversation between a user and an AI assistant:\n\n{shown}\n\nWrite the user's next message, in "
           f"{LANG_NAME[lang]}: a natural follow-up (a clarification, a correction, a deeper question, a related request, or "
           "pushing back). Output only the message text.")
    answer = chat([{"role": "user", "content": ask}], model=WRITER, temperature=1.0, top_p=0.95, max_tokens=600)
    return (answer["choices"][0]["message"].get("content") or "").strip()


def tool_result(call, lang):
    ask = (f"You are the tool `{call['function']['name']}`. It was called with arguments {call['function']['arguments']}. "
           "Return a realistic JSON result for this call (plausible values, 2026 dates; for errors, a JSON error object "
           "occasionally). Output only the JSON.")
    answer = chat([{"role": "user", "content": ask}], model=WRITER, temperature=1.0, top_p=0.95, max_tokens=800)
    text = (answer["choices"][0]["message"].get("content") or "{}").strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text)


def convo_job(seed):
    rng = random.Random(seed["sid"])
    lang = "ko" if seed["lang"] == "mixed" else seed["lang"]
    setting = SERVED[0] if rng.random() < GREEDY_SHARE else SERVED[1]   # one setting for every turn of a conversation
    think_share = {c: t for c, _, _, t in CATEGORIES}.get(seed["category"], 0.2)
    thinking = rng.random() < think_share
    messages, tools, meta = [], None, {}
    if seed["kind"] == "tool_seed":
        tools = [t for t in TOOLS if t["function"]["name"] in seed["tools"]]
        messages.append({"role": "system", "content": rng.choice(AGENT_SYSTEMS[lang])})
        messages.append({"role": "user", "content": seed["text"]})
        for _ in range(3):                                      # call, result, and so on until an answer
            reply, meta = served_answer(messages, rng, thinking=thinking, tools=tools, setting=setting)
            messages.append(reply)
            if not reply.get("tool_calls"):
                break
            for call in reply["tool_calls"]:
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_result(call, lang)})
        shape = "tool"
    elif seed["kind"] == "doc_seed":
        if rng.random() < 0.5:
            messages.append({"role": "system", "content": rng.choice(SYSTEMS[lang])})
        messages.append({"role": "user", "content": seed["doc"] + "\n\n" + seed["text"]})
        reply, meta = served_answer(messages, rng, thinking=thinking, setting=setting)
        messages.append(reply)
        shape = "longdoc"
    else:
        shape = weighted(rng, [("single", 0.8), ("multi", 0.2)])
        if rng.random() < (0.2 if shape == "single" else 0.5):   # most single turns are plain chat: the general half
            messages.append({"role": "system", "content": rng.choice(SYSTEMS[lang])})
        messages.append({"role": "user", "content": seed["text"]})
        turns = rng.choice([2, 2, 3]) if shape == "multi" else 1
        for t in range(turns):
            reply, meta = served_answer(messages, rng, thinking=thinking, setting=setting)
            messages.append(reply)
            if t + 1 < turns:
                nxt = follow_up(messages, lang)
                if not nxt:
                    break
                messages.append({"role": "user", "content": nxt})
    if messages[-1]["role"] != "assistant":
        return []
    return [{"cid": seed["sid"], "shape": shape, "lang": seed["lang"], "category": seed["category"], "thinking": thinking,
             "messages": messages, "tools": tools, "final": meta}]


def convos(out_dir, limit):
    path = os.path.join(out_dir, "convos.jsonl")
    done = {json.loads(l)["cid"] for l in open(path)} if os.path.exists(path) else set()
    seeds_all = [json.loads(l) for l in open(os.path.join(out_dir, "seeds.jsonl"))]
    kept, seen = [], set()
    for s in seeds_all:                                          # exact duplicates and template debris go before any call
        norm = re.sub(r"\W+", "", s["text"].lower())
        if norm in seen or DEBRIS.search(s["text"]):
            continue
        seen.add(norm)
        kept.append(s)
    random.Random(7).shuffle(kept)
    jobs = [s for s in kept[:limit] if s["sid"] not in done]
    parallel(jobs, convo_job, path, "convos")


# -- quality ------------------------------------------------------------------------------------------------------------
def looping(text: str) -> bool:
    """A degenerate loop: some 40-character stretch repeated four or more times."""
    if len(text) < 400:
        return False
    counts = {}
    for i in range(0, len(text) - 40, 20):
        piece = text[i:i + 40]
        counts[piece] = counts.get(piece, 0) + 1
        if counts[piece] >= 4:
            return True
    return False


def shingles(text: str, n: int = 5) -> set:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def quality(out_dir):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "convos.jsonl"))]
    kept, dropped, firsts = [], {}, []

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for r in rows:
        final = r["messages"][-1]
        answer = (final.get("content") or "") + (final.get("reasoning_content") or "")
        users = " ".join(m["content"] for m in r["messages"] if m["role"] == "user")
        if not final.get("content") and not final.get("tool_calls"):
            if not final.get("reasoning_content"):
                drop("empty answer"); continue
            # thinking cut by max_tokens before its answer: the reasoning is exactly what a thinking row drafts, so it
            # stays -- the answer a single space, which continue_final_message resumes after (two tokens of </think>)
            final["content"] = " "
            dropped["kept: reasoning cut by max_tokens"] = dropped.get("kept: reasoning cut by max_tokens", 0) + 1
        if any(looping((m.get("content") or "") + (m.get("reasoning_content") or "")) for m in r["messages"] if m["role"] == "assistant"):
            drop("loop"); continue
        if DEBRIS.search(users):
            drop("template debris"); continue
        if han_leak(users, r["lang"], allowed=2):
            drop("Chinese characters in a Korean user turn"); continue
        answers = "".join(m.get("content") or "" for m in r["messages"] if m["role"] == "assistant")
        if r["lang"] in ("ko", "mixed") and len(HAN.findall(answers)) > max(8, len(answers) // 400):
            drop("Chinese characters throughout a Korean answer"); continue
        if r["lang"] == "ko" and len(re.findall(r"[가-힣]", users)) < 0.2 * len(re.sub(r"\W", "", users) or "x"):
            drop("not Korean"); continue
        first = shingles(r["messages"][1 if r["messages"][0]["role"] == "system" else 0]["content"])
        if any(len(first & f) / max(1, len(first | f)) > 0.6 for f in firsts[-400:]):
            drop("near duplicate"); continue
        firsts.append(first)
        kept.append(r)
    with open(os.path.join(out_dir, "conversations.jsonl"), "w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    docs, seen_docs = [], set()
    for line in open(os.path.join(out_dir, "seeds.jsonl")):
        seed = json.loads(line)
        if seed["kind"] == "doc_seed" and seed["doc"] not in seen_docs and not looping(seed["doc"]):
            seen_docs.add(seed["doc"])
            docs.append({"rid": seed["sid"], "lang": seed["lang"], "category": seed["category"], "text": seed["doc"]})
    extra = os.path.join(out_dir, "rawdocs.jsonl")
    if os.path.exists(extra):                                  # the raw-only documents (rawdocs stage)
        for line in open(extra):
            doc = json.loads(line)
            if doc["text"] not in seen_docs:
                seen_docs.add(doc["text"])
                docs.append(doc)
    with open(os.path.join(out_dir, "raw.jsonl"), "w") as fh:     # the general third: text with no chat around it
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps({"convos": len(rows), "kept": len(kept), "dropped": dropped, "raw_docs": len(docs)}, ensure_ascii=False))


def stats(out_dir):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "conversations.jsonl"))]
    raw = [json.loads(l) for l in open(os.path.join(out_dir, "raw.jsonl"))]
    chars = sum(len(json.dumps(r["messages"], ensure_ascii=False)) for r in rows)
    raw_chars = sum(len(d["text"]) for d in raw)
    plain = sum(len(json.dumps(r["messages"], ensure_ascii=False)) for r in rows
                if r["shape"] == "single" and r["messages"][0]["role"] != "system")

    def share(key):
        out = {}
        for r in rows:
            out[str(r[key])] = out.get(str(r[key]), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    finals = {}
    for r in rows:
        t = (r.get("final") or {}).get("temperature")
        finals[str(t)] = finals.get(str(t), 0) + 1
    total = chars + raw_chars
    print(json.dumps({"conversations": len(rows), "raw_docs": len(raw), "chars": chars, "raw_chars": raw_chars,
                      "approx_tokens": round(total / 2.6),
                      "mix_by_chars": {"plain_chat": round(plain / total, 3), "served_shapes": round((chars - plain) / total, 3),
                                       "raw_text": round(raw_chars / total, 3)},
                      "shape": share("shape"), "lang": share("lang"), "thinking": share("thinking"), "final_temperature": finals,
                      "category": share("category"), "turns": {n: sum(1 for r in rows if sum(m["role"] == "user" for m in r["messages"]) == n)
                                                               for n in (1, 2, 3)}}, ensure_ascii=False))


if __name__ == "__main__":
    stage, out_dir = sys.argv[1], sys.argv[2]
    if stage == "seeds":
        seeds(out_dir, int(sys.argv[3]) if len(sys.argv) > 3 else 500)
    elif stage == "convos":
        convos(out_dir, int(sys.argv[3]) if len(sys.argv) > 3 else 2600)
    elif stage == "rawdocs":
        rawdocs(out_dir, int(sys.argv[3]) if len(sys.argv) > 3 else 120)
    elif stage == "filter":
        quality(out_dir)
    elif stage == "stats":
        stats(out_dir)
    elif stage == "probe":                                       # one job of a kind, printed: probe OUT_DIR seed|tool|doc
        kind = sys.argv[3]
        job = {"seed": "9999999", "lang": "ko", "category": "energy_solar",
               "desc": dict((c, d) for c, _, d, _ in CATEGORIES)["energy_solar"], "style": STYLES["ko"][1], "what": kind,
               "tools": TOOLS[2:5]}
        rows = {"seed": seed_job, "tool": tool_job, "doc": doc_job}[kind](job)
        print(json.dumps([{k: (v[:300] if isinstance(v, str) else v) for k, v in r.items()} for r in rows],
                         ensure_ascii=False, indent=1))
    else:
        raise SystemExit(__doc__)
