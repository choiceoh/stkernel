#!/usr/bin/env python3
"""Email-analysis prompts from Deneb's mail store for the self-distribution data (operator 2026-09-19: "사용자 턴 질문
뿐만 아니라 메일분석은?"). Deneb analyses mail (mail_archive, gmail tools; an email-analysis skill), so its mail is a
served workload. Forty messages are held out for the live evaluation and never enter the data.

    mailgen3.py OUT_DIR [N=600]
        mail_prompts.jsonl  ids mp-*: one message analysed; a day's messages triaged; a reply drafted; fields
                            extracted as JSON; a thread summarised -- half under Deneb's system prompt
        mail_eval.jsonl     ids m-*: ten prompts over the held-out messages, same shapes (live eval)

Runs on srv4 (~/.deneb/mailstore/messages). Prints counts only; the mail stays on srv4/srv2 and out of the repo.
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HOME = os.path.expanduser("~/.deneb")
out_dir = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
rng = random.Random(20260922)
os.makedirs(out_dir, exist_ok=True)
system = "\n\n".join(open(os.path.join(HOME, f), errors="replace").read().strip()
                     for f in ("SOUL.md", "IDENTITY.md", "USER.md", "AGENTS.md", "TOOLS.md")
                     if os.path.exists(os.path.join(HOME, f)))

# the email-analysis skill Deneb ran (retired 2026-08-26): its procedure goes into the system prompt of half the single
# and batch analyses, so the served model does the task the way Deneb asked it (operator: "메일 분석 업무 자체를 api한테
# 시켜서 그 추론이랑 답변을 얻으면")
skill_path = os.path.join(HOME, ".retired-skills", "email-analysis.retired-20260826", "SKILL.md")
skill = open(skill_path, errors="replace").read().strip() if os.path.exists(skill_path) else ""

mails = []
for f in sorted(glob.glob(os.path.join(HOME, "mailstore", "messages", "*.jsonl"))):
    for line in open(f, errors="replace"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        body = (m.get("body") or "").strip()
        if len(body) < 80:
            continue
        mails.append(m)
rng.shuffle(mails)
held, pool = mails[:40], mails[40:]


def korean(text):
    return len(re.findall(r"[가-힣]", text)) > 0.2 * max(1, len(re.sub(r"\W", "", text)))


def show(m):
    return (f"보낸 사람: {m.get('from', '')}\n받는 사람: {m.get('to', '')}\n날짜: {m.get('date', '')}\n"
            f"제목: {m.get('subject', '')}\n\n{m.get('body', '').strip()}")


def norm_subject(s):
    return re.sub(r"^\s*((re|fw|fwd|답장|전달)\s*:\s*)+", "", s or "", flags=re.I).strip().lower()


TASKS = {
    "mail-single": ["다음 메일을 분석해줘. 한 줄 요약, 보낸 사람이 원하는 것, 내가 해야 할 일과 기한, 긴급도(상/중/하)와 이유를 "
                    "정리하고, 답장이 필요하면 짧은 초안도 붙여줘.",
                    "이 메일 핵심만 알려줘. 내가 뭘 해야 하는지, 언제까지인지 중심으로.",
                    "이 메일이 중요한 메일인지, 그냥 알림이나 광고인지 판단하고 이유를 말해줘. 중요하면 할 일도 정리해줘."],
    "mail-reply": ["이 메일에 보낼 공손한 답장 초안을 써줘. 내용은 확인했고 이번 주 안에 회신하겠다는 취지로.",
                   "이 메일에 대한 답장을 써줘. 요청은 받아들이되 일정은 다음 주로 조정하고 싶다고.",
                   "Draft a concise, polite reply to this email in the language it was written in."],
    "mail-extract": ["이 메일에서 회사명, 담당자, 날짜, 금액, 요청 사항을 JSON 으로만 뽑아줘. 없는 항목은 null.",
                     "Extract sender, organisation, dates, amounts and requested actions from this email as JSON only."],
    "mail-batch": ["오늘 받은 메일들이야. 우선순위대로 정리하고, 각각 해야 할 일을 한 줄로 적어줘. 무시해도 되는 메일은 따로 묶어줘.",
                   "아래 메일들을 급한 것, 이번 주 안에 볼 것, 참고만 할 것으로 나눠서 정리해줘."],
    "mail-thread": ["아래는 한 스레드의 메일들이야. 지금까지의 진행 상황과 아직 남은 쟁점, 다음에 내가 할 일을 정리해줘."],
}
SHARES = {"mail-single": 0.4, "mail-reply": 0.15, "mail-extract": 0.1, "mail-batch": 0.2, "mail-thread": 0.15}


def build(source, count, prefix):
    rows = []
    by_day, by_thread = defaultdict(list), defaultdict(list)
    for m in source:
        by_day[(m.get("mailbox"), (m.get("date") or "")[:16])].append(m)
        by_thread[norm_subject(m.get("subject"))].append(m)
    days = [v for v in by_day.values() if len(v) >= 3] or [source[i:i + 5] for i in range(0, len(source), 5)]
    threads = [v for k, v in by_thread.items() if k and len(v) >= 2]
    singles = list(source)
    for kind, share in SHARES.items():
        want = max(1, round(count * share))
        for n in range(want):
            if kind in ("mail-single", "mail-reply", "mail-extract"):
                if not singles:
                    break
                ms = [singles.pop()]
            elif kind == "mail-batch":
                if not days:
                    continue
                day = rng.choice(days)
                ms = rng.sample(day, min(len(day), rng.randint(3, 7)))
            else:
                if not threads:
                    continue
                ms = sorted(rng.choice(threads), key=lambda m: m.get("date") or "")[:6]
            text = "\n\n---\n\n".join(show(m) for m in ms)
            if len(text) > 14000:
                text = text[:14000] + "\n...(생략)"
            task = TASKS[kind][n % len(TASKS[kind])]
            if skill and kind in ("mail-single", "mail-batch") and n % 2 == 0:
                sys_text = (system + "\n\n" if system else "") + "# 활성 스킬: 이메일 분석\n\n" + skill
                messages = [{"role": "system", "content": sys_text},
                            {"role": "user", "content": ("이 메일 분석해줘." if kind == "mail-single" else "오늘 온 메일들 분석해줘.")
                             + "\n\n" + text}]
            else:
                messages = ([{"role": "system", "content": system}] if system and rng.random() < 0.5 else []) + \
                           [{"role": "user", "content": f"{task}\n\n{text}"}]
            rows.append({"id": f"{prefix}{len(rows):04d}", "kind": kind, "category": "deneb_mail",
                         "lang": "ko" if korean(text) else "en", "thinking": True,     # the reasoning is wanted too
                         "messages": messages, "tools": None, "mail_ids": [m.get("id") for m in ms]})
    return rows


data = build(pool, N, "mp-")
evals = build(held, 10, "m-")
with open(os.path.join(out_dir, "mail_prompts.jsonl"), "w") as fh:
    for r in data:
        fh.write(json.dumps({k: v for k, v in r.items() if k != "mail_ids"}, ensure_ascii=False) + "\n")
with open(os.path.join(out_dir, "mail_eval.jsonl"), "w") as fh:
    for r in evals:
        fh.write(json.dumps({"id": r["id"], "kind": r["kind"], "suite": "deneb_mail", "messages": r["messages"],
                             "content": r["messages"][-1]["content"], "max_tokens": 512, "thinking": r["thinking"],
                             "temperature": 0.0, "split": "eval", "lang": r["lang"], "category": "deneb_mail",
                             "system": r["messages"][0]["role"] == "system"}, ensure_ascii=False) + "\n")
print(json.dumps({"mails": len(mails), "held_out": len(held), "prompts": len(data), "eval_prompts": len(evals),
                  "kinds": dict(Counter(r["kind"] for r in data)), "lang": dict(Counter(r["lang"] for r in data)),
                  "with_system": sum(1 for r in data if r["messages"][0]["role"] == "system"),
                  "thinking": sum(r["thinking"] for r in data), "with_skill": sum(1 for r in data if "# 활성 스킬" in r["messages"][0]["content"]),
                  "eval_kinds": dict(Counter(r["kind"] for r in evals))}))
