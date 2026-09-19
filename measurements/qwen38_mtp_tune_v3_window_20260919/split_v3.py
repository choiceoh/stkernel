#!/usr/bin/env python3
"""The v3 self-distribution data split for the next window: a held-out set prefilled in a boot of its own
(mtp_tune data --eval-boots), so no session sits on both sides -- window 5's held-out cut one Deneb session's
overlapping windows onto both. Deneb conversations split by session (found by their context's first user turn),
Deneb record tasks, mail analyses and synthetic conversations by prompt; about one in ten held out.

    split_v3.py V3_DIR V3SYN_DIR     -> V3_DIR/train.jsonl, V3_DIR/heldout.jsonl  (cid, src, messages, tools, thinking)

Counts only on stdout.
"""
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfgen3 import extract_defs                                   # noqa: E402

v3, syn = sys.argv[1], sys.argv[2]
x = extract_defs()
first_user_to_session = {}
for f in sorted(os.listdir(os.path.join(x["HOME"], "transcripts"))):
    if not f.endswith(".jsonl"):
        continue
    records = []
    for line in open(os.path.join(x["HOME"], "transcripts", f), errors="replace"):
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    messages, _ = x["blocks_to_messages"](records)
    for m in messages:
        if m["role"] == "user" and (m.get("content") or "").strip():
            first_user_to_session.setdefault(m["content"].strip(), f)


def held(key: str) -> bool:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 10 == 0


rows, unmatched = [], 0
for src, path in (("deneb", os.path.join(v3, "deneb_answers.jsonl")), ("mail", os.path.join(v3, "mail_answers.jsonl")),
                  ("synthetic", os.path.join(syn, "conversations.named.jsonl"))):
    for line in open(path):
        r = json.loads(line)
        key = r["cid"]
        if src == "deneb" and r["cid"].startswith("dp-"):
            first = next((m["content"].strip() for m in r["messages"] if m["role"] == "user" and (m.get("content") or "").strip()), "")
            session = first_user_to_session.get(first)
            if session is None:
                unmatched += 1
            key = "session:" + (session or r["cid"])
        rows.append((held(key), {"cid": r["cid"], "src": src, "messages": r["messages"], "tools": r.get("tools"),
                                  "thinking": bool(r.get("thinking")), "final": r.get("final")}))
with open(os.path.join(v3, "train.jsonl"), "w") as tr, open(os.path.join(v3, "heldout.jsonl"), "w") as ho:
    for is_held, r in rows:
        (ho if is_held else tr).write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"rows": len(rows), "unmatched_sessions": unmatched,
                  "train": dict(Counter(r["src"] for h, r in rows if not h)),
                  "heldout": dict(Counter(r["src"] for h, r in rows if h))}))
