#!/usr/bin/env python3
"""Forty live-eval prompts from the v2 set that no data boot prefilled: conversations.named.jsonl minus boot_a / boot_b
(prep_boots.py sampled 823 of 1,110 into the boots), no tools (the door answers tools with 400), the conversation up to
its last user turn, greedy, max_tokens 256 like the v1 eval prompts. The mix follows the served shapes: system prompts,
multi-turn, long documents, thinking, Korean and English.

    v2_eval_prompts.py V2_DIR > v2_eval.jsonl
"""
import json
import random
import sys

d = sys.argv[1]
rng = random.Random(20260919)
used = set()
for b in ("a", "b"):
    used |= {json.loads(line)["cid"] for line in open(f"{d}/boot_{b}.jsonl")}
pool = [r for r in map(json.loads, open(f"{d}/conversations.named.jsonl"))
        if r["cid"] not in used and not r.get("tools") and r["shape"] in ("single", "multi", "longdoc")]
rng.shuffle(pool)
want = {"multi": 14, "longdoc": 4, "single": 22}
picked, count, thinking = [], {k: 0 for k in want}, 0
# the unused pool leans to thinking (the boots kept the rich shapes first); the eval keeps the set's share, about 30%
for r in sorted(pool, key=lambda r: (r["shape"] != "longdoc", r["shape"] != "multi")):   # the scarce shapes first
    if count[r["shape"]] < want[r["shape"]] and (not r["thinking"] or thinking < 12):
        picked.append(r)
        count[r["shape"]] += 1
        thinking += bool(r["thinking"])
rng.shuffle(picked)
for r in picked:
    msgs = r["messages"]
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    context = [{k: v for k, v in m.items() if k in ("role", "content", "reasoning_content")} for m in msgs[:last_user + 1]]
    print(json.dumps({"id": f"v2-{r['cid']}", "kind": f"v2-{r['shape']}", "messages": context,
                      "content": context[-1]["content"], "max_tokens": 256, "thinking": bool(r["thinking"]),
                      "temperature": 0.0, "split": "eval", "lang": r["lang"],
                      "system": context[0]["role"] == "system"}, ensure_ascii=False))
print(json.dumps({"pool": len(pool), "picked": len(picked), "shapes": count}), file=sys.stderr)
