#!/usr/bin/env python3
"""The live eval from window 6 on: 80 prompts (operator: "평가 180개가 너무 많아서 시간이 오래걸려 80개까지만 하자"), with
thinking on for about half, as Deneb's traffic has it (63% of its answered turns reasoned; operator: "mtp 수용률 상승은
추론이 있어야되는거 아니야?"). Hand-written 25 (every category, the conversations kept, thinking on where a request calls
for work), Deneb's conversations 30, Deneb's records 14, Deneb's mail analyses 11 (all thinking).

    make_prompts80.py W5_PROMPTS MAIL_EVAL OUT      (counts only on stdout)
"""
import json
import sys
from collections import Counter

w5, mail, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(w5)]
ev = {r["id"]: r for r in rows if r.get("split") == "eval"}
crafted = [r for i, r in ev.items() if i.startswith("s-")]
multi = [r for r in crafted if r["kind"] == "s-multi"]
single = [r for r in crafted if r["kind"] != "s-multi"]
picked, cats = list(multi), {r["category"] for r in multi}
for r in single:                                          # one of each category first
    if r["category"] not in cats:
        picked.append(r)
        cats.add(r["category"])
for r in single:                                          # then the rest in order, to 25
    if len(picked) >= 25:
        break
    if r not in picked:
        picked.append(r)
WORK = {"math", "reasoning", "coding_debug", "coding_write", "data_sql", "energy_solar", "business_finance",
        "legal_tax_admin", "coding_review"}
for r in picked:
    if r["category"] in WORK:
        r["thinking"] = True
deneb = [r for i, r in ev.items() if i.startswith("d-")]
records = []
for kind, keep in (("r-wiki", 6), ("r-files", 5), ("r-code", 3)):
    records += [r for i, r in ev.items() if i.startswith("r-") and r["kind"] == kind][:keep]
mails = [json.loads(l) for l in open(mail)]
final = picked + deneb + records + mails
assert len({r["id"] for r in final}) == len(final)
with open(out, "w") as fh:
    for r in final:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
suite = lambda i: {"s": "crafted", "d": "deneb", "r": "records", "m": "mail"}[i[0]]
print(json.dumps({"prompts": len(final), "suites": dict(Counter(suite(r["id"]) for r in final)),
                  "thinking": sum(1 for r in final if r.get("thinking")),
                  "crafted_categories": len({r["category"] for r in picked}),
                  "lang": dict(Counter(r.get("lang") for r in final))}))
