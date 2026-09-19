#!/usr/bin/env python3
"""Window 5's arms, paired by prompt: tuned4-off - tuned3-off (greedy) and best-t1-block - best-t1-exact (T=1), over the
whole eval set and split by suite (v1 templates, v2 held-out, crafted by hand, Deneb conversations, Deneb records) and,
from the prompts' own fields, by language, shape, thinking and category. Tokens a decode step leave out the prefill's
first token (review, PR #1275). Prints numbers only -- no prompt or answer text (Deneb's text stays on the fleet).

    arms5.py OUT_DIR [PROMPTS=OUT_DIR/prompts.jsonl]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
prompts = {}
for line in open(sys.argv[2] if len(sys.argv) > 2 else d / "prompts.jsonl"):
    p = json.loads(line)
    if p.get("split") == "eval":
        if not p.get("lang"):                                     # the v1 templates carry no language: read it off the text
            p["lang"] = "ko" if any("가" <= ch <= "힣" for ch in p.get("content") or "") else "en"
        if not p.get("category") and p.get("kind") and not p["id"].startswith(("v2-", "d-", "r-")):
            p["category"] = "v1_" + p["kind"]
        prompts[p["id"]] = p
PAIRS = (("tuned3-off", "tuned4-off"), ("best-t1-exact", "best-t1-block"))


def suite(i):
    return ("v2" if i.startswith("v2-") else "crafted" if i.startswith(("s-", "s2-")) else "deneb" if i.startswith("d-")
            else "records" if i.startswith("r-") else "v1")


def load(label):
    path = d / f"{label}-requests.jsonl"
    if not path.exists():
        return None
    rows, errors = {}, 0
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "error" in r:
            errors += 1
            continue
        if "id" not in r or not r.get("decode_steps"):
            continue
        n, s = r["completion_tokens"], r["decode_steps"]
        r["seconds"] = r["ms_a_step"] * s / 1e3
        r["tps"], r["toks"] = (n - 1) / s, (n - 1) / r["seconds"]
        rows[r["id"]] = r
    return rows, errors


def summary(rows):
    steps = sum(x["decode_steps"] for x in rows)
    dec = sum(x["completion_tokens"] - 1 for x in rows)
    secs = sum(x["seconds"] for x in rows)
    return steps, dec, secs


def paired(a, b, ids, key):
    diffs = [b[k][key] - a[k][key] for k in ids]
    m = sum(diffs) / len(diffs)
    se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs)) if len(diffs) > 1 else float("nan")
    base = sum(a[k][key] for k in ids) / len(ids)
    return m, se, base


arms = {}
for pair in PAIRS:
    for label in pair:
        got = load(label)
        if got is None:
            print(f"{label}: missing")
        else:
            arms[label] = got[0]
            steps, dec, secs = summary(got[0].values())
            print(f"{label:14s} answered {len(got[0]):3d} errors {got[1]:2d} steps {steps:6.0f} decoded {dec:6d} "
                  f"ms/step {secs / steps * 1e3:6.2f} tokens/step {dec / steps:.3f} tok/s {dec / secs:6.1f}")

for base, other in PAIRS:
    if base not in arms or other not in arms:
        continue
    a, b = arms[base], arms[other]
    both = [k for k in a if k in b and k in prompts]
    groups = defaultdict(list)
    for k in both:
        p = prompts[k]
        groups["all"].append(k)
        groups["suite:" + suite(k)].append(k)
        groups["lang:" + str(p.get("lang", "?"))].append(k)
        groups["thinking:" + ("on" if p.get("thinking") else "off")].append(k)
        groups["system:" + ("yes" if p.get("system") or (p.get("messages") or [{}])[0].get("role") == "system" else "no")].append(k)
        turns = sum(1 for m in p.get("messages") or [] if m.get("role") == "user") or 1
        groups["turns:" + ("1" if turns == 1 else "2+")].append(k)
        groups["max_tokens:" + str(p.get("max_tokens"))].append(k)
        if p.get("category"):
            groups["category:" + p["category"]].append(k)
    print(f"\n== {other} - {base}: tokens/step (paired mean ± se, relative) | tok/s relative | same text")
    order = sorted(groups, key=lambda g: (g != "all", g.split(":")[0], -len(groups[g]), g))
    for g in order:
        ids = groups[g]
        if len(ids) < 3 and not g.startswith("suite:"):
            continue
        m, se, base_mean = paired(a, b, ids, "tps")
        mt, set_, bt = paired(a, b, ids, "toks")
        same = sum(1 for k in ids if (a[k]["completion_tokens"], a[k].get("text")) == (b[k]["completion_tokens"], b[k].get("text")))
        print(f"  {g:34s} n {len(ids):3d}  {m:+.3f} ± {se:.3f} ({100 * m / base_mean:+5.1f}% ± {100 * se / base_mean:3.1f}%) "
              f"base {base_mean:.3f} | tok/s {100 * mt / bt:+5.1f}% ± {100 * set_ / bt:3.1f}% | same {same}/{len(ids)}")
