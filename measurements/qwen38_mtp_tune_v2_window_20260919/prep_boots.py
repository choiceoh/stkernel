#!/usr/bin/env python3
"""The fourth window's three data boots, each a file the prefill driver takes whole (mtp_requests2.py prefill2 ... all):

    boot_a.jsonl / raw_a.jsonl   the synthetic conversations answered greedy, and the synthetic raw documents
    boot_b.jsonl                 the synthetic conversations answered at T=1
    boot_c.jsonl / raw_c.jsonl   Deneb's transcript windows, and Deneb's wiki, code and documents

Within a budget of characters a boot, every served shape (multi-turn, tools, long documents) and every thinking
conversation stays; the plain single turns are sampled down to fit -- the window's prefill is about half an hour.

    prep_boots.py DIR [A_CHARS=2300000] [B_CHARS=1150000]
"""
import json
import random
import sys

d = sys.argv[1]
budget = {"a": int(sys.argv[2]) if len(sys.argv) > 2 else 2300000, "b": int(sys.argv[3]) if len(sys.argv) > 3 else 1150000}
rng = random.Random(20260919)


def size(r):
    return sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") for m in r["messages"])


convos = [json.loads(line) for line in open(f"{d}/conversations.named.jsonl")]
out = {}
for boot, greedy in (("a", True), ("b", False)):
    rows = [r for r in convos if ((r.get("final") or {}).get("temperature") == 0.0) == greedy]
    rich = [r for r in rows if r["shape"] != "single" or r["thinking"]]
    plain = [r for r in rows if r["shape"] == "single" and not r["thinking"]]
    rng.shuffle(rich)
    rng.shuffle(plain)
    # the operator's mix ("적절히 섞어야해"): plain single-turn chat 45% of a boot's characters, the served shapes and
    # thinking 55% -- each group sampled to its share, a group that runs short leaving its room to the other
    keep, used = [], 0
    for group, share in ((plain, 0.45), (rich, 0.55)):
        room = budget[boot] * share
        for r in group:
            if size(r) <= room:
                keep.append(r)
                room -= size(r)
                used += size(r)
    for r in plain + rich:                                        # what a short group left: the rest, in turn
        if r not in keep and used + size(r) <= budget[boot]:
            keep.append(r)
            used += size(r)
    rng.shuffle(keep)
    out[boot] = (keep, used, len(rows))
    with open(f"{d}/boot_{boot}.jsonl", "w") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
deneb = [json.loads(line) for line in open(f"{d}/deneb_convos.jsonl")]
with open(f"{d}/boot_c.jsonl", "w") as fh:
    for r in deneb:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
for src, dst in (("raw.named.jsonl", "raw_a.jsonl"), ("deneb_raw.jsonl", "raw_c.jsonl")):
    with open(f"{d}/{src}") as fin, open(f"{d}/{dst}", "w") as fout:
        fout.write(fin.read())
raw_a = sum(len(json.loads(l)["text"]) for l in open(f"{d}/raw_a.jsonl"))
raw_c = sum(len(json.loads(l)["text"]) for l in open(f"{d}/raw_c.jsonl"))
system = len(deneb[0]["messages"][0]["content"]) if deneb and deneb[0]["messages"][0]["role"] == "system" else 0
deneb_chars = sum(size(r) for r in deneb) - system * max(0, len(deneb) - 1)          # the system prompt is cached after the first
print(json.dumps({"a": {"convos": len(out["a"][0]), "of": out["a"][2], "chars": out["a"][1], "raw_chars": raw_a},
                  "b": {"convos": len(out["b"][0]), "of": out["b"][2], "chars": out["b"][1]},
                  "c": {"convos": len(deneb), "chars_after_prefix_cache": deneb_chars, "raw_chars": raw_c},
                  "approx_tokens": round((out["a"][1] + raw_a + out["b"][1]) / 2.6 + (deneb_chars + raw_c) / 3.0)}))
