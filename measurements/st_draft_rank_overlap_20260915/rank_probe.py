"""srv2: at each greedy first rejection, where the target's token sits in the drafter's candidates (debug branch
debug-draft-candidate-rank adds target_rank/draft_rank to the door's draft_rejection rows).

Runs the twelve onepass JSON questions at C=1 inside one door latency recording (POST /v1/engine/latency begin/end,
requests carry X-ST-Latency-Token), keeps rank 0's draft_rejection rows, and prints: the reasons, the target's
candidate-rank histogram on selector misses, the walk's own rank, and what a tree that adds one or more siblings at the
rejected position would have caught.

    python3 rank_probe.py --prompts ~/expert-capture/style-prompts.jsonl --out ~/expert-capture/rank-rows.jsonl
"""
import argparse
import http.client
import json
import time
import uuid
from collections import Counter

PORT = 8001


def post(path, body, headers=None, timeout=3600):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        data = json.loads(raw)
    except ValueError:
        data = {"raw": raw[:500].decode(errors="replace")}
    return resp.status, data


def ask(item, token):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": item["content"]}],
            "max_tokens": item["max_tokens"], "temperature": 0.0, "chat_template_kwargs": {"thinking": True},
            "reasoning_budget": item["reasoning_budget"], "retain": False, "cache_salt": uuid.uuid4().hex}
    status, data = post("/v1/chat/completions", body, {"X-ST-Latency-Token": token} if token else None)
    if status != 200:
        raise RuntimeError(f"chat HTTP {status}: {str(data)[:300]}")
    return (data.get("usage") or {}).get("completion_tokens")


def summarize(rows):
    reasons = Counter(r["reason"] for r in rows)
    sel = [r for r in rows if r["reason"] == "selector_miss"]
    cand = [r for r in rows if r["reason"] == "candidate_miss"]
    rejections = len(sel) + len(cand)
    steps = sum(reasons.values())
    print(f"steps {steps}: " + ", ".join(f"{k} {v} ({100 * v / steps:.1f}%)" for k, v in reasons.most_common()))
    print(f"first rejections {rejections}: target among the candidates {len(sel)} ({100 * len(sel) / max(1, rejections):.1f}%)")
    hist = Counter(r["target_rank"] for r in sel)
    cum = 0
    line = []
    for rank in range(16):
        cum += hist.get(rank, 0)
        if rank in (0, 1, 2, 3, 7, 15):
            line.append(f"<= rank {rank}: {100 * cum / max(1, rejections):.1f}%")
    print("target's candidate rank, share of ALL first rejections: " + ", ".join(line))
    print("target rank histogram (selector misses): " + ", ".join(f"{k}:{hist[k]}" for k in sorted(hist)))
    dr = Counter(r["draft_rank"] for r in sel + cand)
    print("walk's own pick rank at the rejected position: " + ", ".join(f"{k}:{dr[k]}" for k in sorted(dr)))
    joint = Counter((r["target_rank"] == 0, r["draft_rank"] == 0) for r in sel)
    print(f"selector misses: target top-1 & walk not top-1 {joint[(True, False)]}, target not top-1 & walk top-1 {joint[(False, True)]}, "
          f"both not top-1 {joint[(False, False)]}")
    for width in (1, 2, 3, 7):
        # a tree adding `width` siblings at the rejected position: the best-ranked candidates other than the walk's pick
        caught = 0
        for r in sel:
            others = [x for x in range(16) if x != r["draft_rank"]][:width]
            caught += r["target_rank"] in others
        print(f"  {width} sibling(s) by drafter rank would catch {caught} of {rejections} first rejections "
              f"({100 * caught / max(1, rejections):.1f}%)")
    by_pos = Counter((r["accepted_prefix"], r["reason"]) for r in sel + cand)
    print("by position (accepted prefix): " + "; ".join(
        f"{p}: sel {by_pos[(p, 'selector_miss')]} cand {by_pos[(p, 'candidate_miss')]}" for p in range(8)
        if by_pos[(p, 'selector_miss')] or by_pos[(p, 'candidate_miss')]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_id = {p["id"]: p for p in map(json.loads, open(a.prompts))}
    ids = [f"json-s{s}-{c}" for s in (100, 101, 102, 103) for c in ("ledger", "portfolio", "logic")]
    ask(dict(content="hi", max_tokens=16, reasoning_budget=8), None)
    token = "rank-" + uuid.uuid4().hex[:12]
    status, begin = post("/v1/engine/latency", {"op": "begin", "token": token, "diagnostic": False, "concurrency": 1})
    print("begin", status, json.dumps(begin)[:300], flush=True)
    if status != 200:
        raise SystemExit(1)
    try:
        for i in ids:
            t0 = time.monotonic()
            n = ask(by_id[i]["item"], token)
            print(f"{i}: {n} tokens in {time.monotonic() - t0:.1f} s", flush=True)
    finally:
        status, end = post("/v1/engine/latency", {"op": "end", "token": token})
        print("end", status, [{k: v for k, v in r.items() if k not in ("rows", "traces")} for r in end.get("ranks", [])][:1], flush=True)
    rank0 = next((r for r in end.get("ranks", []) if r.get("rank") == 0), {})
    rows = [r for r in rank0.get("rows", []) if r.get("kind") == "draft_rejection"]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} draft_rejection rows on rank 0 (all ranks' row counts: "
          f"{[len(r.get('rows', [])) for r in end.get('ranks', [])]})", flush=True)
    if rows and "target_rank" not in rows[0]:
        print("rows carry no target_rank: not the debug engine", flush=True)
        raise SystemExit(2)
    summarize(rows)
    print("RANK done", flush=True)


if __name__ == "__main__":
    main()
