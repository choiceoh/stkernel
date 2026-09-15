"""srv2: how many new MoE experts a tree's sibling row would read (debug branch debug-sibling-expert-overlap).

The debug engine keeps every decode step synchronous and writes one `route_overlap` row a step: the union of the
8 verify rows' routed experts, each row's leave-one-out new experts, and -- when the step continues the previous one
-- how many of the anchor's (layer, expert) routes the previous step's chain did not read. After a rejection the anchor
is the target's own token at the rejected position over the same accepted prefix: the routes a tree's right sibling
takes. Runs the twelve onepass JSON questions at C=1 inside one door latency recording.

    python3 overlap_probe.py --prompts ~/expert-capture/style-prompts.jsonl --out ~/expert-capture/overlap-rows.jsonl
"""
import argparse
import http.client
import json
import statistics
import time
import uuid
from collections import defaultdict

PORT = 8001
UNIFORM_8_OF_288 = 288 * (1 - (280 / 288) ** 8)


def post(path, body, headers=None, timeout=3600):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw)
    except ValueError:
        return resp.status, {"raw": raw[:500].decode(errors="replace")}


def ask(item, token):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": item["content"]}],
            "max_tokens": item["max_tokens"], "temperature": 0.0, "chat_template_kwargs": {"thinking": True},
            "reasoning_budget": item["reasoning_budget"], "retain": False, "cache_salt": uuid.uuid4().hex}
    status, data = post("/v1/chat/completions", body, {"X-ST-Latency-Token": token} if token else None)
    if status != 200:
        raise RuntimeError(f"chat HTTP {status}: {str(data)[:300]}")
    return (data.get("usage") or {}).get("completion_tokens")


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def summarize(rows):
    rows = [r for r in rows if r.get("rows") == 8]
    if not rows:
        print("no 8-row route_overlap rows")
        return
    layers = rows[0]["layers"]
    slots = 8 * layers
    stale = sum(r["stale"] for r in rows)
    print(f"steps {len(rows)} (8 rows, {layers} MoE layers, top-8); stale route sets {stale}")
    per_layer = [r["union_total"] / layers for r in rows]
    print(f"distinct experts per layer read by the 8 verify rows: mean {mean(per_layer):.1f} "
          f"(uniform random top-8 of 288 would be {UNIFORM_8_OF_288:.1f}; one row alone reads 8)")
    loo = [r["loo_draft_mean"] / 8 / layers for r in rows]
    loo_u = [r["loo_draft_mean"] / r["union_total"] for r in rows]
    print(f"a chain draft row's own experts no other row reads: {100 * mean(loo):.1f}% of its routes, "
          f"= +{100 * mean(loo_u):.1f}% of the step's expert reads")
    cont = [r for r in rows if "anchor_new" in r]
    groups = {"after a rejection (anchor = the right sibling)": [r for r in cont if r["prev_accepted"] < r["prev_drafted"]],
              "after all 7 accepted (anchor = the continuation)": [r for r in cont if r["prev_accepted"] == r["prev_drafted"]]}
    for name, sel in groups.items():
        if not sel:
            print(f"{name}: none")
            continue
        new = [r["anchor_new"] / slots for r in sel]
        growth = [r["anchor_new"] / r["prev_union_total"] for r in sel]
        print(f"{name}: {len(sel)} steps; new routes {100 * mean(new):.1f}% of the row's {slots} "
              f"(median {100 * statistics.median(new):.1f}%), adding the row = +{100 * mean(growth):.1f}% expert reads")
    by_pos = defaultdict(list)
    for r in groups["after a rejection (anchor = the right sibling)"]:
        by_pos[r["prev_accepted"]].append(r["anchor_new"] / r["prev_union_total"])
    print("sibling at rejected position j: +expert reads " + ", ".join(
        f"j={p + 1}: {100 * mean(v):.1f}% (n={len(v)})" for p, v in sorted(by_pos.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_id = {p["id"]: p for p in map(json.loads, open(a.prompts))}
    ids = [f"json-s{s}-{c}" for s in (100, 101, 102, 103) for c in ("ledger", "portfolio", "logic")]
    ask(dict(content="hi", max_tokens=16, reasoning_budget=8), None)
    token = "overlap-" + uuid.uuid4().hex[:12]
    status, begin = post("/v1/engine/latency", {"op": "begin", "token": token, "diagnostic": False, "concurrency": 1})
    print("begin", status, json.dumps(begin)[:200], flush=True)
    if status != 200:
        raise SystemExit(1)
    try:
        for i in ids:
            t0 = time.monotonic()
            n = ask(by_id[i]["item"], token)
            print(f"{i}: {n} tokens in {time.monotonic() - t0:.1f} s", flush=True)
    finally:
        status, end = post("/v1/engine/latency", {"op": "end", "token": token})
        print("end", status, flush=True)
    rank0 = next((r for r in end.get("ranks", []) if r.get("rank") == 0), {})
    rows = [r for r in rank0.get("rows", []) if r.get("kind") == "route_overlap"]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} route_overlap rows on rank 0; errors {rank0.get('errors')}", flush=True)
    summarize(rows)
    print("OVERLAP done", flush=True)


if __name__ == "__main__":
    main()
