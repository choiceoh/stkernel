"""srv2: raw routed experts of every C=1 verify step (debug branch debug-ace-route-dump f19573f0).

The debug engine keeps every decode step synchronous; rank 0 appends each step's int16 ids and float32 weights
[42 MoE layers, 8 rows, top-8] to ~/glm53-logs/route-dump/<latency token>.bin on its node and writes one
`route_dump` latency row with the offset. Runs the twelve onepass JSON questions of #966 at C=1, temperature 0,
inside one door latency recording, and saves the rows.

    python3 ace_route_probe.py --prompts ~/expert-capture/style-prompts.jsonl --out ~/expert-capture/ace-routes/rows.jsonl
"""
import argparse
import http.client
import json
import time
import uuid

PORT = 8001


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_id = {p["id"]: p for p in map(json.loads, open(a.prompts))}
    ids = [f"json-s{s}-{c}" for s in (100, 101, 102, 103) for c in ("ledger", "portfolio", "logic")]
    ask(dict(content="hi", max_tokens=16, reasoning_budget=8), None)
    token = "aceroutes-" + uuid.uuid4().hex[:12]
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
    rows = [r for r in rank0.get("rows", []) if r.get("kind") == "route_dump"]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    paths = sorted({r["path"] for r in rows})
    print(f"{len(rows)} route_dump rows on rank 0; errors {rank0.get('errors')}; files {paths}", flush=True)
    if rows:
        last = rows[-1]
        print(f"bytes expected {last['offset'] + last['nbytes']}", flush=True)
    print("ACE-ROUTES done", flush=True)


if __name__ == "__main__":
    main()
