#!/usr/bin/env python3
"""Replay captured ST token IDs on an idle live door, with a fresh cache namespace.

This is a bounded incident reproduction, not a throughput benchmark. The record
contains private prompt tokens: keep request/response artifacts outside git.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(base, path, body=None):
    req = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--temperature", type=float, choices=(0.0, 1.0), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 1200:
        parser.error("this bounded reproduction allows 1..1200 output tokens")
    record = json.loads(args.record.read_text())
    count = record["prompt_len"]
    ids = record["tokens"][:count]
    if len(ids) != count or not ids or any(type(t) is not int or t < 0 for t in ids):
        parser.error("record must contain the complete prompt token IDs")
    base = args.url.rstrip("/")
    before = fetch(base, "/")
    if before.get("engine") != "ST" or any(before.get(k) for k in ("running", "waiting", "queued")):
        raise SystemExit("ST door must be idle before this reproduction")
    if before.get("fleet", {}).get("draining"):
        raise SystemExit("ST door is draining")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    body = dict(ids=ids, temperature=args.temperature, seed=args.seed,
                max_tokens=args.max_tokens, retain=False,
                cache_salt="incident-" + uuid.uuid4().hex)
    (args.out / "request.json").write_text(json.dumps(body))
    (args.out / "before.json").write_text(json.dumps(before, indent=2))
    started = time.monotonic()
    result = fetch(base, "/v1/engine/completions", body)
    elapsed = time.monotonic() - started
    (args.out / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    after = fetch(base, "/")
    (args.out / "after.json").write_text(json.dumps(after, indent=2))
    summary = dict(prompt_tokens=result.get("prompt_tokens"), cached_tokens=result.get("cached_tokens"),
                   completion_tokens=result.get("completion_tokens"), wall_seconds=elapsed,
                   served_delta=after.get("served", 0) - before.get("served", 0),
                   scope="same prompt IDs; fresh prefix; bounded output; seeded replay, not original RNG")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    if result.get("cached_tokens") != 0 or result.get("prompt_tokens") != count:
        raise SystemExit("replay did not satisfy exact-input, fresh-prefix evidence")


if __name__ == "__main__":
    main()
