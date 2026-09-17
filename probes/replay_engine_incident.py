#!/usr/bin/env python3
"""Replay captured ST token IDs on an idle live door, with a fresh cache namespace.

This is a bounded incident reproduction, not a throughput benchmark. The record
contains private prompt tokens: keep request/response artifacts outside git.
"""
import argparse
import json
import re
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(base, path, body=None):
    req = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as response:
        return json.load(response)


def counters(base):
    with urlopen(base + '/metrics', timeout=10) as response:
        raw = response.read().decode()
    names = ('st:async_decode_steps_total', 'st:steps_decode_total',
             'vllm:spec_decode_num_accepted_tokens_total', 'vllm:spec_decode_num_draft_tokens_total')
    out = {}
    for name in names:
        matches = re.findall(r'^' + re.escape(name) + r'(?:\{[^\n]*\})? ([0-9.eE+-]+)$', raw, re.M)
        if not matches:
            raise RuntimeError('missing required execution counter: ' + name)
        out[name] = sum(map(float, matches))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--temperature", type=float, choices=(0.0, 1.0), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument('--diagnostic-mode', type=int, choices=(0, 1, 2, 3, 4, 5, 6), default=0,
                        help='Private diagnostic arm only: 1 host block, 2 host target, 3 host token rejection, 4 device reference, 5 single-position target, 6 device target')
    parser.add_argument('--expect-owner', help='Require this exact fleet owner before sending a request')
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 1200:
        parser.error("this bounded reproduction allows 1..1200 output tokens")
    if args.diagnostic_mode and not args.expect_owner:
        parser.error('diagnostic controls require --expect-owner for their isolated fleet window')
    if args.diagnostic_mode and not 0 <= args.seed < 1000:
        parser.error('the diagnostic seed encoding requires a seed in 0..999')
    record = json.loads(args.record.read_text())
    count = record["prompt_len"]
    ids = record["tokens"][:count]
    if len(ids) != count or not ids or any(type(t) is not int or t < 0 for t in ids):
        parser.error("record must contain the complete prompt token IDs")
    base = args.url.rstrip("/")
    before = fetch(base, "/")
    metrics_before = counters(base)
    if before.get("engine") != "ST" or any(before.get(k) for k in ("running", "waiting", "queued")):
        raise SystemExit("ST door must be idle before this reproduction")
    if before.get("fleet", {}).get("draining"):
        raise SystemExit("ST door is draining")
    owner = before.get('fleet', {}).get('owner')
    if args.expect_owner is not None and owner != args.expect_owner:
        raise SystemExit('the requested fleet owner does not own this door')
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    body = dict(ids=ids, temperature=args.temperature, seed=args.seed + args.diagnostic_mode * 1000,
                max_tokens=args.max_tokens, retain=False,
                cache_salt="incident-" + uuid.uuid4().hex)
    (args.out / "request.json").write_text(json.dumps(body))
    (args.out / "before.json").write_text(json.dumps(before, indent=2))
    started = time.monotonic()
    result = fetch(base, "/v1/engine/completions", body)
    elapsed = time.monotonic() - started
    (args.out / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    after = fetch(base, "/")
    metrics_after = counters(base)
    delta = {k: metrics_after[k] - v for k, v in metrics_before.items()}
    (args.out / "after.json").write_text(json.dumps(after, indent=2))
    summary = dict(prompt_tokens=result.get("prompt_tokens"), cached_tokens=result.get("cached_tokens"),
                   completion_tokens=result.get("completion_tokens"), wall_seconds=elapsed,
                   served_delta=after.get("served", 0) - before.get("served", 0),
                   diagnostic_mode=args.diagnostic_mode, underlying_seed=args.seed,
                   execution_delta=delta,
                   scope="same prompt IDs; fresh prefix; bounded output; seeded replay, not original RNG")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    if result.get("cached_tokens") != 0 or result.get("prompt_tokens") != count:
        raise SystemExit("replay did not satisfy exact-input, fresh-prefix evidence")
    if args.expect_owner is not None and (after.get('fleet', {}).get('owner') != owner
                                         or summary['served_delta'] != 1):
        raise SystemExit('replay did not preserve the isolated fleet owner and single-request control')
    if args.diagnostic_mode in (1, 2, 3, 5) and delta['st:async_decode_steps_total'] != 0:
        raise SystemExit('host control was bypassed by asynchronous decoding')
    if args.diagnostic_mode in (2, 5, 6) and delta['vllm:spec_decode_num_accepted_tokens_total'] != 0:
        raise SystemExit('target-only control accepted draft tokens')
    if args.diagnostic_mode in (4, 6) and delta['st:async_decode_steps_total'] <= 0:
        raise SystemExit('device control did not execute asynchronously')
    if args.diagnostic_mode == 5 and delta['vllm:spec_decode_num_draft_tokens_total'] != 0:
        raise SystemExit('single-position target control proposed draft tokens')


if __name__ == "__main__":
    main()
