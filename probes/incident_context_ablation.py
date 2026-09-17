#!/usr/bin/env python3
"""Replay the incident prompt with its assistant-provenance records neutralized.

The incident's context is an agentic transcript whose tool and memory records
quote the assistant's own reasoning (`[ctx] [assistant] The user asks ...`,
`[assistant] ...`, `[source=session ref=".../assistant"]`). This probe asks
whether that material is the differentiator, by editing the prompt at the id
level: each selected line's span is replaced with the same number of a neutral
filler id, so the token count is unchanged and every other id is byte-identical.

That is a quality control, not a timing one. It needs the private engine record
(the prompt ids) and a door whose owner and served counters it can read:

    python3 probes/incident_context_ablation.py --record /tmp/<private>.json \
        --door http://127.0.0.1:8000 --arm broad --out /tmp/ablation

The door's admission is checked, the served delta is recorded, and outputs land
outside the repository (private prompt text and answers never enter git).
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path

FILLER = 15                     # the token for "0": neutral, single-piece, in-vocabulary
ARMS = {
    'records': (r'\[ctx\] \[assistant\]', r'\[assistant\] The user asks'),
    'broad': (r'\[ctx\] \[assistant\]', r'\[assistant\]', r'\*\*\[assistant\]\*\*',
              r'source=session ref="[^"]*assistant"', r'cl:[^ ]*assistant', r'\[도구'),
}


def http(door, path, body=None, timeout=600):
    request = urllib.request.Request(door + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def detokenize(door, ids):
    out = http(door, '/detokenize', {'tokens': list(ids)})
    for key in ('text', 'prompt', 'content'):
        if isinstance(out.get(key), str):
            return out[key]
    raise SystemExit(f'detokenize returned no text: {sorted(out)}')


def token_at(door, ids, char_offset):
    """Smallest k whose decoded prefix reaches the character offset."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(detokenize(door, ids[:mid])) >= char_offset:
            hi = mid
        else:
            lo = mid + 1
    return lo


def neutralize(door, ids, arm):
    """The ablated ids, the spans, and the filler id; the token count never changes."""
    text = detokenize(door, ids)
    lines, offset = text.splitlines(keepends=True), []
    running = 0
    for line in lines:
        offset.append(running)
        running += len(line)
    patterns = ARMS[arm]
    targets = sorted({i for i, line in enumerate(lines)
                      if any(re.search(p, line) for p in patterns)})
    if not targets:
        raise SystemExit(f'no {arm} record lines in this prompt')
    spans = [(token_at(door, ids, offset[i]), token_at(door, ids, offset[i] + len(lines[i])))
             for i in targets]
    ablated = list(ids)
    for start, end in sorted(spans, reverse=True):
        ablated[start:end] = [FILLER] * (end - start)
    if len(ablated) != len(ids):
        raise SystemExit(f'edit changed the token count: {len(ids)} -> {len(ablated)}')
    return ablated, spans, len(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True, help='the private engine record with prompt ids')
    parser.add_argument('--door', default='http://127.0.0.1:8000')
    parser.add_argument('--arm', choices=('baseline', 'records', 'broad'), default='broad')
    parser.add_argument('--out', required=True, help='directory for the private output; outside the repository')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-tokens', type=int, default=1024)
    args = parser.parse_args()

    record = json.loads(Path(args.record).read_text())
    ids = record['tokens'][:record['prompt_len']]
    spans, lines = [], 0
    prompt = list(ids)
    if args.arm != 'baseline':
        prompt, spans, lines = neutralize(args.door, ids, args.arm)

    before = http(args.door, '/')
    if any(before.get('running', [])) or before.get('waiting') or before.get('queued'):
        raise SystemExit('the door is not idle: this probe must admit into an idle engine')
    owner = before['fleet']['owner']
    started = time.monotonic()
    result = http(args.door, '/v1/engine/completions',
                  dict(ids=prompt, temperature=args.temperature, seed=args.seed, top_p=1, top_k=-1,
                       max_tokens=args.max_tokens, retain=False, cache_salt='incident-ablation-' + uuid.uuid4().hex))
    after = http(args.door, '/')
    if after['fleet']['owner'] != owner:
        raise SystemExit(f'the owner changed mid-request: {owner} -> {after["fleet"]["owner"]}')
    served_delta = after['served'] - before['served']
    text = result.get('text') or detokenize(args.door, result['ids'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'ablation-{args.arm}-{int(time.time())}.json').write_text(json.dumps(result, ensure_ascii=False))
    print(json.dumps(dict(arm=args.arm, lines=lines, spans=spans,
                          prompt_tokens=len(prompt), cached_tokens=result.get('cached_tokens'),
                          completion_tokens=result.get('completion_tokens'), served_delta=served_delta,
                          seconds=round(time.monotonic() - started, 1),
                          text_sha256=hashlib.sha256(text.encode()).hexdigest()), ensure_ascii=False))


if __name__ == '__main__':
    main()
