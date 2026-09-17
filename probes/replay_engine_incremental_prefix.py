#!/usr/bin/env python3
"""Private incident control: build the same prompt in small cached increments.

The final request has the original IDs. This diagnoses prefill chunk size, not
cache equivalence or serving throughput. Artifacts contain private prompt IDs.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

from replay_engine_incident import fetch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8001')
    p.add_argument('--step', type=int, default=6912)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    if a.step < 768 or a.step % 768:
        p.error('step must be a positive whole 768-token prefix block')
    record = json.loads(a.record.read_text())
    ids = record['tokens'][:record['prompt_len']]
    before = fetch(a.url, '/')
    if before.get('engine') != 'ST' or any(before.get(k) for k in ('running', 'waiting', 'queued')):
        raise SystemExit('ST door must be idle')
    a.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    salt = 'incident-incremental-' + uuid.uuid4().hex
    (a.out / 'before.json').write_text(json.dumps(before, indent=2))
    for end in range(a.step, len(ids), a.step):
        started = time.monotonic()
        result = fetch(a.url, '/v1/prefix/warm', dict(ids=ids[:end], cache_salt=salt))
        result['wall_seconds'] = time.monotonic() - started
        (a.out / f'warm-{end}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    body = dict(ids=ids, temperature=1.0, seed=a.seed, max_tokens=1200, cache_salt=salt)
    (a.out / 'request.json').write_text(json.dumps(body))
    started = time.monotonic()
    result = fetch(a.url, '/v1/engine/completions', body)
    (a.out / 'response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    after = fetch(a.url, '/')
    (a.out / 'after.json').write_text(json.dumps(after, indent=2))
    print(json.dumps(dict(prompt_tokens=result['prompt_tokens'], cached_tokens=result['cached_tokens'],
                          completion_tokens=result['completion_tokens'], wall_seconds=time.monotonic()-started)), flush=True)
    if result['prompt_tokens'] != len(ids) or not result['cached_tokens']:
        raise SystemExit('control did not reuse the incrementally built prefix')


if __name__ == '__main__':
    main()
