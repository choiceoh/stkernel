"""Feed bounded C1 greedy requests to the admitted capture boot (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def post(base, route, body):
    request = urllib.request.Request(base + route, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8001')
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.capture / 'manifest.json').read_text())
    assert manifest['rank'] == 0 and manifest['world'] == 4, manifest
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    def cases():
        paths = sorted(args.capture.glob('case-*.json'))
        return [json.loads(path.read_text()) for path in paths]
    for item in json.loads(args.requests.read_text()):
        if len(cases()) >= 16:
            break
        prefix = args.out / f"request-{item['index']:02d}"
        tokenized = post(args.url, '/tokenize', {
            'messages': item['messages'], 'chat_template_kwargs': item['chat_template_kwargs']})
        ids = tokenized['tokens']
        body = dict(ids=ids, max_tokens=item['max_tokens'], temperature=0.0, retain=False,
                    cache_salt=f"draft-replay-0916-{item['index']}")
        # Direct engine dialect preserves the rendered thinking=false prompt and
        # gives exact output ids, without adding seed/grammar/reasoning constraints.
        prefix.with_suffix('.input.json').write_text(json.dumps(body) + '\n')
        before = len(cases())
        started = time.monotonic()
        print(f"request {item['index']}: context={item['context_label']} actual_tokens={len(ids)} cases_before={before}", flush=True)
        response = post(args.url, '/v1/engine/completions', body)
        seconds = time.monotonic() - started
        prefix.with_suffix('.response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n')
        records.append(dict(index=item['index'], context_label=item['context_label'],
                            corpus_seed=item['corpus_seed'], prompt_tokens=len(ids),
                            completion_tokens=response['completion_tokens'], seq=response['seq'],
                            output_ids_sha256=hashlib.sha256(json.dumps(response['ids']).encode()).hexdigest(),
                            wall_seconds=seconds, cases_added=len(cases())-before,
                            performance_baseline=False))
        (args.out / 'requests.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(records[-1]), flush=True)
    labels = cases()
    result = dict(requests=len(records), cases=len(labels), complete=sum(x['complete'] for x in labels),
                  state_sha256=manifest['state_sha256'], performance_baseline=False)
    (args.out / 'collection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if len(labels) != 16 or not all(x['complete'] for x in labels):
        raise SystemExit('capture incomplete; inspect retained cases and request records')


if __name__ == '__main__':
    main()
