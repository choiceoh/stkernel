"""Download a bounded public UltraChat calibration sample, preserving provenance.

Raw conversations remain in the requested output directory. Dataset Viewer
responses are hashed; the Hub revision observed during retrieval is recorded,
not claimed to pin the separately generated Viewer data. MIT-licensed source:
https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


DATASET = 'HuggingFaceH4/ultrachat_200k'


def get(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def page(spec):
    split, offset, length, destination = spec
    url = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(
        dict(dataset=DATASET, config='default', split=split, offset=offset, length=length))
    raw = get(url)
    data = json.loads(raw)
    if len(data['rows']) != length:
        raise ValueError('incomplete dataset page')
    rows, manifest = [], []
    for item in data['rows']:
        if item.get('truncated_cells'):
            raise ValueError('truncated source conversation')
        source = item['row']
        messages = source['messages'][:]
        while messages and messages[-1]['role'] != 'user':
            messages.pop()
        if not messages:
            raise ValueError('source has no user turn')
        row_id = f'ultrachat-{split}-{item["row_idx"]}'
        rows.append(dict(id=row_id, split=destination, messages=messages))
        manifest.append(dict(id=row_id, source_prompt_id=source['prompt_id'],
                             conversation_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()))
    return rows, dict(url=url, response_sha256=hashlib.sha256(raw).hexdigest(), rows=manifest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--local', type=Path, default=Path(__file__).parent/'fixtures/sparse_calibration_prompts.json')
    args = ap.parse_args()
    info = json.loads(get('https://huggingface.co/api/datasets/'+DATASET))
    specs = [('train_sft', n, 32, 'train') for n in (0, 1000, 10000, 50000)]
    specs += [('test_sft', 0, 16, 'validation'), ('test_sft', 1000, 16, 'test')]
    rows = json.loads(args.local.read_text())
    pages = []
    with ThreadPoolExecutor(4) as executor:
        for batch, metadata in executor.map(page, specs):
            rows.extend(batch); pages.append(metadata)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate fixture ID')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    report = dict(dataset=DATASET, source_url='https://huggingface.co/datasets/'+DATASET,
                  license=info['cardData']['license'], observed_hub_revision=info['sha'],
                  viewer_revision_pinned=False, pages=pages,
                  local_fixture_sha256=hashlib.sha256(args.local.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                  prompt_counts={s:sum(r['split']==s for r in rows) for s in ('train','validation','test')})
    args.out.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(prompt_counts=report['prompt_counts'], bytes=args.out.stat().st_size)))


if __name__ == '__main__':
    main()
