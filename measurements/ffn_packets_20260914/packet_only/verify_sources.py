"""Bind each retained report to immutable repository source, without a GPU."""
import hashlib
import json
from pathlib import Path
import subprocess

folder = Path(__file__).resolve().parent
revisions = {
    'cpu': 'a61644a4',
    'cpu-postmerge': 'ff877806',
    'cpu-ci-unscoped': '98d9285c',
    'cpu-ci-fix': 'f218bbf7',
    'compile': 'a61644a4',
    'compile-aligned': '0641070e',
    'compile-vector': 'fa2050ea',
    'compile-router-scale': '57e63935',
    'gpu-v3-failure': 'a61644a4',
    'gpu-v4-failure': 'ff877806',
    'gpu-v5': '0641070e',
    'gpu-v6': 'fa2050ea',
    'gpu-v7': '57e63935',
}
records = {}
for name, short in revisions.items():
    revision = subprocess.check_output(['git', 'rev-parse', short], text=True).strip()
    report = json.loads((folder / (name + '.json')).read_text())
    for path, expected in report['source_sha256'].items():
        source = subprocess.check_output(['git', 'show', revision + ':' + path])
        if hashlib.sha256(source).hexdigest() != expected:
            raise SystemExit(f'{name}: source differs at {path}')
    records[name] = dict(revision=revision, all_sources_match=True,
                         source_count=len(report['source_sha256']))
(folder / 'source_revisions.json').write_text(json.dumps(records, indent=2) + '\n')
print(f'PASS: {len(records)} reports match their frozen source revisions')
