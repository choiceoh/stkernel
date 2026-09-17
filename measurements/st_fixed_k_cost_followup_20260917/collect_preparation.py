"""Retain measured-phase preparation counters from the existing server artifacts.

Run on srv2 with the consumer JSONL path. No server or GPU requests are made.
"""
import hashlib
import json
from pathlib import Path
import sys

rows = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    run = json.loads(line)
    for source in sorted(Path(run['artifacts']).glob('measure-*/server.json')):
        raw = source.read_bytes()
        for rank in json.loads(raw).get('ranks', []):
            rows.append(dict(arm=run['name'], run=run['run_index'], run_id=run['run_id'],
                phase=source.parent.name, rank=rank['rank'], source=str(source),
                source_sha256=hashlib.sha256(raw).hexdigest(),
                preparation_changed=rank.get('preparation_changed'),
                before=rank.get('preparation_before'), after=rank.get('preparation_after')))
print(json.dumps(dict(scope='Observed counters; late compile/load is not waived', records=rows), indent=2))
