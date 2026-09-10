#!/usr/bin/env python3
"""Verify archived stored/original bytes and optional immutable git source; read only."""
import argparse,gzip,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--repo',type=Path);a=p.parse_args();root=a.archive.resolve()
def sha(raw):return hashlib.sha256(raw).hexdigest()
m=json.loads((root/'manifest.json').read_text());originals={}
for name,row in m['files'].items():
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=(root/rel).read_bytes();assert sha(raw)==row['stored_sha256'] and len(raw)==row['stored_size']
 data=gzip.decompress(raw) if row['encoding']=='gzip' else raw
 assert sha(data)==row['original_sha256'] and len(data)==row['original_size'];originals[name]=data
checks={}
for line in (root/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1);assert sha((root/name).read_bytes())==digest;checks[name]=digest
actual={str(x.relative_to(root)) for x in root.rglob('*') if x.is_file()}
assert actual==set(checks)|{'SHA256SUMS'}
assert set(checks)=={r['path'] for r in m['files'].values()}|{'manifest.json'}
v=json.loads(originals['verification.json']);r=json.loads(originals['originals/result.json']);c=json.loads(originals['originals/contracts.json'])
assert sha(originals['originals/result.json'])==v['result_sha256'] and r['contracts']==c['contracts']==v['contracts']
if a.repo:
 s=json.loads(originals['source-verification.json']);assert s['source']==v['source']
 for path,digest in s['verified'].items():
  raw=subprocess.check_output(['git','show',s['source']+':'+path],cwd=a.repo);assert sha(raw)==digest
print(json.dumps(dict(archive=str(root),files=len(actual),source=v['source'],verdict=r['verdict'],contracts=r['contracts'],lowerings=v['lowerings'],manifest_sha256=sha((root/'manifest.json').read_bytes())),sort_keys=True))
