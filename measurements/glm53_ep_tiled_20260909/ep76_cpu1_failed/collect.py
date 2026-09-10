#!/usr/bin/env python3
"""Private, read-only collection of terminal CPU1 failure, never a test invocation."""
import gzip,hashlib,importlib.util,io,json
from pathlib import Path
import tarfile
OLD=Path('/tmp/glm53_archive_prep_cpu10.py')
spec=importlib.util.spec_from_file_location('old_collector',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
old.READ=old.READ.replace("roots={n:pathlib.Path('/home/choiceoh/glm53-ep-prep-%d-cpu-evidence'%n) for n in (10,)}", "roots={1:pathlib.Path('/home/choiceoh/glm53-ep76-1-cpu-evidence')}").replace("prefix='cpu10/'", "prefix=''")
OUT=Path('/tmp/glm53-ep76-cpu1-failed-archive')
def sha(b):return hashlib.sha256(b).hexdigest()
assert not OUT.exists()
ex=Path('/tmp/glm53-ep76-cpu1-execution.json').read_bytes();e=json.loads(ex)
assert e['revision']=='680a899d7de0c823905b89eed1e27cfc375feaf8' and e['returncode']==1 and e['finished']>=e['started']
before_raw=old.remote('inventory');before=json.loads(before_raw)
bundle=old.remote('bundle')
after_raw=old.remote('inventory');after=json.loads(after_raw)
assert before['files']==after['files']
assert before['image']==after['image']==old.IMAGE
assert before['capsule_manifest_sha256']==after['capsule_manifest_sha256']==old.CAPSULE
raw={}
with tarfile.open(fileobj=io.BytesIO(bundle),mode='r:') as t:
 for ent in t.getmembers():
  name=ent.name;p=Path(name)
  assert ent.isfile() and not p.is_absolute() and '..' not in p.parts and name not in raw
  data=t.extractfile(ent).read();r=before['files'][name]
  assert len(data)==r['size'] and sha(data)==r['sha256'];raw[name]=data
assert set(raw)==set(before['files'])
OUT.mkdir()
for name,data in raw.items():
 p=OUT/'originals'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
(OUT/'worker-before.json').write_bytes(before_raw);(OUT/'worker-after.json').write_bytes(after_raw)
(OUT/'execution.json').write_bytes(ex)
(OUT/'fleet.log.gz').write_bytes(gzip.compress(Path('/tmp/glm53-ep76-cpu1.log').read_bytes(),mtime=0))
print(json.dumps({'archive':str(OUT),'originals':len(raw),'original_bytes':sum(map(len,raw.values())), 'receipts':{k:json.loads(v) for k,v in raw.items() if k in ('result.json','contracts.json')}},sort_keys=True))
