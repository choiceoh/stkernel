#!/usr/bin/env python3
"""Private, read-only collection of terminal CPU7 pass, never a test invocation."""
import gzip,hashlib,importlib.util,io,json
from pathlib import Path
import tarfile
OLD=Path('/tmp/glm53_archive_prep_cpu10.py')
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='c786b6125471acb4378c4930bc91e1fcce1dad0ca079a30c30459ce90d723a20', 'collector reference drift'
spec=importlib.util.spec_from_file_location('old_collector',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
old.READ=old.READ.replace("roots={n:pathlib.Path('/home/choiceoh/glm53-ep-prep-%d-cpu-evidence'%n) for n in (10,)}", "roots={7:pathlib.Path('/home/choiceoh/glm53-ep76-7-cpu-evidence')}").replace("prefix='cpu10/'", "prefix=''")
old.READ=old.READ.replace("  image=run(['docker','image','inspect','--format','{{.Id}}','sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211']),\n", "  image_observation='not re-inspected; no Docker invocation during collection',\n")
assert 'docker' not in old.READ
OUT=Path('/tmp/glm53-ep76-cpu7-archive')
def sha(b):return hashlib.sha256(b).hexdigest()
assert not OUT.exists()
ex=Path('/tmp/glm53-ep76-cpu7-execution.json').read_bytes();e=json.loads(ex)
assert e['revision']=='ca076d35e64a6a19e90dffe54054269d1a5e1887' and e['returncode']==0 and e['finished']>=e['started']
assert e['session']=='epdecode76cpu0910v7'
assert e['output']=='/home/choiceoh/glm53-ep76-7-cpu-evidence'
assert e['source']=='/home/choiceoh/stkernel-ep-tiled-0909-ring'
before_raw=old.remote('inventory');before=json.loads(before_raw)
bundle=old.remote('bundle')
after_raw=old.remote('inventory');after=json.loads(after_raw)
assert before['files']==after['files']
assert before['image_observation']==after['image_observation']
assert before['head']==after['head']==e['revision'] and before['status']==after['status']==''
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
(OUT/'fleet.log.gz').write_bytes(gzip.compress(Path('/tmp/glm53-ep76-cpu7.log').read_bytes(),mtime=0))
print(json.dumps({'archive':str(OUT),'originals':len(raw),'original_bytes':sum(map(len,raw.values()))},sort_keys=True))
