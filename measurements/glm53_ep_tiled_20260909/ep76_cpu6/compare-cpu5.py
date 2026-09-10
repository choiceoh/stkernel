#!/usr/bin/env python3
import gzip,hashlib,json,subprocess
from pathlib import Path
OLD=Path('/tmp/glm53-ep76-cpu5-failed-archive');NEW=Path('/tmp/glm53-ep76-cpu6-archive');REPO=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OLDREV='48df51174e0f76a726ca87cc7652dc57f2ad8724';NEWREV='4618859c90131b33c5d9ebd85a67f1537497a357'
def sha(b):return hashlib.sha256(b).hexdigest()
m=json.loads((OLD/'manifest.json').read_text())
def old(name):
 row=m['files']['originals/'+name];b=(OLD/row['path']).read_bytes();b=gzip.decompress(b) if row['encoding']=='gzip' else b
 assert sha(b)==row['original_sha256'];return b
before=json.loads(old('result.json'));after=json.loads((NEW/'originals/result.json').read_bytes())
rows=[]
for name,row in m['files'].items():
 if not name.startswith('originals/') or not name.endswith(('.ptx','.cubin','.resources.log')):continue
 rel=name.removeprefix('originals/');a=old(rel);b=(NEW/name).read_bytes();assert a==b
 rows.append(dict(path=rel,sha256=sha(b),bytes=len(b),identical=True))
assert len(rows)==69
assert before['mounted_sources']==after['mounted_sources'] and before['binding_runtime']==after['binding_runtime'] and before['scatter_helper']==after['scatter_helper']
assert set(before['contract_sources'])==set(after['contract_sources'])
changed=[dict(path=p,before=d,after=after['contract_sources'][p]) for p,d in before['contract_sources'].items() if d!=after['contract_sources'][p]]
assert [r['path'] for r in changed]==['tests/test_glm53_ep_tiled_static.py']
files=subprocess.check_output(['git','diff','--name-only',OLDREV,NEWREV],cwd=REPO,text=True).splitlines();assert files==['tests/test_glm53_ep_tiled_static.py']
report=dict(before_source=OLDREV,after_source=NEWREV,before_verdict=before['verdict'],after_verdict=after['verdict'],
 before_result_sha256=sha(old('result.json')),after_result_sha256=sha((NEW/'originals/result.json').read_bytes()),
 identical_artifacts=69,artifact_types=dict(ptx=23,cubin=23,resources=23),artifacts=rows,mounted_sources_identical=22,contract_sources=46,changed_contracts=changed,
 binding_runtime_identical=True,scatter_helper_identical=True,full_git_changed_files=files,scope='Byte equality and exact immutable source diff; no performance inference.')
(NEW/'cpu5-artifact-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
(NEW/'cpu5-to-cpu6-source.diff').write_bytes(subprocess.check_output(['git','diff',OLDREV,NEWREV],cwd=REPO))
(NEW/'compare-cpu5.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in report.items() if k!='artifacts'},sort_keys=True))
