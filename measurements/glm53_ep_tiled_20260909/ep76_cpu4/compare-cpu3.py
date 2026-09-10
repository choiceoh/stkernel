import gzip,hashlib,json,subprocess
from pathlib import Path
OLD=Path('/tmp/glm53-ep76-cpu3-archive');NEW=Path('/tmp/glm53-ep76-cpu4-archive')
REPO=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
BEFORE='29daef8b95f3dd098ba24b2fae61b322d70ecb38';AFTER='3fab1ce81a79936b3b76fa4d87b447d9f55bacba'
def sha(b):return hashlib.sha256(b).hexdigest()
def git(rev,path):return subprocess.check_output(['git','show',rev+':'+path],cwd=REPO)
m=json.loads((OLD/'manifest.json').read_text());a=json.loads((OLD/'originals/result.json').read_text());b=json.loads((NEW/'originals/result.json').read_text())
rows=[]
for name,row in m['files'].items():
 if not name.startswith('originals/') or not name.endswith(('.ptx','.cubin','.resources.log')):continue
 old=(OLD/row['path']).read_bytes();old=gzip.decompress(old) if row['encoding']=='gzip' else old
 target=NEW/name
 new=target.read_bytes() if target.exists() else gzip.decompress(Path(str(target)+'.gz').read_bytes())
 assert sha(old)==row['original_sha256']
 rows.append(dict(path=name.removeprefix('originals/'),cpu3_sha256=sha(old),cpu4_sha256=sha(new),identical=old==new))
assert len(rows)==69
result=dict(cpu3_source=BEFORE,cpu4_source=AFTER,artifact_files=69,identical_files=sum(x['identical'] for x in rows),different=[x for x in rows if not x['identical']],files=rows)
(NEW/'cpu3-artifact-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
comparisons={}
for key,count in [('mounted_sources',22),('contract_sources',44)]:
 assert set(a[key])==set(b[key]) and len(a[key])==count
 comparisons[key]=dict(count=count,changed=[dict(path=p,cpu3_sha256=a[key][p],cpu4_sha256=b[key][p]) for p in sorted(a[key]) if a[key][p]!=b[key][p]])
core=['overlay/modules/glm53_moe/moe_static_ep_tiled.py','overlay/modules/glm53_runtime/glm53_prep_fused.py','build/glm53/moe_static_ep_tiled.py','build/glm53/glm53_prep_fused.py']
comparisons['native_and_prep']=[dict(path=p,identical=git(BEFORE,p)==git(AFTER,p),sha256=sha(git(AFTER,p))) for p in core]
assert all(x['identical'] for x in comparisons['native_and_prep'])
paths=subprocess.check_output(['git','diff','--name-only',BEFORE,AFTER],cwd=REPO,text=True).splitlines()
comparisons['all_git_changed_paths']=paths
comparisons['all_git_changed_sources']=[dict(path=p,cpu3_sha256=sha(git(BEFORE,p)),cpu4_sha256=sha(git(AFTER,p))) for p in paths]
comparisons['binding_runtime_identical']=a['binding_runtime']==b['binding_runtime'];comparisons['scatter_helper_identical']=a['scatter_helper']==b['scatter_helper']
comparisons['scope']='FP8 dense source/generated startup memory accounting additions; pure register-layout validation is now also called after artifact transfer. Native EP and PREP bytes are identical. No runtime throughput equivalence is inferred.'
(NEW/'cpu3-source-comparison.json').write_text(json.dumps(dict(cpu3_source=BEFORE,cpu4_source=AFTER,**comparisons),indent=2)+'\n')
(NEW/'cpu3-to-cpu4-source.diff').write_bytes(subprocess.check_output(['git','diff',BEFORE,AFTER],cwd=REPO))
print(json.dumps(dict(artifact_files=69,identical_files=result['identical_files'],changed_sources={k:v['changed'] for k,v in comparisons.items() if k in ('mounted_sources','contract_sources')},runtime_equal=comparisons['binding_runtime_identical'],changed_paths=paths),indent=2))
