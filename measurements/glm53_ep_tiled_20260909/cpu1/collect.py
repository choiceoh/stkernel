#!/usr/bin/env python3
"""Read-only collection of the completed eptiledcpu0909v1 worker evidence."""
import ast,base64,datetime,gzip,hashlib,io,json,pathlib,shlex,shutil,subprocess,tarfile
ROOT=pathlib.Path(__file__).resolve().parent
REPO=ROOT.parents[2]
REV='a62d492b90caf712d0522528c0baa02926c256d6'
SOURCE='/home/choiceoh/stkernel-ep-tiled-0909-cpu1'
EVIDENCE='/home/choiceoh/glm53-ep-tiled-cpu1-evidence'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
CAPSULE='/home/choiceoh/glm53-ep-cpu-0909-21/capsule'
LIMIT=128*1024*1024

def sha(b):return hashlib.sha256(b).hexdigest()
def remote_worker(script):
 r=subprocess.run(['ssh','-o','BatchMode=yes','choiceoh@srv2','ssh','-o','BatchMode=yes','choiceoh@10.10.10.4','python3','-c',shlex.quote(shlex.quote(script))],capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
 assert len(r.stdout)<=LIMIT,'128 MiB collection budget exceeded'
 return r.stdout
inventory=f'''import pathlib,json,hashlib,subprocess,time
p=pathlib.Path({EVIDENCE!r}); source=pathlib.Path({SOURCE!r}); files={{}}
for f in sorted(p.rglob('*')):
 if f.is_symlink():raise RuntimeError('symlink in evidence')
 if f.is_file():
  raw=f.read_bytes();files[str(f.relative_to(p))]={{'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}}
assert sum(x['size'] for x in files.values())<={LIMIT}
head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
image=subprocess.check_output(['sudo','-n','docker','image','inspect',{IMAGE!r},'--format','{{{{.Id}}}}'],text=True).strip()
m=pathlib.Path({CAPSULE!r})/'capsule-manifest.json'; manifest=json.loads(m.read_text());bad=[]
for name,row in manifest['files'].items():
 f=m.parent/name
 if f.is_symlink() or not f.is_file() or f.stat().st_size!=row['size'] or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:bad.append(name)
print(json.dumps(dict(captured_at=time.time(),node='10.10.10.4',source=str(source),head=head,status=status,image=image,capsule={CAPSULE!r},capsule_manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),capsule_files=len(manifest['files']),capsule_mismatches=bad,files=files)))
'''
assert not (ROOT/'result.json').exists(),'archive already collected'
before_raw=remote_worker(inventory);(ROOT/'worker-before.json').write_bytes(before_raw);before=json.loads(before_raw)
assert before['head']==REV and not before['status'] and before['image']==IMAGE
assert before['capsule_manifest_sha256']=='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab' and not before['capsule_mismatches']
# Result first: this exact original file is also included in the complete tar.
result_raw=remote_worker(f'import pathlib,sys;sys.stdout.buffer.write(pathlib.Path({EVIDENCE!r},"result.json").read_bytes())')
assert sha(result_raw)==before['files']['result.json']['sha256'];(ROOT/'result.json').write_bytes(result_raw)
result=json.loads(result_raw)
assert result['verdict']=='PASS' and result['phase']=='complete' and result['cuda_initialized'] is False
assert result['compile_only'] is True and result['binding_runtime_rechecked'] is True
assert result['gpu_numerics_acceptance'] is False and result['performance_acceptance'] is False
assert not any(k in result for k in ('error','recheck_error','cleanup_error'))
tar_raw=remote_worker(f'''import pathlib,tarfile,io,sys
p=pathlib.Path({EVIDENCE!r}); buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for f in sorted(p.rglob('*')):
  if f.is_file():
   assert not f.is_symlink();t.add(f,arcname=str(f.relative_to(p)),recursive=False)
assert len(buf.getvalue())<={LIMIT};sys.stdout.buffer.write(buf.getvalue())
''')
(ROOT/'original-evidence.tar.gz').write_bytes(tar_raw)
with tarfile.open(fileobj=io.BytesIO(tar_raw),mode='r:gz') as t:
 members=t.getmembers();assert sum(x.size for x in members)<=LIMIT
 assert {x.name for x in members}==set(before['files'])
 for m in members:
  assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
  raw=t.extractfile(m).read();assert sha(raw)==before['files'][m.name]['sha256'] and len(raw)==m.size
  out=ROOT/m.name
  if out.exists():assert out.read_bytes()==raw
  else:out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
after_raw=remote_worker(inventory);(ROOT/'worker-after.json').write_bytes(after_raw);after=json.loads(after_raw)
for key in ('source','head','status','image','capsule','capsule_manifest_sha256','capsule_files','capsule_mismatches','files'):assert before[key]==after[key],key
for source,dest in [('/tmp/glm53-ep-tiled-cpu1-submission.json','submission.json'),('/tmp/glm53-ep-tiled-cpu1-fleet.log','fleet.log')]:shutil.copyfile(source,ROOT/dest)
submission=json.loads((ROOT/'submission.json').read_text());assert submission['source_revision']==REV and submission['source']==SOURCE and submission['returncode']==0
# Check all receipt source bytes against the immutable local git commit.
manifest=subprocess.check_output(['git','show',REV+':build/glm53/manifest.tsv'],cwd=REPO).decode()
by_target={x.split('\t')[1]:x.split('\t')[0] for x in manifest.splitlines() if x and not x.startswith('#')}
verified={}
for target,expected in result['mounted_sources'].items():
 path='build/glm53/'+by_target[target];raw=subprocess.check_output(['git','show',REV+':'+path],cwd=REPO)
 assert sha(raw)==expected,target;verified[path]=dict(sha256=expected,scope='immutable git source equals actual-mounted receipt')
for path,expected in result['contract_sources'].items():
 raw=subprocess.check_output(['git','show',REV+':'+path],cwd=REPO);assert sha(raw)==expected,path
 out=ROOT/'contract-source'/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
 verified[path]=dict(sha256=expected,scope='immutable git source equals CPU contract receipt')
(ROOT/'source-verification.json').write_text(json.dumps(dict(revision=REV,verified=verified),indent=2)+'\n')
print(json.dumps(dict(verdict='ORIGINALS_COLLECTED',original_files=len(before['files']),original_bytes=sum(x['size'] for x in before['files'].values()),tar_sha256=sha(tar_raw),result_sha256=sha(result_raw),mounted_sources=len(result['mounted_sources']),contract_sources=len(result['contract_sources']))))
