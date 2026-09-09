#!/usr/bin/env python3
"""Read-only CPU24 collection from srv1 payload via srv2/head; never compiles or submits work."""
import ast,base64,gzip,hashlib,io,itertools,json,re,shlex,subprocess,tarfile,tempfile
from pathlib import Path
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/decode24-cpu'
REVISION='82ac3c34173ae63b3dd0a42c49f8421097e96a1a'
EXPECTED_TESTS=165
SOURCE='/home/choiceoh/stkernel-ep-onepass-0909-24'
JOB='/tmp/glm53-ep-decode-cpu-0909-24'
WORKER='choiceoh@10.10.10.1'
WORKER_ROOT='/home/choiceoh/glm53-ep-cpu-0909-24'
CAPSULE='/tmp/glm53-bindings-capsule-cpu0908-2/capsule'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
CAPSULE_SHA='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def ssh(argv):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2',shlex.join(argv)],capture_output=True,timeout=55)
 require(p.returncode==0,p.stderr.decode(errors='replace')[-2000:]);return p.stdout

def command():
 inner=['nice','-n','19','python3','-B',SOURCE+'/probes/run_glm53_ep_short_decode_cpu.py','--image',IMAGE,'--output',WORKER_ROOT+'/evidence','--capsule-root',WORKER_ROOT+'/capsule','--manifest-sha256',CAPSULE_SHA]
 return ['bash',SOURCE+'/bench/fleet.sh','run','--cpu','epdecodecpu0909v24','6','TP SF6 stock/candidate and retained controls lowering on srv1 with same source/image/capsule and unchanged no-device 12GiB/4GiB/2CPU gates','--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',WORKER,shlex.join(inner)]

REMOTE=r'''import base64,hashlib,json,os,pathlib,shlex,subprocess,sys,time
P=pathlib.Path
source,job,revision,image,capsule,manifest_sha,expected_command,expected_tests,worker,worker_root=sys.argv[1:]
source=P(source);job=P(job);expected_command=json.loads(expected_command);expected_tests=int(expected_tests)
def rawfile(path):
 assert path.is_file() and not path.is_symlink()
 a=path.stat();raw=path.read_bytes();b=path.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
 assert len(raw)==a.st_size and len(raw)<128*2**20
 return raw
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'},text=True).strip()
state=dict(revision=git('rev-parse','HEAD'),status=git('status','--porcelain'),shallow=git('rev-parse','--is-shallow-repository'),alternates=(source/'.git/objects/info/alternates').exists())
assert state==dict(revision=revision,status='',shallow='false',alternates=False)
head={}
for name in ('submission.json','exit.json','fleet.log','driver.py','driver.pid'):
 raw=rawfile(job/name);head[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw),base64=base64.b64encode(raw).decode())
submission=json.loads(rawfile(job/'submission.json'));completion=json.loads(rawfile(job/'exit.json'))
assert submission['source']==str(source) and submission['revision']==revision and submission['worker']==worker and submission['worker_root']==worker_root and submission['image']==image and submission['manifest_sha256']==manifest_sha and submission['command']==expected_command
assert completion['returncode']==completion['payload_returncode']==completion['copy_returncode']==0 and completion['ended']>=completion['started']>=submission['created']
evidence={};total=0
for path in sorted((job/'evidence').rglob('*')):
 assert not path.is_symlink()
 if path.is_file():
  raw=rawfile(path);total+=len(raw);assert total<128*2**20
  evidence[path.relative_to(job/'evidence').as_posix()]=dict(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))
assert len(evidence)<512 and 'result.json' in evidence
result=json.loads(rawfile(job/'evidence/result.json'))
assert result['verdict']=='PASS' and result['phase']=='complete' and result['cuda_initialized'] is False and result['binding_runtime_rechecked'] is True
assert result['contracts']==dict(tests_run=expected_tests,errors=0,failures=0,skips=0)
assert completion['started']<=result['started']<=result['finished']<=completion['ended']
assert 'error' not in result and 'recheck_error' not in result
contracts={name:hashlib.sha256(rawfile(source/name)).hexdigest() for name in result['contract_sources']}
mounted={}
for line in rawfile(source/'build/glm53/manifest.tsv').decode().splitlines():
 if not line or line.startswith('#'):continue
 name,target,*_=line.split('\t')
 if target in result['mounted_sources']:mounted[target]=hashlib.sha256(rawfile(source/'build/glm53'/name)).hexdigest()
assert contracts==result['contract_sources'] and mounted==result['mounted_sources']
assert submission['worker_sources']==dict(contract_sources=contracts,mounted_sources=mounted)
assert submission['worker_identity']['image']==image and submission['worker_identity']['mem_available_kib']>=12*1024*1024
sys.path.insert(0,str(source/'probes'))
from glm53_ep_bindings_capsule import validate_capsule
from glm53_ep_capsule_runtime import validate_runtime_receipt
manifest=validate_capsule(P(capsule),manifest_sha);validate_runtime_receipt(result['binding_runtime'])
assert result['binding_runtime']['capsule_manifest_sha256']==manifest_sha
actual_image=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip();assert actual_image==image
worker_code="import hashlib,json,os,pathlib,subprocess,sys\nP=pathlib.Path\nsource,root,revision,image,manifest_sha=sys.argv[1:];source=P(source);root=P(root)\ndef rawfile(path):\n assert path.is_file() and not path.is_symlink()\n a=path.stat();raw=path.read_bytes();b=path.stat()\n assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)\n assert len(raw)==a.st_size and len(raw)<128*2**20\n return raw\ndef git(*args):return subprocess.check_output(['git','-C',str(source),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'},text=True).strip()\nstate=dict(revision=git('rev-parse','HEAD'),status=git('status','--porcelain'),shallow=git('rev-parse','--is-shallow-repository'),alternates=(source/'.git/objects/info/alternates').exists())\nassert state==dict(revision=revision,status='',shallow='false',alternates=False)\nevidence={};total=0\nfor path in sorted((root/'evidence').rglob('*')):\n assert not path.is_symlink()\n if path.is_file():\n  raw=rawfile(path);total+=len(raw);assert total<128*2**20\n  evidence[path.relative_to(root/'evidence').as_posix()]=dict(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))\nassert len(evidence)<512 and 'result.json' in evidence\nresult=json.loads(rawfile(root/'evidence/result.json'))\ncontracts={name:hashlib.sha256(rawfile(source/name)).hexdigest() for name in result['contract_sources']}\nmounted={}\nfor line in rawfile(source/'build/glm53/manifest.tsv').decode().splitlines():\n if not line or line.startswith('#'):continue\n name,target,*_=line.split('\\t')\n if target in result['mounted_sources']:mounted[target]=hashlib.sha256(rawfile(source/'build/glm53'/name)).hexdigest()\nassert contracts==result['contract_sources'] and mounted==result['mounted_sources']\nsys.path.insert(0,str(source/'probes'))\nfrom glm53_ep_bindings_capsule import validate_capsule\nmanifest=validate_capsule(root/'capsule',manifest_sha)\nactual_image=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip();assert actual_image==image\nprint(json.dumps(dict(source_state=state,image_id=actual_image,capsule=dict(root=str(root/'capsule'),manifest_sha256=manifest_sha,files=len(manifest['files']),strict_validation='PASS'),source_receipts=dict(contract_sources=contracts,mounted_sources=mounted),evidence=evidence)))\n"
worker_capture=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',worker,shlex.join(['python3','-B','-c',worker_code,str(source),worker_root,revision,image,manifest_sha])],timeout=45))
assert worker_capture['source_state']==state and worker_capture['image_id']==actual_image
assert worker_capture['evidence']==evidence and worker_capture['source_receipts']==dict(contract_sources=contracts,mounted_sources=mounted)
print(json.dumps(dict(time=time.time(),worker=dict(node=worker,root=worker_root,**worker_capture),source_state=state,image_id=actual_image,capsule=dict(root=capsule,manifest_sha256=manifest_sha,files=len(manifest['files']),strict_validation='PASS'),source_receipts=dict(contract_sources=contracts,mounted_sources=mounted),files=head,evidence=evidence)))
'''
def snapshot():return json.loads(ssh(['python3','-B','-c',REMOTE,SOURCE,JOB,REVISION,IMAGE,CAPSULE,CAPSULE_SHA,json.dumps(command()),str(EXPECTED_TESTS),WORKER,WORKER_ROOT]))
def main():
 require(re.fullmatch('[0-9a-f]{40}',REVISION) and type(EXPECTED_TESTS)is int and EXPECTED_TESTS>0,'bind actual CPU24 revision and successful test count first')
 require(not OUT.exists() and not OUT.with_suffix('.collecting').exists(),'refuse existing archive/staging')
 before=snapshot();head_bytes={name:base64.b64decode(value['base64'],validate=True) for name,value in before['files'].items()}
 for name,raw in head_bytes.items():require(len(raw)==before['files'][name]['size'] and sha(raw)==before['files'][name]['sha256'],'head payload hash mismatch')
 archive=ssh(['tar','-czf','-','-C',JOB,'evidence']);require(len(archive)<128*2**20,'oversized evidence archive')
 files={};total=0
 with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
  for member in tar:
   require(not member.issym() and not member.islnk(),'linked tar member')
   path=Path(member.name)
   require(not path.is_absolute() and '..' not in path.parts and path.parts[0]=='evidence','unsafe tar member')
   if member.isdir():continue
   require(member.isfile() and len(path.parts)>1 and member.size<128*2**20,'non-regular/oversized tar member')
   relative=path.relative_to('evidence').as_posix();require(relative not in files,'duplicate tar member')
   raw=tar.extractfile(member).read();require(len(raw)==member.size,'truncated tar member');total+=len(raw)
   require(total<128*2**20 and len(files)<512,'archive limits');files[relative]=raw
 descriptors={name:dict(sha256=sha(raw),size=len(raw)) for name,raw in files.items()}
 require(descriptors==before['evidence'],'original archive differs from first snapshot')
 result=json.loads(files['result.json'])
 def gitfile(name):return subprocess.check_output(['git','-C',str(ROOT),'show',REVISION+':'+name])
 # Every actual contract/mount must match both the completed head source and
 # the immutable local git object. No current worktree or CUDA import is used.
 require({name:sha(gitfile(name)) for name in result['contract_sources']}==result['contract_sources'],'local frozen contract mismatch')
 expected={}
 for line in gitfile('build/glm53/manifest.tsv').decode().splitlines():
  if line and not line.startswith('#'):
   name,target,*_=line.split('\t')
   if target in result['mounted_sources']:expected[target]=sha(gitfile('build/glm53/'+name))
 require(expected==result['mounted_sources'],'local frozen mount mismatch')
 # Execute only the frozen pure artifact validator function, not module setup,
 # test cases, compilation entrypoints or any CUDA imports.
 runner=gitfile('probes/run_glm53_ep_short_decode_cpu.py');tree=ast.parse(runner)
 fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='validate_artifacts')
 ns=dict(Path=Path,hashlib=hashlib,itertools=itertools,re=re)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'frozen-artifact-validator','exec'),ns)
 with tempfile.TemporaryDirectory(prefix='glm53-cpu24-artifact-check-') as temp:
  directory=Path(temp)
  for name,raw in files.items():p=directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
  ns['validate_artifacts'](directory,result)
 after=snapshot()
 for snap in (before,after):
  for entry in snap['files'].values():entry.pop('base64',None)
 # Neither metadata capture submits or launches a workload.
 for key in ('source_state','image_id','capsule','source_receipts','files','evidence','worker'):require(before[key]==after[key],'snapshot changed: '+key)
 return write_archive(before,after,archive,files,runner,head_bytes)

def write_archive(before,after,archive,files,runner,head_bytes):
 stage=OUT.with_suffix('.collecting');require(not OUT.exists() and not stage.exists(),'archive/staging appeared')
 values={};originals={}
 def add(name,raw,origin,compress=False):
  stored=gzip.compress(raw,mtime=0) if compress else raw
  require(name not in values,'duplicate stored name');values[name]=stored
  originals[name]=dict(origin=origin,original_bytes=len(raw),original_sha256=sha(raw),stored_bytes=len(stored),stored_sha256=sha(stored),compression='gzip-mtime0' if compress else 'none')
 for name,raw in head_bytes.items():add('head/'+name+('.gz' if name.endswith(('.log','.py')) else ''),raw,JOB+'/'+name,name.endswith(('.log','.py')))
 add('evidence.tar.gz',archive,'read-only tar of '+JOB+'/evidence')
 add('result.json',files['result.json'],JOB+'/evidence/result.json')
 add('frozen-runner.py.gz',runner,'git '+REVISION+':probes/run_glm53_ep_short_decode_cpu.py',True)
 capture=dict(schema=1,scope='Completed srv1 CPU24 lowering/contracts via normal head fleet only; no GPU/HTTP/performance acceptance',source=SOURCE,revision=REVISION,head_job=JOB,worker=WORKER,worker_root=WORKER_ROOT,before=before,after=after,evidence_tar_sha256=sha(archive),original_files=len(files),original_bytes=sum(map(len,files.values())),archive_bytes=len(archive),artifact_validator='exact frozen validate_artifacts function; no module entrypoint or CUDA import')
 add('capture.json',(json.dumps(capture,sort_keys=True,indent=2)+'\n').encode(),'verified original snapshots')
 text='# CPU24: completed srv1 no-device compile receipt via head fleet\n\nThis archive was collected after the normal `fleet --cpu` command returned0. Frozen source `'+REVISION+'`, the exact bounded runc/network-none/4GiB/2CPU launcher, immutable image, capsule manifest, actual contracts/mounted sources and original evidence file hashes were checked before and after transfer. Both head and srv1 sources have full independent history and no alternates. The exact normal head fleet command invokes the bounded runner through srv1 SSH; payload and result-copy return codes must both be zero. The srv1 evidence inventory, capsule, image and source bytes must agree with the head copy before and after transfer. The result must contain '+str(EXPECTED_TESTS)+' CPU tests with zero failures/errors/skips and CUDA uninitialized.\n\nThe original tar retains every PTX, cubin, resource log and result byte. Its safe regular-file inventory must exactly equal both head and srv1 snapshots, and the frozen pure artifact validator checks all descriptor hashes, resource strings and the complete compiled artifact set. Source bytes also match local immutable git objects. This is compilation/CPU proof only; it does not establish serving canary, sanitizer, throughput or default adoption. The CPU uses the isolated13.0.3 bindings capsule; production13.3.1 binary identity is not asserted.\n\nNo compilation, GPU execution, HTTP request, queue change or service change is performed by collection. Logs and frozen Python snapshots use deterministic gzip; original/stored hashes are preserved.\n'
 add('README.md',text.encode(),'bounded evidence scope')
 add('collect.py',Path(__file__).read_bytes(),'exact executed collector')
 values['originals.json']=(json.dumps(originals,sort_keys=True,indent=2)+'\n').encode()
 values['SHA256SUMS']=''.join(sha(raw)+'  '+name+'\n' for name,raw in sorted(values.items())).encode()
 stage.mkdir()
 for name,raw in values.items():path=stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 for name,raw in values.items():require(sha((stage/name).read_bytes())==sha(raw),'stored hash mismatch')
 stage.rename(OUT)
 print(json.dumps(dict(archive=str(OUT),files=len(values),original_files=len(files),result_sha256=sha(files['result.json']),sums_sha256=sha(values['SHA256SUMS'])),sort_keys=True))
if __name__=='__main__':main()
