#!/usr/bin/env python3
"""Preserve failed CPU22 contracts; read-only head access, output in /tmp only."""
import ast,base64,gzip,hashlib,io,itertools,json,re,shlex,subprocess,tarfile,tempfile
from pathlib import Path
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=Path('/tmp/glm53-cpu22-failed-archive')
REVISION='2d6b5d6168a4992645fb4daa60cbf27a2e445262'
EXPECTED_TESTS=165
SOURCE='/home/choiceoh/stkernel-ep-onepass-0909-22'
JOB='/tmp/glm53-ep-decode-cpu-0909-22'
CAPSULE='/tmp/glm53-bindings-capsule-cpu0908-2/capsule'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
CAPSULE_SHA='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def ssh(argv):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2',shlex.join(argv)],capture_output=True,timeout=55)
 require(p.returncode==0,p.stderr.decode(errors='replace')[-2000:]);return p.stdout

def command():return ['bash',SOURCE+'/bench/fleet.sh','run','--cpu','epdecodecpu0909v22','6','TP SF6 Q0 stock and candidate actual lowering; no devices, 4GiB and 2CPU','--','python3','-B',SOURCE+'/probes/run_glm53_ep_short_decode_cpu.py','--image',IMAGE,'--output',JOB+'/evidence','--capsule-root',CAPSULE,'--manifest-sha256',CAPSULE_SHA]
REMOTE=r'''import base64,hashlib,json,os,pathlib,subprocess,sys,time
P=pathlib.Path
source,job,revision,image,capsule,manifest_sha,expected_command,expected_tests=sys.argv[1:]
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
assert submission['source']==str(source) and submission['revision']==revision and submission['job']==str(job) and submission['command']==expected_command
assert completion['returncode']==1 and completion['ended']>=completion['started']>=submission['created']
evidence={};total=0
for path in sorted((job/'evidence').rglob('*')):
 assert not path.is_symlink()
 if path.is_file():
  raw=rawfile(path);total+=len(raw);assert total<128*2**20
  evidence[path.relative_to(job/'evidence').as_posix()]=dict(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))
assert len(evidence)<512 and 'result.json' in evidence
result=json.loads(rawfile(job/'evidence/result.json'))
assert result['verdict']=='FAIL' and result['phase']=='cpu-contracts'
assert 'cuda_initialized' not in result and 'binding_runtime_rechecked' not in result
assert result['contracts']==dict(tests_run=expected_tests,errors=1,failures=0,skips=0)
assert completion['started']<=result['started']<=result['finished']<=completion['ended']
assert result['error']=="AssertionError({'tests_run': 165, 'failures': 0, 'errors': 1, 'skips': 0})"
assert 'recheck_error' not in result
assert b"NameError: name '_TP_SF6_Q0_ENABLED' is not defined" in rawfile(job/'fleet.log')
contracts={name:hashlib.sha256(rawfile(source/name)).hexdigest() for name in result['contract_sources']}
mounted={}
for line in rawfile(source/'build/glm53/manifest.tsv').decode().splitlines():
 if not line or line.startswith('#'):continue
 name,target,*_=line.split('\t')
 if target in result['mounted_sources']:mounted[target]=hashlib.sha256(rawfile(source/'build/glm53'/name)).hexdigest()
assert contracts==result['contract_sources'] and mounted==result['mounted_sources']
sys.path.insert(0,str(source/'probes'))
from glm53_ep_bindings_capsule import validate_capsule
from glm53_ep_capsule_runtime import validate_runtime_receipt
manifest=validate_capsule(P(capsule),manifest_sha);validate_runtime_receipt(result['binding_runtime'])
assert result['binding_runtime']['capsule_manifest_sha256']==manifest_sha
actual_image=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip();assert actual_image==image
print(json.dumps(dict(time=time.time(),source_state=state,image_id=actual_image,capsule=dict(root=capsule,manifest_sha256=manifest_sha,files=len(manifest['files']),strict_validation='PASS'),source_receipts=dict(contract_sources=contracts,mounted_sources=mounted),files=head,evidence=evidence)))
'''
def snapshot():return json.loads(ssh(['python3','-B','-c',REMOTE,SOURCE,JOB,REVISION,IMAGE,CAPSULE,CAPSULE_SHA,json.dumps(command()),str(EXPECTED_TESTS)]))
def main():
 require(re.fullmatch('[0-9a-f]{40}',REVISION) and type(EXPECTED_TESTS)is int and EXPECTED_TESTS>0,'bind actual CPU22 revision and successful test count first')
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
 with tempfile.TemporaryDirectory(prefix='glm53-cpu22-artifact-check-') as temp:
  directory=Path(temp)
  for name,raw in files.items():p=directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
  ns['validate_artifacts'](directory,result)
 after=snapshot()
 for snap in (before,after):
  for entry in snap['files'].values():entry.pop('base64',None)
 # Neither metadata capture submits or launches a workload.
 for key in ('source_state','image_id','capsule','source_receipts','files','evidence'):require(before[key]==after[key],'snapshot changed: '+key)
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
 capture=dict(schema=1,scope='Failed CPU22 mock contract; artifacts exist, final CUDA-initialization and runtime recheck fields absent. No GPU/HTTP/performance acceptance',source=SOURCE,revision=REVISION,head_job=JOB,worker=None,before=before,after=after,evidence_tar_sha256=sha(archive),original_files=len(files),original_bytes=sum(map(len,files.values())),archive_bytes=len(archive),artifact_validator='exact frozen validate_artifacts function; no module entrypoint or CUDA import')
 add('capture.json',(json.dumps(capture,sort_keys=True,indent=2)+'\n').encode(),'verified original snapshots')
 text='# CPU22: mock contract failure after lowering\n\nFrozen source `'+REVISION+'` ran through the normal head `fleet --cpu` command and returned1. Its165 focused CPU tests had0 failures,1 error and0 skips: the old fake owner test raised `NameError: _TP_SF6_Q0_ENABLED is not defined`. The original result remains FAIL at phase `cpu-contracts`. The final CUDA initialization assertion and final binding-runtime recheck were not reached; neither status is claimed as passed. GPU22 was not submitted.\n\nOriginal30 PTX and30 cubin files plus resource logs are preserved in the tar. The complete artifact descriptors/hashes/resource strings were checked with the frozen pure validator, and contracts/mounted-source hashes agree with both the completed head source and immutable local git objects. The original no-device launcher, image and capsule identity and all source/file hashes were checked around transfer. These observations do not convert the failed job to accepted CPU proof, GPU numerical proof or throughput evidence.\n\nOnly read-only collection and this private /tmp archive were performed. Logs and frozen Python snapshots use deterministic gzip. All original/stored hashes remain in the manifests; no recompile, test rerun, service change, HTTP request or queue submission occurred.\n'
 add('failure-summary.json',(json.dumps(dict(verdict='CPU_CONTRACT_ERROR',revision=REVISION,returncode=1,phase='cpu-contracts',tests_run=165,failures=0,errors=1,skips=0,error="NameError: _TP_SF6_Q0_ENABLED is not defined",ptx_files=sum(name.endswith('.ptx') for name in files),cubin_files=sum(name.endswith('.cubin') for name in files),cuda_initialized=None,binding_runtime_rechecked=None,cpu_acceptance=False,gpu_submitted=False,performance_acceptance=False),sort_keys=True,indent=2)+'\n').encode(),'exact failed result and original fleet log')
 add('README.md',text.encode(),'bounded failure evidence scope')
 add('collect.py',Path(__file__).read_bytes(),'exact executed collector')
 values['originals.json']=(json.dumps(originals,sort_keys=True,indent=2)+'\n').encode()
 values['SHA256SUMS']=''.join(sha(raw)+'  '+name+'\n' for name,raw in sorted(values.items())).encode()
 stage.mkdir()
 for name,raw in values.items():path=stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 for name,raw in values.items():require(sha((stage/name).read_bytes())==sha(raw),'stored hash mismatch')
 stage.rename(OUT)
 print(json.dumps(dict(archive=str(OUT),files=len(values),original_files=len(files),result_sha256=sha(files['result.json']),sums_sha256=sha(values['SHA256SUMS'])),sort_keys=True))
if __name__=='__main__':main()
