#!/usr/bin/env python3
"""Read-only collection of the already completed CPU13 receipt; never runs tests."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

OUT = Path(__file__).resolve().parent
REVISION = 'a010504ee03842f30892a31d764b74cdbf3a5610'
SOURCE = '/home/choiceoh/stkernel-ep-onepass-0909-13'
WORKER = 'choiceoh@10.10.10.1'
EVIDENCE = '/home/choiceoh/glm53-ep-cpu-0909-13/evidence'
JOB = '/tmp/glm53-ep-decode-cpu-0909-13'


def ssh(command):
    completed = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
         'choiceoh@srv2', shlex.join(command)], capture_output=True, timeout=50)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors='replace')[-2000:])
    return completed.stdout


WORKER_READ = r'''import hashlib,json,pathlib,subprocess,time,sys
root=pathlib.Path('/home/choiceoh/glm53-ep-cpu-0909-13/evidence')
source=pathlib.Path('/home/choiceoh/stkernel-ep-onepass-0909-13')
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
files={}
for p in sorted(root.rglob('*')):
 if p.is_symlink():raise RuntimeError('unexpected symlink: '+str(p))
 if p.is_file():
  raw=p.read_bytes();files[p.relative_to(root).as_posix()]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
sys.path.insert(0,str(source/'probes'))
from glm53_ep_bindings_capsule import validate_capsule
capsule=root.parent/'capsule'
manifest_sha='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
manifest=validate_capsule(capsule,manifest_sha)
image='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
image_actual=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()
assert image_actual==image
result=json.loads((root/'result.json').read_text())
contracts={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in result['contract_sources']}
mounted={}
for line in (source/'build/glm53/manifest.tsv').read_text().splitlines():
 if not line or line.startswith('#'):continue
 name,target,*_=line.split('\t')
 if target in result['mounted_sources']:mounted[target]=hashlib.sha256((source/'build/glm53'/name).read_bytes()).hexdigest()
assert contracts==result['contract_sources'] and mounted==result['mounted_sources']
print(json.dumps({'time':time.time(),'revision':git('rev-parse','HEAD'),'status':git('status','--porcelain'),
 'image_id':image_actual,'capsule':{'root':str(capsule),'manifest_sha256':manifest_sha,'files':len(manifest['files']),'strict_validation':'PASS'},
 'source_receipts':{'contract_sources':contracts,'mounted_sources':mounted},
 'shallow':git('rev-parse','--is-shallow-repository'),'alternates':(source/'.git/objects/info/alternates').exists(),
 'files':files}))
'''


def snapshot():
    code = r'''import base64,hashlib,json,pathlib,subprocess,sys,time,shlex
source=pathlib.Path('/home/choiceoh/stkernel-ep-onepass-0909-13')
job=pathlib.Path('/tmp/glm53-ep-decode-cpu-0909-13')
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
files={}
for name in ('submission.json','exit.json','fleet.log','driver.py','driver.pid'):
 raw=(job/name).read_bytes();files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'base64':base64.b64encode(raw).decode()}
worker=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','choiceoh@10.10.10.1',shlex.join(['python3','-B','-c',sys.argv[1]])],capture_output=True,text=True,timeout=30)
if worker.returncode:raise RuntimeError(worker.stderr[-1000:])
head_evidence={}
for p in sorted((job/'evidence').rglob('*')):
 if p.is_symlink():raise RuntimeError('unexpected symlink')
 if p.is_file():
  raw=p.read_bytes();head_evidence[p.relative_to(job/'evidence').as_posix()]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
worker_data=json.loads(worker.stdout)
assert head_evidence==worker_data['files']
print(json.dumps({'time':time.time(),'revision':git('rev-parse','HEAD'),'status':git('status','--porcelain'),
 'shallow':git('rev-parse','--is-shallow-repository'),'alternates':(source/'.git/objects/info/alternates').exists(),
 'head_evidence':head_evidence,'files':files,'worker':worker_data}))
'''
    value = json.loads(ssh(['python3', '-B', '-c', code, WORKER_READ]))
    for node in (value, value['worker']):
        assert node['revision'] == REVISION and not node['status']
        assert node['shallow'] == 'false' and node['alternates'] is False
    return value


def main():
    assert not (OUT/'capture.json').exists(), 'do not overwrite a completed capture'
    before = snapshot()
    (OUT/'head').mkdir(exist_ok=True)
    for name, value in before['files'].items():
        raw = base64.b64decode(value.pop('base64'), validate=True)
        assert len(raw) == value['size'] and hashlib.sha256(raw).hexdigest() == value['sha256']
        (OUT/'head'/name).write_bytes(raw)
    archive = ssh(['tar', '-czf', '-', '-C', JOB, 'evidence'])
    (OUT/'evidence.tar.gz').write_bytes(archive)
    extracted = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        for member in tar.getmembers():
            assert not member.issym() and not member.islnk()
            if member.isdir():
                continue
            assert member.isfile() and member.name.startswith('evidence/') and '..' not in Path(member.name).parts
            relative = member.name[len('evidence/'):]
            assert relative not in extracted
            raw = tar.extractfile(member).read()
            extracted[relative] = {'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
            if relative == 'result.json':
                assert (OUT/'result.json').read_bytes() == raw
    assert extracted == before['worker']['files']
    after = snapshot()
    for value in after['files'].values():
        value.pop('base64')
    assert before['files'] == after['files']
    assert before['worker']['files'] == after['worker']['files']
    capture = dict(schema=1, scope='read-only archive of completed CPU13',
                   source=SOURCE, revision=REVISION, head_job=JOB, worker=WORKER,
                   worker_evidence=EVIDENCE, before=before, after=after,
                   evidence_tar_sha256=hashlib.sha256(archive).hexdigest(),
                   original_files=len(extracted), original_bytes=sum(x['size'] for x in extracted.values()),
                   archive_bytes=len(archive))
    (OUT/'capture.json').write_text(json.dumps(capture, indent=2)+'\n')
    print(json.dumps({k:capture[k] for k in ('original_files','original_bytes','archive_bytes','evidence_tar_sha256')}))


if __name__ == '__main__':
    main()
