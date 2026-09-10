import gzip
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import shlex
import subprocess
import sys
import tarfile

version=int(sys.argv[1])
root=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
dest=root/f'measurements/glm53_ep_local_20260908/cpu{version}'
remote=r'''import gzip,hashlib,io,json,pathlib,subprocess,sys,tarfile,time
v=int(sys.argv[1]); root=pathlib.Path(f'/tmp/glm53-ep-local-compile0908-{v}-head'); source=pathlib.Path(f'/home/choiceoh/stkernel-ep-local-0908-{v}')
receipt=json.loads((root/'exit.json').read_text())
assert receipt['complete'] and receipt['returncode']==0
assert not subprocess.check_output(['git','--no-optional-locks','-C',str(source),'status','--porcelain'],text=True).strip()
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert revision==receipt['revision']
sys.path.insert(0,str(source/'probes'))
from glm53_ep_local_evidence import validate_compile_evidence
proof=validate_compile_evidence(source,root/'local/result.json')
files={}
for f in sorted(root.rglob('*')):
 if f.is_file():
  assert not f.is_symlink()
  files[str(f.relative_to(root))]=f.read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),source=str(root/n)) for n,b in files.items()}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==m['sha256'] for n,m in manifest.items())
identity=dict(source=str(source),revision=revision,clean=True,captured_epoch=time.time(),compile_receipt_validated=True,tests=proof['contracts']['tests_run'],mounted_files=len(proof['mounted_sources']),contract_files=len(proof['contracts']['files']),original_files_rechecked=len(files))
files['source-identity.json']=(json.dumps(identity,indent=2)+'\n').encode()
files['source-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,data in files.items():
  if name.endswith(('.ptx','.cubin')):name+='.gz';data=gzip.compress(data,mtime=0)
  item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
'''
payload=subprocess.check_output(['ssh','choiceoh@srv2',shlex.join(['python3','-c',remote,str(version)])])
dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
 for entry in archive:
  name=PurePosixPath(entry.name)
  assert entry.isfile() and not name.is_absolute() and '..' not in name.parts
  p=dest.joinpath(*name.parts);assert not p.exists(), 'refuse overwriting prior evidence: '+str(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.extractfile(entry).read())
manifest=json.loads((dest/'source-manifest.json').read_text())
for name,item in manifest.items():
 data=gzip.decompress((dest/(name+'.gz')).read_bytes()) if name.endswith(('.ptx','.cubin')) else (dest/name).read_bytes()
 assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
print(json.dumps(dict(destination=str(dest),verified_original_files=len(manifest),identity=json.loads((dest/'source-identity.json').read_text()))))
