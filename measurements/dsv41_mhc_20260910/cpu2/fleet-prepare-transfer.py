import subprocess,shlex,hashlib,json
from pathlib import Path
source='/home/choiceoh/stkernel-dsv41-megakernel-20260910'
revision='2c4b37b615a089cfd93f97e6423d52dca3846f7d'
stage='/tmp/dsv41-mhc-cpu2-source-shallow'; archive='/tmp/dsv41-mhc-cpu2-source-git.tar.gz'
assert not Path(stage).exists() and not Path(archive).exists()
subprocess.run(['git','clone','--quiet','--no-checkout','--depth','1','file://'+source,stage],check=True)
assert subprocess.check_output(['git','-C',stage,'rev-parse','HEAD'],text=True).strip()==revision
subprocess.run(['tar','-C',stage,'-czf',archive,'.git'],check=True)
digest=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
subprocess.run(['scp','-q',archive,'choiceoh@10.10.10.3:'+archive],check=True)
code="""import json,subprocess,socket,hashlib
from pathlib import Path
assert not Path(SOURCE).exists()
assert hashlib.sha256(Path(ARCHIVE).read_bytes()).hexdigest()==DIGEST
Path(SOURCE).mkdir()
subprocess.run(['tar','-C',SOURCE,'-xzf',ARCHIVE],check=True)
subprocess.run(['git','-C',SOURCE,'checkout','--quiet','--detach',REVISION],check=True)
assert subprocess.check_output(['git','-C',SOURCE,'rev-parse','HEAD'],text=True).strip()==REVISION
assert not subprocess.check_output(['git','-C',SOURCE,'status','--porcelain'],text=True).strip()
print(json.dumps(dict(host=socket.gethostname(),source=SOURCE,revision=REVISION,clean=True,shallow_git_archive_sha256=DIGEST)))
"""
code='\n'.join(f'{k}={v!r}' for k,v in dict(SOURCE=source,REVISION=revision,ARCHIVE=archive,DIGEST=digest).items())+'\n'+code
subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@10.10.10.3','python3 -B -c '+shlex.quote(code)],check=True)
