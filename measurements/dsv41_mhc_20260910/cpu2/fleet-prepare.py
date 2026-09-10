import subprocess,shlex
code="""import json,subprocess,socket
from pathlib import Path
source='/home/choiceoh/stkernel-dsv41-megakernel-20260910'
revision='2c4b37b615a089cfd93f97e6423d52dca3846f7d'
assert not Path(source).exists()
subprocess.run(['git','clone','--quiet','--no-checkout','--depth','1','ssh://choiceoh@10.10.10.2'+source,source],check=True)
subprocess.run(['git','-C',source,'checkout','--quiet','--detach',revision],check=True)
assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()==revision
assert not subprocess.check_output(['git','-C',source,'status','--porcelain'],text=True).strip()
print(json.dumps(dict(host=socket.gethostname(),source=source,revision=revision,clean=True,origin='head frozen source shallow clone')))
"""
subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@10.10.10.3','python3 -B -c '+shlex.quote(code)],check=True)
