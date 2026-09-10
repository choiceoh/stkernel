import hashlib,json,subprocess,shlex
from pathlib import Path
source='/home/choiceoh/stkernel-dsv41-megakernel-20260910'
base='/home/choiceoh/stkernel-ep-tiled-0909-ring'
revision='2c4b37b615a089cfd93f97e6423d52dca3846f7d'
bundle='/tmp/dsv41-mhc-cpu1-source.bundle'
expected='8da539298874276d3e389d1db3b14b6a09236c15dc2d0b44dc95f8658a77f868'
assert hashlib.sha256(Path(bundle).read_bytes()).hexdigest()==expected
subprocess.run(['scp','-q',bundle,'choiceoh@10.10.10.4:'+bundle],check=True)
code="""import hashlib,json,subprocess,socket
from pathlib import Path
s=Path(SOURCE); b=Path(BUNDLE)
assert not s.exists(), 'refuse existing checkout'
assert hashlib.sha256(b.read_bytes()).hexdigest()==EXPECTED
subprocess.run(['git','-C',BASE,'merge-base','--is-ancestor','313e36c2ad915edbbd7823fdc4ec9b2a1af62e08','HEAD'],check=True)
subprocess.run(['git','clone','--quiet','--no-checkout','--shared',BASE,str(s)],check=True)
subprocess.run(['git','-C',str(s),'fetch','--quiet',str(b),'HEAD'],check=True)
subprocess.run(['git','-C',str(s),'checkout','--quiet','--detach',REVISION],check=True)
assert subprocess.check_output(['git','-C',str(s),'rev-parse','HEAD'],text=True).strip()==REVISION
assert not subprocess.check_output(['git','-C',str(s),'status','--porcelain'],text=True).strip()
print(json.dumps(dict(host=socket.gethostname(),source=str(s),revision=REVISION,clean=True,bundle_sha256=EXPECTED)))
"""
code='\n'.join(f'{k}={v!r}' for k,v in dict(SOURCE=source,BASE=base,REVISION=revision,BUNDLE=bundle,EXPECTED=expected).items())+'\n'+code
for host in (None,'choiceoh@10.10.10.4'):
 command=['python3','-B','-c',code]
 if host: command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,shlex.join(command)]
 subprocess.run(command,check=True)
