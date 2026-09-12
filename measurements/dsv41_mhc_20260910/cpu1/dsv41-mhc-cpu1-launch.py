import json,shlex,subprocess,time
from pathlib import Path
source='/home/choiceoh/stkernel-dsv41-megakernel-20260910'
revision='2c4b37b615a089cfd93f97e6423d52dca3846f7d'
output='/home/choiceoh/dsv41-mhc-cpu1-evidence'
session='dsv41mhccpu0910v1'
worker=['nice','-n19','python3','-B',source+'/measurements/dsv41_mhc_20260910/cpu_compile_runner.py','--source',source,'--revision',revision,'--output',output]
command=['env','REPO='+source,'bash',source+'/bench/fleet.sh','run','--cpu',session,'10','DSV4.1 H5120 MHC full-TU AOT; no GPU devices','--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@10.10.10.4',shlex.join(worker)]
metadata=Path('/tmp/dsv41-mhc-cpu1-execution.json'); log=Path('/tmp/dsv41-mhc-cpu1.log')
assert not metadata.exists() and not log.exists()
d=dict(source=source,revision=revision,output=output,session=session,command=command,started=time.time(),scope='Normal fleet CPU admission; AOT/export only, no GPU devices or workloads')
metadata.write_text(json.dumps(d,indent=2)+'\n')
remote="""import json,subprocess,sys,shlex
p=json.load(sys.stdin)
for host in (None,'choiceoh@10.10.10.4'):
 for args,expected in ((['rev-parse','HEAD'],p['revision']),(['status','--porcelain'],'')):
  c=['git','-C',p['source']]+args
  if host: c=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,shlex.join(c)]
  assert subprocess.check_output(c,text=True).strip()==expected
rc=subprocess.call(p['command'])
raise SystemExit(rc)
"""
with log.open('x') as f:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2','python3 -B -c '+shlex.quote(remote)],input=json.dumps(d),text=True,stdout=f,stderr=subprocess.STDOUT)
d.update(returncode=r.returncode,finished=time.time());metadata.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d));raise SystemExit(r.returncode)
