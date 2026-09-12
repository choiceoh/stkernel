"""Read-only all-rank log capture, pinned to one canonical bracket owner/release."""
import argparse,json,shlex,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--session',required=True);p.add_argument('--release',required=True)
p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
owner='queue/'+a.session
nodes=[None,'choiceoh@10.10.10.1','choiceoh@10.10.10.3','choiceoh@10.10.10.4']
def command(node,args):
    return args if node is None else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',node,shlex.join(args)]
started={};children=[];files=[];deadline=time.monotonic()+1800
while len(started)<4 and time.monotonic()<deadline:
    for rank,node in enumerate(nodes):
        if rank in started:continue
        r=subprocess.run(command(node,['docker','inspect','st-glm53']),capture_output=True,text=True,timeout=15)
        if r.returncode:continue
        d=json.loads(r.stdout)[0];env=dict(v.split('=',1) for v in d['Config']['Env'] if '=' in v)
        if env.get('ST_LEASE_OWNER')!=owner or env.get('ST_RELEASE')!=a.release:continue
        ident=d['Id']; row=dict(rank=rank,node=node or 'srv2',container_id=ident,started_at=d['State']['StartedAt'],
            image=d['Config']['Image'],image_id=d['Image'],owner=owner,release=a.release)
        m=subprocess.run(command(node,['docker','exec',ident,'cat','/opt/st/runtime-manifest.json']),capture_output=True,text=True,timeout=15)
        if m.returncode==0:row['runtime_manifest']=json.loads(m.stdout)
        started[rank]=row
        f=(out/('rank'+str(rank)+'.log')).open('w');files.append(f)
        children.append(subprocess.Popen(command(node,['docker','logs','--timestamps','--follow',ident]),stdout=f,stderr=subprocess.STDOUT))
        (out/'identity.json').write_text(json.dumps(list(started.values()),indent=2)+'\n')
        print(json.dumps(dict(rank=rank,container_id=ident,release=a.release,log=str(out/('rank'+str(rank)+'.log')))),flush=True)
    if len(started)<4:time.sleep(4)
if len(started)!=4:raise SystemExit('did not observe all four owned containers')
for child in children:child.wait()
for f in files:f.close()
print(json.dumps(dict(completed=True,exit_codes=[c.returncode for c in children])),flush=True)
