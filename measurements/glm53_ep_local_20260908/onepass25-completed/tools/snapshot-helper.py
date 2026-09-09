#!/usr/bin/env python3
"""Private, immutable live-arm evidence; inspect/log reads only, no requests."""
import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REV='055914aeb719c1769e05cdb863e43a88b2ee47af'
TICKET='1788930114553638'
OWNER_PID='553638'

IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
JOB = '/tmp/glm53-ep-onepass-0909-25'
SOURCE = '/home/choiceoh/stkernel-ep-onepass-0909-25'
NODE = r'''
def exact_mm_limit(argv):
    values=[]
    for index,arg in enumerate(argv):
        key,equal,value=arg.partition('=')
        if key!='--limit-mm-per-prompt': continue
        if not equal:
            if index+1>=len(argv): raise RuntimeError('missing multimodal limit value')
            value=argv[index+1]
        values.append(json.loads(value))
    if len(values)!=1 or not isinstance(values[0],dict):
        raise RuntimeError('missing or duplicate multimodal limit')
    value=values[0]
    if set(value)!={'image','video'} or any(type(v) is not int or v<0 for v in value.values()):
        raise RuntimeError('multimodal limits must be exact nonnegative integer image/video fields')
    return value
import math
def valid_startup_trim(receipt,rank,created):
    try:
        if (type(receipt.get('schema')) is not int or receipt['schema']!=1
                or receipt.get('verdict')!='COMPLETE' or type(receipt.get('rank')) is not int
                or receipt['rank']!=rank or receipt.get('measurement_errors')!=[]
                or any(k in receipt for k in ('error','failed_stage'))): return False
        stages=receipt['stages']
        if ([x['stage'] for x in stages]!=['synchronize','gc_collect','empty_cache','malloc_trim']
                or any(x['status']!='COMPLETE' or 'error' in x for x in stages)
                or type(stages[-1]['returned']) is not int or stages[-1]['returned'] not in (0,1)
                or type(stages[1]['collected']) is not int or stages[1]['collected']<0): return False
        for phase in ('before','after'):
            if set(receipt[phase])!={'allocated','reserved','mem_available','vm_rss'}: return False
            if any(type(v) is not int or v<0 for v in receipt[phase].values()): return False
        started,finished=receipt['started_at'],receipt['completed_at']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (started,finished)): return False
        return created<=started<=finished<=time.time()
    except (KeyError,TypeError,ValueError,AttributeError): return False
import base64,datetime,gzip,hashlib,json,os,re,stat,subprocess,tempfile,time
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,msg):
    if not ok: raise RuntimeError(msg)
def command(argv):
    p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    need(p.returncode==0,'read command failed')
    need(len(p.stdout)<=16*2**20,'command output too large')
    return p.stdout
def inspect(ref):
    raw=command(['docker','inspect',ref]); values=json.loads(raw)
    need(len(values)==1,'container inspect not unique')
    return values[0],raw
def bounded(path,maximum=32*2**20):
    path=Path(path); a=path.lstat()
    need(stat.S_ISREG(a.st_mode) and a.st_size<=maximum,'unsafe or oversized file')
    raw=path.read_bytes(); b=path.lstat()
    need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'file changed')
    return raw
def active(c):
    need(c['State']['Running'] is True and not c['State'].get('Paused') and not c['State'].get('Restarting'),'container not actively running')
    need(c['Image']==cfg['image'],'wrong pinned image')
def fixed(c):
    # Docker returns Mounts in varying order. Compare every original mount
    # field and duplicate exactly, with only this unordered list canonicalized.
    return {k:c[k] for k in ('Id','Created','Image','Config','HostConfig','RestartCount')} | {'Mounts':sorted(c['Mounts'],key=lambda m:json.dumps(m,sort_keys=True)), 'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
def source_evidence(c):
    manifest=bounded('/home/choiceoh/overlays/glm53/manifest.tsv')
    need(sha(manifest)==cfg['manifest_sha256'],'deployed source manifest differs')
    found={}
    for mount in c['Mounts']:
        dest=mount['Destination']
        if dest in cfg['mounts']:
            need(dest not in found and mount['Type']=='bind' and mount['RW'] is False,'ambiguous or writable overlay mount')
            need(mount['Source'].startswith('/home/choiceoh/overlays/glm53/'),'unexpected overlay source')
            found[dest]=sha(bounded(mount['Source']))
    need(found==cfg['mounts'],'mounted overlay source differs')
    return {'manifest_sha256':sha(manifest),'mounts':found}
started=time.time(); c,before=inspect(cfg['name']); active(c)
need(c['Name']=='/'+cfg['name'] and len(c['Id'])==64,'wrong container name/ID')
container_born=datetime.datetime.fromisoformat(c['State']['StartedAt'].replace('Z','+00:00')).timestamp()
container_created=datetime.datetime.fromisoformat(c['Created'].replace('Z','+00:00')).timestamp()
need(container_born>=container_created>=cfg['arm_event']['started_at'],'container creation/start predates or contradicts the selected canonical arm')
if cfg['rank']==0:
    need(c['Id']==cfg['head_id'] and c['State']['StartedAt']==cfg['head_started_at'],'observed head identity changed')
namespace={'__name__':'snapshot_launch_parser'}
exec(compile(base64.b64decode(cfg['parser']),'<frozen-launch-parser>','exec'),namespace)
topology=namespace['launch_parallelism'](c['Config']['Cmd'])
need(topology['enabled']==cfg['ep'] and topology['tensor_parallel_size']==4 and topology['nnodes']==4 and topology['node_rank']==cfg['rank'],'wrong EP/topology')
payload=namespace['_WRAPPER'].fullmatch(c['Config']['Cmd'][1])[1]
script=base64.b64decode(payload,validate=True).decode()
line=script[len(namespace['_GID_PRELUDE']):].removesuffix('\n')
argv=namespace['_literal_argv'](line[:-len(namespace['_REDIRECTION'])])
mm_limit=exact_mm_limit(argv)
need(mm_limit==cfg['mm_limit'],'actual multimodal image/video limits differ')
env={}
for item in c['Config']['Env']:
    key,value=item.split('=',1); need(key and key not in env,'duplicate environment key'); env[key]=value
for key,value in cfg['flags'].items(): need(env.get(key)==value,'unexpected required arm flag: '+key)
endpoint={}
for i,arg in enumerate(argv):
    key,equal,value=arg.partition('=')
    if key in ('--host','--port'):
        need(key not in endpoint,'duplicate endpoint option')
        endpoint[key]=value if equal else argv[i+1]
if cfg['rank']==0: need(endpoint=={'--host':'127.0.0.1','--port':'18000'},'wrong private head endpoint')
source=source_evidence(c)
mounts=[m for m in c['Mounts'] if m['Destination']=='/glmlogs']
need(len(mounts)==1 and mounts[0]['Type']=='bind','serving log bind missing')
path=Path(mounts[0]['Source'])/'glm53.log'; a=path.lstat()
boot=datetime.datetime.fromisoformat(c['State']['StartedAt'].replace('Z','+00:00')).timestamp()
need(stat.S_ISREG(a.st_mode) and 0<a.st_size<=128*2**20 and a.st_mtime>=boot,'unsafe, stale or oversized serving log')
with path.open('rb') as stream: log=stream.read(a.st_size)
b=path.lstat()
need(len(log)==a.st_size and (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and b.st_size>=a.st_size,'serving log rotated or truncated')
graph=bool(re.search(rb'Graph capturing finished in [0-9]+ secs, took ',log))
need(graph,'graph capture is not complete on this node')
marker=b'[tp-sf6-q0-selftest] PASS '; pass_records=[]
for raw_line in log.splitlines():
    if marker not in raw_line: continue
    receipt,_=json.JSONDecoder().raw_decode(raw_line.split(marker,1)[1].decode())
    need(isinstance(receipt,dict) and receipt.get('verdict')=='PASS' and receipt.get('phase')=='complete','invalid TP SF6 Q0 PASS receipt')
    pass_records.append({'line_sha256':sha(raw_line),'receipt_sha256':sha(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())})
if cfg['candidate']:
    need(pass_records and b'[tp-sf6-q0-selftest] FAIL ' not in log,'candidate TP SF6 Q0 selftest is absent or failed')
trim_marker=b'[glm53-startup-trim] '; trim_records=[]
for raw_line in log.splitlines():
    if trim_marker not in raw_line:continue
    receipt,_=json.JSONDecoder().raw_decode(raw_line.split(trim_marker,1)[1].decode())
    need(valid_startup_trim(receipt,cfg['rank'],container_created),'startup trim is not a complete measured operation on this rank')
    trim_records.append({'line_sha256':sha(raw_line),'receipt':receipt})
need(len(trim_records)==1,'startup trim requires exactly one complete receipt per rank')

with tempfile.TemporaryFile() as stream:
    result=subprocess.run(['docker','logs','--timestamps','--since',c['State']['StartedAt'],c['Id']],stdout=stream,stderr=stream,timeout=30)
    need(result.returncode==0,'docker log read failed')
    size=stream.tell(); need(size<=128*2**20,'docker logs too large')
    stream.seek(0); docker_log=stream.read()
again,after=inspect(c['Id']); by_name,_=inspect(cfg['name']); active(again); active(by_name)
need(fixed(c)==fixed(again)==fixed(by_name),'container/start/config changed during capture')
need(source_evidence(again)==source,'source changed during capture')
files={'inspect.before.json':before,'inspect.after.json':after,'serving.log':log,'docker.log':docker_log,
       'manifest.tsv':bounded('/home/choiceoh/overlays/glm53/manifest.tsv')}
summary={'node':cfg['node'],'id':c['Id'],'created_at':c['Created'],'started_at':c['State']['StartedAt'],'image':c['Image'],
         'capture_started_at':started,'capture_finished_at':time.time(),'running_start_config_stable':True,
         'topology':topology,'flags':cfg['flags'],'endpoint':endpoint,'mm_limit':mm_limit,'source':source,
         'readiness':{'graph_finished':graph,'candidate':cfg['candidate'],'tp_sf6_q0_pass_records':pass_records,'startup_trim_records':trim_records},
         'environment_sha256':sha(json.dumps(env,sort_keys=True,separators=(',',':')).encode()),
         'log_source':{'path':str(path),'device':a.st_dev,'inode':a.st_ino,'captured_prefix_bytes':len(log),
                       'size_after':b.st_size,'mtime_ns_before':a.st_mtime_ns,'mtime_ns_after':b.st_mtime_ns},
         'scope':'live prefix snapshot; traffic may still be in progress; no quality/performance acceptance'}
print(json.dumps({'summary':summary,'files':{k:{'sha256':sha(v),'bytes':len(v),'gzip':base64.b64encode(gzip.compress(v,mtime=0)).decode()} for k,v in files.items()}}))
'''

REMOTE = r'''

def current_arm_event(session, ticket, owner_pid):
    import datetime, hashlib, json, pathlib, re, stat
    fleet=pathlib.Path('/home/choiceoh/glm53-logs/fleet')
    pending=json.loads((fleet/'pending'/(hashlib.sha256(session.encode()).hexdigest()+'.json')).read_text())
    if pending.get('session')!=session or str(pending.get('ticket'))!=ticket or str(pending.get('pid'))!=str(owner_pid):
        raise RuntimeError('arm reservation identity differs')
    fields=pathlib.Path('/proc/'+str(owner_pid)+'/stat').read_bytes().rsplit(b')',1)[1].split()
    if fields[0]==b'Z' or fields[19].decode('ascii')!=str(pending.get('start')):
        raise RuntimeError('arm supervisor start differs or exited')
    if pending.get('payload_returncode') is not None or pending.get('returncode') is not None:
        raise RuntimeError('arm payload is already terminal')
    holder=(fleet/'holder').read_text().strip().split('|')
    if len(holder)<2 or holder[:2]!=[session,str(owner_pid)]:
        raise RuntimeError('arm is not held by this supervisor')
    path=fleet/'run-logs'/(hashlib.sha256((session+'\0'+ticket).encode()).hexdigest()+'.log')
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size>16*2**20:
        raise RuntimeError('unsafe or oversized canonical run log')
    with path.open('rb') as stream: raw=stream.read(before.st_size)
    after=path.lstat()
    if len(raw)!=before.st_size or (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino) or after.st_size<before.st_size:
        raise RuntimeError('canonical run log changed identity')
    if raw and not raw.endswith(b'\n'): raw=raw.rsplit(b'\n',1)[0]+b'\n' if b'\n' in raw else b''
    events=list(re.finditer(rb'(?m)^== ([0-9]{2}:[0-9]{2}:[0-9]{2}) arm ([^ :\r\n]+):[^\r\n]*$',raw))
    expected=['EPONEPASS25'+arm for arm in ('B0','B1','A','B2','B3')]
    names=[e[2].decode() for e in events]
    if not names or names!=expected[:len(names)]:
        raise RuntimeError('canonical arm order is absent or differs')
    event=events[-1]; hour,minute,second=map(int,event[1].split(b':'))
    now=datetime.datetime.now().astimezone()
    born=now.replace(hour=hour,minute=minute,second=second,microsecond=0)
    if born>now: born-=datetime.timedelta(days=1)
    if (now-born).total_seconds()>3*3600:
        raise RuntimeError('arm event is stale or clock differs')
    marker={'path':str(path),'device':before.st_dev,'inode':before.st_ino,'offset':event.start(),'line':event[0].decode()}
    return dict(arm=names[-1].removeprefix('EPONEPASS25'),started_at=born.timestamp(),
                started_at_local=born.isoformat(),key=hashlib.sha256(json.dumps(marker,sort_keys=True).encode()).hexdigest(),**marker)
import base64,gzip,hashlib,json,os,stat,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.umask(0o077)
sha=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,msg):
    if not ok: raise RuntimeError(msg)
def run(argv):
    p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    need(p.returncode==0,'source read command failed'); return p.stdout
def holder():
    raw=Path('/home/choiceoh/glm53-logs/fleet/holder').read_bytes()
    need(raw.decode().strip().split('|')[0]==cfg['session'],'different active fleet holder')
    return raw
def source_check():
    need(run(['git','-C',cfg['source'],'rev-parse','HEAD']).decode().strip()==cfg['revision'],'frozen HEAD differs')
    need(not run(['git','-C',cfg['source'],'status','--porcelain','--untracked-files=no']).strip(),'frozen tracked source dirty')
def write(path,data):
    with path.open('xb') as stream: stream.write(data)
    path.chmod(0o600)
before_holder=holder(); source_check()
arm_event=current_arm_event(cfg['session'],cfg['ticket'],cfg['owner_pid'])
need(arm_event['arm']==cfg['arm'] and arm_event['key']==cfg['arm_event_key'],'selected canonical arm changed')
cfg['arm_event']=arm_event
job=Path(cfg['job']); need(job.is_dir() and not job.is_symlink(),'job directory absent/unsafe')
dest=job/('live-'+cfg['label']); dest.mkdir(mode=0o700,exist_ok=False)
source=Path(cfg['source']); manifest=(source/'build/glm53/manifest.tsv').read_bytes()
expected=b'# source_commit='+cfg['revision'].encode()+b'\n'+manifest
mounts={}
for line in manifest.decode().splitlines():
    if not line or line.startswith('#'): continue
    name,target,group=line.split('\t')
    need('/' not in name and name not in ('.','..') and target not in mounts,'unsafe/duplicate manifest entry')
    mounts[target]=sha((source/'build/glm53'/name).read_bytes())
parser=(source/'bench/glm53_launch_metadata.py').read_bytes()
need(parser==run(['git','-C',str(source),'show',cfg['revision']+':bench/glm53_launch_metadata.py']),'launch parser differs')
common=dict(cfg,parser=base64.b64encode(parser).decode(),manifest_sha256=sha(expected),mounts=mounts)
nodes=('local','10.10.10.1','10.10.10.3','10.10.10.4')
def capture(pair):
    rank,node=pair
    data=dict(common,node=node,rank=rank,name='glm53' if rank==0 else 'glm53-worker')
    code='cfg='+repr(data)+'\n'+node_code
    cmd=['python3','-B','-'] if rank==0 else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','choiceoh@'+node,'python3 -B -']
    result=subprocess.run(cmd,input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=150)
    if result.returncode:
        write(dest/(node+'.failure.private.log'),result.stderr)
        raise RuntimeError('node snapshot failed: '+node)
    obj=json.loads(result.stdout); entries={}
    for name,item in obj['files'].items():
        need(name in ('inspect.before.json','inspect.after.json','serving.log','docker.log','manifest.tsv'),'unexpected capture path')
        packed=base64.b64decode(item['gzip'],validate=True); raw=gzip.decompress(packed)
        need(len(raw)==item['bytes'] and sha(raw)==item['sha256'],'capture transport hash differs')
        relative=node+'.'+name+'.gz'; write(dest/relative,packed)
        entries[relative]={'original_sha256':sha(raw),'original_bytes':len(raw),'stored_sha256':sha(packed),'stored_bytes':len(packed)}
    return node,obj['summary'],entries
with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(capture,enumerate(nodes)))
after_holder=holder(); need(before_holder==after_holder,'holder changed during capture'); source_check()
need(current_arm_event(cfg['session'],cfg['ticket'],cfg['owner_pid'])['key']==cfg['arm_event_key'],'canonical arm changed during capture')
write(dest/'holder.before.private',before_holder); write(dest/'holder.after.private',after_holder)
write(dest/'launch-parser.py',parser)
identity={'schema':1,'arm':cfg['arm'],'label':cfg['label'],'revision':cfg['revision'],'source':cfg['source'],
          'session':cfg['session'],'ticket':cfg['ticket'],'owner_pid':cfg['owner_pid'],
          'arm_event':arm_event,'captured_at':time.time(),'parser_sha256':sha(parser),
          'nodes':{n:s for n,s,_ in results},'files':{k:v for _,_,es in results for k,v in es.items()}}
write(dest/'identity.json',(json.dumps(identity,indent=2,sort_keys=True)+'\n').encode())
files={}
for path in sorted(dest.iterdir()):
    need(path.is_file() and not path.is_symlink(),'unsafe saved capture')
    raw=path.read_bytes(); files[path.name]={'sha256':sha(raw),'data':base64.b64encode(raw).decode()}
print(json.dumps({'identity':identity,'remote_directory':str(dest),'files':files}))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('B0','B1','A','B2','B3'),required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--suffix',default='')
    p.add_argument('--head-id',required=True)
    p.add_argument('--head-started-at',required=True)
    p.add_argument('--arm-event-key',required=True)
    p.add_argument('--syntax-check',action='store_true')
    args=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',args.revision): p.error('full revision required')
    if not re.fullmatch('[a-f0-9]{64}',args.head_id) or not re.fullmatch('[a-f0-9]{64}',args.arm_event_key): p.error('exact head and arm event identities required')
    if args.session != 'eplocalonepass0909v25': p.error('wrong onepass25 session')
    if args.suffix and not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,39}',args.suffix): p.error('unsafe suffix')
    label=args.arm+('-'+args.suffix if args.suffix else '')
    config=dict(arm=args.arm,label=label,revision=args.revision,session=args.session,job=JOB,source=SOURCE,
                image=IMAGE,ticket=TICKET,owner_pid=OWNER_PID,head_id=args.head_id,
                head_started_at=args.head_started_at,arm_event_key=args.arm_event_key,
                ep=False,candidate=args.arm == 'A',mm_limit={'image':4,'video':0},flags={
                    'VLLM_GLM53_EP_PREFILL_LOCAL':'0',
                    'VLLM_B12X_EP_WARM_COMPACT':'0',
                    'VLLM_B12X_EP_ZERO_WEIGHT_MICRO':'0',
                    'VLLM_GLM53_TP_SF6_Q0':'1' if args.arm == 'A' else '0',
                    'VLLM_GLM53_STARTUP_TRIM':'1',
                    'VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE':'1'})
    code='cfg='+repr(config)+'\nnode_code='+repr(NODE)+'\n'+REMOTE
    ast.parse(NODE); ast.parse(code)
    if args.syntax_check:
        print(json.dumps({'syntax':'PASS','arm':args.arm,'label':label,'remote_execution':False})); return
    if args.revision!=REV or not re.fullmatch('[a-f0-9]{40}',REV) or not TICKET.isdecimal() or not OWNER_PID.isdecimal() or int(OWNER_PID)<=0:
        p.error('bind actual onepass25 revision, ticket and owner PID before capture')
    os.umask(0o077)
    local=Path('/tmp')/('glm53-onepass25-live-'+label)
    local.mkdir(mode=0o700,exist_ok=False)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','choiceoh@srv2','python3 -B -'],
                          input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=210)
    if result.returncode:
        (local/'failure.private.log').write_bytes(result.stderr)
        (local/'failure.private.log').chmod(0o600)
        raise RuntimeError('snapshot rejected; private failure log: '+str(local/'failure.private.log'))
    obj=json.loads(result.stdout)
    for name,item in obj['files'].items():
        if Path(name).name!=name or name in ('.','..'): raise RuntimeError('unsafe received path')
        raw=base64.b64decode(item['data'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=item['sha256']: raise RuntimeError('received hash differs')
        target=local/name
        with target.open('xb') as stream: stream.write(raw)
        target.chmod(0o600)
    print(json.dumps({'arm':args.arm,'revision':args.revision,'remote_directory':obj['remote_directory'],
                      'local_directory':str(local),'identity_sha256':hashlib.sha256((local/'identity.json').read_bytes()).hexdigest(),
                      'nodes':{n:{k:s[k] for k in ('id','started_at','image','running_start_config_stable')} for n,s in obj['identity']['nodes'].items()}}))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1)
