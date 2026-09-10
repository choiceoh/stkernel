#!/usr/bin/env python3
"""Prepare terminal GPU27 evidence. Running this later performs read-only captures
and writes one fresh archive; it never submits work or signals any process.
Only B/A are valid, and only A enables TP Q0. CPU24 is reused verbatim.
"""
import argparse, base64, datetime, gzip, hashlib, importlib.util, json, math, os, re, subprocess
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass27-completed'
REV='ea413ac4c39ba3e6e4009c73587b0d536053b4bf'; SESSION='eplocalonepass0909v27'; TICKET='1788932740730747'; PID=730747; START='43512869'
CPU_REV='82ac3c34173ae63b3dd0a42c49f8421097e96a1a'
CPU_SHA='1d96612938f200bd86d078e1d8841cc92df808ba4ab802ca51306fe229b67506'
REUSE_SHA='d04ae3cb91a3b49b952d90a2eb5c069169437be989c8093c2b67d417abf944e3'
VALIDATOR_SHA='7e0e31a67f49ef20fca3586e15c8370192d2648c7fbd1ad1fc207d3eb5002cb8'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
ARMS=('B','A'); NODES=('local','10.10.10.1','10.10.10.3','10.10.10.4')
items={}; originals={}; missing=[]
def sha(raw): return hashlib.sha256(raw).hexdigest()
def need(ok,why):
 if not ok: raise ValueError(why)
def read(path):
 path=Path(path); need(path.is_file() and not path.is_symlink(),'unsafe file '+str(path))
 a=path.stat(); need(a.st_size<128*2**20,'oversized file '+str(path)); raw=path.read_bytes(); b=path.stat()
 need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==a.st_size,'changed file '+str(path)); return raw
def save(name,raw,origin,**metadata):
 need(name not in items and not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe archive name')
 items[name]=raw; originals[name]=dict(origin=origin,bytes=len(raw),sha256=sha(raw),**metadata)
def packed(name,raw,origin): save(name,gzip.compress(raw,mtime=0),origin,original_bytes=len(raw),original_sha256=sha(raw))
def record(name,value,origin): save(name,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode(),origin)
def gitfile(name,revision=REV): return subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
def date(value): return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def jsonlines(raw): return [json.loads(line) for line in raw.splitlines() if line.strip()]

REMOTE=r'''
import base64,hashlib,json,os,pathlib,re,subprocess,time
P=pathlib.Path; root=P('/home/choiceoh/stkernel-ep-onepass-0909-27'); job=P('/tmp/glm53-ep-onepass-0909-27'); F=P('/home/choiceoh/glm53-logs/fleet')
session=cfg['session']; ticket=cfg['ticket']; owner=cfg['pid']; log=F/'run-logs'/(hashlib.sha256((session+'\0'+ticket).encode()).hexdigest()+'.log')
def need(ok,why):
 if not ok: raise ValueError(why)
def plain(p):
 need(p.is_file() and not p.is_symlink(),'unsafe file '+str(p)); a=p.stat(); need(a.st_size<128*2**20,'file too large'); raw=p.read_bytes(); b=p.stat()
 need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==a.st_size,'file changed'); return raw,b

def state():
 def git(*args): return subprocess.check_output(['git','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'},timeout=10).decode().strip()
 p=json.loads(plain(F/'pending'/(hashlib.sha256(session.encode()).hexdigest()+'.json'))[0])
 need(p['session']==session and str(p['ticket'])==ticket and p['pid']==owner and str(p['start'])==cfg['start'] and p['repo']==str(root) and p['cwd']==str(root),'reservation binding differs')
 f=json.loads(subprocess.check_output(['bash',str(root/'bench/fleet.sh'),'show',session,'--ticket',ticket,'--json'],timeout=20))
 keys=('session','ticket','state','phase','started_at','payload_finished_at','finished_at','payload_returncode','returncode','outcome','log_path','recovery_policy','recovery_deferred','supervisor_alive','payload_seconds')
 f={k:f[k] for k in keys if k in f}
 holder=(F/'holder').read_text().split('|') if (F/'holder').exists() else []
 own=bool(holder and holder[0]==session)
 need(f['session']==session and str(f['ticket'])==ticket and f['log_path']==str(log),'fleet identity differs')
 need(f['phase']=='finished' and type(f.get('payload_returncode')) is int and type(f.get('returncode')) is int and f['supervisor_alive'] is False and not own,'not terminal/released')
 try:
  fields=(P('/proc')/str(owner)/'stat').read_bytes().rsplit(b')',1)[1].split(); same=fields[0]!=b'Z' and fields[19].decode()==cfg['start']
 except FileNotFoundError: same=False
 need(not same,'bound supervisor still alive')
 source=dict(head=git('rev-parse','HEAD'),status=git('status','--porcelain')); need(source==dict(head=cfg['revision'],status=''),'frozen source changed')
 return dict(source=source,fleet=f,own_holder=own,bound_supervisor_alive=same,reservation={k:p[k] for k in ('session','ticket','pid','start','repo','cwd')})
r=dict(schema=1,captured_at=time.time(),before=state(),files={},absent=[],leg_origins={}); total=0

def capture(p):
 global total
 key=str(p)
 if key in r['files'] or key in r['absent']: return
 if not p.exists(): r['absent'].append(key); return
 raw,meta=plain(p); total+=len(raw); need(total<512*2**20,'capture total exceeds bound')
 r['files'][key]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),mtime_ns=meta.st_mtime_ns,data=base64.b64encode(raw).decode())

for name in ('submission.json','submit.exit.json','submit.stdout','submit.stderr','cpu24-reuse.json','onepass.jsonl','verdicts.jsonl','cancel-request.json'): capture(job/name)
capture(log)
for arm in cfg['arms']: capture(P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS27'+arm+'.log'))
for name in ('pre-signal.json','result.json','leg.raw'): capture(job/'failfast-first-fixed-A'/name)
lograw=plain(log)[0]
for value in cfg['legs']:
 need(re.fullmatch(r'/tmp/leg\.[0-9]+',value),'unsafe explicit leg'); r['leg_origins'][value]=['explicit parent-supplied path; no reconstructed lineage']
for value in re.findall(rb'/tmp/leg\.[0-9]+',lograw): r['leg_origins'].setdefault(value.decode(),[]).append('literal path in terminal fleet log')
for value in r['leg_origins']: capture(P(value))
r['after']=state(); need(r['before']==r['after'],'terminal/source state changed during capture')
print(json.dumps(r))
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--leg',action='append',default=[],help='Known exact /tmp/leg.<pid>; missing originals stay missing')
 args=parser.parse_args(); need(all(re.fullmatch(r'/tmp/leg\.[0-9]+',v) for v in args.leg),'unsafe leg path'); need(not OUT.exists(),'refuse existing archive')
 # Do not fetch or write an archive while the passive observer is still open.
 stream_root=Path('/tmp/glm53-onepass27-streams'); eraw=read(stream_root/'events.jsonl'); events=jsonlines(eraw)
 need(events and events[-1]['kind']=='observer_finished' and events[-1]['owned_go_seen'] is True,'observer is not closed')
 attempted=events[-1]['attempted_arms']; need(len(attempted)==len(set(attempted)) and set(attempted)<=set(ARMS),'invalid attempted arms')
 start_event=next(e for e in events if e['kind']=='observer_start')
 need((start_event['session'],str(start_event['ticket']),start_event['revision'])==(SESSION,TICKET,REV),'observer source binding differs')
 need(str(start_event['owner_start_tick'])==START,'observer supervisor start differs')
 observer_pid=int(read(stream_root/'observer.pid')); need(start_event['pid']==observer_pid,'observer PID differs')
 try: os.kill(observer_pid,0)
 except ProcessLookupError: pass
 else: raise ValueError('observer PID still exists; do not stop it from this collector')
 cfg=dict(session=SESSION,ticket=TICKET,pid=PID,start=START,revision=REV,arms=ARMS,legs=args.leg)
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','choiceoh@srv2','python3','-B','-'],input=('cfg='+repr(cfg)+'\n'+REMOTE).encode(),capture_output=True,timeout=60)
 need(p.returncode==0,p.stderr.decode(errors='replace')[:2000]); remote=json.loads(p.stdout)
 for origin,d in remote['files'].items():
  raw=base64.b64decode(d.pop('data'),validate=True); need(sha(raw)==d['sha256'] and len(raw)==d['bytes'],'transport bytes differ')
  name=Path(origin).name
  if name.startswith('boot-'): packed('boot/'+name+'.gz',raw,origin)
  elif origin==remote['after']['fleet']['log_path']: packed('fleet/terminal-run.log.gz',raw,origin)
  elif re.fullmatch(r'leg\.[0-9]+',name): packed('legs/'+name+'.raw.gz',raw,origin)
  elif '/failfast-first-fixed-A/' in origin:
   if name=='leg.raw': packed('failure/failfast-first-fixed-A/leg.raw.gz',raw,origin)
   else: save('failure/failfast-first-fixed-A/'+name,raw,origin)
  else: save('job/'+name,raw,origin)
 record('terminal-capture.json',remote,'read-only terminal fleet/source/reservation/process/owned-holder capture')
 for required in ('job/submission.json','job/submit.exit.json','job/cpu24-reuse.json','fleet/terminal-run.log.gz'): need(required in items,'missing required receipt '+required)
 need(sha(items['job/cpu24-reuse.json'])==REUSE_SHA,'remote original CPU24 reuse receipt changed')
 reuse_path=Path('/tmp/glm53-onepass27-cpu24-reuse.json'); need(read(reuse_path)==items['job/cpu24-reuse.json'],'local/remote CPU24 reuse receipt differs')
 terminal_log=gzip.decompress(items['fleet/terminal-run.log.gz'])
 begun=[v.decode().removeprefix('EPONEPASS27') for v in re.findall(rb'(?m)^== [0-9]{2}:[0-9]{2}:[0-9]{2} arm (EPONEPASS27[^ :\r\n]+):',terminal_log)]
 need(begun==list(ARMS[:len(begun)]),'canonical arm order differs')
 rows=jsonlines(items.get('job/onepass.jsonl',b'')); names=[r['name'].removeprefix('EPONEPASS27') for r in rows]
 need(names==list(ARMS[:len(names)]) and all(r['name']=='EPONEPASS27'+a for a,r in zip(names,rows)) and begun[:len(names)]==names,'completed records are not a canonical prefix')
 verdicts=jsonlines(items.get('job/verdicts.jsonl',b''))
 record('legs/availability.json',dict(known_paths=[dict(path=p,available=p in remote['files'],origins=o) for p,o in remote['leg_origins'].items()],scope='Only exact parent-supplied or terminal-log paths were read. Cleaned originals stay absent; other unknown leg paths are not enumerated or reconstructed.'),'terminal path evidence plus explicit --leg arguments')
 # Preserve exact source objects; the original CPU24 revision/result is never relabeled.
 manifest_raw=gitfile('build/glm53/manifest.tsv'); manifest=('# source_commit='+REV+'\n').encode()+manifest_raw
 mounts={line.split('\t')[1]:sha(gitfile('build/glm53/'+line.split('\t')[0])) for line in manifest_raw.decode().splitlines() if line and not line.startswith('#')}
 save('source/git-manifest.tsv',manifest_raw,'git '+REV); save('source/frozen-manifest.tsv',manifest,'normal source_commit header + git '+REV); record('source/mounted-hashes.json',mounts,'frozen build '+REV)
 for path in ('bench/onepass.py','profiles/glm53.env'):
  packed('source/'+Path(path).name+'.gz',gitfile(path),'git '+REV+':'+path)
 for name in ('glm53_tp_sf6_q0_selftest.py','glm53_ep_local_selftest.py','moe_dynamic_gated_sf6_q0.py','moe_dynamic_gated_sf6.py','moe_dispatch.py','flashinfer_b12x_moe.py','gpu_worker.py'):
  packed('source/'+name+'.gz',gitfile('build/glm53/'+name),'git '+REV+':build/glm53/'+name)
 helper=Path('/tmp/glm53_extract_onepass27_canary.py'); helper_raw=read(helper); need(sha(helper_raw)==VALIDATOR_SHA,'reviewed pure canary validator changed')
 spec=importlib.util.spec_from_file_location('onepass27_archive_validator',helper); validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
 cpu_path=ROOT/'measurements/glm53_ep_local_20260908/decode24-cpu/result.json'
 expected,cpu=validator.expected_source(SimpleNamespace(cpu_result=cpu_path,cpu_reuse_receipt=reuse_path,repo=ROOT,revision=REV))
 need(sha(read(cpu_path))==CPU_SHA,'original CPU24 result differs')
 reuse_verified=Path('/tmp/glm53-onepass27-cpu24-reuse-verified.json')
 if reuse_verified.exists():
  raw=read(reuse_verified); need(sha(raw)=='38fe5a7588817116e7b67402bf3fceec3ccb5e1e365175b9a4c9603c25864ada','independent CPU24 reuse proof changed')
  save('source/independent-cpu24-reuse-verification.json',raw,str(reuse_verified))

 record('source/cpu24-reference.json',dict(path=str(cpu_path.relative_to(ROOT)),sha256=CPU_SHA,revision=CPU_REV,new_cpu_compile=False,kernel_contract_sources_equal=True,reuse_receipt_sha256=REUSE_SHA,scope='Original CPU24 actual 165-test/30-kernel proof reused by exact source equality. CPU capsule13.0.3 versus serving image13.3.1; no CPU27 compile occurred.'),'original CPU24 archive + exact source objects + original reuse receipt')
 # Verify every passive chunk before using it as an attribution source.
 ends={}; chunks={}; streams={}
 for e in events:
  if e['kind']!='chunk': continue
  key=e['node'],e['channel']; need(key[0] in NODES and key[1] in ('stdout','stderr'),'unexpected stream key'); need(e['offset']==ends.get(key,0),'stream offset gap')
  ends[key]=e['offset']+e['bytes']; chunks.setdefault(key,[]).append(e)
 for key,end in ends.items():
  path=stream_root/(key[0]+'.'+key[1]+'.raw'); raw=read(path); need(len(raw)==end,'stream tail differs')
  for e in chunks[key]: need(sha(raw[e['offset']:e['offset']+e['bytes']])==e['sha256'],'stream chunk hash differs')
  streams[key]=raw; packed('streams/'+path.name+'.gz',raw,str(path))
 need(all((n,'stdout') in streams for n in NODES),'missing node stdout')
 packed('streams/events.jsonl.gz',eraw,str(stream_root/'events.jsonl'))
 record('streams/closure.json',dict(observer_pid=observer_pid,alive=False,attempted_arms=attempted,events=[e for e in events if e['kind'] in ('stream_end','observer_stop','observer_finished')],scope='All original chunk hashes/offsets verified. Reservation-window bytes remain unassigned until matched with a strict snapshot.'),'closed passive observer')
 # Raw Docker Config.Env/Cmd and inspect objects are read privately, never exported.
 identities={}; snapshot_status={}
 frozen_parser=gitfile('bench/glm53_launch_metadata.py'); parser_ns={'__name__':'archive_frozen_launch_parser'}
 exec(compile(frozen_parser,'<frozen-launch-parser>','exec'),parser_ns)
 def fixed(c): return {k:c[k] for k in ('Id','Created','Image','Config','HostConfig','RestartCount')}|{'Mounts':sorted(c['Mounts'],key=lambda m:json.dumps(m,sort_keys=True)),'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
 for arm in ARMS:
  private=Path('/tmp/glm53-onepass27-live-'+arm+'-observer')
  if not (private/'identity.json').exists():
   snapshot_status[arm]=dict(verdict='MISSING',attempted=arm in attempted,started=arm in begun,complete_record=arm in names)
   if (private/'failure.private.log').exists():
    raw=read(private/'failure.private.log'); snapshot_status[arm]['private_failure_log']=dict(path=str(private/'failure.private.log'),bytes=len(raw),sha256=sha(raw),exported=False)
   missing.append(str(private/'identity.json')); continue
  iraw=read(private/'identity.json'); identity=json.loads(iraw)
  need((identity['revision'],identity['session'],str(identity['ticket']),str(identity['owner_pid']),identity['arm'])==(REV,SESSION,TICKET,str(PID),arm),'snapshot binding differs')
  need(identity['source']=='/home/choiceoh/stkernel-ep-onepass-0909-27' and set(identity['nodes'])==set(NODES),'snapshot source/nodes differ')
  need(str(identity['owner_start_tick'])==START,'snapshot supervisor start differs')
  cb=identity['cpu24_binding']; original_reuse=json.loads(items['job/cpu24-reuse.json'])
  need(cb['mode']=='ORIGINAL_CPU24_EXACT_SOURCE_REUSE'
       and cb['reuse_path']=='/tmp/glm53-ep-onepass-0909-27/cpu24-reuse.json'
       and cb['reuse_sha256']==REUSE_SHA and cb['cpu_origin']==original_reuse['cpu_origin']
       and cb['new_source']=='/home/choiceoh/stkernel-ep-onepass-0909-27' and cb['new_revision']==REV
       and cb['new_cpu_compile'] is False and cb['mounted_and_contract_sources_equal'] is True,
       'snapshot original CPU24/source binding differs')

  need(arm in attempted and arm in begun,'snapshot arm was not observed/started')
  allowed={n+'.'+suffix for n in NODES for suffix in ('docker.log.gz','serving.log.gz','manifest.tsv.gz','inspect.before.json.gz','inspect.after.json.gz')}
  need(set(identity['files'])==allowed,'snapshot file allowlist differs')
  for name,d in identity['files'].items():
   stored=read(private/name); original=gzip.decompress(stored)
   need(len(stored)==d['stored_bytes'] and sha(stored)==d['stored_sha256'] and len(original)==d['original_bytes'] and sha(original)==d['original_sha256'],'snapshot file hash differs')
   if '.inspect.' not in name: save('snapshot/'+arm+'/'+name,stored,str(private/name),original_bytes=len(original),original_sha256=sha(original))
  for rank,node in enumerate(NODES):
   d=identity['nodes'][node]; before=json.loads(gzip.decompress(read(private/(node+'.inspect.before.json.gz'))))[0]; after=json.loads(gzip.decompress(read(private/(node+'.inspect.after.json.gz'))))[0]
   need(fixed(before)==fixed(after) and before['State']['Running'] and after['State']['Running'],'snapshot container drift')
   need(before['Id']==d['id'] and before['Image']==d['image']==IMAGE and before['State']['StartedAt']==d['started_at'] and before['Created']==d['created_at'],'snapshot container identity differs')
   need(date(d['started_at'])>=date(d['created_at'])>=identity['arm_event']['started_at'],'container predates arm')
   need(d['source']==dict(manifest_sha256=sha(manifest),mounts=mounts),'snapshot mounted source differs')
   resolved=[m for m in before['Mounts'] if m['Destination'] in mounts]
   need(len(resolved)==len(mounts) and {m['Destination'] for m in resolved}==set(mounts) and all(m['Type']=='bind' and m['RW'] is False and m['Source'].startswith('/home/choiceoh/overlays/glm53/') for m in resolved),'resolved overlay mount contract differs')
   topology=parser_ns['launch_parallelism'](before['Config']['Cmd'])
   need(topology==d['topology'] and all(topology[k]==v for k,v in dict(enabled=False,tensor_parallel_size=4,nnodes=4,node_rank=rank).items()),'actual TP geometry differs')
   payload=parser_ns['_WRAPPER'].fullmatch(before['Config']['Cmd'][1])[1]
   script=base64.b64decode(payload,validate=True).decode(); line=script[len(parser_ns['_GID_PRELUDE']):].removesuffix('\n')
   argv=parser_ns['_literal_argv'](line[:-len(parser_ns['_REDIRECTION'])])
   mm=[]; endpoint={}
   for index,arg in enumerate(argv):
    key,equal,value=arg.partition('=')
    if key in ('--limit-mm-per-prompt','--host','--port'):
     if not equal: need(index+1<len(argv),'missing launch option value'); value=argv[index+1]
     if key=='--limit-mm-per-prompt': mm.append(json.loads(value))
     else: need(key not in endpoint,'duplicate endpoint'); endpoint[key]=value
   need(len(mm)==1 and isinstance(mm[0],dict) and set(mm[0])=={'image','video'} and all(type(v)is int for v in mm[0].values()) and mm[0]==d['mm_limit']=={'image':4,'video':0},'actual MM budget differs')
   if rank==0: need(endpoint=={'--host':'127.0.0.1','--port':'18000'},'actual private endpoint differs')
   env={};
   for entry in before['Config']['Env']:
    k,v=entry.split('=',1); need(k not in env,'duplicate environment key'); env[k]=v
   flags={'VLLM_B12X_EP_ZERO_WEIGHT_MICRO':'0','VLLM_GLM53_EP_PREFILL_LOCAL':'0','VLLM_B12X_EP_WARM_COMPACT':'0','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE':'1','VLLM_GLM53_STARTUP_TRIM':'1','VLLM_GLM53_TP_SF6_Q0':'1' if arm=='A' else '0'}
   need(d['flags']==flags and all(env.get(k)==v for k,v in flags.items()),'unexpected arm flags')
   need(d['environment_sha256']==sha(json.dumps(env,sort_keys=True,separators=(',',':')).encode()),'environment hash differs')
   need(d['mm_limit']=={'image':4,'video':0},'MM budget differs')
   rd=d['readiness']; need(rd['graph_finished'] and rd['candidate']==(arm=='A') and len(rd['tp_sf6_q0_pass_records'])==(1 if arm=='A' else 0),'graph/canary readiness differs')
   trim=rd['startup_trim_records']; need(len(trim)==1,'trim receipt count differs'); tr=trim[0]['receipt']
   need(tr['verdict']=='COMPLETE' and tr['rank']==rank and tr['measurement_errors']==[],'trim not complete')
   need([x['stage'] for x in tr['stages']]==['synchronize','gc_collect','empty_cache','malloc_trim'] and all(x['status']=='COMPLETE' for x in tr['stages']),'trim stages differ')
   need(tr['stages'][-1]['returned'] in (0,1),'malloc trim return invalid')
   need(all(set(tr[phase])=={'allocated','reserved','mem_available','vm_rss'} and all(type(v)is int and v>=0 for v in tr[phase].values()) for phase in ('before','after')),'trim metrics incomplete')
   need(date(d['started_at'])<=tr['started_at']<=tr['completed_at']<=d['capture_finished_at'] and tr['before']['allocated']==tr['after']['allocated'],'trim timing/allocation differs')
   log=gzip.decompress(items['snapshot/'+arm+'/'+node+'.serving.log.gz']); trim_lines=[line for line in log.splitlines() if b'[glm53-startup-trim] ' in line]
   need(len(trim_lines)==1 and sha(trim_lines[0])==trim[0]['line_sha256'] and json.loads(trim_lines[0].split(b'[glm53-startup-trim] ',1)[1])==tr,'trim log bytes differ')
   graph=re.search(rb'Graph capturing finished in [0-9]+ secs, took ',log); need(graph and graph.start()<log.index(trim_lines[0]),'trim precedes graph')
  save('snapshot/'+arm+'/allowlisted-identity.json',iraw,str(private/'identity.json'))
  parser_raw=read(private/'launch-parser.py'); need(parser_raw==frozen_parser and sha(parser_raw)==identity['parser_sha256'],'snapshot parser differs'); save('snapshot/'+arm+'/launch-parser.py',parser_raw,str(private/'launch-parser.py'))
  identities[arm]=identity; snapshot_status[arm]=dict(verdict='STRICT_IDENTITY_VERIFIED',complete_record=arm in names)
 record('snapshot/availability.json',snapshot_status,'original strict snapshot availability; missing snapshots confer no identity proof')
 completed=[]
 for arm,row in zip(names,rows):
  need(row['session']==SESSION and row['git']==REV[:8] and row['overlay']==sha(manifest)[:12],'canonical record source differs')
  if arm in identities: need(row['boot_id']==identities[arm]['nodes']['local']['id']+'|'+identities[arm]['nodes']['local']['started_at'],'record boot differs')
  fixed_requests=[q for q in row.get('requests',[]) if q.get('fixed_decode')]
  eligible=[q for q in fixed_requests if type(q.get('completion_tokens'))is int and q['completion_tokens']>1 and type(q.get('decode_s'))in (int,float) and math.isfinite(q['decode_s']) and q['decode_s']>0]
  pooled=sum(q['completion_tokens']-1 for q in eligible)/sum(q['decode_s'] for q in eligible) if eligible and len(eligible)==len(fixed_requests) else None
  channels=[]
  for index,q in enumerate(row.get('requests',[])):
   channels.append(dict(request_index=index,ctx=q.get('ctx'),rep=q.get('rep'),fixed_decode=q.get('fixed_decode',False),channel_diagnostics=q.get('channel_diagnostics'),status='RECORDED' if 'channel_diagnostics' in q else 'MISSING'))
  record('channels/'+arm+'.json',dict(arm=arm,requests=channels,scope='Exact native diagnostics fields copied from original canonical record. Existing combined-text gates remain authoritative; no regating, attribution waiver, or reconstructed channels.'),'job/onepass.jsonl')
  completed.append(dict(arm=arm,requests=len(row.get('requests',[])),fixed_decode_tok_s=[q.get('decode_tok_s') for q in fixed_requests],fixed_pooled_tok_s=pooled,prefill=row.get('prefill'),quality=row.get('quality'),korean=row.get('korean'),proof_ok=row.get('proof_ok'),channel_diagnostics_recorded=sum('channel_diagnostics' in q for q in row.get('requests',[])),strict_identity=arm in identities,onepass_record_canonical_json_sha256=sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode())))
 # The prepared extractor is pure; use its exact source/runtime validation API,
 # but retain a missing/incomplete extractor receipt instead of fabricating PASS.
 canary_root=Path('/tmp/glm53-onepass27-canary-A'); canary_summary=dict(verdict='MISSING',candidate_started='A' in begun)
 if (canary_root/'validation.json').exists():
  validation_raw=read(canary_root/'validation.json'); validation=json.loads(validation_raw)
  need((validation['revision'],validation['cpu24_result_sha256'],validation['cpu24_source_revision'],validation['cpu24_reuse_sha256'],validation['expected_arm'],validation['occurrence'])==(REV,CPU_SHA,CPU_REV,REUSE_SHA,'A',0),'canary source/reuse binding differs')
  save('canary/A/validation-original.json',validation_raw,str(canary_root/'validation.json')); checked={}
  for node in NODES:
   path=canary_root/(node+'.json')
   if not path.exists(): missing.append(str(path)); continue
   raw=read(path); receipt=json.loads(raw); line=read(canary_root/(node+'.receipt-line.raw')); prov=validation['nodes'][node].get('provenance'); stream=streams[node,'stdout']
   if prov is None:
    # An extractor validation error can omit its derived provenance even when
    # it preserved the original raw receipt. Bind that line without making it PASS.
    offset=stream.find(line); need(offset>=0 and stream.find(line,offset+1)<0,'unbound/ambiguous incomplete canary line')
    payload=re.split(rb'\[tp-sf6-q0-selftest\] (?:PASS|FAIL) ',line,maxsplit=1)
    need(len(payload)==2 and json.JSONDecoder().raw_decode(payload[1].decode())[0]==receipt,'incomplete canary JSON differs')
    prov=dict(raw_offset=offset,raw_end=offset+len(line),line_sha256=sha(line),prefix_sha256=sha(stream[:offset+len(line)]),json_sha256=sha(raw.rstrip(b'\n')),scope='Collector-derived exact-byte binding; original extractor supplied no provenance and no acceptance is inferred')
   need(stream[prov['raw_offset']:prov['raw_end']]==line and sha(line)==prov['line_sha256'] and sha(stream[:prov['raw_end']])==prov['prefix_sha256'] and sha(raw.rstrip(b'\n'))==prov['json_sha256'],'canary stream bytes differ')
   save('canary/A/'+node+'.json',raw,str(path),raw_stream_binding=prov); save('canary/A/'+node+'.receipt-line.raw',line,str(canary_root/(node+'.receipt-line.raw')))
   if validation['verdict']=='RAW_CANARY_SOURCE_VALIDATED':
    detail=validator.validate(receipt,expected); detail.pop('versions',None)
    need(stream.count(b'[tp-sf6-q0-selftest] PASS ')==1 and b'[tp-sf6-q0-selftest] FAIL ' not in stream,'unexpected canary multiplicity/failure')
    bound=False
    if 'A' in identities:
     d=identities['A']['nodes'][node]; log=gzip.decompress(items['snapshot/A/'+node+'.serving.log.gz']); marker=b'[tp-sf6-q0-selftest] PASS '; need(log.count(marker)==1,'snapshot canary count differs')
     logged,_=json.JSONDecoder().raw_decode(log.split(marker,1)[1].decode()); need(logged==receipt,'snapshot canary differs')
     need(date(d['started_at'])<=receipt['started_at']<=receipt['completed_at']<=d['capture_finished_at'] and receipt['completed_at']<=d['readiness']['startup_trim_records'][0]['receipt']['started_at'],'canary precedes selected container or follows trim'); bound=True
    checked[node]=dict(detail,strict_source_container_bound=bound)
  if validation['verdict']=='RAW_CANARY_SOURCE_VALIDATED': need(len(checked)==4,'incomplete purported PASS canary')
  canary_summary=dict(verdict=validation['verdict'],nodes=checked,strict_source_container_bound=len(checked)==4 and all(v['strict_source_container_bound'] for v in checked.values()))
 else: missing.append(str(canary_root/'validation.json'))
 record('canary/verified-summary.json',canary_summary,'original candidate A receipt bytes; explicit missing/incomplete status preserved')
 verification_helper=Path('/tmp/glm53_verify_onepass27.py')
 need(sha(read(verification_helper))=='6b5ed3bc23e726df315d590a21ab4d2c73c23be61d405b032f2c8f7016451469','independent snapshot verifier changed')
 save('tools/independent-snapshot-verifier.py',read(verification_helper),str(verification_helper))
 for arm in ARMS:
  path=Path('/tmp/glm53-onepass27-'+arm+'-verified.json')
  if not path.exists(): missing.append(str(path)); continue
  raw=read(path); verified=json.loads(raw)
  need(verified['verdict']=='STRICT_ARM_SOURCE_MM_TRIM_CANARY_VERIFIED' and verified['revision']==REV
       and verified['arm']==arm and str(verified['ticket'])==TICKET and int(verified['owner_pid'])==PID
       and str(verified['owner_start_tick'])==START and arm in identities
       and verified['snapshot_sha256']==sha(items['snapshot/'+arm+'/allowlisted-identity.json'])
       and verified['cpu24_binding']==identities[arm]['cpu24_binding']
       and verified['cpu24_counts']==dict(mounted_sources=len(cpu['mounted_sources']),contract_sources=len(cpu['contract_sources'])),
       'independent strict snapshot verification binding differs')
  save('snapshot/'+arm+'/independent-verification.json',raw,str(path))
 for source,name in [('/tmp/glm53-onepass27-submit.json','submission/request.json'),('/tmp/glm53-onepass27-submit-receipt.txt','submission/receipt.json'),('/tmp/glm53_onepass27_observer.py','tools/passive-observer.py'),('/tmp/glm53_onepass27_snapshot.py','tools/snapshot-helper.py'),('/tmp/glm53_onepass27_progress.py','tools/progress-helper.py'),('/tmp/glm53_onepass27_failfast.py','tools/failfast-helper.py'),(str(helper),'tools/canary-extractor-validator.py')]: save(name,read(source),source)
 submission=json.loads(items['submission/receipt.json']); need((submission['session'],str(submission['ticket']),submission['pid'])==(SESSION,TICKET,PID),'submission binding differs')
 failfast={}
 for name in ('pre-signal.json','result.json'):
  key='failure/failfast-first-fixed-A/'+name
  if key in items:
   d=json.loads(items[key]); need((d['session'],str(d['ticket']),d['owner_pid'],str(d['owner_start_tick']),d['revision'])==(SESSION,TICKET,PID,START,REV),'failfast receipt binding differs'); failfast[name]=d
 summary=dict(schema=1,verdict='TERMINAL_EVIDENCE_ARCHIVED_NOT_ADOPTION_VERDICT',revision=REV,session=SESSION,ticket=TICKET,supervisor_pid=PID,supervisor_start_tick=START,arm_order=list(ARMS),started_arms=begun,completed_arms=names,partial_or_no_record_arms=[a for a in begun if a not in names],unstarted_arms=[a for a in ARMS if a not in begun],completed=completed,canonical_verdicts=verdicts,fleet=remote['after']['fleet'],own_holder_after=False,bound_supervisor_alive=False,failfast_receipts=failfast,cancellation_receipt_present='job/cancel-request.json' in items,missing_originals=remote['absent']+missing,canary=canary_summary,cpu=dict(original_revision=CPU_REV,original_result_sha256=CPU_SHA,reuse_receipt_sha256=REUSE_SHA,new_cpu_compile=False),performance_acceptance=False,adoption_acceptance=False,full_sanitizer_acceptance=False,public_recovery_verified=False)
 record('result-summary.json',summary,'unmodified canonical records/verdicts + terminal state; summary supplies no new acceptance judgment')
 lines=['# onepass27 terminal evidence','',f'Frozen source `{REV}`; session `{SESSION}`, ticket `{TICKET}`, supervisor `{PID}` / start tick `{START}`.',f'Canonical order: B → A. Started: {", ".join(begun) or "none"}. Complete original records: {", ".join(names) or "none"}. Only A enables TP Q0.',f'Terminal payload rc={remote["after"]["fleet"]["payload_returncode"]}, supervisor rc={remote["after"]["fleet"]["returncode"]}. Bound supervisor and owned holder are absent. Observer is closed. Recovery policy/status is preserved without claiming public service restoration.','', '`job/onepass.jsonl` and `job/verdicts.jsonl` are original bytes when available. Native per-request `channel_diagnostics` remain unmodified; `channels/` copies these fields for review. Existing combined-text gates and their original verdicts are retained. No output-channel attribution waives a quality failure.','']
 for row in completed: lines.append(f'- {row["arm"]}: fixed decode {row["fixed_decode_tok_s"]} tok/s; pooled {row["fixed_pooled_tok_s"]} tok/s; quality {row["quality"]}; Korean {row["korean"]}; diagnostics {row["channel_diagnostics_recorded"]}/{row["requests"]}.')
 lines+=['','Pooled fixed decode is `sum(completion_tokens - 1) / sum(decode_s)` over all recorded fixed repetitions only when every such entry has valid positive timing. This archive does not decide matched throughput, default promotion, or sanitizer acceptance. Missing/partial arms are not reconstructed as results. GPU25 is historical context only: GPU27 uses a newer full source including the required main startup changes, while the original CPU24 kernel/contract bytes remain exact.','', 'Original CPU24 proof is reused verbatim: revision `'+CPU_REV+'`, result SHA256 `'+CPU_SHA+'`, 165 CPU tests and 30 compiled kernels. The original `cpu24-reuse.json` proves exact mounted/contract source equality for GPU27. No CPU27 compile or relabeled CPU receipt is claimed. CPU used capsule13.0.3 while serving used image13.3.1.','', 'Available strict snapshots verify pinned image/source, TP4/4, EP/local/warm/zero0, Q0 only A, skip1/trim1, and actual image4/video0. Raw Docker Env/Cmd/inspect stay private. Each archived non-inspect snapshot and all passive chunks are hash-verified. Snapshot/canary absence is explicit and confers no proof. Unassigned stream bytes keep that scope.','', 'Only explicitly known or terminal-log-referenced temporary legs are considered. Cleaned files remain missing in `terminal-capture.json` / `legs/availability.json`; no reconstruction is labeled original. Deterministic gzip retains original/stored hashes. This collector performs read-only SSH/file/git operations and writes only this fresh archive.']
 save('README.md',('\n'.join(lines)+'\n').encode(),'bounded archival scope'); save('collect-evidence.py',read(__file__),'this collector')
 record('originals.json',originals.copy(),'original/stored byte provenance')
 items['SHA256SUMS']=''.join(f'{sha(raw)}  {name}\n' for name,raw in sorted(items.items())).encode()
 need(not OUT.exists(),'archive appeared during capture'); OUT.mkdir(parents=False)
 for name,raw in items.items():
  path=OUT/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
 for name,raw in items.items(): need(sha(read(OUT/name))==sha(raw),'local archive write differs')
 print(json.dumps(dict(archive=str(OUT),files=len(items),bytes=sum(map(len,items.values())),completed_arms=names,summary_sha256=sha(items['result-summary.json']),sums_sha256=sha(items['SHA256SUMS'])),sort_keys=True))
if __name__=='__main__': main()
