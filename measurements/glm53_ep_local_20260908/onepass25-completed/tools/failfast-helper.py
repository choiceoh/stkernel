#!/usr/bin/env python3
"""Prepared only: poll the owned candidate's first fixed1024 stdout; opt-in SIGTERM.

Run after review with exact ticket/PID/start tick/revision. No HTTP, workload,
deployment or queue writes. Default is report-only, including below threshold.
"""
import argparse, hashlib, json, math, shlex, subprocess, sys
from pathlib import Path

REMOTE = r'''
import datetime, hashlib, json, math, os, re, shlex, signal, stat, subprocess, time
from pathlib import Path
F=Path('/home/choiceoh/glm53-logs/fleet')
S=Path('/home/choiceoh/stkernel-ep-onepass-0909-25')
J=Path('/tmp/glm53-ep-onepass-0909-25')
SESSION='eplocalonepass0909v25'; PREFIX='EPONEPASS25'
owner=cfg['owner_pid']; ticket=cfg['ticket']; revision=cfg['revision']; signal_sent=False
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def need(ok,why):
    if not ok: raise ValueError(why)
def regular(path,limit=16*2**20):
    a=path.lstat();need(stat.S_ISREG(a.st_mode) and a.st_uid==os.getuid() and a.st_size<=limit,'unsafe file: '+str(path))
    with path.open('rb') as f: raw=f.read(a.st_size)
    b=path.lstat();need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and len(raw)==a.st_size and b.st_size>=a.st_size,'file identity changed')
    return raw,dict(path=str(path),device=a.st_dev,inode=a.st_ino,bytes=len(raw),sha256=sha(raw),mtime_ns=a.st_mtime_ns)
def proc(pid):
    p=Path('/proc')/str(pid); f=(p/'stat').read_bytes().rsplit(b')',1)[1].decode('ascii').split()
    if f[0]=='Z': raise ProcessLookupError('process is zombie')
    args=[x.decode(errors='surrogateescape') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    return dict(pid=pid,ppid=int(f[1]),start=f[19],cwd=str((p/'cwd').resolve()),args=args)
def script_arg(p,name):
    matches=[x for x in p['args'] if x.endswith('/bench/'+name) or x=='bench/'+name]
    need(len(matches)==1,'missing/ambiguous process script '+name)
    path=Path(matches[0]);return path if path.is_absolute() else Path(p['cwd'])/path

def source():
    env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
    head=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],env=env,timeout=5).decode().strip()
    dirty=subprocess.check_output(['git','-C',str(S),'status','--porcelain'],env=env,timeout=5).strip()
    need(head==revision and not dirty,'frozen source changed')
    return head

def reservation():
    raw,_=regular(F/'pending'/(sha(SESSION.encode())+'.json'),2**20); p=json.loads(raw)
    need((p.get('session'),str(p.get('ticket')),p.get('pid'),p.get('start'))==(SESSION,ticket,owner,cfg['owner_start_tick']),'pending identity differs')
    need(p.get('repo')==str(S) and p.get('cwd')==str(S),'pending source/cwd differs')
    if p.get('payload_returncode') is not None or p.get('returncode') is not None or p.get('state') in ('finished','failed','cancelled','interrupted'):
        return p,False
    actual=proc(owner);need(actual['start']==cfg['owner_start_tick'] and actual['cwd']==str(S),'supervisor reused/source differs')
    boot=script_arg(actual,'fleet_boot.py');fleet=boot.with_name('fleet.sh')
    need(boot==S/'bench/fleet_boot.py' or (boot.parent.parent.parent==F/'runners' and re.fullmatch('[0-9a-f]{64}',boot.parent.parent.name)), 'supervisor is not source or pinned runner')
    pos=actual['args'].index(str(boot));need(actual['args'][pos+1:pos+3]==[str(fleet),SESSION],'supervisor command/session differs')
    need(regular(boot,2**20)[0]==regular(S/'bench/fleet_boot.py',2**20)[0],'supervisor source bytes differ')
    need(regular(fleet,2**20)[0]==regular(S/'bench/fleet.sh',2**20)[0],'supervisor fleet bytes differ')
    holder=[] if not (F/'holder').exists() else regular(F/'holder',4096)[0].decode().strip().split('|')
    own=holder[:2]==[SESSION,str(owner)]
    return p,own

def arm_event():
    path=F/'run-logs'/(sha((SESSION+'\0'+ticket).encode())+'.log'); raw,meta=regular(path)
    complete=raw if raw.endswith(b'\n') else raw.rsplit(b'\n',1)[0]+b'\n' if b'\n' in raw else b''
    events=list(re.finditer(rb'(?m)^== ([0-9]{2}:[0-9]{2}:[0-9]{2}) arm ([^ :\r\n]+):[^\r\n]*$',complete))
    names=[m[2].decode() for m in events];expected=[PREFIX+a for a in ('B0','B1','A','B2','B3')]
    if not names:return None
    need(names==expected[:len(names)],'arm order different')
    e=events[-1];now=datetime.datetime.now().astimezone();h,m,s=map(int,e[1].split(b':'))
    born=now.replace(hour=h,minute=m,second=s,microsecond=0)
    if born>now:born-=datetime.timedelta(days=1)
    need(0<=(now-born).total_seconds()<3*3600,'stale arm event')
    return dict(arm=names[-1][len(PREFIX):],started_at=born.timestamp(),offset=e.start(),line=e[0].decode(),log_device=meta['device'],log_inode=meta['inode'])

def records():
    if not (J/'onepass.jsonl').exists():return b'',[]
    raw,_=regular(J/'onepass.jsonl',4*2**20)
    need(not raw or raw.endswith(b'\n'),'record append in progress')
    rows=[json.loads(l) for l in raw.splitlines()]
    need(all(r.get('session')==SESSION and r.get('git')==revision[:8] for r in rows),'record source/session differs')
    return raw,[r['name'] for r in rows]

def lineage(arm):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            q=proc(int(p.name))
            if not any(x.endswith('/bench/onepass.py') or x=='bench/onepass.py' for x in q['args']):continue
            if q['cwd']!=str(S):continue
            positions=[i for i,arg in enumerate(q['args']) if arg=='--name']
            if len(positions)!=1 or positions[0]+1>=len(q['args']):continue
            pos=positions[0]
            if q['args'][pos+1]!=PREFIX+arm:continue
            need(script_arg(q,'onepass.py')==S/'bench/onepass.py','onepass source differs')
            chain=[q]
            for _ in range(12):
                parent=proc(chain[-1]['ppid']);chain.append(parent)
                if parent['pid']==owner:break
                if parent['ppid']<=1:break
            if chain[-1]['pid']!=owner:continue
            lever=chain[1];need(script_arg(lever,'ab-lever.sh')==S/'bench/ab-lever.sh','onepass immediate parent is not canonical lever')
            lpos=lever['args'].index(str(S/'bench/ab-lever.sh'))
            need(lever['args'][lpos+1]==PREFIX+arm,'lever arm differs')
            need(all(v['cwd']==str(S) for v in chain),'descendant cwd differs')
            need(chain[-1]['start']==cfg['owner_start_tick'],'supervisor start differs')
            expected={'MM_LIMIT':'{"image":4,"video":0}','ENABLE_EP':'0','VLLM_GLM53_EP_PREFILL_LOCAL':'0','VLLM_B12X_EP_WARM_COMPACT':'0',
                      'VLLM_B12X_EP_ZERO_WEIGHT_MICRO':'0','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE':'1','VLLM_GLM53_TP_SF6_Q0':'1','VLLM_GLM53_STARTUP_TRIM':'1'}
            selected={}
            for raw_entry in (p/'environ').read_bytes().split(b'\0'):
                if not raw_entry:continue
                k,sep,v=raw_entry.partition(b'=');key=k.decode(errors='surrogateescape')
                if key in expected:
                    need(sep and key not in selected,'duplicate/malformed candidate flag')
                    selected[key]=v.decode(errors='surrogateescape')
            # ab-lever applies its arm knobs only to the launcher, not to the
            # benchmark process. Resolve those overrides over inherited flags.
            need(lpos+2<len(lever['args']),'candidate lever overrides absent')
            overrides={}
            for item in shlex.split(lever['args'][lpos+2]):
                key,sep,value=item.partition('=')
                need(sep and key not in overrides,'malformed/duplicate lever override')
                overrides[key]=value
            selected.update({key:value for key,value in overrides.items() if key in expected})
            need(selected==expected,'owned candidate launcher flags differ')
            chain[0]['candidate_launch_flags']=selected
            found.append(chain)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    need(len(found)<=1,'multiple owned candidate onepasses')
    return found[0] if found else None

def evidence(chain,event):
    raw,meta=regular(Path('/tmp')/('leg.'+str(chain[1]['pid'])),4*2**20)
    matches=list(re.finditer(rb'(?m)^\s*fixed2K rep=(\d+) tokens=1024/1024 decode=([0-9]+\.[0-9]+) tok/s[^\r\n]*\n',raw))
    if not matches:return None
    need(len(matches)==1 and matches[0][1]==b'0','first fixed result already stale/multiple repetitions')
    need(time.time()-meta['mtime_ns']/1e9<=cfg['max_age_seconds'],'fixed stdout stale')
    need(meta['mtime_ns']/1e9>=event['started_at'],'leg predates candidate arm')
    e=matches[0];rate=float(e[2]);need(math.isfinite(rate),'nonfinite rate')
    return raw,dict(**meta,offset=e.start(),line=e[0].decode(),line_sha256=sha(e[0]),decode_tok_s=rate,observed_at=time.time())

def run():
    global signal_sent
    source();deadline=time.monotonic()+cfg['timeout_seconds']
    while time.monotonic()<deadline:
        p,own=reservation()
        if p.get('payload_returncode') is not None or p.get('returncode') is not None:return dict(action='REPORT_TERMINAL',signal_sent=False)
        if not own:
            if p.get('state')=='queued':time.sleep(cfg['poll_seconds']);continue
            return dict(action='REPORT_NOT_OWNED',signal_sent=False)
        if p.get('phase')!='payload':time.sleep(cfg['poll_seconds']);continue
        try:event=arm_event()
        except FileNotFoundError:time.sleep(cfg['poll_seconds']);continue
        if event is None:time.sleep(cfg['poll_seconds']);continue
        before_records,names=records()
        if event['arm']!=cfg['arm'] or event['arm'] != 'A':
            return dict(action='REPORT_OTHER_ARM_NO_SIGNAL',arm=event['arm'],completed_records=names,signal_sent=False)
        if PREFIX+cfg['arm'] in names:return dict(action='REPORT_CANDIDATE_COMPLETE_NO_SIGNAL',completed_records=names,signal_sent=False)
        chain=lineage(cfg['arm'])
        if chain is None:time.sleep(cfg['poll_seconds']);continue
        observation=evidence(chain,event)
        if observation is None:time.sleep(cfg['poll_seconds']);continue
        raw,meta=observation
        receipt=dict(schema=1,session=SESSION,ticket=ticket,owner_pid=owner,owner_start_tick=cfg['owner_start_tick'],revision=revision,
                     arm=event,processes=[{k:v[k] for k in ('pid','ppid','start','cwd')} for v in chain],
                     threshold_tok_s=cfg['threshold'],candidate_launch_flags=chain[0]['candidate_launch_flags'],observed=meta,records_sha256=sha(before_records),completed_records=names,
                     helper_sha256=cfg['helper_sha256'],signal_sent=False,scope='early-stop policy on first rounded stdout result; no matched performance or adoption verdict')
        if meta['decode_tok_s']>=cfg['threshold']:return {**receipt,'action':'REPORT_FIRST_RESULT_NOT_BELOW_THRESHOLD'}
        if not cfg['signal']:return {**receipt,'action':'REPORT_BELOW_THRESHOLD_DRY_RUN'}
        need(not (J/'cancel-request.json').exists(),'existing cancellation receipt')
        # pidfd binds the exact process across final checks; never use killpg.
        need(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'pidfd unavailable')
        fd=os.pidfd_open(owner,0)
        try:
            source();p,own=reservation();need(own and p.get('phase')=='payload','reservation moved before signal')
            need(arm_event()==event,'candidate arm changed before signal')
            need(records()[0]==before_records,'candidate completed before signal')
            need(lineage(cfg['arm'])==chain,'process chain changed before signal')
            again=evidence(chain,event);need(again is not None and again[0]==raw,'leg changed before signal')
            target=J/('failfast-first-fixed-'+cfg['arm']);target.mkdir(mode=0o700)
            for name,data in [('leg.raw',raw),('pre-signal.json',(json.dumps({**receipt,'action':'SIGNAL_REQUESTED'},indent=2)+'\n').encode())]:
                with (target/name).open('xb') as f:f.write(data)
                (target/name).chmod(0o600)
            # Recheck after receipt I/O; any change leaves evidence and sends no signal.
            p,own=reservation();need(own and p.get('phase')=='payload','reservation changed after receipt')
            need(arm_event()==event and records()[0]==before_records,'arm completed/changed after receipt')
            need(lineage(cfg['arm'])==chain,'lineage changed after receipt')
            need(time.time()-meta['observed_at']<5,'final admission expired')
            signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
            signal_sent=True
            result={**receipt,'action':'SIGTERM_SENT','signal_sent':True,'signalled_at':time.time()}
            with (target/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
            (target/'result.json').chmod(0o600)
            return result
        finally:os.close(fd)
    return dict(action='REPORT_TIMEOUT_NO_SIGNAL',signal_sent=False)
try: print(json.dumps(run()),flush=True)
except Exception as exc:
    print(json.dumps(dict(action='ERROR_AFTER_SIGNAL' if signal_sent else 'DECLINED_NO_SIGNAL',signal_sent=signal_sent,error_type=type(exc).__name__,error=str(exc))),flush=True)
    raise SystemExit(2)
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',required=True);p.add_argument('--ticket',required=True)
    p.add_argument('--owner-pid',type=int,required=True);p.add_argument('--owner-start-tick',required=True)
    p.add_argument('--arm',choices=('A',),default='A')
    p.add_argument('--threshold',type=float,default=65.)
    p.add_argument('--timeout-seconds',type=int,default=3600)
    p.add_argument('--poll-seconds',type=float,default=1.)
    p.add_argument('--max-age-seconds',type=float,default=15.)
    p.add_argument('--signal',action='store_true',help='Opt in to exact supervisor SIGTERM after all guards; default report only')
    args=p.parse_args();cfg=vars(args)
    if not (len(args.revision)==40 and all(c in '0123456789abcdef' for c in args.revision) and args.ticket.isdecimal() and args.owner_start_tick.isdecimal() and args.owner_pid>1):p.error('exact immutable revision and actual reservation identity required')
    if not (math.isfinite(args.threshold) and 0<args.threshold<1000 and 1<=args.timeout_seconds<=10800 and .25<=args.poll_seconds<=5 and 1<=args.max_age_seconds<=30):p.error('bounded threshold/wait/age required')
    cfg['helper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    script='cfg='+repr(cfg)+'\n'+REMOTE
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','choiceoh@srv2',shlex.join(['python3','-B','-c',script])]
    return subprocess.call(command)

if __name__=='__main__':raise SystemExit(main())
