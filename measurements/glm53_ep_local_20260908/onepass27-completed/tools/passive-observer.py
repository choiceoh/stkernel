#!/usr/bin/env python3
"""Bounded passive evidence observer. Never starts a workload or alters fleet."""
import argparse, ast, base64, hashlib, json, os, signal, subprocess, sys, threading, time
from pathlib import Path

SESSION='eplocalonepass0909v27'
TICKET='1788932740730747'
OWNER_PID='730747'
OWNER_START_TICK='43512869'
REV='ea413ac4c39ba3e6e4009c73587b0d536053b4bf'
SOURCE='/home/choiceoh/stkernel-ep-onepass-0909-27'
ROOT=Path('/tmp/glm53-onepass27-streams')
SNAPSHOT=Path('/tmp/glm53_onepass27_snapshot.py')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','choiceoh@srv2','python3 -B -']
COMMON=r'''
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

def current_arm_event(session, ticket, owner_pid):
    import datetime, hashlib, json, pathlib, re, stat
    fleet=pathlib.Path('/home/choiceoh/glm53-logs/fleet')
    pending=json.loads((fleet/'pending'/(hashlib.sha256(session.encode()).hexdigest()+'.json')).read_text())
    if pending.get('session')!=session or str(pending.get('ticket'))!=ticket or str(pending.get('pid'))!=str(owner_pid) or str(pending.get('start'))!=owner_start_tick:
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
    expected=['EPONEPASS27'+arm for arm in ('B','A')]
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
    return dict(arm=names[-1].removeprefix('EPONEPASS27'),started_at=born.timestamp(),
                started_at_local=born.isoformat(),key=hashlib.sha256(json.dumps(marker,sort_keys=True).encode()).hexdigest(),**marker)
import hashlib,json,os,pathlib,re,subprocess,time
F=pathlib.Path('/home/choiceoh/glm53-logs/fleet')
def holder_fields():
    try: return (F/'holder').read_text().strip().split('|')
    except FileNotFoundError: return []
def state():
    holder=holder_fields()
    p=F/'pending'/(hashlib.sha256(session.encode()).hexdigest()+'.json')
    try: pending=json.loads(p.read_text())
    except FileNotFoundError: pending={'session':session,'ticket':ticket,'pid':owner_pid,'state':'record_absent','phase':'record_absent','start':owner_start_tick}
    if pending.get('session')!=session or pending.get('ticket')!=ticket or str(pending.get('pid'))!=str(owner_pid) or str(pending.get('start'))!=owner_start_tick: raise RuntimeError('reservation identity changed')
    own=len(holder)>1 and holder[0]==session and holder[1]==str(pending['pid'])
    if own:
        fields=pathlib.Path('/proc/'+str(owner_pid)+'/stat').read_bytes().rsplit(b')',1)[1].split()
        if fields[0]==b'Z' or fields[19].decode('ascii')!=str(pending.get('start')):raise RuntimeError('owned supervisor start differs')
    terminal=pending.get('payload_returncode') is not None or pending.get('returncode') is not None or pending.get('state') in ('finished','succeeded','failed','cancelled','interrupted')
    with (F/'log').open('rb') as stream:
        stream.seek(0,2); stream.seek(max(0,stream.tell()-2**20)); lines=stream.read().decode(errors='replace')
    go=bool(re.search(r'(?m)^.*\bGO '+re.escape(session)+r' \(pid '+str(pending['pid'])+r'\)(?:\s|$)',lines))
    return {'own':own,'go':go,'terminal':terminal or (go and not own),'phase':pending.get('phase'),'state':pending.get('state'),
            'payload_returncode':pending.get('payload_returncode'),'returncode':pending.get('returncode'),'at':time.time()}
'''
STATUS=r'''
s=state()
if s['own'] and s['go'] and not s['terminal']:
    try:
        arm_event=current_arm_event(session,ticket,owner_pid)
        c=json.loads(subprocess.check_output(['docker','inspect','glm53'],stderr=subprocess.DEVNULL,timeout=10))[0]
        env={}; duplicate=False
        for item in c['Config']['Env']:
            k,v=item.split('=',1); duplicate=duplicate or k in env; env[k]=v
        manifest=pathlib.Path('/home/choiceoh/overlays/glm53/manifest.tsv').read_bytes()
        frozen=pathlib.Path(source+'/build/glm53/manifest.tsv').read_bytes()
        exact=manifest==b'# source_commit='+revision.encode()+b'\n'+frozen
        exact=exact and subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],timeout=10).decode().strip()==revision
        exact=exact and not subprocess.check_output(['git','-C',source,'status','--porcelain','--untracked-files=no'],timeout=10).strip()
        born=__import__('datetime').datetime.fromisoformat(c['State']['StartedAt'].replace('Z','+00:00')).timestamp()
        created=__import__('datetime').datetime.fromisoformat(c['Created'].replace('Z','+00:00')).timestamp()
        path=pathlib.Path('/home/choiceoh/glm53-logs/glm53.log'); a=path.stat()
        log=''
        if a.st_mtime>=born and a.st_size<=128*2**20: log=path.read_text(errors='replace')
        again=json.loads(subprocess.check_output(['docker','inspect',c['Id']],stderr=subprocess.DEVNULL,timeout=10))[0]
        stable=current_arm_event(session,ticket,owner_pid)['key']==arm_event['key'] and c['Id']==again['Id'] and c['Created']==again['Created'] and c['State']['StartedAt']==again['State']['StartedAt'] and again['State']['Running'] is True
        graph=bool(re.search(r'Graph capturing finished in [0-9]+ secs, took ',log))
        namespace={'__name__':'observer_launch_parser'}
        exec(compile(pathlib.Path(source+'/bench/glm53_launch_metadata.py').read_bytes(),'<frozen-launch-parser>','exec'),namespace)
        topology=namespace['launch_parallelism'](c['Config']['Cmd'])
        payload=namespace['_WRAPPER'].fullmatch(c['Config']['Cmd'][1])[1]
        script=__import__('base64').b64decode(payload,validate=True).decode()
        line=script[len(namespace['_GID_PRELUDE']):].removesuffix('\n')
        argv=namespace['_literal_argv'](line[:-len(namespace['_REDIRECTION'])])
        mm_limit=exact_mm_limit(argv)
        tp_only=all(topology.get(k)==v for k,v in {'enabled':False,'tensor_parallel_size':4,'nnodes':4,'node_rank':0}.items())
        image_exact=c['Image']=='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
        pass_receipts=[]
        for line in log.splitlines():
            if '[tp-sf6-q0-selftest] PASS ' not in line:continue
            receipt,_=json.JSONDecoder().raw_decode(line.split('[tp-sf6-q0-selftest] PASS ',1)[1])
            if isinstance(receipt,dict) and receipt.get('verdict')=='PASS' and receipt.get('phase')=='complete':pass_receipts.append(receipt)
        q0_pass=bool(pass_receipts) and '[tp-sf6-q0-selftest] FAIL ' not in log
        trim_receipts=[]
        for line in log.splitlines():
            if '[glm53-startup-trim] ' not in line:continue
            receipt,_=json.JSONDecoder().raw_decode(line.split('[glm53-startup-trim] ',1)[1])
            trim_receipts.append(receipt)
        trim_complete=(len(trim_receipts)==1 and valid_startup_trim(trim_receipts[0],0,created))

        flags={k:env.get(k) for k in ('VLLM_GLM53_EP_PREFILL_LOCAL','VLLM_B12X_EP_WARM_COMPACT','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE','VLLM_B12X_EP_ZERO_WEIGHT_MICRO','VLLM_GLM53_TP_SF6_Q0','VLLM_GLM53_STARTUP_TRIM')}
        s['head']={'id':c['Id'],'created_at':c['Created'],'started_at':c['State']['StartedAt'],'running':c['State']['Running'],'stable':stable,
                   'arm_event':arm_event,'born_after_arm':born>=created>=arm_event['started_at'],
                   'flags':flags,'manifest_exact':exact,'duplicate_env':duplicate,'graph_finished':graph,
                   'tp_sf6_q0_pass':q0_pass,'tp_only':tp_only,'image_exact':image_exact,'mm_limit':mm_limit,
                   'startup_trim_complete':trim_complete,
                   'log_inode':a.st_ino,'log_bytes':a.st_size}
    except Exception as exc: s['head_unavailable']=type(exc).__name__
print(json.dumps(s))
'''
STREAM=r'''
import base64,selectors,signal
children=[]; sel=selectors.DefaultSelector(); reason='timeout'
def emit(value): print(json.dumps(value),flush=True)
def interrupted(signum,frame): raise InterruptedError('stream interrupted')
signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted); signal.signal(signal.SIGHUP,interrupted)
try:
    s=state()
    if not(s['own'] and s['go'] and not s['terminal']): raise RuntimeError('owned GO not active')
    for node in ('local','10.10.10.1','10.10.10.3','10.10.10.4'):
        cmd=['tail','-n','0','-F','/home/choiceoh/glm53-logs/glm53.log']
        if node!='local': cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','choiceoh@'+node,'exec tail -n 0 -F /home/choiceoh/glm53-logs/glm53.log']
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True); children.append(p)
        for channel,pipe in (('stdout',p.stdout),('stderr',p.stderr)):
            sel.register(pipe,selectors.EVENT_READ,(node,channel))
        emit({'kind':'stream_start','node':node,'pid':p.pid,'at':time.time(),'attribution':'UNASSIGNED: owned reservation window only; logs may precede this arm boot'})
    next_check=0
    while time.time()<deadline and sel.get_map():
        if time.time()>=next_check:
            s=state(); next_check=time.time()+1
            if not s['own'] or s['terminal']:
                reason='release_or_terminal'; break
        for key,_ in sel.select(timeout=.5):
            current_holder=holder_fields()
            if len(current_holder)<2 or current_holder[0]!=session or current_holder[1]!=str(owner_pid):
                reason='release'; raise InterruptedError('reservation released')
            raw=os.read(key.fileobj.fileno(),65536)
            if not raw: sel.unregister(key.fileobj); continue
            node,channel=key.data
            emit({'kind':'chunk','node':node,'channel':channel,'at':time.time(),'data':base64.b64encode(raw).decode()})
    if not sel.get_map(): reason='all_streams_closed'
except BaseException as exc:
    reason=reason if reason!='timeout' else type(exc).__name__
finally:
    for p in children:
        if p.poll() is None:
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for p in children:
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
    emit({'kind':'stream_end','at':time.time(),'reason':reason})
'''


def remote_code(body,deadline):
    return ('session='+repr(SESSION)+'\nticket='+repr(TICKET)+'\nrevision='+repr(REV)+
            '\nsource='+repr(SOURCE)+'\nowner_pid='+repr(OWNER_PID)+'\nowner_start_tick='+repr(OWNER_START_TICK)+'\ndeadline='+repr(deadline)+'\n'+COMMON+body)


def main():
    args=argparse.ArgumentParser(); args.add_argument('--syntax-check',action='store_true'); options=args.parse_args()
    for body in (STATUS,STREAM): ast.parse(remote_code(body,0))
    if options.syntax_check:
        print('outer + status + stream AST PASS; no remote execution'); return
    if len(REV)!=40 or any(c not in '0123456789abcdef' for c in REV) or not TICKET.isdecimal() or not OWNER_PID.isdecimal() or int(OWNER_PID)<=0 or not OWNER_START_TICK.isdecimal():
        args.error('bind actual onepass27 revision, ticket, owner PID and start tick before observing')
    os.umask(0o077); ROOT.mkdir(mode=0o700,exist_ok=False)
    (ROOT/'observer.pid').write_text(str(os.getpid())+'\n')
    (ROOT/'README.private.txt').write_text('All stream bytes are UNASSIGNED raw evidence from the owned reservation window. They are not attributed to B/A by arrival time. Use strict immutable snapshots and container start/source identities; tail stderr preserves truncation/replacement notices. This observer issues no HTTP/GPU requests.\n')
    events=(ROOT/'events.jsonl').open('x',buffering=1)
    lock=threading.Lock(); stopped=threading.Event(); processes=[]; tasks=[]; deadline=time.time()+3*3600
    def event(value):
        with lock: events.write(json.dumps(dict(value,local_at=time.time()))+'\n')
    def stop(signum=None,frame=None): stopped.set()
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def stream_reader(p):
        handles={}
        try:
            for line in p.stdout:
                value=json.loads(line)
                if value['kind']=='chunk':
                    key=(value['node'],value['channel'])
                    if key not in handles: handles[key]=(ROOT/(key[0]+'.'+key[1]+'.raw')).open('xb')
                    raw=base64.b64decode(value.pop('data'),validate=True); handle=handles[key]
                    value.update(offset=handle.tell(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),attribution='UNASSIGNED')
                    handle.write(raw); handle.flush()
                event(value)
        except Exception as exc: event({'kind':'stream_reader_error','error':type(exc).__name__})
        finally:
            for h in handles.values(): h.close()
    def snapshot(arm, head):
        cmd=[sys.executable,str(SNAPSHOT),'--arm',arm,'--revision',REV,'--session',SESSION,'--suffix','observer',
             '--head-id',head['id'],'--head-started-at',head['started_at'],
             '--arm-event-key',head['arm_event']['key']]
        with (ROOT/(arm+'.snapshot.private.log')).open('xb') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); processes.append(p)
            try: rc=p.wait(timeout=240)
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGTERM); rc=p.wait(timeout=10)
        event({'kind':'strict_snapshot_finished','arm':arm,'returncode':rc,'path':'/tmp/glm53-onepass27-live-'+arm+'-observer'})
    event({'kind':'observer_start','pid':os.getpid(),'session':SESSION,'ticket':TICKET,'revision':REV,'owner_start_tick':OWNER_START_TICK,'deadline':deadline})
    seen_go=False; attempted=set(); old_state=None
    try:
        while not stopped.is_set() and time.time()<deadline:
            try:
                p=subprocess.run(SSH,input=remote_code(STATUS,deadline).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
                if p.returncode: raise RuntimeError('status unavailable')
                status=json.loads(p.stdout)
            except Exception as exc:
                event({'kind':'status_error','error':type(exc).__name__}); stopped.wait(15); continue
            compact={k:status[k] for k in ('own','go','terminal','phase','state','payload_returncode','returncode')}
            if compact!=old_state: event({'kind':'status','status':status}); old_state=compact
            if status['terminal'] or(seen_go and not status['own']): event({'kind':'observer_stop','reason':'terminal_or_release'}); break
            if status['own'] and status['go'] and not seen_go:
                seen_go=True
                err=(ROOT/'stream-ssh.private.log').open('xb')
                stream=subprocess.Popen(SSH,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,start_new_session=True)
                processes.append(stream); stream.stdin.write(remote_code(STREAM,deadline).encode()); stream.stdin.close()
                t=threading.Thread(target=stream_reader,args=(stream,),daemon=True); t.start(); tasks.append(t)
                event({'kind':'owned_go_streams_requested','pid':stream.pid})
            head=status.get('head')
            if seen_go and head and head['manifest_exact'] and head['stable'] and head['running'] and head['born_after_arm'] and not head['duplicate_env']:
                flags=head['flags'];arm=head['arm_event']['arm']
                expected='1' if arm=='A' else '0'; eligible_arm=arm in ('B','A')
                eligible=eligible_arm and all(flags[key]=='0' for key in ('VLLM_GLM53_EP_PREFILL_LOCAL','VLLM_B12X_EP_WARM_COMPACT','VLLM_B12X_EP_ZERO_WEIGHT_MICRO'))
                eligible=eligible and flags['VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE']=='1' and flags['VLLM_GLM53_TP_SF6_Q0']==expected
                eligible=eligible and head['tp_only'] and head['image_exact'] and head['graph_finished']
                eligible=eligible and head['mm_limit']=={'image':4,'video':0}
                eligible=eligible and flags['VLLM_GLM53_STARTUP_TRIM']=='1' and head['startup_trim_complete']
                eligible=eligible and (arm=='B' or head['tp_sf6_q0_pass'])
                # The canonical event and a container born after it identify
                # this arm; B/A name is checked against the owned canonical event.
                if eligible and arm not in attempted:
                    attempted.add(arm); event({'kind':'strict_snapshot_requested','arm':arm,'head':head})
                    t=threading.Thread(target=snapshot,args=(arm,head),daemon=True); t.start(); tasks.append(t)
            stopped.wait(30)
    finally:
        stopped.set()
        for p in processes:
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        for p in processes:
            try: p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try: os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                p.wait()
        for task in tasks: task.join(timeout=2)
        event({'kind':'observer_finished','attempted_arms':sorted(attempted),'owned_go_seen':seen_go})
        events.close()


if __name__=='__main__': main()
