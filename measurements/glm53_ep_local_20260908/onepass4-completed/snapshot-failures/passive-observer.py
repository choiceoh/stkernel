#!/usr/bin/env python3
"""Bounded passive evidence observer. Never starts a workload or alters fleet."""
import argparse, ast, base64, hashlib, json, os, signal, subprocess, sys, threading, time
from pathlib import Path

SESSION='eplocalonepass0909v4'
TICKET='17888996582431852'
REV='96cb599d8816ee2585988fc7b750a21e2eb66a0b'
SOURCE='/home/choiceoh/stkernel-ep-onepass-0909-4'
ROOT=Path('/tmp/glm53-onepass4-streams')
SNAPSHOT=Path('/tmp/glm53_onepass4_snapshot.py')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','choiceoh@srv2','python3 -B -']
COMMON=r'''
import hashlib,json,os,pathlib,re,subprocess,time
F=pathlib.Path('/home/choiceoh/glm53-logs/fleet')
def state():
    holder=(F/'holder').read_text().strip().split('|')
    p=F/'pending'/(hashlib.sha256(session.encode()).hexdigest()+'.json')
    pending=json.loads(p.read_text())
    if pending.get('session')!=session or pending.get('ticket')!=ticket: raise RuntimeError('reservation identity changed')
    own=len(holder)>1 and holder[0]==session and holder[1]==str(pending['pid'])
    terminal=pending.get('payload_returncode') is not None or pending.get('returncode') is not None or pending.get('state') in ('finished','failed','cancelled','interrupted')
    with (F/'log').open('rb') as stream:
        stream.seek(0,2); stream.seek(max(0,stream.tell()-2**20)); lines=stream.read().decode(errors='replace')
    go=bool(re.search(r'(?m)^.*\bGO '+re.escape(session)+r' \(pid '+str(pending['pid'])+r'\)(?:\s|$)',lines))
    return {'own':own,'go':go,'terminal':terminal,'phase':pending.get('phase'),'state':pending.get('state'),
            'payload_returncode':pending.get('payload_returncode'),'returncode':pending.get('returncode'),'at':time.time()}
'''
STATUS=r'''
s=state()
if s['own'] and s['go'] and not s['terminal']:
    try:
        c=json.loads(subprocess.check_output(['docker','inspect','glm53'],stderr=subprocess.DEVNULL,timeout=10))[0]
        env={}; duplicate=False
        for item in c['Config']['Env']:
            k,v=item.split('=',1); duplicate=duplicate or k in env; env[k]=v
        manifest=pathlib.Path('/home/choiceoh/overlays/glm53/manifest.tsv').read_bytes()
        frozen=pathlib.Path(source+'/build/glm53/manifest.tsv').read_bytes()
        exact=manifest==b'# source_commit='+revision.encode()+b'\n'+frozen
        born=__import__('datetime').datetime.fromisoformat(c['State']['StartedAt'].replace('Z','+00:00')).timestamp()
        path=pathlib.Path('/home/choiceoh/glm53-logs/glm53.log'); a=path.stat()
        log=''
        if a.st_mtime>=born and a.st_size<=128*2**20: log=path.read_text(errors='replace')
        again=json.loads(subprocess.check_output(['docker','inspect',c['Id']],stderr=subprocess.DEVNULL,timeout=10))[0]
        stable=c['Id']==again['Id'] and c['State']['StartedAt']==again['State']['StartedAt'] and again['State']['Running'] is True
        graph=bool(re.search(r'Graph capturing finished in [0-9]+ secs, took ',log))
        flags={k:env.get(k) for k in ('VLLM_GLM53_EP_PREFILL_LOCAL','VLLM_B12X_EP_WARM_COMPACT','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE')}
        s['head']={'id':c['Id'],'started_at':c['State']['StartedAt'],'running':c['State']['Running'],'stable':stable,
                   'flags':flags,'manifest_exact':exact,'duplicate_env':duplicate,'graph_finished':graph,
                   'compact_warm_complete':'[b12x EP compact warmup] COMPLETE ' in log,
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
            current_holder=(F/'holder').read_text().strip().split('|')
            if len(current_holder)<2 or current_holder[0]!=session or current_holder[1]!=str(json.loads((F/'pending'/(hashlib.sha256(session.encode()).hexdigest()+'.json')).read_text())['pid']):
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
            '\nsource='+repr(SOURCE)+'\ndeadline='+repr(deadline)+'\n'+COMMON+body)


def main():
    args=argparse.ArgumentParser(); args.add_argument('--syntax-check',action='store_true'); options=args.parse_args()
    for body in (STATUS,STREAM): ast.parse(remote_code(body,0))
    if options.syntax_check:
        print('outer + status + stream AST PASS; no remote execution'); return
    os.umask(0o077); ROOT.mkdir(mode=0o700,exist_ok=False)
    (ROOT/'observer.pid').write_text(str(os.getpid())+'\n')
    (ROOT/'README.private.txt').write_text('All stream bytes are UNASSIGNED raw evidence from the owned reservation window. They are not attributed to B1/A/B2 by arrival time. Use strict immutable snapshots and container start/source identities; tail stderr preserves truncation/replacement notices. This observer issues no HTTP/GPU requests.\n')
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
    def snapshot(arm):
        cmd=[sys.executable,str(SNAPSHOT),'--arm',arm,'--revision',REV,'--session',SESSION,'--suffix','observer']
        with (ROOT/(arm+'.snapshot.private.log')).open('xb') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); processes.append(p)
            try: rc=p.wait(timeout=240)
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGTERM); rc=p.wait(timeout=10)
        event({'kind':'strict_snapshot_finished','arm':arm,'returncode':rc,'path':'/tmp/glm53-onepass4-live-'+arm+'-observer'})
    event({'kind':'observer_start','pid':os.getpid(),'session':SESSION,'ticket':TICKET,'revision':REV,'deadline':deadline})
    seen_go=False; seen_a=False; attempted=set(); old_state=None
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
            if seen_go and head and head['manifest_exact'] and head['stable'] and head['running'] and not head['duplicate_env']:
                flags=head['flags']; local=flags['VLLM_GLM53_EP_PREFILL_LOCAL']; warm=flags['VLLM_B12X_EP_WARM_COMPACT']
                if local=='1': seen_a=True
                arm='A' if local=='1' else('B2' if seen_a else 'B1')
                eligible=local in ('0','1') and warm==local and flags['VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE']=='1'
                eligible=eligible and head['graph_finished'] and(local=='0' or head['compact_warm_complete'])
                if eligible and arm not in attempted:
                    attempted.add(arm); event({'kind':'strict_snapshot_requested','arm':arm,'head':head})
                    t=threading.Thread(target=snapshot,args=(arm,),daemon=True); t.start(); tasks.append(t)
            stopped.wait(10 if seen_go else 15)
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
