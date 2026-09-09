#!/usr/bin/env python3
import json,shlex,subprocess,re
REV='055914aeb719c1769e05cdb863e43a88b2ee47af'
TICKET='1788930114553638'
OWNER_PID='553638'
REMOTE=r"""import json,os,re,hashlib,subprocess
from pathlib import Path
session='eplocalonepass0909v25';owner=int(owner_pid);ticket=ticket
fleet=Path('/home/choiceoh/glm53-logs/fleet');job=Path('/tmp/glm53-ep-onepass-0909-25')
pending=json.loads((fleet/'pending'/(hashlib.sha256(session.encode()).hexdigest()+'.json')).read_text())
assert pending['session']==session and str(pending['ticket'])==ticket and pending['pid']==owner
assert pending['repo']==source and pending['cwd']==source
actual=subprocess.check_output(['git','-C',source,'rev-parse','HEAD']).decode().strip();assert actual==revision
assert not subprocess.check_output(['git','-C',source,'status','--porcelain']).strip()
holder=(fleet/'holder').read_text().strip().split('|') if (fleet/'holder').exists() else []
result={k:pending.get(k) for k in ('state','phase','payload_returncode','returncode')};result['own']=holder[:2]==[session,str(owner)]
if result['own']:
 st=(Path('/proc')/str(owner)/'stat').read_bytes().rsplit(b')',1)[1].decode('ascii').split()
 assert st[0]!='Z' and st[19]==pending['start']
result['records']=[]
record=job/'onepass.jsonl'
if record.exists():
 for line in record.read_text().splitlines():
  r=json.loads(line)
  fixed=[q for q in r.get('requests',[]) if q.get('fixed_decode')]
  row={k:r.get(k) for k in ('name','t','git','quality','korean','proof_ok','cold_compile')}
  row['prefill']=[{k:q[k] for k in ('ctx','cold_s','warm_s','cold_tok_s','warm_tok_s')} for q in r.get('prefill',[])]
  row['fixed']=[q.get('decode_tok_s') for q in fixed]
  row['pooled_decode_tok_s']=sum(q['completion_tokens']-1 for q in fixed)/sum(q['decode_s'] for q in fixed) if fixed else None
  row['channel_issues']=[]
  for q in r.get('requests',[]):
   d=q.get('channel_diagnostics',{})
   if any(d.get('combined_gated_counts',{}).values()):
    row['channel_issues'].append(dict(ctx=q.get('ctx'),rep=q.get('rep'),question=q.get('question'),first=d.get('first_offending_channel'),counts={k:v['gated_counts'] for k,v in d.get('channels',{}).items()},offenses=d.get('offenses',[])))
  result['records'].append(row)
result['active_legs']=[]
if result['own'] and pending.get('returncode') is None:
 for entry in Path('/proc').iterdir():
  if not entry.name.isdecimal():continue
  try:
   cmd=(entry/'cmdline').read_bytes().split(b'\0')
   if not any(x.endswith(b'/bench/onepass.py') or x==b'bench/onepass.py' for x in cmd):continue
   if (entry/'cwd').resolve()!=Path(source):continue
   child=int(entry.name);chain=[];pid=child
   for _ in range(12):
    st=(Path('/proc')/str(pid)/'stat').read_bytes().rsplit(b')',1)[1].decode('ascii').split()
    if st[0]=='Z':raise ProcessLookupError('process is zombie')
    parent=int(st[1]);chain.append(pid)
    if pid==owner:break
    if parent<=1:break
    pid=parent
   if chain[-1]!=owner:continue
   lever=chain[1];arg=(Path('/proc')/str(lever)/'cmdline').read_bytes()
   if b'ab-lever.sh' not in arg:continue
   leg=Path('/tmp')/('leg.'+str(lever));rows=[]
   if leg.is_file():rows=[x for x in leg.read_text(errors='replace').splitlines() if re.search(r'fixed2K|tok/s|TTFT|prefill|quality|PASS|FAIL|ERROR|^\s+[0-9]+\s+[0-9]+\s+',x)][-12:]
   result['active_legs'].append(dict(pid=child,lever_pid=lever,lines=rows))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
print(json.dumps(result))
"""
if not re.fullmatch('[0-9a-f]{40}',REV) or not TICKET.isdecimal() or not OWNER_PID.isdecimal() or int(OWNER_PID)<=1:raise SystemExit('bind actual onepass25 revision, ticket and owner PID first')
REMOTE='revision='+repr(REV)+'\nsource='+repr('/home/choiceoh/stkernel-ep-onepass-0909-25')+'\nowner_pid='+repr(OWNER_PID)+'\nticket='+repr(TICKET)+'\n'+REMOTE
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2',shlex.join(['python3','-B','-c',REMOTE])],capture_output=True,text=True)
print(p.stdout,end='');print(p.stderr,end='');raise SystemExit(p.returncode)
