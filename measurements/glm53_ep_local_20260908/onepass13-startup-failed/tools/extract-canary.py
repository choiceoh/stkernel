#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
root=Path('/tmp/glm53-onepass13-streams');out=Path('/tmp/glm53-onepass13-canary');out.mkdir(exist_ok=True)
result={}
for node in ('local','10.10.10.1','10.10.10.3','10.10.10.4'):
 p=root/(node+'.stdout.raw')
 if not p.exists():result[node]={'status':'pending'};continue
 raw=p.read_bytes();matches=[];offset=0
 for line in raw.splitlines(keepends=True):
  for state in ('PASS','FAIL'):
   marker=b'[ep-local-selftest] '+state.encode()+b' '
   if marker in line:
    payload=line.split(marker,1)[1].strip()
    try:x=json.loads(payload)
    except json.JSONDecodeError:continue
    assert x['verdict']==state
    matches.append((payload,x,offset))
  offset+=len(line)
 assert len(matches)<=1,(node,'duplicate complete canary receipts')
 if not matches:result[node]={'status':'pending'};continue
 payload,x,offset=matches[0];dest=out/(node+'.json')
 if dest.exists():assert dest.read_bytes()==payload+b'\n'
 else:dest.write_bytes(payload+b'\n')
 result[node]={'status':x['verdict'],'raw_offset':offset,'json_sha256':hashlib.sha256(payload).hexdigest(),'cases':len(x.get('cases',[]))}
source=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908/measurements/glm53_ep_local_20260908/decode13-cpu/result.json')
if source.exists():
 x=json.loads(source.read_text());Path('/tmp/glm53-onepass13-mounted-hashes.json').write_text(json.dumps(x['mounted_sources'],sort_keys=True,indent=2)+'\n')
print(json.dumps(result,sort_keys=True))
