#!/usr/bin/env python3
"""Optional collector for GPU26's pre-hold freshness refusal. Preparation only;
when explicitly run it reads exact existing files and writes one fresh archive.
No fleet command, workload, HTTP, GPU, signal, queue, or serving mutation.
"""
import base64,gzip,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass26-prehold-refused'
SESSION='eplocalonepass0909v26';PID=713884
REV='055914aeb719c1769e05cdb863e43a88b2ee47af'
RECEIPT_SHA='148497d17dfcd651a4a0d846859ecf4e37ec35adfec051c31c4e81a33104d749'
CONFIG_SHA='0ceb9ed4292039f64b024e3a0a23f77b5245c27f1752cbd8364f0328b9b1f778'
LOG='/home/choiceoh/glm53-logs/fleet/launches/fa96764ce2a3164798f8492c334bcd8f6d232bf90fa809333b9dd547f0897778.ce29b8fa4eb24a3085c73af39a8ef7b1.log'
REFUSAL='PREPARE REFUSED (no GPU hold): candidate must include relevant changes from 396675e62d5734e07412624f2b1a53778adef167:'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
 p=Path(path);assert p.is_file() and not p.is_symlink();a=p.stat();assert a.st_size<16*2**20;raw=p.read_bytes();b=p.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns);return raw
REMOTE=r'''
import base64,hashlib,json,pathlib
P=pathlib.Path;job=P('/tmp/glm53-ep-onepass-0909-26');out=dict(files={},absent=[])
for p in [job/x for x in ('submission.json','submit.exit.json','submit.stdout','submit.stderr','cpu24-reuse.json','onepass.jsonl','verdicts.jsonl')]+[P(log)]:
 if not p.exists():out['absent'].append(str(p));continue
 assert p.is_file() and not p.is_symlink();a=p.stat();assert a.st_size<16*2**20;raw=p.read_bytes();b=p.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
 out['files'][str(p)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode())
print(json.dumps(out))
'''
def main():
 assert not OUT.exists(),'refuse existing archive'
 receipt_raw=read('/tmp/glm53-onepass26-submit-receipt.txt');assert sha(receipt_raw)==RECEIPT_SHA;receipt=json.loads(receipt_raw)
 assert receipt['session']==SESSION and receipt['pid']==PID and receipt['accepted'] is False
 assert receipt['state']=='startup-failed' and receipt['returncode']==3 and not receipt.get('ticket')
 assert receipt['startup_log']==receipt['log_path']==LOG and REFUSAL in receipt['log_tail']
 config_raw=read('/tmp/glm53-onepass26-submit.json');assert sha(config_raw)==CONFIG_SHA;cfg=json.loads(config_raw)
 assert cfg['revision']==REV and cfg['source']=='/home/choiceoh/stkernel-ep-onepass-0909-25'
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','choiceoh@srv2','python3','-B','-'],input=('log='+repr(LOG)+'\n'+REMOTE).encode(),capture_output=True,timeout=30)
 assert p.returncode==0,p.stderr.decode(errors='replace')[:2000];remote=json.loads(p.stdout);items={};provenance={}
 def save(name,raw,origin,compress=False):
  assert name not in items;stored=gzip.compress(raw,mtime=0) if compress else raw
  items[name]=stored;provenance[name]=dict(origin=origin,bytes=len(raw),sha256=sha(raw),stored_bytes=len(stored),stored_sha256=sha(stored))
 original={}
 for path,d in remote['files'].items():
  raw=base64.b64decode(d.pop('data'),validate=True);assert sha(raw)==d['sha256'] and len(raw)==d['bytes'];original[path]=raw
  if path==LOG:save('startup-refusal.log.gz',raw,path,True)
  else:save('job/'+Path(path).name,raw,path)
 assert REFUSAL.encode() in original[LOG]
 job='/tmp/glm53-ep-onepass-0909-26/'
 assert original[job+'submit.stdout']+original[job+'submit.stderr']==receipt_raw
 assert json.loads(original[job+'submit.exit.json'])['returncode']==3
 for name in ('onepass.jsonl','verdicts.jsonl'):assert not original.get(job+name,b'').strip(),'unexpected GPU26 result; refuse pre-hold-only classification'
 save('submission/request.json',config_raw,'/tmp/glm53-onepass26-submit.json')
 save('submission/receipt.json',receipt_raw,'/tmp/glm53-onepass26-submit-receipt.txt')
 summary=dict(schema=1,verdict='PRE_HOLD_SOURCE_FRESHNESS_REFUSAL',session=SESSION,supervisor_pid=PID,revision=REV,accepted=False,accepted_ticket=None,returncode=3,canonical_records=0,source=cfg['source'],new_cpu_compile=False,reason=REFUSAL,required_main_revision='396675e62d5734e07412624f2b1a53778adef167',scope='No GPU hold is the explicit startup admission refusal. No GPU numerics, serving result, throughput, recovery, or candidate-regression verdict exists for26.',absent=remote['absent'])
 save('result-summary.json',(json.dumps(summary,indent=2,sort_keys=True)+'\n').encode(),'original admission receipt and exact startup log')
 save('README.md',('# onepass26 refused before GPU hold\n\nThe normal freshness guard rejected source `'+REV+'` before admission: required main `396675e62d5734e07412624f2b1a53778adef167` startup/audit changes were absent. The launch response is `accepted=false`, startup-failed, return code3, and has no accepted ticket. The exact refusal log explicitly says `no GPU hold`.\n\nNo canonical measurement record or GPU26/CPU26 result is invented. Source25 and original CPU24 reuse provenance remain their original identities. The new matched B/A on source27 is separate work. This archive retains original submit/config/job/startup bytes and missing paths; it does not attribute a performance or numerical failure to the candidate.\n').encode(),'bounded refusal scope')
 save('collect-evidence.py',read(__file__),'this optional collector')
 items['originals.json']=(json.dumps(provenance,sort_keys=True,indent=2)+'\n').encode()
 items['SHA256SUMS']=''.join(sha(raw)+'  '+name+'\n' for name,raw in sorted(items.items())).encode()
 assert not OUT.exists();OUT.mkdir()
 for name,raw in items.items():
  p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);assert read(p)==raw
 print(json.dumps(dict(archive=str(OUT),files=len(items),sums_sha256=sha(items['SHA256SUMS']))))
if __name__=='__main__':main()
