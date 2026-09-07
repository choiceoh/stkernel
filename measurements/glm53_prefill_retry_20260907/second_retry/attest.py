import concurrent.futures,datetime,hashlib,json,os,pathlib,re,shlex,subprocess,sys
job=pathlib.Path(os.environ['RETRY_JOB']);name=sys.argv[1];rev=os.environ['RETRY_REV'];repo=pathlib.Path(os.environ['REPO'])
script=r'''
import base64,datetime,hashlib,json,pathlib,re,subprocess
name=next(n for n in subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines() if n.startswith('glm53'))
c=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
env=dict(e.split('=',1) for e in c['Config']['Env'] if '=' in e)
cmd=' '.join(c['Config'].get('Cmd') or [])
payload=re.search(r'echo ([A-Za-z0-9+/=]+) \| base64 -d',cmd)
if payload:cmd=base64.b64decode(payload.group(1),validate=True).decode()
args={}
for k in ['max-model-len','num-gpu-blocks-override','gpu-memory-utilization','max-num-batched-tokens','max-num-seqs','host','port']:
 m=re.search(r'--'+k+r'(?:=|\s+)([^\s]+)',cmd);args[k]=m.group(1) if m else None
mounts={m['Destination']: {'source':m['Source'],'sha256':hashlib.sha256(pathlib.Path(m['Source']).read_bytes()).hexdigest()} for m in c['Mounts'] if m['Source'].startswith('/home/choiceoh/overlays/glm53/')}
mem={l.split(':')[0]:int(l.split()[1]) for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.split(':')[0] in ['MemTotal','MemAvailable']}
print(json.dumps({'t':datetime.datetime.now().isoformat(),'container':name,'id':c['Id'],'image':c['Image'],'started_at':c['State']['StartedAt'],'args':args,'memory_kib':mem,'env':{k:v for k,v in env.items() if k.startswith('VLLM_GLM53_PREFILL_SP') or k=='VLLM_GLM53_NVFP4_STATIC_SCALE'},'mounts':mounts,'manifest_sha':hashlib.sha256(pathlib.Path('/home/choiceoh/overlays/glm53/manifest.tsv').read_bytes()).hexdigest(), 'stamp':(pathlib.Path('/home/choiceoh/glm53-cache/.overlay-sha').read_text().strip() if pathlib.Path('/home/choiceoh/glm53-cache/.overlay-sha').exists() else None)}))
'''
def one(ip):
 cmd=['python3','-c',script] if ip=='10.10.10.2' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=3','choiceoh@'+ip,'python3 -c '+shlex.quote(script)]
 raw=subprocess.check_output(cmd,text=True,timeout=15)
 return ip,json.loads(raw)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: nodes=dict(pool.map(one,['10.10.10.2','10.10.10.1','10.10.10.3','10.10.10.4']))
if name=='--preflight':
 print(json.dumps({'nodes':nodes},ensure_ascii=False,indent=2));raise SystemExit(0)
rows=[json.loads(l) for l in pathlib.Path(os.environ['ONEPASS_JSONL']).read_text().splitlines() if l.strip()]
r=next(r for r in reversed(rows) if r['name']==name)
result={'name':name,'source':rev,'record':r,'nodes':nodes,'expected_capacity':{'max_len':262144,'kv_tokens':524288,'blocks':415},'issues':[]}
issues=result['issues']
head=nodes['10.10.10.2']
if r.get('boot_id')!=head['id']+'|'+head['started_at']:issues.append('head boot identity mismatch')
if r['git']!=rev[:7]:issues.append('benchmark source mismatch')
if r.get('evidence_issues') or r.get('traffic',{}).get('issues'):issues.append('invalid traffic evidence')
if r.get('quality')!={'ok':15,'total':15} or r.get('korean',{}).get('dirty')!=0:issues.append('quality gate failed')
if any(p['tok']<=0 for p in r['prefill']) or len(r['prefill'])!=5:issues.append('incomplete prefill ladder')
if len(r.get('requests',[]))!=11 or not r.get('workload',{}).get('require_exclusive'):issues.append('workload contract mismatch')
candidate=name.endswith('A')
expected={'VLLM_GLM53_PREFILL_SP_FUSE_MHC':'1' if candidate else '0','VLLM_GLM53_PREFILL_SP_DIRECT_NCCL':'0','VLLM_GLM53_PREFILL_SP_FP8_AG_MIN_TOKENS':'2048' if candidate else '-1','VLLM_GLM53_PREFILL_SP_FP8_RS_MIN_TOKENS':'4096' if candidate else '-1','VLLM_GLM53_NVFP4_STATIC_SCALE':'16'}
manifest=(repo/'build/glm53/manifest.tsv').read_text()
expected_manifest=hashlib.sha256(('# source_commit='+rev+'\n'+manifest).encode()).hexdigest()
expected_mounts={}
for line in manifest.splitlines():
 if not line or line.startswith('#'):continue
 src,dst,contract=line.split('\t');expected_mounts[dst]=hashlib.sha256((repo/'build/glm53'/src).read_bytes()).hexdigest()
for ip,n in nodes.items():
 if n['image']!=os.environ['IMAGE']:issues.append(ip+': image mismatch')
 if n['args']['host']!='127.0.0.1' or n['args']['port']!='18000':issues.append(ip+': private endpoint mismatch')
 if n['args']['max-model-len']!='262144' or n['args']['num-gpu-blocks-override']!='415':issues.append(ip+': capacity mismatch')
 if any(n['env'].get(k)!=v for k,v in expected.items()):issues.append(ip+': knob mismatch')
 if {k:v['sha256'] for k,v in n['mounts'].items()}!=expected_mounts:issues.append(ip+': overlay fingerprint mismatch')
 if n['manifest_sha']!=expected_manifest or n['manifest_sha'][:12]!=r['overlay']:issues.append(ip+': manifest mismatch')
 if ip=='10.10.10.2' and n['stamp']!=expected_manifest:issues.append(ip+': head cache stamp mismatch')
if candidate and r.get('proof_ok')!='3/3':issues.append('candidate serving proof incomplete')
(job/(name+'.attest.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'name':name,'issues':issues,'quality':r['quality'],'overlay':r['overlay'],'nodes':len(nodes)},ensure_ascii=False))
raise SystemExit(3 if issues else 0)
