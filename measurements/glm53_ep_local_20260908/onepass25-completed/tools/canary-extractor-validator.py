#!/usr/bin/env python3
"""Offline TP Q0 receipt extraction for the sole candidate A of GPU25.

B0 and B1 have Q0 disabled and produce no TP candidate receipt. Occurrence0
is expected to belong to A; strict snapshot proof is required separately.
Original CPU24 is reused only when every actual mounted and contract source
matches its immutable source objects and the new frozen source exactly.
No CUDA, SSH, HTTP or new test execution occurs in this extractor.
"""
import argparse, ast, hashlib, json, math, os, re, struct, subprocess
from pathlib import Path

ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
REV='055914aeb719c1769e05cdb863e43a88b2ee47af'
CPU_SHA='1d96612938f200bd86d078e1d8841cc92df808ba4ab802ca51306fe229b67506'
CPU_REV='82ac3c34173ae63b3dd0a42c49f8421097e96a1a'
REUSE_SHA='8b27f9505d18724d295977b08a4a942e2b3bf203b5904bbcca8926b8a7b636a7'
EXPECTED_TESTS=165
NODES=('local','10.10.10.1','10.10.10.3','10.10.10.4')
CASES=(('balanced4096',4096),('concentrated6912',6912),('zeros4097',4097),('duplicate8192',8192))
PHASES=[p+'-'+c for p in ('initial','changed') for c in ('C1-eager','C2-graph-current','C3-graph-side')]
FILES=dict(selftest='glm53_tp_sf6_q0_selftest.py',numerical='glm53_ep_local_selftest.py',
 dispatch='moe_dispatch.py',candidate='moe_dynamic_gated_sf6_q0.py',sf6='moe_dynamic_gated_sf6.py',
 wrapper='flashinfer_b12x_moe.py')
STOCK_SHA='993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445'
MARKER=b'[tp-sf6-q0-selftest] '

def sha(raw): return hashlib.sha256(raw).hexdigest()
def require(ok, message):
 if not ok: raise ValueError(message)
def ishash(x): return isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None
def plain(path):
 path=Path(path)
 require(path.is_file() and not path.is_symlink(),f'not a regular file: {path}')
 return path.read_bytes()
def no_errors(value):
 if isinstance(value,dict):
  for key,item in value.items():
   require(not (key in ('error','cleanup_error','candidate_first_failure') or key.endswith('_error')),f'failure key: {key}')
   no_errors(item)
 elif isinstance(value,list):
  for item in value: no_errors(item)
def parse_cache_key(value):
 require(isinstance(value,str) and len(value)<=8192,'invalid cache key text')
 class DtypeLiteral(ast.NodeTransformer):
  def visit_Attribute(self,node):
   require(isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='int32',
           'unsupported cache key attribute')
   return ast.copy_location(ast.Constant(value='torch.int32'),node)
  def visit_Call(self,node):
   raise ValueError('cache key calls are forbidden')
 tree=DtypeLiteral().visit(ast.parse(value,mode='eval'))
 result=ast.literal_eval(tree)
 require(isinstance(result,tuple),'cache key is not a tuple')
 return result

def identity(value):
 require(isinstance(value,dict) and set(value)=={'shape','dtype','sha256','data_ptr'},'tensor identity fields')
 require(isinstance(value['shape'],list) and all(type(n)is int and n>0 for n in value['shape']),'tensor shape')
 require(isinstance(value['dtype'],str) and value['dtype'].startswith('torch.'),'tensor dtype')
 require(ishash(value['sha256']) and type(value['data_ptr'])is int and value['data_ptr']>0,'tensor hash/pointer')
 return value
def content(value): return {k:value[k] for k in ('shape','dtype','sha256')}
def metrics(value,control=False):
 require(value.get('bad_rows')==0 and type(value['bad_rows'])is int,'nonzero/missing bad rows')
 for key in ('max_row_relative_l2','max_row_relative_abs'):
  require(type(value.get(key))in (int,float) and math.isfinite(value[key]) and value[key]>=0,'invalid numeric metric')
 if control:
  require(value['max_row_relative_l2']<=struct.unpack('<f',struct.pack('<f',.02))[0] and value['max_row_relative_abs']<=struct.unpack('<f',struct.pack('<f',.04))[0],'stock control exceeds original floors')
 else:
  for key,limit in (('stock_max_row_relative_l2',.02),('stock_max_row_relative_abs',.04)):
   require(type(value.get(key))in (int,float) and math.isfinite(value[key]) and 0<=value[key]<=struct.unpack('<f',struct.pack('<f',limit))[0],'invalid candidate control noise')
  # Necessary global bound only. Exact per-row acceptance is the source-bound
  # comparison's bad_rows field, not reconstructable from aggregate maxima.
  require(value['max_row_relative_l2']<=max(.02,3*value['stock_max_row_relative_l2'])+1e-7,'candidate L2 global necessary bound')
  require(value['max_row_relative_abs']<=max(.04,3*value['stock_max_row_relative_abs'])+1e-7,'candidate peak global necessary bound')

def expected_source(args):
 raw=plain(args.cpu_result);require(sha(raw)==CPU_SHA,'CPU24 result hash changed')
 cpu=json.loads(raw)
 require(cpu['verdict']=='PASS' and cpu['phase']=='complete' and cpu['binding_runtime_rechecked'] is True
         and cpu['cuda_initialized'] is False,'CPU24 incomplete')
 require(cpu['contracts']==dict(tests_run=EXPECTED_TESTS,failures=0,errors=0,skips=0),'CPU24 contract counts')
 require([x['arm'] for x in cpu['tp_sf6_passes']]==['stock','q0-cache'],'CPU24 TP compile pair missing')
 reuse_raw=plain(args.cpu_reuse_receipt);require(sha(reuse_raw)==REUSE_SHA,'CPU24 reuse receipt hash changed')
 reuse=json.loads(reuse_raw)
 require(reuse['schema']==1 and reuse['mode']=='ORIGINAL_CPU24_EXACT_SOURCE_REUSE' and reuse['new_revision']==REV
         and reuse['new_source']=='/home/choiceoh/stkernel-ep-onepass-0909-25'
         and reuse['mounted_and_contract_sources_equal'] is True and reuse['artifact_and_runtime_validation'] is True
         and reuse['new_cpu_compile'] is False,'incorrect CPU24 reuse admission')
 origin=reuse['cpu_origin'];require(origin['revision']==CPU_REV and origin['result_sha256']==CPU_SHA
         and origin['source']=='/home/choiceoh/stkernel-ep-onepass-0909-24'
         and origin['job']=='/tmp/glm53-ep-decode-cpu-0909-24'
         and origin['archive_result']=='measurements/glm53_ep_local_20260908/decode24-cpu/result.json','CPU origin mismatch')
 for key,name in (('submission_sha256','submission.json'),('exit_sha256','exit.json')):
  require(origin[key]==sha(plain(ROOT/'measurements/glm53_ep_local_20260908/decode24-cpu/head'/name)),'original CPU24 head receipt mismatch')
 for revision in (CPU_REV,REV):
  def gitbytes(path):return subprocess.check_output(['git','-C',str(args.repo),'show',revision+':'+path])
  require({path:sha(gitbytes(path)) for path in cpu['contract_sources']}==cpu['contract_sources'],'reused contract sources changed: '+revision)
  found={}
  for line in gitbytes('build/glm53/manifest.tsv').decode().splitlines():
   if not line or line.startswith('#'):continue
   name,target,*_=line.split('\t')
   if target in cpu['mounted_sources']:
    require(target not in found,'duplicate mounted source')
    found[target]=sha(gitbytes('build/glm53/'+name))
  require(found==cpu['mounted_sources'],'reused mounted sources changed: '+revision)

 result={}
 for key,filename in FILES.items():
  relative='overlay/modules/glm53_moe/'+filename
  data=subprocess.check_output(['git','-C',str(args.repo),'show',args.revision+':'+relative])
  found=[(path,digest) for path,digest in cpu['mounted_sources'].items() if Path(path).name==filename]
  require(len(found)==1 and found[0][1]==sha(data),'frozen source differs from CPU24: '+key)
  result[key]=dict(path=found[0][0],sha256=found[0][1])
 result['stock']=dict(path='/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py',sha256=STOCK_SHA)
 sf6=subprocess.check_output(['git','-C',str(args.repo),'show',args.revision+':overlay/modules/glm53_moe/moe_dynamic_gated_sf6.py'])
 tree=ast.parse(sf6)
 pinned=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STOCK_GATED_SHA256' for t in n.targets))
 require(pinned==STOCK_SHA,'inherited stock pin changed')
 return result,cpu

def extract(path,occurrence):
 require(path.is_file() and not path.is_symlink(),'stream missing/unsafe: '+str(path))
 prefix=hashlib.sha256();offset=0;matches=[]
 with path.open('rb') as stream:
  stat=os.fstat(stream.fileno())
  require(stat.st_size<=512*1024*1024,'unexpectedly large observer stream')
  remaining=stat.st_size
  while remaining:
   line=stream.readline(min(remaining,4*1024*1024+1));remaining-=len(line)
   require(len(line)<=4*1024*1024,'oversized stream line')
   prefix.update(line)
   if MARKER in line:
    require(line.endswith(b'\n'),'partial TP receipt line; retry after publication completes')
    state,separator,payload=line.split(MARKER,1)[1].strip().partition(b' ')
    require(separator and state in (b'PASS',b'FAIL'),'unrecognized TP receipt marker')
    receipt=json.loads(payload)
    require(receipt.get('verdict')==state.decode(),'marker/verdict mismatch')
    matches.append((payload,receipt,dict(stream=str(path),stream_device=stat.st_dev,stream_inode=stat.st_ino,
      raw_offset=offset,raw_end=offset+len(line),line_sha256=sha(line),prefix_sha256=prefix.hexdigest()),line))
   offset+=len(line)
 require(len(matches)>occurrence,'selected TP receipt is not yet present')
 payload,receipt,provenance,line=matches[occurrence]
 # Appends are allowed, but the selected original bytes/prefix cannot drift.
 with path.open('rb') as stream:
  stat2=os.fstat(stream.fileno());require((stat.st_dev,stat.st_ino)==(stat2.st_dev,stat2.st_ino),'observer stream replaced')
  require(sha(stream.read(provenance['raw_end']))==provenance['prefix_sha256'],'observer prefix changed')
 provenance.update(occurrence=occurrence,complete_receipts_seen=len(matches),json_sha256=sha(payload))
 return payload,receipt,provenance,line

def validate(receipt,expected):
 require(receipt.get('schema')==1 and receipt.get('verdict')=='PASS' and receipt.get('phase')=='complete','canary not complete PASS')
 no_errors(receipt)
 for key in ('actual_packed_owner','caller_preserved'): require(receipt.get(key)is True,'missing '+key)
 for key in ('performance_acceptance','full_sanitizer_acceptance'): require(receipt.get(key)is False,'unsupported acceptance claim')
 require(receipt['geometry']==dict(E=288,K=4096,I=512,top8=8),'TP geometry')
 require(receipt['source']['source']==expected,'CPU/frozen/runtime source mismatch')
 versions=receipt['source']['versions']
 require(set(versions)=={'torch','flashinfer','cuda.bindings'},'runtime version set')
 for name,value in versions.items():
  require(isinstance(value.get('version'),str) and value['version'] not in ('','unknown'),'runtime version missing')
  require(isinstance(value.get('path'),str) and value['path'].startswith('/usr/local/lib/python3.12/dist-packages/')
          and ishash(value.get('sha256')),'runtime module identity missing')
 require(versions['cuda.bindings']['version']=='13.3.1','serving bindings version changed')
 require(versions['cuda.bindings'].get('metadata_path')=='/usr/local/lib/python3.12/dist-packages/cuda_bindings-13.3.1.dist-info/METADATA'
         and ishash(versions['cuda.bindings'].get('metadata_sha256')),'bindings metadata identity missing')
 before,after=receipt['weights_before'],receipt['weights_after']
 names={'w13','w2','sf1','sf2','fc1_alpha','fc2_alpha','fc1_input','fc2_input'}
 require(set(before)==names and before==after,'weight backing identity/content changed')
 caller=receipt['caller_state'];require(set(caller['tensors'])==names,'caller tensor set')
 require(caller['raw_parameters']==[None,None],'raw packed scales retained')
 for name,value in before.items():
  identity(value);state=caller['tensors'][name]
  require(state[1]==value['data_ptr'] and state[3]==value['shape'] and state[5]==value['dtype'],'caller backing disagreement: '+name)
 require(before['w13']['shape']==[288,1024,2048] and before['w2']['shape']==[288,4096,256],'actual packed weights shape')
 require(len(receipt['cases'])==4,'not all four TP fixtures')
 for case,(name,rows) in zip(receipt['cases'],CASES):
  require(case['case']==name and case['rows']==rows and case['verdict']=='PASS' and case['phase']=='complete'
          and case['graph_replay']is True,'case completion/identity')
  require(receipt['started_at']<=case['started_at']<=case['completed_at']<=receipt['completed_at'],'case timestamps')
  require(len(case['controls'])==2 and all(len(group)==3 for group in case['controls']),'B1/B2/B3 control coverage')
  for group in case['controls']:
   for item in group: metrics(item,True)
  require([x['phase'] for x in case['candidate']]==PHASES,'candidate phase coverage')
  for item in case['candidate']: metrics(item)
  require([x['phase'] for x in case['q0']]==PHASES,'Q0 phase coverage')
  for item in case['q0']:
   require(item['routes']==rows*8 and ishash(item['sha256']) and item['scope']=='all route IDs/weights/counts/prefixes; A/SFA bytes only for tokens 0,T//2,T-1','Q0 route/sample scope')
  for start in (0,3): require(len({x['sha256'] for x in case['q0'][start:start+3]})==1,'same-phase Q0 bytes differ')
  require(len(case['inputs'])==2,'initial/changed input identities')
  initial,changed=case['inputs'];inputnames={'X','ids','weights','fc1_input','fc2_input','fc1_alpha','fc2_alpha'}
  require(set(initial)==inputnames and set(changed)==inputnames,'input names')
  for key in inputnames:
   a,b=identity(initial[key]),identity(changed[key])
   require(a['data_ptr']==b['data_ptr'] and a['shape']==b['shape'] and a['dtype']==b['dtype'],'changed storage/metadata replaced: '+key)
   if key in ('X','ids','weights','fc1_input'): require(a['sha256']!=b['sha256'],'changed fixture unchanged: '+key)
   else: require(a==b,'immutable input changed: '+key)
  for key,shape,dtype in (('X',[rows,4096],'torch.bfloat16'),('ids',[rows,8],'torch.int32'),('weights',[rows,8],'torch.float32')):
   require(initial[key]['shape']==shape and initial[key]['dtype']==dtype,'input shape/dtype: '+key)
  for key in ('fc1_alpha','fc2_alpha'):require(initial[key]==before[key],'actual alpha backing changed')
 pairs=receipt['cache_pairs'];require(bool(pairs),'missing compiled keys')
 for pair in pairs:
  base=parse_cache_key(pair['baseline']);candidate=parse_cache_key(pair['candidate'])
  require(base[:7]==('dynamic','fp4','nvfp4',288,4096,512,8) and base[-1]=='sf6_direct_prefill_v1'
          and candidate==base+('glm53_tp_sf6_q0_v1',),'compiled Q0/stock key separation')
 return dict(cases=4,control_comparisons=24,candidate_comparisons=24,q0_comparisons=24,
             weights_preserved=True,inputs_same_address=True,source_matches_cpu24=True,
             started_at=receipt['started_at'],completed_at=receipt['completed_at'],versions=versions)

def continuity(new,old):
 differences=[]
 require(set(old)==set(new)==set(NODES),'within-rank comparison requires all ranks')
 for node in NODES:
  a,b=old[node],new[node]
  if a['source']!=b['source']:differences.append(node+':source/runtime')
  for key in a['weights_before']:
   if content(a['weights_before'][key])!=content(b['weights_before'][key]):differences.append(node+':weight:'+key)
  for ac,bc in zip(a['cases'],b['cases']):
   for phase,(ai,bi) in enumerate(zip(ac['inputs'],bc['inputs'])):
    for key in ai:
     if content(ai[key])!=content(bi[key]):differences.append(node+':'+ac['case']+':'+str(phase)+':'+key)
 return dict(match=not differences,differences=differences[:32],difference_count=len(differences),
             scope='same rank weight/input dtype/shape/hash; cross-boot base addresses may differ')

def save(path,raw):
 if path.exists(): require(plain(path)==raw,'refuse overwriting changed evidence: '+str(path));return
 with path.open('xb') as handle:handle.write(raw)
 os.chmod(path,0o600)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--streams',type=Path,default=Path('/tmp/glm53-onepass25-streams'))
 parser.add_argument('--output',type=Path,default=Path('/tmp/glm53-onepass25-canary-A'))
 parser.add_argument('--cpu-result',type=Path,default=ROOT/'measurements/glm53_ep_local_20260908/decode24-cpu/result.json')
 parser.add_argument('--cpu-reuse-receipt',type=Path,default=Path('/tmp/glm53-onepass25-cpu24-reuse.json'))
 parser.add_argument('--repo',type=Path,default=ROOT);parser.add_argument('--revision',default=REV)
 parser.add_argument('--occurrence',type=int,default=0);parser.add_argument('--compare-to',type=Path)
 args=parser.parse_args();require(re.fullmatch('[0-9a-f]{40}',REV) and ishash(CPU_SHA) and type(EXPECTED_TESTS)is int and EXPECTED_TESTS>0,'bind actual CPU24 revision, result SHA and test count first');require(args.revision==REV and args.occurrence==0,'wrong revision/occurrence')
 require(args.output.parent.resolve()==Path('/tmp').resolve() and not args.output.is_symlink(),'output must be one private /tmp directory')
 expected,cpu=expected_source(args)
 report=dict(schema=1,revision=args.revision,cpu24_result_sha256=CPU_SHA,cpu24_source_revision=CPU_REV,cpu24_reuse_sha256=REUSE_SHA,expected_arm='A',occurrence=args.occurrence,
   verdict='NOT_VALIDATED',arm_container_proof=False,strict_snapshot_required_separately=True,
   performance_acceptance=False,full_sanitizer_acceptance=False,nodes={},
   runtime_scope=dict(cpu_bindings=cpu['binding_runtime']['binding_identity']['version'],serving_bindings='13.3.1',
    cpu_serving_runtime_equal=False,serving_runtime_four_rank_comparison=True,
    cuda_python_paired_metadata_recorded=False))
 extracted={};receipts={}
 for node in NODES:
  try:
   payload,receipt,provenance,line=extract(args.streams/(node+'.stdout.raw'),args.occurrence)
   extracted[node]=(payload,line);receipts[node]=receipt
   report['nodes'][node]=dict(provenance=provenance,receipt_verdict=receipt.get('verdict'),**validate(receipt,expected))
  except (ValueError,KeyError,TypeError,IndexError,StopIteration) as exc:
   report['nodes'][node]=dict(validation_error=str(exc)[:500])
 complete=len(receipts)==4 and all('validation_error' not in v for v in report['nodes'].values())
 if complete:
  versions=[r['source']['versions'] for r in receipts.values()]
  if any(value!=versions[0] for value in versions[1:]):
   complete=False;report['runtime_error']='serving module/metadata identities differ across ranks'
 if complete and args.compare_to:
  old={node:json.loads(plain(args.compare_to/(node+'.json'))) for node in NODES}
  for receipt in old.values():validate(receipt,expected)
  report['within_rank_continuity']=continuity(receipts,old)
  complete=report['within_rank_continuity']['match']
 report['verdict']='RAW_CANARY_SOURCE_VALIDATED' if complete else 'INCOMPLETE_OR_REJECTED'
 # Incomplete/failed attempts retain their exact available receipts too; a new
 # invocation must choose a fresh output suffix instead of revising a report.
 args.output.mkdir(mode=0o700,exist_ok=True);os.chmod(args.output,0o700)
 for node,(payload,line) in extracted.items():
  save(args.output/(node+'.json'),payload+b'\n');save(args.output/(node+'.receipt-line.raw'),line)
 save(args.output/'validation.json',(json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
 print(json.dumps(dict(verdict=report['verdict'],output=str(args.output),nodes={k:{x:v[x] for x in ('validation_error','cases','candidate_comparisons','q0_comparisons') if x in v} for k,v in report['nodes'].items()},arm_container_proof=False),sort_keys=True))
 return 0 if complete else 2
if __name__=='__main__':
 try:raise SystemExit(main())
 except (ValueError,OSError,subprocess.CalledProcessError) as exc:
  print(json.dumps(dict(verdict='REJECTED',error=str(exc)[:800]),sort_keys=True));raise SystemExit(2)
