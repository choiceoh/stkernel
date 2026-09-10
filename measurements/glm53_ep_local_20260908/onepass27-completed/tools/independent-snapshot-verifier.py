#!/usr/bin/env python3
"""Offline strict B/A snapshot validation; no snapshot requests or GPU imports."""
import argparse,base64,datetime,gzip,hashlib,importlib.util,json,re,subprocess
from pathlib import Path
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
REV='ea413ac4c39ba3e6e4009c73587b0d536053b4bf';SESSION='eplocalonepass0909v27';TICKET='1788932740730747';PID='730747';START='43512869'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
NODES=('local','10.10.10.1','10.10.10.3','10.10.10.4')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
date=lambda value:datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def plain(path):
 assert path.is_file() and not path.is_symlink();a=path.stat();raw=path.read_bytes();b=path.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==a.st_size
 return raw
def gitfile(path):return subprocess.check_output(['git','-C',str(ROOT),'show',REV+':'+path])
def fixed(c):return {k:c[k] for k in ('Id','Created','Image','Config','HostConfig','RestartCount')}|{'Mounts':sorted(c['Mounts'],key=lambda x:json.dumps(x,sort_keys=True)),'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True,choices=['B','A']);a=p.parse_args()
 private=Path('/tmp/glm53-onepass27-live-'+a.arm+'-observer');dest=Path('/tmp/glm53-onepass27-'+a.arm+'-verified.json');assert not dest.exists()
 raw=plain(private/'identity.json');r=json.loads(raw)
 assert (r['revision'],r['session'],r['ticket'],r['owner_pid'],r['arm'])==(REV,SESSION,TICKET,PID,a.arm) and set(r['nodes'])==set(NODES) and r['owner_start_tick']==START
 helper=Path('/tmp/glm53_extract_onepass27_canary.py');assert sha(plain(helper))=='7e0e31a67f49ef20fca3586e15c8370192d2648c7fbd1ad1fc207d3eb5002cb8'
 spec=importlib.util.spec_from_file_location('strict_canary27',helper);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
 reuse_path=Path('/tmp/glm53-onepass27-cpu24-reuse.json');reuse=json.loads(plain(reuse_path))
 expected_sources,cpu=validator.expected_source(argparse.Namespace(cpu_result=ROOT/'measurements/glm53_ep_local_20260908/decode24-cpu/result.json',cpu_reuse_receipt=reuse_path,repo=ROOT,revision=REV))
 binding=r['cpu24_binding'];assert binding['mode']==reuse['mode'] and binding['reuse_sha256']==sha(plain(reuse_path)) and binding['cpu_origin']==reuse['cpu_origin']
 assert binding['new_revision']==REV and binding['new_source']==reuse['new_source'] and binding['reuse_path']=='/tmp/glm53-ep-onepass-0909-27/cpu24-reuse.json'
 assert binding['new_cpu_compile'] is False and binding['mounted_and_contract_sources_equal'] is True
 candidate=a.arm=='A'
 if candidate:
  cv_path=Path('/tmp/glm53-onepass27-canary-A/validation.json');cv=json.loads(plain(cv_path));assert cv['verdict']=='RAW_CANARY_SOURCE_VALIDATED' and cv['revision']==REV and cv['cpu24_result_sha256']==validator.CPU_SHA and cv['cpu24_reuse_sha256']==validator.REUSE_SHA and cv['occurrence']==0 and cv['expected_arm']=='A'
  bp=Path('/tmp/glm53-onepass27-live-B-observer');br=json.loads(plain(bp/'identity.json'));bv=json.loads(plain(Path('/tmp/glm53-onepass27-B-verified.json')))
  assert bv['revision']==REV and bv['arm']=='B' and bv['verdict']=='STRICT_ARM_SOURCE_MM_TRIM_CANARY_VERIFIED' and bv['snapshot_sha256']==sha(plain(bp/'identity.json'))
 for name,d in r['files'].items():
  stored=plain(private/name);original=gzip.decompress(stored)
  assert (len(stored),sha(stored),len(original),sha(original))==(d['stored_bytes'],d['stored_sha256'],d['original_bytes'],d['original_sha256'])
 manifest=('# source_commit='+REV+'\n').encode()+gitfile('build/glm53/manifest.tsv')
 mounts={line.split('\t')[1]:sha(gitfile('build/glm53/'+line.split('\t')[0])) for line in manifest.decode().splitlines() if line and not line.startswith('#')}
 parser_raw=gitfile('bench/glm53_launch_metadata.py');assert plain(private/'launch-parser.py')==parser_raw and sha(parser_raw)==r['parser_sha256']
 ns={};exec(compile(parser_raw,'frozen-launch-parser','exec'),ns)
 out=dict(verdict='STRICT_ARM_SOURCE_MM_TRIM_CANARY_VERIFIED',revision=REV,arm=a.arm,ticket=TICKET,owner_pid=PID,owner_start_tick=START,cpu24_binding=binding,cpu24_counts={'mounted_sources':len(cpu['mounted_sources']),'contract_sources':len(cpu['contract_sources'])},snapshot_sha256=sha(raw),nodes={},performance_acceptance=False,quality_acceptance=False,scope='Read-only validation of existing strict observer snapshots; raw Env/Cmd/inspect remain private')
 expected={'VLLM_B12X_EP_WARM_COMPACT':'0','VLLM_B12X_EP_ZERO_WEIGHT_MICRO':'0','VLLM_GLM53_EP_PREFILL_LOCAL':'0','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE':'1','VLLM_GLM53_STARTUP_TRIM':'1','VLLM_GLM53_TP_SF6_Q0':'1' if candidate else '0'}
 for rank,node in enumerate(NODES):
  d=r['nodes'][node];before=json.loads(gzip.decompress(plain(private/(node+'.inspect.before.json.gz'))))[0];after=json.loads(gzip.decompress(plain(private/(node+'.inspect.after.json.gz'))))[0]
  assert fixed(before)==fixed(after) and before['State']['Running'] and after['State']['Running']
  assert before['Id']==d['id'] and before['Image']==d['image']==IMAGE and before['Created']==d['created_at'] and before['State']['StartedAt']==d['started_at']
  assert date(d['started_at'])>=date(d['created_at'])>=r['arm_event']['started_at'] and d['source']==dict(manifest_sha256=sha(manifest),mounts=mounts)
  assert all(d['topology'][k]==v for k,v in dict(enabled=False,nnodes=4,tensor_parallel_size=4,node_rank=rank).items())
  env={}
  for item in before['Config']['Env']:
   key,value=item.split('=',1);assert key not in env;env[key]=value
  assert d['flags']==expected and all(env[k]==v for k,v in expected.items()) and d['environment_sha256']==sha(json.dumps(env,sort_keys=True,separators=(',',':')).encode())
  payload=ns['_WRAPPER'].fullmatch(before['Config']['Cmd'][1])[1];script=base64.b64decode(payload,validate=True).decode();line=script[len(ns['_GID_PRELUDE']):].removesuffix('\n');argv=ns['_literal_argv'](line[:-len(ns['_REDIRECTION'])])
  fields={}
  for i,arg in enumerate(argv):
   key,eq,value=arg.partition('=')
   if key in ('--limit-mm-per-prompt','--host','--port'):
    assert key not in fields;fields[key]=value if eq else argv[i+1]
  mm=json.loads(fields['--limit-mm-per-prompt']);assert mm=={'image':4,'video':0} and all(type(v)is int for v in mm.values()) and d['mm_limit']==mm
  if rank==0:assert fields['--host']=='127.0.0.1' and fields['--port']=='18000'
  readiness=d['readiness'];assert readiness['graph_finished'] and readiness['candidate'] is candidate and len(readiness['tp_sf6_q0_pass_records'])==(1 if candidate else 0)
  log=gzip.decompress(plain(private/(node+'.serving.log.gz')))
  assert b'[tp-sf6-q0-selftest] FAIL ' not in log
  candidate_details=None;pair=None;receipt=None
  if candidate:
   marker=b'[tp-sf6-q0-selftest] PASS ';marker_lines=[line for line in log.splitlines() if marker in line];assert len(marker_lines)==1
   receipt_path=Path('/tmp/glm53-onepass27-canary-A')/(node+'.json');receipt_raw=plain(receipt_path);receipt=json.loads(receipt_raw)
   assert json.loads(marker_lines[0].split(marker,1)[1])==receipt and sha(receipt_raw.rstrip(b'\n'))==cv['nodes'][node]['provenance']['json_sha256']
   candidate_details=validator.validate(receipt,expected_sources);candidate_details.pop('versions')
   bpath=bp/(node+'.inspect.before.json.gz');bstored=plain(bpath);bdesc=br['files'][bpath.name];boriginal=gzip.decompress(bstored)
   assert (sha(bstored),len(bstored),sha(boriginal),len(boriginal))==(bdesc['stored_sha256'],bdesc['stored_bytes'],bdesc['original_sha256'],bdesc['original_bytes'])
   bcontainer=json.loads(boriginal)[0];benv={}
   for item in bcontainer['Config']['Env']:
    key,value=item.split('=',1);assert key not in benv;benv[key]=value
   assert set(benv)==set(env)
   diff={key:{'baseline':benv[key],'candidate':env[key]} for key in env if env[key]!=benv[key]}
   assert diff=={'VLLM_GLM53_TP_SF6_Q0':{'baseline':'0','candidate':'1'}}
   assert before['Image']==bcontainer['Image'] and all(before['Config'][key]==bcontainer['Config'][key] for key in ('Cmd','Entrypoint'))
   pair=dict(full_environment_exact_except_Q0=True,environment_differences=diff,full_command_entrypoint_image_exact=True,baseline_container_id=bcontainer['Id'])
  else:assert b'[tp-sf6-q0-selftest] PASS ' not in log
  records=readiness['startup_trim_records'];assert len(records)==1;tr=records[0]['receipt'];lines=[line for line in log.splitlines() if b'[glm53-startup-trim] ' in line]
  assert len(lines)==1 and sha(lines[0])==records[0]['line_sha256'] and json.loads(lines[0].split(b'[glm53-startup-trim] ',1)[1])==tr
  assert tr['verdict']=='COMPLETE' and tr['rank']==rank and tr['measurement_errors']==[]
  assert [x['stage'] for x in tr['stages']]==['synchronize','gc_collect','empty_cache','malloc_trim'] and all(x['status']=='COMPLETE' for x in tr['stages'])
  assert tr['before']['allocated']==tr['after']['allocated'] and date(d['started_at'])<=tr['started_at']<=tr['completed_at']<=d['capture_finished_at']
  graph=re.search(rb'Graph capturing finished in [0-9]+ secs, took ',log);assert graph and graph.start()<log.index(lines[0])
  if candidate:assert date(d['started_at'])<=receipt['started_at']<=receipt['completed_at']<=tr['started_at'] and log.index(marker_lines[0])<graph.start()
  out['nodes'][node]=dict(container_id=d['id'],created_at=d['created_at'],started_at=d['started_at'],image=IMAGE,source_match=True,private_before_after_equal=True,topology=d['topology'],flags=expected,mm_limit=mm,graph_then_trim=True,trim=tr,canary=candidate_details,baseline_pair=pair)
 dest.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
 print(json.dumps(dict(path=str(dest),sha256=sha(plain(dest)),verdict=out['verdict'],ranks=len(out['nodes']))))
if __name__=='__main__':main()
