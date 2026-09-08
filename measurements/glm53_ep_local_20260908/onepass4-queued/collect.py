import base64, datetime, hashlib, json, pathlib, subprocess
ROOT = pathlib.Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT = ROOT / 'measurements/glm53_ep_local_20260908/onepass4-queued'
REV = '96cb599d8816ee2585988fc7b750a21e2eb66a0b'
REMOTE = r'''import base64,datetime,hashlib,json,os,pathlib,subprocess
root=pathlib.Path('/home/choiceoh/stkernel-ep-onepass-0909-4')
job=pathlib.Path('/tmp/glm53-ep-onepass-0909-4')
def run(args):
 p=subprocess.run(args,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
 return {'argv':args,'returncode':p.returncode,'stdout':p.stdout.decode(),'stderr':p.stderr.decode()}
def identity():
 return {'head':run(['git','rev-parse','HEAD']),'status':run(['git','status','--porcelain'])}
def file(path):
 before=path.stat();data=path.read_bytes();after=path.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 return {'path':str(path),'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mtime_ns':after.st_mtime_ns,'base64':base64.b64encode(data).decode()}
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(root),'job':str(job),'before':identity()}
out['queue']=run(['bash',str(root/'bench/fleet.sh'),'show','eplocalonepass0909v4'])
out['cpu_session']=run(['bash',str(root/'bench/fleet.sh'),'show','eponepass4logic0909'])
out['job_inventory']=[{'name':p.name,'size':p.stat().st_size} for p in sorted(job.iterdir()) if p.is_file()]
names=['submission.json','submission-output.txt','cpu-report.json','cpu-report-logic-0.log']
out['files']={name:file(job/name) for name in names if (job/name).is_file()}
paths={root/'build/glm53/manifest.tsv'}
manifest=(root/'build/glm53/manifest.tsv').read_text()
module_names={line.split('\t')[0] for line in manifest.splitlines() if line}
paths.update(root/'build/glm53'/name for name in module_names)
paths.update(p for p in (root/'overlay/modules').rglob('*') if p.is_file() and p.name in module_names)
harness=['bench/fleet.sh','bench/chain.sh','bench/ab-lever.sh','bench/onepass.py','bench/proof.py','bench/experiment_baselines.py','bench/fleet_onepass.py','bench/experiment_resources.py','bench/cpu_evidence.py','profiles/glm53.env','probes/run_mhc_glm53_bench.sh','tests/test_glm53_ep_compact_warmup.py','tests/test_onepass_scoped_baseline.py','tests/test_logic.py','tests/test_glm53_overlay_sync.py','tests/test_fleet_prepare.py']
paths.update(root/p for p in harness)
out['source_files']={}
for path in sorted(paths):
 if not path.is_file(): raise RuntimeError('missing requested source '+str(path))
 data=path.read_bytes()
 out['source_files'][str(path.relative_to(root))]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
out['manifest']=file(root/'build/glm53/manifest.tsv')
out['after']=identity()
print(json.dumps(out))
'''
received = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','choiceoh@srv2','python3 -c '+__import__('shlex').quote(REMOTE)],capture_output=True,text=True)
if received.returncode:
    raise RuntimeError(received.stderr)
r = json.loads(received.stdout)
for when in ('before','after'):
    assert r[when]['head']['returncode'] == 0
    assert r[when]['head']['stdout'].strip() == REV
    assert r[when]['status']['returncode'] == 0 and r[when]['status']['stdout'] == ''
assert r['queue']['returncode'] == 0 and 'ticket: 17888996582431852' in r['queue']['stdout']
OUT.mkdir(parents=True, exist_ok=False)
provenance={}
def save(relative,data,origin):
    p=OUT/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    expected=hashlib.sha256(data).hexdigest();assert actual==expected
    provenance[relative]={'origin':origin,'size':len(data),'sha256':actual}
local={'submission/local-submit.json':'/tmp/glm53-onepass4-submit.json','submission/local-submit-output.txt':'/tmp/glm53-onepass4-submit-output.txt','submission/local-submit.py':'/tmp/glm53-onepass4-submit.py'}
logs=['glm53-onepass-scoped-baseline-tests.log','glm53-onepass4-fixture-repair-tests.log','glm53-onepass4-core.log','glm53-onepass4-overlay-sync.log','glm53-onepass4-prepare-final.log','glm53-onepass4-logic.log','glm53-onepass4-logic-final.log','glm53-onepass4-logic-passed.log','glm53-onepass4-compose.log']
local.update({'validation/local/'+name:'/tmp/'+name for name in logs})
for relative,origin in local.items():
    save(relative,pathlib.Path(origin).read_bytes(),origin)
for name,record in r.pop('files').items():
    data=base64.b64decode(record.pop('base64'));assert hashlib.sha256(data).hexdigest()==record['sha256']
    sub='validation/linux' if name.startswith('cpu-report') else 'submission/remote'
    save(sub+'/'+name,data,record)
manifest=r.pop('manifest');data=base64.b64decode(manifest.pop('base64'));assert hashlib.sha256(data).hexdigest()==manifest['sha256']
save('source/frozen-manifest.tsv',data,manifest)
source_files=r.pop('source_files')
save('source/hashes.json',(json.dumps({'frozen_revision':REV,'remote_source':r['source'],'files':source_files},indent=2)+'\n').encode(),'read-only SHA256 of frozen source files')
save('queue-snapshot.json',(json.dumps(r,indent=2)+'\n').encode(),'read-only ssh source identities and fleet show')
reported={'status':'reported_pass_tool_output_only','command':'python3 -B -m unittest discover -s tests -p test_glm53_ep_compact_warmup.py -v','tests_run':8,'errors':0,'skips':0,'reported_seconds':0.415,'source':'ep_kernel_extra tool-output transcript in this thread, relayed to root; original stdout was not saved to a file','source_sha256_at_test_report':'23079af7d6d7277a759226a2ea3c60af96352dd8f9ae43ed0cb5fe24db0de1bb','test_sha256':'74e70bbf6ab9e4838e458e9eb636be786e47f1c3ab5b3c76777b171f91fc84b1','note':'Source SHA was taken after a comment-only clarification following the successful run. This is a report, not a reconstructed test log or a fresh execution.'}
save('validation/warmup-focused-reported.json',(json.dumps(reported,indent=2)+'\n').encode(),'agent/root observed report only')
save('originals.json',(json.dumps(provenance,indent=2)+'\n').encode(),'collector originals provenance (self excluded)')
print(json.dumps({'archive':str(OUT),'captured_utc':r['captured_utc'],'source_file_count':len(source_files),'original_count':len(provenance),'queue':r['queue']['stdout'].splitlines()[0],'linux_report':(OUT/'validation/linux/cpu-report.json').exists()},indent=2))
