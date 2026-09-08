from pathlib import Path
import os,sys,subprocess,json,hashlib,datetime
repo=Path('/Users/ost/.worktrees/stkernel2/glm53-fleet-restore-test-env-20260908')
out=Path('/tmp/glm53-fleet-restore-test-env0908');out.mkdir(exist_ok=True)
phase=sys.argv[1]
assert phase in ('before','after')
manifest=out/'parent-preparation.json'
if not manifest.exists():manifest.write_text('{"spec_path": null, "marker": "synthetic test parent only"}\n')
parent_bytes=manifest.read_bytes()
env=dict(os.environ,FLEET_PREPARE_MANIFEST=str(manifest))
# Do not borrow any real supervisor/experiment identity for unit fixtures.
for key in ('FLEET_LAUNCH_ID','FLEET_EXPERIMENT_ID'):env.pop(key,None)
command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_fleet_pending.py','-v']
started=datetime.datetime.now().astimezone().isoformat()
result=subprocess.run(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
log=out/(phase+'.log');log.open('xb').write(result.stdout)
receipt={'phase':phase,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'cwd':str(repo),'command':command,'started':started,'ended':datetime.datetime.now().astimezone().isoformat(),'returncode':result.returncode,'test_file_sha256':hashlib.sha256((repo/'tests/test_fleet_pending.py').read_bytes()).hexdigest(),'injected_environment':{'FLEET_PREPARE_MANIFEST':str(manifest)},'parent_manifest_sha256':hashlib.sha256(parent_bytes).hexdigest(),'parent_manifest_unchanged':manifest.read_bytes()==parent_bytes,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stdout_bytes':len(result.stdout),'scope':'Local synthetic parent environment; no fleet hold, queue, remote or GPU execution.'}
(out/(phase+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
print(result.stdout.decode())
