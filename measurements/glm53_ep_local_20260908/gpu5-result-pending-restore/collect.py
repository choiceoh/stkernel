from pathlib import Path
import base64,gzip,hashlib,json,shlex,subprocess
root=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
out=root/'measurements/glm53_ep_local_20260908/gpu5-result-pending-restore'
remote=r'''from pathlib import Path
import base64,hashlib,json,time
j=Path('/tmp/glm53-ep-local-gpu-0908-5');c=j/'capture';f=Path('/home/choiceoh/glm53-logs/fleet')
complete=json.loads((c/'completion.json').read_text());assert complete['exit_code']==1 and complete['restored_original'] is True
raw={str(p.relative_to(c)):p.read_bytes() for p in sorted(c.rglob('*')) if p.is_file() and not p.is_symlink()}
assert sum(map(len,raw.values()))<32*1024*1024
assert all((c/n).read_bytes()==b for n,b in raw.items()),'closed capture changed during collection'
b=json.loads(raw['before.json']);s=json.loads(raw['stopped.json']);r=json.loads(raw['restored.json'])
assert b==s==r
state=dict(captured=time.time(),scope='offline capture complete FAIL; separate fleet public restoration not yet claimed complete',incoming_before_stopped_restored_equal=True,job_exit=json.loads((j/'exit.json').read_text()) if (j/'exit.json').exists() else None,holder=(f/'holder').read_text() if (f/'holder').exists() else None,queue=(f/'queue').read_text() if (f/'queue').exists() else None)
raw['fleet-state-at-capture.json']=(json.dumps(state,indent=2)+'\n').encode()
print(json.dumps({name:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()) for name,b in raw.items()}))
'''
result=json.loads(subprocess.check_output(['ssh','choiceoh@srv2',shlex.join(['python3','-B','-c',remote])]))
out.mkdir(exist_ok=False);manifest={}
for name,info in result.items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 b=base64.b64decode(info['data'],validate=True);assert len(b)==info['bytes'] and hashlib.sha256(b).hexdigest()==info['sha256']
 stored_name=name+'.gz' if name.endswith('.log') else name
 stored=gzip.compress(b,mtime=0) if name.endswith('.log') else b
 dest=out/stored_name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(stored)
 manifest[name]=dict(original_bytes=len(b),original_sha256=info['sha256'],stored_name=stored_name,stored_bytes=len(stored),stored_sha256=hashlib.sha256(stored).hexdigest())
(out/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'collect.py').write_bytes(Path(__file__).read_bytes())
c=json.loads((out/'completion.json').read_text());r=json.loads((out/'concentrated6912.json').read_text())
cells=[{k:x[k] for k in ('case','sanitizer','exit_code','started','ended') if k in x} for x in c['cells']]
summary=dict(verdict='FAIL',offline_complete=True,full_fleet_restore_complete=False,source=c['source_revision'],cells=cells,passed_cells=4,failed_cells=1,unrun_cells=12,sanitizer_cells_run=0,incoming_before_stopped_restored_equal=True,failure=dict(case='concentrated6912',phase=r['phase'],error=r['error'],binding_runtime_rechecked=r['binding_runtime_rechecked']),scope='Partial full-suite GPU evidence. Not performance acceptance or public-service restoration proof.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'README.md').write_text('''# Full GPU v5: numerical failure, public restore pending

Offline validation received GO at22:17:24 KST and ended at22:20:32 with FAIL.
The 24-specialization remap and balanced4096/6912/8192 MoE fixtures passed.
Concentrated6912 failed after input/routes/scales changed at the same
addresses: one row exceeded the fixed numerical limits. Maximum relative L2
was0.0118141882, normalized peak0.0406976752; stock controls passed. Initial
candidate and its nondefault-stream replay had passed. Runtime identity was
successfully rechecked after the failure. The tolerance is unchanged.

The suite stopped at this failure. Four remaining MoE fixtures and all eight
sanitizer cells were not run. This is not full GPU acceptance; successful
balanced timings do not establish TTFT or the incremental value of CPU16.
No silent retry or default promotion was performed.

The offline wrapper restored the exact four incoming stopped records:
before==stopped==restored, including configuration/source/mount/running state.
The normal fleet then started its separate public-serving restore and health
wait. At this snapshot the fleet job had no exit.json and retained its holder.
Do not conflate the wrapper's restored_original=true with completed fleet
public restoration. Final exit/release/service evidence must be collected
when that phase finishes. This directory is intentionally named pending-restore.

Every closed capture file was reread remotely and verified after transfer.
Logs use deterministic gzip; archive-manifest.json records original and stored
hashes. Frozen source is .../stkernel-ep-local-gpu-0908-5b atd53fd44f; CPU16's
original passing compiler receipt and all source/runtime checks remain in the
source and gpu5-queued archive. No frozen source or original result was edited.
''')
files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
(out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(out))+'\n' for p in files))
print(json.dumps(dict(archive=str(out),original_capture_files=len(result)-1,summary=summary)))
