#!/usr/bin/env python3
"""Owned GLM prefill arm collection and B1/A/B2 serving execution.

Use run only after the frozen GPU gate passes, through fleet.sh run --gpu.
The source checkout must be clean and based on current origin/main. This
uses fleet deploy and chain, then always restores the public default arm.
No candidate is promoted. `arm` is the chain's after hook, not a standalone
unreserved client. All metrics, metadata and archive work is outside TTFT.
"""
import argparse
import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

import prefill_compare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
from glm53_offline_checks import PINS, check_holder, pinned, remote

IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
NODES = ('10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4')
CANDIDATES = {
    'moe': ('VLLM_GLM53_B12X_PREFILL_STREAM_FC2', '[b12x prefill stream] LAUNCHED m=',
            '924b1be06e146381019fb1ee1144bd6d64b2914d', '/home/choiceoh/stkernel-moe-stream-check2-0907'),
    'mla': ('VLLM_GLM53_MK_MLA_PREFILL32', '[megakernel] mla prefill32 LAUNCHED T=',
            '3eb219dd2d938479326a5a6704f3789d854367dd', '/home/choiceoh/stkernel-prefill32-check5-0907'),
}

SNAPSHOT = r'''
import base64,gzip,hashlib,json,pathlib,re,shlex,subprocess
sha=lambda value:hashlib.sha256(value).hexdigest()
names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
if name not in names:raise RuntimeError('expected serving container missing: '+name)
c=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
cmd=' '.join(c['Config'].get('Cmd') or [])
payload=re.search(r'echo ([A-Za-z0-9+/=]+) \| base64 -d',cmd)
if payload:cmd=base64.b64decode(payload.group(1),validate=True).decode()
args={}
for key in ['host','port','max-model-len','num-gpu-blocks-override','gpu-memory-utilization',
            'max-num-batched-tokens','max-num-seqs','max-cudagraph-capture-size']:
    match=re.search(r'--'+key+r'(?:=|\s+)([^\s]+)',cmd)
    args[key]=match.group(1) if match else None
args['command_sha256']=sha(cmd.encode())
env=dict(e.split('=',1) for e in c['Config']['Env'] if '=' in e)
# Hash other environment values, including any credentials, while still
# checking that every non-candidate value remains identical across boots.
env={k:(v if k==knob else 'sha256:'+sha(v.encode())) for k,v in env.items()}
mounts={m['Destination']:sha(pathlib.Path(m['Source']).read_bytes()) for m in c['Mounts']
        if m['Source'].startswith('/home/choiceoh/overlays/glm53/')}
model={}
for m in c['Mounts']:
    if not m['Destination'].startswith('/models/'):continue
    path=pathlib.Path(m['Source'])
    if not path.is_dir():continue
    files={f.name:sha(f.read_bytes()) for f in path.iterdir() if f.is_file() and
           (f.name in ('config.json','tokenizer.json','tokenizer_config.json','generation_config.json') or f.name.endswith('.index.json'))}
    weights=sorted((str(f.relative_to(path)),f.stat().st_size,f.stat().st_mtime_ns)
                   for f in path.rglob('*.safetensors'))
    if not files or not weights:raise RuntimeError('model identity incomplete: '+str(path))
    model[m['Destination']]=dict(path=str(path.resolve()),metadata=files,weight_files=weights)
manifest=pathlib.Path('/home/choiceoh/overlays/glm53/manifest.tsv').read_bytes()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
if not gpu or not model:raise RuntimeError('model/hardware identity unavailable')
state=dict(id=c['Id'],started_at=c['State']['StartedAt'],image=c['Image'],args=args,env=env,
           mounts=mounts,manifest_sha=sha(manifest),model=model,hardware=gpu)
if archive:
    raw=subprocess.check_output(['docker','logs','--since',c['State']['StartedAt'],c['Id']],stderr=subprocess.STDOUT)
    text=raw.decode(errors='replace')
    state['launch_proof']=marker in text
    state['launch_lines']=[line for line in text.splitlines() if marker in line]
    state['log_sha256']=sha(raw)
    state['log_gzip_base64']=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
print(json.dumps(state))
'''


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def capture(candidate, out=None, label='snapshot'):
    knob, marker, _, _ = CANDIDATES[candidate]
    def one(node):
        name = 'glm53' if node == '10.10.10.2' else 'glm53-worker'
        code = f'name={name!r}\nknob={knob!r}\nmarker={marker!r}\narchive={out is not None!r}\n'+SNAPSHOT
        result = remote('local' if node == '10.10.10.2' else node, code, timeout=90)
        if out is not None:
            data=base64.b64decode(result.pop('log_gzip_base64'),validate=True)
            (out/(label+'.'+node+'.log.gz')).write_bytes(data)
        return node,result
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(one, NODES))


def source_contract(repo, revision):
    manifest=(repo/'build/glm53/manifest.tsv').read_text()
    mounts={}
    for line in manifest.splitlines():
        if not line or line.startswith('#'):continue
        source,target,_=line.split('\t')
        mounts[target]=hashlib.sha256((repo/'build/glm53'/source).read_bytes()).hexdigest()
    return dict(manifest_sha=hashlib.sha256(('# source_commit='+revision+'\n'+manifest).encode()).hexdigest(),mounts=mounts)


def attest(nodes, contract, knob, enabled, public=False):
    issues=[]
    expected_args = dict(host='0.0.0.0' if public else '127.0.0.1', port='8000' if public else '18000',
                         **{'max-model-len':'1048576' if public else '262144',
                            'num-gpu-blocks-override':'1056' if public else '415'})
    for node in NODES:
        state=nodes[node]
        if state['image'] != IMAGE:issues.append(node+': image mismatch')
        for key,value in contract.items():
            if state[key] != value:issues.append(node+': source '+key+' mismatch')
        if state['env'].get(knob) != str(int(enabled)):issues.append(node+': knob mismatch')
        if any(state['args'].get(k)!=v for k,v in expected_args.items()):issues.append(node+': endpoint/capacity mismatch')
    if issues:raise RuntimeError('; '.join(issues))


def verify_gate(candidate, directory, repo):
    _,_,revision,frozen=CANDIDATES[candidate]
    complete=json.loads((directory/'completion.json').read_text())
    gate=complete['probes'][candidate]
    if not complete.get('ended') or complete.get('error') or gate['exit_code'] != 0 or gate['revision'] != revision:
        raise RuntimeError('candidate GPU gate or recovery is incomplete/failed')
    if not complete.get('restored_original') and not complete.get('public_restore'):
        raise RuntimeError('offline serving recovery evidence missing')
    log=(directory/(candidate+'.log')).read_bytes()
    if not log or (candidate=='moe' and b'MOE_STREAM_ALL_GATES_PASS' not in log):
        raise RuntimeError('GPU gate log incomplete')
    pinned(frozen,revision)
    paths = (['overlay/modules/glm53_moe/'+n for n in
              ('moe_dynamic_prefill_n128.py','moe_dynamic_prefill.py','moe_dynamic_gated_tiled.py',
               'moe_dispatch.py','flashinfer_b12x_moe.py')]
             if candidate=='moe' else ['overlay/modules/glm53_megakernel/glm53_megakernel.cu'])
    hashes={}
    for path in paths:
        old=(Path(frozen)/path).read_bytes();new=(repo/path).read_bytes()
        if old!=new:raise RuntimeError('GPU-validated source changed: '+path)
        hashes[path]=hashlib.sha256(new).hexdigest()
    if candidate=='mla':
        path='overlay/modules/glm53_megakernel/glm53_megakernel.py'
        def function(p):
            tree=ast.parse(p.read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mla_prefill32'),include_attributes=False)
        if function(Path(frozen)/path)!=function(repo/path):
            raise RuntimeError('GPU-validated MLA call function changed')
    return dict(revision=revision,files=hashes,log_sha256=hashlib.sha256(log).hexdigest(),
                completion_sha256=hashlib.sha256((directory/'completion.json').read_bytes()).hexdigest())


def run_owned(command, **kwargs):
    """Terminate only this owned process group before outer recovery."""
    child=subprocess.Popen(command,start_new_session=True,**kwargs)
    try:
        rc=child.wait()
        if rc:raise subprocess.CalledProcessError(rc,command)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


def boot_controls(node):
    mapping={'gpu-memory-utilization':'GMU','max-num-batched-tokens':'MAX_BATCHED',
             'max-num-seqs':'MAX_SEQS','max-cudagraph-capture-size':'GRAPH_CAP'}
    controls={v:node['args'][k] for k,v in mapping.items()}
    if not 0<float(controls['GMU'])<1 or any(not str(controls[k]).isdigit() for k in ('MAX_BATCHED','MAX_SEQS','GRAPH_CAP')):
        raise RuntimeError('invalid measured boot controls')
    # B1 is read after the launcher's graph-budget correction. Applying the
    # deduction a second time would silently change A/B2's memory budget.
    controls['CG_UTIL_DELTA']='0'
    return controls


def boot_arm(args):
    check_holder()
    out=Path(os.environ['PREFILL_SERVING_OUT']);repo=Path(os.environ['REPO'])
    knob=CANDIDATES[os.environ['PREFILL_SERVING_CANDIDATE']][0]
    if args.knobs not in ('',knob+'=1'):raise RuntimeError('unexpected arm settings')
    env=dict(os.environ)
    control_file=out/'boot-controls.json'
    if control_file.exists():
        controls=json.loads(control_file.read_text())
        if set(controls)!={'GMU','MAX_BATCHED','MAX_SEQS','GRAPH_CAP','CG_UTIL_DELTA'} or controls['CG_UTIL_DELTA']!='0':
            raise RuntimeError('invalid frozen boot controls')
        env.update(controls)
    elif args.name!=os.environ['PREFILL_SERVING_FIRST_ARM']:
        raise RuntimeError('baseline boot controls missing; refusing the next arm')
    run_owned(['bash',str(repo/'bench/ab-lever.sh'),args.name,args.knobs],cwd=repo,env=env)


def collect_arm(args):
    check_holder()
    repo=Path(os.environ['REPO']);revision=os.environ['PREFILL_SERVING_REV']
    out=Path(os.environ['PREFILL_SERVING_OUT']);candidate=os.environ['PREFILL_SERVING_CANDIDATE']
    knob=CANDIDATES[candidate][0]
    pinned(str(repo),revision)
    before=capture(candidate)
    contract=source_contract(repo,revision)
    attest(before,contract,knob,args.enabled)
    controls=boot_controls(before['10.10.10.2'])
    control_file=out/'boot-controls.json'
    if args.name==os.environ['PREFILL_SERVING_FIRST_ARM']:
        if control_file.exists():raise RuntimeError('duplicate first arm')
        save(control_file,controls)
    elif json.loads(control_file.read_text())!=controls:
        raise RuntimeError('effective boot controls differ from B1')
    arm=dict(revision=revision,knob=knob,enabled=args.enabled,before=before)
    save(out/(args.name+'.incomplete.json'),arm)
    salts=set()
    for phase in ('priming','measured'):
        name=args.name+'PRIME' if phase=='priming' else args.name
        env=dict(os.environ,MK_COLD_COMPILE='0')
        with (out/(name+'.client.log')).open('x') as log:
            run_owned(['python3',str(ROOT/'bench/onepass_memory.py'),'--minimum-gib','12',
                       '--report',str(out/(name+'.memory.jsonl')),'--',
                       'python3',str(ROOT/'bench/onepass_fresh.py'),'--out',str(out),'--name',name],
                      cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        records=[json.loads(l) for l in (out/'onepass.jsonl').read_text().splitlines() if l.strip()]
        selected=[r for r in records if r['name']==name]
        if len(selected)!=1:raise RuntimeError('missing or duplicate phase record')
        arm[phase]=dict(record=selected[0],fresh=json.loads((out/(name+'.fresh.json')).read_text()))
        issues=[]
        prefill_compare.validate_phase(arm[phase],arm,salts,issues,name)
        save(out/(args.name+'.incomplete.json'),arm)
        if issues:raise RuntimeError(str(issues))
    after=capture(candidate,out,args.name)
    attest(after,contract,knob,args.enabled)
    arm['after']=after
    arm['launch_proof']={node:state['launch_proof'] for node,state in after.items()}
    if any(before[n].get(k)!=after[n].get(k) for n in NODES for k in prefill_compare.NODE_FIELDS):
        raise RuntimeError('node identity changed during traffic')
    if args.enabled and not all(arm['launch_proof'].values()):
        raise RuntimeError('actual candidate launch missing on a rank')
    save(out/(args.name+'.json'),arm)
    print(json.dumps(dict(name=args.name,complete=True,launch_proof=arm['launch_proof'])),flush=True)


def refresh_gpu_gate(candidate, source, revision, directory):
    """Revalidate the frozen current-main kernel before any serving deploy."""
    _, _, probe_revision, probe_source = CANDIDATES[candidate]
    expected = ((candidate, probe_source, probe_revision,
                 ['bash', 'probes/run_mk_mla_prefill32_check.sh']),)
    if candidate != 'mla' or PINS != expected:
        raise RuntimeError('refresh requires the exact single-candidate full GPU plan')
    run_owned(['python3', str(source/'probes/glm53_offline_checks.py'),
               '--out', str(directory)], cwd=source,
              env=dict(os.environ, OFFLINE_SOURCE_REV=revision))


def run_bracket(args):
    check_holder()
    source=args.source.resolve();out=args.out.resolve()
    if source!=ROOT.resolve():raise RuntimeError('runner and canonical harness must belong to the source checkout')
    pinned(str(source),args.revision)
    subprocess.run(['git','-C',str(source),'fetch','--quiet','origin','main'],check=True)
    subprocess.run(['git','-C',str(source),'merge-base','--is-ancestor','origin/main',args.revision],check=True)
    if getattr(args, 'refresh_gate', False):
        refresh_gpu_gate(args.candidate, source, args.revision, args.gate_dir)
    gate=verify_gate(args.candidate,args.gate_dir,source)
    out.mkdir(parents=True,exist_ok=False)
    save(out/'gpu-gate.json',gate)
    resources={n:remote('local' if n=='10.10.10.2' else n,
        "import json,shutil; print(json.dumps(dict(disk_free_gib=shutil.disk_usage('/home/choiceoh').free/2**30)))") for n in NODES}
    save(out/'resources.json',resources)
    if any(v['disk_free_gib']<128 for v in resources.values()):raise RuntimeError('128 GiB per-node disk reserve required')
    repo=Path('/home/choiceoh/stkernel');fleet=repo/'bench/fleet.sh'
    knob=CANDIDATES[args.candidate][0]
    # Start from the deployed source's profile, not a predecessor's experiment
    # env. Only the tested knob differs between the three private boots.
    controlled=('VLLM_','ONEPASS_','FLEET_WORKLOAD','FLEET_CONTEXT','FLEET_EXPERIMENT_ID','FLEET_OBJECTIVE')
    profile_keys=set(re.findall(r'^([A-Z][A-Z0-9_]*)=',(source/'profiles/glm53.env').read_text(),re.M))
    env={k:v for k,v in os.environ.items() if not k.startswith(controlled) and k not in profile_keys}
    env.update(REPO=str(repo),IMAGE=IMAGE,LEGS='none',PREFILL_WARMUP='0',
        HEALTH_BUDGET_S='1800',BENCH_MODEL='glm-5.3-flash',KV_TOKENS='524288',MAX_LEN='262144',
        GLM53_API_PORT='18000',GLM53_API_HOST='127.0.0.1',HEAD='127.0.0.1',HEAD_URL='http://127.0.0.1:18000',
        FLEET=str(fleet),LEVER=str(ROOT/'bench/prefill_boot.sh'),ONEPASS_JSONL=str(out/'onepass.jsonl'),
        PREFILL_SERVING_RUNNER=str(ROOT/'bench/prefill_serving.py'),PREFILL_SERVING_FIRST_ARM=args.name+'B1',
        PREFILL_SERVING_OUT=str(out),PREFILL_SERVING_CANDIDATE=args.candidate,PREFILL_SERVING_REV=args.revision)
    result=dict(started=time.time(),exit_code=1)
    changed=False
    try:
        changed=True
        run_owned(['bash',str(fleet),'deploy',os.environ['FLEET_SESSION'],args.revision],env=env)
        pinned(str(repo),args.revision)
        contract=source_contract(repo,args.revision)
        save(out/'source-contract.json',contract)
        names=[args.name+s for s in ('B1','A','B2')]
        command=['bash',str(repo/'bench/chain.sh'),names[0]+'=',names[1]+'='+knob+'=1',names[2]+'=']
        for name,enabled in zip(names,(False,True,False)):
            hook=['python3',str(ROOT/'bench/prefill_serving.py'),'arm','--name',name]
            if enabled:hook.append('--enabled')
            command+=['--legs',name,'none','--after',name,shlex.join(hook)]
        with (out/'chain.log').open('x') as log:
            run_owned(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        comparison=prefill_compare.compare([json.loads((out/(n+'.json')).read_text()) for n in names])
        save(out/'comparison.json',comparison)
        if comparison['issues']:raise RuntimeError(str(comparison['issues']))
        result['exit_code']=0
    except BaseException as exc:
        result['error']=repr(exc)
    finally:
        if changed:
            previous=signal.signal(signal.SIGTERM,signal.SIG_IGN)
            try:
                check_holder()
                restore=dict(env,KV_TOKENS='2000000',MAX_LEN='1048576',
                    GLM53_API_PORT='8000',GLM53_API_HOST='0.0.0.0',HEAD='10.10.10.2',HEAD_URL='http://10.10.10.2:8000')
                with (out/'public-restore.log').open('x') as log:
                    run_owned(['bash',str(repo/'bench/ab-lever.sh'),args.name+'RESTORE',''],
                              cwd=repo,env=restore,stdout=log,stderr=subprocess.STDOUT)
                restored=capture(args.candidate,out,'RESTORE')
                attest(restored,source_contract(repo,args.revision),knob,False,public=True)
                baseline_file=out/(args.name+'B1.json')
                if baseline_file.exists():
                    baseline=json.loads(baseline_file.read_text())['before']
                    if any(restored[n][k]!=baseline[n][k] for n in NODES for k in ('model','hardware')):
                        raise RuntimeError('restored model/hardware differs from measured baseline')
                with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=10) as r:
                    if r.status!=200:raise RuntimeError('public health failed')
                save(out/'public-restored.json',restored)
                result['restored']=True
            except BaseException as exc:
                result.update(exit_code=1,restore_error=repr(exc))
            finally:
                signal.signal(signal.SIGTERM,previous)
        result['ended']=time.time()
        save(out/'completion.json',result)
    return result['exit_code']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='mode',required=True)
    arm=sub.add_parser('arm');arm.add_argument('--name',required=True);arm.add_argument('--enabled',action='store_true')
    boot=sub.add_parser('boot');boot.add_argument('name');boot.add_argument('knobs',nargs='?',default='')
    run=sub.add_parser('run')
    run.add_argument('--name',required=True);run.add_argument('--candidate',choices=CANDIDATES,required=True)
    run.add_argument('--source',type=Path,required=True);run.add_argument('--revision',required=True)
    run.add_argument('--gate-dir',type=Path,required=True);run.add_argument('--out',type=Path,required=True)
    run.add_argument('--refresh-gate',action='store_true',
                     help='Run the pinned full MLA GPU gate and recover before serving, in this same owned turn')
    args=ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',args.name):ap.error('invalid arm name')
    def interrupted(signum, frame):raise InterruptedError('termination requested')
    signal.signal(signal.SIGTERM,interrupted)
    if args.mode=='arm':return collect_arm(args)
    if args.mode=='boot':return boot_arm(args)
    return run_bracket(args)


if __name__=='__main__':
    raise SystemExit(main())
