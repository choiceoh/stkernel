#!/usr/bin/env python3
"""Existing normal-chain experimental onepass glue; no numerical/default acceptance."""
import base64, gzip, hashlib, json, os, re, shlex, signal, subprocess, sys, time
from pathlib import Path
REV='540dfb7eea44a7356af897268fa6d6be2edba683'
REPO=Path('/home/choiceoh/stkernel-ep-onepass-0909-1')
SCHEDULER=Path('/home/choiceoh/stkernel-ep-local-scheduler-0909')
JOB=Path('/tmp/glm53-ep-onepass-0909-1'); OUT=JOB/'capture'
SESSION='eplocalonepass0909v1'
WITNESS=Path('/home/choiceoh/glm53-logs/mb9216-0909.log')
WITNESS_SHA='d1aff5c7bdf593213d4801c4e96d94b843d08ebe301f0b8e039c772196374b4a'
NAMES=('EPONEPASS1B1','EPONEPASS1A','EPONEPASS1B2')
sys.path[:0]=[str(REPO/'bench'),str(REPO/'probes')]
import prefill_serving as serving
import glm53_ep_serving_contract as contract
import glm53_launch_metadata as launch
import glm53_probe_lifecycle as lifecycle
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def require(value,message):
    if not value:raise RuntimeError(message)
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(name):return json.loads((OUT/name).read_text())
def pinned():
    lifecycle.pinned(str(REPO),REV)
def run(command,**kwargs):serving.run_owned(command,**kwargs)
# Reuse existing inspected-ID/model/source/fresh-file-log capture. Extend only
# its data record: raw Cmd/Env stay in the private job, never printed.
serving.CANDIDATES['ep']=(contract.EP_LOCAL,'[ep-prefill-local] LAUNCHED full-token E72/I2048/top8 T=',REV,str(REPO))
ending='print(json.dumps(state))'
require(serving.SNAPSHOT.count(ending)==1,'snapshot extension point changed')
metadata_code="import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ('cuda-bindings','cuda-python','cuda-pathfinder')}))"
serving.SNAPSHOT=serving.SNAPSHOT.replace(ending,"state.update(cmd=c['Config']['Cmd'],env=c['Config']['Env'],host_config=c['HostConfig'],docker_mounts=c['Mounts'])\nstate['binding_runtime']=json.loads(subprocess.check_output(['docker','exec',c['Id'],'python3','-B','-c',"+repr(metadata_code)+"],text=True,timeout=20))\n"+ending)
def capture(label=None):return serving.capture('ep',OUT if label else None,label or 'snapshot')
def config(nodes):
    return {node:{k:v for k,v in state.items() if k in ('cmd','env','id','started_at','image','model','hardware','host_config','mounts','manifest_sha')} for node,state in nodes.items()}
def envdict(node):
    result={}
    for item in node['env']:
        key,value=item.split('=',1);require(key not in result,'duplicate runtime Env');result[key]=value
    return result
def spec(cmd):
    launch.launch_parallelism(cmd)
    script=base64.b64decode(launch._WRAPPER.fullmatch(cmd[1])[1],validate=True).decode()
    line=script[len(launch._GID_PRELUDE):].removesuffix('\n')
    argv=launch._literal_argv(line[:-len(launch._REDIRECTION)])
    values=[a.split('=',1)[1] if '=' in a else argv[i+1] for i,a in enumerate(argv) if a.split('=',1)[0]=='--speculative-config']
    require(len(values)==1,'exact speculative config required');return values[0]
def arm_record(nodes,enabled,controls):
    mapping={n:contract.EndpointSubstitution(contract.Endpoint('0.0.0.0',8000),contract.Endpoint('127.0.0.1',18000)) for n in contract.NODES}
    return contract.configured_arm(config(nodes),enabled=enabled,expected_knobs={contract.EP_LOCAL:str(int(enabled)),contract.EP_COMPACT:'1' if enabled else None},kv_witness={k:controls[k] for k in ('KV_TOKENS','KV_HYBRID_BLOCKS')},endpoint_mapping=mapping)
def attest(nodes,original,controls,enabled):
    expected=serving.source_contract(REPO,REV)
    record=arm_record(nodes,enabled,controls)
    for node,state in nodes.items():
        require(state['image']==original[node]['image'] and state['model']==original[node]['model'],'original image/model changed')
        require(state['mounts']==expected['mounts'] and state['manifest_sha']==expected['manifest_sha'],'frozen deployed source differs')
        require(record['nodes'][node]['capacity']==contract.launch_capacity(original[node]['cmd'],public=True)['capacity'],'original capacity changed')
        require(spec(state['cmd'])==spec(original[node]['cmd']),'original speculative config changed')
    require(record['controls']==controls,'resolved capacity replay changed')
    return record

def after(name):
    require(name in NAMES,'unknown after hook');lifecycle.check_holder();pinned()
    original=read('original.json');controls=read('controls.json');enabled=name==NAMES[1]
    before=capture();record=attest(before,original,controls,enabled);save(name+'.before.json',before)
    if name!=NAMES[0]:
        baseline=read(NAMES[0]+'.configured.json')
        require(record['controls']==baseline['controls'],'arm controls changed')
        for node,current in record['nodes'].items():
            reference=baseline['nodes'][node]
            for key in ('capacity','endpoint','other_env_sha256'):
                require(current[key]==reference[key],'arm configuration changed: '+key)
            require(current['parallelism']['serve_argv_without_ep_sha256']==reference['parallelism']['serve_argv_without_ep_sha256'],'non-EP argv changed')
            require(current['provenance']['fields']==reference['provenance']['fields'],'arm source/image/model/host identity changed')
    save(name+'.configured.json',record)
    command=['python3',str(REPO/'bench/onepass_memory.py'),'--minimum-gib','12','--report',str(OUT/(name+'.memory.jsonl')),'--','python3',str(REPO/'bench/onepass_fresh.py'),'--out',str(OUT),'--name',name,'--ctx','2000,32000,128000']
    failure=None
    try:
        with (OUT/(name+'.client.log')).open('x') as log:run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    except BaseException as exc:failure=exc
    try:observed=capture(name);save(name+'.after.json',observed)
    except BaseException as exc:
        save(name+'.capture-error.json',dict(error=repr(exc)))
        if failure is None:raise
    if failure is not None:raise failure
    require(attest(observed,original,controls,enabled)==record,'container/config changed during onepass')
    markers={}
    for node,state in observed.items():
        raw=gzip.decompress((OUT/(name+'.'+node+'.log.gz')).read_bytes());text=raw.decode(errors='replace')
        markers[node]=dict(ep_local=bool(state['launch_proof']),mhc_shard_selection='MHC token shards selected' in text,log_sha256=sha(raw))
    save(name+'.markers.json',dict(ranks=markers,scope='Selection/launch evidence only; not layer completion or numerics'))
    rows=[json.loads(line) for line in (OUT/'onepass.jsonl').read_text().splitlines() if line.strip()]
    selected=[row for row in rows if row['name']==name]
    require(len(selected)==1,'onepass record missing/duplicated')
    require([row['ctx'] for row in selected[0]['prefill']]==[2000,32000,128000],'onepass context ladder changed')
    require(all(v['mhc_shard_selection'] for v in markers.values()) and (not enabled or all(v['ep_local'] for v in markers.values())),'actual four-rank selection/launch missing')
    save(name+'.result.json',dict(record=selected[0],fresh=read(name+'.fresh.json'),markers=markers,experimental=True,full_model_acceptance=False))

def main():
    os.umask(0o077)
    require(os.environ.get('FLEET_SESSION')==SESSION,'wrong normal fleet session')
    if len(sys.argv)==3 and sys.argv[1]=='after':after(sys.argv[2]);return 0
    require(sys.argv[1:]==['run'],'unsupported arguments');lifecycle.check_holder();pinned()
    OUT.mkdir(parents=True,exist_ok=False)
    result=dict(started=time.time(),exit_code=1,experimental=True,full_model_acceptance=False,default_promotion=False,prior_numerics_failure=True,supervisor_recovery='pending normal restore or handoff',offline_binding_runtime='13.0.3',serving_binding_runtime='normal pinned-image runtime; not forced to offline capsule')
    try:
        original=capture();save('original.json',original)
        lifecycle.idle(8000)
        witness=WITNESS.read_bytes();require(sha(witness)==WITNESS_SHA,'KV witness log changed')
        require(re.search(rb'KV pinned: 2000000 tokens -> 1056 blocks \(2304/block \+ 187 hybrid\)',witness) is not None,'resolved KV witness missing')
        save('kv-witness.json',dict(path=str(WITNESS),sha256=WITNESS_SHA,KV_TOKENS='2000000',KV_HYBRID_BLOCKS='187'))
        capacities={n:contract.launch_capacity(s['cmd'],public=True) for n,s in original.items()}
        controls={n:contract.replay_controls(c,{'KV_TOKENS':'2000000','KV_HYBRID_BLOCKS':'187'}) for n,c in capacities.items()}
        require(len({json.dumps(c,sort_keys=True) for c in controls.values()})==1,'original ranks disagree on capacity')
        controls=next(iter(controls.values()));save('controls.json',controls)
        image=original[contract.NODES[0]]['image'];require(all(s['image']==image for s in original.values()),'original rank images differ')
        env=dict(os.environ,REPO=str(REPO),FLEET=str(SCHEDULER/'bench/fleet.sh'),LEVER=str(REPO/'bench/ab-lever.sh'),IMAGE=image,ENABLE_EP='0',VLLM_GLM53_EP_PREFILL_LOCAL='0',**contract.COMMON_KNOBS)
        for key in ('VLLM_B12X_EP_COMPACT','FLEET_WORKLOAD','FLEET_CONTEXT','FLEET_EXPERIMENT_ID','FLEET_OBJECTIVE'):env.pop(key,None)
        env.update(controls);env.update(LEGS='none',PREFILL_WARMUP='0',GLM53_API_HOST='127.0.0.1',GLM53_API_PORT='18000',HEAD='127.0.0.1',HEAD_URL='http://127.0.0.1:18000',BENCH_MODEL='glm-5.3-flash',ONEPASS_JSONL=str(OUT/'onepass.jsonl'))
        with (OUT/'deploy.log').open('x') as log:run(['bash',str(SCHEDULER/'bench/fleet.sh'),'deploy',SESSION,REV],env=env,stdout=log,stderr=subprocess.STDOUT)
        pinned();save('deployed-source.json',serving.source_contract(REPO,REV))
        # The new launcher needs its admitted boot-authorization helper; the
        # queued outer supervisor and normal deploy validation stay unchanged.
        save('helper-selection.json',dict(outer_runner=os.environ.get('FLEET_RUNNER_REPO'),chain_helper_source=str(REPO),revision=REV))
        env['FLEET_RUNNER_REPO']=str(REPO)
        command=['bash',str(REPO/'bench/chain.sh'),NAMES[0]+'=',NAMES[1]+'=ENABLE_EP=1 VLLM_GLM53_EP_PREFILL_LOCAL=1',NAMES[2]+'=']
        for name in NAMES:command+=['--legs',name,'none','--after',name,shlex.join(['python3','-B',str(Path(__file__).resolve()),'after',name])]
        save('chain-command.json',command)
        with (OUT/'chain.log').open('x') as log:run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        result.update(exit_code=0,arms={name:read(name+'.result.json') for name in NAMES})
    except BaseException as exc:result['error']=repr(exc)
    finally:result['ended']=time.time();save('completion.json',result)
    return result['exit_code']
if __name__=='__main__':
    def interrupted(signum,frame):raise InterruptedError('termination requested')
    signal.signal(signal.SIGTERM,interrupted)
    raise SystemExit(main())
