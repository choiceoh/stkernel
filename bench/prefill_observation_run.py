#!/usr/bin/env python3
"""Collect a current-default prefill baseline in a private clone of serving.

Run through fleet.sh run --gpu. The public originals remain intact and are
restarted in finally. CPU trace analysis is a separate command after the hold.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

import prefill_serving
from prefill_observation import PrivateObserverAPI,idle_observers
from prefill_observation_requests import validate_baseline,prefill_identity
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'probes'))
import glm53_offline_checks as lifecycle
import glm53_prefill_trace as trace
from glm53_prefill_observer import validate_ranks


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def settled(nodes,call):
    results,errors={},{}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={node:pool.submit(call,node) for node in nodes}
        for node,future in futures.items():
            try:results[node]=future.result()
            except Exception as exc:errors[node]=repr(exc)
    if errors:raise RuntimeError(str(errors))
    return results


class Run:
    def __init__(self,source,revision,out):
        self.source,self.revision,self.out=source,revision,out
        self.session=os.environ['FLEET_SESSION']
        self.name='glm53-observe-'+self.session
        self.node_dir=out/'worker'
        self.sha=hashlib.sha256((source/'probes/glm53_prefill_observer.py').read_bytes()).hexdigest()

    def host(self,node,action):
        lifecycle.check_holder()
        kwargs=dict(action=action,name=self.name,session=self.session,directory=str(self.node_dir),
                    source=str(self.source),original=lifecycle.name(node))
        code='import json,runpy\nm=runpy.run_path('+repr(str(self.source/'probes/glm53_observation_host.py'))+')\n'
        return lifecycle.remote(node,code+'print(json.dumps(m["dispatch"](**'+repr(kwargs)+')))',timeout=180)

    def all(self,action):return settled(lifecycle.NODES,lambda n:self.host(n,action))

    def cleanup(self):
        errors=[]
        try:
            logs=self.all('logs')
            for node,record in logs.items():
                if record['exists']:
                    raw=base64.b64decode(record.pop('gzip_base64'),validate=True)
                    if hashlib.sha256(gzip.decompress(raw)).hexdigest()!=record['sha256']:
                        raise ValueError('diagnostic log transfer hash mismatch')
                    (self.out/('diagnostic-'+node+'.log.gz')).write_bytes(raw)
            save(self.out/'diagnostic-logs.json',logs)
        except Exception as exc:errors.append('logs: '+repr(exc))
        # A transient Docker/SSH failure must not leave a clone holding memory
        # when the existing lifecycle helper restarts the original containers.
        for attempt in range(2):
            try:
                self.all('remove')
                save(self.out/'cleanup.json',dict(removed=True,errors=errors))
                return
            except Exception as exc:errors.append('remove: '+repr(exc))
        save(self.out/'cleanup.json',dict(removed=False,errors=errors))
        raise RuntimeError('owned clone cleanup failed: '+str(errors))

    def snapshot(self,original=False,archive=False):
        def one(node):
            name=lifecycle.name(node) if original else self.name
            code=f'name={name!r}\nknob="__no_candidate__"\nmarker="GLM53_PREFILL_OBSERVER"\narchive={archive!r}\n'
            result=lifecycle.remote(node,code+prefill_serving.SNAPSHOT,timeout=120)
            if archive:
                raw=base64.b64decode(result.pop('log_gzip_base64'),validate=True)
                (self.out/('serving-'+node+'.log.gz')).write_bytes(raw)
            return result
        return settled(lifecycle.NODES,one)

    def ready(self,prepared):
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            states=self.all('state')
            for node,state in states.items():
                if any(state[k]!=prepared[node][k] for k in ('id','config','host_config','image')) or not state['running']:
                    raise RuntimeError('clone exited or changed while booting: '+node)
            if lifecycle.healthy(18000):return states
            time.sleep(10)
        raise TimeoutError('private serving readiness timeout')

    def attest_clone(self,original,cloned):
        for node in lifecycle.NODES:
            a,b=original[node],cloned[node]
            if any(a[k]!=b[k] for k in ('image','mounts','manifest_sha','model','hardware')):
                raise ValueError('clone model/runtime differs from incoming: '+node)
            for key in a['args']:
                if key not in ('host','port','command_sha256') and a['args'][key]!=b['args'][key]:
                    raise ValueError('clone changed capacity/control: '+key)
            if b['args']['host']!='127.0.0.1' or b['args']['port']!='18000':
                raise ValueError('clone endpoint is not isolated')
            clean=lambda env:{k:v for k,v in env.items() if k!='PYTHONPATH'}
            if clean(a['env'])!=clean(b['env']):raise ValueError('clone changed runtime environment')

    def phase(self,mode,name,ctx=None):
        command=['python3',str(self.source/'bench/prefill_observation_requests.py'),
            '--mode',mode,'--name',name,'--out',str(self.out/name),'--observer-sha',self.sha]
        if ctx is not None:command+=['--ctx',str(ctx)]
        env=dict(os.environ,HEAD_URL='http://127.0.0.1:18000',GLM53_API_PORT='18000',BENCH_MODEL='glm-5.3-flash')
        for key in ('FLEET_WORKLOAD','FLEET_CONTEXT','FLEET_EXPERIMENT_ID','FLEET_OBJECTIVE'):
            env.pop(key,None)
        with (self.out/(name+'.client.log')).open('x') as log:
            prefill_serving.run_owned(['python3',str(self.source/'bench/onepass_memory.py'),
                '--minimum-gib','12','--report',str(self.out/(name+'.memory.jsonl')),'--',*command],
                cwd=self.source,env=env,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((self.out/name/'result.json').read_text())
        if not result['complete']:raise RuntimeError('incomplete request phase: '+name)
        return result

    def collect_traces(self,before,name):
        deadline=time.monotonic()+180;previous=None
        while time.monotonic()<deadline:
            current=self.all('traces')
            if current==previous:
                try:chosen={node:trace.fresh_trace(before[node],current[node]) for node in lifecycle.NODES}
                except ValueError:
                    # Extra/replaced files are a hard error; absent files may
                    # still be flushing from a worker after stop_profile.
                    if all(set(current[n])-set(before[n]) for n in lifecycle.NODES):raise
                else:break
            previous=current;time.sleep(2)
        else:raise TimeoutError('all-rank stable trace files missing')
        hashes=self.all('trace_hashes');evidence={}
        for rank,node in enumerate(lifecycle.NODES):
            path=chosen[node]
            if (Path(path).parent!=self.node_dir/'prof'
                    or not re.fullmatch(r'[A-Za-z0-9_.-]+',Path(path).name)):
                raise ValueError('unexpected trace filename or nesting')
            state=hashes[node][path]
            if {k:v for k,v in state.items() if k!='sha256'}!=current[node][path]:raise ValueError('trace changed before copy')
            target=self.out/name/('rank'+str(rank)+'-'+Path(path).name)
            if node=='local':shutil.copyfile(path,target)
            else:subprocess.run(['scp','-q','choiceoh@'+node+':'+path,str(target)],check=True,timeout=180)
            with target.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=state['sha256']:raise ValueError('trace copy hash mismatch')
            evidence[node]=dict(rank=rank,path=target.name,source=path,identity=state)
        after=self.all('trace_hashes')
        if hashes!=after:raise ValueError('trace changed during transfer')
        save(self.out/name/'traces.json',evidence)

    def collect(self,prepared,original):
        try:
            settled(lifecycle.NODES[1:],lambda n:self.host(n,'start'))
            self.host('local','start');self.ready(prepared)
            initial=self.snapshot();self.attest_clone(original,initial)
            save(self.out/'private-before.json',initial)
            api=PrivateObserverAPI('http://127.0.0.1:18000')
            idle_observers(api.post('/glm53/prefill-observe',{'op':'status'}),self.sha)
            phases=[('PRIME',self.phase('baseline','PRIME'))]
            for rep in range(5):phases.append(('BASE'+str(rep),self.phase('baseline','BASE'+str(rep))))
            salts=[]
            for _,phase in phases:
                salts.extend(r['cache_salt'] for r in phase['fresh_requests'])
            if len(salts)!=len(set(salts)):raise ValueError('duplicate baseline cache salt')
            identities=[r['prefill_identity_sha256'] for r in phases[0][1]['raw']]
            if any([r['prefill_identity_sha256'] for r in p['raw']]!=identities for _,p in phases[1:]):
                raise ValueError('canonical baseline request changed across repetitions')
            for ctx in (2000,32000,128000):
                for mode in ('profile','routes'):
                    name=mode.upper()+str(ctx);before=self.all('traces')
                    save(self.out/(name+'.traces-before.json'),before)
                    phase=self.phase(mode,name,ctx)
                    salt=phase['fresh_requests'][0]['cache_salt']
                    if salt in salts:raise ValueError('duplicate instrumented cache salt')
                    salts.append(salt)
                    index={2000:0,32000:3,128000:4}[ctx]
                    if phase['raw'][0]['prefill_identity_sha256']!=identities[index]:
                        raise ValueError('instrumented prompt/sampling differs from canonical baseline')
                    if mode=='profile':self.collect_traces(before,name)
                    elif self.all('traces')!=before:raise ValueError('routes pass unexpectedly profiled')
            final=self.snapshot(archive=True)
            if initial!=final:
                # Archive-only fields are additional, never changes to identity.
                if any(any(final[n].get(k)!=v for k,v in initial[n].items()) for n in lifecycle.NODES):
                    raise ValueError('private runtime changed during collection')
            save(self.out/'private-after.json',final)
        finally:
            self.cleanup()

    def run(self):
        lifecycle.check_holder();lifecycle.pinned(str(self.source),self.revision)
        if os.environ.get('FLEET_OBSERVATION_CLONES')!='1':
            raise RuntimeError('fleet-owned observation clone cleanup is required')
        if not (Path(os.environ['FLEET_RUNNER_REPO'])/'bench/fleet_observation_cleanup.py').is_file():
            raise RuntimeError('frozen fleet runner lacks observation clone cleanup')
        self.out.mkdir(parents=True,exist_ok=False)
        result=dict(source_revision=self.revision,observer_sha256=self.sha,complete=False,performance_acceptance=False)
        save(self.out/'incomplete.json',result)
        before=None;pause_attempted=False
        try:
            # An existing runtime is the observation source. Normalizing it
            # would spend an unmeasured recovery boot before the experiment.
            before=lifecycle.snapshot()
            if lifecycle.validate_before(before)!='present':raise RuntimeError('existing compatible serving required; run after an observed baseline or idle recovery')
            lifecycle.idle(8000);save(self.out/'before.json',before)
            resources=settled(lifecycle.NODES,lambda n:lifecycle.remote(n,
                'import json,shutil,subprocess\np='+repr(str(self.source))+'\nr='+repr(self.revision)+'\n'
                'assert subprocess.check_output(["git","-C",p,"rev-parse","HEAD"],text=True).strip()==r\n'
                'assert not subprocess.check_output(["git","-C",p,"status","--porcelain"],text=True).strip()\n'
                'print(json.dumps(dict(disk_free_gib=shutil.disk_usage("/home/choiceoh").free/2**30)))'))
            save(self.out/'resources.json',resources)
            if any(r['disk_free_gib']<128 for r in resources.values()):raise RuntimeError('128 GiB disk reserve required')
            original=self.snapshot(original=True);save(self.out/'original-runtime.json',original)
            try:
                prepared=self.all('prepare');save(self.out/'prepared.json',prepared)
                pause_attempted=True
                lifecycle.with_paused(before,lambda:self.collect(prepared,original),lambda name,value:save(self.out/name,value))
            finally:
                previous=signal.signal(signal.SIGTERM,signal.SIG_IGN)
                try:
                    self.cleanup()
                    result['cleanup_complete']=True
                finally:signal.signal(signal.SIGTERM,previous)
            result['complete']=True
        except BaseException as exc:result['error']=repr(exc)
        finally:
            result['original_stop_attempted']=pause_attempted
            result['public_recovery']='central idle controller'
            result['ended']=time.time();save(self.out/'completion.json',result)
        if result['complete']:print('GLM53_PREFILL_OBSERVATION_CAPTURE_COMPLETE',flush=True)
        return 0 if result['complete'] else 1


def analyze(directory):
    sys.path.insert(0,str(ROOT/'tools'))
    import trace_prefill_attribution as attributed
    complete=json.loads((directory/'completion.json').read_text())
    if not complete.get('complete') or not (complete.get('cleanup_complete') or complete.get('restored_original')):
        raise ValueError('complete collection and owned clone cleanup required')
    baseline=[];salts=set();identities=None;results={};routing={}
    def phase(name):
        path=directory/name
        report=json.loads((path/'result.json').read_text())
        if not report['complete'] or report['name']!=name:raise ValueError('incomplete or mismatched phase')
        requests,raw=report['fresh_requests'],report['raw']
        if not requests or len(requests)!=len(raw):raise ValueError('missing raw request evidence')
        for i,(request,record) in enumerate(zip(requests,raw)):
            wire=(path/f'{i}.request.json').read_bytes()
            response=gzip.decompress((path/f'{i}.sse.gz').read_bytes())
            if (hashlib.sha256(wire).hexdigest()!=record['request_sha256']
                    or record['request_sha256']!=request['wire_sha256']
                    or hashlib.sha256(response).hexdigest()!=record['sse_sha256']
                    or prefill_identity(wire)!=record['prefill_identity_sha256']
                    or json.loads(wire)['cache_salt']!=request['cache_salt']
                    or request['cache_salt'] in salts or request.get('issues') or request.get('error')):
                raise ValueError('raw request/response identity or fresh-cache evidence mismatch')
            salts.add(request['cache_salt'])
        return report
    for name in ['PRIME']+['BASE'+str(i) for i in range(5)]:
        report=phase(name);validate_baseline(report['onepass'],report['fresh_requests'])
        current=[r['prefill_identity_sha256'] for r in report['raw']]
        if identities is None:identities=current
        if current!=identities:raise ValueError('baseline request identity changed')
        if name!='PRIME':
            for timing,fresh in zip(report['onepass']['requests'],report['fresh_requests']):
                if (timing['ttft_s']!=fresh['ttft_s'] or timing['prompt_tokens']!=fresh['prompt_tokens']
                        or timing['completion_tokens']!=fresh['completion_tokens']):
                    raise ValueError('baseline timing/usage join mismatch')
                baseline.append(dict(repetition=name,**timing))
    for ctx in (2000,32000,128000):
        path=directory/('PROFILE'+str(ctx))
        observed={}
        for mode in ('profile','routes'):
            name=mode.upper()+str(ctx);report=phase(name)
            if len(report['raw'])!=1 or report['raw'][0]['prefill_identity_sha256']!=identities[{2000:0,32000:3,128000:4}[ctx]]:
                raise ValueError('instrumented request differs from canonical baseline')
            data=report['instrumented']
            if not data['complete']:raise ValueError('incomplete observer phase')
            ranks=validate_ranks(data['observation']['ranks'],request_id=name,mode=mode,
                                 source_sha256=complete['observer_sha256'])
            observed[mode]={r['rank']:r for r in ranks}
        observations=observed['profile']
        traces=json.loads((path/'traces.json').read_text())
        if (set(traces)!=set(lifecycle.NODES) or
                any(traces[n]['rank']!=r for r,n in enumerate(lifecycle.NODES))):
            raise ValueError('exact all-rank trace inventory required')
        for record in traces.values():
            source=path/record['path']
            with source.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=record['identity']['sha256']:raise ValueError('archived trace hash differs')
            # Existing streamed pure-prefill attribution provides the scheduler
            # ranges; our observer independently proves exact MoE call coverage.
            checked=trace.analyze(attributed.events(source),observations[record['rank']])
            checked['pure_prefill']=attributed.analyze(source)
            results[str(ctx)+'-rank'+str(record['rank'])]=checked
        for rank,report in observed['routes'].items():
            if (report['moe_forward_groups']!=observations[rank]['moe_forward_groups']
                    or report['layers']!=observations[rank]['layers']):
                raise ValueError('profile/routes execution coverage differs')
            groups=[];layers=len(report['layers'])
            for group in report['moe_forward_groups']:
                records=report['records'][group['index']*layers:(group['index']+1)*layers]
                padded={str(tile):sum(r['padded_rows'][str(tile)] for r in records) for tile in (64,128)}
                groups.append(dict(**group,padded_rows_across_layers=padded,
                    padded_work_reduction_pct=100*(1-padded['64']/padded['128']),
                    per_layer=[dict(layer=r['layer'],active_experts=sum(c>0 for c in r['expert_counts']),
                        max_expert_slots=max(r['expert_counts']),mean_expert_slots=r['routed_slots']/288,
                        zero_weight_slots=r['zero_weight_slots'],
                        padded_work_reduction_pct=r['padded_work_reduction_pct']) for r in records]))
            routing[str(ctx)+'-rank'+str(rank)]=groups
    summary={}
    for ctx in (2000,32000,128000):
        rows=[r for r in baseline if r['ctx']==ctx]
        times=[r['ttft_s'] for r in rows]
        summary[str(ctx)]=dict(samples=len(rows),ttft_median_s=statistics.median(times),
            ttft_min_s=min(times),ttft_max_s=max(times),prompt_tokens=sorted({r['prompt_tokens'] for r in rows}))
    save(directory/'attribution.json',dict(results=results,routing=routing,
        baseline=dict(summary=summary,requests=baseline,quality_checks=45,prime_excluded=True),
        workload='Canonical Korean onepass; observed model routing, not a production traffic distribution',
        performance_acceptance=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--revision',required=True);run.add_argument('--out',type=Path,required=True)
    analysis=sub.add_parser('analyze');analysis.add_argument('directory',type=Path)
    args=ap.parse_args()
    if args.command=='analyze':analyze(args.directory);return 0
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(SystemExit(143)))
    return Run(ROOT,args.revision,args.out.resolve()).run()


if __name__=='__main__':raise SystemExit(main())
