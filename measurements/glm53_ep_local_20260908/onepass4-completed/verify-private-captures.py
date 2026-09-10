"""Read archived local bytes only; write one scoped verification receipt."""
import base64,datetime,gzip,hashlib,json,os,re,sys,time
from pathlib import Path

WORK=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
sys.path.insert(0,str(WORK/'bench'))
import glm53_ep_serving_contract as contract
import glm53_launch_metadata as launch
S=Path('/tmp/glm53-onepass4-streams')
DIRS={'A':Path('/tmp/glm53-onepass4-live-A-manualfix'),'B2':Path('/tmp/glm53-onepass4-live-B2-observer')}
OUT=Path('/tmp/glm53-onepass4-final-validation.json')
REV='96cb599d8816ee2585988fc7b750a21e2eb66a0b'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
ids={a:json.loads((p/'identity.json').read_bytes())for a,p in DIRS.items()}
events_raw=(S/'events.jsonl').read_bytes();events=[json.loads(l)for l in events_raw.splitlines()]
nodes=('local','10.10.10.1','10.10.10.3','10.10.10.4')
raws={n:(S/(n+'.stdout.raw')).read_bytes()for n in nodes}
verified={}
for a,p in DIRS.items():
    assert ids[a]['arm']==a and ids[a]['revision']==REV
    assert sha((p/'launch-parser.py').read_bytes())==ids[a]['parser_sha256']
    assert(p/'launch-parser.py').read_bytes()==(WORK/'bench/glm53_launch_metadata.py').read_bytes()
    for name,meta in ids[a]['files'].items():
        stored=(p/name).read_bytes();raw=gzip.decompress(stored)
        assert sha(stored)==meta['stored_sha256'] and len(stored)==meta['stored_bytes']
        assert sha(raw)==meta['original_sha256'] and len(raw)==meta['original_bytes']
    verified[a]={'identity_sha256':sha((p/'identity.json').read_bytes()),'original_and_stored_files_verified':len(ids[a]['files'])}

def env(c):
    pairs=[x.split('=',1)for x in c['Config']['Env']];assert len({k for k,v in pairs})==len(pairs)
    return dict(pairs)
def normalize(c):
    h=dict(c['HostConfig']);h['Binds']=sorted(h.get('Binds')or[])
    return {'Image':c['Image'],'Config':c['Config'],'HostConfig':h,'Mounts':sorted(c['Mounts'],key=canon),
            'Id':c['Id'],'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid'],'RestartCount':c['RestartCount']}
def readc(a,n,which):return json.loads(gzip.decompress((DIRS[a]/(n+'.inspect.'+which+'.json.gz')).read_bytes()))[0]
def stamp(line):
    m=re.search(r'(?:2026-)?(09-08) (\d\d:\d\d:\d\d)(?:[,.](\d+))?',line)
    if not m:return None
    return datetime.datetime.fromisoformat('2026-'+m[1]+'T'+m[2]+'.'+(m[3]or'0')).replace(tzinfo=datetime.timezone.utc).timestamp()
def chunk_for(n,offset):
    r=next(r for r in events if r.get('kind')=='chunk'and r.get('node')==n and r.get('channel')=='stdout'and r['offset']<=offset<r['offset']+r['bytes'])
    return {k:r[k]for k in('at','local_at','offset','bytes','sha256')}
def line_records(raw):
    offset=0;result=[]
    for line in raw.splitlines(keepends=True):
        result.append((offset,line.decode(errors='replace').rstrip('\r\n')));offset+=len(line)
    return result

runtime={};streams={};analysis={}
allow={'VLLM_B12X_EP_COMPACT':('1',None),'VLLM_B12X_EP_WARM_COMPACT':('1','0'),'VLLM_GLM53_EP_PREFILL_LOCAL':('1','0')}
end=datetime.datetime(2026,9,8,21,10,tzinfo=datetime.timezone.utc).timestamp()
for rank,n in enumerate(nodes):
    cs={a:readc(a,n,'before')for a in DIRS}
    for a,c in cs.items():
        after=readc(a,n,'after');assert normalize(c)==normalize(after)
        assert c['State']['Running'] and after['State']['Running'] and c['Image']==IMAGE
        assert c['Id']==ids[a]['nodes'][n]['id'] and c['State']['StartedAt']==ids[a]['nodes'][n]['started_at']
    es={a:env(c)for a,c in cs.items()}
    changed={k for k in es['A'].keys()|es['B2'].keys()if es['A'].get(k)!=es['B2'].get(k)}
    assert changed==set(allow)
    assert all((es['A'].get(k),es['B2'].get(k))==values for k,values in allow.items())
    assert es['A']['VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE']==es['B2']['VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE']=='1'
    na,nb=normalize(cs['A']),normalize(cs['B2'])
    for key in('Image','HostConfig','Mounts'):assert na[key]==nb[key]
    assert {k:v for k,v in na['Config'].items()if k not in('Cmd','Env')}=={k:v for k,v in nb['Config'].items()if k not in('Cmd','Env')}
    top={a:launch.launch_parallelism(c['Config']['Cmd'])for a,c in cs.items()}
    assert top['A']['enabled'] and not top['B2']['enabled']
    assert top['A']['serve_argv_without_ep_sha256']==top['B2']['serve_argv_without_ep_sha256']
    assert all(t['node_rank']==rank and t['nnodes']==4 and t['tensor_parallel_size']==4 for t in top.values())
    caps={a:contract.launch_capacity(c['Config']['Cmd'],public=False)for a,c in cs.items()}
    assert caps['A']['capacity']==caps['B2']['capacity']
    assert ids['A']['nodes'][n]['source']==ids['B2']['nodes'][n]['source']
    runtime[n]={'matched':True,'capacity':caps['A']['capacity'],'environment_differences':{k:{'A':v[0],'B2':v[1]}for k,v in allow.items()},
        'non_ep_argv_sha256':top['A']['serve_argv_without_ep_sha256'],'source':ids['A']['nodes'][n]['source'],
        'full_hostconfig_and_mounts_equal_after_order_only_normalization':True,
        'container_ids':{a:c['Id']for a,c in cs.items()},'image':IMAGE}
    raw=raws[n]; chunks=[r for r in events if r.get('kind')=='chunk'and r.get('node')==n and r.get('channel')=='stdout'];offset=0
    for r in chunks:
        assert r['offset']==offset;part=raw[offset:offset+r['bytes']];assert sha(part)==r['sha256'];offset+=r['bytes']
    assert offset==len(raw)
    snapshots={a:gzip.decompress((p/(n+'.serving.log.gz')).read_bytes())for a,p in DIRS.items()}
    starts={a:raw.find(snap)for a,snap in snapshots.items()}
    assert all(v>=0 and raw.count(snapshots[a])==1 for a,v in starts.items())
    assert starts['A']+len(snapshots['A'])<starts['B2']
    segment=raw[starts['A']:starts['B2']];records=line_records(segment)
    warm=[(off,l)for off,l in records if '[b12x EP compact warmup] COMPLETE 'in l]
    assert len(warm)==1 and 'specializations=14 static=10 dynamic=4 required=14 ready=14 'in warm[0][1]
    wo,wl=warm[0];prefix=re.match(r'\(Worker_TP'+str(rank)+r'_EP'+str(rank)+r' pid=\d+\)',wl)[0]
    graphs=[(off,l)for off,l in records if l.startswith(prefix)and'Graph capturing finished'in l];assert len(graphs)==1
    go,gl=graphs[0];ready=stamp(gl);assert stamp(wl)<ready<end
    cute=[];unknown=[];jit=[]
    for off,line in records:
        t=stamp(line)
        if 'Compiling CuTe-DSL kernel' in line:
            if t is None:unknown.append({'offset':starts['A']+off,'line':line})
            elif ready<=t<=end:cute.append({'offset':starts['A']+off,'line':line})
        if line.startswith(prefix)and'JIT compilation during inference:'in line and t is not None and ready<=t<=end:
            name=line.split('JIT compilation during inference:',1)[1].split('. This causes',1)[0].strip()
            category='MHC TileLang'if'mhc_'in name else('sampler Triton'if name in('_compute_local_logits_stats_kernel','_rejection_kernel','_resample_kernel')else('EP remap Triton'if name=='_remap_ep_local_kernel'else'other Triton helper'))
            jit.append({'name':name,'category':category,'timestamp_utc':t,'offset':starts['A']+off,'line':line})
    assert not cute and not unknown
    streams[n]={'sha256':sha(raw),'bytes':len(raw),'all_chunk_hashes_and_offsets_verified':True,'chunk_count':len(chunks),
                'A_snapshot_contiguous_offset':starts['A'],'B2_snapshot_contiguous_offset':starts['B2']}
    analysis[n]={'worker_prefix':prefix,'warm_complete':{'line':wl,'stream_offset':starts['A']+wo,'chunk':chunk_for(n,starts['A']+wo)},
        'graph_complete':{'line':gl,'stream_offset':starts['A']+go},'audit_window_utc':[ready,end],
        'post_graph_through_A_end_CuTe_compile_count':len(cute),'unparseable_CuTe_compile_lines':unknown,
        'other_request_window_jit_count':len(jit),'other_jit':jit}
    if n=='local':
        post=next((off,l)for off,l in records if 'POST /v1/'in l)
        first_post={'stream_offset':starts['A']+post[0],'chunk':chunk_for(n,starts['A']+post[0]),'kind':'HTTP response log; chunk receipt time is not exact request start'}
        assert all(stamp(x['warm_complete']['line'])<first_post['chunk']['at']for x in analysis.values())

pid=int((S/'observer.pid').read_text())
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
receipt={'schema':1,'generated_at':time.time(),'verdict':'SCOPED_PASS','scope':'A versus B2 configuration/source/capacity and logged CuTe warmup coverage only; no B1 strict snapshot, quality, speed or default acceptance',
 'source_revision':REV,'snapshot_hash_verification':verified,'runtime_nodes':runtime,'stream_files':streams,'A_warmup_and_jit':analysis,
 'first_A_HTTP_response_log':first_post,'all_warm_complete_before_first_observed_A_HTTP_response':True,
 'A_end_utc_epoch':end,'A_end_basis':'parent-reported canonical A completion at 2026-09-09 06:10:00 KST',
 'observer':{'pid':pid,'alive':alive,'finished_events':[r for r in events if r['kind']=='observer_finished'],
             'stream_end_events':[r for r in events if r['kind']=='stream_end'],
             'note':'stream supervisor completed its child-termination finally block before stream_end; remote live process absence not independently rechecked'},
 'input_hashes':{'events.jsonl':sha(events_raw),'launch_parser':sha((WORK/'bench/glm53_launch_metadata.py').read_bytes()),
                 'capacity_parser':sha((WORK/'bench/glm53_ep_serving_contract.py').read_bytes()),'verifier':sha(Path(__file__).read_bytes())}}
with OUT.open('x')as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
OUT.chmod(0o600)
print(json.dumps({'output':str(OUT),'sha256':sha(OUT.read_bytes()),'runtime_nodes_matched':len(runtime),'CuTe_compile_count':sum(x['post_graph_through_A_end_CuTe_compile_count']for x in analysis.values()),'other_jit_count':sum(x['other_request_window_jit_count']for x in analysis.values()),'observer_alive':alive}))
