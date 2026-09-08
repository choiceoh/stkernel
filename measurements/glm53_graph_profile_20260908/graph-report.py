"""Read-only summary and verification of the unused graph-profile bracket."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import subprocess
import sys
root, repo = map(Path, sys.argv[1:3])
subprocess.run([sys.executable, str(Path(__file__).with_name('startup-cache-report.py')), str(root)], check=True, stdout=subprocess.DEVNULL)
data = json.loads((root/'report.json').read_text())
ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
for arm, row in data['arms'].items():
    for suffix, key in (('first-requests.json','first_requests'), ('graph-env.json','graph_environment')):
        p = root/f'{arm}-{suffix}'
        if p.exists(): row[key] = json.loads(p.read_text())
    boot = root/f'{arm}-boot.out'
    row['compile_cache'] = [json.loads(x) for x in re.findall(r'\[compile-cache\] (\{[^\n]+\})', boot.read_text())] if boot.exists() else []
    for node, detail in row.get('nodes',{}).items():
        text = ansi.sub('', (root/f'{arm}-{node}.log').read_text(errors='replace'))
        detail['graph_profile_skipped'] = '[glm53-graph-profile] skipped unused estimate' in text
        detail['kv_memory_gib'] = [float(x) for x in re.findall(r'Available KV cache memory: ([\d.-]+) GiB', text)]
        detail['kv_blocks'] = [int(x) for x in re.findall(r'num_gpu_blocks[=: ]+(\d+)',text)]
        detail['pack_io'] = [{k: float(v) if k.endswith('_s') else int(v) for k,v in re.findall(r'(\w+)=([\d.]+)',fields)} for fields in re.findall(r'\[mk-pack-io\] \S+ ([^\n]*)', text)]
        if node == 'srv2':
            health = text.find('GET /health HTTP/1.1" 200')
            posts = list(re.finditer(r'(\d+\.\d+\.\d+\.\d+):\d+ - "POST /',text))
            row['post_completions'] = dict(loopback=sum(m[1].startswith('127.') for m in posts), non_loopback=sum(not m[1].startswith('127.') for m in posts), before_health=sum(m.start()<health for m in posts))
summary = {}
for kind in ('BASE','FAST'):
    rows = [r for a,r in data['arms'].items() if re.search(kind+r'[12]$',a)]
    values = {'health_wall_s':[r['health_wall_s'] for r in rows if 'health_wall_s' in r]}
    for phase in ('load-model','encoder-profile','profile-run','cudagraph-memory-profile','profile/determine-memory','cudagraph-capture','compile+warmup'):
        values[phase] = [r['nodes']['srv2']['phase_s'].get(phase,[0])[0] for r in rows if 'srv2' in r.get('nodes',{})]
    for label in ('text','image','video'):
        values['first_'+label+'_ttft_s']=[x['ttft_s'] for r in rows for x in r.get('first_requests',{}).get('requests',[]) if x['kind']==label]
    summary[kind] = {k:dict(samples=v, mean=statistics.mean(v), min=min(v), max=max(v)) for k,v in values.items() if v and all(x is not None for x in v)}
data['head_comparison'] = summary
lines = ['# GLM unused CUDA graph memory profiling', '', f"Runtime source: `{data['source_commit']}`.", '',
    'PRIME is excluded. Timed order: BASE1, FAST1, FAST2, BASE2. Same source and image; only VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE changes. Both arms retain the model/MM memory profile, actual graph capture and kernel warmup; both use early CPU MM warmup. First text/image/video requests precede the canonical 2K/32K Korean onepass.', '',
    '| Arm | Health s | Model s | Encoder s | Dry capture s | Memory profile s | Real capture s | Compile/warmup s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for arm,r in data['arms'].items():
    p=r.get('nodes',{}).get('srv2',{}).get('phase_s',{})
    vals=[r.get('health_wall_s'),*[p.get(k,[0])[0] for k in ('load-model','encoder-profile','cudagraph-memory-profile','profile/determine-memory','cudagraph-capture','compile+warmup')]]
    lines.append('| '+arm+' | '+' | '.join(map(str,vals))+' |')
lines += ['', '| Metric | BASE mean | FAST mean | Difference |', '|---|---:|---:|---:|']
for key,b in summary['BASE'].items():
    f=summary['FAST'].get(key)
    if f: lines.append(f"| {key} | {b['mean']:.3f} | {f['mean']:.3f} | {f['mean']-b['mean']:+.3f} |")
lines += ['', 'Phase timers are nested and include host work and existing synchronizations. First-request samples do not establish general throughput/quality. POST completions and response evidence are retained to detect interference.', '']
for arm,r in data['arms'].items():
    lines.append(f"- {arm}: quality={r.get('onepass',{}).get('quality')}; corruption={r.get('onepass',{}).get('korean')}; posts={r.get('post_completions')}")
if '--verify' in sys.argv:
    assert data['exit_code']==0
    assert list(data['arms'])==['GRAPHMEMPRIME','GRAPHMEMBASE1','GRAPHMEMFAST1','GRAPHMEMFAST2','GRAPHMEMBASE2']
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'build/glm53'/name for name in ('gpu_worker.py','deneb_boot_stamps.py','glm53_megakernel.py','glm53_rank_cache.py','glm53_startup_cache.py'))}
    images,content=set(),set()
    for arm,r in data['arms'].items():
        fast='FAST' in arm or arm.endswith('PRIME')
        assert dict(x.split('=',1) for x in r['graph_environment'])=={'VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE':str(int(fast)),'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS':'0'}
        assert r['first_requests']['ok'] and len(r['first_requests']['requests'])==3
        assert r['onepass']['quality']=={'ok':6,'total':6} and r['onepass']['korean']['dirty']==0
        assert r['post_completions']['before_health']==0
        assert len(r['nodes'])==4
        assert len(r['compile_cache'])==1
        content.add(r['compile_cache'][0]['content_sha256'])
        if not arm.endswith('PRIME'): assert r['compile_cache'][0]['action']=='reuse'
        for node,d in r['nodes'].items():
            assert d['graph_profile_skipped']==fast
            assert ('cudagraph-memory-profile' in d['phase_s'])!=fast
            for label in ('encoder-profile','profile-run','cudagraph-capture','compile+warmup'): assert label in d['phase_s']
            assert not d['cache_warnings'] and not any(d['copy_disarmed'])
            assert sum(x['errors'] for x in d['fp8'])==0
            if not arm.endswith('PRIME'):
                assert len(d['rank'])==1 and d['rank'][0]['kind']=='hit'
                assert sum(x['hit'] for x in d['fp8'])==244 and sum(x['miss'] for x in d['fp8'])==0
            io=d['pack_io'][-1]
            assert io['sha_hits']==io['fast_hits']==255
            assert io['legacy_hits']==io['md5_fallback']==io['aliases']==io['alias_errors']==0
            receipt=(root/f'{arm}-{node}.{"sha256" if node=="srv2" else "state"}').read_text()
            actual={Path(p).name:h for h,p in re.findall(r'([0-9a-f]{64})\s+(\S+\.py)',receipt)}
            assert actual==expected,(arm,node,actual,expected)
            state=(root/f'{arm}-{node}.state').read_text()
            assert state.startswith('running 0 false sha256:')
            images.add(state.splitlines()[0].split()[-1])
    assert len(content)==len(images)==1
    data['verification']=dict(ok=True, source_hashes=expected, images=list(images), compile_content=list(content), boots=5, nodes_per_boot=4)
(root/'report.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(root/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:16]))
